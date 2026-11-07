import { SIMD_WASM, SIMD_WASM_SHARED } from "./simd-bytes";

/** The exports of simd.wat: one x row against N weight rows. Pointers are byte offsets. */
export interface SimdKernels {
  gemv_f32(x: number, w: number, K: number, N: number, y: number): void;
  gemv_q8(x: number, q: number, s: number, mn: number, xsum: number, K: number, N: number, y: number): void;
  gemv_q4(x: number, q: number, s: number, mn: number, xsum: number, K: number, N: number, y: number): void;
}

/** Which gemv a threaded job runs. */
export const GEMV_F32 = 0;
export const GEMV_Q8 = 1;
export const GEMV_Q4 = 2;

/**
 * Control block of a threaded job, as i32 slots in the shared memory: the job generation (bumped
 * to start one, -1 to stop), helpers done, then the job.
 */
const C_GEN = 0, C_DONE = 1, C_FN = 2, C_M = 3, C_N = 4, C_K = 5, C_X = 6, C_W = 7, C_S = 8, C_MN = 9, C_XSUM = 10, C_Y = 11, C_PARTS = 12, C_READY = 13;
const CTRL_BYTES = 64;

/**
 * Helper thread: instantiates the SIMD module on the shared memory, then waits for jobs. Its index
 * is taken once it is waiting (generation read first, so a job started meanwhile is either seen or
 * not counted on), and only helpers the job counts report back.
 */
const HELPER = `onmessage = (e) => {
  const { memory, bytes, ctrl } = e.data;
  const k = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory } }).exports;
  const c = new Int32Array(memory.buffer, ctrl, ${CTRL_BYTES / 4});
  let gen = Atomics.load(c, ${C_GEN});
  const idx = Atomics.add(c, ${C_READY}, 1);
  for (;;) {
    Atomics.wait(c, ${C_GEN}, gen);
    gen = Atomics.load(c, ${C_GEN});
    if (gen < 0) return;
    const parts = c[${C_PARTS}];
    if (idx + 1 >= parts) continue;
    const [fn, M, N, K, x, w, s, mn, xsum, y] = [c[${C_FN}], c[${C_M}], c[${C_N}], c[${C_K}], c[${C_X}], c[${C_W}], c[${C_S}], c[${C_MN}], c[${C_XSUM}], c[${C_Y}]];
    const [n0, n1] = slice(N, parts, idx + 1);
    if (n0 < n1) run(k, fn, M, N, K, x, w, s, mn, xsum, y, n0, n1);
    Atomics.add(c, ${C_DONE}, 1);
    Atomics.notify(c, ${C_DONE});
  }
};
const slice = ${slice.toString()};
const run = ${runRows.toString()};`;

/** Rows [n0, n1) of part i of N rows split into `parts` (multiples of 8 rows). */
function slice(N: number, parts: number, i: number): [number, number] {
  const per = Math.ceil(N / parts / 8) * 8;
  return [Math.min(N, i * per), Math.min(N, (i + 1) * per)];
}

/** Every x row against weight rows [n0, n1): offsets into the weight planes and the output rows. */
function runRows(k: SimdKernels, fn: number, M: number, N: number, K: number, x: number, w: number, s: number, mn: number, xsum: number, y: number, n0: number, n1: number): void {
  const KB = K >>> 5;
  const rows = n1 - n0;
  for (let m = 0; m < M; m++) {
    const xm = x + m * K * 4, ym = y + (m * N + n0) * 4;
    if (fn === 0) k.gemv_f32(xm, w + n0 * K * 4, K, rows, ym);
    else {
      const sm = s + n0 * KB * 4, mm = mn ? mn + n0 * KB * 4 : 0, xs = xsum + m * KB * 4;
      if (fn === 1) k.gemv_q8(xm, w + n0 * K, sm, mm, xs, K, rows, ym);
      else k.gemv_q4(xm, w + n0 * (K >>> 1), sm, mm, xs, K, rows, ym);
    }
  }
}

/** Multiply-adds below which a matmul stays on one thread (waking helpers costs tens of µs). */
const THREAD_MIN_MACS = 1 << 20;

const PAGE = 65536;
const ALIGN = 16;
const up = (n: number) => Math.ceil(n / ALIGN) * ALIGN;

/**
 * One WebAssembly memory holding every buffer of a CPU backend, so the SIMD kernels can address
 * weights, activations and scratch directly. A first-fit free list with coalescing; offset 0 is
 * never handed out, so a 0 pointer can mean "none". Growing the memory detaches earlier views of
 * it: read `buffer` again after any `alloc` or `scratch`.
 */
export class WasmHeap {
  private holes: { at: number; size: number }[] = [];
  private top = ALIGN;
  private scratchAt = 0;
  private scratchSize = 0;

  /** Helper threads and the control block they wait on (threaded heaps only). */
  private helpers: Worker[] = [];
  private ctrl: Int32Array | undefined;

  private constructor(
    readonly memory: WebAssembly.Memory,
    readonly kernels: SimdKernels,
  ) {}

  /**
   * A heap, or undefined where WebAssembly SIMD is unavailable. With `threads` > 1 the memory is
   * shared and matmuls split across that many threads (this one and helper workers), where that is
   * possible: a cross-origin isolated page (SharedArrayBuffer), Worker available, and the heap used
   * off a page's main thread (which may not block on Atomics.wait). Otherwise one thread.
   */
  static create(opts: { threads?: number } = {}): WasmHeap | undefined {
    try {
      if (typeof WebAssembly === "undefined") return undefined;
      const threads = Math.max(1, Math.floor(opts.threads ?? 1));
      const g = globalThis as { crossOriginIsolated?: boolean; document?: unknown; Worker?: unknown; Blob?: unknown };
      const shared = threads > 1 && g.crossOriginIsolated === true && typeof g.Worker === "function" && typeof g.Blob === "function" && g.document === undefined;
      const bytes = Uint8Array.from(atob(shared ? SIMD_WASM_SHARED : SIMD_WASM), (c) => c.charCodeAt(0));
      if (!WebAssembly.validate(bytes)) return shared ? WasmHeap.create() : undefined;
      const memory = new WebAssembly.Memory({ initial: 16, maximum: 65536, ...(shared ? { shared: true } : {}) });
      const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory } });
      const heap = new WasmHeap(memory, inst.exports as unknown as SimdKernels);
      if (shared) heap.startHelpers(bytes, threads - 1);
      return heap;
    } catch {
      return undefined;
    }
  }

  /** Threads a large matmul is split across (1 without helpers). */
  get threads(): number {
    return 1 + (this.ctrl ? Atomics.load(this.ctrl, C_READY) : 0);
  }

  private startHelpers(bytes: Uint8Array, count: number): void {
    const at = this.alloc(CTRL_BYTES);
    this.ctrl = new Int32Array(this.memory.buffer, at, CTRL_BYTES / 4);
    const url = URL.createObjectURL(new Blob([HELPER], { type: "text/javascript" }));
    for (let i = 0; i < count; i++) {
      const w = new Worker(url, { name: `kiln-cpu-${i + 1}` });
      w.postMessage({ memory: this.memory, bytes, ctrl: at });
      this.helpers.push(w);
    }
    URL.revokeObjectURL(url);
  }

  /**
   * y rows for every x row against N weight rows (`fn`: GEMV_F32 / GEMV_Q8 / GEMV_Q4; pointers as
   * the gemv exports take them), split across the helper threads when large enough.
   */
  gemm(fn: number, M: number, N: number, K: number, x: number, w: number, s: number, mn: number, xsum: number, y: number): void {
    const c = this.ctrl;
    const parts = c ? Math.min(1 + Atomics.load(c, C_READY), Math.ceil(N / 8)) : 1;
    if (!c || parts < 2 || M * N * K < THREAD_MIN_MACS) {
      runRows(this.kernels, fn, M, N, K, x, w, s, mn, xsum, y, 0, N);
      return;
    }
    c[C_FN] = fn; c[C_M] = M; c[C_N] = N; c[C_K] = K; c[C_X] = x; c[C_W] = w; c[C_S] = s; c[C_MN] = mn; c[C_XSUM] = xsum; c[C_Y] = y; c[C_PARTS] = parts;
    Atomics.store(c, C_DONE, 0);
    Atomics.add(c, C_GEN, 1);
    Atomics.notify(c, C_GEN);
    const [n0, n1] = slice(N, parts, 0);
    runRows(this.kernels, fn, M, N, K, x, w, s, mn, xsum, y, n0, n1);
    for (let d = Atomics.load(c, C_DONE); d < parts - 1; d = Atomics.load(c, C_DONE)) Atomics.wait(c, C_DONE, d);
  }

  /** Stops the helper threads. */
  destroy(): void {
    if (this.ctrl) {
      Atomics.store(this.ctrl, C_GEN, -1);
      Atomics.notify(this.ctrl, C_GEN);
    }
    for (const w of this.helpers) w.terminate();
    this.helpers = [];
    this.ctrl = undefined;
  }

  get buffer(): ArrayBuffer {
    return this.memory.buffer;
  }

  /** Bytes the heap spans (its high-water mark). */
  get bytes(): number {
    return this.top;
  }

  alloc(bytes: number): number {
    const size = up(Math.max(bytes, 1));
    for (let i = 0; i < this.holes.length; i++) {
      const h = this.holes[i]!;
      if (h.size < size) continue;
      const at = h.at;
      if (h.size === size) this.holes.splice(i, 1);
      else {
        h.at += size;
        h.size -= size;
      }
      return at;
    }
    const at = this.top;
    this.reserve(at + size);
    this.top = at + size;
    return at;
  }

  release(at: number, bytes: number): void {
    const size = up(Math.max(bytes, 1));
    if (at + size === this.top) {
      this.top = at;
      // The hole now ending at the top goes back to it too.
      const last = this.holes[this.holes.length - 1];
      if (last && last.at + last.size === this.top) {
        this.top = last.at;
        this.holes.pop();
      }
      return;
    }
    let i = 0;
    while (i < this.holes.length && this.holes[i]!.at < at) i++;
    this.holes.splice(i, 0, { at, size });
    // Merge with the next hole, then with the previous one.
    const next = this.holes[i + 1];
    if (next && at + size === next.at) {
      this.holes[i]!.size += next.size;
      this.holes.splice(i + 1, 1);
    }
    const prev = this.holes[i - 1];
    if (prev && prev.at + prev.size === at) {
      prev.size += this.holes[i]!.size;
      this.holes.splice(i, 1);
    }
  }

  /** A scratch region of at least `bytes`, reused by every kernel call (its contents don't persist). */
  scratch(bytes: number): number {
    if (bytes > this.scratchSize) {
      if (this.scratchAt) this.release(this.scratchAt, this.scratchSize);
      this.scratchSize = up(bytes * 1.25);
      this.scratchAt = this.alloc(this.scratchSize);
    }
    return this.scratchAt;
  }

  private reserve(end: number): void {
    const have = this.memory.buffer.byteLength;
    if (end <= have) return;
    // Grow by at least a quarter, to keep the number of grows (each a copy, on some engines) low.
    const pages = Math.ceil(Math.max(end - have, have / 4) / PAGE);
    const max = 65536 - have / PAGE;
    if (Math.ceil((end - have) / PAGE) > max) throw new Error(`CPU backend: out of memory (${(end / 2 ** 30).toFixed(2)} GiB wanted, 4 GiB at most)`);
    this.memory.grow(Math.min(pages, max));
  }
}
