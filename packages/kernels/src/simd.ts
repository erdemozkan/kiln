import { SIMD_WASM, SIMD_WASM_SHARED } from "./simd-bytes";

/** The exports of simd.wat. Pointers are byte offsets. */
export interface SimdKernels {
  /** One float x row against N f32 weight rows. */
  gemv_f32(x: number, w: number, K: number, N: number, y: number): void;
  /** One x row (int16 per block, xq, with block scales xs; q4's in nibble order, see simd.wat) against N q8 / q4 rows. */
  gemv_q8i(xq: number, xs: number, q: number, s: number, mn: number, xsum: number, K: number, N: number, y: number): void;
  gemv_q4i(xq: number, xs: number, q: number, s: number, mn: number, xsum: number, K: number, N: number, y: number): void;
  /** The same for four x rows at once (xq rows K apart, xs/xsum rows K/32 apart, outputs ys floats apart), identical per row. */
  gemm4_q8i(xq: number, xs: number, q: number, s: number, mn: number, xsum: number, K: number, N: number, y: number, ys: number): void;
  gemm4_q4i(xq: number, xs: number, q: number, s: number, mn: number, xsum: number, K: number, N: number, y: number, ys: number): void;
  /**
   * rows × K floats at x as int16 per 32-value block (xq; q4's lane order when q4 is 1), block scales xs
   * and float sums xsum; RMS-normalised first with weights nw and eps when nw is not 0.
   */
  quant_x(x: number, K: number, rows: number, nw: number, eps: number, xq: number, xs: number, xsum: number, q4: number): void;
  /**
   * For G query heads sharing a KV head (queries D floats apart): out[g · os + j] = q_g · K_j for keys j
   * in [first, n), key j at k + (j mod cap) · stride bytes, and each head's largest score at mx[g].
   * f16: the cache holds halves (widened through tmp, D floats).
   */
  att_scores(q: number, k: number, stride: number, cap: number, first: number, n: number, D: number, out: number, G: number, os: number, mx: number, f16: number, tmp: number): void;
  /** acc[g · D + i] = Σ_j p[g · ps + j] · V_j[i] over the same keys (acc overwritten). */
  att_wsum(p: number, v: number, stride: number, cap: number, first: number, n: number, D: number, acc: number, G: number, ps: number, f16: number, tmp: number): void;
}

/** Which gemv a threaded job runs. */
export const GEMV_F32 = 0;
export const GEMV_Q8 = 1;
export const GEMV_Q4 = 2;

/**
 * Control block of a threaded job, as i32 slots in the shared memory: the job generation (bumped
 * to start one, -1 to stop), helpers done, then the job.
 */
const C_GEN = 0, C_DONE = 1, C_FN = 2, C_M = 3, C_N = 4, C_K = 5, C_X = 6, C_W = 7, C_S = 8, C_MN = 9, C_XSUM = 10, C_Y = 11, C_PARTS = 12, C_READY = 13, C_XS = 14;
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
    const [fn, M, N, K, x, xs, w, s, mn, xsum, y] = [c[${C_FN}], c[${C_M}], c[${C_N}], c[${C_K}], c[${C_X}], c[${C_XS}], c[${C_W}], c[${C_S}], c[${C_MN}], c[${C_XSUM}], c[${C_Y}]];
    const [n0, n1] = slice(N, parts, idx + 1);
    if (n0 < n1) run(k, fn, M, N, K, x, xs, w, s, mn, xsum, y, n0, n1);
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

/**
 * Every x row against weight rows [n0, n1): offsets into the weight planes and the output rows. x is
 * float rows for f32 weights, else int16 rows (xq) with block scales at xs.
 */
function runRows(k: SimdKernels, fn: number, M: number, N: number, K: number, x: number, xs: number, w: number, s: number, mn: number, xsum: number, y: number, n0: number, n1: number): void {
  const KB = K >>> 5;
  const rows = n1 - n0;
  if (fn === 0) {
    for (let m = 0; m < M; m++) k.gemv_f32(x + m * K * 4, w + n0 * K * 4, K, rows, y + (m * N + n0) * 4);
    return;
  }
  const wq = w + n0 * (fn === 2 ? K >>> 1 : K), sm = s + n0 * KB * 4, mm = mn ? mn + n0 * KB * 4 : 0;
  let m = 0;
  // Four x rows per pass over the weights where there are four (identical results to one at a time).
  const four = fn === 2 ? k.gemm4_q4i : k.gemm4_q8i;
  for (; m + 4 <= M; m += 4) four(x + m * K * 2, xs + m * KB * 4, wq, sm, mm, xsum + m * KB * 4, K, rows, y + (m * N + n0) * 4, N);
  const one = fn === 2 ? k.gemv_q4i : k.gemv_q8i;
  for (; m < M; m++) one(x + m * K * 2, xs + m * KB * 4, wq, sm, mm, xsum + m * KB * 4, K, rows, y + (m * N + n0) * 4);
}

/** Multiply-adds below which a matmul stays on one thread (waking helpers costs tens of µs). */
const THREAD_MIN_MACS = 1 << 20;

/**
 * Picks, per matmul shape, whether splitting across the helper threads pays, by timing both ways.
 * Whether it does depends on things only a run shows: on a browser reporting 2 cores with the page
 * in the background, waking a helper took long enough that SmolLM2-135M decoded at 8.6 tok/s on two
 * threads against 55 on one, while elsewhere the same split is a gain. Each shape is first timed
 * twice each way; after that the faster way runs, and the slower one is timed again every `every`
 * calls (16, doubling up to 1024 while it stays slower), so a change of conditions is picked up.
 */
export class ThreadTuner {
  private readonly shapes = new Map<string, { one: number; many: number; nOne: number; nMany: number; best: boolean; calls: number; every: number }>();

  /** True to split this call across the helpers. */
  choose(key: string): boolean {
    const t = this.shapes.get(key);
    if (!t) return false;
    if (t.nOne < 2 || t.nMany < 2) return t.nMany < t.nOne;
    if (++t.calls >= t.every) {
      t.calls = 0;
      return !t.best;
    }
    return t.best;
  }

  /** Records how long a call took (ms) the way it ran. */
  record(key: string, threaded: boolean, ms: number): void {
    let t = this.shapes.get(key);
    if (!t) this.shapes.set(key, (t = { one: 0, many: 0, nOne: 0, nMany: 0, best: false, calls: 0, every: 16 }));
    const probe = t.nOne >= 2 && t.nMany >= 2 && threaded !== t.best;
    if (threaded) t.many = t.nMany++ ? 0.7 * t.many + 0.3 * ms : ms;
    else t.one = t.nOne++ ? 0.7 * t.one + 0.3 * ms : ms;
    if (t.nOne < 2 || t.nMany < 2) return;
    const best = t.many < t.one;
    if (best !== t.best) {
      t.best = best;
      t.every = 16;
    } else if (probe) t.every = Math.min(t.every * 2, 1024);
  }

  /** The way each shape currently runs (for profiling and tests). */
  get choices(): Record<string, "one" | "threads"> {
    const out: Record<string, "one" | "threads"> = {};
    for (const [k, t] of this.shapes) if (t.nOne >= 2 && t.nMany >= 2) out[k] = t.best ? "threads" : "one";
    return out;
  }
}

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

  /** Whether each matmul shape runs split across the helpers (see ThreadTuner). */
  readonly tuner = new ThreadTuner();
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
   * y rows for every x row against N weight rows (`fn`: GEMV_F32 / GEMV_Q8 / GEMV_Q4; x as float
   * rows, or int16 rows with block scales xs; pointers as the kernels take them), split across the
   * helper threads when large enough.
   */
  gemm(fn: number, M: number, N: number, K: number, x: number, xs: number, w: number, s: number, mn: number, xsum: number, y: number): void {
    const c = this.ctrl;
    const parts = c ? Math.min(1 + Atomics.load(c, C_READY), Math.ceil(N / 8)) : 1;
    if (!c || parts < 2 || M * N * K < THREAD_MIN_MACS) {
      runRows(this.kernels, fn, M, N, K, x, xs, w, s, mn, xsum, y, 0, N);
      return;
    }
    const key = `${fn}:${M}:${N}:${K}`;
    const threaded = this.tuner.choose(key);
    const t0 = performance.now();
    if (threaded) this.gemmThreaded(c, parts, fn, M, N, K, x, xs, w, s, mn, xsum, y);
    else runRows(this.kernels, fn, M, N, K, x, xs, w, s, mn, xsum, y, 0, N);
    this.tuner.record(key, threaded, performance.now() - t0);
  }

  private gemmThreaded(c: Int32Array, parts: number, fn: number, M: number, N: number, K: number, x: number, xs: number, w: number, s: number, mn: number, xsum: number, y: number): void {
    c[C_FN] = fn; c[C_M] = M; c[C_N] = N; c[C_K] = K; c[C_X] = x; c[C_XS] = xs; c[C_W] = w; c[C_S] = s; c[C_MN] = mn; c[C_XSUM] = xsum; c[C_Y] = y; c[C_PARTS] = parts;
    Atomics.store(c, C_DONE, 0);
    Atomics.add(c, C_GEN, 1);
    Atomics.notify(c, C_GEN);
    const [n0, n1] = slice(N, parts, 0);
    runRows(this.kernels, fn, M, N, K, x, xs, w, s, mn, xsum, y, n0, n1);
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
