import { SIMD_WASM } from "./simd-bytes";

/** The exports of simd.wat: one x row against N weight rows. Pointers are byte offsets. */
export interface SimdKernels {
  gemv_f32(x: number, w: number, K: number, N: number, y: number): void;
  gemv_q8(x: number, q: number, s: number, mn: number, xsum: number, K: number, N: number, y: number): void;
  gemv_q4(x: number, q: number, s: number, mn: number, xsum: number, K: number, N: number, y: number): void;
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

  private constructor(
    readonly memory: WebAssembly.Memory,
    readonly kernels: SimdKernels,
  ) {}

  /** A heap, or undefined where WebAssembly SIMD is unavailable. */
  static create(): WasmHeap | undefined {
    try {
      if (typeof WebAssembly === "undefined") return undefined;
      const bytes = Uint8Array.from(atob(SIMD_WASM), (c) => c.charCodeAt(0));
      if (!WebAssembly.validate(bytes)) return undefined;
      const memory = new WebAssembly.Memory({ initial: 16, maximum: 65536 });
      const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory } });
      return new WasmHeap(memory, inst.exports as unknown as SimdKernels);
    } catch {
      return undefined;
    }
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
