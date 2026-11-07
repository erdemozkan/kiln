import { WasmHeap } from "./simd";
import { physical, summarizeProfile, type Backend, type Binding, type Buf, type BufUsage, type CpuKernel, type CpuView, type DeviceLimits, type Dispatch, type KernelSpec, type Pipeline, type Profile } from "@kiln/core";

/**
 * A buffer of the CPU backend, with the views kernels read it through: a region of the backend's
 * WASM heap when it has one (so the SIMD kernels can address it), else its own ArrayBuffer.
 */
class CpuBuf implements Buf, CpuView {
  private data: ArrayBuffer | undefined;
  private views!: { bytes: Uint8Array; f32: Float32Array; u32: Uint32Array; i32: Int32Array; i8: Int8Array };
  readonly ptr: number;

  constructor(
    readonly size: number,
    readonly label: string,
    readonly heap?: WasmHeap,
  ) {
    this.ptr = heap ? heap.alloc(size) : 0;
    if (!heap) this.view(new ArrayBuffer(Math.ceil(size / 4) * 4));
  }

  /** The views, made again when the heap has grown (which detaches the old ones). */
  private get v() {
    const buf = this.heap ? this.heap.buffer : this.data!;
    if (buf !== this.data) this.view(buf);
    return this.views;
  }

  private view(buf: ArrayBuffer): void {
    this.data = buf;
    const at = this.ptr;
    const words = Math.ceil(this.size / 4);
    this.views = {
      bytes: new Uint8Array(buf, at, words * 4),
      f32: new Float32Array(buf, at, words),
      u32: new Uint32Array(buf, at, words),
      i32: new Int32Array(buf, at, words),
      i8: new Int8Array(buf, at, words * 4),
    };
  }

  get bytes(): Uint8Array { return this.v.bytes; }
  get f32(): Float32Array { return this.v.f32; }
  get u32(): Uint32Array { return this.v.u32; }
  get i32(): Int32Array { return this.v.i32; }
  get i8(): Int8Array { return this.v.i8; }

  private freed = false;

  release(): void {
    if (!this.heap || this.freed) return;
    this.freed = true;
    this.heap.release(this.ptr, Math.ceil(this.size / 4) * 4);
  }
}

interface CpuPipeline extends Pipeline {
  readonly run: CpuKernel;
}

interface CpuBinding extends Binding {
  readonly pipeline: CpuPipeline;
  readonly bufs: readonly CpuBuf[];
}

const asCpu = (buf: Buf): CpuBuf => {
  // A plan intermediate resolves to the pool buffer it was placed in.
  const b = physical(buf);
  if (!(b instanceof CpuBuf)) throw new Error(`buffer "${b.label}" does not belong to the CPU backend`);
  return b;
};

/**
 * Runs every kernel on the CPU (each kernel spec's `cpu` implementation), for browsers and devices
 * without WebGPU: matmuls in WASM SIMD where available (simd.wat), everything else in TypeScript.
 * Work runs when it is submitted, on the calling thread (the model's worker), so queue order is
 * program order. Much slower than a GPU: meant for small models.
 */
export class CpuBackend implements Backend {
  readonly name = "cpu";
  readonly deviceId = "cpu";
  readonly limits: DeviceLimits = {
    maxBufferSize: 2 ** 31,
    maxStorageBufferBindingSize: 2 ** 31,
    maxComputeWorkgroupsPerDimension: 65535,
    features: new Set<string>(),
    maxWorkgroupStorage: 16384,
  };

  /** Where buffers live: a WASM heap the SIMD kernels use, when WebAssembly SIMD is available. */
  readonly heap: WasmHeap | undefined;

  constructor(opts: { simd?: boolean } = {}) {
    this.heap = opts.simd === false ? undefined : WasmHeap.create();
  }

  /** Whether matmuls run in WASM SIMD (else in plain TypeScript). */
  get simd(): boolean {
    return !!this.heap;
  }

  alloc(bytes: number, _usage: BufUsage = "storage", label = "buf"): Buf {
    return new CpuBuf(bytes, label, this.heap);
  }

  free(buf: Buf): void {
    // Heap regions go back to the free list; plain ArrayBuffers are garbage collected.
    if (buf instanceof CpuBuf) buf.release();
  }

  write(buf: Buf, offset: number, data: ArrayBufferView): void {
    asCpu(buf).bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
  }

  copy(src: Buf, srcOffset: number, dst: Buf, dstOffset: number, bytes: number): void {
    asCpu(dst).bytes.set(asCpu(src).bytes.subarray(srcOffset, srcOffset + bytes), dstOffset);
  }

  pipeline(spec: KernelSpec): Pipeline {
    if (!spec.cpu) throw new Error(`kernel ${spec.key} has no CPU implementation`);
    return { key: spec.key, run: spec.cpu } as CpuPipeline;
  }

  bind(pipeline: Pipeline, buffers: readonly Buf[]): Binding {
    const binding: CpuBinding = { pipeline: pipeline as CpuPipeline, bufs: buffers.map(asCpu) };
    return binding;
  }

  submit(dispatches: readonly Dispatch[]): void {
    const prof = this.profiling;
    for (const d of dispatches) {
      const b = d.binding as CpuBinding;
      const t0 = prof ? performance.now() : 0;
      b.pipeline.run(b.bufs, d.grid);
      if (prof) prof.push({ label: d.label, ms: performance.now() - t0 });
    }
  }

  private profiling: { label: string; ms: number }[] | undefined;

  /** Times every kernel until `stopProfile` (wall clock around each one). */
  startProfile(): void {
    this.profiling = [];
  }

  async stopProfile(): Promise<Profile> {
    const samples = this.profiling;
    if (!samples) throw new Error("[kiln] stopProfile without startProfile");
    this.profiling = undefined;
    return summarizeProfile(samples, "wall-clock time per kernel on the CPU backend");
  }

  async readback(buf: Buf, offset: number, bytes: number): Promise<ArrayBuffer> {
    return asCpu(buf).bytes.slice(offset, offset + bytes).buffer;
  }

  destroy(): void {}
}
