import { physical, type Backend, type Binding, type Buf, type BufUsage, type CpuKernel, type CpuView, type DeviceLimits, type Dispatch, type KernelSpec, type Pipeline } from "@kiln/core";

/** A buffer of the CPU backend: one ArrayBuffer with the views kernels read it through. */
class CpuBuf implements Buf, CpuView {
  readonly bytes: Uint8Array;
  readonly f32: Float32Array;
  readonly u32: Uint32Array;
  readonly i32: Int32Array;
  readonly i8: Int8Array;

  constructor(
    readonly size: number,
    readonly label: string,
  ) {
    const data = new ArrayBuffer(Math.ceil(size / 4) * 4);
    this.bytes = new Uint8Array(data);
    this.f32 = new Float32Array(data);
    this.u32 = new Uint32Array(data);
    this.i32 = new Int32Array(data);
    this.i8 = new Int8Array(data);
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
 * Runs every kernel on the CPU, in TypeScript (each kernel spec's `cpu` implementation), for
 * browsers and devices without WebGPU. Work runs when it is submitted, on the calling thread (the
 * model's worker), so queue order is program order. Much slower than a GPU: meant for small models.
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

  alloc(bytes: number, _usage: BufUsage = "storage", label = "buf"): Buf {
    return new CpuBuf(bytes, label);
  }

  free(_buf: Buf): void {
    // Garbage collected once nothing references it.
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
    for (const d of dispatches) {
      const b = d.binding as CpuBinding;
      b.pipeline.run(b.bufs, d.grid);
    }
  }

  async readback(buf: Buf, offset: number, bytes: number): Promise<ArrayBuffer> {
    return asCpu(buf).bytes.slice(offset, offset + bytes).buffer;
  }

  destroy(): void {}
}
