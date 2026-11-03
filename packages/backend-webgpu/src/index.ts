import {
  physical,
  roundUp,
  type Backend,
  type Binding,
  type Buf,
  type BufUsage,
  type DeviceLimits,
  type Dispatch,
  type KernelSpec,
  type Pipeline,
} from "@kiln/core";

class GPUBuf implements Buf {
  constructor(
    readonly gpu: GPUBuffer,
    readonly size: number,
    readonly label: string,
  ) {}
}

/** The GPU buffer behind a handle (a compiled plan intermediate resolves to its pool buffer). */
const gpuOf = (buf: Buf): GPUBuffer => (physical(buf) as GPUBuf).gpu;

class GPUPipe implements Pipeline {
  constructor(
    readonly key: string,
    readonly gpu: GPUComputePipeline,
    readonly layout: GPUBindGroupLayout,
  ) {}
}

class GPUBinding implements Binding {
  constructor(
    readonly pipeline: GPUPipe,
    readonly group: GPUBindGroup,
  ) {}
}

const OPTIONAL_FEATURES = ["shader-f16", "timestamp-query", "subgroups"] as GPUFeatureName[];

export interface WebGPUOptions {
  powerPreference?: GPUPowerPreference;
  /** Validate every kernel and report compile errors with the kernel key. On by default. */
  validateKernels?: boolean;
}

export async function createWebGPUBackend(opts: WebGPUOptions = {}): Promise<WebGPUBackend> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) throw new Error("WebGPU is not available in this environment (navigator.gpu is missing)");
  const adapter = await gpu.requestAdapter({ powerPreference: opts.powerPreference ?? "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter found");
  const requiredFeatures = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    },
  });
  const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo & { subgroupMinSize?: number } }).info;
  const deviceId = info ? [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join("/") : "unknown";
  const subgroupMinSize = device.features.has("subgroups" as GPUFeatureName) ? (info?.subgroupMinSize ?? 0) : 0;
  return new WebGPUBackend(device, opts.validateKernels ?? true, deviceId, subgroupMinSize);
}

export class WebGPUBackend implements Backend {
  readonly name = "webgpu";
  readonly limits: DeviceLimits;
  /** Set once the device is lost; every later call fails with this reason. */
  lostReason: string | undefined;

  private readonly pipelines = new Map<string, GPUPipe>();
  private readonly staging = new Map<number, GPUBuffer[]>();

  constructor(
    readonly device: GPUDevice,
    private readonly validate: boolean,
    readonly deviceId = "unknown",
    subgroupMinSize = 0,
  ) {
    this.limits = {
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
      features: new Set(device.features as unknown as Iterable<string>),
      subgroupMinSize,
    };
    device.lost.then((info) => {
      this.lostReason = `${info.reason}: ${info.message}`;
      console.error(`[kiln] WebGPU device lost (${this.lostReason})`);
    });
    device.addEventListener("uncapturederror", (e) => {
      console.error("[kiln] uncaptured WebGPU error:", (e as GPUUncapturedErrorEvent).error.message);
    });
  }

  alloc(bytes: number, usage: BufUsage = "storage", label = "buf"): Buf {
    this.checkAlive();
    const size = roundUp(Math.max(bytes, 4), 4);
    if (size > this.limits.maxStorageBufferBindingSize && usage === "storage") {
      throw new Error(
        `[kiln] buffer "${label}" needs ${(size / 2 ** 20).toFixed(1)} MiB but this device binds at most ` +
          `${(this.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(1)} MiB per buffer. Try quant: "q4" or a smaller model.`,
      );
    }
    const flags =
      usage === "uniform"
        ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    return new GPUBuf(this.device.createBuffer({ size, usage: flags, label }), size, label);
  }

  free(buf: Buf): void {
    gpuOf(buf).destroy();
  }

  write(buf: Buf, offset: number, data: ArrayBufferView): void {
    let bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    if (bytes.byteLength % 4 !== 0) {
      const padded = new Uint8Array(roundUp(bytes.byteLength, 4));
      padded.set(bytes);
      bytes = padded;
    }
    this.device.queue.writeBuffer(gpuOf(buf), offset, bytes);
  }

  copy(src: Buf, srcOffset: number, dst: Buf, dstOffset: number, bytes: number): void {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(gpuOf(src), srcOffset, gpuOf(dst), dstOffset, bytes);
    this.device.queue.submit([enc.finish()]);
  }

  pipeline(spec: KernelSpec): Pipeline {
    const cached = this.pipelines.get(spec.key);
    if (cached) return cached;
    if (this.validate) this.device.pushErrorScope("validation");
    const module = this.device.createShaderModule({ code: spec.code, label: spec.key });
    const layout = this.device.createBindGroupLayout({
      label: spec.key,
      entries: spec.bindings.map((kind, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: kind === "uniform" ? "uniform" : kind === "read" ? "read-only-storage" : "storage" },
      })),
    });
    const gpu = this.device.createComputePipeline({
      label: spec.key,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: spec.entry ?? "main" },
    });
    if (this.validate) {
      void this.device.popErrorScope().then(async (err) => {
        if (!err) return;
        const info = await module.getCompilationInfo();
        const lines = info.messages.map((m) => `  ${m.type} ${m.lineNum}:${m.linePos} ${m.message}`).join("\n");
        console.error(`[kiln] kernel "${spec.key}" failed: ${err.message}\n${lines}`);
      });
    }
    const pipe = new GPUPipe(spec.key, gpu, layout);
    this.pipelines.set(spec.key, pipe);
    return pipe;
  }

  bind(pipeline: Pipeline, buffers: readonly Buf[]): Binding {
    const pipe = pipeline as GPUPipe;
    const group = this.device.createBindGroup({
      label: pipe.key,
      layout: pipe.layout,
      entries: buffers.map((b, binding) => ({ binding, resource: { buffer: gpuOf(b) } })),
    });
    return new GPUBinding(pipe, group);
  }

  submit(dispatches: readonly Dispatch[]): void {
    this.checkAlive();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    let current: GPUPipe | undefined;
    for (const d of dispatches) {
      const b = d.binding as GPUBinding;
      if (b.pipeline !== current) {
        pass.setPipeline(b.pipeline.gpu);
        current = b.pipeline;
      }
      pass.setBindGroup(0, b.group);
      pass.dispatchWorkgroups(d.grid[0], d.grid[1], d.grid[2]);
    }
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  async readback(buf: Buf, offset: number, bytes: number): Promise<ArrayBuffer> {
    this.checkAlive();
    const size = roundUp(bytes, 4);
    const pool = this.staging.get(size) ?? [];
    this.staging.set(size, pool);
    const staging =
      pool.pop() ??
      this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "staging" });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(gpuOf(buf), offset, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = staging.getMappedRange().slice(0, bytes);
    staging.unmap();
    pool.push(staging);
    return out;
  }

  destroy(): void {
    for (const pool of this.staging.values()) for (const b of pool) b.destroy();
    this.staging.clear();
    this.pipelines.clear();
    this.device.destroy();
  }

  private checkAlive(): void {
    if (this.lostReason) throw new Error(`[kiln] WebGPU device was lost: ${this.lostReason}`);
  }
}
