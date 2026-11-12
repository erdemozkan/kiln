import {
  physical,
  roundUp,
  summarizeProfile,
  type Backend,
  type Binding,
  type Buf,
  type BufUsage,
  type DeviceLimits,
  type Dispatch,
  type KernelSpec,
  type Pipeline,
  type Profile,
} from "@kiln/core";

/** Timestamp queries per query set (WebGPU allows at most 4096). */
const PROFILE_QUERIES = 4096;

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
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    },
  });
  const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo & { subgroupMinSize?: number } }).info;
  const deviceId = info ? [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join("/") : "unknown";
  const subgroupMinSize = device.features.has("subgroups" as GPUFeatureName) ? (info?.subgroupMinSize ?? 0) : 0;
  return new WebGPUBackend(device, opts.validateKernels ?? true, deviceId, subgroupMinSize);
}

/** Largest readback staging buffer kept for reuse. */
const STAGING_POOL_MAX = 4 * 2 ** 20;

export class WebGPUBackend implements Backend {
  readonly name = "webgpu";
  readonly limits: DeviceLimits;
  /** Set once the device is lost; every later call fails with this reason. */
  lostReason: string | undefined;

  private readonly pipelines = new Map<string, GPUPipe>();
  private readonly staging = new Map<number, GPUBuffer[]>();
  /** While profiling: the timings read so far and the reads still in flight. */
  private profiling: { samples: { label: string; ms: number }[]; pending: Promise<void>[] } | undefined;

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
      maxWorkgroupStorage: device.limits.maxComputeWorkgroupStorageSize,
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
    if (this.profiling) {
      for (let i = 0; i < dispatches.length; i += PROFILE_QUERIES / 2) this.submitTimed(dispatches.slice(i, i + PROFILE_QUERIES / 2));
      return;
    }
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

  /**
   * Times every dispatch until `stopProfile`, with timestamp queries: each dispatch runs in a compute
   * pass of its own, which adds a little overhead per dispatch. Browsers may coarsen timestamps (Chrome
   * can round them to 100 µs), so one short dispatch is imprecise; totals over many calls are not.
   */
  startProfile(): void {
    if (!this.limits.features.has("timestamp-query")) throw new Error("[kiln] profiling needs the timestamp-query feature, which this device does not offer");
    this.profiling = { samples: [], pending: [] };
  }

  async stopProfile(): Promise<Profile> {
    const p = this.profiling;
    if (!p) throw new Error("[kiln] stopProfile without startProfile");
    this.profiling = undefined;
    await Promise.all(p.pending);
    return summarizeProfile(p.samples, "GPU time per dispatch from timestamp queries (each dispatch in its own pass; a single short dispatch may be imprecise, totals over many calls are reliable)");
  }

  private submitTimed(dispatches: readonly Dispatch[]): void {
    const p = this.profiling!;
    const n = dispatches.length * 2;
    const querySet = this.device.createQuerySet({ type: "timestamp", count: n, label: "profile" });
    const resolve = this.device.createBuffer({ size: n * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC, label: "profile" });
    const read = this.device.createBuffer({ size: n * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "profile" });
    const enc = this.device.createCommandEncoder();
    dispatches.forEach((d, i) => {
      const b = d.binding as GPUBinding;
      const pass = enc.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      pass.setPipeline(b.pipeline.gpu);
      pass.setBindGroup(0, b.group);
      pass.dispatchWorkgroups(d.grid[0], d.grid[1], d.grid[2]);
      pass.end();
    });
    enc.resolveQuerySet(querySet, 0, n, resolve, 0);
    enc.copyBufferToBuffer(resolve, 0, read, 0, n * 8);
    this.device.queue.submit([enc.finish()]);
    const labels = dispatches.map((d) => d.label);
    p.pending.push(
      read.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigUint64Array(read.getMappedRange());
        labels.forEach((label, i) => p.samples.push({ label, ms: t[2 * i + 1]! > t[2 * i]! ? Number(t[2 * i + 1]! - t[2 * i]!) / 1e6 : 0 }));
        read.unmap();
        read.destroy();
        resolve.destroy();
        querySet.destroy();
      }),
    );
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
    // Large ones (weights read back for the cache) are not kept: a pool of them would hold their memory for good.
    if (size <= STAGING_POOL_MAX) pool.push(staging);
    else staging.destroy();
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
