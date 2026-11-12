import { byteSize, f32, packUniform, q4ScaleBytes, roundUp, u32, type Backend, type Buf, type Tensor } from "@kiln/core";
import { GPU_EXACT_TYPES, convertSpec, exactSpec, gpuConverts, scaleSpec, type GpuExactType, grid1d, MAX_X, type GpuConvertOut, type GpuConvertType } from "@kiln/kernels";
import { exactPlane, exactQ4, exactQ8, openConverted, quantizeQ4, quantizeQ8, repackQ4_0, repackQ8_0, rowBytes, toF32, type ByteSource, type ConvertedReader, type ConvertedWriter, type TensorEntry } from "@kiln/loader";
import { isQuantizable } from "@kiln/models";

/**
 * Weight storage on the GPU for linear and embedding matrices: f32, q8 (8-bit, a quarter of f32 and
 * close to it in accuracy) or q4 (4-bit, an eighth). Norms and biases always stay f32.
 *
 * GGUF block types that Kiln's planes hold exactly load that way under every setting but "f32" (and
 * "q4" for the 5- and 8-bit ones): Q4_0, Q4_1 and Q4_K as q4; Q5_0, Q5_1, Q5_K and Q8_0 as q8.
 * "auto" (the default) also keeps Q6_K at 8 bits (q8, one fine rounding) and stores float weights as
 * q4, so a GGUF file keeps the precision its quantizer chose per tensor.
 */
export type Quant = "auto" | "f32" | "q8" | "q4";

export interface LoadProgress {
  file: string;
  /** Bytes of this file uploaded to the GPU so far. */
  loaded: number;
  total: number;
}

export interface WeightFile {
  name: string;
  source: ByteSource;
  tensors: TensorEntry[];
  /** Maps the file's tensor names to model names; undefined skips the tensor. */
  rename?: (name: string) => string | undefined;
  /**
   * Edits values in place before upload, by model name, for tensors stored as f32 (norms and biases are
   * never quantized), e.g. Gemma's 1 + w norms.
   */
  transform?: (name: string, values: Float32Array) => void;
  /**
   * A factor to multiply a tensor by, by model name (Granite's residual and logit multipliers, folded
   * into the weights). Applied to f32 values, or to a quantized tensor's scales.
   */
  scale?: (name: string) => number | undefined;
}

/** Bytes read per step. Bounds peak CPU memory regardless of model size. */
const CHUNK = 8 * 2 ** 20;

let gpuConvert = true;

/**
 * Converts weights to q4 or q8 on the GPU (default true) from F32/F16/BF16 safetensors and every GGUF
 * block type Kiln reads: raw blocks are uploaded and a kernel writes the planes, which takes the
 * conversion off the CPU, where it was most of a cached load. False converts on the CPU; the planes
 * are the same bits either way.
 */
export function setGpuConvert(on: boolean): void {
  gpuConvert = on;
}

/** One staging buffer for the raw blocks and one uniform, reused for every chunk in queue order. */
class GpuConverter {
  private staging: Buf | undefined;
  private readonly uniform: Buf;

  constructor(private readonly backend: Backend) {
    this.uniform = backend.alloc(16, "uniform", "convert/params");
  }

  convert(type: GpuConvertType, out: GpuConvertOut, raw: Uint8Array, blocks: number, dst0: number, data: Buf, scales: Buf): void {
    const be = this.backend;
    this.stage(raw);
    be.write(this.uniform, 0, packUniform([u32(blocks), u32(dst0)]));
    const binding = be.bind(be.pipeline(convertSpec(type, out)), [this.uniform, this.staging!, data, scales]);
    be.submit([{ binding, grid: grid1d(blocks), label: `convert/${type}/${out}` }]);
  }

  /** Writes exact planes (codes, scales, and mins for the affine types) from GGUF blocks. */
  exact(type: GpuExactType, raw: Uint8Array, blocks: number, dst0: number, data: Buf, scales: Buf, mins?: Buf): void {
    const be = this.backend;
    this.stage(raw);
    be.write(this.uniform, 0, packUniform([u32(blocks), u32(dst0)]));
    const binding = be.bind(be.pipeline(exactSpec(type)), mins ? [this.uniform, this.staging!, data, scales, mins] : [this.uniform, this.staging!, data, scales]);
    be.submit([{ binding, grid: grid1d(blocks), label: `convert/${type}/exact` }]);
  }

  private stage(raw: Uint8Array): void {
    const be = this.backend;
    if (!this.staging || this.staging.size < raw.byteLength) {
      if (this.staging) be.free(this.staging);
      this.staging = be.alloc(roundUp(Math.max(raw.byteLength, CHUNK), 4), "storage", "convert/raw");
    }
    be.write(this.staging, 0, raw);
  }

  /** Multiplies the first `n` floats of `buf` by `f`, after the conversions already queued. */
  scale(buf: Buf, n: number, f: number): void {
    const be = this.backend;
    be.write(this.uniform, 0, packUniform([u32(n), f32(f)]));
    const binding = be.bind(be.pipeline(scaleSpec()), [this.uniform, buf]);
    be.submit([{ binding, grid: grid1d(n), label: "convert/scale" }]);
  }

  destroy(): void {
    // Destroying a buffer leaves already-submitted work that uses it to finish.
    if (this.staging) this.backend.free(this.staging);
    this.backend.free(this.uniform);
  }
}

/** How a tensor is held on the GPU: decided by its name, shape, stored type and the `Quant` setting. */
interface Layout {
  shape: number[];
  rows: number;
  cols: number;
  dtype: "f32" | "q4" | "q8";
  /** Planes that hold the file's own codes (a repack, or code·scale + min), never a second rounding. */
  isExact: boolean;
  /** Exact planes with a minimum per block beside the scale. */
  affine: boolean;
  bufBytes: number;
  /** Bytes of scales (and of mins, when affine); 0 for f32. */
  scaleBytes: number;
}

function layoutOf(e: TensorEntry, name: string, quant: Quant): Layout {
  if (e.transposed) throw new Error(`${name}: stored transposed (non-contiguous); only tensors read on the CPU can be`);
  const shape = e.shape.length === 1 ? e.shape : [e.shape.slice(0, -1).reduce((a, b) => a * b, 1), e.shape[e.shape.length - 1]!];
  const cols = shape[shape.length - 1]!;
  const rows = shape.length === 2 ? shape[0]! : 1;
  const quantizable = isQuantizable(name, shape);
  if (e.type === "Q4_0" && !quantizable) throw new Error(`${name}: Q4_0 tensor with shape [${shape}] cannot stay quantized`);
  const exact = exactPlane(e.type);
  const dtype: "f32" | "q4" | "q8" = !quantizable
    ? "f32"
    : e.type === "Q4_0"
      ? "q4"
      : quant === "f32"
        ? "f32"
        : exact === "q4" || (exact === "q8" && quant !== "q4")
          ? exact
          : quant === "auto"
            ? e.type === "Q6_K" ? "q8" : "q4"
            : quant;
  const isExact = dtype === exact;
  const affine = isExact && (e.type === "Q4_1" || e.type === "Q4_K" || e.type === "Q5_1" || e.type === "Q5_K");
  return { shape, rows, cols, dtype, isExact, affine, bufBytes: dtype === "f32" ? rows * cols * 4 : rows * byteSize([cols], dtype), scaleBytes: dtype === "f32" ? 0 : q4ScaleBytes(shape) };
}

const layoutBytes = (l: Layout): number => l.bufBytes + l.scaleBytes * (l.affine ? 2 : 1);

/**
 * Bumped whenever the same file and setting would give different bytes on the GPU (a converter, a
 * model's `transform` or `scale`), so converted weights cached by an older build are made again.
 */
const CONVERTED_VERSION = 1;
/** Converted weights are cached when they are at most this fraction of the bytes they were read from. */
const CONVERTED_MAX_RATIO = 0.75;
/** Readbacks waiting to be written to the cache before the load pauses for them. */
const CONVERTED_PENDING = 64 * 2 ** 20;

/** Where `loadWeights` keeps converted weights between loads. */
export interface ConvertedCache {
  /** Names the weights among everything cached: the repo, revision and server, and which part of the model. */
  key: string;
}

/** 32-bit FNV-1a, for a short fingerprint of a tensor list. */
function fnv1a(text: string, h = 0x811c9dc5): number {
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193);
  return h >>> 0;
}

interface Planned {
  file: WeightFile;
  entry: TensorEntry;
  name: string;
  layout: Layout;
}

/** Every tensor a load keeps, in order, with its place on the GPU. */
function planOf(files: readonly WeightFile[], quant: Quant): Planned[] {
  const plan: Planned[] = [];
  for (const file of files) {
    for (const entry of file.tensors) {
      const name = file.rename ? file.rename(entry.name) : entry.name;
      if (name) plan.push({ file, entry, name, layout: layoutOf(entry, name, quant) });
    }
  }
  return plan;
}

/**
 * Opens the cache entry for a load, or nothing when caching would not pay: the converted weights
 * must be clearly smaller than their source (f32 from an F32 file, or a GGUF's own planes, are read
 * as fast from the original).
 */
async function openCache(cache: ConvertedCache | undefined, plan: readonly Planned[], quant: Quant): Promise<ConvertedReader | ConvertedWriter | undefined> {
  if (!cache || !plan.length) return undefined;
  let bytes = 0;
  let source = 0;
  let print = fnv1a(quant);
  for (const { entry, name, layout } of plan) {
    bytes += layoutBytes(layout);
    source += entry.byteLength;
    print = fnv1a(`|${name}:${entry.type}:${layout.dtype}:${layout.affine ? "a" : ""}:${layout.shape.join("x")}:${entry.byteLength}`, print);
  }
  if (bytes > source * CONVERTED_MAX_RATIO) return undefined;
  return openConverted(`${cache.key}|${quant}`, `v${CONVERTED_VERSION}/${plan.length}/${source}/${print.toString(16)}`, bytes);
}

/** Loads every planned tensor from a cache entry: the bytes go to the GPU as they were read back from it. */
function loadConverted(backend: Backend, reader: ConvertedReader, plan: readonly Planned[], out: Map<string, Tensor>, onProgress?: (p: LoadProgress) => void): void {
  const total = plan.reduce((s, p) => s + layoutBytes(p.layout), 0);
  const file = plan[0]!.file.name;
  let at = 0;
  onProgress?.({ file, loaded: 0, total });
  const fill = (buf: Buf, bytes: number) => {
    for (let o = 0; o < bytes; o += CHUNK) {
      const n = Math.min(CHUNK, bytes - o);
      backend.write(buf, o, reader.read(at, n));
      at += n;
      onProgress?.({ file, loaded: at, total });
    }
  };
  for (const { name, layout: l } of plan) {
    const t: Tensor = {
      shape: l.shape,
      dtype: l.dtype,
      buf: backend.alloc(l.bufBytes, "storage", name),
      scales: l.dtype !== "f32" ? backend.alloc(l.scaleBytes, "storage", `${name}/scales`) : undefined,
      mins: l.affine ? backend.alloc(l.scaleBytes, "storage", `${name}/mins`) : undefined,
    };
    // In the map before its bytes arrive, so a failed read frees it with the rest.
    out.set(name, t);
    fill(t.buf, l.bufBytes);
    if (t.scales) fill(t.scales, l.scaleBytes);
    if (t.mins) fill(t.mins, l.scaleBytes);
  }
}

/** Reads loaded tensors back from the GPU, in order, into a cache entry; writes trail the readbacks. */
class ConvertedSink {
  private readonly queue: Promise<ArrayBuffer>[] = [];
  private pending = 0;
  private ok = true;

  constructor(
    private readonly backend: Backend,
    private readonly writer: ConvertedWriter,
  ) {}

  async add(t: Tensor, l: Layout): Promise<void> {
    if (!this.ok) return;
    for (const [buf, bytes] of [[t.buf, l.bufBytes], [t.scales, l.scaleBytes], [t.mins, l.scaleBytes]] as const) {
      if (!buf) continue;
      for (let o = 0; o < bytes; o += CHUNK) {
        const n = Math.min(CHUNK, bytes - o);
        // Enqueued now, so it reads the tensor as loaded whatever the model does with the buffer next.
        const read = this.backend.readback(buf, o, n);
        read.catch(() => undefined);
        this.queue.push(read);
        this.pending += n;
      }
    }
    if (this.pending >= CONVERTED_PENDING) await this.drain();
  }

  private async drain(): Promise<void> {
    const queue = this.queue.splice(0);
    this.pending = 0;
    for (const read of queue) {
      if (!this.ok) continue;
      try {
        this.ok = this.writer.append(new Uint8Array(await read));
      } catch {
        // A failed readback (a lost device) fails the load elsewhere; here it only ends the caching.
        this.ok = false;
        this.writer.abort();
      }
    }
  }

  async finish(): Promise<void> {
    await this.drain();
    if (this.ok) await this.writer.commit();
  }

  abort(): void {
    this.ok = false;
    this.writer.abort();
  }
}

/**
 * Streams tensors from their sources straight into GPU buffers, converting row-chunks on the way:
 * F16/BF16/K-quants → f32, q8 or q4; GGUF Q8_0 → repacked q8 (unless q4 is asked for), GGUF Q4_0 →
 * repacked q4 whatever the setting. The full model never sits in CPU memory.
 */
export async function loadWeights(
  backend: Backend,
  files: readonly WeightFile[],
  quant: Quant,
  onProgress?: (p: LoadProgress) => void,
  cache?: ConvertedCache,
): Promise<Map<string, Tensor>> {
  const out = new Map<string, Tensor>();
  const plan = planOf(files, quant);
  let entry = await openCache(cache, plan, quant);
  if (entry?.mode === "read") {
    try {
      loadConverted(backend, entry, plan, out, onProgress);
      return out;
    } catch (err) {
      // The entry was dropped by the failed read: load from the files as if there were none.
      freeWeights(backend, out);
      console.warn(`[kiln] cached converted weights could not be read (${String(err)}); converting again`);
    } finally {
      entry.close();
    }
    entry = undefined;
  }
  const sink = entry ? new ConvertedSink(backend, entry) : undefined;
  // The CPU backend has no convert kernels; its planes come from the CPU converters, the same bits.
  const gpu = gpuConvert && backend.name !== "cpu" ? new GpuConverter(backend) : undefined;
  try {
    for (const file of files) {
      const total = file.tensors.reduce((s, t) => s + t.byteLength, 0);
      let loaded = 0;
      onProgress?.({ file: file.name, loaded, total });
      // Report per chunk, not per tensor: a single embedding can be tens of MB.
      const onBytes = (n: number) => {
        loaded += n;
        onProgress?.({ file: file.name, loaded, total });
      };
      for (const entry of file.tensors) {
        const name = file.rename ? file.rename(entry.name) : entry.name;
        if (!name) {
          onBytes(entry.byteLength);
          continue;
        }
        const layout = layoutOf(entry, name, quant);
        const t = await loadTensor(backend, file.source, entry, name, layout, onBytes, file.transform, gpu, file.scale?.(name));
        out.set(name, t);
        await sink?.add(t, layout);
      }
    }
    await sink?.finish();
  } catch (err) {
    sink?.abort();
    // A file that fails part-way must not strand the tensors already on the GPU.
    freeWeights(backend, out);
    throw err;
  } finally {
    gpu?.destroy();
  }
  return out;
}

/** Frees every buffer of loaded weights (planes, scales and mins). */
export function freeWeights(backend: Backend, weights: Map<string, Tensor>): void {
  for (const t of weights.values()) {
    backend.free(t.buf);
    if (t.scales) backend.free(t.scales);
    if (t.mins) backend.free(t.mins);
  }
  weights.clear();
}

async function loadTensor(
  backend: Backend,
  src: ByteSource,
  e: TensorEntry,
  name: string,
  layout: Layout,
  onBytes: (n: number) => void,
  transform?: (name: string, values: Float32Array) => void,
  gpu?: GpuConverter,
  factor?: number,
): Promise<Tensor> {
  const scaleAll = (v: Float32Array | undefined) => {
    if (!v) return;
    // An f32 factor, as the GPU uses: the product of two f32 is exact in a double, so one rounding.
    const f = Math.fround(factor ?? 1);
    if (f !== 1) for (let i = 0; i < v.length; i++) v[i] = v[i]! * f;
  };
  const { shape, rows, cols, dtype, isExact, affine } = layout;

  const srcRow = rowBytes(e.type, cols);
  const rowsPerChunk = Math.max(1, Math.floor(CHUNK / srcRow));

  if (dtype !== "f32") {
    const perRow = byteSize([cols], dtype);
    const buf = backend.alloc(rows * perRow, "storage", name);
    const scales = backend.alloc(q4ScaleBytes(shape), "storage", `${name}/scales`);
    const mins = affine ? backend.alloc(q4ScaleBytes(shape), "storage", `${name}/mins`) : undefined;
    const fits = grid1d((rowsPerChunk * cols) / 32)[1] <= MAX_X;
    const exactOnGpu = gpu && isExact && GPU_EXACT_TYPES.has(e.type) && fits;
    const onGpu = gpu && (!isExact || e.type === "Q4_0") && gpuConverts(e.type, dtype) && fits;
    for (let r = 0; r < rows; r += rowsPerChunk) {
      const n = Math.min(rowsPerChunk, rows - r);
      const bytes = await src.read(e.offset + r * srcRow, n * srcRow);
      if (exactOnGpu) {
        gpu.exact(e.type as GpuExactType, bytes, (n * cols) / 32, (r * cols) / 32, buf, scales, mins);
        onBytes(n * srcRow);
        continue;
      }
      if (onGpu) {
        gpu.convert(e.type as GpuConvertType, dtype, bytes, (n * cols) / 32, (r * cols) / 32, buf, scales);
        onBytes(n * srcRow);
        continue;
      }
      let planes: { data: Uint32Array; scales: Float32Array; mins?: Float32Array };
      const asF32 = () => toF32(e.type as Exclude<typeof e.type, "Q4_0">, bytes, n * cols);
      if (dtype === "q4") {
        const p =
          e.type === "Q4_0" ? repackQ4_0(bytes, n * cols) : isExact ? exactQ4(e.type as "Q4_1" | "Q4_K", bytes, n * cols) : quantizeQ4(asF32());
        planes = { data: p.nibbles, scales: p.scales, mins: p.mins };
      } else {
        const p =
          e.type === "Q8_0" ? repackQ8_0(bytes, n * cols) : isExact ? exactQ8(e.type as "Q5_0" | "Q5_1" | "Q5_K", bytes, n * cols) : quantizeQ8(asF32());
        planes = { data: p.bytes, scales: p.scales, mins: p.mins };
      }
      scaleAll(planes.scales);
      scaleAll(planes.mins);
      backend.write(buf, r * perRow, planes.data);
      backend.write(scales, (r * cols) / 8, planes.scales);
      if (mins && planes.mins) backend.write(mins, (r * cols) / 8, planes.mins);
      onBytes(n * srcRow);
    }
    // GPU-converted scales get the factor once every chunk is in; CPU ones got it per chunk.
    if ((onGpu || exactOnGpu) && factor !== undefined && factor !== 1) {
      gpu.scale(scales, (rows * cols) / 32, factor);
      if (mins) gpu.scale(mins, (rows * cols) / 32, factor);
    }
    return { shape, dtype, buf, scales, mins };
  }

  const buf = backend.alloc(rows * cols * 4, "storage", name);
  for (let r = 0; r < rows; r += rowsPerChunk) {
    const n = Math.min(rowsPerChunk, rows - r);
    const bytes = await src.read(e.offset + r * srcRow, n * srcRow);
    const values = toF32(e.type as Exclude<typeof e.type, "Q4_0">, bytes, n * cols);
    transform?.(name, values);
    scaleAll(values);
    backend.write(buf, r * cols * 4, values);
    onBytes(n * srcRow);
  }
  return { shape, dtype: "f32", buf };
}
