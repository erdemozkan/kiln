import { byteSize, f32, packUniform, q4ScaleBytes, roundUp, u32, type Backend, type Buf, type Tensor } from "@kiln/core";
import { GPU_EXACT_TYPES, convertSpec, exactSpec, gpuConverts, scaleSpec, type GpuExactType, grid1d, MAX_X, type GpuConvertOut, type GpuConvertType } from "@kiln/kernels";
import { exactPlane, exactQ4, exactQ8, quantizeQ4, quantizeQ8, repackQ4_0, repackQ8_0, rowBytes, toF32, type ByteSource, type TensorEntry } from "@kiln/loader";
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
): Promise<Map<string, Tensor>> {
  const out = new Map<string, Tensor>();
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
        if (name) out.set(name, await loadTensor(backend, file.source, entry, name, quant, onBytes, file.transform, gpu, file.scale?.(name)));
        else onBytes(entry.byteLength);
      }
    }
  } finally {
    gpu?.destroy();
  }
  return out;
}

async function loadTensor(
  backend: Backend,
  src: ByteSource,
  e: TensorEntry,
  name: string,
  quant: Quant,
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
  // Exact planes: a repack (Q4_0, Q8_0) or code·scale (+ min) per block, never a second rounding.
  const isExact = dtype === exact;
  const affine = isExact && (e.type === "Q4_1" || e.type === "Q4_K" || e.type === "Q5_1" || e.type === "Q5_K");

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
