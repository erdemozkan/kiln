import { byteSize, packUniform, q4ScaleBytes, roundUp, u32, type Backend, type Buf, type Tensor } from "@kiln/core";
import { convertSpec, grid1d, MAX_X, type GpuConvertType } from "@kiln/kernels";
import { quantizeQ4, quantizeQ8, repackQ4_0, repackQ8_0, rowBytes, toF32, type ByteSource, type TensorEntry } from "@kiln/loader";
import { isQuantizable } from "@kiln/models";

/**
 * Weight storage on the GPU for linear and embedding matrices: f32, q8 (8-bit, a quarter of f32 and
 * close to it in accuracy) or q4 (4-bit, an eighth). Norms and biases always stay f32.
 */
export type Quant = "f32" | "q8" | "q4";

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
}

/** Bytes read per step. Bounds peak CPU memory regardless of model size. */
const CHUNK = 8 * 2 ** 20;

let gpuConvert = true;

/**
 * Converts GGUF Q4_0 and Q6_K blocks to q4 on the GPU (default true): raw blocks are uploaded and a
 * kernel writes the planes, which takes the conversion off the CPU, where it was most of a cached
 * load. False converts on the CPU; the planes are the same bits either way.
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

  convert(type: GpuConvertType, raw: Uint8Array, blocks: number, dst0: number, nibbles: Buf, scales: Buf): void {
    const be = this.backend;
    if (!this.staging || this.staging.size < raw.byteLength) {
      if (this.staging) be.free(this.staging);
      this.staging = be.alloc(roundUp(Math.max(raw.byteLength, CHUNK), 4), "storage", "convert/raw");
    }
    be.write(this.staging, 0, raw);
    be.write(this.uniform, 0, packUniform([u32(blocks), u32(dst0)]));
    const binding = be.bind(be.pipeline(convertSpec(type)), [this.uniform, this.staging, nibbles, scales]);
    be.submit([{ binding, grid: grid1d(blocks), label: `convert/${type}` }]);
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
  const gpu = gpuConvert ? new GpuConverter(backend) : undefined;
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
        if (name) out.set(name, await loadTensor(backend, file.source, entry, name, quant, onBytes, file.transform, gpu));
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
): Promise<Tensor> {
  const shape = e.shape.length === 1 ? e.shape : [e.shape.slice(0, -1).reduce((a, b) => a * b, 1), e.shape[e.shape.length - 1]!];
  const cols = shape[shape.length - 1]!;
  const rows = shape.length === 2 ? shape[0]! : 1;
  const quantizable = isQuantizable(name, shape);
  if (e.type === "Q4_0" && !quantizable) throw new Error(`${name}: Q4_0 tensor with shape [${shape}] cannot stay quantized`);
  const dtype = !quantizable
    ? "f32"
    : quant === "q4" || e.type === "Q4_0"
      ? "q4"
      : quant === "q8" || (e.type === "Q8_0" && quant !== "f32")
        ? "q8"
        : "f32";

  const srcRow = rowBytes(e.type, cols);
  const rowsPerChunk = Math.max(1, Math.floor(CHUNK / srcRow));

  if (dtype !== "f32") {
    const perRow = byteSize([cols], dtype);
    const buf = backend.alloc(rows * perRow, "storage", name);
    const scales = backend.alloc(q4ScaleBytes(shape), "storage", `${name}/scales`);
    const onGpu = gpu && dtype === "q4" && (e.type === "Q4_0" || e.type === "Q6_K") && grid1d((rowsPerChunk * cols) / 32)[1] <= MAX_X;
    for (let r = 0; r < rows; r += rowsPerChunk) {
      const n = Math.min(rowsPerChunk, rows - r);
      const bytes = await src.read(e.offset + r * srcRow, n * srcRow);
      if (onGpu) {
        gpu.convert(e.type as GpuConvertType, bytes, (n * cols) / 32, (r * cols) / 32, buf, scales);
        onBytes(n * srcRow);
        continue;
      }
      let planes: { data: Uint32Array; scales: Float32Array };
      if (dtype === "q4") {
        const p = e.type === "Q4_0" ? repackQ4_0(bytes, n * cols) : quantizeQ4(toF32(e.type as Exclude<typeof e.type, "Q4_0">, bytes, n * cols));
        planes = { data: p.nibbles, scales: p.scales };
      } else {
        const p = e.type === "Q8_0" ? repackQ8_0(bytes, n * cols) : quantizeQ8(toF32(e.type as Exclude<typeof e.type, "Q4_0">, bytes, n * cols));
        planes = { data: p.bytes, scales: p.scales };
      }
      backend.write(buf, r * perRow, planes.data);
      backend.write(scales, (r * cols) / 8, planes.scales);
      onBytes(n * srcRow);
    }
    return { shape, dtype, buf, scales };
  }

  const buf = backend.alloc(rows * cols * 4, "storage", name);
  for (let r = 0; r < rows; r += rowsPerChunk) {
    const n = Math.min(rowsPerChunk, rows - r);
    const bytes = await src.read(e.offset + r * srcRow, n * srcRow);
    const values = toF32(e.type as Exclude<typeof e.type, "Q4_0">, bytes, n * cols);
    transform?.(name, values);
    backend.write(buf, r * cols * 4, values);
    onBytes(n * srcRow);
  }
  return { shape, dtype: "f32", buf };
}
