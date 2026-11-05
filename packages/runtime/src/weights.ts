import { byteSize, q4ScaleBytes, type Backend, type Tensor } from "@kiln/core";
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
      if (name) out.set(name, await loadTensor(backend, file.source, entry, name, quant, onBytes, file.transform));
      else onBytes(entry.byteLength);
    }
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
): Promise<Tensor> {
  const shape = e.shape.length === 1 ? e.shape : [e.shape.slice(0, -1).reduce((a, b) => a * b, 1), e.shape[e.shape.length - 1]!];
  const cols = shape[shape.length - 1]!;
  const rows = shape.length === 2 ? shape[0]! : 1;
  const quantizable = isQuantizable(name, shape);
  if (e.type === "Q4_0" && !quantizable) throw new Error(`${name}: Q4_0 tensor with shape [${shape}] cannot stay quantized`);
  const dtype = !quantizable ? "f32" : quant === "q4" || e.type === "Q4_0" ? "q4" : quant === "q8" || e.type === "Q8_0" && quant !== "f32" ? "q8" : "f32";

  const srcRow = rowBytes(e.type, cols);
  const rowsPerChunk = Math.max(1, Math.floor(CHUNK / srcRow));

  if (dtype !== "f32") {
    const perRow = byteSize([cols], dtype);
    const buf = backend.alloc(rows * perRow, "storage", name);
    const scales = backend.alloc(q4ScaleBytes(shape), "storage", `${name}/scales`);
    for (let r = 0; r < rows; r += rowsPerChunk) {
      const n = Math.min(rowsPerChunk, rows - r);
      const bytes = await src.read(e.offset + r * srcRow, n * srcRow);
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
