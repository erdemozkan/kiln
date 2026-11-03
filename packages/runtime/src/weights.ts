import { q4ScaleBytes, type Backend, type Tensor } from "@kiln/core";
import { quantizeQ4, repackQ4_0, rowBytes, toF32, type ByteSource, type TensorEntry } from "@kiln/loader";
import { isQuantizable } from "@kiln/models";

export type Quant = "f32" | "q4";

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
}

/** Bytes read per step. Bounds peak CPU memory regardless of model size. */
const CHUNK = 8 * 2 ** 20;

/**
 * Streams tensors from their sources straight into GPU buffers, converting row-chunks on the way:
 * F16/BF16/Q8_0 → f32 or q4, GGUF Q4_0 → repacked q4. The full model never sits in CPU memory.
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
      if (name) out.set(name, await loadTensor(backend, file.source, entry, name, quant, onBytes));
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
): Promise<Tensor> {
  const shape = e.shape.length === 1 ? e.shape : [e.shape.slice(0, -1).reduce((a, b) => a * b, 1), e.shape[e.shape.length - 1]!];
  const cols = shape[shape.length - 1]!;
  const rows = shape.length === 2 ? shape[0]! : 1;
  const toQ4 = isQuantizable(name, shape) && (quant === "q4" || e.type === "Q4_0");
  if (e.type === "Q4_0" && !toQ4) throw new Error(`${name}: Q4_0 tensor with shape [${shape}] cannot stay quantized`);

  const srcRow = rowBytes(e.type, cols);
  const rowsPerChunk = Math.max(1, Math.floor(CHUNK / srcRow));

  if (toQ4) {
    const buf = backend.alloc(rows * cols / 2, "storage", name);
    const scales = backend.alloc(q4ScaleBytes(shape), "storage", `${name}/scales`);
    for (let r = 0; r < rows; r += rowsPerChunk) {
      const n = Math.min(rowsPerChunk, rows - r);
      const bytes = await src.read(e.offset + r * srcRow, n * srcRow);
      const planes = e.type === "Q4_0" ? repackQ4_0(bytes, n * cols) : quantizeQ4(toF32(e.type, bytes, n * cols));
      backend.write(buf, (r * cols) / 2, planes.nibbles);
      backend.write(scales, (r * cols) / 8, planes.scales);
      onBytes(n * srcRow);
    }
    return { shape, dtype: "q4", buf, scales };
  }

  const buf = backend.alloc(rows * cols * 4, "storage", name);
  for (let r = 0; r < rows; r += rowsPerChunk) {
    const n = Math.min(rowsPerChunk, rows - r);
    const bytes = await src.read(e.offset + r * srcRow, n * srcRow);
    backend.write(buf, r * cols * 4, toF32(e.type as Exclude<typeof e.type, "Q4_0">, bytes, n * cols));
    onBytes(n * srcRow);
  }
  return { shape, dtype: "f32", buf };
}
