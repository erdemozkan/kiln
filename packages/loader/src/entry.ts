/** Source encodings the loader can convert. Anything else fails with a clear message. */
export type SrcType = "F32" | "F16" | "BF16" | "Q4_0" | "Q4_1" | "Q5_0" | "Q5_1" | "Q8_0" | "Q4_K" | "Q5_K" | "Q6_K";

/** Block-quantized GGUF types: elements per block and bytes per block (llama.cpp layouts). */
export const BLOCKS: Partial<Record<SrcType, readonly [elems: number, bytes: number]>> = {
  Q4_0: [32, 18],
  Q4_1: [32, 20],
  Q5_0: [32, 22],
  Q5_1: [32, 24],
  Q8_0: [32, 34],
  Q4_K: [256, 144],
  Q5_K: [256, 176],
  Q6_K: [256, 210],
};

/** One tensor inside a model file, independent of the container format. */
export interface TensorEntry {
  name: string;
  /** Row-major, outermost first (GGUF dims are reversed into this order). */
  shape: number[];
  type: SrcType;
  /** Absolute byte offset inside the file. */
  offset: number;
  byteLength: number;
  /** A 2-D tensor stored column-major (a transposed view saved as is): the bytes hold [cols, rows]. */
  transposed?: boolean;
}

export function rowBytes(type: SrcType, cols: number): number {
  switch (type) {
    case "F32":
      return cols * 4;
    case "F16":
    case "BF16":
      return cols * 2;
    default: {
      const [elems, bytes] = BLOCKS[type]!;
      if (cols % elems !== 0) throw new UnsupportedTensorError(`${type} rows must be a multiple of ${elems} (got ${cols})`);
      return (cols / elems) * bytes;
    }
  }
}

export class UnsupportedTensorError extends Error {}

/** Largest header Kiln reads from a weight file (GGUF and legacy PyTorch grow a window to find its end). */
export const MAX_HEADER_BYTES = 256 * 2 ** 20;
/** Most elements one tensor may have (64 G: far above any model, far below 2^53). */
const MAX_ELEMENTS = 2 ** 36;

/** Bytes a tensor of `shape` takes in `type`. */
export function tensorBytes(type: SrcType, shape: readonly number[]): number {
  if (!shape.length) return rowBytes(type, 1);
  const cols = shape[shape.length - 1]!;
  return shape.slice(0, -1).reduce((a, b) => a * b, 1) * rowBytes(type, cols);
}

/**
 * Checks a weight file's tensor table against the file: shapes of non-negative integers, offsets and
 * lengths that are integers inside the file, and lengths that match shape and type. A file that fails
 * is corrupt or hostile; reading it anyway would mean garbage weights or multi-GB reads.
 */
export function checkEntries(file: string, tensors: readonly TensorEntry[], fileSize: number): void {
  for (const t of tensors) {
    const bad = (why: string) => new Error(`${file}: tensor ${t.name} ${why}`);
    if (!Array.isArray(t.shape) || t.shape.some((d) => !Number.isSafeInteger(d) || d < 0)) throw bad(`has an invalid shape [${String(t.shape)}]`);
    if (t.shape.reduce((a, b) => a * b, 1) > MAX_ELEMENTS) throw bad("is implausibly large");
    if (!Number.isSafeInteger(t.offset) || !Number.isSafeInteger(t.byteLength) || t.offset < 0 || t.byteLength < 0) throw bad("has an invalid offset or length");
    if (t.offset + t.byteLength > fileSize) throw bad(`runs past the end of the file (${t.offset + t.byteLength} > ${fileSize} bytes)`);
    if (t.byteLength !== tensorBytes(t.type, t.shape)) throw bad(`has ${t.byteLength} bytes, not the ${tensorBytes(t.type, t.shape)} its shape and type need`);
  }
}
