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
