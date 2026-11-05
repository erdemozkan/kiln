import type { Buf } from "./backend";

/**
 * GPU storage formats.
 * - f32: plain floats.
 * - q4: symmetric 4-bit blocks of 32 along the last axis, stored as two planes:
 *   `buf` holds 8 nibbles per u32 (nibble j = element 8i+j, value q-8), `scales` holds one f32 per block.
 * - q8: symmetric 8-bit blocks of 32 along the last axis, the same two planes: `buf` holds 4 int8 per
 *   u32 (byte j = element 4i+j), `scales` one f32 per block.
 * - u32: token ids and other integer data.
 */
export type DType = "f32" | "q4" | "q8" | "u32";
/** How a weight matrix is stored on the GPU. */
export type WeightDType = "f32" | "q4" | "q8";
/** q4 and q8: block-quantized, with a scales plane. */
export const isBlockQuant = (dtype: DType): dtype is "q4" | "q8" => dtype === "q4" || dtype === "q8";

export interface Tensor {
  readonly shape: readonly number[];
  readonly dtype: DType;
  readonly buf: Buf;
  /** Per-block scales, present only for q4 and q8. */
  readonly scales?: Buf;
}

export const Q4_BLOCK = 32;

export const numel = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1);

/** Bytes of the main plane (`buf`). */
export function byteSize(shape: readonly number[], dtype: DType): number {
  const n = numel(shape);
  return dtype === "q4" ? n / 2 : dtype === "q8" ? n : n * 4;
}

export const q4ScaleBytes = (shape: readonly number[]): number => (numel(shape) / Q4_BLOCK) * 4;

export function view(t: Tensor, shape: readonly number[]): Tensor {
  if (numel(shape) !== numel(t.shape)) {
    throw new Error(`view: cannot reshape [${t.shape}] to [${shape}]`);
  }
  return { ...t, shape };
}

export function lastDim(t: Tensor): number {
  return t.shape[t.shape.length - 1]!;
}
