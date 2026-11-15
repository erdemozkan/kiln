/**
 * Values a q4 cache rotates together in a head of D (a multiple of 32): the largest power of two that
 * divides D, at most 512 (D itself for heads of 64, 128, 256 or 512; 32 for heads of 96).
 */
export function q4Block(D: number): number {
  let b = 32;
  while (b < 512 && D % (2 * b) === 0) b *= 2;
  return b;
}
