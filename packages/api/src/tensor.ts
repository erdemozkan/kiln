/** Element types a Tensor holds, by transformers.js's names. */
export type TensorType = "float32" | "int64" | "int32" | "uint8" | "bool";

type TensorData = Float32Array | BigInt64Array | Int32Array | Uint8Array;

/** transformers.js's Tensor, as far as Kiln returns them: a typed array with dims, row-major. */
export class Tensor {
  constructor(
    readonly type: TensorType,
    readonly data: TensorData,
    readonly dims: number[],
  ) {}

  get size(): number {
    return this.data.length;
  }

  /** Nested arrays, outermost dimension first (bigints for int64, as transformers.js returns them). */
  tolist(): unknown[] {
    const rec = (offset: number, d: number): unknown[] => {
      const n = this.dims[d]!;
      if (d === this.dims.length - 1) return Array.from(this.data.subarray(offset, offset + n) as ArrayLike<number | bigint>);
      const stride = this.dims.slice(d + 1).reduce((a, b) => a * b, 1);
      return Array.from({ length: n }, (_, i) => rec(offset + i * stride, d + 1));
    };
    return rec(0, 0);
  }
}
