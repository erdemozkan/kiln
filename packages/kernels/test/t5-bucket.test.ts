import { describe, expect, it } from "vitest";
import { t5Bucket } from "../src/cpu";

// transformers' T5Attention._relative_position_bucket for relative positions -300…300 (32 buckets,
// max distance 128), run-length encoded as [bucket, count].
const BIDIRECTIONAL = [[15, 210], [14, 27], [13, 18], [12, 14], [11, 9], [10, 7], [9, 4], [8, 4], [7, 1], [6, 1], [5, 1], [4, 1], [3, 1], [2, 1], [1, 1], [0, 1], [17, 1], [18, 1], [19, 1], [20, 1], [21, 1], [22, 1], [23, 1], [24, 4], [25, 4], [26, 7], [27, 9], [28, 14], [29, 18], [30, 27], [31, 210]];
const CAUSAL = [[31, 188], [30, 14], [29, 12], [28, 10], [27, 10], [26, 8], [25, 7], [24, 6], [23, 6], [22, 5], [21, 4], [20, 4], [19, 3], [18, 3], [17, 2], [16, 3], [15, 1], [14, 1], [13, 1], [12, 1], [11, 1], [10, 1], [9, 1], [8, 1], [7, 1], [6, 1], [5, 1], [4, 1], [3, 1], [2, 1], [1, 1], [0, 301]];

const expand = (r: number[][]) => r.flatMap(([b, n]) => new Array<number>(n!).fill(b!));

describe("T5 relative position buckets", () => {
  it.each([["bidirectional", BIDIRECTIONAL, true], ["causal", CAUSAL, false]] as const)("match transformers (%s)", (_, rle, bidir) => {
    const got = Array.from({ length: 601 }, (_, i) => t5Bucket(i - 300, bidir, 32, 128));
    expect(got).toEqual(expand(rle as unknown as number[][]));
  });
});
