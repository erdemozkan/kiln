import { describe, expect, it } from "vitest";
import { bicubicResize, vitGrid } from "../src";

describe("bicubicResize", () => {
  // torch 1.7.1: interpolate(x [1, 2, 4, 5], size=(6, 3), mode="bicubic", align_corners=False).
  const x = [-5, -2, 1, 4, -4, 2.1, 5.1, -2.9, 0.1, 3.1, -1.8, 1.2, 4.2, -3.8, -0.8, 5.3, -2.7, 0.3, 3.3, -4.7, 1, 1.5, 2, 2.5, 3, 2, 3, 0.5, 1.5, 2.5, 3, 1, 2.5, 0.5, 2, 0.5, 2.5, 1, 3, 1.5];
  const torch = [
    -4.891591, 1.338542, -1.63053, 0.056249, -1.615625, 0.926622, 3.447683, -2.151852, 1.987488, -0.556636, 3.451852, -1.493432, 0.373381, 2.915625, -2.52477, 2.93053, -0.038541, -1.999703, 1.014694, 2.130209,
    2.925025, 1.807292, 1.0625, 2.632812, 2.563368, 0.682871, 2.056241, 2.427181, 2.317131, 1.466134, 1.734376, 1.9375, 1.672742, 1.119728, 0.869792, 2.141203,
  ];
  it("matches torch's bicubic interpolation (align_corners=False)", () => {
    const y = bicubicResize(Float32Array.from(x), 2, 4, 5, 6, 3);
    torch.forEach((v, i) => expect(y[i]).toBeCloseTo(v, 5));
  });
  it("is the identity at the same size", () => {
    const y = bicubicResize(Float32Array.from(x), 2, 4, 5, 4, 5);
    x.forEach((v, i) => expect(y[i]).toBeCloseTo(v, 6));
  });
});

describe("vitGrid", () => {
  it("drops the pixels a stride-p convolution leaves over", () => {
    expect(vitGrid({ patchSize: 16 }, 770, 515)).toEqual([32, 48]);
  });
});
