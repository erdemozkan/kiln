import { describe, expect, it } from "vitest";
import { donutPixels, type DonutProcessing } from "../src/donut";

describe("Donut image processing", () => {
  // DonutImageProcessor's steps (transformers v4) run with PIL 8 and numpy at size 40 × 30: resize the
  // shortest edge (bilinear), thumbnail (bicubic, reducing_gap 2), pad around the center, normalize.
  const p: DonutProcessing = { size: [40, 30], resample: 2, alignLongAxis: false, resize: true, thumbnail: true, pad: true, rescale: 0.00392156862745098, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] };
  const cases: [number, number, number, number[]][] = [
    [50, 20, -2518.3685945272446, [-1, 0.13725495338439941, -1, -1, -0.05882352590560913, -1, -1, -0.03529411554336548, -1]],
    // 200 × 10 and 10 × 100 shrink enough for PIL to reduce first.
    [200, 10, -3509.8431346416473, [-1, -1, -1, -1, -0.027450978755950928, -1, -1, -1, -1]],
    [10, 100, -3120.2352799773216, [-1, -1, -1, -1, -1, -1, -1, -1, -1]],
    [25, 25, -901.3097369074821, [-1, -0.5607843399047852, -0.5686274766921997, -1, 0.3960784673690796, 0.41960787773132324, -0.6470588445663452, -0.6549019813537598, -1]],
  ];
  it.each(cases)("a %i × %i image", (w, h, sum, probes) => {
    const out = donutPixels({ width: w, height: h, data: Uint8Array.from({ length: w * h * 3 }, (_, i) => (i * 37) % 256) }, p);
    expect(out.length).toBe(3 * 40 * 30);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(sum, 6);
    // Nine evenly spaced values (numpy's linspace over the flat array, truncated).
    const at = Array.from({ length: 9 }, (_, i) => Math.trunc((i * (out.length - 1)) / 8));
    expect(at.map((i) => out[i])).toEqual(probes);
  });
});
