import { describe, expect, it } from "vitest";
import { centerCrop, detrResizeSize, preprocessImage, resizeImage } from "../src/image";

// A 5 × 4 RGB image and PIL 8's Image.resize of it (checked byte for byte; the 97 × 61 case and
// whole DeiT / CLIP pipelines were also checked against PIL + numpy, all exact).
const src = { width: 5, height: 4, data: Uint8Array.from({ length: 60 }, (_, i) => (i * 13) % 256) };

describe("image preprocessing", () => {
  it("resizes as PIL does (bicubic, bilinear up, lanczos)", () => {
    expect([...resizeImage(src, 3, 2, 3).data]).toEqual([120, 134, 95, 89, 95, 78, 106, 105, 122, 133, 150, 149, 177, 160, 166, 160, 121, 135]);
    expect([...resizeImage(src, 2, 3, 1).data]).toEqual([61, 75, 52, 99, 124, 133, 186, 194, 148, 108, 61, 69, 122, 131, 156, 203, 180, 194]);
    const up = resizeImage(src, 7, 6, 2).data;
    expect([...up.slice(0, 12)]).toEqual([0, 13, 26, 22, 35, 48, 50, 63, 76, 78, 91, 104]);
    expect([...up.slice(-6)]).toEqual([207, 220, 233, 229, 242, 255]);
  });

  it("crops the middle and normalizes in float32", () => {
    const c = centerCrop(src, 3, 2);
    expect([...c.data.slice(0, 3)]).toEqual([...src.data.slice(18, 21)]);
    const p = preprocessImage(src, { size: { height: 4, width: 5 }, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5] });
    expect(p.data[0]).toBe(-1);
    expect(p.data[1]).toBeCloseTo((39 / 255 - 0.5) / 0.5, 6);
  });
});

describe("DETR-family resize sizes", () => {
  // [width, height, shortest_edge, longest_edge] → transformers' YOLOS and DETR output [width, height].
  const cases: [number, number, number, number, [number, number], [number, number]][] = [
    [612, 408, 512, 1333, [768, 512], [768, 512]],
    [640, 480, 800, 1333, [1056, 800], [1066, 800]],
    [2000, 500, 800, 1333, [1328, 320], [1333, 333]],
    [333, 1000, 800, 1333, [432, 1328], [444, 1333]],
    [1203, 800, 800, 1333, [1200, 800], [1203, 800]],
    [2000, 750, 800, 1333, [1328, 496], [1333, 500]],
  ];
  it("matches transformers' get_size_with_aspect_ratio", () => {
    for (const [w, h, s, m, yolos, detr] of cases) {
      expect(detrResizeSize(w, h, s, m, true)).toEqual(yolos);
      expect(detrResizeSize(w, h, s, m, false)).toEqual(detr);
    }
  });
  it("rounds a capped short side half to even, as Python's round does", () => {
    expect(detrResizeSize(2, 1, 3, 5, false)).toEqual([5, 2]);
  });
});
