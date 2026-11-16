import { describe, expect, it } from "vitest";
import { samPositional, samSparsePrompt, type SamPromptWeights } from "../src/sam";

const F = 4;
const gaussian = Float32Array.from({ length: 2 * F }, (_, i) => (i - 3) / 5);
const row = (v: number) => new Float32Array(2 * F).fill(v);
const w: SamPromptWeights = { gaussian, pointEmbed: [row(10), row(20), row(30), row(40)], notAPoint: row(-1) };

describe("SAM prompt encoder", () => {
  it("embeds a point as transformers does: sines then cosines of 2π (2·xy − 1)·G", () => {
    const [x, y] = [0.25, 0.75];
    const e = samPositional(gaussian, x, y);
    for (let i = 0; i < F; i++) {
      const v = 2 * Math.PI * ((2 * x - 1) * gaussian[i]! + (2 * y - 1) * gaussian[F + i]!);
      expect(e[i]).toBeCloseTo(Math.sin(v), 5);
      expect(e[F + i]).toBeCloseTo(Math.cos(v), 5);
    }
  });

  it("adds each label's embedding, pads points without boxes, and gives boxes two corners", () => {
    const pts = samSparsePrompt(w, { points: [{ x: 10, y: 20, label: 1 }, { x: 5, y: 5, label: 0 }] }, 64);
    expect(pts.length).toBe(3);
    const pe = samPositional(gaussian, 10.5 / 64, 20.5 / 64);
    expect(Array.from(pts[0]!, (v, i) => v - pe[i]!).every((d) => Math.abs(d - 20) < 1e-4)).toBe(true);
    expect(Array.from(pts[2]!)).toEqual(Array.from(row(-1)));
    const withBox = samSparsePrompt(w, { points: [{ x: 10, y: 20, label: 1 }], boxes: [[1, 2, 30, 40]] }, 64);
    expect(withBox.length).toBe(3);
    const corner = samPositional(gaussian, 30.5 / 64, 40.5 / 64);
    expect(Array.from(withBox[2]!, (v, i) => v - corner[i]!).every((d) => Math.abs(d - 40) < 1e-4)).toBe(true);
    expect(() => samSparsePrompt(w, {}, 64)).toThrow(/at least one point or box/);
  });
});
