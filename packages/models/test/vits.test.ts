import { describe, expect, it } from "vitest";
import { VitsTokenizer, normalStream, splineInverse, vitsConfigFromHF } from "../src";

const MMS_VOCAB: Record<string, number> = { " ": 19, "'": 1, "-": 14, "0": 23, "1": 15, "2": 28, "3": 11, "4": 27, "5": 35, "6": 36, _: 30, a: 26, b: 24, c: 12, d: 5, e: 7, f: 20, g: 37, h: 6, i: 18, j: 16, k: 0, l: 21, m: 17, n: 29, o: 22, p: 13, q: 34, r: 25, s: 8, t: 33, u: 4, v: 32, w: 9, x: 31, y: 3, z: 2, "–": 10 };

/** The forward rational-quadratic spline, written from transformers' _rational_quadratic_spline (reverse=False). */
function splineForward(x: number, uw: number[], uh: number[], ud: number[], tail: number): number {
  if (x < -tail || x > tail) return x;
  const nb = uw.length;
  const cum = (u: number[]) => {
    const m = Math.max(...u);
    const e = u.map((v) => Math.exp(v - m));
    const z = e.reduce((a, b) => a + b, 0);
    const p = e.map((v) => 1e-3 + (1 - 1e-3 * nb) * (v / z));
    const c = [0];
    for (const v of p) c.push(c[c.length - 1]! + v);
    const out = c.map((v) => 2 * tail * v - tail);
    out[0] = -tail;
    out[nb] = tail;
    return out;
  };
  const cw = cum(uw);
  const ch = cum(uh);
  const k = Math.log(Math.exp(1 - 1e-3) - 1);
  const d = [k, ...ud, k].map((v) => 1e-3 + Math.log1p(Math.exp(v)));
  const locs = cw.slice();
  locs[nb] = locs[nb]! + 1e-6;
  let i = -1;
  for (const l of locs) if (x >= l) i++;
  const w = cw[i + 1]! - cw[i]!;
  const h = ch[i + 1]! - ch[i]!;
  const delta = h / w;
  const th = (x - cw[i]!) / w;
  const tt = th * (1 - th);
  return ch[i]! + (h * (delta * th * th + d[i]! * tt)) / (delta + (d[i]! + d[i + 1]! - 2 * delta) * tt);
}

describe("VITS", () => {
  it("tokenizes as VitsTokenizer: unknown characters lowercased or dropped, blanks between", () => {
    const tok = new VitsTokenizer(MMS_VOCAB, { addBlank: true, normalize: true });
    // "Hello, World!" → "hello world".
    expect(tok.encode("Hello, World!")).toEqual([0, 6, 0, 7, 0, 21, 0, 21, 0, 22, 0, 19, 0, 9, 0, 22, 0, 25, 0, 21, 0, 5, 0]);
    expect(tok.encode("  ok  ")).toEqual([0, 22, 0, 0, 0]);
    expect(() => new VitsTokenizer(MMS_VOCAB, { isUroman: true }).encode("привет")).toThrow(/uroman/);
  });

  it("inverts the rational-quadratic spline inside the tail bound and is the identity outside", () => {
    const nb = 10;
    for (let t = 0; t < 6; t++) {
      const uw = Array.from({ length: nb }, (_, j) => Math.sin(0.3 * (t * nb + j)));
      const uh = Array.from({ length: nb }, (_, j) => 1.5 * Math.cos(0.7 * (t * nb + j)));
      const ud = Array.from({ length: nb - 1 }, (_, j) => Math.sin(1.1 * (t * 9 + j) + 0.5));
      for (const x of [-4.9, -2.3, -0.4, 0, 0.8, 3.7, 4.95]) expect(splineInverse(splineForward(x, uw, uh, ud, 5), uw, uh, ud, 5)).toBeCloseTo(x, 9);
      expect(splineInverse(6.5, uw, uh, ud, 5)).toBe(6.5);
    }
  });

  it("draws the same noise for the same seed", () => {
    const a = normalStream(42);
    const b = normalStream(42);
    const xs = Array.from({ length: 1000 }, a);
    expect(Array.from({ length: 1000 }, b)).toEqual(xs);
    const mean = xs.reduce((s, v) => s + v, 0) / xs.length;
    const sd = Math.sqrt(xs.reduce((s, v) => s + (v - mean) ** 2, 0) / xs.length);
    expect(Math.abs(mean)).toBeLessThan(0.1);
    expect(Math.abs(sd - 1)).toBeLessThan(0.1);
  });

  it("reads facebook/mms-tts-eng's config and refuses multi-speaker models", () => {
    expect(vitsConfigFromHF({ model_type: "vits", hidden_size: 192, upsample_rates: [8, 8, 2, 2] })).toMatchObject({ hidden: 192, flows: 4, wavenetLayers: 4, bins: 10, upsampleRates: [8, 8, 2, 2], noiseScale: 0.667, samplingRate: 16000 });
    expect(() => vitsConfigFromHF({ model_type: "vits", num_speakers: 4 })).toThrow(/multi-speaker/);
  });
});
