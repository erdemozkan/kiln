import { describe, expect, it } from "vitest";
import { FFT, melFilters, readAudio, readWav, resample, whisperFeatures } from "../src";

function dft(x: Float64Array): [Float64Array, Float64Array] {
  const n = x.length;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let k = 0; k < n; k++)
    for (let j = 0; j < n; j++) {
      re[k]! += x[j]! * Math.cos((-2 * Math.PI * j * k) / n);
      im[k]! += x[j]! * Math.sin((-2 * Math.PI * j * k) / n);
    }
  return [re, im];
}

function wav(samples: number[][], rate: number, bits: 16 | 32, float = false): Uint8Array {
  const ch = samples.length;
  const n = samples[0]!.length;
  const w = bits / 8;
  const buf = new ArrayBuffer(44 + n * ch * w);
  const dv = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((c, i) => dv.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  dv.setUint32(4, 36 + n * ch * w, true);
  str(8, "WAVE");
  str(12, "fmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, float ? 3 : 1, true);
  dv.setUint16(22, ch, true);
  dv.setUint32(24, rate, true);
  dv.setUint32(28, rate * ch * w, true);
  dv.setUint16(32, ch * w, true);
  dv.setUint16(34, bits, true);
  str(36, "data");
  dv.setUint32(40, n * ch * w, true);
  for (let i = 0; i < n; i++)
    for (let c = 0; c < ch; c++) {
      const o = 44 + (i * ch + c) * w;
      if (float) dv.setFloat32(o, samples[c]![i]!, true);
      else dv.setInt16(o, Math.round(samples[c]![i]! * 32767), true);
    }
  return new Uint8Array(buf);
}

describe("FFT", () => {
  it.each([400, 64, 60, 75, 2, 1])("matches a direct DFT (n = %i)", (n) => {
    const x = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) + Math.cos(i * i * 0.01));
    const [wr, wi] = dft(x);
    const re = x.slice();
    const im = new Float64Array(n);
    new FFT(n).transform(re, im);
    for (let k = 0; k < n; k++) {
      expect(re[k]).toBeCloseTo(wr[k]!, 9);
      expect(im[k]).toBeCloseTo(wi[k]!, 9);
    }
  });

  it("rejects lengths with other prime factors", () => {
    expect(() => new FFT(14)).toThrow(/prime factors/);
  });
});

describe("Whisper features", () => {
  it("builds slaney mel triangles: non-negative, each band one peak", () => {
    const f = melFilters(201, 80, 16000);
    expect(f.length).toBe(201 * 80);
    expect(Math.min(...f)).toBeGreaterThanOrEqual(0);
    for (let m = 0; m < 80; m += 13) expect(Array.from({ length: 201 }, (_, b) => f[b * 80 + m]!).some((v) => v > 0)).toBe(true);
  });

  it("gives 3000 frames × 80 mels, clamped to 8 below the maximum", () => {
    const tone = Float32Array.from({ length: 16000 }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / 16000) * 0.5);
    const f = whisperFeatures(tone, { mels: 80 });
    expect(f.length).toBe(3000 * 80);
    const max = f.reduce((a, b) => Math.max(a, b), -Infinity);
    const min = f.reduce((a, b) => Math.min(a, b), Infinity);
    expect(max - min).toBeCloseTo(2, 5); // (8 / 4) after the scaling
    // Silence after the tone sits at the floor.
    expect(f[2999 * 80 + 5]).toBeCloseTo(min, 6);
  });
});

describe("audio input", () => {
  it("reads 16-bit stereo and float WAV, averaging channels", () => {
    const s16 = readWav(wav([[0.5, -0.5, 0.25], [0.5, 0.5, 0.25]], 22050, 16));
    expect(s16.sampleRate).toBe(22050);
    expect(Array.from(s16.samples).map((v) => +v.toFixed(3))).toEqual([0.5, 0, 0.25]);
    const f32 = readWav(wav([[0.1, -0.7]], 16000, 32, true));
    expect(Array.from(f32.samples)).toEqual([Math.fround(0.1), Math.fround(-0.7)]);
  });

  it("resamples linearly", () => {
    expect(Array.from(resample(Float32Array.from([0, 1, 2, 3]), 2, 1))).toEqual([0, 2]);
    expect(Array.from(resample(Float32Array.from([0, 2]), 1, 2))).toEqual([0, 1, 2, 2]);
  });

  it("decodes WAV bytes to 16 kHz samples without Web Audio", async () => {
    const out = await readAudio(wav([[0, 0.5, 0, -0.5]], 32000, 16), 16000);
    expect(out.length).toBe(2);
  });
});
