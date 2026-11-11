import { describe, expect, it } from "vitest";
import { Processor } from "../src/auto";
import { RawImage } from "../src/compat";
import { Tensor } from "../src/tensor";

type Callable = Processor & ((input: unknown) => Promise<Record<string, unknown>>);

describe("Processor (transformers.js AutoProcessor)", () => {
  it("makes pixel_values [batch, 3, h, w] with the image processor's steps", async () => {
    const p = new Processor({ do_resize: true, size: { height: 2, width: 2 }, resample: 0, do_rescale: true, rescale_factor: 1 / 255, do_normalize: true, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5] }) as Callable;
    const img = new RawImage(new Uint8ClampedArray(4 * 4 * 3).fill(255), 4, 4, 3);
    const out = await p([img, img.grayscale()]);
    const px = out.pixel_values as Tensor;
    expect(px.dims).toEqual([2, 3, 2, 2]);
    expect(Array.from(px.data as Float32Array).every((v) => Math.abs(v - 1) < 1e-6)).toBe(true);
    expect(out.original_sizes).toEqual([[4, 4], [4, 4]]);
    expect(out.reshaped_input_sizes).toEqual([[2, 2], [2, 2]]);
  });

  it("normalizes wav2vec2 audio and returns its mask", async () => {
    const p = new Processor({ feature_size: 1, sampling_rate: 16000, do_normalize: true, return_attention_mask: false }) as Callable;
    const out = await p(Float32Array.from([1, 2, 3, 4]));
    const v = Array.from((out.input_values as Tensor).data as Float32Array);
    expect((out.input_values as Tensor).dims).toEqual([1, 4]);
    expect(v.reduce((a, b) => a + b, 0)).toBeCloseTo(0, 5);
    expect(v[3]! - v[0]!).toBeCloseTo(3 / Math.sqrt(1.25 + 1e-7), 4);
    expect((out.attention_mask as Tensor).type).toBe("int64");
  });

  it("makes Whisper's input_features mel-major [1, mels, 3000]", async () => {
    const p = new Processor({ feature_extractor_type: "WhisperFeatureExtractor", feature_size: 80, sampling_rate: 16000, n_fft: 400, hop_length: 160, chunk_length: 30 }) as Callable;
    const tone = Float32Array.from({ length: 16000 }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / 16000));
    const f = (await p(tone)).input_features as Tensor;
    expect(f.dims).toEqual([1, 80, 3000]);
    // The tone's energy sits in a low mel band for the first second (100 frames), not after it.
    const d = f.data as Float32Array;
    const band = Array.from({ length: 80 }, (_, m) => d[m * 3000 + 50]!);
    const peak = band.indexOf(Math.max(...band));
    expect(peak).toBeGreaterThan(5);
    expect(peak).toBeLessThan(30);
    expect(d[peak * 3000 + 50]!).toBeGreaterThan(d[peak * 3000 + 2000]!);
  });
});
