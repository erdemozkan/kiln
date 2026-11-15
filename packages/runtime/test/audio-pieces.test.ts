import { describe, expect, it } from "vitest";
import { GEMMA4_AUDIO_MAX_SAMPLES, gemma4AudioFeatures } from "@kiln/loader";
import { audioKey, audioPieces } from "../src/engine";

describe("audio pieces", () => {
  const ramp = (n: number) => Float32Array.from({ length: n }, (_, i) => i);

  it("keeps a clip of up to 30 s whole", () => {
    expect(audioPieces(ramp(1000)).map((p) => p.length)).toEqual([1000]);
    expect(audioPieces(ramp(GEMMA4_AUDIO_MAX_SAMPLES)).map((p) => p.length)).toEqual([GEMMA4_AUDIO_MAX_SAMPLES]);
    // Too short for a token, but all there is: the engine reports it.
    expect(audioPieces(ramp(50)).map((p) => p.length)).toEqual([50]);
    expect(audioPieces(new Float32Array(0)).length).toBe(1);
  });

  it("cuts longer audio into consecutive pieces of 30 s", () => {
    const x = ramp(2 * GEMMA4_AUDIO_MAX_SAMPLES + 5000);
    const p = audioPieces(x);
    expect(p.map((q) => q.length)).toEqual([GEMMA4_AUDIO_MAX_SAMPLES, GEMMA4_AUDIO_MAX_SAMPLES, 5000]);
    // In order, without gaps or copies.
    expect(p[1]![0]).toBe(GEMMA4_AUDIO_MAX_SAMPLES);
    expect(p[2]![4999]).toBe(x.length - 1);
    expect(p[2]!.buffer).toBe(x.buffer);
  });

  it("drops a last piece too short to make a token", () => {
    expect(audioPieces(ramp(GEMMA4_AUDIO_MAX_SAMPLES + 160)).length).toBe(1);
    const kept = audioPieces(ramp(GEMMA4_AUDIO_MAX_SAMPLES + 161));
    expect(kept.length).toBe(2);
    expect(gemma4AudioFeatures(kept[1]!).frames).toBe(1);
    expect(gemma4AudioFeatures(ramp(160)).frames).toBe(0);
  });

  it("keys a piece by its samples", () => {
    const x = ramp(4000);
    expect(audioKey(x.subarray(0, 2000))).toBe(audioKey(x.slice(0, 2000)));
    expect(audioKey(x.subarray(0, 2000))).not.toBe(audioKey(x.subarray(2000)));
  });
});
