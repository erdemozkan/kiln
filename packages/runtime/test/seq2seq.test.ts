import { describe, expect, it } from "vitest";
import { sinusoidalTable } from "@kiln/models";
import { bannedNgramTokens, decodeAsr, findLongestCommonSequence, whisperTimestampChoice } from "../src/seq2seq";

describe("no_repeat_ngram_size", () => {
  it("bans tokens that would repeat an n-gram", () => {
    // "a b c a b" with n = 3: "a b" was followed by c.
    expect([...bannedNgramTokens([1, 2, 3, 1, 2], 3)]).toEqual([3]);
    expect([...bannedNgramTokens([1, 2, 3, 1, 2], 2)].sort()).toEqual([3]);
    expect([...bannedNgramTokens([1, 2, 1], 2)]).toEqual([2]);
    expect(bannedNgramTokens([1, 2], 3).size).toBe(0);
    expect(bannedNgramTokens([1, 2, 3], 0).size).toBe(0);
  });
});

describe("sinusoidal positions", () => {
  it("lays out sin then cos halves, as Marian and M2M100 do", () => {
    const d = 8;
    const m = sinusoidalTable("sinusoidal-marian", 3, d);
    expect(m[2 * d + 1]).toBeCloseTo(Math.sin(2 / Math.pow(10000, 2 / d)), 6);
    expect(m[2 * d + 4 + 1]).toBeCloseTo(Math.cos(2 / Math.pow(10000, 2 / d)), 6);
    const n = sinusoidalTable("sinusoidal-m2m100", 4, d, 1);
    expect(n[3 * d + 2]).toBeCloseTo(Math.sin(3 * Math.exp((-2 * Math.log(10000)) / 3)), 6);
    expect(Array.from(n.subarray(d, 2 * d))).toEqual(new Array(d).fill(0));
  });
});

describe("Whisper timestamps", () => {
  const TS = 100; // first timestamp token in these toy vocabularies
  const decode = (ids: number[]) => ids.map((i) => ` w${i}`).join("");
  const opts = { timestampBegin: TS, firstSpecial: 90, timePrecision: 0.02, timestamps: true, decode };

  it("picks a timestamp when the timestamps' total probability beats the best text token", () => {
    // Three timestamps at logit 2 sum to 2 + ln 3 ≈ 3.10.
    expect(whisperTimestampChoice(7, 3.0, Float32Array.of(2, 2, 2), TS)).toBe(TS);
    expect(whisperTimestampChoice(7, 3.2, Float32Array.of(2, 2, 2), TS)).toBe(7);
    expect(whisperTimestampChoice(7, 0, Float32Array.of(-1, 5, 1), TS)).toBe(TS + 1);
  });

  it("splits one window into segments on timestamp pairs, ignoring special tokens", () => {
    const r = decodeAsr([{ tokens: [91, TS, 1, 2, TS + 50, TS + 50, 3, TS + 100, 90], stride: [2, 0, 0] }], opts);
    expect(r).toEqual({
      text: " w1 w2 w3",
      chunks: [
        { timestamp: [0, 1], text: " w1 w2" },
        { timestamp: [1, 2], text: " w3" },
      ],
    });
    expect(decodeAsr([{ tokens: [91, 1, 2, 90], stride: [2, 0, 0] }], { ...opts, timestamps: false })).toEqual({ text: " w1 w2" });
  });

  it("merges overlapping windows where their tokens agree", () => {
    expect(findLongestCommonSequence([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]])).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(findLongestCommonSequence([[1, 2], [8, 9]])).toEqual([1, 2, 8, 9]);
    // Two windows without timestamps: [1 2 3 4] then [3 4 5].
    expect(decodeAsr([{ tokens: [1, 2, 3, 4], stride: [30, 0, 5] }, { tokens: [3, 4, 5], stride: [13, 5, 0] }], { ...opts, timestamps: false }).text).toBe(" w1 w2 w3 w4 w5");
  });
});
