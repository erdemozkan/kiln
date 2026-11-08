import { describe, expect, it } from "vitest";
import { sinusoidalTable } from "@kiln/models";
import { bannedNgramTokens } from "../src/seq2seq";

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
