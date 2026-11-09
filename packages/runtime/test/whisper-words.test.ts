import { describe, expect, it } from "vitest";
import { decodeAsr } from "../src/seq2seq";
import { collateWords, dynamicTimeWarping, medianFilter, whisperTokenTimestamps } from "../src/whisper-words";

describe("Whisper word timestamps", () => {
  it("median-filters with mirrored edges", () => {
    expect([...medianFilter(Float32Array.of(5, 1, 9, 2, 7), 3)]).toEqual([1, 5, 2, 7, 2]);
  });

  it("warps along the cheapest monotonic path", () => {
    // Row 0 is cheap in columns 0-1, row 1 in columns 2-3.
    const m = [0, 0, 9, 9, 9, 9, 0, 0];
    expect(dynamicTimeWarping(m, 2, 4)).toEqual([[0, 0, 1, 1], [0, 1, 2, 3]]);
  });

  it("starts each token where its attention peaks begin", () => {
    // One head, prompt of 1 row, then 3 tokens attending to frames 0-1, 2-3 and 4-5 of 6.
    const S = 8;
    const rows = 4;
    const w = new Float32Array(rows * S);
    const peak = [[0, 1, 2, 3, 4, 5], [0, 1], [2, 3], [4, 5]];
    peak.forEach((fs, r) => fs.forEach((f) => (w[r * S + f] = 1)));
    const t = whisperTokenTimestamps(w, { rows, heads: 1, S, frames: 6, nInit: 1, medianWidth: 1 });
    expect(t.map((x) => Math.round(x * 100) / 100)).toEqual([0, 0, 0.04, 0.08, 0.08]);
  });

  it("groups tokens into words and attaches punctuation", () => {
    const pieces: Record<number, string> = { 1: " Hello", 2: ",", 3: " wor", 4: "ld", 5: "." };
    const decode = (ids: number[]) => ids.map((i) => pieces[i]).join("");
    const times: [number, number][] = [[0, 0.5], [0.5, 0.52], [0.6, 0.8], [0.8, 1], [1, 1.02]];
    expect(collateWords([1, 2, 3, 4, 5], times, { decode, eos: 50 })).toEqual([
      { text: " Hello,", timestamp: [0, 0.52] },
      { text: " world.", timestamp: [0.6, 1.02] },
    ]);
  });

  it("returns the words of every segment, capped at the segment's end", () => {
    const TS = 100;
    const pieces: Record<number, string> = { 1: " a", 2: " b" };
    const r = decodeAsr([{ tokens: [TS, 1, 2, TS + 50, 90], stride: [2, 0, 0], tokenTimestamps: [0, 0.1, 0.5, 1.4, 1.4] }], {
      timestampBegin: TS,
      firstSpecial: 90,
      timePrecision: 0.02,
      timestamps: true,
      decode: (ids) => ids.map((i) => pieces[i] ?? "").join(""),
      words: { eos: 90 },
    });
    expect(r).toEqual({ text: " a b", chunks: [{ text: " a", timestamp: [0.1, 0.5] }, { text: " b", timestamp: [0.5, 1] }] });
  });
});
