import { describe, expect, it } from "vitest";
import { chunkText, splitSentences } from "../src/retrieval";

const words = (s: string) => s.split(/\s+/).filter(Boolean).length;

describe("retrieval chunking", () => {
  it("splits sentences and paragraphs, keeping all the text", () => {
    const t = "First one. Second one! Third?\n\nNew paragraph without a stop\nNext line.";
    const parts = splitSentences(t);
    expect(parts.map((p) => p.trim())).toEqual(["First one.", "Second one!", "Third?", "New paragraph without a stop", "Next line."]);
    expect(parts.join("")).toBe(t);
  });

  it("packs sentences up to the token budget with sentence overlap", () => {
    const t = Array.from({ length: 12 }, (_, i) => `Sentence number ${i} has six words.`).join(" ");
    const chunks = chunkText(t, words, 20, 6);
    for (const c of chunks) expect(words(c)).toBeLessThanOrEqual(20);
    // Each chunk after the first starts with the last sentence of the one before.
    for (let i = 1; i < chunks.length; i++) {
      const prevLast = splitSentences(chunks[i - 1]!).at(-1)!.trim();
      expect(chunks[i]!.startsWith(prevLast)).toBe(true);
    }
    // Every sentence is in some chunk.
    for (let i = 0; i < 12; i++) expect(chunks.some((c) => c.includes(`number ${i} `))).toBe(true);
  });

  it("splits a sentence longer than the budget at words", () => {
    const long = Array.from({ length: 50 }, (_, i) => `w${i}`).join(" ") + ".";
    const chunks = chunkText(long, words, 16, 0);
    expect(chunks.length).toBe(4);
    for (const c of chunks) expect(words(c)).toBeLessThanOrEqual(16);
    expect(chunks.join(" ").split(/\s+/).length).toBe(50);
  });
});
