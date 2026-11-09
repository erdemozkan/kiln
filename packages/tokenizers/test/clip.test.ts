import { describe, expect, it } from "vitest";
import { ClipTokenizer } from "../src/clip";

// A toy CLIP vocabulary: words end in "</w>", merges apply by rank. (The real openai/clip-vit-base-patch32
// tokenizer was checked against transformers.js on punctuation, accents, CJK and emoji: all equal.)
const vocab: Record<string, number> = { "c": 0, "a": 1, "t": 2, "t</w>": 3, "ca": 4, "cat</w>": 5, "!": 6, "!</w>": 7, "!!</w>": 8, "s</w>": 9, "<|startoftext|>": 98, "<|endoftext|>": 99 };
const tok = new ClipTokenizer({
  model: { type: "BPE", vocab, merges: ["c a", "ca t</w>", "! !</w>"], end_of_word_suffix: "</w>", unk_token: "<|endoftext|>" },
  post_processor: { cls: ["<|startoftext|>", 98], sep: ["<|endoftext|>", 99] },
});

describe("CLIP tokenizer", () => {
  it("lowercases, drops whitespace, closes words with </w> and wraps in start / end", () => {
    expect(tok.encode("  CAT   cat!!")).toEqual([98, 5, 5, 8, 99]);
  });
  it("splits 's off as its own piece and maps unknown symbols to unk", () => {
    expect(tok.encode("cat's")).toEqual([98, 5, 99, 9, 99]);
  });
  it("truncates to the maximum length, keeping the end token", () => {
    const short = new ClipTokenizer({ model: { type: "BPE", vocab, merges: ["c a", "ca t</w>"] } }, 4);
    expect(short.encode("cat cat cat cat")).toEqual([98, 5, 5, 99]);
  });
});
