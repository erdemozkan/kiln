import { describe, expect, it } from "vitest";
import { Tokenizer, type TokenizerJSON } from "../src";

/** A small vocabulary in Gemma 4's layout: ▁ for spaces, merges, byte tokens as the fallback, markers as added tokens. */
const PIECES = ["<pad>", "<eos>", "<bos>", "<unk>", "<|turn>", "<turn|>", "<|channel>", "▁", "a", "b", "c", "\n", "▁a", "ab", "▁ab", "bc", ...Array.from({ length: 256 }, (_, i) => `<0x${i.toString(16).toUpperCase().padStart(2, "0")}>`)];
const MERGES = ["a b", "▁ ab", "▁ a", "b c"];
const TYPES = PIECES.map((p, i) => (i < 4 || p === "<|turn>" || p === "<turn|>" ? 3 : p === "<|channel>" ? 4 : p.startsWith("<0x") ? 6 : 1));

const json: TokenizerJSON = {
  normalizer: { type: "Replace", pattern: { String: " " }, content: "▁" },
  pre_tokenizer: { type: "Split", pattern: { String: " " }, behavior: "MergedWithPrevious", invert: false },
  model: { type: "BPE", byte_fallback: true, vocab: Object.fromEntries(PIECES.map((p, i) => [p, i])), merges: MERGES },
  added_tokens: PIECES.flatMap((p, i) => (TYPES[i] === 3 || TYPES[i] === 4 ? [{ id: i, content: p, special: true }] : [])),
} as unknown as TokenizerJSON;

const meta = new Map<string, unknown>([
  ["tokenizer.ggml.model", "gemma4"],
  ["tokenizer.ggml.tokens", PIECES],
  ["tokenizer.ggml.token_type", TYPES],
  ["tokenizer.ggml.merges", MERGES],
  ["tokenizer.ggml.bos_token_id", 2],
  ["tokenizer.ggml.eos_token_id", 5],
  ["tokenizer.chat_template", "{{ bos_token }}{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}"],
]);

describe("Gemma 4's tokenizer from GGUF metadata", () => {
  const [hf, gguf] = [Tokenizer.fromJSON(json), Tokenizer.fromGGUF(meta)];

  it("encodes as the checkpoint's tokenizer.json does", () => {
    for (const text of ["a ab abc", " ab", "ab ab\nab", "<|turn>ab<turn|>\n", "é b", "<|channel>a", "", "a  b"]) {
      expect(gguf.encode(text), text).toEqual(hf.encode(text));
      // Decoding leaves the markers out, as it does for every model.
      expect(gguf.decode(gguf.encode(text)), text).toBe(hf.decode(hf.encode(text)));
    }
    // Merges run across the spaces: "▁ab" is one piece; markers are single tokens; unknown bytes fall back.
    expect(gguf.encode(" ab")).toEqual([PIECES.indexOf("▁ab")]);
    expect(gguf.encode("<|turn>ab<turn|>")).toEqual([4, PIECES.indexOf("ab"), 5]);
    expect(gguf.encode("é")).toEqual([PIECES.indexOf("<0xC3>"), PIECES.indexOf("<0xA9>")]);
  });

  it("carries the chat template and its tokens", () => {
    expect(gguf.chatTemplate).toMatchObject({ bosToken: "<bos>", eosToken: "<turn|>" });
    expect(gguf.tokenId("<turn|>")).toBe(5);
    expect(gguf.specialIds.has(PIECES.indexOf("<|channel>"))).toBe(true);
  });
});
