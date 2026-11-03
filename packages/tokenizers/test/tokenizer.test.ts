import { describe, expect, it } from "vitest";
import { QWEN2_PATTERN, Tokenizer, byteToUnicode, chatml, detectChatFormat, toJsRegex, type TokenizerJSON } from "../src";

/** Byte-level BPE with every byte, a few merges and two special tokens. */
function tiny(preTokenizer: TokenizerJSON["pre_tokenizer"] = { type: "ByteLevel", add_prefix_space: false, use_regex: true }): Tokenizer {
  const vocab: Record<string, number> = {};
  for (let b = 0; b < 256; b++) vocab[byteToUnicode(b)] = b;
  const merges = ["h e", "l l", "he ll", "hell o", "Ġ w", "o r", "Ġw or", "l d", "Ġwor ld"];
  let id = 256;
  for (const m of merges) vocab[m.replace(" ", "")] = id++;
  return Tokenizer.fromJSON({
    model: { type: "BPE", vocab, merges },
    added_tokens: [
      { id: 500, content: "<|im_start|>", special: true },
      { id: 501, content: "<|im_end|>", special: true },
    ],
    pre_tokenizer: preTokenizer,
  });
}

describe("regex conversion", () => {
  it("expands scoped case-insensitive groups", () => {
    const re = toJsRegex(String.raw`(?i:'s|'ll)|\p{L}+`);
    expect("It'S fine'LL".match(re)).toEqual(["It", "'S", "fine", "'LL"]);
  });

  it("compiles the Qwen2 split pattern", () => {
    expect("Hello  world123!".match(toJsRegex(QWEN2_PATTERN))).toEqual(["Hello", " ", " world", "1", "2", "3", "!"]);
  });
});

describe("byte-level BPE", () => {
  const tok = tiny();

  it("applies merges by rank", () => {
    expect(tok.encode("hello world")).toEqual([259, 264]);
  });

  it("round-trips arbitrary unicode through byte fallback", () => {
    const text = "héllo 🌍 wörld\n\ttabs";
    expect(tok.decode(tok.encode(text))).toBe(text);
  });

  it("streams multi-byte characters without emitting broken text", () => {
    const d = tok.decoder();
    const pieces = tok.encode("🌍").map((id) => d.push(id));
    expect(pieces.slice(0, -1).every((p) => p === "")).toBe(true);
    expect(pieces.join("") + d.flush()).toBe("🌍");
  });

  it("parses special tokens as single ids and skips them when decoding", () => {
    const ids = tok.encode("<|im_start|>hello<|im_end|>");
    expect(ids).toEqual([500, 259, 501]);
    expect(tok.decode(ids)).toBe("hello");
    expect(tok.encode("<|im_start|>", { allowSpecial: false })).not.toContain(500);
  });

  it("supports Split + ByteLevel sequences (Qwen style)", () => {
    const qwen = tiny({ type: "Sequence", pretokenizers: [{ type: "Split", pattern: { Regex: QWEN2_PATTERN } }, { type: "ByteLevel", use_regex: false }] });
    expect(qwen.decode(qwen.encode("it's 42, ok?"))).toBe("it's 42, ok?");
  });
});

describe("chat formats", () => {
  it("detects chatml and renders a generation prompt", () => {
    const tok = tiny();
    expect(detectChatFormat(tok).name).toBe("chatml");
    expect(chatml.render([{ role: "user", content: "hi" }], { enableThinking: false })).toBe(
      "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    );
  });
});
