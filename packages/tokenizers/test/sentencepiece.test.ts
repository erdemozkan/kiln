import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { Tokenizer, type TokenizerJSON } from "../src";

const S = "▁";

/** SentencePiece-style BPE: "▁" for spaces, a few merges, byte tokens for everything else. */
function tiny(pre: Partial<TokenizerJSON> = {}): Tokenizer {
  const vocab: Record<string, number> = { "<unk>": 0, "<s>": 1, "</s>": 2 };
  for (let b = 0; b < 256; b++) vocab[`<0x${b.toString(16).toUpperCase().padStart(2, "0")}>`] = 3 + b;
  let id = 259;
  for (const t of [S, "h", "e", "l", "o", "w", "r", "d", "é", `${S}h`, "el", "lo", `${S}hel`, `${S}hello`, `${S}w`, "or", `${S}wor`, `${S}world`]) vocab[t] = id++;
  const merges = [`${S} h`, "e l", `${S}h el`, "l o", `${S}hel lo`, `${S} w`, "o r", `${S}w or`, `${S}wor l`, `${S}worl d`];
  return Tokenizer.fromJSON({
    model: { type: "BPE", vocab, merges, byte_fallback: true },
    added_tokens: [
      { id: 0, content: "<unk>", special: true },
      { id: 1, content: "<s>", special: true },
      { id: 2, content: "</s>", special: true },
    ],
    normalizer: { type: "Sequence", normalizers: [{ type: "Prepend", prepend: S }, { type: "Replace", content: S }] },
    pre_tokenizer: null,
    ...pre,
  });
}

describe("SentencePiece-style BPE", () => {
  it("prepends ▁, merges by rank and falls back to bytes", () => {
    const tok = tiny();
    const pieces = (t: string) => tok.encode(t).map((i) => tok.decode([i], false) || `#${i}`);
    expect(tok.encode("hello")).toEqual([tok.tokenId(`${S}hello`)]);
    // Merges run through pieces that are not tokens themselves ("▁worl") to reach ones that are.
    expect(tok.encode("hello world")).toEqual([tok.tokenId(`${S}hello`), tok.tokenId(`${S}world`)]);
    // "x" and "😀" are not in the vocabulary: one byte token per UTF-8 byte.
    const ids = tok.encode("hex😀");
    expect(ids.slice(-5)).toEqual([0x78, 0xf0, 0x9f, 0x98, 0x80].map((b) => b + 3));
    expect(tok.decode(ids)).toBe("hex😀");
    expect(pieces("hé")).toEqual(["h", "é"]);
  });

  it("decodes ▁ as spaces and drops the prepended one", () => {
    const tok = tiny();
    for (const t of ["hello world", "  hello", "hello  ", "héllo\n\twörld 😀"]) expect(tok.decode(tok.encode(t))).toBe(t);
    // Streaming: the leading space goes only from the first piece.
    const d = tok.decoder();
    expect(tok.encode("hello hello").map((i) => d.push(i)).join("") + d.flush()).toBe("hello hello");
  });

  it("prepends ▁ only at the start with Metaspace prepend_scheme first", () => {
    const tok = tiny({ normalizer: null, pre_tokenizer: { type: "Metaspace", replacement: S, prepend_scheme: "first", split: false } as never });
    const always = tiny();
    // After a special token the text gets no ▁ with "first", but does with the Prepend normalizer.
    expect(tok.encode("<s>hello")[1]).not.toBe(tok.tokenId(`${S}hello`));
    expect(always.encode("<s>hello")[1]).toBe(always.tokenId(`${S}hello`));
    expect(tok.encode("hello")).toEqual([tok.tokenId(`${S}hello`)]);
  });

  it("merges by score for GGUF llama vocabularies", () => {
    const tokens = ["<unk>", "<s>", "</s>", "<0x0A>", S, "a", "b", `${S}a`, "ab", `${S}ab`];
    const scores = [0, 0, 0, 0, -1, -2, -2, -3, -1, -4];
    const tok = Tokenizer.fromGGUF(
      new Map<string, unknown>([
        ["tokenizer.ggml.model", "llama"],
        ["tokenizer.ggml.tokens", tokens],
        ["tokenizer.ggml.scores", scores],
        ["tokenizer.ggml.token_type", [2, 3, 3, 6, 1, 1, 1, 1, 1, 1]],
      ]),
    );
    // "ab" (score -1) merges before "▁a" (-3), so "▁ab" forms via "▁" + "ab".
    expect(tok.encode("ab")).toEqual([9]);
    expect(tok.encode("ab\n")).toEqual([9, 3]);
    expect(tok.decode([1, 9, 3, 2])).toBe("ab\n");
  });
});

/**
 * Real tokenizers against SentencePiece's own ids. KILN_SPM_REFS names a JSON file of
 * { texts, models: { name: { tokenizer, ids } } } with tokenizer.json paths relative to it,
 * as written by encoding each text with the model's tokenizer.model.
 */
const refs = process.env.KILN_SPM_REFS;
describe.skipIf(!refs)("SentencePiece reference ids", () => {
  const data = refs ? (JSON.parse(readFileSync(refs, "utf8")) as { texts: string[]; models: Record<string, { tokenizer: string; ids: number[][] }> }) : { texts: [], models: {} };
  for (const [name, m] of Object.entries(data.models)) {
    it(`${name} encodes like sentencepiece`, () => {
      const tok = Tokenizer.fromJSON(JSON.parse(readFileSync(join(dirname(refs!), m.tokenizer), "utf8")));
      data.texts.forEach((t, i) => {
        expect(tok.encode(t, { allowSpecial: false }), JSON.stringify(t)).toEqual(m.ids[i]);
        if (!t.includes(S)) expect(tok.decode(m.ids[i]!), JSON.stringify(t)).toBe(t);
      });
    });
  }
});
