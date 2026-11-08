import { describe, expect, it } from "vitest";
import { encoderTokenizerFromJSON, wordPieceFromVocab, type EncoderTokenizerJSON } from "../src";

const VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", ",", "world", "!", "un", "##believ", "##able", "cafe", "the", "is", "paris", ".", "日", "本"];

function bert(lowercase = true): ReturnType<typeof encoderTokenizerFromJSON> {
  const vocab = Object.fromEntries(VOCAB.map((t, i) => [t, i]));
  const json: EncoderTokenizerJSON = {
    model: { type: "WordPiece", vocab, unk_token: "[UNK]", continuing_subword_prefix: "##" },
    normalizer: { type: "BertNormalizer", lowercase, clean_text: true, handle_chinese_chars: true, strip_accents: null },
    post_processor: {
      type: "TemplateProcessing",
      single: [{ SpecialToken: { id: "[CLS]", type_id: 0 } }, { Sequence: { id: "A", type_id: 0 } }, { SpecialToken: { id: "[SEP]", type_id: 0 } }],
      pair: [
        { SpecialToken: { id: "[CLS]", type_id: 0 } },
        { Sequence: { id: "A", type_id: 0 } },
        { SpecialToken: { id: "[SEP]", type_id: 0 } },
        { Sequence: { id: "B", type_id: 1 } },
        { SpecialToken: { id: "[SEP]", type_id: 1 } },
      ],
      special_tokens: { "[CLS]": { ids: [2] }, "[SEP]": { ids: [3] } },
    },
    added_tokens: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"].map((t, i) => ({ id: i, content: t, special: true })),
  };
  return encoderTokenizerFromJSON(json);
}

describe("WordPiece encoder tokenizer", () => {
  it("normalizes, splits punctuation and matches longest pieces", () => {
    const tok = bert();
    expect(tok.encode("Hello, WORLD!").ids).toEqual([2, 5, 6, 7, 8, 3]);
    expect(tok.encode("unbelievable").ids).toEqual([2, 9, 10, 11, 3]);
    // Accents stripped with lowercase; an uncovered word is one [UNK].
    expect(tok.encode("Café xyz").ids).toEqual([2, 12, 1, 3]);
    // CJK ideographs are words of their own.
    expect(tok.encode("日本").ids).toEqual([2, 17, 18, 3]);
  });

  it("keeps added tokens such as [MASK] whole, before lowercasing", () => {
    expect(bert().encode("The capital is [MASK].").ids).toEqual([2, 13, 1, 14, 4, 16, 3]);
    expect(bert().maskToken).toBe("[MASK]");
  });

  it("builds pairs with segment ids and truncates the longer text first", () => {
    const tok = bert();
    const e = tok.encode("hello world", "the cafe is paris");
    expect(e.ids).toEqual([2, 5, 7, 3, 13, 12, 14, 15, 3]);
    expect(e.typeIds).toEqual([0, 0, 0, 0, 1, 1, 1, 1, 1]);
    expect(e.seq).toEqual([-1, 0, 0, -1, 1, 1, 1, 1, -1]);
    expect(tok.encode("hello world", "the cafe is paris", 7).ids).toEqual([2, 5, 7, 3, 13, 12, 3]);
  });

  it("decodes without ## joins and special tokens", () => {
    const tok = bert();
    expect(tok.decode(tok.encode("unbelievable, hello world!").ids)).toBe("unbelievable, hello world!");
  });

  it("builds BERT's tokenizer from vocab.txt", () => {
    const tok = wordPieceFromVocab(VOCAB.join("\n") + "\n");
    expect(tok.encode("Hello [MASK]", "world").ids).toEqual([2, 5, 4, 3, 7, 3]);
    expect(tok.encode("Hello [MASK]", "world").typeIds).toEqual([0, 0, 0, 0, 1, 1]);
  });
});
