import { describe, expect, it } from "vitest";
import { CharTokenizer, encoderTokenizerFromJSON, marianTokenizer, parseSentencePieceModel, type SentencePieceTokenizer } from "../src";

const S = "▁";
const post = {
  type: "TemplateProcessing",
  single: [{ Sequence: { id: "A" as const, type_id: 0 } }, { SpecialToken: { id: "</s>", type_id: 0 } }],
  special_tokens: { "</s>": { ids: [1] } },
};
const added = [
  { id: 0, content: "<pad>", special: true },
  { id: 1, content: "</s>", special: true },
  { id: 2, content: "<unk>", special: true },
];

function unigram(): SentencePieceTokenizer {
  const vocab: [string, number][] = [["<pad>", 0], ["</s>", 0], ["<unk>", 0], [S, -2], [`${S}hello`, -3], [`${S}he`, -4], ["llo", -4], ["l", -6], ["o", -6], [`${S}world`, -3], ["h", -7], ["e", -7]];
  return encoderTokenizerFromJSON({
    model: { type: "Unigram", vocab, unk_id: 2 } as never,
    normalizer: { type: "Sequence", normalizers: [{ type: "Replace", pattern: { Regex: " {2,}" }, content: " " }] } as never,
    pre_tokenizer: { type: "Metaspace", replacement: S, add_prefix_space: true } as never,
    post_processor: post,
    added_tokens: added,
  }) as SentencePieceTokenizer;
}

describe("SentencePiece Unigram", () => {
  it("picks the best-scoring segmentation per word", () => {
    const t = unigram();
    // ▁hello (-3) beats ▁he + llo (-8).
    expect(t.pieces("hello world")).toEqual([`${S}hello`, `${S}world`]);
    expect(t.encode("hello world").ids).toEqual([4, 9, 1]);
  });

  it("fuses runs of unknown characters, collapses and trims spaces", () => {
    const t = unigram();
    expect(t.encode("  hello   xyz world  ").ids).toEqual([4, 3, 2, 9, 1]);
    expect(t.decode(t.encode("hello world").ids)).toBe("hello world");
  });
});

describe("SentencePiece BPE (NLLB-style)", () => {
  it("merges characters by rank under Metaspace", () => {
    const vocab = { "<pad>": 0, "</s>": 1, "<unk>": 2, [S]: 3, h: 4, i: 5, [`${S}h`]: 6, [`${S}hi`]: 7 };
    const t = encoderTokenizerFromJSON({
      model: { type: "BPE", vocab, merges: [`${S} h`, `${S}h i`], unk_token: "<unk>" } as never,
      pre_tokenizer: { type: "Metaspace", replacement: S, add_prefix_space: true } as never,
      post_processor: post,
      added_tokens: added,
    }) as SentencePieceTokenizer;
    expect(t.encode("hi hih").ids).toEqual([7, 7, 4, 1]);
    expect(t.encode("hiq").ids).toEqual([7, 2, 1]);
  });
});

/** A SentencePiece ModelProto: pieces (field 1) and trainer_spec.model_type = UNIGRAM (field 2.3). */
function spmBytes(pieces: [string, number, number][], modelType = 1): Uint8Array {
  const out: number[] = [];
  const varint = (v: number) => {
    while (v > 0x7f) {
      out.push((v & 0x7f) | 0x80);
      v >>>= 7;
    }
    out.push(v);
  };
  for (const [p, score, type] of pieces) {
    const s = new TextEncoder().encode(p);
    const msg: number[] = [0x0a, s.length, ...s, 0x15, ...new Uint8Array(new Float32Array([score]).buffer), 0x18, type];
    out.push(0x0a);
    varint(msg.length);
    out.push(...msg);
  }
  out.push(0x12, 2, 0x18, modelType);
  return Uint8Array.from(out);
}

describe("Marian (.spm + vocab.json)", () => {
  it("segments with the spm and numbers pieces through vocab.json", () => {
    const spm = spmBytes([["<unk>", 0, 2], ["<s>", 0, 3], ["</s>", 0, 3], [`${S}Hallo`, -2, 1], [`${S}Welt`, -2, 1], [`${S}`, -3, 1], ["W", -5, 1]]);
    expect(parseSentencePieceModel(spm).pieces.map((p) => p.piece)).toEqual(["<unk>", "<s>", "</s>", `${S}Hallo`, `${S}Welt`, S, "W"]);
    const vocab = { "</s>": 0, "<unk>": 1, [`${S}Welt`]: 7, [`${S}Hallo`]: 9, "<pad>": 58100 };
    const t = marianTokenizer(spm, vocab);
    expect(t.encode("Hallo Welt").ids).toEqual([9, 7, 0]);
    expect(t.decode([9, 7, 0])).toBe("Hallo Welt");
  });
});

describe("character SentencePiece (SpeechT5)", () => {
  it("maps characters to ids after the dummy prefix, collapses spaces, appends </s>", () => {
    const spm = spmBytes([["<s>", 0, 3], ["<pad>", 0, 3], ["</s>", 0, 3], ["<unk>", 0, 2], [S, 0, 1], ["H", 0, 1], ["i", 0, 1], ["!", 0, 1]], 4);
    const t = new CharTokenizer(spm);
    expect(t.encode("Hi  Hi!")).toEqual([4, 5, 6, 4, 5, 6, 7, 2]);
    expect(t.encode(" Hi?")).toEqual([4, 5, 6, 3, 2]);
    expect(() => new CharTokenizer(spmBytes([["<unk>", 0, 2]], 1))).toThrow(/not a character model/);
  });
});
