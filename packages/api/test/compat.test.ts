import { describe, expect, it } from "vitest";
import { PreTrainedTokenizer, RawImage } from "../src/compat";
import { Tensor } from "../src/tensor";
import { Tokenizer, type TokenizerJSON } from "@kiln/tokenizers";

const VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", ",", "world", "!", "the", "is", "paris", "."];

function bert(config: Record<string, unknown> = {}) {
  const vocab = Object.fromEntries(VOCAB.map((t, i) => [t, i]));
  const json = {
    model: { type: "WordPiece", vocab, unk_token: "[UNK]", continuing_subword_prefix: "##" },
    normalizer: { type: "BertNormalizer", lowercase: true, clean_text: true, handle_chinese_chars: true, strip_accents: null },
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
  return new PreTrainedTokenizer(json as never, { pad_token: "[PAD]", mask_token: "[MASK]", model_max_length: 512, ...config }) as PreTrainedTokenizer & ((t: string | string[], o?: object) => Record<string, Tensor | number[] | number[][]>);
}

describe("PreTrainedTokenizer (transformers.js API)", () => {
  it("is callable and returns int64 tensors [batch, tokens]", () => {
    const tok = bert();
    const out = tok("Hello, world!");
    const ids = out.input_ids as Tensor;
    expect(ids).toBeInstanceOf(Tensor);
    expect(ids.type).toBe("int64");
    expect(ids.dims).toEqual([1, 6]);
    expect(Array.from(ids.data as BigInt64Array, Number)).toEqual([2, 5, 6, 7, 8, 3]);
    expect((out.attention_mask as Tensor).tolist()).toEqual([[1n, 1n, 1n, 1n, 1n, 1n]]);
    expect(out.token_type_ids).toBeUndefined();
  });

  it("pads, truncates and returns plain arrays", () => {
    const tok = bert({ padding_side: "left" });
    const out = tok(["hello", "the world is paris."], { padding: true, return_tensor: false });
    expect(out.input_ids).toEqual([[0, 0, 0, 0, 2, 5, 3], [2, 9, 7, 10, 11, 12, 3]]);
    expect(out.attention_mask).toEqual([[0, 0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]);
    expect(tok("the world is paris.", { max_length: 4, truncation: true, return_tensor: false }).input_ids).toEqual([2, 9, 7, 10]);
    expect(() => tok(["hello", "hello world"])).toThrow(/padding/);
  });

  it("encodes pairs with segment ids when asked", () => {
    const tok = bert();
    const out = tok("hello", { text_pair: "world", return_tensor: false, return_token_type_ids: true });
    expect(out.input_ids).toEqual([2, 5, 3, 7, 3]);
    expect(out.token_type_ids).toEqual([0, 0, 0, 1, 1]);
  });

  it("decodes arrays, bigints and tensors", () => {
    const tok = bert();
    expect(tok.decode([2, 5, 6, 7, 8, 3])).toBe("[CLS] hello, world! [SEP]");
    expect(tok.decode([2, 5, 6, 7, 8, 3], { skip_special_tokens: true })).toBe("hello, world!");
    expect(tok.decode([5n, 7n])).toBe("hello world");
    const t = tok(["hello", "world"], { padding: true }).input_ids as Tensor;
    expect(tok.batch_decode(t, { skip_special_tokens: true })).toEqual(["hello", "world"]);
    expect(tok.tokenize("Hello world")).toEqual(["hello", "world"]);
    expect(tok.encode("hello", { add_special_tokens: false })).toEqual([5]);
    expect(tok.convert_tokens_to_ids(["[MASK]", "paris"])).toEqual([4, 11]);
    expect([tok.mask_token_id, tok.pad_token_id, tok.model_max_length]).toEqual([4, 0, 512]);
  });

  it("applies the chat template, rendering or tokenizing", () => {
    const vocab: Record<string, number> = {};
    for (const ch of "abcdefghijklmnopqrstuvwxyz:Ġ\nĊ") vocab[ch] = Object.keys(vocab).length;
    const json: TokenizerJSON = {
      model: { type: "BPE", vocab, merges: [] },
      added_tokens: [
        { id: 100, content: "<|im_start|>", special: true },
        { id: 101, content: "<|im_end|>", special: true },
      ],
      pre_tokenizer: { type: "ByteLevel", add_prefix_space: false, use_regex: true } as never,
    };
    const template = "{% for m in messages %}<|im_start|>{{ m.role }}:{{ m.content }}<|im_end|>{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant:{% endif %}";
    const tok = new PreTrainedTokenizer(json, { chat_template: template, eos_token: "<|im_end|>" });
    const chat = [{ role: "user", content: "hi" }];
    expect(tok.apply_chat_template(chat, { tokenize: false, add_generation_prompt: true })).toBe("<|im_start|>user:hi<|im_end|><|im_start|>assistant:");
    const ids = tok.apply_chat_template(chat, { return_dict: false, return_tensor: false }) as number[];
    expect(ids[0]).toBe(100);
    expect(ids.at(-1)).toBe(101);
    expect(tok.decode(ids)).toBe("<|im_start|>user:hi<|im_end|>");
    expect(tok.pad_token).toBe("<|im_end|>");
    expect(Tokenizer.fromJSON(json).encode("hi")).toEqual(ids.slice(6, 8));
  });
});

describe("RawImage", () => {
  const img = new RawImage(Uint8ClampedArray.from([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]), 2, 2, 3);

  it("converts channels", () => {
    expect(Array.from(img.grayscale().data)).toEqual([76, 150, 29, 255]);
    expect(img.rgba().channels).toBe(4);
    expect(Array.from(img.grayscale().rgb().data.slice(0, 3))).toEqual([76, 76, 76]);
  });

  it("crops and makes tensors in CHW or HWC order", async () => {
    const c = await img.crop([1, 0, 1, 1]);
    expect([c.width, c.height]).toEqual([1, 2]);
    expect(Array.from(c.data)).toEqual([0, 255, 0, 255, 255, 255]);
    const t = img.toTensor();
    expect(t.dims).toEqual([3, 2, 2]);
    expect(Array.from(t.data as Uint8Array)).toEqual([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
    expect(Array.from(RawImage.fromTensor(t).data)).toEqual(Array.from(img.data));
    expect(img.toTensor("HWC").dims).toEqual([2, 2, 3]);
  });

  it("resizes with PIL's filters, keeping the aspect ratio for a null side", async () => {
    const r = await img.resize(4, null, { resample: "nearest" });
    expect([r.width, r.height, r.channels]).toEqual([4, 4, 3]);
    expect(Array.from(r.data.slice(0, 6))).toEqual([255, 0, 0, 255, 0, 0]);
  });
});

describe("the pipeline's lazy tokenizer", () => {
  it("builds on first use and then behaves as the tokenizer", async () => {
    const { lazyTokenizer } = await import("../src/pipeline");
    let built = 0;
    const lazy = lazyTokenizer(() => (built++, bert())) as ReturnType<typeof bert>;
    expect(built).toBe(0);
    expect(lazy instanceof PreTrainedTokenizer).toBe(true);
    expect(built).toBe(1);
    const real = bert();
    expect(lazy.encode("hello world")).toEqual(real.encode("hello world"));
    expect(lazy("hello")).toEqual(real("hello"));
    expect(lazy.pad_token_id).toBe(real.pad_token_id);
    expect(Object.keys(lazy)).toEqual(Object.keys(real));
    expect(built).toBe(1);
  });
});
