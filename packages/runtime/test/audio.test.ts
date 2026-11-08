import { describe, expect, it } from "vitest";
import { wav2vec2ConfigFromHF, wav2vec2Frames, wav2vec2Name } from "@kiln/models";
import { CtcTokenizer, ctcTokenizer } from "@kiln/tokenizers";
import { argmaxRows, classifyHidden, convRows, normalizeAudio, positionalConvWeight } from "../src/audio";

const base = {
  model_type: "wav2vec2",
  architectures: ["Wav2Vec2ForCTC"],
  hidden_size: 768,
  num_hidden_layers: 12,
  num_attention_heads: 12,
  intermediate_size: 3072,
  conv_dim: [512, 512, 512, 512, 512, 512, 512],
  conv_kernel: [10, 3, 3, 3, 3, 2, 2],
  conv_stride: [5, 2, 2, 2, 2, 2, 2],
  feat_extract_norm: "group",
  num_conv_pos_embeddings: 128,
  num_conv_pos_embedding_groups: 16,
  vocab_size: 32,
};

describe("wav2vec2", () => {
  it("reads the config and counts frames as the conv stack does (20 ms per frame)", () => {
    const cfg = wav2vec2ConfigFromHF(base);
    expect(cfg.conv.length).toBe(7);
    expect(cfg.stable).toBe(false);
    expect(wav2vec2Frames(cfg, 16000)).toBe(49);
    expect(wav2vec2Frames(cfg, 176000)).toBe(549);
    expect(() => wav2vec2ConfigFromHF({ ...base, model_type: "wavlm" })).toThrow(/not supported/);
  });

  it("names the backend tensors and leaves conv weights and heads to the loader", () => {
    expect(wav2vec2Name("wav2vec2.encoder.layers.3.attention.q_proj.weight")).toBe("l3.q.weight");
    expect(wav2vec2Name("hubert.encoder.layers.0.final_layer_norm.bias")).toBe("l0.ln2.bias");
    expect(wav2vec2Name("wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight")).toBe("fe.0.norm.weight");
    expect(wav2vec2Name("wav2vec2.feature_extractor.conv_layers.2.conv.weight")).toBeUndefined();
    expect(wav2vec2Name("wav2vec2.encoder.pos_conv_embed.conv.weight_v")).toBeUndefined();
    expect(wav2vec2Name("lm_head.weight")).toBe("head.weight");
    expect(wav2vec2Name("projector.weight")).toBeUndefined();
    expect(wav2vec2Name("wav2vec2.masked_spec_embed")).toBeUndefined();
  });

  it("lays conv weights out as im2col rows, tap-major and padded to 32", () => {
    const raw = Float32Array.from({ length: 2 * 3 * 2 }, (_, i) => i); // [O 2, C 3, k 2]
    const { data, row } = convRows([2, 3, 2], raw);
    expect(row).toBe(32);
    // Output 1, tap 1, channel 2 = raw[(1·3 + 2)·2 + 1].
    expect(data[1 * row + 1 * 3 + 2]).toBe(raw[(1 * 3 + 2) * 2 + 1]);
    expect(data[row - 1]).toBe(0);
  });

  it("applies weight norm per tap (dim 2) and transposes to [k, Cg, O]", () => {
    const [O, Cg, k] = [2, 2, 3];
    const v = Float32Array.from({ length: O * Cg * k }, (_, i) => i + 1);
    const g = { shape: [1, 1, k], data: Float32Array.of(1, 2, 3) };
    const w = positionalConvWeight([O, Cg, k], v, g);
    for (let j = 0; j < k; j++) {
      let n = 0;
      for (let i = 0; i < O * Cg; i++) n += v[i * k + j]! ** 2;
      for (let o = 0; o < O; o++) for (let c = 0; c < Cg; c++) expect(w[(j * Cg + c) * O + o]).toBeCloseTo((g.data[j]! * v[(o * Cg + c) * k + j]!) / Math.sqrt(n), 6);
    }
  });

  it("normalizes audio to zero mean, unit variance", () => {
    const x = normalizeAudio(Float32Array.from({ length: 1000 }, (_, i) => Math.sin(i) * 0.3 + 0.1));
    const m = x.reduce((a, b) => a + b, 0) / x.length;
    const v = x.reduce((a, b) => a + (b - m) ** 2, 0) / x.length;
    expect(Math.abs(m)).toBeLessThan(1e-6);
    expect(v).toBeCloseTo(1, 4);
  });

  it("pools, projects and classifies hidden states", () => {
    const h = Float32Array.of(1, 2, 3, 4); // 2 frames × 2
    const logits = classifyHidden(h, 2, 2, { projector: Float32Array.of(1, 0, 0, 1, 1, 1), projectorBias: Float32Array.of(0, 0, 1), classifier: Float32Array.of(1, 1, 1), classifierBias: Float32Array.of(-1) });
    // mean [2, 3] → projector [2, 3, 6] → classifier 11 − 1.
    expect(Array.from(logits)).toEqual([10]);
    expect(Array.from(argmaxRows(Float32Array.of(0, 2, 1, 5, 0, 0), 2, 3))).toEqual([1, 0]);
  });
});

describe("CTC tokenizer", () => {
  const vocab = { "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, E: 5, L: 6, H: 7, O: 8, "'": 9, S: 10 };
  it("merges repeats, drops blanks and turns the delimiter into spaces", () => {
    const t = new CtcTokenizer(vocab);
    // H E L <pad> L O | | H E ' S
    expect(t.decode([0, 7, 7, 5, 6, 6, 0, 6, 8, 4, 4, 7, 5, 9, 10, 0])).toBe("HELLO HE'S");
    expect(t.decode([6, 6, 6, 4])).toBe("L");
  });
  it("builds from tokenizer.json's CTC decoder or vocab.json", () => {
    const fromJson = ctcTokenizer({ tokenizerJson: { model: { vocab }, decoder: { type: "CTC", pad_token: "<pad>", word_delimiter_token: "|", cleanup: true } } });
    expect(fromJson.decode([7, 8, 4, 8])).toBe("HO O");
    expect(ctcTokenizer({ vocab, config: { pad_token: "<pad>" } }).blank).toBe(0);
  });
});
