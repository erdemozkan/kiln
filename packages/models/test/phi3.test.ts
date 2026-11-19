import { describe, expect, it } from "vitest";
import { UnsupportedModelError, configFromGGUF, configFromHF, defaultMaxSeq, ggufToHF, phi3Name, ropeInvFreq } from "../src";

// Phi-3-mini-4k-instruct's config.json, as far as Kiln reads it.
const mini4k = {
  model_type: "phi3",
  hidden_size: 3072,
  intermediate_size: 8192,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_key_value_heads: 32,
  vocab_size: 32064,
  rms_norm_eps: 1e-5,
  rope_theta: 10000,
  rope_scaling: null,
  max_position_embeddings: 4096,
  original_max_position_embeddings: 4096,
  tie_word_embeddings: false,
  bos_token_id: 1,
  eos_token_id: 32000,
};
const short = Array.from({ length: 48 }, (_, i) => 1 + i / 100);

describe("Phi-3", () => {
  it("maps the 4k config as a plain Llama-shaped decoder", () => {
    const c = configFromHF(mini4k, { eos_token_id: [32000, 32001, 32007] });
    expect(c).toMatchObject({ family: "phi3", headDim: 96, numKvHeads: 32, attentionBias: false, qkNorm: false, ropeInterleaved: false, tieEmbeddings: false, contextLength: 4096 });
    expect(c.eosTokenIds).toEqual([32000, 32001, 32007]);
    expect(c.ropeFactors).toBeUndefined();
    expect(c.attnScale).toBeUndefined();
    expect(c.maxPositions).toBeUndefined();
  });

  it("slides every layer over the config's window, when it is shorter than the context", () => {
    const c = configFromHF({ ...mini4k, sliding_window: 2047 });
    expect(c.slidingWindow).toBe(2047);
    expect(c.slidingLayers).toEqual(Array.from({ length: 32 }, () => true));
    // Phi-3.5 mini and the 128k variants: a window as long as the context cuts nothing.
    for (const window of [4096, 262144, null, undefined]) expect(configFromHF({ ...mini4k, sliding_window: window }).slidingLayers, String(window)).toBeUndefined();
    const long = configFromHF({ ...mini4k, max_position_embeddings: 131072, sliding_window: 262144, rope_scaling: { type: "longrope", short_factor: short, long_factor: short } });
    expect(long.slidingLayers).toBeUndefined();
  });

  it("applies LongRoPE's short factors and attention factor, within the original context", () => {
    const c = configFromHF({ ...mini4k, max_position_embeddings: 131072, rope_scaling: { type: "longrope", short_factor: short, long_factor: short.map((v) => v * 10) } });
    expect(c.ropeFactors).toEqual(short);
    // √(1 + ln(32) / ln(4096)) on cos and sin, so its square on the scores.
    expect(c.attnScale).toBeCloseTo((1 + Math.log(32) / Math.log(4096)) / Math.sqrt(96), 12);
    expect(c.contextLength).toBe(4096);
    expect(c.maxPositions).toBe(4096);
    expect(defaultMaxSeq(c)).toBeLessThanOrEqual(4096);
    const plain = ropeInvFreq({ ...c, ropeFactors: undefined });
    const scaled = ropeInvFreq(c);
    for (const i of [0, 7, 47]) expect(scaled[i]).toBeCloseTo(plain[i]! / short[i]!, 6);
  });

  it("rotates part of each head for Phi-4 mini", () => {
    // Phi-4-mini-instruct's shape: heads of 128, three quarters rotated, 48 LongRoPE factors (one per rotated pair).
    const phi4 = { ...mini4k, num_attention_heads: 24, num_key_value_heads: 8, vocab_size: 200064, tie_word_embeddings: true, partial_rotary_factor: 0.75, max_position_embeddings: 131072 };
    const c = configFromHF({ ...phi4, rope_scaling: { type: "longrope", short_factor: new Array(48).fill(1), long_factor: short } });
    expect(c).toMatchObject({ headDim: 128, numKvHeads: 8, rotaryDims: 96, tieEmbeddings: true, maxPositions: 4096 });
    const f = ropeInvFreq(c);
    expect(f.length).toBe(48);
    // Frequencies spread over the rotated part: 10000^(-2i / 96).
    for (const i of [0, 1, 47]) expect(f[i]).toBeCloseTo(10000 ** ((-2 * i) / 96), 6);
    expect(() => configFromHF({ ...phi4, rope_scaling: { type: "longrope", short_factor: new Array(64).fill(1) } })).toThrow("48 positive short factors");
    expect(configFromHF(mini4k).rotaryDims).toBeUndefined();
  });

  it("refuses what it cannot run", () => {
    expect(() => configFromHF({ ...mini4k, partial_rotary_factor: 0.01 })).toThrow(/rotary embedding on 0 of 96/);
    expect(() => configFromHF({ ...mini4k, partial_rotary_factor: 1.5 })).toThrow(UnsupportedModelError);
    expect(() => configFromHF({ ...mini4k, rope_scaling: { type: "longrope", short_factor: [1, 2, 3] } })).toThrow("48 positive short factors");
    expect(() => configFromHF({ ...mini4k, rope_scaling: { type: "yarn" } })).toThrow(UnsupportedModelError);
    // Values that would give a zero or endless context, or a scale that is not a number.
    const long = (over: object, scaling: object = {}) => () => configFromHF({ ...mini4k, max_position_embeddings: 131072, ...over, rope_scaling: { type: "longrope", short_factor: short, ...scaling } });
    for (const original of [0, 1, -4096, 4096.5, 2 ** 30]) expect(long({ original_max_position_embeddings: original }), String(original)).toThrow(UnsupportedModelError);
    // A factor that is not positive is ignored, and the computed one applies.
    expect(long({}, { attention_factor: -1 })).not.toThrow();
    expect(long({}, { attention_factor: 1e9 })).toThrow(/attention factor/);
    expect(long({}, { short_factor: short.map((v, i) => (i === 5 ? Infinity : v)) })).toThrow("48 positive short factors");
    expect(long({})).not.toThrow();
  });

  it("names the stacked projections as Kiln's stacked matmuls", () => {
    expect(phi3Name("model.layers.3.self_attn.qkv_proj.weight")).toBe("model.layers.3.self_attn.qkv.weight");
    expect(phi3Name("model.layers.3.mlp.gate_up_proj.weight")).toBe("model.layers.3.mlp.gate_up.weight");
    expect(phi3Name("model.layers.3.mlp.down_proj.weight")).toBe("model.layers.3.mlp.down_proj.weight");
    expect(ggufToHF("blk.3.attn_qkv.weight", "phi3")).toBe("model.layers.3.self_attn.qkv.weight");
    expect(ggufToHF("blk.3.ffn_up.weight", "phi3")).toBe("model.layers.3.mlp.gate_up.weight");
    expect(ggufToHF("blk.3.ffn_up.weight")).toBe("model.layers.3.mlp.up_proj.weight");
    expect(ggufToHF("rope_factors_short.weight", "phi3")).toBeUndefined();
  });

  it("reads a GGUF, with and without LongRoPE", () => {
    const meta = new Map<string, unknown>([
      ["general.architecture", "phi3"],
      ["phi3.context_length", 4096],
      ["phi3.embedding_length", 3072],
      ["phi3.feed_forward_length", 8192],
      ["phi3.block_count", 32],
      ["phi3.attention.head_count", 32],
      ["phi3.attention.head_count_kv", 32],
      ["phi3.attention.layer_norm_rms_epsilon", 1e-5],
      ["phi3.rope.dimension_count", 96],
      ["tokenizer.ggml.tokens", new Array(32064).fill("x")],
      ["tokenizer.ggml.eos_token_id", 32000],
      ["tokenizer.ggml.bos_token_id", 1],
    ]);
    const names = ["token_embd.weight", "output.weight", "blk.0.attn_qkv.weight", "blk.0.ffn_up.weight"];
    const c = configFromGGUF(meta, names);
    // Unlike llama files, phi3 files keep rotary pairs in halves (no permuted q/k).
    expect(c).toMatchObject({ family: "phi3", headDim: 96, vocabSize: 32064, ropeInterleaved: false, tieEmbeddings: false, attentionBias: false, contextLength: 4096 });
    const long = new Map([...meta, ["phi3.context_length", 131072], ["phi3.rope.scaling.original_context_length", 4096], ["phi3.rope.scaling.attn_factor", 1.19]]);
    const withFactors = [...names, "rope_factors_short.weight", "rope_factors_long.weight"];
    expect(() => configFromGGUF(long, withFactors)).toThrow("rope_factors_short.weight are needed");
    const l = configFromGGUF(long, withFactors, Float32Array.from(short));
    expect(l.ropeFactors!.length).toBe(48);
    expect(l.attnScale).toBeCloseTo(1.19 ** 2 / Math.sqrt(96), 10);
    expect(l.maxPositions).toBe(4096);
    // Phi-4 mini files: fewer rotated dimensions than the head has.
    const partial = configFromGGUF(new Map([...meta, ["phi3.rope.dimension_count", 72]]), names);
    expect(partial.rotaryDims).toBe(72);
    expect(ropeInvFreq(partial).length).toBe(36);
    expect(() => configFromGGUF(new Map([...meta, ["phi3.rope.dimension_count", 73]]), names)).toThrow(/73 of 96/);
    expect(() => configFromGGUF(new Map([...meta, ["phi3.rope.dimension_count", 128]]), names)).toThrow(/128 of 96/);
  });
});
