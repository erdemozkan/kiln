import { describe, expect, it } from "vitest";
import { UnsupportedModelError, configFromGGUF, configFromHF, ggufToHF, isQuantizable, ropeInvFreq } from "../src";

const qwen3 = {
  model_type: "qwen3",
  hidden_size: 1024,
  intermediate_size: 3072,
  num_hidden_layers: 28,
  num_attention_heads: 16,
  num_key_value_heads: 8,
  head_dim: 128,
  vocab_size: 151936,
  rms_norm_eps: 1e-6,
  rope_theta: 1000000,
  tie_word_embeddings: true,
  attention_bias: false,
  eos_token_id: 151645,
};

describe("configFromHF", () => {
  it("maps Qwen3 (explicit head_dim, qk-norm, merged eos ids)", () => {
    const c = configFromHF(qwen3, { eos_token_id: [151645, 151643] });
    expect(c).toMatchObject({ family: "qwen3", headDim: 128, numKvHeads: 8, qkNorm: true, attentionBias: false, tieEmbeddings: true });
    expect(c.eosTokenIds).toEqual([151645, 151643]);
  });

  it("maps Qwen2 with biases and Llama with llama3 rope scaling", () => {
    expect(configFromHF({ ...qwen3, model_type: "qwen2", head_dim: undefined }).attentionBias).toBe(true);
    const llama = configFromHF({
      ...qwen3,
      model_type: "llama",
      head_dim: undefined,
      rope_scaling: { rope_type: "llama3", factor: 32, low_freq_factor: 1, high_freq_factor: 4, original_max_position_embeddings: 8192 },
    });
    expect(llama.headDim).toBe(64);
    expect(llama.ropeScaling?.factor).toBe(32);
  });

  it("maps Gemma 3 text models (sliding layers, two rope bases, bf16 embedding scale)", () => {
    const gemma = {
      model_type: "gemma3_text",
      hidden_size: 640,
      intermediate_size: 2048,
      num_hidden_layers: 18,
      num_attention_heads: 4,
      num_key_value_heads: 1,
      head_dim: 256,
      vocab_size: 262144,
      rms_norm_eps: 1e-6,
      rope_theta: 1000000,
      rope_local_base_freq: 10000,
      query_pre_attn_scalar: 256,
      sliding_window: 512,
      hidden_activation: "gelu_pytorch_tanh",
      attn_logit_softcapping: null,
      final_logit_softcapping: null,
      rope_scaling: null,
      torch_dtype: "bfloat16",
      eos_token_id: 106,
    };
    const c = configFromHF({ ...gemma, layer_types: Array.from({ length: 18 }, (_, l) => ((l + 1) % 6 ? "sliding_attention" : "full_attention")) }, { eos_token_id: [1, 106] });
    expect(c).toMatchObject({ family: "gemma3", qkNorm: true, tieEmbeddings: true, activation: "gelu", sandwichNorms: true, slidingWindow: 512, ropeLocalTheta: 10000, attnScale: 1 / 16 });
    // sqrt(640) = 25.298… is 25.25 in bf16, which is what HF multiplies by.
    expect(c.embedScale).toBe(25.25);
    expect(c.eosTokenIds).toEqual([106, 1]);
    expect(c.slidingLayers!.map((s) => (s ? "s" : "F")).join("")).toBe("sssssFsssssFsssssF");
    // Older configs give a period instead of a list.
    expect(configFromHF({ ...gemma, sliding_window_pattern: 6 }).slidingLayers).toEqual(c.slidingLayers);
    expect(() => configFromHF({ ...gemma, final_logit_softcapping: 30 })).toThrow(/softcapping/);
    expect(() => configFromHF({ ...gemma, model_type: "gemma3" })).toThrow(/multimodal/);
  });

  it("refuses unknown architectures by name", () => {
    expect(() => configFromHF({ ...qwen3, model_type: "mamba" })).toThrow(UnsupportedModelError);
  });
});

describe("GGUF", () => {
  it("builds config from metadata and maps tensor names", () => {
    const meta = new Map<string, unknown>([
      ["general.architecture", "llama"],
      ["llama.embedding_length", 576],
      ["llama.feed_forward_length", 1536],
      ["llama.block_count", 30],
      ["llama.attention.head_count", 9],
      ["llama.attention.head_count_kv", 3],
      ["llama.vocab_size", 49152],
      ["tokenizer.ggml.eos_token_id", 2],
    ]);
    const c = configFromGGUF(meta, ["token_embd.weight", "blk.0.attn_q.weight"]);
    expect(c).toMatchObject({ headDim: 64, ropeInterleaved: true, tieEmbeddings: true, eosTokenIds: [2] });
    expect(ggufToHF("blk.3.ffn_down.weight")).toBe("model.layers.3.mlp.down_proj.weight");
    expect(ggufToHF("output.weight")).toBe("lm_head.weight");
    expect(ggufToHF("rope_freqs.weight")).toBeUndefined();
  });

  it("maps gemma3 the way llama.cpp writes it (Gemma 3 270M-it metadata)", () => {
    const tokens = Array.from({ length: 262144 }, (_, i) => (i === 106 ? "<end_of_turn>" : i === 1 ? "<eos>" : `t${i}`));
    const meta = new Map<string, unknown>([
      ["general.architecture", "gemma3"],
      ["gemma3.embedding_length", 640],
      ["gemma3.block_count", 18],
      ["gemma3.feed_forward_length", 2048],
      ["gemma3.attention.head_count", 4],
      ["gemma3.attention.head_count_kv", 1],
      ["gemma3.attention.key_length", 256],
      ["gemma3.attention.layer_norm_rms_epsilon", 1e-6],
      ["gemma3.rope.freq_base", 1000000],
      ["gemma3.attention.sliding_window", 512],
      ["tokenizer.ggml.tokens", tokens],
      ["tokenizer.ggml.eos_token_id", 1],
      ["tokenizer.ggml.bos_token_id", 2],
    ]);
    const c = configFromGGUF(meta, ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_q_norm.weight"]);
    expect(c).toMatchObject({
      family: "gemma3",
      headDim: 256,
      vocabSize: 262144,
      ropeInterleaved: false,
      ropeTheta: 1000000,
      ropeLocalTheta: 10000,
      tieEmbeddings: true,
      qkNorm: true,
      activation: "gelu",
      sandwichNorms: true,
      embedScale: 25.25,
      attnScale: 1 / 16,
      slidingWindow: 512,
      eosTokenIds: [1, 106],
    });
    expect(c.slidingLayers!.map((s) => (s ? "s" : "F")).join("")).toBe("sssssFsssssFsssssF");
    expect(ggufToHF("blk.2.ffn_norm.weight", "gemma3")).toBe("model.layers.2.pre_feedforward_layernorm.weight");
    expect(ggufToHF("blk.2.post_attention_norm.weight", "gemma3")).toBe("model.layers.2.post_attention_layernorm.weight");
    expect(ggufToHF("blk.2.post_ffw_norm.weight", "gemma3")).toBe("model.layers.2.post_feedforward_layernorm.weight");
    expect(ggufToHF("blk.2.ffn_norm.weight")).toBe("model.layers.2.post_attention_layernorm.weight");
    // The 62-layer 27B scales by hidden / heads, not head_dim.
    const big = new Map(meta);
    big.set("gemma3.block_count", 62).set("gemma3.embedding_length", 5376).set("gemma3.attention.head_count", 32).set("gemma3.attention.key_length", 128);
    expect(configFromGGUF(big, ["token_embd.weight"]).attnScale).toBeCloseTo(168 ** -0.5, 12);
    big.set("gemma3.rope.scaling.type", "linear");
    expect(() => configFromGGUF(big, [])).toThrow(/rope scaling/);
  });
});

describe("helpers", () => {
  it("computes plain RoPE frequencies", () => {
    const f = ropeInvFreq({ headDim: 8, ropeTheta: 10000 });
    expect([...f].map((v) => +v.toPrecision(4))).toEqual([1, 0.1, 0.01, 0.001]);
  });

  it("only quantizes 2-D non-norm weights", () => {
    expect(isQuantizable("model.layers.0.mlp.up_proj.weight", [1536, 576])).toBe(true);
    expect(isQuantizable("model.layers.0.input_layernorm.weight", [576])).toBe(false);
    expect(isQuantizable("model.layers.0.self_attn.q_proj.bias", [576, 32])).toBe(false);
    expect(isQuantizable("x.weight", [10, 33])).toBe(false);
  });
});
