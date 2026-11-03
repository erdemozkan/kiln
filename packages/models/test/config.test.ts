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
