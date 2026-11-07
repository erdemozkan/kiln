import { describe, expect, it } from "vitest";
import { UnsupportedModelError, configFromGGUF, configFromHF, defaultMaxSeq, ggufToHF, graniteName, isQuantizable, kvCacheBytes, multimodalTextName, ringSlots, ropeInvFreq, weightScale } from "../src";

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
    expect(() => configFromHF({ ...gemma, model_type: "gemma3" })).toThrow(/text_config/);
  });

  it("maps the multimodal Gemma 3 (4B) to its language model, with linear rope scaling on global layers", () => {
    // google/gemma-3-4b-it's text_config leaves most fields to Gemma3TextConfig's defaults.
    const c = configFromHF(
      {
        model_type: "gemma3",
        eos_token_id: [1, 106],
        text_config: { model_type: "gemma3_text", hidden_size: 2560, intermediate_size: 10240, num_hidden_layers: 34, rope_scaling: { factor: 8.0, rope_type: "linear" }, sliding_window: 1024 },
        vision_config: { model_type: "siglip_vision_model" },
        torch_dtype: "bfloat16",
      },
      { eos_token_id: [1, 106] },
    );
    expect(c).toMatchObject({ family: "gemma3", numHeads: 8, numKvHeads: 4, headDim: 256, vocabSize: 262208, slidingWindow: 1024, ropeLinear: 8, ropeTheta: 1e6, ropeLocalTheta: 10000, attnScale: 1 / 16, eosTokenIds: [1, 106] });
    expect(c.embedScale).toBe(50.5); // sqrt(2560) = 50.596 in bf16
    expect(c.slidingLayers!.filter((s) => !s)).toHaveLength(5);
    // The trained context comes from Gemma3TextConfig's default. Sliding layers keep a ring of
    // window + 128 slots, so 8192 tokens take 305 MB (1.1 GB with a full cache per layer).
    expect(c.contextLength).toBe(131072);
    expect(ringSlots(c, 8192, 0)).toBe(1152);
    expect(ringSlots(c, 8192, 5)).toBe(8192);
    expect(ringSlots(c, 1000, 0)).toBe(1000);
    expect(kvCacheBytes(c, 8192)).toBe(2 * (5 * 8192 + 29 * 1152) * 4 * 256 * 2);
    expect(defaultMaxSeq(c)).toBe(8192);
    expect(defaultMaxSeq(c, "f32")).toBe(6400);
    const global = ropeInvFreq(c);
    const local = ropeInvFreq({ headDim: 256, ropeTheta: 10000 });
    expect(global[1]).toBeCloseTo(1e6 ** (-2 / 256) / 8, 7);
    expect(local[1]).toBeCloseTo(1e4 ** (-2 / 256), 7);
    expect(() => configFromHF({ model_type: "gemma3", text_config: { rope_scaling: { rope_type: "yarn", factor: 4 } } })).toThrow(/yarn/);
    expect(multimodalTextName("language_model.model.layers.3.mlp.up_proj.weight")).toBe("model.layers.3.mlp.up_proj.weight");
    expect(multimodalTextName("model.language_model.layers.3.mlp.up_proj.weight")).toBe("model.layers.3.mlp.up_proj.weight");
    expect(multimodalTextName("model.language_model.embed_tokens.weight")).toBe("model.embed_tokens.weight");
    expect(multimodalTextName("vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight")).toBeUndefined();
    expect(multimodalTextName("multi_modal_projector.mm_soft_emb_norm.weight")).toBeUndefined();
  });

  it("reads the trained context length", () => {
    expect(configFromHF({ ...qwen3, max_position_embeddings: 40960 }).contextLength).toBe(40960);
    expect(configFromHF(qwen3).contextLength).toBeUndefined();
  });

  it("refuses unknown architectures by name", () => {
    expect(() => configFromHF({ ...qwen3, model_type: "mamba" })).toThrow(UnsupportedModelError);
  });
});

describe("Granite", () => {
  // ibm-granite/granite-4.0-350m's config.json, trimmed.
  const g4 = {
    model_type: "granitemoehybrid",
    hidden_size: 1024,
    intermediate_size: 2048,
    num_hidden_layers: 28,
    num_attention_heads: 16,
    num_key_value_heads: 4,
    vocab_size: 100352,
    rms_norm_eps: 1e-5,
    rope_theta: 10000000,
    attention_multiplier: 0.015625,
    embedding_multiplier: 12,
    residual_multiplier: 0.263,
    logits_scaling: 4,
    layer_types: Array(28).fill("attention"),
    num_local_experts: 0,
    position_embedding_type: "rope",
    tie_word_embeddings: true,
    eos_token_id: 100257,
  };

  it("reads the multipliers, folding the tied head's 1/4 into the embedding scale", () => {
    const c = configFromHF(g4);
    expect([c.family, c.headDim, c.attnScale, c.residualScale, c.logitScale, c.embedScale]).toEqual(["llama", 64, 0.015625, 0.263, 0.25, 48]);
    expect(weightScale(c, "model.layers.3.self_attn.o_proj.weight")).toBe(0.263);
    expect(weightScale(c, "model.layers.3.mlp.down_proj.weight")).toBe(0.263);
    expect(weightScale(c, "model.embed_tokens.weight")).toBe(0.25);
    expect(weightScale(c, "model.layers.3.self_attn.q_proj.weight")).toBeUndefined();
    const untied = configFromHF({ ...g4, model_type: "granite", tie_word_embeddings: false });
    expect([untied.embedScale, weightScale(untied, "model.embed_tokens.weight"), weightScale(untied, "lm_head.weight")]).toEqual([12, undefined, 0.25]);
    expect(weightScale(configFromHF({ ...g4, model_type: "llama" }), "model.layers.0.mlp.down_proj.weight")).toBeUndefined();
    expect(configFromHF({ ...g4, intermediate_size: 512, shared_intermediate_size: 2048 }).intermediateSize).toBe(2048);
  });

  it("maps Granite 4's shared MLP onto the fused gate_up and down_proj", () => {
    expect(graniteName("model.layers.7.shared_mlp.input_linear.weight")).toBe("model.layers.7.mlp.gate_up.weight");
    expect(graniteName("model.layers.7.shared_mlp.output_linear.weight")).toBe("model.layers.7.mlp.down_proj.weight");
  });

  it("refuses the hybrid and mixture-of-experts variants", () => {
    expect(() => configFromHF({ ...g4, layer_types: ["mamba", ...Array(27).fill("attention")] })).toThrow(/Mamba/);
    expect(() => configFromHF({ ...g4, num_local_experts: 32 })).toThrow(/experts/);
    expect(() => configFromHF({ ...g4, position_embedding_type: "nope" })).toThrow(/nope/);
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
    // No context_length: the old 2048 default. With one, the smaller of it and 8192.
    expect(c.contextLength).toBeUndefined();
    expect(defaultMaxSeq(c)).toBe(2048);
    expect(defaultMaxSeq(configFromGGUF(new Map([...meta, ["llama.context_length", 8192]]), []))).toBe(8192);
    expect(defaultMaxSeq(configFromGGUF(new Map([...meta, ["llama.context_length", 1024]]), []))).toBe(1024);
    expect(defaultMaxSeq(configFromGGUF(new Map([...meta, ["llama.context_length", 131072]]), []))).toBe(8192);
    expect(ggufToHF("blk.3.ffn_down.weight")).toBe("model.layers.3.mlp.down_proj.weight");
    expect(ggufToHF("output.weight")).toBe("lm_head.weight");
    expect(ggufToHF("rope_freqs.weight")).toBeUndefined();
  });

  it("reads granite metadata: per-layer kv heads and the four multipliers (granite-4.0-350m Q8_0)", () => {
    const meta = new Map<string, unknown>([
      ["general.architecture", "granite"],
      ["granite.block_count", 28],
      ["granite.embedding_length", 1024],
      ["granite.feed_forward_length", 2048],
      ["granite.attention.head_count", 16],
      ["granite.attention.head_count_kv", Array(28).fill(4)],
      ["granite.vocab_size", 100352],
      ["granite.attention.scale", 0.015625],
      ["granite.embedding_scale", 12],
      ["granite.residual_scale", Math.fround(0.263)],
      ["granite.logit_scale", 4],
      ["granite.expert_count", 0],
    ]);
    const c = configFromGGUF(meta, ["token_embd.weight"]);
    expect(c).toMatchObject({ numKvHeads: 4, headDim: 64, ropeInterleaved: true, attnScale: 0.015625, logitScale: 0.25, embedScale: 48, residualScale: Math.fround(0.263) });
    expect(weightScale(c, ggufToHF("blk.2.attn_output.weight")!)).toBe(Math.fround(0.263));
    expect(configFromGGUF(meta, ["token_embd.weight", "output.weight"]).embedScale).toBe(12);
    expect(() => configFromGGUF(new Map([...meta, ["granite.attention.head_count_kv", [4, 8]]]), [])).toThrow(/differs between layers/);
    expect(() => configFromGGUF(new Map([...meta, ["granite.expert_count", 32]]), [])).toThrow(/experts/);
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
    big.set("gemma3.rope.scaling.type", "linear").set("gemma3.rope.scaling.factor", 8);
    expect(configFromGGUF(big, []).ropeLinear).toBe(8);
    big.set("gemma3.rope.scaling.type", "yarn");
    expect(() => configFromGGUF(big, [])).toThrow(/rope scaling "yarn"/);
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
