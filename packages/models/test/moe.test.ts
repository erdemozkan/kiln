import { describe, expect, it } from "vitest";
import { configFromHF, graniteName, isQuantizable, qwen3MoeName, weightScale } from "../src";

describe("mixture-of-experts configs", () => {
  it("reads Qwen3-MoE: experts in every decoder_sparse_step-th layer but mlp_only_layers", () => {
    const c = configFromHF({
      model_type: "qwen3_moe",
      hidden_size: 64,
      intermediate_size: 128,
      moe_intermediate_size: 96,
      num_hidden_layers: 6,
      num_attention_heads: 2,
      num_key_value_heads: 1,
      head_dim: 32,
      vocab_size: 1000,
      num_experts: 8,
      num_experts_per_tok: 2,
      norm_topk_prob: true,
      decoder_sparse_step: 2,
      mlp_only_layers: [3],
    });
    expect(c.family).toBe("qwen3");
    expect(c.qkNorm).toBe(true);
    expect(c.moe).toEqual({ experts: 8, topK: 2, intermediate: 96, renorm: true, layers: [false, true, false, false, false, true] });
    expect(qwen3MoeName("model.layers.1.mlp.gate.weight")).toBe("model.layers.1.mlp.router.weight");
    expect(qwen3MoeName("model.layers.1.mlp.experts.3.gate_proj.weight")).toBe("model.layers.1.mlp.experts.3.gate_proj.weight");
  });

  it("reads Granite-MoE: every layer has experts, routed by a softmax over the picked ones, outputs scaled", () => {
    const c = configFromHF({
      model_type: "granitemoe",
      hidden_size: 1024,
      intermediate_size: 512,
      num_hidden_layers: 24,
      num_attention_heads: 16,
      num_key_value_heads: 8,
      vocab_size: 49155,
      num_local_experts: 32,
      num_experts_per_tok: 8,
      residual_multiplier: 0.22,
      logits_scaling: 6,
      embedding_multiplier: 12,
      attention_multiplier: 0.015625,
      tie_word_embeddings: true,
    });
    expect(c.moe?.experts).toBe(32);
    expect(c.moe?.topK).toBe(8);
    expect(c.moe?.intermediate).toBe(512);
    expect(c.moe?.renorm).toBe(true);
    expect(c.moe?.layers.every(Boolean)).toBe(true);
    const down = graniteName("model.layers.0.block_sparse_moe.output_linear.weight");
    expect(down).toBe("model.layers.0.mlp.experts.down.weight");
    expect(graniteName("model.layers.0.block_sparse_moe.input_linear.weight")).toBe("model.layers.0.mlp.experts.gate_up.weight");
    expect(graniteName("model.layers.0.block_sparse_moe.router.layer.weight")).toBe("model.layers.0.mlp.router.weight");
    expect(weightScale(c, down)).toBeCloseTo(0.22);
  });

  it("keeps routers exact and quantizes experts", () => {
    expect(isQuantizable("model.layers.0.mlp.router.weight", [32, 1024])).toBe(false);
    expect(isQuantizable("model.layers.0.mlp.experts.gate_up.weight", [32768, 1024])).toBe(true);
  });
});
