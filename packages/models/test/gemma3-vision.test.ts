import { describe, expect, it } from "vitest";
import { gemma3VisionName, visionConfigFromGemma3 } from "@kiln/models";

// google/gemma-3-4b-it's config.json (vision_config and the multimodal fields) and preprocessor_config.json.
const CONFIG = {
  model_type: "gemma3",
  mm_tokens_per_image: 256,
  text_config: { hidden_size: 2560 },
  vision_config: { hidden_act: "gelu_pytorch_tanh", hidden_size: 1152, image_size: 896, intermediate_size: 4304, layer_norm_eps: 1e-6, num_attention_heads: 16, num_hidden_layers: 27, patch_size: 14 },
};
const PRE = { image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5], size: { height: 896, width: 896 } };

describe("Gemma 3's vision tower from safetensors", () => {
  it("reads the config", () => {
    expect(visionConfigFromGemma3(CONFIG, PRE)).toEqual({
      projector: "gemma3",
      scaleFactor: 4,
      imageSize: 896,
      patchSize: 14,
      hidden: 1152,
      intermediate: 4304,
      layers: 27,
      heads: 16,
      eps: 1e-6,
      projectionDim: 2560,
      tokensPerSide: 16,
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
    });
    expect(() => visionConfigFromGemma3({ ...CONFIG, mm_tokens_per_image: 200 })).toThrow(/pool/);
  });

  it("maps both transformers layouts to mmproj names", () => {
    expect(gemma3VisionName("vision_tower.vision_model.encoder.layers.26.self_attn.q_proj.weight")).toBe("v.blk.26.attn_q.weight");
    expect(gemma3VisionName("model.vision_tower.vision_model.encoder.layers.0.mlp.fc2.bias")).toBe("v.blk.0.ffn_down.bias");
    expect(gemma3VisionName("vision_tower.vision_model.embeddings.patch_embedding.bias")).toBe("v.patch_embd.bias");
    expect(gemma3VisionName("vision_tower.vision_model.post_layernorm.weight")).toBe("v.post_ln.weight");
    expect(gemma3VisionName("multi_modal_projector.mm_input_projection_weight")).toBe("mm.input_projection.weight");
    expect(gemma3VisionName("model.multi_modal_projector.mm_soft_emb_norm.weight")).toBe("mm.soft_emb_norm.weight");
    expect(gemma3VisionName("language_model.model.layers.0.mlp.up_proj.weight")).toBeUndefined();
  });
});
