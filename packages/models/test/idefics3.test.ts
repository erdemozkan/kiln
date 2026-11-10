import { describe, expect, it } from "vitest";
import { idefics3Processing, idefics3TextName, idefics3VisionName, imagePatches, patchK, patchOrder, visionConfigFromIdefics3 } from "@kiln/models";

const CONFIG = {
  model_type: "idefics3",
  image_token_id: 49190,
  scale_factor: 4,
  text_config: { model_type: "llama", hidden_size: 576 },
  vision_config: { hidden_act: "gelu_pytorch_tanh", hidden_size: 768, image_size: 512, intermediate_size: 3072, num_attention_heads: 12, num_hidden_layers: 12, patch_size: 16, layer_norm_eps: 1e-6 },
};
const PRE = { do_image_splitting: true, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5], max_image_size: { longest_edge: 512 }, resample: 1, size: { longest_edge: 2048 } };

/** torch's permute of a contiguous tensor, as a new contiguous array. */
function permute(data: number[], shape: number[], perm: number[]): { data: number[]; shape: number[] } {
  const out = perm.map((p) => shape[p]!);
  const stride = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  const res: number[] = [];
  const idx = out.map(() => 0);
  for (let n = 0; n < data.length; n++) {
    res.push(data[idx.reduce((o, v, i) => o + v * stride[perm[i]!]!, 0)]!);
    for (let i = out.length - 1; i >= 0 && ++idx[i]! === out[i]; i--) idx[i] = 0;
  }
  return { data: res, shape: out };
}

describe("Idefics3 (SmolVLM)", () => {
  it("reads the vision config and the processor", () => {
    const v = visionConfigFromIdefics3(CONFIG, PRE);
    expect(v).toMatchObject({ projector: "idefics3", scaleFactor: 4, imageSize: 512, patchSize: 16, hidden: 768, layers: 12, heads: 12, projectionDim: 576, tokensPerSide: 8 });
    expect(idefics3Processing(CONFIG, PRE, v)).toEqual({ longestEdge: 2048, tileSize: 512, split: true, resample: 1, imageTokenId: 49190 });
    expect(() => visionConfigFromIdefics3({ ...CONFIG, vision_config: { ...CONFIG.vision_config, hidden_act: "relu" } })).toThrow(/activation/);
  });

  it("maps tensor names", () => {
    expect(idefics3VisionName("model.vision_model.encoder.layers.3.self_attn.out_proj.weight")).toBe("v.blk.3.attn_out.weight");
    expect(idefics3VisionName("model.vision_model.encoder.layers.0.mlp.fc1.bias")).toBe("v.blk.0.ffn_up.bias");
    expect(idefics3VisionName("model.vision_model.encoder.layers.11.layer_norm2.weight")).toBe("v.blk.11.ln2.weight");
    expect(idefics3VisionName("model.vision_model.embeddings.position_embedding.weight")).toBe("v.position_embd.weight");
    expect(idefics3VisionName("model.connector.modality_projection.proj.weight")).toBe("mm.input_projection.weight");
    expect(idefics3VisionName("model.text_model.layers.0.mlp.up_proj.weight")).toBeUndefined();
    expect(idefics3TextName("model.text_model.layers.0.mlp.up_proj.weight")).toBe("model.layers.0.mlp.up_proj.weight");
    expect(idefics3TextName("lm_head.weight")).toBe("lm_head.weight");
    expect(idefics3TextName("model.vision_model.post_layernorm.bias")).toBeUndefined();
  });

  it("orders patches so the post-norm rows are the connector's pixel shuffle", () => {
    // A 6 × 6 patch grid, scale 3, embedding 2: transformers' pixel_shuffle, step by step.
    const cfg = { ...visionConfigFromIdefics3({ ...CONFIG, scale_factor: 3, vision_config: { ...CONFIG.vision_config, image_size: 96 } }), hidden: 2 };
    const [side, s, E] = [6, 3, 2];
    let t = { data: Array.from({ length: side * side * E }, (_, i) => i), shape: [side, side / s, E * s] };
    t = permute(t.data, t.shape, [1, 0, 2]);
    t = permute(t.data, [side / s, side / s, E * s * s], [1, 0, 2]);
    // Kiln: rows in patchOrder, read as [tokens, s² · E].
    const order = patchOrder(cfg);
    const kiln = [...order].flatMap((p) => [p * E, p * E + 1]);
    expect(kiln).toEqual(t.data);
  });

  it("writes patches in that order", () => {
    const cfg = visionConfigFromIdefics3({ ...CONFIG, scale_factor: 2, vision_config: { ...CONFIG.vision_config, image_size: 4, patch_size: 1 } });
    const data = new Uint8Array(4 * 4 * 3);
    for (let i = 0; i < 16; i++) data[i * 3] = i * 10; // red = 10 · raster index
    const rows = imagePatches(cfg, { width: 4, height: 4, data });
    const K = patchK(cfg);
    const red = Array.from({ length: 16 }, (_, r) => Math.round(((rows[r * K]! * 0.5 + 0.5) * 255) / 10));
    // Token (0, 0) takes patches 0, 1, 4, 5; token (0, 1) patches 2, 3, 6, 7.
    expect(red.slice(0, 8)).toEqual([0, 1, 4, 5, 2, 3, 6, 7]);
  });
});
