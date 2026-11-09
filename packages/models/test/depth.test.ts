import { describe, expect, it } from "vitest";
import { bicubicResize, convTransposeMatrix, depthConfigFromHF, depthName, dinov2Positions } from "../src";

describe("Depth Anything", () => {
  it("reads Depth-Anything-V2-Small-hf's config", () => {
    const c = depthConfigFromHF({
      model_type: "depth_anything",
      backbone_config: { model_type: "dinov2", hidden_size: 384, num_attention_heads: 6, image_size: 518, patch_size: 14, out_indices: [3, 6, 9, 12] },
      fusion_hidden_size: 64,
      head_hidden_size: 32,
      neck_hidden_sizes: [48, 96, 192, 384],
      reassemble_factors: [4, 2, 1, 0.5],
    });
    expect(c).toMatchObject({ hidden: 384, layers: 12, heads: 6, intermediate: 1536, patchSize: 14, eps: 1e-6, outIndices: [3, 6, 9, 12], fusion: 64, headHidden: 32 });
    expect(() => depthConfigFromHF({ model_type: "depth_anything", depth_estimation_type: "metric" })).toThrow(/metric/);
  });

  it("names DINOv2's weights and leaves the layer-scaled projections to be folded", () => {
    expect(depthName("backbone.encoder.layer.11.attention.attention.query.weight")).toBe("l11.q.weight");
    expect(depthName("backbone.encoder.layer.0.norm2.bias")).toBe("l0.ln2.bias");
    expect(depthName("backbone.encoder.layer.0.mlp.fc1.weight")).toBe("l0.up.weight");
    expect(depthName("backbone.layernorm.weight")).toBe("lnF.weight");
    expect(depthName("backbone.encoder.layer.0.attention.output.dense.weight")).toBeUndefined();
    expect(depthName("backbone.encoder.layer.0.mlp.fc2.weight")).toBeUndefined();
    expect(depthName("neck.convs.0.weight")).toBeUndefined();
  });

  it("fits position embeddings channels-last with the same bicubic resize as planar", () => {
    const [C, n0, gh, gw] = [3, 4, 4, 6];
    const table = Float32Array.from({ length: n0 * n0 * C }, (_, i) => Math.sin(i * 0.7));
    const planar = new Float32Array(C * n0 * n0);
    for (let t = 0; t < n0 * n0; t++) for (let c = 0; c < C; c++) planar[c * n0 * n0 + t] = table[t * C + c]!;
    const ref = bicubicResize(planar, C, n0, n0, gh, gw);
    const out = dinov2Positions(table, C, n0, gh, gw);
    for (let t = 0; t < gh * gw; t++) for (let c = 0; c < C; c++) expect(out[t * C + c]).toBeCloseTo(ref[c * gh * gw + t]!, 6);
  });

  it("reorders a ConvTranspose2d weight [C, O, f, f] to [f·f, O, C]", () => {
    const [C, O, f] = [2, 3, 2];
    const w = Float32Array.from({ length: C * O * f * f }, (_, i) => i);
    const m = convTransposeMatrix(w, [C, O, f, f]);
    for (let c = 0; c < C; c++) for (let o = 0; o < O; o++) for (let t = 0; t < f * f; t++) expect(m[(t * O + o) * C + c]).toBe(w[(c * O + o) * f * f + t]);
  });
});
