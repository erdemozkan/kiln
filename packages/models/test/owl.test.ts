import { describe, expect, it } from "vitest";
import { owlBoxBias, owlConfigFromHF, owlTextName, owlVisionName } from "@kiln/models";

describe("OWL-ViT / OWLv2", () => {
  it("reads the config with OWL's defaults", () => {
    const v2 = owlConfigFromHF({ model_type: "owlv2", projection_dim: 512, text_config: { model_type: "owlv2_text_model" }, vision_config: { image_size: 960, patch_size: 16 } });
    expect(v2).toMatchObject({ kind: "owlv2", queryDim: 512 });
    expect(v2.vision).toMatchObject({ imageSize: 960, patchSize: 16, hidden: 768, layers: 12, act: "quick_gelu", eps: 1e-5, preLn: true, finalLn: false });
    // Text: 16 positions, pooled at the highest id.
    expect(v2.text).toMatchObject({ maxPositions: 16, hidden: 512, eosId: 2, projectionDim: 512 });
    expect(owlConfigFromHF({ model_type: "owlvit" }).vision).toMatchObject({ imageSize: 768, patchSize: 32 });
    expect(() => owlConfigFromHF({ model_type: "clip" })).toThrow(/OWL/);
  });

  it("maps tensor names", () => {
    expect(owlVisionName("owlv2.vision_model.encoder.layers.3.self_attn.q_proj.weight")).toBe("l3.q.weight");
    expect(owlVisionName("owlv2.vision_model.pre_layernorm.bias")).toBe("lnPre.bias");
    expect(owlVisionName("owlvit.vision_model.post_layernorm.weight")).toBe("lnPost.weight");
    expect(owlVisionName("layer_norm.bias")).toBe("lnOwl.bias");
    expect(owlVisionName("box_head.dense2.weight")).toBe("box.2.weight");
    expect(owlVisionName("objectness_head.dense0.weight")).toBeUndefined();
    expect(owlVisionName("class_head.dense0.weight")).toBeUndefined();
    expect(owlVisionName("owlv2.vision_model.embeddings.position_embedding.weight")).toBeUndefined();
    expect(owlVisionName("owlv2.text_model.encoder.layers.0.mlp.fc1.weight")).toBeUndefined();
    expect(owlTextName("owlv2.text_model.encoder.layers.0.mlp.fc1.weight")).toBe("t0.up.weight");
    expect(owlTextName("owlvit.text_model.embeddings.token_embedding.weight")).toBe("tok.weight");
    expect(owlTextName("owlv2.vision_model.post_layernorm.bias")).toBeUndefined();
  });

  it("computes the box bias as transformers does", () => {
    // compute_box_bias(2, 3) in float32 (numpy, the same operations).
    const want = [-0.6929972171783447, 5.960464477539063e-8, -0.6929972171783447, 5.960464477539063e-8, 0.6929973363876343, 5.960464477539063e-8, -0.6929972171783447, 5.960464477539063e-8, 9.210274696350098, 5.960464477539063e-8, -0.6929972171783447, 5.960464477539063e-8, -0.6929972171783447, 9.210274696350098, -0.6929972171783447, 5.960464477539063e-8, 0.6929973363876343, 9.210274696350098, -0.6929972171783447, 5.960464477539063e-8, 9.210274696350098, 9.210274696350098, -0.6929972171783447, 5.960464477539063e-8];
    const got = owlBoxBias(2, 3);
    want.forEach((v, i) => expect(got[i]).toBeCloseTo(v, 6));
  });
});
