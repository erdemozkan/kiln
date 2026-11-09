import { describe, expect, it } from "vitest";
import { convNetConfigFromHF, convNetConvs, convNextName, vitConfigFromHF, vitName } from "../src";

describe("ResNet / ConvNeXt / DINOv2 classifiers", () => {
  it("lists ResNet-50's convolutions as microsoft/resnet-50 names them (shortcuts only where the shape changes)", () => {
    const cfg = convNetConfigFromHF({ model_type: "resnet", depths: [3, 4, 6, 3], hidden_sizes: [256, 512, 1024, 2048], embedding_size: 64, layer_type: "bottleneck" });
    const convs = convNetConvs(cfg);
    // Stem, 16 blocks of 3 convolutions, 4 shortcuts: 53 conv + batch-norm pairs.
    expect(convs.length).toBe(53);
    expect(convs[0]).toMatchObject({ path: "resnet.embedder.embedder", to: "stem", kind: "convbn", cp: 4 });
    expect(convs.filter((c) => c.path.endsWith("shortcut")).map((c) => c.to)).toEqual(["s0.0.short", "s1.0.short", "s2.0.short", "s3.0.short"]);
    expect(convs.some((c) => c.path === "resnet.encoder.stages.2.layers.5.layer.2")).toBe(true);
  });

  it("lists ConvNeXt's patchify, downsampling and depthwise convolutions, and names its matrices", () => {
    const cfg = convNetConfigFromHF({ model_type: "convnext", depths: [3, 3, 9, 3], hidden_sizes: [96, 192, 384, 768] });
    const convs = convNetConvs(cfg);
    expect(convs.filter((c) => c.kind === "dw").length).toBe(18);
    expect(convs.filter((c) => c.to.endsWith(".down")).map((c) => c.path)).toEqual([1, 2, 3].map((s) => `convnext.encoder.stages.${s}.downsampling_layer.1`));
    expect(convNextName("convnext.encoder.stages.2.layers.8.pwconv1.weight")).toBe("s2.8.up.weight");
    expect(convNextName("convnext.encoder.stages.1.downsampling_layer.0.bias")).toBe("s1.downLn.bias");
    expect(convNextName("convnext.embeddings.layernorm.weight")).toBe("stemLn.weight");
  });

  it("reads DINOv2's config (MLP ratio, eps 1e-6) and names its norms and first MLP matrix", () => {
    const c = vitConfigFromHF({ model_type: "dinov2", hidden_size: 384, num_hidden_layers: 12, num_attention_heads: 6, mlp_ratio: 4, patch_size: 14, image_size: 518, layer_norm_eps: 1e-6 });
    expect(c).toMatchObject({ modelType: "dinov2", intermediate: 1536, eps: 1e-6, prefix: 1, finalLn: true, patchSize: 14 });
    expect(vitName("dinov2.encoder.layer.3.norm1.weight")).toBe("l3.ln1.weight");
    expect(vitName("dinov2.encoder.layer.3.mlp.fc1.bias")).toBe("l3.up.bias");
    expect(vitName("dinov2.layernorm.bias")).toBe("lnF.bias");
    expect(() => vitConfigFromHF({ model_type: "dinov2", use_swiglu_ffn: true })).toThrow(/SwiGLU/);
  });
});
