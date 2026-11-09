import { describe, expect, it } from "vitest";
import { vitConfigFromHF, vitName, vitPatchRows } from "@kiln/models";
import { topLabels } from "../src/vision";

describe("vision transformers", () => {
  it("maps ViT and DeiT tensor names", () => {
    expect(vitName("vit.encoder.layer.3.attention.attention.query.weight")).toBe("l3.q.weight");
    expect(vitName("deit.encoder.layer.0.layernorm_after.bias")).toBe("l0.ln2.bias");
    expect(vitName("vit.encoder.layer.11.output.dense.weight")).toBe("l11.down.weight");
    expect(vitName("vit.layernorm.weight")).toBe("lnF.weight");
    expect(vitName("vit.embeddings.cls_token")).toBeUndefined();
    expect(vitName("classifier.weight")).toBeUndefined();
  });

  it("lays out patch rows after the prefix tokens, in (channel, row, column) order", () => {
    const cfg = vitConfigFromHF({ model_type: "deit", hidden_size: 8, num_hidden_layers: 1, num_attention_heads: 2, intermediate_size: 16, image_size: 4, patch_size: 2 });
    expect(cfg.prefix).toBe(2);
    const px = Float32Array.from({ length: 3 * 16 }, (_, i) => i);
    const rows = vitPatchRows(cfg, px);
    const K = 32; // 3 · 2 · 2 = 12, padded to 32
    expect(rows.length).toBe((2 + 4) * K);
    expect([...rows.subarray(0, 2 * K)].every((v) => v === 0)).toBe(true);
    // Patch (0, 1): channel 0 rows 0-1, columns 2-3 → pixels 2, 3, 6, 7; channel 1 starts at 16.
    expect([...rows.subarray(3 * K, 3 * K + 6)]).toEqual([2, 3, 6, 7, 18, 19]);
  });

  it("softmaxes logits into the best labels", () => {
    const r = topLabels(Float32Array.of(0, Math.log(3), -10), ["a", "b", "c"], 2);
    expect(r.map((x) => x.label)).toEqual(["b", "a"]);
    expect(r[0]!.score).toBeCloseTo(0.75, 3);
  });
});
