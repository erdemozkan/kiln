import { describe, expect, it } from "vitest";
import { segformerConfigFromHF, segformerName } from "../src";

describe("SegFormer", () => {
  it("reads nvidia/segformer-b0's config and refuses what it cannot run", () => {
    const c = segformerConfigFromHF({ model_type: "segformer", architectures: ["SegformerForSemanticSegmentation"], hidden_sizes: [32, 64, 160, 256], id2label: { "0": "wall", "1": "building" } });
    expect(c).toMatchObject({ hidden: [32, 64, 160, 256], depths: [2, 2, 2, 2], heads: [1, 2, 5, 8], srRatios: [8, 4, 2, 1], decoderHidden: 256, labels: ["wall", "building"] });
    expect(() => segformerConfigFromHF({ model_type: "segformer", architectures: ["SegformerForImageClassification"] })).toThrow(/Semantic/);
  });

  it("names the encoder's matrices and norms and the decoder's projections, leaving the convolutions", () => {
    expect(segformerName("segformer.encoder.block.2.1.attention.self.query.weight")).toBe("b2.1.q.weight");
    expect(segformerName("segformer.encoder.block.0.0.attention.self.layer_norm.bias")).toBe("b0.0.srLn.bias");
    expect(segformerName("segformer.encoder.block.3.0.mlp.dense2.bias")).toBe("b3.0.down.bias");
    expect(segformerName("segformer.encoder.patch_embeddings.1.layer_norm.weight")).toBe("s1.patchLn.weight");
    expect(segformerName("segformer.encoder.layer_norm.3.bias")).toBe("s3.ln.bias");
    expect(segformerName("decode_head.linear_c.2.proj.weight")).toBe("dec.c2.weight");
    expect(segformerName("segformer.encoder.block.0.0.attention.self.sr.weight")).toBeUndefined();
    expect(segformerName("segformer.encoder.block.0.0.mlp.dwconv.dwconv.weight")).toBeUndefined();
    expect(segformerName("decode_head.linear_fuse.weight")).toBeUndefined();
  });
});
