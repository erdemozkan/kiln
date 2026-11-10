import { describe, expect, it } from "vitest";
import { FLORENCE2_ADDED_TOKENS, davitName, davitPositions, florence2ConfigFromHF, florence2Parse, florence2Prompt } from "../src";

// microsoft/Florence-2-base-ft's config.json, the fields read.
const CONFIG = {"model_type": "florence2", "projection_dim": 768, "vision_config": {"model_type": "davit", "dim_embed": [128, 256, 512, 1024], "num_heads": [4, 8, 16, 32], "num_groups": [4, 8, 16, 32], "depths": [1, 1, 9, 1], "patch_size": [7, 3, 3, 3], "patch_stride": [4, 2, 2, 2], "patch_padding": [3, 1, 1, 1], "patch_prenorm": [false, true, true, true], "window_size": 12, "projection_dim": 768, "image_pos_embed": {"type": "learned_abs_2d", "max_pos_embeddings": 50}, "visual_temporal_embedding": {"type": "COSINE", "max_temporal_embeddings": 100}, "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"]}, "text_config": {"d_model": 768, "encoder_layers": 6, "decoder_layers": 6, "encoder_attention_heads": 12, "decoder_attention_heads": 12, "encoder_ffn_dim": 3072, "decoder_ffn_dim": 3072, "vocab_size": 51289, "max_position_embeddings": 1024, "activation_function": "gelu", "scale_embedding": false, "pad_token_id": 1, "eos_token_id": 2, "decoder_start_token_id": 2}};

describe("Florence-2", () => {
  it("reads the config: DaViT stages and a BART language model", () => {
    const c = florence2ConfigFromHF(CONFIG);
    expect(c.vision).toMatchObject({ dims: [128, 256, 512, 1024], depths: [1, 1, 9, 1], window: 12, projection: 768, patchPrenorm: [false, true, true, true] });
    expect(c.text).toMatchObject({ family: "bart", modelType: "florence2", dModel: 768, encLayers: 6, decLayers: 6, posOffset: 2, layernormEmbedding: true, preNorm: false, decoderStart: 2 });
    expect(() => florence2ConfigFromHF({ ...CONFIG, vision_config: { ...CONFIG.vision_config, image_feature_source: ["last_frame"] } })).toThrow(/feature source/);
  });

  it("maps the vision tower's names", () => {
    expect(davitName("vision_tower.convs.2.proj.weight")).toBe("conv2.weight");
    expect(davitName("vision_tower.convs.0.norm.bias")).toBe("conv0.norm.bias");
    expect(davitName("vision_tower.blocks.2.7.spatial_block.window_attn.fn.qkv.weight")).toBe("b2.7.s.qkv.weight");
    expect(davitName("vision_tower.blocks.1.0.channel_block.conv2.fn.dw.bias")).toBe("b1.0.c.dw2.bias");
    expect(davitName("vision_tower.blocks.3.0.channel_block.ffn.fn.net.fc1.weight")).toBe("b3.0.c.fc1.weight");
    expect(davitName("vision_tower.blocks.3.0.channel_block.channel_attn.norm.weight")).toBe("b3.0.c.ln1.weight");
    expect(davitName("language_model.model.shared.weight")).toBeUndefined();
  });

  it("builds the position addend: column half, row half, plus cos 0 at odd indices", () => {
    const C = 4;
    const row = Float32Array.from([10, 11, 20, 21]); // rows 0, 1 (C/2 each)
    const col = Float32Array.from([1, 2, 3, 4]); // columns 0, 1
    const p = davitPositions(row, col, 2, C);
    // y = 1, x = 0: [col0, row1] + [0, 1, 0, 1].
    expect(Array.from(p.subarray(2 * C, 3 * C))).toEqual([1, 3, 20, 22]);
    expect(Array.from(p.subarray(1 * C, 2 * C))).toEqual([3, 5, 10, 12]);
  });

  it("adds the processor's tokens at the ids of onnx-community's added_tokens.json", () => {
    const id = (t: string) => 50265 + FLORENCE2_ADDED_TOKENS.indexOf(t);
    expect([id("</od>"), id("</ocr>"), id("</cap>"), id("</poly>"), id("<and>"), FLORENCE2_ADDED_TOKENS.length]).toEqual([50266, 50268, 51270, 51287, 51288, 1024]);
  });

  it("builds prompts as the processors do", () => {
    expect(florence2Prompt("<OD>")).toBe("Locate the objects with category name in the image.");
    expect(florence2Prompt("<CAPTION_TO_PHRASE_GROUNDING>A green car")).toBe("Locate the phrases in the caption: A green car");
    expect(florence2Prompt("What is this?")).toBe("What is this?");
  });

  it("parses answers into pixels: boxes, quad boxes, polygons", () => {
    const od = florence2Parse("</s><s>car<loc_52><loc_333><loc_932><loc_774>door<loc_710><loc_442><loc_869><loc_651></s>", "<OD>", 1000, 500);
    expect(od).toEqual({ "<OD>": { labels: ["car", "door"], bboxes: [[52.5, 166.75, 932.5, 387.25], [710.5, 221.25, 869.5, 325.75]] } });
    // A label-less box takes the previous label (phrase grounding of one phrase twice).
    const g = florence2Parse("<s>A car<loc_0><loc_0><loc_9><loc_9><loc_10><loc_10><loc_19><loc_19></s>", "<CAPTION_TO_PHRASE_GROUNDING>", 1000, 1000);
    expect(g["<CAPTION_TO_PHRASE_GROUNDING>"]).toEqual({ labels: ["A car", "A car"], bboxes: [[0.5, 0.5, 9.5, 9.5], [10.5, 10.5, 19.5, 19.5]] });
    const ocr = florence2Parse("<s>CUDA<loc_1><loc_2><loc_3><loc_2><loc_3><loc_4><loc_1><loc_4></s>", "<OCR_WITH_REGION>", 1000, 1000);
    expect(ocr["<OCR_WITH_REGION>"]).toEqual({ labels: ["CUDA"], quad_boxes: [[1.5, 2.5, 3.5, 2.5, 3.5, 4.5, 1.5, 4.5]] });
    const seg = florence2Parse("<s><loc_10><loc_20><loc_30><loc_20><loc_30><loc_40></s>", "<REFERRING_EXPRESSION_SEGMENTATION>", 1000, 1000);
    // As Python's parser: with no label, its [^<]+ swallows the first "loc_10>", and the now odd
    // trailing coordinate is dropped.
    expect(seg["<REFERRING_EXPRESSION_SEGMENTATION>"]).toEqual({ labels: [""], polygons: [[[20.5, 30.5, 20.5, 30.5]]] });
    const seg2 = florence2Parse("<s>cat<loc_10><loc_20><loc_30><loc_20><loc_30><loc_40></s>", "<REFERRING_EXPRESSION_SEGMENTATION>", 1000, 1000);
    expect(seg2["<REFERRING_EXPRESSION_SEGMENTATION>"]).toEqual({ labels: ["cat"], polygons: [[[10.5, 20.5, 30.5, 20.5, 30.5, 40.5]]] });
    expect(florence2Parse("<s>A cat.</s>", "<CAPTION>", 10, 10)).toEqual({ "<CAPTION>": "A cat." });
  });
});
