import { describe, expect, it } from "vitest";
import { visionDecoderConfigFromHF } from "../src";

describe("VisionEncoderDecoder (ViT + GPT-2)", () => {
  it("reads the GPT-2 decoder of nlpconnect/vit-gpt2-image-captioning", () => {
    const c = visionDecoderConfigFromHF({
      model_type: "vision-encoder-decoder",
      decoder_start_token_id: 50256,
      eos_token_id: 50256,
      decoder: { model_type: "gpt2", add_cross_attention: true, n_embd: 768, n_layer: 12, n_head: 12, n_positions: 1024, vocab_size: 50257, activation_function: "gelu_new" },
    });
    expect(c).toMatchObject({ family: "gpt2", dModel: 768, encLayers: 0, decLayers: 12, decHeads: 12, decFfn: 3072, vocab: 50257, preNorm: true, posOffset: 0, embedScale: 1, act: "gelu_tanh", eos: 50256, decoderStart: 50256 });
  });

  it("refuses a decoder without cross-attention or of another family", () => {
    expect(() => visionDecoderConfigFromHF({ decoder: { model_type: "gpt2", add_cross_attention: false } })).toThrow(/cross-attention/);
    expect(() => visionDecoderConfigFromHF({ decoder: { model_type: "bert" } })).toThrow(/gpt2/);
  });
});
