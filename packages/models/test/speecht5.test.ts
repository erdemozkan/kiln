import { describe, expect, it } from "vitest";
import { speecht5ConfigFromHF, speecht5Name, speecht5Positional } from "../src";

describe("SpeechT5", () => {
  it("reads the config", () => {
    const c = speecht5ConfigFromHF({ model_type: "speecht5", architectures: ["SpeechT5ForTextToSpeech"], hidden_size: 768, encoder_layers: 12, decoder_layers: 6, vocab_size: 81, max_speech_positions: 1876 });
    expect(c.seq).toMatchObject({ family: "speecht5", dModel: 768, encLayers: 12, decLayers: 6, preNorm: false, relativeMax: 160, maxPositions: 600 });
    expect(c).toMatchObject({ mel: 80, reduction: 2, maxSpeechPositions: 1876, postnetLayers: 5, postnetKernel: 5 });
    expect(() => speecht5ConfigFromHF({ model_type: "speecht5", architectures: ["SpeechT5ForSpeechToText"] })).toThrow(/SpeechT5ForTextToSpeech/);
  });

  it("maps checkpoint names onto the seq2seq layers", () => {
    expect(speecht5Name("speecht5.encoder.wrapped_encoder.layers.3.attention.k_proj.weight")).toBe("enc.l3.k.weight");
    expect(speecht5Name("speecht5.encoder.wrapped_encoder.layers.0.feed_forward.output_dense.bias")).toBe("enc.l0.fc2.bias");
    expect(speecht5Name("speecht5.encoder.wrapped_encoder.layer_norm.weight")).toBe("enc.lnE.weight");
    expect(speecht5Name("speecht5.encoder.wrapped_encoder.embed_positions.pe_k.weight")).toBe("enc.relpos");
    expect(speecht5Name("speecht5.encoder.prenet.embed_tokens.weight")).toBe("shared.weight");
    expect(speecht5Name("speecht5.decoder.wrapped_decoder.layers.5.encoder_attn.out_proj.weight")).toBe("dec.l5.co.weight");
    expect(speecht5Name("speecht5.decoder.wrapped_decoder.layers.2.self_attn_layer_norm.bias")).toBe("dec.l2.ln1.bias");
    expect(speecht5Name("speecht5.decoder.wrapped_decoder.layers.2.encoder_attn_layer_norm.bias")).toBe("dec.l2.lnc.bias");
    expect(speecht5Name("speecht5.decoder.prenet.final_layer.weight")).toBeUndefined();
    expect(speecht5Name("speech_decoder_postnet.feat_out.weight")).toBeUndefined();
  });

  it("builds the sinusoidal table as torch does", () => {
    // torch: pe[7] of SpeechT5ScaledPositionalEncoding(dim 8).
    const ref = [0.6569865942001343, 0.7539022564888, 0.6442176103591919, 0.7648422122001648, 0.06994284689426422, 0.9975510239601135, 0.006999942008405924, 0.9999755024909973];
    const pe = speecht5Positional(7, 8);
    ref.forEach((v, i) => expect(pe[i]).toBeCloseTo(v, 6));
  });
});
