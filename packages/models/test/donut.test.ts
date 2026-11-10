import { describe, expect, it } from "vitest";
import { donutSwinConfigFromHF, donutSwinName, swinRelativeBias, visionDecoderConfigFromHF } from "@kiln/models";

const ENC = { model_type: "donut-swin", depths: [2, 2, 14, 2], embed_dim: 128, hidden_act: "gelu", image_size: [2560, 1920], layer_norm_eps: 1e-5, num_heads: [4, 8, 16, 32], patch_size: 4, window_size: 10 };

describe("Donut", () => {
  it("reads the encoder and decoder configs", () => {
    expect(donutSwinConfigFromHF(ENC)).toEqual({ imageHW: [2560, 1920], patchSize: 4, embedDim: 128, depths: [2, 2, 14, 2], heads: [4, 8, 16, 32], window: 10, eps: 1e-5 });
    // 2560 / 4 / 8 = 80 rows at the last stage: whole windows; 2000 / 4 = 500 rows, then 250, 125: not mergeable.
    expect(() => donutSwinConfigFromHF({ ...ENC, image_size: [2000, 1920] })).toThrow(/merged|windows/);
    const d = visionDecoderConfigFromHF({ decoder: { model_type: "mbart", d_model: 1024, decoder_layers: 4, decoder_attention_heads: 16, decoder_ffn_dim: 4096, vocab_size: 57532, max_position_embeddings: 128, scale_embedding: true, activation_function: "gelu", add_final_layer_norm: true } });
    expect(d).toMatchObject({ family: "mbart", encLayers: 0, decLayers: 4, preNorm: true, layernormEmbedding: true, posOffset: 2, embedScale: 32, eos: 2, pad: 1 });
  });

  it("maps tensor names", () => {
    expect(donutSwinName("encoder.layers.2.blocks.13.attention.self.query.weight")).toBe("s2.b13.q.weight");
    expect(donutSwinName("encoder.layers.0.blocks.1.layernorm_after.bias")).toBe("s0.b1.ln2.bias");
    expect(donutSwinName("encoder.layers.1.downsample.reduction.weight")).toBe("m1.red.weight");
    expect(donutSwinName("embeddings.norm.weight")).toBe("embNorm.weight");
    expect(donutSwinName("encoder.layers.0.blocks.0.attention.self.relative_position_bias_table")).toBeUndefined();
    expect(donutSwinName("embeddings.patch_embeddings.projection.weight")).toBeUndefined();
  });

  it("expands the relative-position table as transformers indexes it", () => {
    const [win, heads] = [3, 2];
    const table = Float32Array.from({ length: (2 * win - 1) ** 2 * heads }, (_, i) => i);
    const bias = swinRelativeBias(table, heads, win);
    // transformers: coords (y, x) per token; relative (Δy + w − 1) · (2w − 1) + (Δx + w − 1).
    const N = win * win;
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        const idx = (Math.floor(i / win) - Math.floor(j / win) + win - 1) * (2 * win - 1) + ((i % win) - (j % win) + win - 1);
        for (let h = 0; h < heads; h++) expect(bias[(h * N + i) * N + j]).toBe(table[idx * heads + h]);
      }
    expect(bias[0 * N * N + 0]).toBe(table[((win - 1) * (2 * win - 1) + win - 1) * heads]);
  });
});
