import { describe, expect, it } from "vitest";
import { gemma4TableLeads } from "../src/load";

const t = (name: string, offset: number, byteLength: number) => ({ name, offset, byteLength });

describe("Gemma 4 load order", () => {
  it("loads the per-layer table first when it leads its file", () => {
    // A GGUF: token embeddings, the table, then the layers.
    const gguf = [t("token_embd.weight", 0, 100), t("per_layer_token_embd.weight", 100, 1600), t("blk.0.attn_q.weight", 1700, 700), t("blk.1.attn_q.weight", 2400, 700)];
    expect(gemma4TableLeads([{ tensors: gguf }])).toBe(true);
    // A checkpoint whose table comes after the layers keeps the usual order.
    const late = [t("model.language_model.layers.0.mlp.up_proj.weight", 0, 900), t("model.language_model.layers.1.mlp.up_proj.weight", 900, 900), t("model.language_model.embed_tokens_per_layer.weight", 1800, 1600), t("model.language_model.norm.weight", 3400, 10)];
    expect(gemma4TableLeads([{ tensors: late }])).toBe(false);
    // Only the file that holds the table counts; none at all is the usual order.
    expect(gemma4TableLeads([{ tensors: [t("a", 0, 5000)] }, { tensors: gguf }])).toBe(true);
    expect(gemma4TableLeads([{ tensors: [t("a", 0, 5000)] }])).toBe(false);
  });
});
