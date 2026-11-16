import { describe, expect, it } from "vitest";
import { loraFromTensors } from "../src/lora";

const t = (shape: number[], v = 1) => ({ shape, data: new Float32Array(shape[0]! * shape[1]!).fill(v) });
const pair = (l: number, group: string, m: string, r: number, out = 8, inp = 4) => [
  [`base_model.model.model.layers.${l}.${group}.${m}.lora_A.weight`, t([r, inp])],
  [`base_model.model.model.layers.${l}.${group}.${m}.lora_B.weight`, t([out, r], 2)],
] as const;

describe("PEFT LoRA adapters", () => {
  it("fold lora_alpha / r into B, per module patterns and rsLoRA", () => {
    const tensors = new Map([...pair(0, "self_attn", "q_proj", 4), ...pair(1, "mlp", "down_proj", 2)]);
    const a = loraFromTensors("x", { r: 4, lora_alpha: 8, alpha_pattern: { "layers.1.mlp.down_proj": 1 } }, tensors);
    expect(a.rank).toBe(4);
    expect(a.modules.sort()).toEqual(["down_proj", "q_proj"]);
    expect(a.weights.get("0.q_proj")!.B[0]).toBe(2 * (8 / 4));
    expect(a.weights.get("1.down_proj")!.B[0]).toBe(2 * (1 / 2));
    const rs = loraFromTensors("x", { r: 4, lora_alpha: 8, use_rslora: true }, new Map(pair(0, "self_attn", "q_proj", 4)));
    expect(rs.weights.get("0.q_proj")!.B[0]).toBeCloseTo(2 * (8 / 2), 6);
  });

  it("refuses what it cannot apply", () => {
    const ok = new Map(pair(0, "self_attn", "q_proj", 4));
    expect(() => loraFromTensors("x", { use_dora: true }, ok)).toThrow(/DoRA/);
    expect(() => loraFromTensors("x", { modules_to_save: ["lm_head"] }, ok)).toThrow(/modules_to_save/);
    expect(() => loraFromTensors("x", {}, new Map([["base_model.model.model.embed_tokens.lora_embedding_A", t([4, 4])]]))).toThrow(/cannot/);
    expect(() => loraFromTensors("x", {}, new Map([pair(0, "self_attn", "q_proj", 4)[0]]))).toThrow(/lora_B/);
  });

  it("takes Gemma 4's text layers and leaves its vision and audio towers out", () => {
    const text = (l: number, m: string) =>
      [
        [`base_model.model.model.language_model.layers.${l}.self_attn.${m}.lora_A.weight`, t([4, 4])],
        [`base_model.model.model.language_model.layers.${l}.self_attn.${m}.lora_B.weight`, t([8, 4], 2)],
      ] as const;
    const tower = [
      ["base_model.model.model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.lora_A.weight", t([4, 4])],
      ["base_model.model.model.audio_tower.layers.0.self_attn.q_proj.linear.lora_B.weight", t([8, 4])],
      ["base_model.model.model.embed_vision.embedding_projection.lora_A.weight", t([4, 4])],
    ] as const;
    const a = loraFromTensors("g", { r: 4, lora_alpha: 4 }, new Map([...text(0, "q_proj"), ...text(3, "v_proj"), ...tower]));
    expect([...a.weights.keys()].sort()).toEqual(["0.q_proj", "3.v_proj"]);
    expect(a.towerTensors).toBe(3);
    expect(() => loraFromTensors("g", {}, new Map(tower))).toThrow(/only the vision or audio tower/);
  });
});
