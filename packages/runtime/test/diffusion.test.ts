import { describe, expect, it } from "vitest";
import { taesdConfigFromDiffusers, taesdLayers, timestepFeatures, unetConfigFromDiffusers, unetResnets } from "@kiln/models";
import { alphasCumprod, gaussianNoise, scheduleTimesteps, schedulerFromDiffusers } from "../src/diffusion";

// IDKiro/sdxs-512-dreamshaper's configs, as far as they matter here.
const UNET = {
  _class_name: "UNet2DConditionModel",
  attention_head_dim: 8,
  block_out_channels: [320, 640, 1280],
  cross_attention_dim: 768,
  down_block_types: ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
  up_block_types: ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
  layers_per_block: 1,
  mid_block_type: null,
  only_cross_attention: [true, false, false],
  in_channels: 4,
  out_channels: 4,
  norm_num_groups: 32,
  norm_eps: 1e-5,
  flip_sin_to_cos: true,
  freq_shift: 0,
};
const SCHEDULER = { _class_name: "EulerDiscreteScheduler", beta_end: 0.012, beta_schedule: "scaled_linear", beta_start: 0.00085, num_train_timesteps: 1000, prediction_type: "epsilon", set_alpha_to_one: false, steps_offset: 1, timestep_spacing: "trailing" };

describe("text-to-image", () => {
  it("reads SDXS's UNet config", () => {
    const c = unetConfigFromDiffusers(UNET);
    expect(c.mid).toBe(false);
    expect(c.downAttn).toEqual([false, true, true]);
    expect(c.upAttn).toEqual([true, true, false]);
    expect(c.heads).toEqual([8, 8, 8]);
  });

  it("lists the ResNet blocks with diffusers' channel counts", () => {
    const r = unetResnets(unetConfigFromDiffusers(UNET));
    expect(r.map((b) => `${b.prefix} ${b.inC}→${b.outC}`)).toEqual([
      "down_blocks.0.resnets.0 320→320",
      "down_blocks.1.resnets.0 320→640",
      "down_blocks.2.resnets.0 640→1280",
      "up_blocks.0.resnets.0 2560→1280",
      "up_blocks.0.resnets.1 1920→1280",
      "up_blocks.1.resnets.0 1920→640",
      "up_blocks.1.resnets.1 960→640",
      "up_blocks.2.resnets.0 960→320",
      "up_blocks.2.resnets.1 640→320",
    ]);
  });

  it("refuses UNet parts it does not run", () => {
    expect(() => unetConfigFromDiffusers({ ...UNET, addition_embed_type: "text_time" })).toThrow(/addition_embed_type/);
    expect(() => unetConfigFromDiffusers({ ...UNET, down_block_types: ["SimpleCrossAttnDownBlock2D"] })).toThrow(/not supported/);
  });

  it("numbers TAESD's decoder layers as DecoderTiny's Sequential does", () => {
    const c = taesdConfigFromDiffusers({ _class_name: "AutoencoderTiny", decoder_block_out_channels: [64, 64, 64, 64], num_decoder_blocks: [3, 3, 3, 1] });
    const l = taesdLayers(c);
    const convs = l.filter((x) => x.kind === "conv").map((x) => (x as { index: number }).index);
    const blocks = l.filter((x) => x.kind === "block").map((x) => (x as { index: number }).index);
    expect(convs).toEqual([0, 6, 11, 16, 18]);
    expect(blocks).toEqual([2, 3, 4, 7, 8, 9, 12, 13, 14, 17]);
    expect(l.filter((x) => x.kind === "up")).toHaveLength(3);
  });

  it("computes the timestep features as diffusers' Timesteps (cos first)", () => {
    const f = timestepFeatures(999, 320, true, 0);
    expect(f[0]).toBeCloseTo(Math.cos(999), 4);
    expect(f[160]).toBeCloseTo(Math.sin(999), 4);
    expect(f[159]).toBeCloseTo(Math.cos(999 * Math.exp((-Math.log(10000) * 159) / 160)), 5);
  });

  it("schedules one trailing step at 999, σ as diffusers' (14.6146)", () => {
    const s = schedulerFromDiffusers("DEISMultistepScheduler", SCHEDULER);
    expect(s.kind).toBe("ddim");
    expect(scheduleTimesteps(s, 1)).toEqual([999]);
    expect(scheduleTimesteps(s, 4)).toEqual([999, 749, 499, 249]);
    const ac = alphasCumprod(s);
    expect(Math.sqrt((1 - ac[999]!) / ac[999]!)).toBeCloseTo(14.614646911621094, 3);
    expect(() => schedulerFromDiffusers("LMSDiscreteScheduler", SCHEDULER)).toThrow(/not supported/);
  });

  it("makes the same noise from the same seed", () => {
    const a = gaussianNoise(42, 10_000);
    expect(gaussianNoise(42, 10_000)).toEqual(a);
    expect(gaussianNoise(43, 10)).not.toEqual(a.subarray(0, 10));
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    const sd = Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length);
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(sd).toBeGreaterThan(0.95);
    expect(sd).toBeLessThan(1.05);
  });
});
