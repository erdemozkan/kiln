import { describe, expect, it } from "vitest";
import { qwenImageGrid, qwenPatches, qwenPositionEmbeddings, qwenRotaryAngles, qwenVisionConfig, qwenVisionName, type QwenVisionConfig } from "../src/qwen-vision";

const config = {
  vision_config: { depth: 12, hidden_size: 768, intermediate_size: 3072, num_heads: 12, num_position_embeddings: 2304, out_hidden_size: 1024, patch_size: 16, spatial_merge_size: 2, temporal_patch_size: 2, deepstack_visual_indexes: [] },
};
const preprocessor = { size: { longest_edge: 16777216, shortest_edge: 65536 }, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5] };

describe("Qwen 3.5 vision", () => {
  const c = qwenVisionConfig(config, preprocessor);

  it("reads the config", () => {
    expect(c).toMatchObject({ hidden: 768, heads: 12, layers: 12, patchSize: 16, mergeSize: 2, posGrid: 48, projectionDim: 1024, minTokens: 64, maxTokens: 16384 });
    expect(() => qwenVisionConfig({ vision_config: { ...config.vision_config, deepstack_visual_indexes: [3] } })).toThrow(/deepstack/);
    expect(() => qwenVisionConfig({})).toThrow(/vision_config/);
  });

  it("picks the token grid as smart_resize does", () => {
    // Rounded to whole tokens of 32 pixels.
    expect(qwenImageGrid(640, 480, c)).toEqual({ rows: 15, cols: 20 });
    expect(qwenImageGrid(650, 470, c)).toEqual({ rows: 15, cols: 20 });
    // Too small: scaled up to at least 64 tokens, keeping the shape.
    expect(qwenImageGrid(100, 100, c)).toEqual({ rows: 8, cols: 8 });
    expect(qwenImageGrid(200, 50, c)).toEqual({ rows: 4, cols: 16 });
    // Too large for the limit: scaled down to fit under it.
    const g = qwenImageGrid(4000, 3000, c, 576);
    expect(g).toEqual({ rows: 20, cols: 27 });
    expect(g.rows * g.cols).toBeLessThanOrEqual(576);
    expect(() => qwenImageGrid(4000, 10, c)).toThrow(/200 times/);
  });

  it("orders patches by merge block, values by channel then row then column", () => {
    // 64 × 32 pixels: 4 × 2 patches, two merge blocks side by side. Each pixel's red is its patch column.
    const [W, H] = [64, 32];
    const data = new Uint8Array(W * H * 3);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        data[(y * W + x) * 3] = Math.floor(x / 16) * 10;
        data[(y * W + x) * 3 + 1] = Math.floor(y / 16) * 10;
        data[(y * W + x) * 3 + 2] = y === 0 && x === 17 ? 255 : 0;
      }
    const p = qwenPatches(c, { width: W, height: H, data });
    expect(p.length).toBe(8 * 768);
    const red = (r: number) => Math.round((p[r * 768]! * 0.5 + 0.5) * 255);
    const green = (r: number) => Math.round((p[r * 768 + 256]! * 0.5 + 0.5) * 255);
    // Block (0, 0): patches (0,0) (0,1) (1,0) (1,1); then block (0, 1): (0,2) (0,3) (1,2) (1,3).
    expect([0, 1, 2, 3, 4, 5, 6, 7].map(red)).toEqual([0, 10, 0, 10, 20, 30, 20, 30]);
    expect([0, 1, 2, 3, 4, 5, 6, 7].map(green)).toEqual([0, 0, 10, 10, 0, 0, 10, 10]);
    // The blue pixel at (row 0, column 17) is patch (0, 1), row 1 of the input, at blue · row 0 · column 1.
    expect(p[1 * 768 + 2 * 256 + 1]).toBeCloseTo(1, 5);
    expect(() => qwenPatches(c, { width: 48, height: 32, data: new Uint8Array(48 * 32 * 3) })).toThrow(/whole tokens/);
  });

  it("resamples the position table with aligned corners", () => {
    const small: QwenVisionConfig = { ...c, hidden: 2, posGrid: 3 };
    // table[y · 3 + x] = [y, x]: interpolation then gives each patch's own (row, column) scaled to 0…2.
    const table = new Float32Array(18);
    for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) table.set([y, x], (y * 3 + x) * 2);
    const e = qwenPositionEmbeddings(table, small, 2, 4);
    // Merge order: (0,0) (0,1) (1,0) (1,1) (0,2) (0,3) (1,2) (1,3); rows map to 0 and 2, columns to 0, 2/3, 4/3, 2.
    const want = [[0, 0], [0, 2 / 3], [2, 0], [2, 2 / 3], [0, 4 / 3], [0, 2], [2, 4 / 3], [2, 2]].flat();
    expect(e.length).toBe(16);
    want.forEach((v, i) => expect(e[i]).toBeCloseTo(v, 5));
    // The trained grid itself comes back as stored (in merge order).
    const same = qwenPositionEmbeddings(table, { ...small, mergeSize: 1 }, 3, 3);
    expect([...same]).toEqual([...table]);
  });

  it("gives rows the first half of the rotary pairs and columns the second", () => {
    const a = qwenRotaryAngles(c, 2, 4);
    // Heads of 64: 32 pairs, 16 frequencies per axis.
    expect(a.length).toBe(8 * 32);
    // Row 5 of the input is patch (0, 3).
    expect(a[5 * 32]).toBe(0);
    expect(a[5 * 32 + 16]).toBe(3);
    expect(a[5 * 32 + 17]).toBeCloseTo(3 / 10000 ** (1 / 16), 6);
    // Row 2 is patch (1, 0).
    expect(a[2 * 32]).toBe(1);
    expect(a[2 * 32 + 16]).toBe(0);
  });

  it("names a checkpoint's tensors as the mmproj file does", () => {
    expect(qwenVisionName("model.visual.blocks.3.attn.qkv.weight")).toBe("v.blk.3.attn_qkv.weight");
    expect(qwenVisionName("model.visual.blocks.11.mlp.linear_fc2.bias")).toBe("v.blk.11.ffn_down.bias");
    expect(qwenVisionName("model.visual.merger.linear_fc2.weight")).toBe("mm.2.weight");
    expect(qwenVisionName("model.visual.merger.norm.bias")).toBe("v.post_ln.bias");
    expect(qwenVisionName("model.visual.patch_embed.proj.weight")).toBe("v.patch_embd.weight");
    expect(qwenVisionName("model.visual.pos_embed.weight")).toBe("v.position_embd.weight");
    expect(qwenVisionName("model.language_model.layers.0.mlp.up_proj.weight")).toBeUndefined();
  });
});
