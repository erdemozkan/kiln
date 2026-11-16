import { describe, expect, it } from "vitest";
import { gemma4ImageGrid, GEMMA4_VIDEO_FRAME_TOKENS, QWEN_VIDEO_PAIR_TOKENS, qwenImageGrid, qwenPatches, type Gemma4VisionConfig, type QwenVisionConfig } from "@kiln/models";
import { pythonFixed1, videoFrames } from "../src/engine";

const frame = (w: number, h: number) => ({ width: w, height: h, data: new Uint8Array(w * h * 4) });

describe("video parts", () => {
  it("takes frames with their times, or times from fps (default 1)", () => {
    const f = [frame(8, 8), frame(8, 8), frame(8, 8)];
    expect(videoFrames({ frames: f, times: [0, 2.5, 61] }).times).toEqual([0, 2.5, 61]);
    expect(videoFrames({ frames: f, fps: 2 }).times).toEqual([0, 0.5, 1]);
    expect(videoFrames({ frames: f }).times).toEqual([0, 1, 2]);
  });

  it("refuses what is not decoded frames with one time each", () => {
    expect(() => videoFrames({ frames: [] })).toThrow(/at least one frame/);
    expect(() => videoFrames({ frames: ["clip.mp4"] })).toThrow(/decoded pixels/);
    expect(() => videoFrames({ frames: [frame(8, 8)], times: [0, 1] })).toThrow(/one non-negative/);
    expect(() => videoFrames({ frames: [frame(8, 8)], fps: 0 })).toThrow(/fps/);
  });

  it("gives a frame at most 70 tokens, at its own shape", () => {
    // The grid reads only the patch and pooling sizes.
    const cfg = { patchSize: 16, poolSize: 3, maxTokens: 280 } as Gemma4VisionConfig;
    const wide = gemma4ImageGrid(1280, 720, cfg, GEMMA4_VIDEO_FRAME_TOKENS);
    expect(wide.rows * wide.cols).toBeLessThanOrEqual(70);
    expect(wide.cols).toBeGreaterThan(wide.rows);
    const square = gemma4ImageGrid(640, 640, cfg, GEMMA4_VIDEO_FRAME_TOKENS);
    expect(square.rows).toBe(square.cols);
  });

  it("writes a time as Python's one-decimal format does (Qwen's <1.5 seconds>)", () => {
    // f"{x:.1f}" in Python: 0.25 → 0.2, 0.75 → 0.8, 1.25 → 1.2, 0.35 → 0.3, 2.0 → 2.0, 10.05 → 10.1.
    expect([0.25, 0.75, 1.25, 0.35, 2, 10.05, 0, 3.14159].map(pythonFixed1)).toEqual(["0.2", "0.8", "1.2", "0.3", "2.0", "10.1", "0.0", "3.1"]);
  });

  it("gives a pair of Qwen frames at most 64 tokens, and lays both frames' patches side by side", () => {
    const cfg = { patchSize: 2, mergeSize: 2, minTokens: 64, maxTokens: 16384, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] } as QwenVisionConfig;
    const real = { ...cfg, patchSize: 16 };
    const g = qwenImageGrid(1280, 720, real, QWEN_VIDEO_PAIR_TOKENS);
    expect(g.rows * g.cols).toBeLessThanOrEqual(64);
    expect(g.cols).toBeGreaterThan(g.rows);
    const a = { width: 4, height: 4, data: new Uint8Array(4 * 4 * 3).fill(255) };
    const b = { width: 4, height: 4, data: new Uint8Array(4 * 4 * 3).fill(0) };
    const one = qwenPatches(cfg, a);
    const two = qwenPatches(cfg, a, b);
    const K = 3 * 2 * 2;
    expect(two.length).toBe(2 * one.length);
    for (let r = 0; r < 4; r++) {
      expect(Array.from(two.subarray(r * 2 * K, r * 2 * K + K))).toEqual(Array.from(one.subarray(r * K, (r + 1) * K)));
      expect(Array.from(two.subarray(r * 2 * K + K, (r + 1) * 2 * K))).toEqual(new Array(K).fill(-1));
    }
    expect(() => qwenPatches(cfg, a, { width: 8, height: 4, data: new Uint8Array(8 * 4 * 3) })).toThrow(/same size/);
  });
});
