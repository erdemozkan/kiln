import { describe, expect, it } from "vitest";
import { gemma4ImageGrid, GEMMA4_VIDEO_FRAME_TOKENS, type Gemma4VisionConfig } from "@kiln/models";
import { videoFrames } from "../src/engine";

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
});
