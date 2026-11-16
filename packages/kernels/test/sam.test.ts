import { describe, expect, it } from "vitest";
import { Ctx, STEP_BYTES, packStep, type Buf, type Tensor } from "@kiln/core";
import { CpuBackend, ops, ref } from "../src";

let seed = 11;
const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32) * 2 - 1;
const randn = (n: number) => Float32Array.from({ length: n }, rand);

describe("SAM windowed attention (CPU)", () => {
  const be = new CpuBackend({ simd: false });
  const step = be.alloc(STEP_BYTES);
  be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 1, temperature: 0 }));
  const upload = (data: Float32Array | Uint32Array, shape: number[]): Tensor => {
    const buf = be.alloc(data.byteLength);
    be.write(buf, 0, data);
    return { shape, dtype: "f32", buf };
  };

  for (const [G, W, win] of [
    [10, 10, 4],
    [7, 9, 7],
    [6, 6, 6],
  ] as const) {
    it(`matches transformers' attention on a ${G} × ${W} grid in windows of ${win}`, async () => {
      const [H, D] = [2, 8];
      const qkv = randn(G * W * 3 * H * D);
      const bias = randn(3 * H * D);
      const rh = randn((2 * win - 1) * D);
      const rw = randn((2 * win - 1) * D);
      const ctx = new Ctx(be, step, "t");
      const w = ops.samWindow(ctx, upload(qkv, [G * W, 3 * H * D]), upload(rh, [2 * win - 1, D]), upload(rw, [2 * win - 1, D]), upload(bias, [3 * H * D]), { G, W, win, heads: H, D });
      const lens: Buf = upload(new Uint32Array(w.B).fill(win * win), [w.B]).buf;
      const a = ops.attentionBatched(ctx, w.q, w.k.buf, w.v.buf, { B: w.B, L: w.L, H, D: w.DP, scale: 1, lens });
      const out = ops.samUnwindow(ctx, a, { G, W, win, heads: H, D, L: w.L, DP: w.DP, nwx: w.nwx });
      be.submit(ctx.dispatches);
      const got = new Float32Array(await be.readback(out.buf, 0, G * W * H * D * 4));
      const want = ref.samAttentionRef(qkv, bias, rh, rw, G, W, win, H, D);
      let err = 0;
      for (let i = 0; i < want.length; i++) err = Math.max(err, Math.abs(got[i]! - want[i]!));
      expect(err).toBeLessThan(1e-5);
    });
  }
});

describe("SAM mask post-processing (CPU)", () => {
  const be = new CpuBackend({ simd: false });
  const step = be.alloc(STEP_BYTES);
  be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 1, temperature: 0 }));

  it("matches SamProcessor.post_process_masks: up to the input size, crop, to the image size, threshold", async () => {
    const [side, M, inSize, cw, ch, W, H] = [16, 4, 64, 64, 40, 103, 61];
    const low = randn(side * side * M);
    const buf = be.alloc(low.byteLength);
    be.write(buf, 0, low);
    const ctx = new Ctx(be, step, "t");
    const y = ops.samMasks(ctx, { shape: [side * side, M], dtype: "f32", buf }, { side, first: 1, count: 3, inSize, cw, ch, W, H, threshold: 0.1 });
    be.submit(ctx.dispatches);
    const words = Math.ceil((W * H) / 4);
    const got = new Uint8Array(await be.readback(y.buf, 0, 3 * words * 4));
    for (let k = 0; k < 3; k++) {
      const want = ref.samMasksRef(low, side, M, 1 + k, inSize, cw, ch, W, H, 0.1);
      let wrong = 0;
      for (let i = 0; i < W * H; i++) if (want[i] !== got[k * words * 4 + i]) wrong++;
      expect(wrong).toBe(0);
    }
  });
});
