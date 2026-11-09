import { describe, expect, it } from "vitest";
import { Ctx, STEP_BYTES, packStep, type Tensor } from "@kiln/core";
import { CpuBackend, ops } from "../src";

let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32) * 2 - 1;
const randn = (n: number) => Float32Array.from({ length: n }, rand);

describe("speech front-end kernels (CPU)", () => {
  const be = new CpuBackend({ simd: false });
  const step = be.alloc(STEP_BYTES);
  be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 1, temperature: 0 }));
  const upload = (data: Float32Array, shape: number[]): Tensor => {
    const buf = be.alloc(data.byteLength);
    be.write(buf, 0, data);
    return { shape, dtype: "f32", buf };
  };
  const run = async (build: (ctx: Ctx) => Tensor, n: number) => {
    const ctx = new Ctx(be, step, "t");
    const out = build(ctx);
    be.submit(ctx.dispatches);
    return new Float32Array(await be.readback(out.buf, 0, n * 4));
  };

  it("channelNorm normalizes each column over time (GroupNorm, one group per channel)", async () => {
    const [T, C] = [37, 5];
    const x = randn(T * C);
    const w = randn(C);
    const b = randn(C);
    const y = await run((ctx) => ops.channelNorm(ctx, upload(x, [T, C]), upload(w, [C]), upload(b, [C]), 1e-5), T * C);
    for (let c = 0; c < C; c++) {
      let m = 0;
      for (let t = 0; t < T; t++) m += x[t * C + c]!;
      m /= T;
      let v = 0;
      for (let t = 0; t < T; t++) v += (x[t * C + c]! - m) ** 2;
      const inv = 1 / Math.sqrt(v / T + 1e-5);
      for (let t = 0; t < T; t++) expect(y[t * C + c]).toBeCloseTo((x[t * C + c]! - m) * inv * w[c]! + b[c]!, 5);
    }
  });

  it("groupConv1d matches a grouped convolution with padding (weights [k, C/groups, O])", async () => {
    const [T, C, O, k, groups, pad] = [11, 8, 6, 4, 2, 2];
    const cg = C / groups;
    const og = O / groups;
    const x = randn(T * C);
    const wt = randn(O * cg * k); // PyTorch layout [O, Cg, k]
    const bias = randn(O);
    const w = new Float32Array(k * cg * O);
    for (let o = 0; o < O; o++) for (let c = 0; c < cg; c++) for (let j = 0; j < k; j++) w[(j * cg + c) * O + o] = wt[(o * cg + c) * k + j]!;
    const tout = T; // drops the even kernel's extra frame, as wav2vec2 does
    const y = await run((ctx) => ops.groupConv1d(ctx, upload(x, [T, C]), upload(w, [k, cg, O]), upload(bias, [O]), { k, pad, groups, tout }), tout * O);
    for (let t = 0; t < tout; t++) {
      for (let o = 0; o < O; o++) {
        const g = Math.floor(o / og);
        let s = bias[o]!;
        for (let j = 0; j < k; j++) {
          const src = t + j - pad;
          if (src < 0 || src >= T) continue;
          for (let c = 0; c < cg; c++) s += x[src * C + g * cg + c]! * wt[(o * cg + c) * k + j]!;
        }
        expect(y[t * O + o]).toBeCloseTo(s, 5);
      }
    }
  });

  it("scaleAdd sets then accumulates a weighted sum", async () => {
    const a = randn(10);
    const b = randn(10);
    const y = await run((ctx) => {
      const acc = ctx.alloc([10]);
      ops.scaleAdd(ctx, acc, upload(a, [10]), 0.25, false);
      ops.scaleAdd(ctx, acc, upload(b, [10]), 0.75, true);
      return acc;
    }, 10);
    for (let i = 0; i < 10; i++) expect(y[i]).toBeCloseTo(0.25 * a[i]! + 0.75 * b[i]!, 6);
  });
});

describe("attention probabilities (CPU)", () => {
  it("attnProbs writes softmax(q·K) of the selected heads at the step's row", async () => {
    const be = new CpuBackend({ simd: false });
    const step = be.alloc(STEP_BYTES);
    be.write(step, 0, packStep({ pos0: 2, lastRow: 0, seed: 1, temperature: 0 }));
    const [S, H, D] = [9, 3, 4];
    const q = Float32Array.from({ length: H * D }, (_, i) => Math.sin(i));
    const k = Float32Array.from({ length: S * H * D }, (_, i) => Math.cos(i * 0.7));
    const qb = be.alloc(q.byteLength);
    be.write(qb, 0, q);
    const kb = be.alloc(k.byteLength);
    be.write(kb, 0, k);
    const nSlots = 3;
    const out = be.alloc(4 * nSlots * S * 4);
    const ctx = new Ctx(be, step, "t");
    ops.attnProbs(ctx, { shape: [1, H * D], dtype: "f32", buf: qb }, kb, out, { S, H, D, mask: 0b101, scale: 0.5, slotBase: 1, nSlots });
    be.submit(ctx.dispatches);
    const y = new Float32Array(await be.readback(out, 0, 4 * nSlots * S * 4));
    [0, 2].forEach((h, i) => {
      const s = Array.from({ length: S }, (_, j) => {
        let d = 0;
        for (let e = 0; e < D; e++) d += q[h * D + e]! * k[j * H * D + h * D + e]!;
        return d * 0.5;
      });
      const m = Math.max(...s);
      const z = s.reduce((a, v) => a + Math.exp(v - m), 0);
      for (let j = 0; j < S; j++) expect(y[(2 * nSlots + 1 + i) * S + j]).toBeCloseTo(Math.exp(s[j]! - m) / z, 6);
    });
  });
});

describe("block attention (CPU)", () => {
  it("each row attends causally within its own block", async () => {
    const be = new CpuBackend({ simd: false });
    const step = be.alloc(STEP_BYTES);
    const [B, T, H, D] = [3, 5, 2, 4];
    const N = B * T;
    const C = H * D;
    const mk = (f: (i: number) => number): Tensor => {
      const data = Float32Array.from({ length: N * C }, (_, i) => f(i));
      const buf = be.alloc(data.byteLength + 64);
      be.write(buf, 0, data);
      return { shape: [N, C], dtype: "f32", buf };
    };
    const q = mk((i) => Math.sin(i * 0.3));
    const k = mk((i) => Math.cos(i * 0.7));
    const v = mk((i) => Math.sin(i * 1.1 + 2));
    const [qd, kd, vd] = await Promise.all([q, k, v].map(async (t) => new Float32Array(await be.readback(t.buf, 0, N * C * 4))));
    const ctx = new Ctx(be, step, "t");
    const out = ops.blockAttention(ctx, q, k, v, { T, H, D, scale: 0.5 });
    be.submit(ctx.dispatches);
    const y = new Float32Array(await be.readback(out.buf, 0, N * C * 4));
    for (let r = 0; r < N; r++) {
      const k0 = r - (r % T);
      for (let h = 0; h < H; h++) {
        const s: number[] = [];
        for (let j = k0; j <= r; j++) {
          let d = 0;
          for (let e = 0; e < D; e++) d += qd[r * C + h * D + e]! * kd[j * C + h * D + e]!;
          s.push(d * 0.5);
        }
        const m = Math.max(...s);
        const p = s.map((x) => Math.exp(x - m));
        const z = p.reduce((a, b) => a + b, 0);
        for (let e = 0; e < D; e++) {
          let acc = 0;
          p.forEach((w, j) => (acc += w * vd[(k0 + j) * C + h * D + e]!));
          expect(y[r * C + h * D + e]).toBeCloseTo(acc / z, 6);
        }
      }
    }
  });
});

describe("image convolution kernels (CPU)", () => {
  const be = new CpuBackend({ simd: false });
  const step = be.alloc(STEP_BYTES);
  be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 1, temperature: 0 }));
  const upload = (data: Float32Array, shape: number[]): Tensor => {
    const buf = be.alloc(data.byteLength);
    be.write(buf, 0, data);
    return { shape, dtype: "f32", buf };
  };
  const run = async (build: (ctx: Ctx) => Tensor, n: number) => {
    const ctx = new Ctx(be, step, "t");
    const out = build(ctx);
    be.submit(ctx.dispatches);
    return new Float32Array(await be.readback(out.buf, 0, n * 4));
  };

  /** PyTorch's conv2d on planar [C, H, W] with weight [O, C, k, k], for checking. */
  const torchConv = (x: Float32Array, C: number, H: number, W: number, w: Float32Array, O: number, k: number, stride: number, pad: number) => {
    const Ho = Math.floor((H + 2 * pad - k) / stride) + 1;
    const Wo = Math.floor((W + 2 * pad - k) / stride) + 1;
    const y = new Float32Array(O * Ho * Wo);
    for (let o = 0; o < O; o++)
      for (let oy = 0; oy < Ho; oy++)
        for (let ox = 0; ox < Wo; ox++) {
          let s = 0;
          for (let c = 0; c < C; c++)
            for (let ky = 0; ky < k; ky++)
              for (let kx = 0; kx < k; kx++) {
                const iy = oy * stride - pad + ky;
                const ix = ox * stride - pad + kx;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) s += x[(c * H + iy) * W + ix]! * w[((o * C + c) * k + ky) * k + kx]!;
              }
          y[(o * Ho + oy) * Wo + ox] = s;
        }
    return { y, Ho, Wo };
  };

  it("conv2d matches PyTorch's convolution (stride, padding, padded K, bias, residual, ReLU)", async () => {
    for (const [C, Cp, H, W, O, k, stride, pad] of [
      [3, 4, 9, 11, 6, 7, 2, 3],
      [8, 8, 7, 5, 70, 3, 1, 1],
      [8, 8, 7, 6, 5, 1, 2, 0],
    ] as const) {
      const x = randn(C * H * W);
      const wt = randn(O * C * k * k);
      const bias = randn(O);
      const { y: ref, Ho, Wo } = torchConv(x, C, H, W, wt, O, k, stride, pad);
      const res = randn(Ho * Wo * O);
      // Channels last, padded to Cp; weight [O, K] tap-major.
      const xl = new Float32Array(H * W * Cp);
      for (let c = 0; c < C; c++) for (let t = 0; t < H * W; t++) xl[t * Cp + c] = x[c * H * W + t]!;
      const K = Math.ceil((k * k * Cp) / 32) * 32;
      const wl = new Float32Array(O * K);
      for (let o = 0; o < O; o++) for (let c = 0; c < C; c++) for (let t = 0; t < k * k; t++) wl[o * K + t * Cp + c] = wt[(o * C + c) * k * k + t]!;
      for (const relu of [false, true]) {
        for (const withRes of [false, true]) {
          const y = await run((ctx) => ops.conv2d(ctx, upload(xl, [H * W, Cp]), upload(wl, [O, K]), upload(bias, [O]), { H, W, k, stride, pad, relu, residual: withRes ? upload(res, [Ho * Wo, O]) : undefined }).y, Ho * Wo * O);
          for (let t = 0; t < Ho * Wo; t++)
            for (let o = 0; o < O; o++) {
              let v = ref[o * Ho * Wo + t]! + bias[o]! + (withRes ? res[t * O + o]! : 0);
              if (relu) v = Math.max(v, 0);
              expect(y[t * O + o]).toBeCloseTo(v, 5);
            }
        }
      }
    }
  });

  it("maxPool2d takes the window maximum, ignoring padding", async () => {
    const [H, W, C] = [7, 6, 4];
    const x = randn(H * W * C).map((v) => v - 3); // all negative: padding must not win
    const y = await run((ctx) => ops.maxPool2d(ctx, upload(x, [H * W, C]), { H, W, k: 3, stride: 2, pad: 1 }).y, 4 * 3 * C);
    for (let oy = 0; oy < 4; oy++)
      for (let ox = 0; ox < 3; ox++)
        for (let c = 0; c < C; c++) {
          let m = -Infinity;
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              const iy = oy * 2 + dy;
              const ix = ox * 2 + dx;
              if (iy >= 0 && iy < H && ix >= 0 && ix < W) m = Math.max(m, x[(iy * W + ix) * C + c]!);
            }
          expect(y[(oy * 3 + ox) * C + c]).toBe(m);
        }
  });
});
