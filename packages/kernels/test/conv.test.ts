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

  it("mulFirstRow multiplies every row after the first by the first", async () => {
    const [T, D] = [4, 3];
    const x = randn(T * D);
    const y = await run((ctx) => ops.mulFirstRow(ctx, upload(x, [T, D])), (T - 1) * D);
    for (let t = 1; t < T; t++) for (let d = 0; d < D; d++) expect(y[(t - 1) * D + d]).toBeCloseTo(x[t * D + d]! * x[d]!, 6);
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

  it("resizeBilinear matches torch's bilinear interpolate, with and without align_corners", async () => {
    // x [2, 3, 4] (channels, rows, columns) and torch's outputs, channels last.
    const [H, W, C] = [3, 4, 2];
    const x = new Float32Array(H * W * C);
    for (let c = 0; c < C; c++) for (let i = 0; i < H * W; i++) x[i * C + c] = ((i * 7) % 11) - 5 + 0.25 * c;
    const torch: Record<string, number[]> = {
      "true-5x7": [-5.0, -4.75, -1.5, -1.25, 2.0, 2.25, 0.0, 0.25, -2.0, -1.75, 1.5, 1.75, 5.0, 5.25, -2.0, -1.75, -1.25, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.25, 1.75, 2.0, 2.5, 2.75, 1.0, 1.25, -1.0, -0.75, -3.0, -2.75, 0.5, 0.75, 4.0, 4.25, 2.0, 2.25, 0.0, 0.25, -1.5, -1.25, -0.75, -0.5, 0.0, 0.25, 0.75, 1.0, 1.5, 1.75, -0.5, -0.25, -2.5, -2.25, -4.0, -3.75, -0.5, -0.25, 3.0, 3.25, 1.0, 1.25, -1.0, -0.75, -3.0, -2.75, -5.0, -4.75],
      "true-2x3": [-5.0, -4.75, 0.0, 0.25, 5.0, 5.25, -4.0, -3.75, 1.0, 1.25, -5.0, -4.75],
      "false-5x7": [-5.0, -4.75, -2.5, -2.25, 1.5, 1.75, -0.0, 0.25, -1.499999, -1.249999, 2.500001, 2.750001, 5.0, 5.25, -2.6, -2.35, -1.671428, -1.421428, -0.185714, 0.064286, 0.2, 0.45, 0.585715, 0.835715, 2.071429, 2.321429, 3.0, 3.25, 1.0, 1.25, -0.428572, -0.178572, -2.714286, -2.464286, 0.500001, 0.750001, 3.714285, 3.964285, 1.428571, 1.678571, 0.0, 0.25, -2.000001, -1.750001, -1.071429, -0.821428, 0.414287, 0.664287, 0.8, 1.05, 0.714284, 0.964284, -1.57143, -1.32143, -3.0, -2.750001, -4.0, -3.75, -1.5, -1.25, 2.5, 2.75, 1.0, 1.25, -1.285715, -1.035715, -3.571429, -3.321429, -5.0, -4.75],
      "false-2x3": [-2.791667, -2.541667, 0.125, 0.375, 3.041667, 3.291667, -2.041667, -1.791667, 0.875, 1.125, -3.083334, -2.833334],
    };
    for (const [key, ref] of Object.entries(torch)) {
      const [ac, size] = key.split("-") as [string, string];
      const [Ho, Wo] = size.split("x").map(Number) as [number, number];
      const y = await run((ctx) => ops.resizeBilinear(ctx, upload(x, [H * W, C]), { H, W, Ho, Wo, alignCorners: ac === "true" }), Ho * Wo * C);
      ref.forEach((v, i) => expect(Math.abs(y[i]! - v)).toBeLessThan(1e-5));
    }
  });

  it("convTranspose matches PyTorch's ConvTranspose2d with kernel = stride", async () => {
    const [H, W, C, O, f] = [3, 2, 5, 4, 2];
    const x = randn(H * W * C);
    const wt = randn(C * O * f * f); // PyTorch layout [C, O, f, f]
    const bias = randn(O);
    const w = new Float32Array(f * f * O * C);
    for (let c = 0; c < C; c++) for (let o = 0; o < O; o++) for (let t = 0; t < f * f; t++) w[(t * O + o) * C + c] = wt[(c * O + o) * f * f + t]!;
    const y = await run((ctx) => ops.convTranspose(ctx, upload(x, [H * W, C]), upload(w, [f * f, O, C]), upload(bias, [O]), { H, W, f }).y, H * f * W * f * O);
    for (let iy = 0; iy < H; iy++)
      for (let ix = 0; ix < W; ix++)
        for (let ky = 0; ky < f; ky++)
          for (let kx = 0; kx < f; kx++)
            for (let o = 0; o < O; o++) {
              let s = bias[o]!;
              for (let c = 0; c < C; c++) s += x[(iy * W + ix) * C + c]! * wt[((c * O + o) * f + ky) * f + kx]!;
              expect(y[((iy * f + ky) * W * f + ix * f + kx) * O + o]).toBeCloseTo(s, 5);
            }
  });

  it("conv2d with reluIn convolves the rectified input", async () => {
    const [H, W, C, O] = [4, 3, 4, 3];
    const x = randn(H * W * C);
    const K = 32;
    const w = new Float32Array(O * K);
    w.set(randn(O * C * 1).map((v) => v), 0);
    const wl = new Float32Array(O * K);
    for (let o = 0; o < O; o++) for (let c = 0; c < C; c++) wl[o * K + c] = w[o * C + c]!;
    const bias = randn(O);
    const y = await run((ctx) => ops.conv2d(ctx, upload(x, [H * W, C]), upload(wl, [O, K]), upload(bias, [O]), { H, W, k: 1, stride: 1, pad: 0, reluIn: true }).y, H * W * O);
    for (let t = 0; t < H * W; t++)
      for (let o = 0; o < O; o++) {
        let s = bias[o]!;
        for (let c = 0; c < C; c++) s += Math.max(x[t * C + c]!, 0) * w[o * C + c]!;
        expect(y[t * O + o]).toBeCloseTo(s, 5);
      }
  });

  it("conv2d convolves a batch of stacked images as each alone", async () => {
    const [n, H, W, C, O] = [3, 5, 4, 4, 6];
    const x = randn(n * H * W * C);
    const K = 64;
    const w = randn(O * K);
    for (let o = 0; o < O; o++) w.fill(0, o * K + 9 * C, (o + 1) * K);
    const bias = randn(O);
    const opts = { H, W, k: 3, stride: 1, pad: 1 };
    const all = await run((ctx) => ops.conv2d(ctx, upload(x, [n * H * W, C]), upload(w, [O, K]), upload(bias, [O]), { ...opts, n }).y, n * H * W * O);
    for (let i = 0; i < n; i++) {
      const one = await run((ctx) => ops.conv2d(ctx, upload(x.slice(i * H * W * C, (i + 1) * H * W * C), [H * W, C]), upload(w, [O, K]), upload(bias, [O]), opts).y, H * W * O);
      one.forEach((v, j) => expect(all[i * H * W * O + j]).toBeCloseTo(v, 5));
    }
  });

  it("groupNorm normalizes each image's channel groups (GroupNorm), with ReLU", async () => {
    const [n, T, C, G] = [2, 7, 12, 3];
    const x = randn(n * T * C);
    const w = randn(C);
    const b = randn(C);
    const y = await run((ctx) => ops.groupNorm(ctx, upload(x, [n * T, C]), upload(w, [C]), upload(b, [C]), { n, groups: G, eps: 1e-5, relu: true }), n * T * C);
    const cg = C / G;
    for (let i = 0; i < n; i++)
      for (let g = 0; g < G; g++) {
        const vals: number[] = [];
        for (let t = 0; t < T; t++) for (let c = 0; c < cg; c++) vals.push(x[(i * T + t) * C + g * cg + c]!);
        const m = vals.reduce((a, v) => a + v, 0) / vals.length;
        const inv = 1 / Math.sqrt(vals.reduce((a, v) => a + (v - m) ** 2, 0) / vals.length + 1e-5);
        for (let t = 0; t < T; t++)
          for (let c = 0; c < cg; c++) {
            const k = g * cg + c;
            expect(y[(i * T + t) * C + k]).toBeCloseTo(Math.max((x[(i * T + t) * C + k]! - m) * inv * w[k]! + b[k]!, 0), 5);
          }
      }
  });

  it("nearestAdd picks torch's nearest source pixels (3×5 → 7×11) and adds the shared image", async () => {
    // torch.nn.functional.interpolate(size=(7, 11), mode="nearest") source rows and columns.
    const rows = [0, 0, 0, 1, 1, 2, 2];
    const cols = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    const [n, H, W, C, Ho, Wo] = [2, 3, 5, 3, 7, 11];
    const x = randn(n * H * W * C);
    const f = randn(Ho * Wo * C);
    const y = await run((ctx) => ops.nearestAdd(ctx, upload(x, [n * H * W, C]), upload(f, [Ho * Wo, C]), { n, H, W, Ho, Wo }), n * Ho * Wo * C);
    for (let i = 0; i < n; i++)
      for (let oy = 0; oy < Ho; oy++)
        for (let ox = 0; ox < Wo; ox++)
          for (let c = 0; c < C; c++) {
            const t = oy * Wo + ox;
            expect(y[(i * Ho * Wo + t) * C + c]).toBeCloseTo(f[t * C + c]! + x[(i * H * W + rows[oy]! * W + cols[ox]!) * C + c]!, 6);
          }
  });

  it("bboxAttention softmaxes each selected query's head scores over all heads and pixels together", async () => {
    const [Q, S, Hh, D] = [4, 5, 2, 3];
    const q = randn(Q * Hh * D);
    const k = randn(S * Hh * D);
    const sel = new Uint32Array([2, 0]);
    const selBuf = be.alloc(sel.byteLength);
    be.write(selBuf, 0, sel);
    const scale = D ** -0.5;
    const y = await run((ctx) => ops.bboxAttention(ctx, upload(q, [Q, Hh * D]), upload(k, [S, Hh * D]), selBuf, { n: 2, heads: Hh, scale }), 2 * S * Hh);
    sel.forEach((row, i) => {
      const s: number[] = [];
      for (let p = 0; p < S; p++)
        for (let h = 0; h < Hh; h++) {
          let d = 0;
          for (let e = 0; e < D; e++) d += q[row * Hh * D + h * D + e]! * scale * k[p * Hh * D + h * D + e]!;
          s.push(d);
        }
      const m = Math.max(...s);
      const z = s.reduce((a, v) => a + Math.exp(v - m), 0);
      s.forEach((v, j) => expect(y[i * S * Hh + j]).toBeCloseTo(Math.exp(v - m) / z, 6));
    });
  });

  it("conv2d with dilation spaces its taps (padding = dilation keeps the size)", async () => {
    const [H, W, C, O, dil] = [9, 7, 4, 5, 2];
    const x = randn(H * W * C);
    const K = 64;
    const wt = randn(O * C * 9); // [O, C, 3, 3]
    const w = new Float32Array(O * K);
    for (let o = 0; o < O; o++) for (let c = 0; c < C; c++) for (let t = 0; t < 9; t++) w[o * K + t * C + c] = wt[(o * C + c) * 9 + t]!;
    const bias = randn(O);
    const r = await run((ctx) => ops.conv2d(ctx, upload(x, [H * W, C]), upload(w, [O, K]), upload(bias, [O]), { H, W, k: 3, stride: 1, pad: dil, dilation: dil }).y, H * W * O);
    for (let oy = 0; oy < H; oy++)
      for (let ox = 0; ox < W; ox++)
        for (let o = 0; o < O; o++) {
          let s = bias[o]!;
          for (let ky = 0; ky < 3; ky++)
            for (let kx = 0; kx < 3; kx++) {
              const iy = oy - dil + ky * dil;
              const ix = ox - dil + kx * dil;
              if (iy >= 0 && iy < H && ix >= 0 && ix < W) for (let c = 0; c < C; c++) s += x[(iy * W + ix) * C + c]! * wt[(o * C + c) * 9 + ky * 3 + kx]!;
            }
          expect(r[(oy * W + ox) * O + o]).toBeCloseTo(s, 5);
        }
  });

  it("maxPool2d with ceil_mode keeps the partial window at an odd edge, as torch does", async () => {
    expect(ops.convOut(7, 2, 2, 0, 1, true)).toBe(4);
    expect(ops.convOut(8, 2, 2, 0, 1, true)).toBe(4);
    expect(ops.convOut(7, 2, 2, 0)).toBe(3);
    const [H, W, C] = [3, 5, 4];
    const x = randn(H * W * C);
    const y = await run((ctx) => ops.maxPool2d(ctx, upload(x, [H * W, C]), { H, W, k: 2, stride: 2, pad: 0, ceil: true }).y, 2 * 3 * C);
    for (let oy = 0; oy < 2; oy++)
      for (let ox = 0; ox < 3; ox++)
        for (let c = 0; c < C; c++) {
          let m = -Infinity;
          for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) if (oy * 2 + dy < H && ox * 2 + dx < W) m = Math.max(m, x[((oy * 2 + dy) * W + ox * 2 + dx) * C + c]!);
          expect(y[(oy * 3 + ox) * C + c]).toBe(m);
        }
  });

  it("dwConv is PyTorch's depthwise convolution (groups = channels, padding k / 2), 3×3 and 7×7", async () => {
    for (const k of [3, 7]) {
      const [H, W, C] = [9, 8, 6];
      const kk = k * k;
      const half = (k - 1) / 2;
      const x = randn(H * W * C);
      const wt = randn(C * kk); // [C, 1, k, k]
      const w = new Float32Array(kk * C);
      for (let c = 0; c < C; c++) for (let t = 0; t < kk; t++) w[t * C + c] = wt[c * kk + t]!;
      const bias = randn(C);
      const y = await run((ctx) => ops.dwConv(ctx, upload(x, [H * W, C]), upload(w, [kk, C]), upload(bias, [C]), { H, W, k }), H * W * C);
      for (let py = 0; py < H; py++)
        for (let px = 0; px < W; px++)
          for (let c = 0; c < C; c++) {
            let s = bias[c]!;
            for (let ky = 0; ky < k; ky++)
              for (let kx = 0; kx < k; kx++) {
                const iy = py + ky - half;
                const ix = px + kx - half;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) s += x[(iy * W + ix) * C + c]! * wt[c * kk + ky * k + kx]!;
              }
            expect(y[(py * W + px) * C + c]).toBeCloseTo(s, 5);
          }
    }
  });

  it("resizeArgmax picks the class transformers.js's interpolate + argmax picks", async () => {
    const [H, W, L, Ho, Wo] = [4, 5, 7, 11, 9];
    const x = randn(H * W * L);
    const y = await run((ctx) => ops.resizeArgmax(ctx, upload(x, [H * W, L]), { H, W, Ho, Wo }), Ho * Wo);
    const got = new Uint32Array(y.buffer);
    // transformers.js's interpolate_data (planar, Float32 output), then the first maximum.
    const planes = Array.from({ length: L }, (_, l) => {
      const out = new Float32Array(Ho * Wo);
      for (let i = 0; i < Ho; i++)
        for (let j = 0; j < Wo; j++) {
          const fx = (j + 0.5) / (Wo / W) - 0.5;
          const fy = (i + 0.5) / (Ho / H) - 0.5;
          let x1 = Math.floor(fx);
          let y1 = Math.floor(fy);
          const x2 = Math.min(x1 + 1, W - 1);
          const y2 = Math.min(y1 + 1, H - 1);
          x1 = Math.max(x1, 0);
          y1 = Math.max(y1, 0);
          const s = fx - x1;
          const t = fy - y1;
          const v = (a: number, b: number) => x[(a * W + b) * L + l]!;
          out[i * Wo + j] = (1 - s) * (1 - t) * v(y1, x1) + s * (1 - t) * v(y1, x2) + (1 - s) * t * v(y2, x1) + s * t * v(y2, x2);
        }
      return out;
    });
    for (let k = 0; k < Ho * Wo; k++) {
      let best = 0;
      for (let l = 1; l < L; l++) if (planes[l]![k]! > planes[best]![k]!) best = l;
      expect(got[k]).toBe(best);
    }
  });

  it("swinAttention equals roll → window partition → cosine attention with bias and mask → window reverse → roll back", async () => {
    const [H, W, heads, D, win, row] = [8, 12, 2, 3, 4, 8];
    const q = randn(H * W * row);
    const k = randn(H * W * row);
    const v = randn(H * W * row);
    const N = win * win;
    const bias = randn(heads * N * N);
    const scale = Float32Array.from([3.5, 7]);
    for (const shift of [0, 2]) {
      const y = await run((ctx) => ops.swinAttention(ctx, upload(q, [H * W, row]), upload(k, [H * W, row]), upload(v, [H * W, row]), upload(bias, [heads * N * N]), upload(scale, [heads]), { H, W, heads, D, win, shift }), H * W * row);
      // Reference: explicit torch.roll(-shift), windows, transformers' region mask (-100, added twice).
      const src = (sy: number, sx: number) => ((sy + shift) % H) * W + ((sx + shift) % W);
      const region = (s: number, n: number) => (shift ? (s >= n - win ? 1 : 0) + (s >= n - shift ? 1 : 0) : 0);
      for (let wy = 0; wy < H / win; wy++)
        for (let wx = 0; wx < W / win; wx++)
          for (let h = 0; h < heads; h++) {
            const toks = Array.from({ length: N }, (_, i) => ({ sy: wy * win + Math.floor(i / win), sx: wx * win + (i % win) }));
            const norm = (t: Float32Array, r: number) => {
              const a = Array.from({ length: D }, (_, d) => t[r * row + h * D + d]!);
              const n = Math.sqrt(a.reduce((x, e) => x + e * e, 0));
              return a.map((e) => e / n);
            };
            toks.forEach((ti, i) => {
              const qi = norm(q, src(ti.sy, ti.sx));
              const sc = toks.map((tj, j) => {
                const kj = norm(k, src(tj.sy, tj.sx));
                const mask = region(ti.sy, H) * 3 + region(ti.sx, W) !== region(tj.sy, H) * 3 + region(tj.sx, W) ? -200 : 0;
                return qi.reduce((a, e, d) => a + e * kj[d]!, 0) * scale[h]! + bias[(h * N + i) * N + j]! + mask;
              });
              const m = Math.max(...sc);
              const e = sc.map((x) => Math.exp(x - m));
              const z = e.reduce((a, x) => a + x, 0);
              for (let d = 0; d < D; d++) {
                const want = toks.reduce((a, tj, j) => a + e[j]! * v[src(tj.sy, tj.sx) * row + h * D + d]!, 0) / z;
                expect(y[src(ti.sy, ti.sx) * row + h * D + d]).toBeCloseTo(want, 5);
              }
              for (let c = heads * D; c < row; c++) expect(y[src(ti.sy, ti.sx) * row + c]).toBe(0);
            });
          }
    }
  });

  it("pixelShuffle rearranges channels into r × r blocks as torch's PixelShuffle", async () => {
    const [H, W, C, r] = [2, 3, 2, 2];
    const x = randn(H * W * C * r * r);
    const y = await run((ctx) => ops.pixelShuffle(ctx, upload(x, [H * W, C * r * r]), { H, W, r }), H * r * W * r * C);
    for (let c = 0; c < C; c++)
      for (let i = 0; i < r; i++)
        for (let j = 0; j < r; j++)
          for (let h = 0; h < H; h++)
            for (let w = 0; w < W; w++) expect(y[((h * r + i) * W * r + w * r + j) * C + c]).toBe(x[(h * W + w) * C * r * r + c * r * r + i * r + j]);
  });

  it("layernorm with cols takes its statistics from the first cols only", async () => {
    const [R, D, n] = [3, 8, 5];
    const x = randn(R * D);
    const w = randn(D).map((v, i) => (i < n ? v : 0));
    const b = randn(D).map((v, i) => (i < n ? v : 0));
    const y = await run((ctx) => ops.layernorm(ctx, upload(x, [R, D]), upload(w, [D]), upload(b, [D]), 1e-5, n), R * D);
    for (let r = 0; r < R; r++) {
      const a = Array.from(x.subarray(r * D, r * D + n));
      const m = a.reduce((s, e) => s + e, 0) / n;
      const inv = 1 / Math.sqrt(a.reduce((s, e) => s + (e - m) ** 2, 0) / n + 1e-5);
      for (let i = 0; i < D; i++) expect(y[r * D + i]).toBeCloseTo(i < n ? (x[r * D + i]! - m) * inv * w[i]! + b[i]! : 0, 5);
    }
  });

  it("conv2d as a 1-D convolution (H = 1, kh = 1) with dilation and LeakyReLU on the input", async () => {
    const [T, C, O, k, dil, slope] = [13, 4, 6, 3, 2, 0.1];
    const x = randn(T * C);
    const wt = randn(O * C * k); // [O, C, k]
    const K = 32;
    const w = new Float32Array(O * K);
    for (let o = 0; o < O; o++) for (let c = 0; c < C; c++) for (let j = 0; j < k; j++) w[o * K + j * C + c] = wt[(o * C + c) * k + j]!;
    const bias = randn(O);
    const pad = (k * dil - dil) / 2;
    const y = await run((ctx) => ops.conv2d(ctx, upload(x, [T, C]), upload(w, [O, K]), upload(bias, [O]), { H: 1, W: T, k, kh: 1, stride: 1, pad, padH: 0, dilation: dil, leakyIn: slope }).y, T * O);
    for (let t = 0; t < T; t++)
      for (let o = 0; o < O; o++) {
        let s = bias[o]!;
        for (let j = 0; j < k; j++) {
          const ti = t - pad + j * dil;
          if (ti >= 0 && ti < T) for (let c = 0; c < C; c++) { const a = x[ti * C + c]!; s += (a > 0 ? a : a * slope) * wt[(o * C + c) * k + j]!; }
        }
        expect(y[t * O + o]).toBeCloseTo(s, 5);
      }
  });

  it("convTranspose1d matches PyTorch's ConvTranspose1d with overlapping taps (kernel 2·stride)", async () => {
    const [Tin, C, O, K, s, pad] = [5, 4, 3, 8, 4, 2];
    const x = randn(Tin * C);
    const wt = randn(C * O * K); // [C, O, K]
    const bias = randn(O);
    const w = new Float32Array(K * O * C);
    for (let c = 0; c < C; c++) for (let o = 0; o < O; o++) for (let k = 0; k < K; k++) w[(k * O + o) * C + c] = wt[(c * O + o) * K + k]!;
    const Tout = (Tin - 1) * s - 2 * pad + K;
    const y = await run((ctx) => ops.convTranspose1d(ctx, upload(x, [Tin, C]), upload(w, [K, O, C]), upload(bias, [O]), { s, pad }), Tout * O);
    // Scatter form: every input sample adds its kernel at t·s − pad.
    const ref = new Float64Array(Tout * O);
    for (let t = 0; t < Tout; t++) for (let o = 0; o < O; o++) ref[t * O + o] = bias[o]!;
    for (let ti = 0; ti < Tin; ti++)
      for (let k = 0; k < K; k++) {
        const t = ti * s - pad + k;
        if (t < 0 || t >= Tout) continue;
        for (let o = 0; o < O; o++) for (let c = 0; c < C; c++) ref[t * O + o] += x[ti * C + c]! * wt[(c * O + o) * K + k]!;
      }
    ref.forEach((v, i) => expect(y[i]).toBeCloseTo(v, 5));
  });

  it("relAttention equals transformers' VitsAttention (its pad-and-reshape relative position maps)", async () => {
    const [L, H, D, W] = [7, 2, 4, 2];
    const row = H * D;
    const q = randn(L * row), k = randn(L * row), v = randn(L * row);
    const ek = randn((2 * W + 1) * D), ev = randn((2 * W + 1) * D);
    const scale = D ** -0.5;
    const y = await run((ctx) => ops.relAttention(ctx, upload(q, [L, row]), upload(k, [L, row]), upload(v, [L, row]), upload(ek, [2 * W + 1, D]), upload(ev, [2 * W + 1, D]), { heads: H, window: W, scale }), L * row);
    // _get_relative_embeddings: the table padded to 2L − 1 rows, centred.
    const rel = (e: Float32Array) => {
      const padLen = Math.max(L - (W + 1), 0);
      const rows: number[][] = [];
      for (let i = 0; i < padLen; i++) rows.push(new Array(D).fill(0));
      for (let i = 0; i < 2 * W + 1; i++) rows.push(Array.from(e.subarray(i * D, (i + 1) * D)));
      for (let i = 0; i < padLen; i++) rows.push(new Array(D).fill(0));
      const start = Math.max(W + 1 - L, 0);
      return rows.slice(start, start + 2 * L - 1);
    };
    // _relative_position_to_absolute_position / _absolute_position_to_relative_position, literally.
    const relToAbs = (x: number[][]) => {
      const flat: number[] = [];
      for (const r of x) flat.push(...r, 0);
      for (let i = 0; i < L - 1; i++) flat.push(0);
      const out: number[][] = [];
      for (let i = 0; i < L; i++) out.push(flat.slice(i * (2 * L - 1), (i + 1) * (2 * L - 1)).slice(L - 1));
      return out;
    };
    const absToRel = (x: number[][]) => {
      const flat: number[] = new Array(L).fill(0);
      for (const r of x) flat.push(...r, ...new Array(L - 1).fill(0));
      const out: number[][] = [];
      for (let i = 0; i < L; i++) out.push(flat.slice(i * 2 * L, (i + 1) * 2 * L).slice(1));
      return out;
    };
    const rk = rel(ek), rv = rel(ev);
    for (let h = 0; h < H; h++) {
      const Q = Array.from({ length: L }, (_, i) => Array.from({ length: D }, (_, d) => q[i * row + h * D + d]! * scale));
      const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i]!, 0);
      const logits = Q.map((qi) => Array.from({ length: L }, (_, j) => dot(qi, Array.from({ length: D }, (_, d) => k[j * row + h * D + d]!))));
      const bias = relToAbs(Q.map((qi) => rk.map((r) => dot(qi, r))));
      const probs = logits.map((r, i) => {
        const s = r.map((x, j) => x + bias[i]![j]!);
        const m = Math.max(...s);
        const e = s.map((x) => Math.exp(x - m));
        const z = e.reduce((a, b) => a + b, 0);
        return e.map((x) => x / z);
      });
      const relW = absToRel(probs);
      for (let i = 0; i < L; i++)
        for (let d = 0; d < D; d++) {
          let want = probs[i]!.reduce((s, p, j) => s + p * v[j * row + h * D + d]!, 0);
          want += relW[i]!.reduce((s, p, r) => s + p * rv[r]![d]!, 0);
          expect(y[i * row + h * D + d]).toBeCloseTo(want, 5);
        }
    }
  });

  it("gatedTanh and cols: WaveNet's gate, channel copies, flips and adds", async () => {
    const [T, H] = [3, 4];
    const x = randn(T * 2 * H);
    const g = await run((ctx) => ops.gatedTanh(ctx, upload(x, [T, 2 * H])), T * H);
    for (let t = 0; t < T; t++) for (let c = 0; c < H; c++) expect(g[t * H + c]).toBeCloseTo(Math.tanh(x[t * 2 * H + c]!) / (1 + Math.exp(-x[t * 2 * H + H + c]!)), 6);
    const src = randn(T * 6);
    const base = randn(T * 5);
    const y = await run((ctx) => {
      const dst = upload(base, [T, 5]);
      ops.cols(ctx, dst, upload(src, [T, 6]), { n: 3, srcStart: 2, dstStart: 1, reverse: true, accumulate: true });
      return dst;
    }, T * 5);
    for (let t = 0; t < T; t++)
      for (let j = 0; j < 5; j++) expect(y[t * 5 + j]).toBeCloseTo(base[t * 5 + j]! + (j >= 1 && j < 4 ? src[t * 6 + 2 + (2 - (j - 1))]! : 0), 6);
  });

  it("relAttention with clamp adds SpeechT5's relative bias q_i · pe_k[clamp(i − j) + M] and no value term", async () => {
    const [L, H, D, M] = [6, 2, 4, 2];
    const row = H * D;
    const q = randn(L * row), k = randn(L * row), v = randn(L * row);
    const pk = randn(2 * M * D);
    const scale = D ** -0.5;
    const y = await run((ctx) => ops.relAttention(ctx, upload(q, [L, row]), upload(k, [L, row]), upload(v, [L, row]), upload(pk, [2 * M, D]), upload(pk, [2 * M, D]), { heads: H, window: M, scale, clamp: true }), L * row);
    for (let h = 0; h < H; h++)
      for (let i = 0; i < L; i++) {
        const s = Array.from({ length: L }, (_, j) => {
          const e = Math.min(Math.max(i - j, -M), M - 1) + M;
          let d0 = 0;
          for (let d = 0; d < D; d++) d0 += q[i * row + h * D + d]! * scale * (k[j * row + h * D + d]! + pk[e * D + d]!);
          return d0;
        });
        const m = Math.max(...s);
        const e = s.map((x) => Math.exp(x - m));
        const z = e.reduce((a, b) => a + b, 0);
        for (let d = 0; d < D; d++) expect(y[i * row + h * D + d]).toBeCloseTo(e.reduce((a, p, j) => a + p * v[j * row + h * D + d]!, 0) / z, 5);
      }
  });

  it("windowAttention equals window partition → softmax(q·scale·kᵀ)·v → window reverse, from a fused qkv", async () => {
    const [H, W, heads, D, win] = [4, 6, 2, 3, 2];
    const C = heads * D;
    const qkv = randn(H * W * 3 * C);
    const y = await run((ctx) => ops.windowAttention(ctx, upload(qkv, [H * W, 3 * C]), { H, W, heads, win }), H * W * C);
    const at = (t: number, part: number, h: number, d: number) => qkv[t * 3 * C + part * C + h * D + d]!;
    for (let t = 0; t < H * W; t++) {
      const [ty, tx] = [Math.floor(t / W), t % W];
      const mates = Array.from({ length: H * W }, (_, j) => j).filter((j) => Math.floor(Math.floor(j / W) / win) === Math.floor(ty / win) && Math.floor((j % W) / win) === Math.floor(tx / win));
      expect(mates.length).toBe(win * win);
      for (let h = 0; h < heads; h++) {
        const sc = mates.map((j) => Array.from({ length: D }, (_, d) => at(t, 0, h, d) * D ** -0.5 * at(j, 1, h, d)).reduce((a, b) => a + b, 0));
        const m = Math.max(...sc);
        const e = sc.map((x) => Math.exp(x - m));
        const z = e.reduce((a, b) => a + b, 0);
        for (let d = 0; d < D; d++) expect(y[t * C + h * D + d]).toBeCloseTo(mates.reduce((a, j, k) => a + e[k]! * at(j, 2, h, d), 0) / z, 5);
      }
    }
  });

  it("channelAttention equals DaViT's softmax((q·N^-½)ᵀ k) applied to vᵀ per channel group", async () => {
    const [N, C, G] = [7, 8, 2];
    const Ch = C / G;
    const qkv = randn(N * 3 * C);
    const y = await run((ctx) => ops.channelAttention(ctx, upload(qkv, [N, 3 * C]), { groups: G }), N * C);
    const at = (n: number, part: number, c: number) => qkv[n * 3 * C + part * C + c]!;
    for (let g = 0; g < G; g++)
      for (let i = 0; i < Ch; i++) {
        const sc = Array.from({ length: Ch }, (_, j) => Array.from({ length: N }, (_, n) => at(n, 0, g * Ch + i) * N ** -0.5 * at(n, 1, g * Ch + j)).reduce((a, b) => a + b, 0));
        const m = Math.max(...sc);
        const e = sc.map((x) => Math.exp(x - m));
        const z = e.reduce((a, b) => a + b, 0);
        for (let n = 0; n < N; n++) expect(y[n * C + g * Ch + i]).toBeCloseTo(e.reduce((a, p, j) => a + (p / z) * at(n, 2, g * Ch + j), 0), 5);
      }
  });
});
