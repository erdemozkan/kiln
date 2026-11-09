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
});
