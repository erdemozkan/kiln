import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** Swin2SR (Swin2SRConfig) with the pixel-shuffle upsampler (the classical super-resolution models). */
export interface Swin2srConfig {
  dim: number;
  depths: number[];
  heads: number[];
  window: number;
  mlpRatio: number;
  upscale: number;
  eps: number;
  /** The patch grid the layers were built for; shifted windows only when it is larger than a window. */
  gridSize: number;
  imgRange: number;
}

export function swin2srConfigFromHF(config: Record<string, unknown>): Swin2srConfig {
  if (config.model_type !== "swin2sr") throw new UnsupportedModelError(`model type "${String(config.model_type)}" is not Swin2SR`);
  const n = (k: string, d: number) => Number(config[k] ?? d);
  const upsampler = String(config.upsampler ?? "pixelshuffle");
  if (upsampler !== "pixelshuffle") throw new UnsupportedModelError(`Swin2SR upsampler "${upsampler}" is not supported (pixelshuffle)`);
  if ((config.resi_connection ?? "1conv") !== "1conv") throw new UnsupportedModelError("Swin2SR with resi_connection 3conv is not supported");
  if (config.use_absolute_embeddings === true) throw new UnsupportedModelError("Swin2SR with absolute position embeddings is not supported");
  if (n("num_channels", 3) !== 3 || n("num_channels_out", n("num_channels", 3)) !== 3) throw new UnsupportedModelError("only RGB Swin2SR models are supported");
  if (n("patch_size", 1) !== 1) throw new UnsupportedModelError("only Swin2SR with patch_size 1 is supported");
  if ((config.hidden_act ?? "gelu") !== "gelu") throw new UnsupportedModelError(`Swin2SR activation "${String(config.hidden_act)}" is not supported`);
  const upscale = n("upscale", 2);
  if (upscale & (upscale - 1)) throw new UnsupportedModelError(`Swin2SR upscale ${upscale} is not supported (powers of 2)`);
  return {
    dim: n("embed_dim", 180),
    depths: ((config.depths as number[] | undefined) ?? [6, 6, 6, 6, 6, 6]).map(Number),
    heads: ((config.num_heads as number[] | undefined) ?? [6, 6, 6, 6, 6, 6]).map(Number),
    window: n("window_size", 8),
    mlpRatio: n("mlp_ratio", 2),
    upscale,
    eps: n("layer_norm_eps", 1e-5),
    gridSize: n("image_size", 64),
    imgRange: n("img_range", 1),
  };
}

/** The per-channel mean Swin2SR subtracts from RGB inputs (and adds back). */
export const SWIN2SR_MEAN = [0.4488, 0.4371, 0.404];

/**
 * A Swin V2 layer's attention bias [heads, win², win²]: the continuous position bias MLP
 * (Linear(2, 512), ReLU, Linear(512, heads)) on the log-spaced relative coordinate table, picked by
 * relative position index, then 16 · sigmoid, as transformers' Swin2SRSelfAttention computes it.
 */
export function swin2srBias(w1: Float32Array, b1: Float32Array, w2: Float32Array, heads: number, win: number): Float32Array {
  const f = Math.fround;
  const span = 2 * win - 1;
  const hidden = b1.length;
  const coord = (c: number) => {
    const x = f(f(c / (win - 1)) * 8);
    return f(f(Math.sign(x) * f(Math.log2(Math.abs(x) + 1))) / f(Math.log2(8)));
  };
  // The table: one row per relative offset (dh, dw), heads values each.
  const table = new Float32Array(span * span * heads);
  const h1 = new Float32Array(hidden);
  for (let a = 0; a < span; a++)
    for (let b = 0; b < span; b++) {
      const [ch, cw] = [coord(a - (win - 1)), coord(b - (win - 1))];
      for (let j = 0; j < hidden; j++) h1[j] = Math.max(0, w1[j * 2]! * ch + w1[j * 2 + 1]! * cw + b1[j]!);
      for (let h = 0; h < heads; h++) {
        let s = 0;
        for (let j = 0; j < hidden; j++) s += w2[h * hidden + j]! * h1[j]!;
        table[(a * span + b) * heads + h] = s;
      }
    }
  const N = win * win;
  const out = new Float32Array(heads * N * N);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      const idx = (Math.floor(i / win) - Math.floor(j / win) + win - 1) * span + ((i % win) - (j % win) + win - 1);
      for (let h = 0; h < heads; h++) out[(h * N + i) * N + j] = 16 / (1 + Math.exp(-table[idx * heads + h]!));
    }
  return out;
}

/**
 * Swin2SR on the GPU, with every row padded from the model's width (180) to a multiple of 32 so the
 * matmul and conv kernels take it; padded weight rows and columns are zero, LayerNorms take their
 * statistics over the real width, and the attention kernel zeroes the padding it writes. Shallow
 * conv → patch 1×1 conv + LayerNorm → residual groups (Swin V2 post-norm layers with shifted-window
 * cosine attention, then 3×3 conv and 1×1 conv plus the group's input) → LayerNorm → 3×3 conv plus
 * the shallow features → pixel-shuffle upsampler. One plan per input size.
 */
export class Swin2srModel {
  private readonly plans = new Map<string, { dispatches: readonly Dispatch[]; input: Buf; out: Tensor; ctx: Ctx; step: Buf }>();
  private readonly pool: ActivationPool;
  /** The padded width. */
  readonly row: number;

  constructor(
    readonly backend: Backend,
    readonly cfg: Swin2srConfig,
    private readonly weights: Map<string, Tensor>,
  ) {
    this.pool = new ActivationPool(backend, "swin2sr");
    this.row = Math.ceil(cfg.dim / 32) * 32;
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`Swin2SR weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`) });
  }

  private build(H: number, W: number) {
    const c = this.cfg;
    const be = this.backend;
    const step = be.alloc(STEP_BYTES, "storage", "swin2sr/step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
    const ctx = new Ctx(be, step, `swin2sr/${H}x${W}`, this.pool);
    const input = ctx.buffer(H * W * 4 * 4, "swin2sr/pixels");
    const conv = (x: Tensor, p: string, k: number, o: { h?: number; w?: number; residual?: Tensor; leaky?: boolean } = {}) =>
      ops.conv2d(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), { H: o.h ?? H, W: o.w ?? W, k, stride: 1, pad: (k - 1) / 2, residual: o.residual, leaky: o.leaky }).y;
    const ln = (x: Tensor, p: string) => ops.layernorm(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), c.eps, c.dim);

    const shallow = conv({ shape: [H * W, 4], dtype: "f32", buf: input }, "first", 3);
    const x = ln(conv(shallow, "patch", 1), "patchLn");
    const shiftable = c.gridSize > c.window;
    c.depths.forEach((depth, s) => {
      const res = ctx.alloc(x.shape);
      ops.scaleAdd(ctx, res, x, 1, false);
      const heads = c.heads[s]!;
      for (let l = 0; l < depth; l++) {
        const p = `s${s}.${l}`;
        const a = ops.swinAttention(ctx, this.lin(ctx, x, `${p}.q`), this.lin(ctx, x, `${p}.k`), this.lin(ctx, x, `${p}.v`), this.w(`${p}.bias`), this.w(`${p}.scale`), {
          H,
          W,
          heads,
          D: c.dim / heads,
          win: c.window,
          shift: l % 2 && shiftable ? c.window / 2 : 0,
        });
        ops.addInPlace(ctx, x, ln(this.lin(ctx, a, `${p}.o`), `${p}.ln1`));
        const up = this.lin(ctx, x, `${p}.up`);
        ops.act(ctx, up, "gelu");
        ops.addInPlace(ctx, x, ln(this.lin(ctx, up, `${p}.down`), `${p}.ln2`));
      }
      const g = conv(conv(x, `s${s}.conv`, 3), `s${s}.proj`, 1, { residual: res });
      ops.scaleAdd(ctx, x, g, 1, false);
    });
    const body = conv(ln(x, "lnF"), "afterBody", 3, { residual: shallow });
    let u = conv(body, "beforeUp", 3, { leaky: true });
    let [h, w] = [H, W];
    for (let i = 0; 2 ** i < c.upscale; i++) {
      u = ops.pixelShuffle(ctx, conv(u, `up${i}`, 3, { h, w }), { H: h, W: w, r: 2 });
      [h, w] = [h * 2, w * 2];
    }
    const out = conv(u, "final", 3, { h, w });
    return { dispatches: ctx.dispatches, input, out, ctx, step };
  }

  /**
   * The upscaled image [3, height · upscale, width · upscale] (before the mean is added back and
   * values clamped) for normalized pixel values [3, height, width] (mean subtracted; sides
   * multiples of the window).
   */
  async upscale(pixels: Float32Array, width: number, height: number): Promise<Float32Array> {
    const n = width * height;
    if (pixels.length !== 3 * n) throw new Error(`Swin2SR: expected 3 × ${height} × ${width} pixel values, got ${pixels.length}`);
    const key = `${height}x${width}`;
    let plan = this.plans.get(key);
    if (!plan) {
      for (const p of this.plans.values()) this.free(p);
      this.plans.clear();
      plan = this.build(height, width);
      this.plans.set(key, plan);
    }
    const nhwc = new Float32Array(n * 4);
    for (let ch = 0; ch < 3; ch++) for (let i = 0; i < n; i++) nhwc[i * 4 + ch] = pixels[ch * n + i]!;
    const be = this.backend;
    be.write(plan.input, 0, nhwc);
    be.submit(plan.dispatches);
    const m = n * this.cfg.upscale ** 2;
    const hwc = new Float32Array(await be.readback(physical(plan.out.buf), 0, m * 3 * 4));
    const out = new Float32Array(3 * m);
    for (let i = 0; i < m; i++) for (let ch = 0; ch < 3; ch++) out[ch * m + i] = hwc[i * 3 + ch]!;
    return out;
  }

  private free(p: { ctx: Ctx; step: Buf }): void {
    p.ctx.destroy();
    this.backend.free(p.step);
  }

  destroy(): void {
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    for (const p of this.plans.values()) this.free(p);
    this.pool.destroy();
  }
}
