import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";
import { bicubicResize } from "./vit";

/** Depth Anything (DepthAnythingConfig): a DINOv2 backbone and a DPT-style neck and head. */
export interface DepthAnythingConfig {
  hidden: number;
  layers: number;
  heads: number;
  intermediate: number;
  patchSize: number;
  /** The image size the position embeddings were trained at (square). */
  imageSize: number;
  eps: number;
  /** Layers (1-based) whose outputs feed the neck. */
  outIndices: number[];
  neck: number[];
  factors: number[];
  fusion: number;
  headHidden: number;
}

export function depthConfigFromHF(config: Record<string, unknown>): DepthAnythingConfig {
  if (config.model_type !== "depth_anything") throw new UnsupportedModelError(`model type "${String(config.model_type)}" is not Depth Anything`);
  const b = (config.backbone_config as Record<string, unknown> | undefined) ?? {};
  if ((b.model_type ?? "dinov2") !== "dinov2") throw new UnsupportedModelError(`Depth Anything backbone "${String(b.model_type)}" is not supported (DINOv2)`);
  if (b.use_swiglu_ffn === true) throw new UnsupportedModelError("DINOv2 with a SwiGLU MLP is not supported");
  if ((b.hidden_act ?? "gelu") !== "gelu") throw new UnsupportedModelError(`DINOv2 activation "${String(b.hidden_act)}" is not supported`);
  if ((config.depth_estimation_type ?? "relative") !== "relative") throw new UnsupportedModelError("only relative Depth Anything models are supported (not metric depth)");
  const n = (o: Record<string, unknown>, k: string, d: number) => Number(o[k] ?? d);
  const hidden = n(b, "hidden_size", 768);
  const layers = n(b, "num_hidden_layers", 12);
  return {
    hidden,
    layers,
    heads: n(b, "num_attention_heads", 12),
    intermediate: hidden * n(b, "mlp_ratio", 4),
    patchSize: n(b, "patch_size", 14),
    imageSize: n(b, "image_size", 518),
    eps: n(b, "layer_norm_eps", 1e-6),
    outIndices: ((b.out_indices as number[] | undefined) ?? [1, 2, 3, 4].map((i) => (i * layers) / 4)).map(Number),
    neck: ((config.neck_hidden_sizes as number[] | undefined) ?? [96, 192, 384, 768]).map(Number),
    factors: ((config.reassemble_factors as number[] | undefined) ?? [4, 2, 1, 0.5]).map(Number),
    fusion: n(config, "fusion_hidden_size", 256),
    headHidden: n(config, "head_hidden_size", 32),
  };
}

/** Kiln's name for a DINOv2 tensor loaded as a weight, or undefined for those loaded otherwise. */
export function depthName(name: string): string | undefined {
  const m = /^backbone\.encoder\.layer\.(\d+)\.(.+)\.(weight|bias)$/.exec(name);
  if (m) {
    const part: Record<string, string> = {
      "attention.attention.query": "q",
      "attention.attention.key": "k",
      "attention.attention.value": "v",
      "mlp.fc1": "up",
      norm1: "ln1",
      norm2: "ln2",
    };
    const p = part[m[2]!];
    return p ? `l${m[1]}.${p}.${m[3]}` : undefined;
  }
  const f = /^backbone\.layernorm\.(weight|bias)$/.exec(name);
  return f ? `lnF.${f[1]}` : undefined;
}

/** A convolution weight [O, C, k, k] as the conv kernel takes it: [O, K], taps major, channels padded to `cp`, K to a multiple of 32. */
export function convMatrix(w: Float32Array, shape: readonly number[], cp = shape[1]!): { w: Float32Array; K: number } {
  const [O, C, kh, kw] = shape as [number, number, number, number];
  const taps = kh * kw;
  const K = Math.ceil((taps * cp) / 32) * 32;
  const out = new Float32Array(O * K);
  for (let o = 0; o < O; o++) for (let c = 0; c < C; c++) for (let t = 0; t < taps; t++) out[o * K + t * cp + c] = w[(o * C + c) * taps + t]!;
  return { w: out, K };
}

/** A ConvTranspose2d weight [C, O, f, f] as the transposed-conv kernel takes it: [f·f, O, C]. */
export function convTransposeMatrix(w: Float32Array, shape: readonly number[]): Float32Array {
  const [C, O, f] = shape as [number, number, number];
  const out = new Float32Array(f * f * O * C);
  for (let c = 0; c < C; c++) for (let o = 0; o < O; o++) for (let t = 0; t < f * f; t++) out[(t * O + o) * C + c] = w[(c * O + o) * f * f + t]!;
  return out;
}

/**
 * DINOv2's patch position embeddings [n0 · n0, C] (raster order) fitted to a gh × gw grid, as
 * Dinov2Embeddings.interpolate_pos_encoding does it (torch bicubic, align_corners=False, no antialias).
 */
export function dinov2Positions(table: Float32Array, C: number, n0: number, gh: number, gw: number): Float32Array {
  const planar = new Float32Array(C * n0 * n0);
  for (let t = 0; t < n0 * n0; t++) for (let c = 0; c < C; c++) planar[c * n0 * n0 + t] = table[t * C + c]!;
  const r = bicubicResize(planar, C, n0, n0, gh, gw);
  const out = new Float32Array(gh * gw * C);
  for (let t = 0; t < gh * gw; t++) for (let c = 0; c < C; c++) out[t * C + c] = r[c * gh * gw + t]!;
  return out;
}

interface DepthPlan {
  dispatches: readonly Dispatch[];
  input: Buf;
  out: Tensor;
  ctx: Ctx;
  owned: Buf[];
  H: number;
  W: number;
}

/**
 * Depth Anything: DINOv2 (patch matmul plus a per-grid addend, pre-norm layers with the layer scales
 * folded into the output projections; patches first, CLS last, so the patch rows are a prefix) →
 * the chosen layers' outputs through the final LayerNorm → reassembled to four scales (1×1
 * projection, transposed or strided convolution) → 3×3 convolutions to the fusion width → fusion
 * from coarse to fine (pre-activation residual blocks, bilinear upsampling, 1×1 projection) → head
 * (3×3 convolution, upsampling to the input size, 3×3 + ReLU, 1×1 + ReLU). One plan per patch grid
 * (two kept); `addend(gh, gw)` gives the [tokens, hidden] embedding addend for a grid.
 */
export class DepthAnythingModel {
  private readonly plans = new Map<string, DepthPlan>();
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: DepthAnythingConfig,
    private readonly weights: Map<string, Tensor>,
    private readonly addend: (gh: number, gw: number) => Float32Array,
  ) {
    this.pool = new ActivationPool(backend, "depth");
  }

  /** Patch features padded to a multiple of 32 for the matmul. */
  get patchK(): number {
    return Math.ceil((3 * this.cfg.patchSize ** 2) / 32) * 32;
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`Depth Anything weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string, residual?: Tensor): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
  }

  private build(gh: number, gw: number): DepthPlan {
    const c = this.cfg;
    const be = this.backend;
    const owned: Buf[] = [];
    const own = (data: Float32Array | Uint32Array, label: string) => {
      const b = be.alloc(data.byteLength, "storage", label);
      be.write(b, 0, data);
      owned.push(b);
      return b;
    };
    const P = gh * gw;
    const T = P + 1;
    const step = own(new Uint32Array(STEP_BYTES / 4), "depth/step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: T }));
    const ctx = new Ctx(be, step, `depth/${gh}x${gw}`, this.pool);
    const input = ctx.buffer(T * this.patchK * 4, "depth/patches");
    const add: Tensor = { shape: [T, c.hidden], dtype: "f32", buf: own(this.addend(gh, gw), "depth/embed") };
    const kc = own(new Float32Array(T * c.hidden), "depth/k");
    const vc = own(new Float32Array(T * c.hidden), "depth/v");

    // DINOv2.
    const D = c.hidden / c.heads;
    const shape = { T, H: c.heads, Hkv: c.heads, D, maxSeq: T, kv: "f32" as const, scale: D ** -0.5, span: true };
    const x = ops.linear(ctx, { shape: [T, this.patchK], dtype: "f32", buf: input }, this.w("patch.weight"), {});
    ops.addInPlace(ctx, x, add);
    const feats: Tensor[] = [];
    for (let l = 0; l < c.layers; l++) {
      const p = `l${l}`;
      const h = ops.layernorm(ctx, x, this.w(`${p}.ln1.weight`), this.w(`${p}.ln1.bias`), c.eps);
      const q = this.lin(ctx, h, `${p}.q`);
      ops.kvWrite(ctx, this.lin(ctx, h, `${p}.k`), this.lin(ctx, h, `${p}.v`), kc, vc, T, T, "f32");
      this.lin(ctx, ops.attention(ctx, q, kc, vc, shape), `${p}.o`, x);
      const h2 = ops.layernorm(ctx, x, this.w(`${p}.ln2.weight`), this.w(`${p}.ln2.bias`), c.eps);
      const up = this.lin(ctx, h2, `${p}.up`);
      ops.act(ctx, up, "gelu");
      this.lin(ctx, up, `${p}.down`, x);
      if (c.outIndices.includes(l + 1)) {
        const f = ops.layernorm(ctx, x, this.w("lnF.weight"), this.w("lnF.bias"), c.eps);
        feats.push({ shape: [P, c.hidden], dtype: "f32", buf: f.buf });
      }
    }

    // Neck.
    type Map2 = { y: Tensor; H: number; W: number };
    const conv = (m: Map2, p: string, k: number, o: { stride?: number; relu?: boolean; reluIn?: boolean; residual?: Tensor } = {}) =>
      ops.conv2d(ctx, m.y, this.w(`${p}.weight`), this.w(`${p}.bias`), { H: m.H, W: m.W, k, stride: o.stride ?? 1, pad: (k - 1) / 2, relu: o.relu, reluIn: o.reluIn, residual: o.residual });
    const maps = feats.map((f, i) => {
      let m = conv({ y: f, H: gh, W: gw }, `neck.re${i}.proj`, 1);
      const factor = c.factors[i]!;
      if (factor > 1) m = ops.convTranspose(ctx, m.y, this.w(`neck.re${i}.resize.weight`), this.w(`neck.re${i}.resize.bias`), { H: m.H, W: m.W, f: factor });
      else if (factor < 1) m = conv(m, `neck.re${i}.resize`, 3, { stride: Math.round(1 / factor) });
      return conv(m, `neck.conv${i}`, 3);
    });
    // Pre-activation residual block: x + conv2(relu(conv1(relu(x)))).
    const residual = (m: Map2, p: string): Map2 => conv(conv(m, `${p}.c1`, 3, { reluIn: true, relu: true }), `${p}.c2`, 3, { residual: m.y });
    let h: Map2 | undefined;
    for (let j = 0; j < maps.length; j++) {
      const f = maps[maps.length - 1 - j]!;
      const p = `neck.fu${j}`;
      if (h) {
        const r = residual(f, `${p}.res1`);
        ops.addInPlace(ctx, r.y, h.y);
        h = r;
      } else h = f;
      h = residual(h, `${p}.res2`);
      const next = maps[maps.length - 2 - j];
      const [Ho, Wo] = next ? [next.H, next.W] : [h.H * 2, h.W * 2];
      h = conv({ y: ops.resizeBilinear(ctx, h.y, { H: h.H, W: h.W, Ho, Wo, alignCorners: true }), H: Ho, W: Wo }, `${p}.proj`, 1);
    }

    // Head.
    const h1 = conv(h!, "head.conv1", 3);
    const [H, W] = [gh * c.patchSize, gw * c.patchSize];
    const up = { y: ops.resizeBilinear(ctx, h1.y, { H: h1.H, W: h1.W, Ho: H, Wo: W, alignCorners: true }), H, W };
    const out = conv(conv(up, "head.conv2", 3, { relu: true }), "head.conv3", 1, { relu: true }).y;
    return { dispatches: ctx.dispatches, input, out, ctx, owned, H, W };
  }

  /** Relative depth [height', width'] (the patch grid times the patch size) for pixel values [3, height, width]. */
  async predict(pixels: Float32Array, width: number, height: number): Promise<{ depth: Float32Array; width: number; height: number }> {
    const c = this.cfg;
    const n = width * height;
    if (pixels.length !== 3 * n) throw new Error(`Depth Anything: expected 3 × ${height} × ${width} pixel values, got ${pixels.length}`);
    const [gh, gw] = [Math.floor(height / c.patchSize), Math.floor(width / c.patchSize)];
    if (gh < 2 || gw < 2) throw new Error(`Depth Anything: a ${width} × ${height} image is too small`);
    const key = `${gh}x${gw}`;
    let plan = this.plans.get(key);
    if (!plan) {
      if (this.plans.size >= 2) {
        const [k, old] = this.plans.entries().next().value!;
        this.free(old);
        this.plans.delete(k);
      }
      plan = this.build(gh, gw);
      this.plans.set(key, plan);
    }
    // Patch rows in (channel, row, column) order as the patch convolution weight flattens; CLS row last, zero.
    const Pz = c.patchSize;
    const K = this.patchK;
    const rows = new Float32Array((gh * gw + 1) * K);
    for (let py = 0; py < gh; py++)
      for (let px = 0; px < gw; px++) {
        const o = (py * gw + px) * K;
        for (let ch = 0; ch < 3; ch++)
          for (let y = 0; y < Pz; y++) for (let x = 0; x < Pz; x++) rows[o + (ch * Pz + y) * Pz + x] = pixels[ch * n + (py * Pz + y) * width + px * Pz + x]!;
      }
    const be = this.backend;
    be.write(plan.input, 0, rows);
    be.submit(plan.dispatches);
    const depth = new Float32Array(await be.readback(physical(plan.out.buf), 0, plan.H * plan.W * 4));
    return { depth, width: plan.W, height: plan.H };
  }

  private free(p: DepthPlan): void {
    p.ctx.destroy();
    for (const b of p.owned) this.backend.free(b);
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
