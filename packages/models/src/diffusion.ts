// Latent diffusion from diffusers checkpoints: the UNet that predicts the noise in a latent image
// (UNet2DConditionModel, Stable Diffusion 1.x and 2.x layouts and their distilled versions such as
// SDXS) and TAESD, the tiny decoder that turns latents into pixels (AutoencoderTiny).
import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** A UNet2DConditionModel's shape (its config.json). */
export interface UNetConfig {
  inChannels: number;
  outChannels: number;
  /** Channels per level, finest first. */
  channels: number[];
  /** Per level going down: whether it has cross-attention transformers (CrossAttnDownBlock2D). */
  downAttn: boolean[];
  /** Per level going up, coarsest first: whether it has cross-attention transformers (CrossAttnUpBlock2D). */
  upAttn: boolean[];
  layersPerBlock: number;
  /** Attention heads per level (diffusers' attention_head_dim, which older configs use as the head count). */
  heads: number[];
  /** Whether there is a middle block (UNetMidBlock2DCrossAttn). */
  mid: boolean;
  groups: number;
  eps: number;
  crossDim: number;
  /** Width of the sinusoidal timestep features (the first level's channels). */
  timeDim: number;
  flipSinToCos: boolean;
  freqShift: number;
  /** 1×1 convolutions (false) or linear layers (true) around each transformer. */
  linearProjection: boolean;
}

export function unetConfigFromDiffusers(config: Record<string, unknown>): UNetConfig {
  if (config._class_name !== "UNet2DConditionModel") throw new UnsupportedModelError(`"${String(config._class_name)}" is not a UNet2DConditionModel`);
  const channels = (config.block_out_channels as number[]).map(Number);
  const down = (config.down_block_types as string[]) ?? [];
  const up = (config.up_block_types as string[]) ?? [];
  const known = new Set(["DownBlock2D", "CrossAttnDownBlock2D", "UpBlock2D", "CrossAttnUpBlock2D"]);
  for (const b of [...down, ...up]) if (!known.has(b)) throw new UnsupportedModelError(`UNet block "${b}" is not supported`);
  const midType = config.mid_block_type === undefined ? "UNetMidBlock2DCrossAttn" : config.mid_block_type;
  if (midType !== null && midType !== "UNetMidBlock2DCrossAttn") throw new UnsupportedModelError(`UNet middle block "${String(midType)}" is not supported`);
  for (const [k, ok] of [
    ["addition_embed_type", config.addition_embed_type == null],
    ["class_embed_type", config.class_embed_type == null],
    ["time_cond_proj_dim", config.time_cond_proj_dim == null],
    ["encoder_hid_dim", config.encoder_hid_dim == null],
    ["dual_cross_attention", !config.dual_cross_attention],
    ["upcast/act", (config.act_fn ?? "silu") === "silu"],
    ["resnet_time_scale_shift", (config.resnet_time_scale_shift ?? "default") === "default"],
    ["time_embedding_type", (config.time_embedding_type ?? "positional") === "positional"],
  ] as const) {
    if (!ok) throw new UnsupportedModelError(`UNet option ${k} is not supported`);
  }
  const perLevel = (v: unknown, d: number): number[] => (Array.isArray(v) ? v.map(Number) : channels.map(() => Number(v ?? d)));
  const onlyCross = config.only_cross_attention;
  const attnDown = down.map((b) => b.startsWith("CrossAttn"));
  if (Array.isArray(onlyCross) ? onlyCross.some((o, i) => o && attnDown[i]) : onlyCross === true) throw new UnsupportedModelError("UNets with only cross-attention are not supported");
  for (const k of ["transformer_layers_per_block", "layers_per_block"]) if (Array.isArray(config[k])) throw new UnsupportedModelError(`a UNet ${k} per level is not supported`);
  if (Number(config.transformer_layers_per_block ?? 1) !== 1) throw new UnsupportedModelError("UNets with several transformer layers per block are not supported");
  if (config.num_attention_heads != null) throw new UnsupportedModelError("UNet num_attention_heads is not supported (use attention_head_dim)");
  return {
    inChannels: Number(config.in_channels ?? 4),
    outChannels: Number(config.out_channels ?? 4),
    channels,
    downAttn: attnDown,
    upAttn: up.map((b) => b.startsWith("CrossAttn")),
    layersPerBlock: Number(config.layers_per_block ?? 2),
    heads: perLevel(config.attention_head_dim, 8),
    mid: midType !== null,
    groups: Number(config.norm_num_groups ?? 32),
    eps: Number(config.norm_eps ?? 1e-5),
    crossDim: Number(config.cross_attention_dim ?? 768),
    timeDim: channels[0]!,
    flipSinToCos: config.flip_sin_to_cos !== false,
    freqShift: Number(config.freq_shift ?? 0),
    linearProjection: config.use_linear_projection === true,
  };
}

/** diffusers' Timesteps: sinusoidal features [dim] of a timestep (cos first with flipSinToCos). */
export function timestepFeatures(t: number, dim: number, flipSinToCos: boolean, freqShift: number): Float32Array {
  const half = dim / 2;
  const out = new Float32Array(dim);
  for (let i = 0; i < half; i++) {
    // torch computes the frequencies and products in float32.
    const f = Math.fround(Math.exp(Math.fround((-Math.log(10000) * i) / (half - freqShift))));
    const a = Math.fround(t * f);
    out[flipSinToCos ? half + i : i] = Math.sin(a);
    out[flipSinToCos ? i : half + i] = Math.cos(a);
  }
  return out;
}

/** y = W x + b for a row-major W [rows, cols]. */
export function matVec(w: Float32Array, b: Float32Array | undefined, x: Float32Array): Float32Array {
  const rows = w.length / x.length;
  const y = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = b ? b[r]! : 0;
    const o = r * x.length;
    for (let c = 0; c < x.length; c++) s += w[o + c]! * x[c]!;
    y[r] = s;
  }
  return y;
}

export const silu = (v: Float32Array): Float32Array => v.map((x) => x / (1 + Math.exp(-x)));

/** One ResNet block's place in the UNet: its weight prefix and the image side it runs at. */
export interface UNetResnet {
  prefix: string;
  inC: number;
  outC: number;
}

/** The ResNet blocks of a UNet in the order of its weights (each block's conv1 bias takes the time embedding). */
export function unetResnets(c: UNetConfig): UNetResnet[] {
  const out: UNetResnet[] = [];
  let ch = c.channels[0]!;
  c.channels.forEach((oc, i) => {
    for (let j = 0; j < c.layersPerBlock; j++) {
      out.push({ prefix: `down_blocks.${i}.resnets.${j}`, inC: j === 0 ? ch : oc, outC: oc });
    }
    ch = oc;
  });
  const last = c.channels[c.channels.length - 1]!;
  if (c.mid) for (const j of [0, 1]) out.push({ prefix: `mid_block.resnets.${j}`, inC: last, outC: last });
  // Skip widths, consumed from the end going up.
  const skips = [c.channels[0]!];
  c.channels.forEach((oc, i) => {
    for (let j = 0; j < c.layersPerBlock; j++) skips.push(oc);
    if (i < c.channels.length - 1) skips.push(oc);
  });
  const rev = [...c.channels].reverse();
  let prev = last;
  rev.forEach((oc, i) => {
    for (let j = 0; j <= c.layersPerBlock; j++) {
      const skip = skips.pop()!;
      out.push({ prefix: `up_blocks.${i}.resnets.${j}`, inC: (j === 0 ? prev : oc) + skip, outC: oc });
    }
    prev = oc;
  });
  return out;
}

/**
 * The UNet of a latent diffusion model, one plan per latent size: conv_in → down levels (ResNet
 * blocks, cross-attention transformers, strided downsampling; each output kept for its skip) →
 * middle block → up levels (skips concatenated along channels, ResNet blocks, transformers, nearest
 * upsampling and a convolution) → GroupNorm, SiLU, conv_out. Images are channels last [H·W, C].
 *
 * The timestep enters only through each ResNet block's time projection, a per-channel constant
 * added after its first convolution; the caller folds it into that convolution's bias
 * (`setTimeBiases`), so a timestep costs no kernel. Weights by diffusers' names: convolutions as
 * conv matrices ([O, K]), linear layers and 1×1 projections as [O, C], norms as they are; each
 * transformer's GEGLU input projection with its gate half first.
 */
export class UNetModel {
  private readonly pool: ActivationPool;
  private readonly step: Buf;
  private plan?: { dispatches: readonly Dispatch[]; latent: Buf; context: Buf; out: Tensor; ctx: Ctx; owned: Buf[]; H: number; W: number };

  constructor(
    readonly backend: Backend,
    readonly cfg: UNetConfig,
    private readonly weights: Map<string, Tensor>,
    /** Tokens of the text context. */
    readonly contextLength = 77,
  ) {
    this.pool = new ActivationPool(backend, "unet");
    this.step = backend.alloc(STEP_BYTES, "storage", "unet/step");
    // Every query sees every key: self-attention over the image, cross-attention over the text.
    backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: 0x7fffffff }));
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`UNet weight "${name}" was not loaded`);
    return t;
  }

  /** Writes each ResNet block's conv1 bias plus its time projection (`biases`, by block prefix). */
  setTimeBiases(biases: Map<string, Float32Array>): void {
    for (const [p, b] of biases) this.backend.write(this.w(`${p}.conv1.bias`).buf, 0, b);
  }

  private build(H0: number, W0: number) {
    const c = this.cfg;
    const be = this.backend;
    const owned: Buf[] = [];
    const ctx = new Ctx(be, this.step, `unet/${H0}x${W0}`, this.pool);
    const latent = ctx.buffer(H0 * W0 * c.inChannels * 4, "unet/latent");
    const context = ctx.buffer(this.contextLength * c.crossDim * 4, "unet/context");
    const text: Tensor = { shape: [this.contextLength, c.crossDim], dtype: "f32", buf: context };
    type Img = { y: Tensor; H: number; W: number };
    const own = (bytes: number, label: string) => {
      const b = be.alloc(bytes, "storage", label);
      owned.push(b);
      return b;
    };
    const conv = (m: Img, p: string, k: number, o: { stride?: number; residual?: Tensor } = {}): Img =>
      ops.conv2d(ctx, m.y, this.w(`${p}.weight`), this.w(`${p}.bias`), { H: m.H, W: m.W, k, stride: o.stride ?? 1, pad: (k - 1) / 2, residual: o.residual });
    const norm = (x: Tensor, p: string, eps: number, act: boolean) => {
      const y = ops.groupNorm(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), { n: 1, groups: c.groups, eps });
      if (act) ops.act(ctx, y, "silu");
      return y;
    };
    const resnet = (m: Img, p: string): Img => {
      const h = conv({ y: norm(m.y, `${p}.norm1`, c.eps, true), H: m.H, W: m.W }, `${p}.conv1`, 3);
      const skip = this.weights.has(`${p}.conv_shortcut.weight`) ? conv(m, `${p}.conv_shortcut`, 1).y : m.y;
      return conv({ y: norm(h.y, `${p}.norm2`, c.eps, true), H: m.H, W: m.W }, `${p}.conv2`, 3, { residual: skip });
    };
    // Cross-attention keys and values have the text's length; self-attention's the image's.
    const lin = (x: Tensor, p: string, residual?: Tensor) => ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
    const transformer = (m: Img, p: string, heads: number): Img => {
      const C = m.y.shape[1]!;
      const T = m.H * m.W;
      const D = C / heads;
      const x = lin(norm(m.y, `${p}.norm`, 1e-6, false), `${p}.proj_in`);
      const b = `${p}.transformer_blocks.0`;
      const kc = own(T * C * 4, "unet/k");
      const vc = own(T * C * 4, "unet/v");
      const n1 = ops.layernorm(ctx, x, this.w(`${b}.norm1.weight`), this.w(`${b}.norm1.bias`), 1e-5);
      ops.kvWrite(ctx, lin(n1, `${b}.attn1.to_k`), lin(n1, `${b}.attn1.to_v`), kc, vc, T, T, "f32");
      const self = { T, H: heads, Hkv: heads, D, maxSeq: T, kv: "f32" as const, scale: D ** -0.5, span: true };
      lin(ops.attention(ctx, lin(n1, `${b}.attn1.to_q`), kc, vc, self), `${b}.attn1.to_out.0`, x);
      const L = this.contextLength;
      const kx = own(L * C * 4, "unet/kx");
      const vx = own(L * C * 4, "unet/vx");
      const n2 = ops.layernorm(ctx, x, this.w(`${b}.norm2.weight`), this.w(`${b}.norm2.bias`), 1e-5);
      ops.kvWrite(ctx, lin(text, `${b}.attn2.to_k`), lin(text, `${b}.attn2.to_v`), kx, vx, L, L, "f32");
      const cross = { T, H: heads, Hkv: heads, D, maxSeq: L, kv: "f32" as const, scale: D ** -0.5 };
      lin(ops.attention(ctx, lin(n2, `${b}.attn2.to_q`), kx, vx, cross), `${b}.attn2.to_out.0`, x);
      const n3 = ops.layernorm(ctx, x, this.w(`${b}.norm3.weight`), this.w(`${b}.norm3.bias`), 1e-5);
      lin(ops.gatedAct(ctx, lin(n3, `${b}.ff.net.0.proj`), "gelu"), `${b}.ff.net.2`, x);
      return { y: lin(x, `${p}.proj_out`, m.y), H: m.H, W: m.W };
    };

    let h: Img = conv({ y: { shape: [H0 * W0, c.inChannels], dtype: "f32", buf: latent }, H: H0, W: W0 }, "conv_in", 3);
    const skips: Img[] = [h];
    c.channels.forEach((_, i) => {
      for (let j = 0; j < c.layersPerBlock; j++) {
        h = resnet(h, `down_blocks.${i}.resnets.${j}`);
        if (c.downAttn[i]) h = transformer(h, `down_blocks.${i}.attentions.${j}`, c.heads[i]!);
        skips.push(h);
      }
      if (i < c.channels.length - 1) {
        h = conv(h, `down_blocks.${i}.downsamplers.0.conv`, 3, { stride: 2 });
        skips.push(h);
      }
    });
    if (c.mid) {
      const last = c.channels.length - 1;
      h = resnet(h, "mid_block.resnets.0");
      h = transformer(h, "mid_block.attentions.0", c.heads[last]!);
      h = resnet(h, "mid_block.resnets.1");
    }
    const rev = [...c.heads].reverse();
    c.upAttn.forEach((attn, i) => {
      for (let j = 0; j <= c.layersPerBlock; j++) {
        const s = skips.pop()!;
        const a = h.y.shape[1]!;
        const b = s.y.shape[1]!;
        const cat = ctx.alloc([h.H * h.W, a + b]);
        ops.cols(ctx, cat, h.y, { n: a });
        ops.cols(ctx, cat, s.y, { n: b, dstStart: a });
        h = resnet({ y: cat, H: h.H, W: h.W }, `up_blocks.${i}.resnets.${j}`);
        if (attn) h = transformer(h, `up_blocks.${i}.attentions.${j}`, rev[i]!);
      }
      if (i < c.upAttn.length - 1) {
        const C = h.y.shape[1]!;
        const [Ho, Wo] = [h.H * 2, h.W * 2];
        const zero = own(Ho * Wo * C * 4, "unet/zero");
        be.write(zero, 0, new Float32Array(Ho * Wo * C));
        const up = ops.nearestAdd(ctx, h.y, { shape: [Ho * Wo, C], dtype: "f32", buf: zero }, { n: 1, H: h.H, W: h.W, Ho, Wo });
        h = conv({ y: up, H: Ho, W: Wo }, `up_blocks.${i}.upsamplers.0.conv`, 3);
      }
    });
    const out = conv({ y: norm(h.y, "conv_norm_out", c.eps, true), H: h.H, W: h.W }, "conv_out", 3).y;
    return { dispatches: ctx.dispatches, latent, context, out, ctx, owned, H: H0, W: W0 };
  }

  /** The predicted noise [H·W, outChannels] for a latent [H·W, inChannels] (channels last) and the text context [contextLength, crossDim]. */
  async predict(latent: Float32Array, H: number, W: number, context: Float32Array): Promise<Float32Array> {
    const c = this.cfg;
    if (latent.length !== H * W * c.inChannels) throw new Error(`UNet: a ${H} × ${W} latent has ${H * W * c.inChannels} values, got ${latent.length}`);
    if (context.length !== this.contextLength * c.crossDim) throw new Error(`UNet: the text context has ${this.contextLength} × ${c.crossDim} values, got ${context.length}`);
    const f = 2 ** (c.channels.length - 1);
    if (H % f || W % f) throw new Error(`UNet: latent sides must be multiples of ${f}, got ${H} × ${W}`);
    if (!this.plan || this.plan.H !== H || this.plan.W !== W) {
      if (this.plan) this.free(this.plan);
      this.plan = undefined;
      this.plan = this.build(H, W);
    }
    const p = this.plan;
    const be = this.backend;
    be.write(p.latent, 0, latent);
    be.write(p.context, 0, context);
    be.submit(p.dispatches);
    return new Float32Array(await be.readback(physical(p.out.buf), 0, H * W * c.outChannels * 4));
  }

  private free(p: NonNullable<UNetModel["plan"]>): void {
    p.ctx.destroy();
    for (const b of p.owned) this.backend.free(b);
  }

  destroy(): void {
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    if (this.plan) this.free(this.plan);
    this.pool.destroy();
    this.backend.free(this.step);
  }
}

/** TAESD's decoder (AutoencoderTiny's DecoderTiny), from its config.json. */
export interface TaesdConfig {
  latentChannels: number;
  /** Channels per stage. */
  channels: number[];
  /** Blocks per stage. */
  blocks: number[];
  outChannels: number;
  upsample: number;
}

export function taesdConfigFromDiffusers(config: Record<string, unknown>): TaesdConfig {
  if (config._class_name !== "AutoencoderTiny") throw new UnsupportedModelError(`"${String(config._class_name)}" is not an AutoencoderTiny`);
  if ((config.act_fn ?? "relu") !== "relu") throw new UnsupportedModelError(`TAESD activation "${String(config.act_fn)}" is not supported`);
  const channels = ((config.decoder_block_out_channels ?? config.block_out_channels) as number[]).map(Number);
  return {
    latentChannels: Number(config.latent_channels ?? 4),
    channels,
    blocks: ((config.num_decoder_blocks as number[] | undefined) ?? [3, 3, 3, 1]).map(Number),
    outChannels: Number(config.out_channels ?? 3),
    upsample: Number(config.upsampling_scaling_factor ?? 2),
  };
}

/** One layer of TAESD's decoder (`decoder.layers.<index>` in the weights). */
export type TaesdLayer = { kind: "conv"; index: number; bias: boolean; relu: boolean } | { kind: "block"; index: number; skip: boolean } | { kind: "up" };

/** The layers of DecoderTiny's Sequential, by index: conv + ReLU, then per stage its blocks, an upsampling and a conv. */
export function taesdLayers(c: TaesdConfig): TaesdLayer[] {
  const out: TaesdLayer[] = [{ kind: "conv", index: 0, bias: true, relu: true }];
  let i = 2;
  c.blocks.forEach((n, s) => {
    for (let b = 0; b < n; b++) out.push({ kind: "block", index: i++, skip: false });
    const last = s === c.blocks.length - 1;
    if (!last) {
      out.push({ kind: "up" });
      i++;
    }
    out.push({ kind: "conv", index: i++, bias: last, relu: false });
  });
  return out;
}

/**
 * TAESD's decoder: latents (already passed through tanh(x / 3) · 3) → 3×3 conv and ReLU → stages of
 * residual blocks (three 3×3 convolutions with ReLU between, the input added, ReLU), nearest
 * upsampling and a 3×3 convolution → pixels in about 0…1 (before DecoderTiny's · 2 − 1, which the
 * caller undoes). Convolution weights as conv matrices by diffusers' names; a conv without bias
 * gets a zero one.
 */
export class TaesdDecoder {
  private readonly pool: ActivationPool;
  private readonly step: Buf;
  private plan?: { dispatches: readonly Dispatch[]; latent: Buf; out: Tensor; ctx: Ctx; owned: Buf[]; H: number; W: number; Ho: number; Wo: number };

  constructor(
    readonly backend: Backend,
    readonly cfg: TaesdConfig,
    private readonly weights: Map<string, Tensor>,
  ) {
    this.pool = new ActivationPool(backend, "taesd");
    this.step = backend.alloc(STEP_BYTES, "storage", "taesd/step");
    backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`TAESD weight "${name}" was not loaded`);
    return t;
  }

  private build(H0: number, W0: number) {
    const c = this.cfg;
    const be = this.backend;
    const owned: Buf[] = [];
    const ctx = new Ctx(be, this.step, `taesd/${H0}x${W0}`, this.pool);
    const latent = ctx.buffer(H0 * W0 * c.latentChannels * 4, "taesd/latent");
    type Img = { y: Tensor; H: number; W: number };
    const conv = (m: Img, p: string, o: { relu?: boolean; residual?: Tensor } = {}): Img =>
      ops.conv2d(ctx, m.y, this.w(`${p}.weight`), this.w(`${p}.bias`), { H: m.H, W: m.W, k: 3, stride: 1, pad: 1, relu: o.relu, residual: o.residual });
    let h: Img = { y: { shape: [H0 * W0, c.latentChannels], dtype: "f32", buf: latent }, H: H0, W: W0 };
    for (const l of taesdLayers(c)) {
      if (l.kind === "conv") h = conv(h, `decoder.layers.${l.index}`, { relu: l.relu });
      else if (l.kind === "block") {
        const p = `decoder.layers.${l.index}.conv`;
        const a = conv(conv(h, `${p}.0`, { relu: true }), `${p}.2`, { relu: true });
        // fuse(conv(x) + skip(x)), the fuse a ReLU.
        h = conv(a, `${p}.4`, { relu: true, residual: h.y });
      } else {
        const C = h.y.shape[1]!;
        const f = c.upsample;
        const [Ho, Wo] = [h.H * f, h.W * f];
        const zero = be.alloc(Ho * Wo * C * 4, "storage", "taesd/zero");
        be.write(zero, 0, new Float32Array(Ho * Wo * C));
        owned.push(zero);
        h = { y: ops.nearestAdd(ctx, h.y, { shape: [Ho * Wo, C], dtype: "f32", buf: zero }, { n: 1, H: h.H, W: h.W, Ho, Wo }), H: Ho, W: Wo };
      }
    }
    return { dispatches: ctx.dispatches, latent, out: h.y, ctx, owned, H: H0, W: W0, Ho: h.H, Wo: h.W };
  }

  /** Pixels [Ho·Wo, outChannels] (about 0…1) for latents [H·W, latentChannels], both channels last. */
  async decode(latent: Float32Array, H: number, W: number): Promise<{ pixels: Float32Array; width: number; height: number }> {
    const c = this.cfg;
    if (latent.length !== H * W * c.latentChannels) throw new Error(`TAESD: a ${H} × ${W} latent has ${H * W * c.latentChannels} values, got ${latent.length}`);
    if (!this.plan || this.plan.H !== H || this.plan.W !== W) {
      if (this.plan) this.free(this.plan);
      this.plan = undefined;
      this.plan = this.build(H, W);
    }
    const p = this.plan;
    this.backend.write(p.latent, 0, latent);
    this.backend.submit(p.dispatches);
    const pixels = new Float32Array(await this.backend.readback(physical(p.out.buf), 0, p.Ho * p.Wo * c.outChannels * 4));
    return { pixels, width: p.Wo, height: p.Ho };
  }

  private free(p: NonNullable<TaesdDecoder["plan"]>): void {
    p.ctx.destroy();
    for (const b of p.owned) this.backend.free(b);
  }

  destroy(): void {
    for (const t of this.weights.values()) this.backend.free(t.buf);
    if (this.plan) this.free(this.plan);
    this.pool.destroy();
    this.backend.free(this.step);
  }
}
