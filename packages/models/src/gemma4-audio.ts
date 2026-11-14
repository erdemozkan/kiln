import { ActivationPool, Ctx, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError, configNumber } from "./config";
import type { Gemma4Clip } from "./gemma4-vision";

/**
 * Gemma 4's audio tower: log-mel frames halved twice in time by two strided convolutions, then a
 * conformer (per layer two half-weight feed-forward blocks around an attention over a short causal
 * window with relative positions and a causal depthwise convolution), projected to the text width.
 * A token is 40 ms of audio.
 */
export interface Gemma4AudioConfig {
  hidden: number;
  heads: number;
  layers: number;
  /** Mel bands of the input (128). */
  mels: number;
  /** Channels of the two subsampling convolutions (128, 32). */
  convChannels: [number, number];
  /** The model's `attention_context_left` (13): a query sees itself and the contextLeft − 2 rows before it, 12 rows in all. */
  contextLeft: number;
  /** Scores are squashed to ± this with tanh (50). */
  logitCap: number;
  /** Taps of a layer's depthwise convolution (5). */
  convKernel: number;
  /** Weight of a feed-forward block's output on the residual stream (0.5). */
  residualWeight: number;
  /** Width the tower's last layer projects to, before the projection to the text model. */
  outputDim: number;
  /** Width of the tokens made: the text model's. */
  projectionDim: number;
  eps: number;
  /** Most tokens one clip becomes (750: 30 s). */
  maxTokens: number;
}

/** The projections of a layer whose input and output the tower clips. */
export const GEMMA4_AUDIO_CLIPPED = ["ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v", "attn_out", "conv_pw1", "conv_pw2", "ffn_up_1", "ffn_down_1"] as const;

const num = (o: Record<string, unknown>, k: string, d: number): number => (o[k] === undefined || o[k] === null ? d : configNumber(k, o[k]));

function check(c: Gemma4AudioConfig): Gemma4AudioConfig {
  const D = c.hidden / c.heads;
  if (!Number.isInteger(D)) throw new UnsupportedModelError(`Gemma 4 audio: ${c.hidden} does not split into ${c.heads} heads`);
  if (c.hidden % 32 || c.outputDim % 32) throw new UnsupportedModelError("Gemma 4 audio: widths must be multiples of 32");
  if (c.contextLeft < 2 || c.contextLeft > 33) throw new UnsupportedModelError(`Gemma 4 audio: a window of ${c.contextLeft} rows is not supported`);
  if (c.mels % 4) throw new UnsupportedModelError(`Gemma 4 audio: ${c.mels} mel bands do not halve twice`);
  return c;
}

/** From a transformers checkpoint's config.json (`audio_config`) and its processor's settings. */
export function gemma4AudioConfig(config: Record<string, unknown>, processor?: Record<string, unknown>): Gemma4AudioConfig {
  const a = config.audio_config as Record<string, unknown> | undefined;
  if (!a) throw new UnsupportedModelError("this checkpoint has no audio_config");
  if (num(a, "attention_context_right", 0) !== 0) throw new UnsupportedModelError("Gemma 4 audio: attention to later rows is not supported");
  const text = (config.text_config ?? {}) as Record<string, unknown>;
  const ch = Array.isArray(a.subsampling_conv_channels) ? a.subsampling_conv_channels.map(Number) : [128, 32];
  const hidden = num(a, "hidden_size", 1024);
  const outputDim = num(a, "output_proj_dims", 1536);
  const extractor = (processor?.feature_extractor ?? {}) as Record<string, unknown>;
  return check({
    hidden,
    heads: num(a, "num_attention_heads", 8),
    layers: num(a, "num_hidden_layers", 12),
    mels: num(extractor, "feature_size", 128),
    convChannels: [ch[0] ?? 128, ch[1] ?? 32],
    contextLeft: num(a, "attention_context_left", 13),
    logitCap: num(a, "attention_logit_cap", 50),
    convKernel: num(a, "conv_kernel_size", 5),
    residualWeight: num(a, "residual_weight", 0.5),
    outputDim,
    projectionDim: num(text, "hidden_size", outputDim),
    eps: num(a, "rms_norm_eps", 1e-6),
    maxTokens: num(processor ?? {}, "audio_seq_length", 750),
  });
}

/** From an mmproj GGUF's metadata (`clip.audio.projector_type` "gemma4a") and the shapes of its convolutions. */
export function gemma4AudioConfigFromGGUF(meta: Map<string, unknown>, convChannels: [number, number], convKernel: number, outputDim: number): Gemma4AudioConfig {
  const n = (k: string): number => {
    const v = meta.get(k);
    if (typeof v !== "number") throw new UnsupportedModelError(`mmproj: missing ${k}`);
    return v;
  };
  return check({
    hidden: n("clip.audio.embedding_length"),
    heads: n("clip.audio.attention.head_count"),
    layers: n("clip.audio.block_count"),
    mels: n("clip.audio.num_mel_bins"),
    convChannels,
    contextLeft: 13,
    logitCap: 50,
    convKernel,
    residualWeight: 0.5,
    outputDim,
    projectionDim: n("clip.audio.projection_dim"),
    // The file's metadata says 1e-5, but the model (and llama.cpp's graph) uses 1e-6.
    eps: 1e-6,
    maxTokens: 750,
  });
}

/** Rows after one of the tower's strided convolutions (kernel 3, stride 2, padding 1) over `n` rows. */
export const gemma4AudioHalf = (n: number): number => (n <= 0 ? 0 : Math.floor((n - 1) / 2) + 1);

/** Tokens a clip of `frames` mel frames becomes, within the tower's limit. */
export function gemma4AudioTokens(frames: number, c: Gemma4AudioConfig): number {
  return Math.min(gemma4AudioHalf(gemma4AudioHalf(frames)), c.maxTokens);
}

/** The tower's two subsampling convolutions' weights, as the checkpoint stores them. */
export interface Gemma4AudioConvs {
  /** [c0, 1, 3, 3] and [c1, c0, 3, 3]: output channel, input channel, time, frequency. */
  w: [Float32Array, Float32Array];
  /** The LayerNorm weight over each convolution's channels (no bias). */
  norm: [Float32Array, Float32Array];
}

/**
 * One subsampling layer on the CPU: a 3 × 3 convolution with stride 2 and padding 1 over x
 * [T, F, cin] (time, frequency, channel), LayerNorm over the channels with a weight and no bias,
 * ReLU → [ceil(T / 2), ceil(F / 2), cout].
 */
function subsampleLayer(x: Float32Array, T: number, F: number, cin: number, w: Float32Array, norm: Float32Array, eps: number): { y: Float32Array; T: number; F: number } {
  const cout = norm.length;
  const [To, Fo] = [gemma4AudioHalf(T), gemma4AudioHalf(F)];
  const y = new Float32Array(To * Fo * cout);
  // Weights by (time tap, frequency tap, input channel) with the output channel last: the inner loop runs over outputs.
  const wt = new Float32Array(9 * cin * cout);
  for (let o = 0; o < cout; o++) for (let i = 0; i < cin; i++) for (let k = 0; k < 9; k++) wt[(k * cin + i) * cout + o] = w[(o * cin + i) * 9 + k]!;
  const acc = new Float64Array(cout);
  for (let t = 0; t < To; t++)
    for (let f = 0; f < Fo; f++) {
      acc.fill(0);
      for (let dt = 0; dt < 3; dt++) {
        const ti = 2 * t - 1 + dt;
        if (ti < 0 || ti >= T) continue;
        for (let df = 0; df < 3; df++) {
          const fi = 2 * f - 1 + df;
          if (fi < 0 || fi >= F) continue;
          const xb = (ti * F + fi) * cin;
          const wb = (dt * 3 + df) * cin;
          for (let i = 0; i < cin; i++) {
            const v = x[xb + i]!;
            if (v === 0) continue;
            const row = (wb + i) * cout;
            for (let o = 0; o < cout; o++) acc[o] += v * wt[row + o]!;
          }
        }
      }
      let mean = 0;
      for (let o = 0; o < cout; o++) mean += acc[o]!;
      mean /= cout;
      let variance = 0;
      for (let o = 0; o < cout; o++) variance += (acc[o]! - mean) ** 2;
      const r = 1 / Math.sqrt(variance / cout + eps);
      const yb = (t * Fo + f) * cout;
      for (let o = 0; o < cout; o++) y[yb + o] = Math.max(0, (acc[o]! - mean) * r * norm[o]!);
    }
  return { y, T: To, F: Fo };
}

/**
 * The tower's input rows from mel features [frames, mels]: the two subsampling layers, each row then
 * its frequencies' channels side by side → [tokens, (mels / 4) · c1].
 */
export function gemma4AudioSubsample(c: Gemma4AudioConfig, convs: Gemma4AudioConvs, features: Float32Array, frames: number): { rows: Float32Array; tokens: number } {
  if (features.length !== frames * c.mels) throw new Error(`audio features hold ${features.length} values, not ${frames} frames of ${c.mels}`);
  const a = subsampleLayer(features, frames, c.mels, 1, convs.w[0], convs.norm[0], c.eps);
  const b = subsampleLayer(a.y, a.T, a.F, c.convChannels[0], convs.w[1], convs.norm[1], c.eps);
  return { rows: b.y, tokens: b.T };
}

/** Sinusoidal embeddings [contextLeft − 1, hidden] of the distances a query sees, 0 … contextLeft − 2: sines of hidden / 2 timescales from 1 to 10000, then their cosines. */
export function gemma4AudioPositions(c: Gemma4AudioConfig): Float32Array {
  const half = c.hidden / 2;
  const step = Math.log(10000) / Math.max(half - 1, 1);
  const out = new Float32Array((c.contextLeft - 1) * c.hidden);
  for (let d = 0; d < c.contextLeft - 1; d++)
    for (let i = 0; i < half; i++) {
      const a = d * Math.exp(-i * step);
      out[d * c.hidden + i] = Math.sin(a);
      out[d * c.hidden + half + i] = Math.cos(a);
    }
  return out;
}

interface Plan {
  dispatches: readonly Dispatch[];
  rows: Buf;
  features: Tensor;
  ctx: Ctx;
  used: number;
}

/**
 * The tower on the GPU, after the subsampling convolutions (which run on the CPU). Weights use
 * llama.cpp's mmproj names (the loader renames a transformers checkpoint's): `a.input_projection`,
 * `a.blk.N.*` (`per_dim_scale` already through softplus), `a.pre_encode.out` and
 * `mm.a.input_projection`. In a layer's convolution module the file's `conv_norm` is the norm before
 * the module and `norm_conv` the one after the convolution (checked against the checkpoint's values). `clips` holds the limits of the clipped projections by `a.blk.N.<name>`.
 */
export class Gemma4AudioTower {
  readonly kind = "gemma4-audio";
  private readonly plans = new Map<number, Plan>();
  private clock = 0;
  private readonly step: Buf;
  private readonly zeros: Buf;
  private readonly ones: Tensor;
  private readonly positions: Tensor;
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: Gemma4AudioConfig,
    private readonly weights: Map<string, Tensor>,
    private readonly clips: ReadonlyMap<string, Gemma4Clip>,
    private readonly convs: Gemma4AudioConvs,
  ) {
    const inWidth = (cfg.mels / 4) * cfg.convChannels[1];
    const proj = weights.get("a.input_projection.weight");
    if (proj && proj.shape[1] !== inWidth) throw new Error(`Gemma 4 audio: the input projection takes ${proj.shape[1]} values, but the convolutions make ${inWidth}`);
    this.step = backend.alloc(64, "storage", "audio/step");
    this.zeros = backend.alloc((cfg.convKernel - 1) * cfg.hidden * 4, "storage", "audio/zeros");
    backend.write(this.zeros, 0, new Float32Array((cfg.convKernel - 1) * cfg.hidden));
    const width = Math.max(cfg.hidden, cfg.outputDim);
    const ones = backend.alloc(width * 4, "storage", "audio/ones");
    backend.write(ones, 0, new Float32Array(width).fill(1));
    this.ones = { shape: [cfg.outputDim], dtype: "f32", buf: ones };
    const pos = backend.alloc((cfg.contextLeft - 1) * cfg.hidden * 4, "storage", "audio/positions");
    backend.write(pos, 0, gemma4AudioPositions(cfg));
    this.positions = { shape: [cfg.contextLeft - 1, cfg.hidden], dtype: "f32", buf: pos };
    this.pool = new ActivationPool(backend, "audio");
  }

  /** Feature rows a slot holds (a clip may use fewer). */
  get tokens(): number {
    return this.cfg.maxTokens;
  }

  /** Tokens a clip of `frames` mel frames becomes. */
  tokensFor(frames: number): number {
    return gemma4AudioTokens(frames, this.cfg);
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`audio weight "${name}" was not loaded`);
    return t;
  }

  private build(T: number): Plan {
    const c = this.cfg;
    const inWidth = (c.mels / 4) * c.convChannels[1];
    const D = c.hidden / c.heads;
    const ctx = new Ctx(this.backend, this.step, "audio", this.pool);
    const rows = ctx.buffer(T * inWidth * 4, "audio/rows");
    const norm = (t: Tensor, name: string) => ops.rmsnorm(ctx, t, this.w(name), c.eps);
    const clipped = (t: Tensor, at: string): Tensor => {
      const lim = this.clips.get(at);
      const x = lim && (lim[0] !== -Infinity || lim[1] !== Infinity) ? ops.clamp(ctx, t, lim[0], lim[1]) : t;
      const y = ops.linear(ctx, x, this.w(`${at}.weight`));
      return lim && (lim[2] !== -Infinity || lim[3] !== Infinity) ? ops.clamp(ctx, y, lim[2], lim[3]) : y;
    };
    let x = ops.linear(ctx, { shape: [T, inWidth], dtype: "f32", buf: rows }, this.w("a.input_projection.weight"));
    // A feed-forward block, added to x at the layer's residual weight.
    const feedForward = (p: string, suffix: string) => {
      const up = clipped(norm(x, `${p}ffn_norm${suffix}.weight`), `${p}ffn_up${suffix}`);
      ops.act(ctx, up, "silu");
      const down = norm(clipped(up, `${p}ffn_down${suffix}`), `${p}ffn_post_norm${suffix}.weight`);
      ops.scaleAdd(ctx, x, down, c.residualWeight, true);
    };
    for (let l = 0; l < c.layers; l++) {
      const p = `a.blk.${l}.`;
      feedForward(p, "");
      const h = norm(x, p + "attn_pre_norm.weight");
      const [q, k, v] = [clipped(h, p + "attn_q"), clipped(h, p + "attn_k"), clipped(h, p + "attn_v")];
      const rel = ops.linear(ctx, this.positions, this.w(p + "attn_k_rel.weight"));
      // The model scales queries by 1 / (√D · ln 2) and a learned factor per element, keys by log2(1 + e).
      const scale = this.w(p + "per_dim_scale.weight");
      const scaled = ctx.alloc([D]);
      ops.scaleAdd(ctx, scaled, scale, D ** -0.5 / Math.LN2, false);
      const a = ops.localRelAttention(ctx, q, k, v, rel, scaled, { heads: c.heads, kscale: Math.log(1 + Math.E) / Math.LN2, cap: c.logitCap });
      ops.addInPlace(ctx, x, norm(clipped(a, p + "attn_out"), p + "attn_post_norm.weight"));
      // The convolution module: a gated projection, the depthwise convolution over time, a norm and SiLU, a projection back.
      const start = clipped(norm(x, p + "conv_norm.weight"), p + "conv_pw1");
      const gated = ctx.alloc([T, c.hidden]);
      const gate = ctx.alloc([T, c.hidden]);
      ops.cols(ctx, gated, start, { n: c.hidden });
      ops.cols(ctx, gate, start, { n: c.hidden, srcStart: c.hidden });
      ops.sigmoidMul(ctx, gated, gate);
      const conv = norm(ops.causalConvPlain(ctx, gated, this.w(p + "conv_dw.weight"), this.zeros), p + "norm_conv.weight");
      ops.act(ctx, conv, "silu");
      ops.addInPlace(ctx, x, clipped(conv, p + "conv_pw2"));
      feedForward(p, "_1");
      x = norm(x, p + "ln2.weight");
    }
    const out = ops.linear(ctx, x, this.w("a.pre_encode.out.weight"), { bias: this.w("a.pre_encode.out.bias") });
    const features = ops.linear(ctx, ops.rmsnorm(ctx, out, this.ones, c.eps), this.w("mm.a.input_projection.weight"));
    return { dispatches: ctx.dispatches, rows, features, ctx, used: 0 };
  }

  /**
   * Runs the tower on mel features [frames, mels] and copies its tokens' features into `dst` at byte
   * offset `dstOffset`; returns the token count (frames past the tower's limit are dropped). Plans
   * are kept for the three most recent lengths.
   */
  async encode(features: Float32Array, frames: number, dst: Buf, dstOffset: number): Promise<number> {
    const c = this.cfg;
    const sub = gemma4AudioSubsample(c, this.convs, features, frames);
    const T = Math.min(sub.tokens, c.maxTokens);
    if (T < 1) throw new Error("the audio is too short to make a token");
    let plan = this.plans.get(T);
    if (!plan) {
      if (this.plans.size >= 3) {
        const [oldest] = [...this.plans].sort((a, b) => a[1].used - b[1].used)[0]!;
        this.plans.get(oldest)!.ctx.destroy();
        this.plans.delete(oldest);
      }
      plan = this.build(T);
      this.plans.set(T, plan);
    }
    plan.used = ++this.clock;
    const inWidth = (c.mels / 4) * c.convChannels[1];
    this.backend.write(plan.rows, 0, sub.rows.subarray(0, T * inWidth));
    const d = plan.dispatches;
    const per = Math.max(1, Math.ceil(d.length / (c.layers + 2)));
    let pending: Promise<ArrayBuffer> | undefined;
    for (let i = 0; i < d.length; i += per) {
      this.backend.submit(d.slice(i, i + per));
      const done = this.backend.readback(this.step, 0, 4);
      if (pending) await pending;
      pending = done;
    }
    await pending;
    this.backend.copy(physical(plan.features.buf), 0, dst, dstOffset, T * c.projectionDim * 4);
    return T;
  }

  /** GPU bytes the tower's weights hold. */
  get weightBytes(): number {
    let n = 0;
    for (const t of this.weights.values()) n += t.buf.size + (t.scales?.size ?? 0) + (t.mins?.size ?? 0);
    return n;
  }

  destroy(): void {
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    for (const p of this.plans.values()) p.ctx.destroy();
    this.plans.clear();
    this.pool.destroy();
    for (const b of [this.step, this.zeros, this.ones.buf, this.positions.buf]) this.backend.free(b);
  }
}

/**
 * A transformers checkpoint's audio tensor name as the tower's (mmproj) name, or undefined for other
 * tensors. A clipped projection's limits come out as `<projection>.input_min` and so on.
 */
export function gemma4AudioName(name: string): string | undefined {
  if (/^(?:model\.)?embed_audio\.embedding_projection\.weight$/.test(name)) return "mm.a.input_projection.weight";
  const m = /^(?:model\.)?audio_tower\.(.+)$/.exec(name);
  if (!m) return undefined;
  const n = m[1]!;
  const top: Record<string, string> = {
    "subsample_conv_projection.layer0.conv.weight": "a.conv1d.0.weight",
    "subsample_conv_projection.layer0.norm.weight": "a.conv1d.0.norm.weight",
    "subsample_conv_projection.layer1.conv.weight": "a.conv1d.1.weight",
    "subsample_conv_projection.layer1.norm.weight": "a.conv1d.1.norm.weight",
    "subsample_conv_projection.input_proj_linear.weight": "a.input_projection.weight",
    "output_proj.weight": "a.pre_encode.out.weight",
    "output_proj.bias": "a.pre_encode.out.bias",
  };
  if (top[n]) return top[n];
  const blk = /^layers\.(\d+)\.(.+)$/.exec(n);
  if (!blk) return undefined;
  const p = `a.blk.${blk[1]}.`;
  const rest = blk[2]!;
  const plain: Record<string, string> = {
    "feed_forward1.pre_layer_norm.weight": "ffn_norm.weight",
    "feed_forward1.post_layer_norm.weight": "ffn_post_norm.weight",
    "feed_forward2.pre_layer_norm.weight": "ffn_norm_1.weight",
    "feed_forward2.post_layer_norm.weight": "ffn_post_norm_1.weight",
    "norm_pre_attn.weight": "attn_pre_norm.weight",
    "norm_post_attn.weight": "attn_post_norm.weight",
    "norm_out.weight": "ln2.weight",
    "self_attn.relative_k_proj.weight": "attn_k_rel.weight",
    "self_attn.per_dim_scale": "per_dim_scale.weight",
    "lconv1d.pre_layer_norm.weight": "conv_norm.weight",
    "lconv1d.conv_norm.weight": "norm_conv.weight",
    "lconv1d.depthwise_conv1d.weight": "conv_dw.weight",
  };
  if (plain[rest]) return p + plain[rest];
  const lin = /^(.+)\.(linear\.weight|input_min|input_max|output_min|output_max)$/.exec(rest);
  if (!lin) return undefined;
  const to: Record<string, string> = {
    "feed_forward1.ffw_layer_1": "ffn_up", "feed_forward1.ffw_layer_2": "ffn_down",
    "feed_forward2.ffw_layer_1": "ffn_up_1", "feed_forward2.ffw_layer_2": "ffn_down_1",
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.post": "attn_out",
    "lconv1d.linear_start": "conv_pw1", "lconv1d.linear_end": "conv_pw2",
  };
  const t = to[lin[1]!];
  return t ? `${p}${t}.${lin[2] === "linear.weight" ? "weight" : lin[2]}` : undefined;
}
