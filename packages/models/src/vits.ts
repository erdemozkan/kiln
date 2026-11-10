import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** VITS (VitsConfig: MMS-TTS and the original VITS checkpoints). */
export interface VitsConfig {
  hidden: number;
  layers: number;
  heads: number;
  window: number;
  ffnDim: number;
  ffnKernel: number;
  eps: number;
  flowSize: number;
  flows: number;
  wavenetLayers: number;
  wavenetKernel: number;
  wavenetDilation: number;
  dsLayers: number;
  dsKernel: number;
  durationFlows: number;
  bins: number;
  tailBound: number;
  upsampleInitial: number;
  upsampleRates: number[];
  upsampleKernels: number[];
  resblockKernels: number[];
  resblockDilations: number[][];
  leakySlope: number;
  noiseScale: number;
  noiseScaleDuration: number;
  speakingRate: number;
  samplingRate: number;
}

export function vitsConfigFromHF(c: Record<string, unknown>): VitsConfig {
  if (c.model_type !== "vits") throw new UnsupportedModelError(`model type "${String(c.model_type)}" is not VITS`);
  const n = (k: string, d: number) => Number(c[k] ?? d);
  if (n("num_speakers", 1) > 1 || n("speaker_embedding_size", 0) > 0) throw new UnsupportedModelError("multi-speaker VITS models are not supported yet");
  if (c.use_stochastic_duration_prediction === false) throw new UnsupportedModelError("VITS with the deterministic duration predictor is not supported yet");
  if ((c.hidden_act ?? "relu") !== "relu") throw new UnsupportedModelError(`VITS activation "${String(c.hidden_act)}" is not supported`);
  if (n("depth_separable_channels", 2) !== 2) throw new UnsupportedModelError("VITS duration flows with more than 2 channels are not supported");
  const arr = (k: string, d: number[]) => ((c[k] as number[] | undefined) ?? d).map(Number);
  return {
    hidden: n("hidden_size", 192),
    layers: n("num_hidden_layers", 6),
    heads: n("num_attention_heads", 2),
    window: n("window_size", 4),
    ffnDim: n("ffn_dim", 768),
    ffnKernel: n("ffn_kernel_size", 3),
    eps: n("layer_norm_eps", 1e-5),
    flowSize: n("flow_size", 192),
    flows: n("prior_encoder_num_flows", 4),
    wavenetLayers: n("prior_encoder_num_wavenet_layers", 4),
    wavenetKernel: n("wavenet_kernel_size", 5),
    wavenetDilation: n("wavenet_dilation_rate", 1),
    dsLayers: n("depth_separable_num_layers", 3),
    dsKernel: n("duration_predictor_kernel_size", 3),
    durationFlows: n("duration_predictor_num_flows", 4),
    bins: n("duration_predictor_flow_bins", 10),
    tailBound: n("duration_predictor_tail_bound", 5),
    upsampleInitial: n("upsample_initial_channel", 512),
    upsampleRates: arr("upsample_rates", [8, 8, 2, 2]),
    upsampleKernels: arr("upsample_kernel_sizes", [16, 16, 4, 4]),
    resblockKernels: arr("resblock_kernel_sizes", [3, 7, 11]),
    resblockDilations: ((c.resblock_dilation_sizes as number[][] | undefined) ?? [[1, 3, 5], [1, 3, 5], [1, 3, 5]]).map((d) => d.map(Number)),
    leakySlope: n("leaky_relu_slope", 0.1),
    noiseScale: n("noise_scale", 0.667),
    noiseScaleDuration: n("noise_scale_duration", 0.8),
    speakingRate: n("speaking_rate", 1),
    samplingRate: n("sampling_rate", 16000),
  };
}

/**
 * VitsTokenizer (transformers): characters kept when the vocabulary has them (others lowercased
 * first, and dropped if still unknown), the text stripped, and with add_blank token 0 between and
 * around the characters. Phonemizing and uroman romanization are not supported.
 */
export class VitsTokenizer {
  private readonly words: string[];

  constructor(
    private readonly vocab: Record<string, number>,
    private readonly opts: { addBlank?: boolean; normalize?: boolean; phonemize?: boolean; isUroman?: boolean } = {},
  ) {
    // Longest first would differ from transformers, which tries the vocabulary in its stored order.
    this.words = Object.keys(vocab);
    if (opts.phonemize) throw new UnsupportedModelError("this VITS tokenizer phonemizes its input (espeak), which Kiln does not do");
  }

  encode(text: string): number[] {
    let s = text;
    if (this.opts.normalize !== false) {
      let out = "";
      for (let i = 0; i < s.length; ) {
        const w = this.words.find((v) => s.startsWith(v, i));
        if (w) {
          out += w;
          i += w.length;
        } else {
          out += s[i]!.toLowerCase();
          i += 1;
        }
      }
      s = out;
    }
    // eslint-disable-next-line no-control-regex
    if (this.opts.isUroman && /[^\x00-\x7F]/.test(s)) throw new Error("this model needs uroman romanization of non-Latin text, which Kiln does not do; romanize the text first");
    if (this.opts.normalize !== false) s = Array.from(s).filter((ch) => ch in this.vocab).join("").trim();
    const ids = Array.from(s).map((ch) => this.vocab[ch]!).filter((i) => i !== undefined);
    if (this.opts.addBlank === false) return ids;
    const out = new Array<number>(ids.length * 2 + 1).fill(0);
    ids.forEach((id, i) => (out[2 * i + 1] = id));
    return out;
  }
}

/** A seeded standard normal stream (mulberry32 + Box-Muller). */
export function normalStream(seed: number): () => number {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare: number | undefined;
  return () => {
    if (spare !== undefined) {
      const s = spare;
      spare = undefined;
      return s;
    }
    let u = 0;
    while (u === 0) u = uniform();
    const v = uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

// ---- The stochastic duration predictor, on the CPU (text length × 192 channels) ----

/** erf, Abramowitz & Stegun 7.1.26 (absolute error under 1.5e-7), as Kiln's GELU kernel uses it. */
function erf(x: number): number {
  const s = Math.sign(x);
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));

/** Channels-last [L, C] features and the operations the duration predictor needs on them. */
type Feat = Float64Array;

function pointwise(x: Feat, L: number, w: Float32Array, b: Float32Array, cin: number, cout: number): Feat {
  const y = new Float64Array(L * cout);
  for (let t = 0; t < L; t++)
    for (let o = 0; o < cout; o++) {
      let s = b[o]!;
      for (let c = 0; c < cin; c++) s += w[o * cin + c]! * x[t * cin + c]!;
      y[t * cout + o] = s;
    }
  return y;
}

function layerNormRows(x: Feat, L: number, C: number, w: Float32Array, b: Float32Array, eps: number): Feat {
  const y = new Float64Array(L * C);
  for (let t = 0; t < L; t++) {
    let m = 0;
    for (let c = 0; c < C; c++) m += x[t * C + c]!;
    m /= C;
    let v = 0;
    for (let c = 0; c < C; c++) v += (x[t * C + c]! - m) ** 2;
    const inv = 1 / Math.sqrt(v / C + eps);
    for (let c = 0; c < C; c++) y[t * C + c] = (x[t * C + c]! - m) * inv * w[c]! + b[c]!;
  }
  return y;
}

/** The duration predictor's weights, by module path relative to `duration_predictor.`. */
export type VitsCpuWeights = (name: string) => Float32Array;

/** VitsDilatedDepthSeparableConv (with conditioning g added first), dropout off. */
function dds(cfg: VitsConfig, w: VitsCpuWeights, p: string, x0: Feat, L: number, g?: Feat): Feat {
  const C = cfg.hidden;
  const k = cfg.dsKernel;
  let x = Float64Array.from(x0);
  if (g) for (let i = 0; i < x.length; i++) x[i]! += g[i]!;
  for (let i = 0; i < cfg.dsLayers; i++) {
    const dil = k ** i;
    const pad = (k * dil - dil) / 2;
    const dw = w(`${p}.convs_dilated.${i}.weight`);
    const db = w(`${p}.convs_dilated.${i}.bias`);
    let h = new Float64Array(L * C);
    for (let t = 0; t < L; t++)
      for (let c = 0; c < C; c++) {
        let s = db[c]!;
        for (let j = 0; j < k; j++) {
          const ti = t - pad + j * dil;
          if (ti >= 0 && ti < L) s += dw[c * k + j]! * x[ti * C + c]!;
        }
        h[t * C + c] = s;
      }
    h = layerNormRows(h, L, C, w(`${p}.norms_1.${i}.weight`), w(`${p}.norms_1.${i}.bias`), 1e-5).map(gelu);
    h = pointwise(h, L, w(`${p}.convs_pointwise.${i}.weight`), w(`${p}.convs_pointwise.${i}.bias`), C, C);
    h = layerNormRows(h, L, C, w(`${p}.norms_2.${i}.weight`), w(`${p}.norms_2.${i}.bias`), 1e-5).map(gelu);
    for (let j = 0; j < x.length; j++) x[j]! += h[j]!;
  }
  return x;
}

/**
 * The inverse of the rational-quadratic spline transform (transformers' _unconstrained_rational_quadratic_spline
 * with reverse=True): identity outside ±tail, otherwise the bin found on the cumulative heights and
 * the quadratic in that bin solved for the input.
 */
export function splineInverse(x: number, uw: ArrayLike<number>, uh: ArrayLike<number>, ud: ArrayLike<number>, tail: number): number {
  if (x < -tail || x > tail) return x;
  const nb = uw.length;
  const minW = 1e-3;
  const minH = 1e-3;
  const minD = 1e-3;
  const softmax = (u: ArrayLike<number>) => {
    let m = -Infinity;
    for (let i = 0; i < u.length; i++) m = Math.max(m, u[i]!);
    const e = Array.from(u, (v) => Math.exp(v - m));
    const z = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / z);
  };
  const cum = (u: ArrayLike<number>, minBin: number) => {
    const p = softmax(u).map((v) => minBin + (1 - minBin * nb) * v);
    const c = [0];
    for (const v of p) c.push(c[c.length - 1]! + v);
    const out = c.map((v) => 2 * tail * v - tail);
    out[0] = -tail;
    out[nb] = tail;
    return out;
  };
  const cw = cum(uw, minW);
  const ch = cum(uh, minH);
  const widths = cw.slice(1).map((v, i) => v - cw[i]!);
  const heights = ch.slice(1).map((v, i) => v - ch[i]!);
  const constant = Math.log(Math.exp(1 - minD) - 1);
  const dPad = [constant, ...Array.from(ud), constant];
  const derivs = dPad.map((v) => minD + (v > 20 ? v : Math.log1p(Math.exp(v))));
  const locs = ch.slice();
  locs[nb] = locs[nb]! + 1e-6;
  let idx = -1;
  for (const l of locs) if (x >= l) idx++;
  idx = Math.min(Math.max(idx, 0), nb - 1);
  const delta = heights[idx]! / widths[idx]!;
  const d0 = derivs[idx]!;
  const d1 = derivs[idx + 1]!;
  const h = heights[idx]!;
  const i1 = d0 + d1 - 2 * delta;
  const i2 = x - ch[idx]!;
  const i3 = i2 * i1;
  const a = h * (delta - d0) + i3;
  const b = h * d0 - i3;
  const c = -delta * i2;
  const disc = b * b - 4 * a * c;
  const root = (2 * c) / (-b - Math.sqrt(Math.max(disc, 0)));
  return root * widths[idx]! + cw[idx]!;
}

/**
 * VitsStochasticDurationPredictor in reverse (inference): conditioning from the text encoder's hidden
 * state through conv_pre, the depth-separable stack and conv_proj; then latents [2, L] (noise ×
 * noise_scale) through the flows in reverse (the last conv flow skipped, as transformers does),
 * flipping the two channels before each. Returns the log durations [L].
 */
export function vitsLogDurations(cfg: VitsConfig, w: VitsCpuWeights, hidden: Float32Array, L: number, noise: () => number, noiseScale: number): Float64Array {
  const C = cfg.hidden;
  const h0 = Float64Array.from(hidden);
  let g = pointwise(h0, L, w("conv_pre.weight"), w("conv_pre.bias"), C, C);
  g = dds(cfg, w, "conv_dds", g, L);
  g = pointwise(g, L, w("conv_proj.weight"), w("conv_proj.bias"), C, C);
  // Latents channels-first: z[0] and z[1], each [L].
  let z0 = Float64Array.from({ length: L }, () => noise() * noiseScale);
  let z1 = Float64Array.from({ length: L }, () => noise() * noiseScale);
  const order = [...Array.from({ length: cfg.durationFlows }, (_, i) => cfg.durationFlows - i).slice(0, -1), 0];
  const nb = cfg.bins;
  for (const f of order) {
    [z0, z1] = [z1, z0];
    if (f === 0) {
      const tr = w("flows.0.translate");
      const ls = w("flows.0.log_scale");
      z0 = z0.map((v) => (v - tr[0]!) * Math.exp(-ls[0]!));
      z1 = z1.map((v) => (v - tr[1]!) * Math.exp(-ls[1]!));
      continue;
    }
    const p = `flows.${f}`;
    const pw = w(`${p}.conv_pre.weight`);
    const pb = w(`${p}.conv_pre.bias`);
    let h: Feat = new Float64Array(L * C);
    for (let t = 0; t < L; t++) for (let c = 0; c < C; c++) h[t * C + c] = pw[c]! * z0[t]! + pb[c]!;
    h = dds(cfg, w, `${p}.conv_dds`, h, L, g);
    const out = 3 * nb - 1;
    const proj = pointwise(h, L, w(`${p}.conv_proj.weight`), w(`${p}.conv_proj.bias`), C, out);
    const s = Math.sqrt(C);
    const z1n = new Float64Array(L);
    for (let t = 0; t < L; t++) {
      const r = proj.subarray(t * out, (t + 1) * out);
      z1n[t] = splineInverse(z1[t]!, Array.from(r.subarray(0, nb), (v) => v / s), Array.from(r.subarray(nb, 2 * nb), (v) => v / s), r.subarray(2 * nb), cfg.tailBound);
    }
    z1 = z1n;
  }
  return z0;
}

// ---- The text encoder, the flow and the vocoder, on the GPU ----

/**
 * VITS text-to-speech: the text encoder (pre-scaled embeddings, post-norm layers of windowed
 * relative attention and a kernel-3 convolutional FFN, projection to prior means and log-variances)
 * on the GPU; the stochastic duration predictor on the CPU; durations rounded up, the priors expanded
 * to frames and sampled; the residual coupling flow in reverse (WaveNet layers, channel flips) and
 * the HiFi-GAN vocoder on the GPU. One plan per text length and one per frame count (two kept each).
 */
export class VitsModel {
  private readonly textPlans = new Map<number, { dispatches: readonly Dispatch[]; ids: Buf; hidden: Tensor; stats: Tensor; ctx: Ctx }>();
  private readonly wavePlans = new Map<number, { dispatches: readonly Dispatch[]; input: Buf; out: Tensor; ctx: Ctx }>();
  private readonly pool: ActivationPool;
  private readonly step: Buf;

  constructor(
    readonly backend: Backend,
    readonly cfg: VitsConfig,
    private readonly weights: Map<string, Tensor>,
    private readonly cpu: VitsCpuWeights,
  ) {
    this.pool = new ActivationPool(backend, "vits");
    this.step = backend.alloc(STEP_BYTES, "storage", "vits/step");
    backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`VITS weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string, residual?: Tensor): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
  }

  /** A 1-D convolution of [T, C] (stride 1, "same" padding for its dilation). */
  private conv1d(ctx: Ctx, x: Tensor, p: string, k: number, o: { dil?: number; relu?: boolean; leakyIn?: number; residual?: Tensor } = {}): Tensor {
    const T = x.shape[0]!;
    const dil = o.dil ?? 1;
    return ops.conv2d(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), { H: 1, W: T, k, kh: 1, stride: 1, pad: (k * dil - dil) / 2, padH: 0, dilation: dil, relu: o.relu, leakyIn: o.leakyIn, residual: o.residual }).y;
  }

  private buildText(L: number) {
    const c = this.cfg;
    const ctx = new Ctx(this.backend, this.step, `vits/text L${L}`, this.pool);
    const ids = ctx.buffer(L * 4, "vits/ids");
    let x = ops.embed(ctx, this.w("te.emb.weight"), ids, L, Math.sqrt(c.hidden));
    const D = c.hidden / c.heads;
    for (let l = 0; l < c.layers; l++) {
      const p = `te.l${l}`;
      const a = ops.relAttention(ctx, this.lin(ctx, x, `${p}.q`), this.lin(ctx, x, `${p}.k`), this.lin(ctx, x, `${p}.v`), this.w(`${p}.ek`), this.w(`${p}.ev`), { heads: c.heads, window: c.window, scale: D ** -0.5 });
      this.lin(ctx, a, `${p}.o`, x);
      x = ops.layernorm(ctx, x, this.w(`${p}.ln1.weight`), this.w(`${p}.ln1.bias`), c.eps);
      const h = this.conv1d(ctx, x, `${p}.ff1`, c.ffnKernel, { relu: true });
      x = ops.layernorm(ctx, this.conv1d(ctx, h, `${p}.ff2`, c.ffnKernel, { residual: x }), this.w(`${p}.ln2.weight`), this.w(`${p}.ln2.bias`), c.eps);
    }
    const stats = this.lin(ctx, x, "te.proj");
    return { dispatches: ctx.dispatches, ids, hidden: x, stats, ctx };
  }

  private buildWave(T: number) {
    const c = this.cfg;
    const ctx = new Ctx(this.backend, this.step, `vits/wave T${T}`, this.pool);
    const F = c.flowSize;
    const half = F / 2;
    const H = c.hidden;
    const input = ctx.buffer(T * F * 4, "vits/latents");
    let x: Tensor = { shape: [T, F], dtype: "f32", buf: input };
    // The flow in reverse: flip, then each coupling layer subtracts its WaveNet's mean from the second half.
    for (let f = c.flows - 1; f >= 0; f--) {
      const p = `fl${f}`;
      const flipped = ctx.alloc([T, F]);
      ops.cols(ctx, flipped, x, { n: F, reverse: true });
      const first = ctx.alloc([T, half]);
      ops.cols(ctx, first, flipped, { n: half });
      const inp = this.lin(ctx, first, `${p}.pre`);
      const skip = ctx.alloc([T, H]);
      for (let j = 0; j < c.wavenetLayers; j++) {
        const acts = ops.gatedTanh(ctx, this.conv1d(ctx, inp, `${p}.in${j}`, c.wavenetKernel, { dil: c.wavenetDilation ** j }));
        const rs = this.lin(ctx, acts, `${p}.rs${j}`);
        if (j < c.wavenetLayers - 1) {
          ops.cols(ctx, inp, rs, { n: H, accumulate: true });
          ops.cols(ctx, skip, rs, { n: H, srcStart: H, accumulate: j > 0 });
        } else ops.cols(ctx, skip, rs, { n: H, accumulate: j > 0 });
      }
      // "post" is stored negated, so adding it subtracts the mean.
      const negMean = this.lin(ctx, skip, `${p}.post`);
      ops.cols(ctx, flipped, negMean, { n: half, dstStart: half, accumulate: true });
      x = flipped;
    }
    // HiFi-GAN.
    let h = this.conv1d(ctx, x, "dec.pre", 7);
    c.upsampleRates.forEach((s, i) => {
      const k = c.upsampleKernels[i]!;
      h = ops.convTranspose1d(ctx, h, this.w(`dec.up${i}.weight`), this.w(`dec.up${i}.bias`), { s, pad: (k - s) / 2, leakyIn: c.leakySlope });
      const nk = c.resblockKernels.length;
      const sum = ctx.alloc(h.shape);
      c.resblockKernels.forEach((kr, r) => {
        const p = `dec.rb${i * nk + r}`;
        let y = h;
        c.resblockDilations[r]!.forEach((d, j) => {
          const t = this.conv1d(ctx, y, `${p}.c1_${j}`, kr, { dil: d, leakyIn: c.leakySlope });
          y = this.conv1d(ctx, t, `${p}.c2_${j}`, kr, { leakyIn: c.leakySlope, residual: y });
        });
        if (r === 0) ops.scaleAdd(ctx, sum, y, 1, false);
        else ops.addInPlace(ctx, sum, y);
      });
      const avg = ctx.alloc(h.shape);
      ops.scaleAdd(ctx, avg, sum, 1 / nk, false);
      h = avg;
    });
    const out = this.conv1d(ctx, h, "dec.post", 7, { leakyIn: 0.01 });
    return { dispatches: ctx.dispatches, input, out, ctx };
  }

  private textPlan(L: number) {
    let p = this.textPlans.get(L);
    if (!p) {
      if (this.textPlans.size >= 2) {
        const [k, old] = this.textPlans.entries().next().value!;
        old.ctx.destroy();
        this.textPlans.delete(k);
      }
      p = this.buildText(L);
      this.textPlans.set(L, p);
    }
    return p;
  }

  private wavePlan(T: number) {
    let p = this.wavePlans.get(T);
    if (!p) {
      if (this.wavePlans.size >= 2) {
        const [k, old] = this.wavePlans.entries().next().value!;
        old.ctx.destroy();
        this.wavePlans.delete(k);
      }
      p = this.buildWave(T);
      this.wavePlans.set(T, p);
    }
    return p;
  }

  /**
   * The waveform for token ids (VitsTokenizer's): `noiseScale` and `noiseScaleDuration` scale the
   * sampled noise (0 gives the deterministic mean path), `speakingRate` divides the durations, and
   * `seed` fixes the noise.
   */
  async synthesize(ids: readonly number[], o: { noiseScale?: number; noiseScaleDuration?: number; speakingRate?: number; seed?: number } = {}): Promise<Float32Array> {
    const c = this.cfg;
    const L = ids.length;
    if (!L) throw new Error("VITS: nothing to say (the text has no characters in the model's vocabulary)");
    const be = this.backend;
    const tp = this.textPlan(L);
    be.write(tp.ids, 0, Uint32Array.from(ids));
    be.submit(tp.dispatches);
    const F = c.flowSize;
    const [hiddenBuf, statsBuf] = await Promise.all([be.readback(physical(tp.hidden.buf), 0, L * c.hidden * 4), be.readback(physical(tp.stats.buf), 0, L * 2 * F * 4)]);
    const hidden = new Float32Array(hiddenBuf);
    const stats = new Float32Array(statsBuf);
    const noise = normalStream(o.seed ?? Math.floor(Math.random() * 2 ** 32));
    const logDur = vitsLogDurations(c, this.cpu, hidden, L, noise, o.noiseScaleDuration ?? c.noiseScaleDuration);
    const lengthScale = 1 / (o.speakingRate ?? c.speakingRate);
    const dur = Array.from(logDur, (v) => Math.ceil(Math.fround(Math.fround(Math.exp(v)) * lengthScale)));
    const T = Math.max(dur.reduce((a, b) => a + b, 0), 1);
    // Prior means and log-variances expanded to frames, then sampled.
    const ns = o.noiseScale ?? c.noiseScale;
    const z = new Float32Array(T * F);
    let f = 0;
    for (let i = 0; i < L && f < T; i++)
      for (let r = 0; r < dur[i]! && f < T; r++, f++)
        for (let ch = 0; ch < F; ch++) {
          const m = stats[i * 2 * F + ch]!;
          const lv = stats[i * 2 * F + F + ch]!;
          z[f * F + ch] = m + noise() * Math.exp(lv) * ns;
        }
    const wp = this.wavePlan(T);
    be.write(wp.input, 0, z);
    be.submit(wp.dispatches);
    const out = new Float32Array(await be.readback(physical(wp.out.buf), 0, T * c.upsampleRates.reduce((a, b) => a * b, 1) * 4));
    for (let i = 0; i < out.length; i++) out[i] = Math.tanh(out[i]!);
    return out;
  }

  destroy(): void {
    for (const t of this.weights.values()) this.backend.free(t.buf);
    for (const p of this.textPlans.values()) p.ctx.destroy();
    for (const p of this.wavePlans.values()) p.ctx.destroy();
    this.pool.destroy();
    this.backend.free(this.step);
  }
}
