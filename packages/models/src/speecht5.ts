import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { UnsupportedModelError } from "./config";
import { conv1d, hifiGan, type HifiGanConfig } from "./hifigan";
import { Seq2SeqModel, type Seq2SeqConfig } from "./seq2seq";
import { normalStream } from "./vits";

/** SpeechT5 text-to-speech (SpeechT5ForTextToSpeech) and its HiFi-GAN vocoder. */
export interface SpeechT5Config {
  seq: Seq2SeqConfig;
  mel: number;
  reduction: number;
  prenetLayers: number;
  prenetDropout: number;
  postnetLayers: number;
  postnetKernel: number;
  maxSpeechPositions: number;
}

const need = (c: Record<string, unknown>, k: string, d: number): number => Number(c[k] ?? d);

export function speecht5ConfigFromHF(c: Record<string, unknown>): SpeechT5Config {
  if (c.model_type !== "speecht5") throw new UnsupportedModelError(`model type "${String(c.model_type)}" is not SpeechT5`);
  if (!((c.architectures as string[] | undefined) ?? []).includes("SpeechT5ForTextToSpeech")) throw new UnsupportedModelError("only SpeechT5ForTextToSpeech checkpoints are supported");
  if ((c.hidden_act ?? "gelu") !== "gelu") throw new UnsupportedModelError(`SpeechT5 activation "${String(c.hidden_act)}" is not supported`);
  const d = need(c, "hidden_size", 768);
  return {
    seq: {
      family: "speecht5",
      modelType: "speecht5",
      dModel: d,
      encLayers: need(c, "encoder_layers", 12),
      decLayers: need(c, "decoder_layers", 6),
      encHeads: need(c, "encoder_attention_heads", 12),
      decHeads: need(c, "decoder_attention_heads", 12),
      encFfn: need(c, "encoder_ffn_dim", 3072),
      decFfn: need(c, "decoder_ffn_dim", 3072),
      vocab: need(c, "vocab_size", 81),
      maxPositions: need(c, "max_text_positions", 600),
      preNorm: false,
      layernormEmbedding: true,
      positions: "learned",
      posOffset: 0,
      embedScale: 1,
      act: "gelu",
      pad: need(c, "pad_token_id", 1),
      eos: need(c, "eos_token_id", 2),
      decoderStart: 0,
      raw: c,
      relativeMax: need(c, "encoder_max_relative_position", 160),
    },
    mel: need(c, "num_mel_bins", 80),
    reduction: need(c, "reduction_factor", 2),
    prenetLayers: need(c, "speech_decoder_prenet_layers", 2),
    prenetDropout: need(c, "speech_decoder_prenet_dropout", 0.5),
    postnetLayers: need(c, "speech_decoder_postnet_layers", 5),
    postnetKernel: need(c, "speech_decoder_postnet_kernel", 5),
    maxSpeechPositions: need(c, "max_speech_positions", 1876),
  };
}

/** Kiln's seq2seq name for a SpeechT5 encoder or decoder tensor, or undefined for the rest (prenets, postnet). */
export function speecht5Name(name: string): string | undefined {
  if (name === "speecht5.encoder.prenet.embed_tokens.weight") return "shared.weight";
  if (name === "speecht5.encoder.wrapped_encoder.embed_positions.pe_k.weight") return "enc.relpos";
  let m = /^speecht5\.encoder\.wrapped_encoder\.layer_norm\.(weight|bias)$/.exec(name);
  if (m) return `enc.lnE.${m[1]}`;
  m = /^speecht5\.(encoder|decoder)\.wrapped_(?:encoder|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$/.exec(name);
  if (!m) return undefined;
  const part: Record<string, string> = {
    "attention.q_proj": "q",
    "attention.k_proj": "k",
    "attention.v_proj": "v",
    "attention.out_proj": "o",
    layer_norm: "ln1",
    "self_attn.q_proj": "q",
    "self_attn.k_proj": "k",
    "self_attn.v_proj": "v",
    "self_attn.out_proj": "o",
    self_attn_layer_norm: "ln1",
    "encoder_attn.q_proj": "cq",
    "encoder_attn.k_proj": "ck",
    "encoder_attn.v_proj": "cv",
    "encoder_attn.out_proj": "co",
    encoder_attn_layer_norm: "lnc",
    "feed_forward.intermediate_dense": "fc1",
    "feed_forward.output_dense": "fc2",
    final_layer_norm: "ln2",
  };
  const p = part[m[3]!];
  return p ? `${m[1] === "encoder" ? "enc" : "dec"}.l${m[2]}.${p}.${m[4]}` : undefined;
}

/** SpeechT5ScaledPositionalEncoding's table row `pos` (dim values), as torch builds it in float32. */
export function speecht5Positional(pos: number, dim: number): Float32Array {
  const f = Math.fround;
  const out = new Float32Array(dim);
  const k = f(-(Math.log(10000) / dim));
  for (let i = 0; i < dim; i += 2) {
    const div = f(Math.exp(f(i * k)));
    const a = f(pos * div);
    out[i] = Math.sin(a);
    if (i + 1 < dim) out[i + 1] = Math.cos(a);
  }
  return out;
}

/** Weights used on the CPU (speech prenet, output heads), by checkpoint name. */
export type CpuWeights = (name: string) => Float32Array;

function dense(w: Float32Array, b: Float32Array, x: ArrayLike<number>, out: number): Float64Array {
  const n = x.length;
  const y = new Float64Array(out);
  for (let o = 0; o < out; o++) {
    let s = b[o]!;
    for (let i = 0; i < n; i++) s += w[o * n + i]! * x[i]!;
    y[o] = s;
  }
  return y;
}

export interface SpeechT5Options {
  /** Stop once the stop probabilities of a step sum to this (default 0.5). */
  threshold?: number;
  minlenratio?: number;
  maxlenratio?: number;
  /**
   * The speech prenet's dropout, which transformers keeps on at inference: "sample" (default) draws
   * the masks, "keep" keeps every unit (still scaled by 1 / (1 − p)), a deterministic stand-in.
   */
  prenetDropout?: "sample" | "keep";
  /** Seed for the dropout masks. */
  seed?: number;
}

/**
 * SpeechT5 text-to-speech: the text encoder and the speech decoder on the seq2seq code (one decoder
 * step per pair of frames, reading its input vector), the speech prenet and the output heads on the
 * CPU, as transformers' _generate_speech runs them; then the postnet (convolutions with batch norm
 * folded, tanh, residual) and the HiFi-GAN vocoder (its input normalized by the vocoder's mean and
 * scale) on the GPU.
 */
export class SpeechT5Model {
  private readonly plans = new Map<number, { post: { dispatches: readonly Dispatch[]; input: Buf; out: Tensor; ctx: Ctx }; voc: { dispatches: readonly Dispatch[]; input: Buf; out: Tensor; ctx: Ctx } }>();
  private readonly pool: ActivationPool;
  private readonly step: Buf;

  constructor(
    readonly backend: Backend,
    readonly cfg: SpeechT5Config,
    readonly vocoderCfg: HifiGanConfig & { samplingRate: number },
    private readonly seq: Seq2SeqModel,
    /** Postnet and vocoder weights. */
    private readonly weights: Map<string, Tensor>,
    private readonly cpu: CpuWeights,
    /** The vocoder's input normalization (normalize_before), or undefined. */
    private readonly norm: { mean: Float32Array; scale: Float32Array } | undefined,
  ) {
    this.pool = new ActivationPool(backend, "speecht5");
    this.step = backend.alloc(STEP_BYTES, "storage", "speecht5/step");
    backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`SpeechT5 weight "${name}" was not loaded`);
    return t;
  }

  private plan(F: number) {
    let p = this.plans.get(F);
    if (p) return p;
    for (const old of this.plans.values()) {
      old.post.ctx.destroy();
      old.voc.ctx.destroy();
    }
    this.plans.clear();
    const c = this.cfg;
    const w = (n: string) => this.w(n);
    // Postnet: conv + folded batch norm (+ tanh but the last), the last adding the input back.
    const pctx = new Ctx(this.backend, this.step, `speecht5/postnet F${F}`, this.pool);
    const pin = pctx.buffer(F * c.mel * 4, "speecht5/spectrum");
    const spec: Tensor = { shape: [F, c.mel], dtype: "f32", buf: pin };
    let h = spec;
    for (let i = 0; i < c.postnetLayers; i++) {
      const last = i === c.postnetLayers - 1;
      h = conv1d(pctx, w, h, `post.l${i}`, c.postnetKernel, { tanh: !last, residual: last ? spec : undefined });
    }
    const post = { dispatches: pctx.dispatches, input: pin, out: h, ctx: pctx };
    const vctx = new Ctx(this.backend, this.step, `speecht5/vocoder F${F}`, this.pool);
    const vin = vctx.buffer(F * c.mel * 4, "speecht5/mel");
    const out = hifiGan(vctx, w, this.vocoderCfg, { shape: [F, c.mel], dtype: "f32", buf: vin }, "voc");
    p = { post, voc: { dispatches: vctx.dispatches, input: vin, out, ctx: vctx } };
    this.plans.set(F, p);
    return p;
  }

  /** The speech prenet for the frame at decoder position `pos`, with the (normalized) speaker embedding. */
  private prenet(frame: Float32Array, pos: number, speaker: Float64Array, masks: () => Float64Array | undefined): Float32Array {
    const c = this.cfg;
    const p = "speecht5.decoder.prenet.";
    let x: Float64Array = Float64Array.from(frame);
    const keep = 1 / (1 - c.prenetDropout);
    for (let i = 0; i < c.prenetLayers; i++) {
      const w = this.cpu(`${p}layers.${i}.weight`);
      const b = this.cpu(`${p}layers.${i}.bias`);
      x = dense(w, b, x, b.length).map((v) => Math.max(v, 0));
      const m = masks();
      for (let j = 0; j < x.length; j++) x[j] = (m && !m[j] ? 0 : x[j]!) * keep;
    }
    const fb = this.cpu(`${p}final_layer.bias`);
    const y = dense(this.cpu(`${p}final_layer.weight`), fb, x, fb.length);
    const alpha = this.cpu(`${p}encode_positions.alpha`)[0]!;
    const pe = speecht5Positional(pos, y.length);
    const cat = new Float64Array(y.length + speaker.length);
    for (let j = 0; j < y.length; j++) cat[j] = y[j]! + Math.fround(alpha * pe[j]!);
    cat.set(speaker, y.length);
    const sb = this.cpu(`${p}speaker_embeds_layer.bias`);
    return Float32Array.from(dense(this.cpu(`${p}speaker_embeds_layer.weight`), sb, cat, sb.length), (v) => Math.max(v, 0));
  }

  /** The waveform for token ids and a speaker embedding (an x-vector, 512 values for the Microsoft checkpoints). */
  async synthesize(ids: readonly number[], speakerEmbedding: Float32Array, o: SpeechT5Options = {}): Promise<Float32Array> {
    const c = this.cfg;
    const seq = this.seq;
    await seq.encode(ids);
    const L = ids.length;
    const maxlen = Math.min(Math.floor((L * (o.maxlenratio ?? 20)) / c.reduction), seq.maxTarget);
    const minlen = Math.floor((L * (o.minlenratio ?? 0)) / c.reduction);
    const threshold = o.threshold ?? 0.5;
    // F.normalize: the speaker embedding over its L2 norm.
    let n = 0;
    for (const v of speakerEmbedding) n += v * v;
    n = Math.max(Math.sqrt(n), 1e-12);
    const speaker = Float64Array.from(speakerEmbedding, (v) => v / n);
    const rand = o.prenetDropout === "keep" ? undefined : normalStream(o.seed ?? Math.floor(Math.random() * 2 ** 32));
    const units = this.cpu("speecht5.decoder.prenet.layers.0.bias").length;
    // Bernoulli(1 − p) keep masks from the normal stream (P(z > Φ⁻¹(p)) = 1 − p; p = 0.5: z > 0).
    const masks = () => (rand ? Float64Array.from({ length: units }, () => (rand() > 0 ? 1 : 0)) : undefined);
    if (rand && c.prenetDropout !== 0.5) throw new UnsupportedModelError("SpeechT5 prenet dropout other than 0.5 is not supported");
    const featW = this.cpu("speech_decoder_postnet.feat_out.weight");
    const featB = this.cpu("speech_decoder_postnet.feat_out.bias");
    const probW = this.cpu("speech_decoder_postnet.prob_out.weight");
    const probB = this.cpu("speech_decoder_postnet.prob_out.bias");
    const frames: Float32Array[] = [];
    let last = new Float32Array(c.mel);
    for (let idx = 1; ; idx++) {
      const h = await seq.stepHidden(this.prenet(last, idx - 1, speaker, masks), idx - 1);
      const spectrum = Float32Array.from(dense(featW, featB, h, featB.length));
      for (let r = 0; r < c.reduction; r++) frames.push(spectrum.slice(r * c.mel, (r + 1) * c.mel));
      last = frames[frames.length - 1]!;
      if (idx < minlen) continue;
      const prob = dense(probW, probB, h, probB.length).reduce((s, v) => s + 1 / (1 + Math.exp(-v)), 0);
      if (prob >= threshold || idx >= maxlen) break;
    }
    const F = frames.length;
    const be = this.backend;
    const p = this.plan(F);
    const spec = new Float32Array(F * c.mel);
    frames.forEach((fr, i) => spec.set(fr, i * c.mel));
    be.write(p.post.input, 0, spec);
    be.submit(p.post.dispatches);
    const mel = new Float32Array(await be.readback(physical(p.post.out.buf), 0, F * c.mel * 4));
    if (this.norm) for (let i = 0; i < mel.length; i++) mel[i] = Math.fround(Math.fround(mel[i]! - this.norm.mean[i % c.mel]!) / this.norm.scale[i % c.mel]!);
    be.write(p.voc.input, 0, mel);
    be.submit(p.voc.dispatches);
    const out = new Float32Array(await be.readback(physical(p.voc.out.buf), 0, F * this.vocoderCfg.upsampleRates.reduce((a, b) => a * b, 1) * 4));
    for (let i = 0; i < out.length; i++) out[i] = Math.tanh(out[i]!);
    return out;
  }

  destroy(): void {
    this.seq.destroy();
    for (const t of this.weights.values()) this.backend.free(t.buf);
    for (const p of this.plans.values()) {
      p.post.ctx.destroy();
      p.voc.ctx.destroy();
    }
    this.pool.destroy();
    this.backend.free(this.step);
  }
}
