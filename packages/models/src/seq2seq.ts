import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops, type Activation } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/**
 * BART-style encoder-decoder models: BART / DistilBART, Marian (OPUS-MT), M2M100 / NLLB, Whisper; and
 * "gpt2", the GPT-2 decoder with cross-attention of a VisionEncoderDecoder, whose "encoder" output
 * is given (an image model's hidden states).
 */
export type Seq2SeqFamily = "bart" | "marian" | "m2m100" | "t5" | "whisper" | "gpt2";

export interface Seq2SeqConfig {
  family: Seq2SeqFamily;
  modelType: string;
  dModel: number;
  encLayers: number;
  decLayers: number;
  encHeads: number;
  decHeads: number;
  encFfn: number;
  decFfn: number;
  vocab: number;
  /** Longest source and target, in tokens. */
  maxPositions: number;
  /** Pre-LayerNorm layers with a final LayerNorm (M2M100, mBART), else post-LayerNorm (BART, Marian). */
  preNorm: boolean;
  /** A LayerNorm right after the embeddings (BART). */
  layernormEmbedding: boolean;
  /** Learned position tables (BART), or sinusoidal ones in Marian's or M2M100's layout. */
  positions: "learned" | "sinusoidal-marian" | "sinusoidal-m2m100";
  /** Row of the first position (BART and M2M100: 2). */
  posOffset: number;
  /** Token embeddings times sqrt(d_model) (Marian, M2M100). */
  embedScale: number;
  act: Activation;
  pad: number;
  eos: number;
  decoderStart: number;
  /** Raw config.json, for generation settings, prefixes and task parameters. */
  raw: Record<string, unknown>;
  /** Whisper: log-mel input of `frames` × `mels` through two convolutions to `frames / 2` positions. */
  audio?: { mels: number; frames: number };
}

const num = (c: Record<string, unknown>, k: string): number => {
  if (typeof c[k] !== "number") throw new UnsupportedModelError(`config.json is missing ${k}`);
  return c[k] as number;
};

function activation(name: unknown): Activation {
  switch (name) {
    case "gelu":
      return "gelu";
    case "gelu_new":
    case "gelu_pytorch_tanh":
      return "gelu_tanh";
    case "swish":
    case "silu":
      return "silu";
    case "relu":
      return "relu";
  }
  throw new UnsupportedModelError(`activation "${String(name)}" is not supported`);
}

/** The config of a BART, Marian or M2M100 (NLLB) checkpoint. */
export function seq2seqConfigFromHF(c: Record<string, unknown>): Seq2SeqConfig {
  const type = String(c.model_type);
  if (type === "whisper") {
    const src = num(c, "max_source_positions");
    return {
      family: "whisper",
      modelType: type,
      dModel: num(c, "d_model"),
      encLayers: num(c, "encoder_layers"),
      decLayers: num(c, "decoder_layers"),
      encHeads: num(c, "encoder_attention_heads"),
      decHeads: num(c, "decoder_attention_heads"),
      encFfn: num(c, "encoder_ffn_dim"),
      decFfn: num(c, "decoder_ffn_dim"),
      vocab: num(c, "vocab_size"),
      maxPositions: num(c, "max_target_positions"),
      preNorm: true,
      layernormEmbedding: false,
      positions: "learned",
      posOffset: 0,
      embedScale: c.scale_embedding ? Math.sqrt(num(c, "d_model")) : 1,
      act: activation(c.activation_function),
      pad: Number(c.pad_token_id ?? 50257),
      eos: Number(c.eos_token_id ?? 50257),
      decoderStart: Number(c.decoder_start_token_id ?? 50258),
      raw: c,
      audio: { mels: num(c, "num_mel_bins"), frames: 2 * src },
    };
  }
  const family: Seq2SeqFamily | undefined = type === "bart" ? "bart" : type === "marian" ? "marian" : type === "m2m_100" || type === "m2m100" || type === "nllb-moe" ? "m2m100" : undefined;
  if (!family || type === "nllb-moe") throw new UnsupportedModelError(`encoder-decoder model_type "${type}" is not supported (bart, marian and m2m100 / NLLB are)`);
  const d = num(c, "d_model");
  const pad = Number(c.pad_token_id ?? 1);
  return {
    family,
    modelType: type,
    dModel: d,
    encLayers: num(c, "encoder_layers"),
    decLayers: num(c, "decoder_layers"),
    encHeads: num(c, "encoder_attention_heads"),
    decHeads: num(c, "decoder_attention_heads"),
    encFfn: num(c, "encoder_ffn_dim"),
    decFfn: num(c, "decoder_ffn_dim"),
    vocab: num(c, "vocab_size"),
    maxPositions: num(c, "max_position_embeddings"),
    preNorm: family === "m2m100" || c.normalize_before === true,
    layernormEmbedding: family === "bart",
    positions: family === "bart" ? "learned" : family === "marian" ? "sinusoidal-marian" : "sinusoidal-m2m100",
    posOffset: family === "marian" ? 0 : 2,
    embedScale: c.scale_embedding ? Math.sqrt(d) : 1,
    act: activation(c.activation_function),
    pad,
    eos: Number(c.eos_token_id ?? 2),
    decoderStart: Number(c.decoder_start_token_id ?? (family === "marian" ? pad : 2)),
    raw: c,
  };
}

/**
 * The decoder of a VisionEncoderDecoder (`decoder` in its config.json): GPT-2 with cross-attention,
 * pre-LayerNorm, learned positions from row 0, tanh-GELU, logits from the tied token embedding.
 */
export function visionDecoderConfigFromHF(top: Record<string, unknown>): Seq2SeqConfig {
  const c = (top.decoder as Record<string, unknown> | undefined) ?? {};
  if (c.model_type !== "gpt2") throw new UnsupportedModelError(`VisionEncoderDecoder decoder "${String(c.model_type)}" is not supported (gpt2)`);
  if (c.add_cross_attention !== true) throw new UnsupportedModelError("the GPT-2 decoder has no cross-attention");
  const d = num(c, "n_embd");
  const eos = Number(top.eos_token_id ?? c.eos_token_id ?? 50256);
  return {
    family: "gpt2",
    modelType: "gpt2",
    dModel: d,
    encLayers: 0,
    decLayers: num(c, "n_layer"),
    encHeads: num(c, "n_head"),
    decHeads: num(c, "n_head"),
    encFfn: 0,
    decFfn: Number(c.n_inner ?? 4 * d),
    vocab: num(c, "vocab_size"),
    maxPositions: num(c, "n_positions"),
    preNorm: true,
    layernormEmbedding: false,
    positions: "learned",
    posOffset: 0,
    embedScale: 1,
    act: activation(c.activation_function ?? "gelu_new"),
    pad: Number(top.pad_token_id ?? eos),
    eos,
    decoderStart: Number(top.decoder_start_token_id ?? c.bos_token_id ?? eos),
    raw: top,
  };
}

/** A sinusoidal position table [rows, d] as Marian or M2M100 builds it (positions from row 0). */
export function sinusoidalTable(kind: "sinusoidal-marian" | "sinusoidal-m2m100", rows: number, d: number, padRow?: number): Float32Array {
  const half = d / 2;
  const out = new Float32Array(rows * d);
  for (let p = 0; p < rows; p++) {
    for (let c = 0; c < half; c++) {
      // Marian: pos / 10000^(2c / d); M2M100: pos · exp(-c · ln(10000) / (half - 1)).
      const a = kind === "sinusoidal-marian" ? p / Math.pow(10000, (2 * c) / d) : p * Math.exp((-c * Math.log(10000)) / (half - 1));
      out[p * d + c] = Math.sin(a);
      out[p * d + half + c] = Math.cos(a);
    }
  }
  if (padRow !== undefined) out.fill(0, padRow * d, (padRow + 1) * d);
  return out;
}

/**
 * The model's tensors under one naming ("shared.weight", "enc.l3.fc1.weight", "dec.l0.cq.bias"…),
 * or undefined for tensors Kiln does not use. Decoder cross-attention projections are c[q|k|v|o],
 * its LayerNorm "lnc".
 */
export function seq2seqName(name: string): string | undefined {
  const n = name.replace(/^model\./, "");
  if (n === "shared.weight" || n === "decoder.embed_tokens.weight" || n === "encoder.embed_tokens.weight") return "shared.weight";
  if (name === "final_logits_bias") return "flb";
  if (name === "lm_head.weight") return "lm_head.weight";
  // Whisper: conv biases load as they are (the weights are rearranged by the loader); proj_out is tied.
  const conv = /^model\.encoder\.(conv[12])\.bias$/.exec(name);
  if (conv) return `enc.${conv[1]}.bias`;
  let m = /^(encoder|decoder)\.(embed_positions\.weight|layernorm_embedding\.(weight|bias)|layer_norm\.(weight|bias))$/.exec(n);
  if (m) {
    const side = m[1] === "encoder" ? "enc" : "dec";
    if (m[2] === "embed_positions.weight") return `${side}.pos`;
    return `${side}.${m[2]!.startsWith("layernorm") ? "lnE" : "lnF"}.${m[3] ?? m[4]}`;
  }
  m = /^(encoder|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$/.exec(n);
  if (!m) return undefined;
  const part: Record<string, string> = {
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
    fc1: "fc1",
    fc2: "fc2",
    final_layer_norm: "ln2",
  };
  const p = part[m[3]!];
  return p ? `${m[1] === "encoder" ? "enc" : "dec"}.l${m[2]}.${p}.${m[4]}` : undefined;
}

/**
 * Greedy token choice on the backend for encoder-decoder decoders: the shared sampling kernel at
 * temperature 0 over the step's logits, under an allowed-token mask (processors such as suppressed
 * tokens, min_length's end-of-sequence ban and no-repeat n-grams become banned bits) and the
 * repetition penalty's seen-token bits. Each step reads back 4 bytes instead of the vocabulary.
 */
export class DecoderSampler {
  readonly seen: Buf;
  readonly allowed: Buf;
  readonly out: Buf;
  private readonly words: number;
  private maskKey = "";
  private seenKey = "";

  constructor(
    private readonly backend: Backend,
    readonly vocab: number,
  ) {
    this.words = Math.ceil(vocab / 32);
    this.seen = backend.alloc(this.words * 4, "storage", "seq2seq/seen");
    this.allowed = backend.alloc((1 + this.words) * 4, "storage", "seq2seq/allowed");
    this.out = backend.alloc(16, "storage", "seq2seq/sampled");
    backend.write(this.allowed, 0, new Uint32Array(1 + this.words));
    backend.write(this.seen, 0, new Uint32Array(this.words));
  }

  /** Writes the masks for the next sampling: `banned` token ids, and the ids the penalty applies to. */
  prepare(banned: Iterable<number>, seenIds?: readonly number[]): void {
    const ban = [...new Set(banned)].filter((i) => i >= 0 && i < this.vocab).sort((a, b) => a - b);
    const key = ban.join(",");
    if (key !== this.maskKey) {
      const m = new Uint32Array(1 + this.words);
      if (ban.length) {
        m.fill(0xffffffff);
        m[0] = 1;
        for (const i of ban) m[1 + (i >>> 5)]! &= ~(1 << (i & 31));
      }
      this.backend.write(this.allowed, 0, m);
      this.maskKey = key;
    }
    if (seenIds) {
      const k = seenIds.join(",");
      if (k !== this.seenKey) {
        const m = new Uint32Array(this.words);
        for (const i of seenIds) if (i >= 0 && i < this.vocab) m[i >>> 5]! |= 1 << (i & 31);
        this.backend.write(this.seen, 0, m);
        this.seenKey = k;
      }
    }
  }

  async read(): Promise<number> {
    return new Uint32Array(await this.backend.readback(this.out, 0, 4))[0]!;
  }

  destroy(): void {
    for (const b of [this.seen, this.allowed, this.out]) this.backend.free(b);
  }
}

/** What a decoding step of either encoder-decoder model takes besides the token and its position. */
export interface StepOptions {
  /** Token ids that must not be chosen. */
  banned?: Iterable<number>;
  /** HF-style repetition penalty over `seen` (1 = off). */
  repetitionPenalty?: number;
  seen?: readonly number[];
}

interface Pair {
  /** Source length these plans are recorded for. */
  S: number;
  enc: { ctx: Ctx; dispatches: readonly Dispatch[]; step: Buf; ids: Buf; pos: Buf; kc: Buf; vc: Buf };
  /** Whisper: the log-mel input [frames, mels]. */
  feats?: Buf;
  /** GPT-2 decoder: the given encoder hidden states [S, d_model]. */
  hidden?: Buf;
  dec: { ctx: Ctx; dispatches: readonly Dispatch[]; logits: Tensor };
  /** The decoder plan with the alignment heads' cross-attention recorded (see enableAlignment). */
  decAlign?: readonly Dispatch[];
  /** Cross-attention K/V per decoder layer, written by the encoder plan. */
  crossK: Buf[];
  crossV: Buf[];
}

/**
 * Encoder-decoder transformer: the encoder runs a source once (full attention over a span) and writes
 * every decoder layer's cross-attention K/V; the decoder then runs one token per step against its own
 * K/V cache, causal attention, and cross-attention through a step block whose span covers every
 * position, ending in logits over the vocabulary (the shared embedding, plus final_logits_bias).
 */
export class Seq2SeqModel {
  private readonly pairs = new Map<number, Pair>();
  private readonly pool: ActivationPool;
  private readonly decStep: Buf;
  private readonly crossStep: Buf;
  private readonly tok: Buf;
  private readonly posBuf: Buf;
  private readonly selfK: Buf[] = [];
  private readonly selfV: Buf[] = [];
  private readonly sampler: DecoderSampler;
  /** Longest target, in tokens. */
  readonly maxTarget: number;
  /** Cross-attention heads [layer, head] whose weights the decoder records, and where. */
  private alignHeads: [number, number][] = [];
  private align: Buf | undefined;
  private alignS = 0;
  /** Whether next/nextDetail/step record the alignment heads (enableAlignment first). */
  recordAlignment = false;

  constructor(
    readonly backend: Backend,
    readonly cfg: Seq2SeqConfig,
    private readonly weights: Map<string, Tensor>,
    opts: { maxTarget?: number } = {},
  ) {
    for (const [d, h] of [[cfg.dModel, cfg.encHeads], [cfg.dModel, cfg.decHeads]] as const) {
      if (d % h) throw new UnsupportedModelError(`d_model ${d} is not divisible by ${h} heads`);
    }
    this.maxTarget = Math.min(opts.maxTarget ?? cfg.maxPositions, cfg.maxPositions);
    this.pool = new ActivationPool(backend, "seq2seq");
    this.decStep = backend.alloc(STEP_BYTES, "storage", "seq2seq/step");
    this.crossStep = backend.alloc(STEP_BYTES, "storage", "seq2seq/cross-step");
    // Every decoder row inside the span: all S encoder keys, whatever the decoding position.
    backend.write(this.crossStep, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: 0x7fffffff }));
    this.tok = backend.alloc(16, "storage", "seq2seq/token");
    this.posBuf = backend.alloc(16, "storage", "seq2seq/pos");
    this.sampler = new DecoderSampler(backend, cfg.vocab);
    for (let l = 0; l < cfg.decLayers; l++) {
      this.selfK.push(backend.alloc(this.maxTarget * cfg.dModel * 4, "storage", "seq2seq/self-k"));
      this.selfV.push(backend.alloc(this.maxTarget * cfg.dModel * 4, "storage", "seq2seq/self-v"));
    }
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`seq2seq weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string, residual?: Tensor): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
  }

  private ln(ctx: Ctx, x: Tensor, p: string): Tensor {
    return ops.layernorm(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), 1e-5);
  }

  /** Token + position embeddings (scaled, then LayerNorm for BART). */
  private embed(ctx: Ctx, side: "enc" | "dec", ids: Buf, pos: Buf, T: number): Tensor {
    let x = ops.embed(ctx, this.w("shared.weight"), ids, T, this.cfg.embedScale);
    ops.addInPlace(ctx, x, ops.embed(ctx, this.w(`${side}.pos`), pos, T));
    if (this.cfg.layernormEmbedding) x = this.ln(ctx, x, `${side}.lnE`);
    return x;
  }

  /** Attention sub-block with its residual, pre- or post-norm. `kv` writes the cache; `attend` reads it. */
  private block(ctx: Ctx, x: Tensor, norm: string, f: (h: Tensor) => Tensor, out: string): Tensor {
    if (this.cfg.preNorm) {
      this.lin(ctx, f(this.ln(ctx, x, norm)), out, x);
      return x;
    }
    this.lin(ctx, f(x), out, x);
    return this.ln(ctx, x, norm);
  }

  private ffn(ctx: Ctx, x: Tensor, p: string): Tensor {
    const run = (h: Tensor) => {
      const up = this.lin(ctx, h, `${p}.fc1`);
      ops.act(ctx, up, this.cfg.act);
      return up;
    };
    return this.block(ctx, x, `${p}.ln2`, run, `${p}.fc2`);
  }

  private build(S: number): Pair {
    const c = this.cfg;
    const be = this.backend;
    const d = c.dModel;
    // Encoder over the S source tokens, bidirectional.
    const step = be.alloc(STEP_BYTES, "storage", "seq2seq/enc-step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: S }));
    const ectx = new Ctx(be, step, `seq2seq/enc S${S}`, this.pool);
    const ids = ectx.buffer(S * 4, "seq2seq/src");
    const pos = ectx.buffer(S * 4, "seq2seq/src-pos");
    const kc = be.alloc(S * d * 4, "storage", "seq2seq/enc-k");
    const vc = be.alloc(S * d * 4, "storage", "seq2seq/enc-v");
    let feats: Buf | undefined;
    let hidden: Buf | undefined;
    let x: Tensor;
    if (c.family === "gpt2") {
      hidden = ectx.buffer(S * d * 4, "seq2seq/hidden");
      x = { shape: [S, d], dtype: "f32", buf: hidden };
    } else if (c.audio) {
      // Whisper: log-mel frames → conv1 (k 3) → GELU → conv2 (k 3, stride 2) → GELU → + positions.
      const { mels, frames } = c.audio;
      feats = ectx.buffer(frames * mels * 4, "seq2seq/mel");
      const conv = (inp: Tensor, n: string, stride: number) => {
        const y = ops.linear(ectx, ops.im2col1d(ectx, inp, 3, stride, 1), this.w(`enc.${n}.weight`), { bias: this.w(`enc.${n}.bias`) });
        ops.gelu(ectx, y, true);
        return y;
      };
      x = conv(conv({ shape: [frames, mels], dtype: "f32", buf: feats }, "conv1", 1), "conv2", 2);
      ops.addInPlace(ectx, x, this.w("enc.pos"));
    } else x = this.embed(ectx, "enc", ids, pos, S);
    const De = d / c.encHeads;
    const eshape = { T: S, H: c.encHeads, Hkv: c.encHeads, D: De, maxSeq: S, kv: "f32" as const, scale: De ** -0.5, span: true };
    for (let l = 0; l < c.encLayers; l++) {
      const p = `enc.l${l}`;
      x = this.block(ectx, x, `${p}.ln1`, (h) => {
        const q = this.lin(ectx, h, `${p}.q`);
        ops.kvWrite(ectx, this.lin(ectx, h, `${p}.k`), this.lin(ectx, h, `${p}.v`), kc, vc, S, S, "f32");
        return ops.attention(ectx, q, kc, vc, eshape);
      }, `${p}.o`);
      x = this.ffn(ectx, x, p);
    }
    if (c.preNorm && c.encLayers) x = this.ln(ectx, x, "enc.lnF");
    const crossK: Buf[] = [];
    const crossV: Buf[] = [];
    for (let l = 0; l < c.decLayers; l++) {
      const k = be.alloc(S * d * 4, "storage", "seq2seq/cross-k");
      const v = be.alloc(S * d * 4, "storage", "seq2seq/cross-v");
      ops.kvWrite(ectx, this.lin(ectx, x, `dec.l${l}.ck`), this.lin(ectx, x, `dec.l${l}.cv`), k, v, S, S, "f32");
      crossK.push(k);
      crossV.push(v);
    }

    // Decoder: one token at the step's position.
    const dctx = new Ctx(be, this.decStep, `seq2seq/dec S${S}`, this.pool);
    let y = this.embed(dctx, "dec", this.tok, this.posBuf, 1);
    const Dd = d / c.decHeads;
    const self = { T: 1, H: c.decHeads, Hkv: c.decHeads, D: Dd, maxSeq: this.maxTarget, kv: "f32" as const, scale: Dd ** -0.5 };
    const cross = { T: 1, H: c.decHeads, Hkv: c.decHeads, D: Dd, maxSeq: S, kv: "f32" as const, scale: Dd ** -0.5, step: this.crossStep };
    for (let l = 0; l < c.decLayers; l++) {
      const p = `dec.l${l}`;
      y = this.block(dctx, y, `${p}.ln1`, (h) => {
        const q = this.lin(dctx, h, `${p}.q`);
        ops.kvWrite(dctx, this.lin(dctx, h, `${p}.k`), this.lin(dctx, h, `${p}.v`), this.selfK[l]!, this.selfV[l]!, 1, this.maxTarget, "f32");
        return ops.attention(dctx, q, this.selfK[l]!, this.selfV[l]!, self);
      }, `${p}.o`);
      y = this.block(dctx, y, `${p}.lnc`, (h) => {
        const q = this.lin(dctx, h, `${p}.cq`);
        const out = ops.attention(dctx, q, crossK[l]!, crossV[l]!, cross);
        let mask = 0;
        let slotBase = -1;
        this.alignHeads.forEach(([al, ah], i) => {
          if (al !== l) return;
          mask |= 1 << ah;
          if (slotBase < 0) slotBase = i;
        });
        if (mask && this.align && S === this.alignS) ops.attnProbs(dctx, q, crossK[l]!, this.align, { S, H: c.decHeads, D: Dd, mask, scale: Dd ** -0.5, slotBase, nSlots: this.alignHeads.length });
        return out;
      }, `${p}.co`);
      y = this.ffn(dctx, y, p);
    }
    if (c.preNorm) y = this.ln(dctx, y, "dec.lnF");
    const head = this.weights.get("lm_head.weight") ?? this.w("shared.weight");
    const logits = ops.linear(dctx, y, head, { bias: this.weights.get("flb") });
    ops.sample(dctx, logits, this.sampler.out, this.sampler.seen, this.sampler.allowed);
    ops.pickLogit(dctx, logits, this.sampler.out);
    const all = dctx.dispatches;
    const plain = all.filter((x) => x.label !== "attn_probs");
    return {
      S,
      feats,
      hidden,
      enc: { ctx: ectx, dispatches: ectx.dispatches, step, ids, pos, kc, vc },
      dec: { ctx: dctx, dispatches: plain, logits },
      decAlign: plain.length < all.length ? all : undefined,
      crossK,
      crossV,
    };
  }

  private current: Pair | undefined;

  /** Runs the encoder on a source (token ids, special tokens included) and readies the decoder for it. */
  async encode(ids: readonly number[]): Promise<void> {
    const S = ids.length;
    if (!S) throw new Error("seq2seq: empty source");
    if (S > this.cfg.maxPositions) throw new Error(`seq2seq: ${S} source tokens; the model takes at most ${this.cfg.maxPositions}`);
    let pair = this.pairs.get(S);
    if (!pair) {
      if (this.pairs.size >= 4) {
        const [k, old] = this.pairs.entries().next().value!;
        this.dropPair(old);
        this.pairs.delete(k);
      }
      pair = this.build(S);
      this.pairs.set(S, pair);
    }
    const be = this.backend;
    be.write(pair.enc.ids, 0, Uint32Array.from(ids));
    be.write(pair.enc.pos, 0, Uint32Array.from({ length: S }, (_, i) => i + this.cfg.posOffset));
    be.submit(pair.enc.dispatches);
    this.current = pair;
  }

  /** Whisper: runs the encoder on log-mel features [frames, mels] and readies the decoder. */
  async encodeAudio(features: Float32Array): Promise<void> {
    const a = this.cfg.audio;
    if (!a) throw new Error("seq2seq: this model takes text, not audio");
    if (features.length !== a.frames * a.mels) throw new Error(`seq2seq: expected ${a.frames} × ${a.mels} features, got ${features.length}`);
    const S = a.frames / 2;
    let pair = this.pairs.get(S);
    if (!pair) {
      pair = this.build(S);
      this.pairs.set(S, pair);
    }
    this.backend.write(pair.feats!, 0, features);
    this.backend.submit(pair.enc.dispatches);
    this.current = pair;
  }

  /** GPT-2 decoder: takes an image model's hidden states [S, d_model] as the encoder output and readies the decoder. */
  async encodeHidden(hidden: Float32Array, S: number): Promise<void> {
    if (this.cfg.family !== "gpt2") throw new Error("seq2seq: this model encodes its own source");
    if (hidden.length !== S * this.cfg.dModel) throw new Error(`seq2seq: expected ${S} × ${this.cfg.dModel} hidden states, got ${hidden.length}`);
    let pair = this.pairs.get(S);
    if (!pair) {
      pair = this.build(S);
      this.pairs.set(S, pair);
    }
    this.backend.write(pair.hidden!, 0, hidden);
    this.backend.submit(pair.enc.dispatches);
    this.current = pair;
  }

  /** Feeds the target token at position t (0 = the decoder start token); resolves to the next token's logits. */
  async step(token: number, t: number): Promise<Float32Array> {
    const pair = this.current;
    if (!pair) throw new Error("seq2seq: encode a source first");
    if (t >= this.maxTarget) throw new Error(`seq2seq: target longer than ${this.maxTarget} tokens`);
    const be = this.backend;
    be.write(this.tok, 0, Uint32Array.of(token));
    be.write(this.posBuf, 0, Uint32Array.of(t + this.cfg.posOffset));
    be.write(this.decStep, 0, packStep({ pos0: t, lastRow: 0, seed: 0, temperature: 0 }));
    be.submit(this.decPlan(pair));
    return new Float32Array(await be.readback(physical(pair.dec.logits.buf), 0, this.cfg.vocab * 4));
  }

  /** Feeds the token at position t and returns the greedy next token, chosen on the backend under `opts`. */
  async next(token: number, t: number, opts: StepOptions = {}): Promise<number> {
    const pair = this.current;
    if (!pair) throw new Error("seq2seq: encode a source first");
    if (t >= this.maxTarget) throw new Error(`seq2seq: target longer than ${this.maxTarget} tokens`);
    const be = this.backend;
    const rep = opts.repetitionPenalty ?? 1;
    this.sampler.prepare(opts.banned ?? [], rep !== 1 ? opts.seen : undefined);
    be.write(this.tok, 0, Uint32Array.of(token));
    be.write(this.posBuf, 0, Uint32Array.of(t + this.cfg.posOffset));
    be.write(this.decStep, 0, packStep({ pos0: t, lastRow: 0, seed: 0, temperature: 0, repetitionPenalty: rep }));
    be.submit(this.decPlan(pair));
    return this.sampler.read();
  }

  /**
   * As next(), also returning the chosen token's logit and, with `tail`, the logits of every token
   * from `tail` up (Whisper's timestamp rule compares the timestamp tokens' total probability with
   * the best text token's). One wait for all of it.
   */
  async nextDetail(token: number, t: number, opts: StepOptions & { tail?: number } = {}): Promise<{ id: number; logit: number; tail?: Float32Array }> {
    const pair = this.current;
    if (!pair) throw new Error("seq2seq: encode a source first");
    if (t >= this.maxTarget) throw new Error(`seq2seq: target longer than ${this.maxTarget} tokens`);
    const be = this.backend;
    this.sampler.prepare(opts.banned ?? []);
    be.write(this.tok, 0, Uint32Array.of(token));
    be.write(this.posBuf, 0, Uint32Array.of(t + this.cfg.posOffset));
    be.write(this.decStep, 0, packStep({ pos0: t, lastRow: 0, seed: 0, temperature: 0 }));
    be.submit(this.decPlan(pair));
    const V = this.cfg.vocab;
    const [out, tail] = await Promise.all([
      be.readback(this.sampler.out, 0, 8),
      opts.tail !== undefined && opts.tail < V ? be.readback(physical(pair.dec.logits.buf), opts.tail * 4, (V - opts.tail) * 4) : undefined,
    ]);
    const u = new Uint32Array(out);
    return { id: u[0]!, logit: new Float32Array(out)[1]!, tail: tail ? new Float32Array(tail) : undefined };
  }

  private decPlan(pair: Pair): readonly Dispatch[] {
    return this.recordAlignment && pair.decAlign ? pair.decAlign : pair.dec.dispatches;
  }

  /**
   * Makes the decoder able to record the cross-attention weights of `heads` ([layer, head], in the order
   * alignment() returns them) for source length S. Plans built before are rebuilt.
   */
  enableAlignment(heads: readonly (readonly [number, number])[], S: number): void {
    const c = this.cfg;
    for (const [l, h] of heads) {
      if (l < 0 || l >= c.decLayers || h < 0 || h >= c.decHeads || h >= 32) throw new Error(`seq2seq: alignment head [${l}, ${h}] is out of range`);
    }
    const same = this.align && this.alignS === S && heads.length === this.alignHeads.length && heads.every(([l, h], i) => this.alignHeads[i]![0] === l && this.alignHeads[i]![1] === h);
    if (same) return;
    if (this.align) this.backend.free(this.align);
    this.alignHeads = heads.map(([l, h]) => [l, h]);
    this.alignS = S;
    this.align = heads.length ? this.backend.alloc(this.maxTarget * heads.length * S * 4, "storage", "seq2seq/align") : undefined;
    for (const p of this.pairs.values()) this.dropPair(p);
    this.pairs.clear();
  }

  /** Recorded alignment weights of decoder positions 0..rows-1: [rows, heads, S]. */
  async alignment(rows: number): Promise<Float32Array> {
    if (!this.align) throw new Error("seq2seq: enableAlignment first");
    const n = this.alignHeads.length * this.alignS;
    return new Float32Array(await this.backend.readback(this.align, 0, rows * n * 4));
  }

  private dropPair(p: Pair): void {
    p.enc.ctx.destroy();
    p.dec.ctx.destroy();
    for (const b of [p.enc.step, p.enc.kc, p.enc.vc, ...p.crossK, ...p.crossV]) this.backend.free(b);
    if (this.current === p) this.current = undefined;
  }

  destroy(): void {
    for (const p of this.pairs.values()) this.dropPair(p);
    this.pairs.clear();
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    this.pool.destroy();
    this.sampler.destroy();
    for (const b of [this.decStep, this.crossStep, this.tok, this.posBuf, ...this.selfK, ...this.selfV]) this.backend.free(b);
    if (this.align) this.backend.free(this.align);
  }
}
