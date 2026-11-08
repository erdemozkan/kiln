import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops, type Activation } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** BART-style encoder-decoder models: BART / DistilBART, Marian (OPUS-MT), M2M100 / NLLB. */
export type Seq2SeqFamily = "bart" | "marian" | "m2m100" | "t5";

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

interface Pair {
  /** Source length these plans are recorded for. */
  S: number;
  enc: { ctx: Ctx; dispatches: readonly Dispatch[]; step: Buf; ids: Buf; pos: Buf; kc: Buf; vc: Buf };
  dec: { ctx: Ctx; dispatches: readonly Dispatch[]; logits: Tensor };
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
  /** Longest target, in tokens. */
  readonly maxTarget: number;

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
    let x = this.embed(ectx, "enc", ids, pos, S);
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
    if (c.preNorm) x = this.ln(ectx, x, "enc.lnF");
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
      y = this.block(dctx, y, `${p}.lnc`, (h) => ops.attention(dctx, this.lin(dctx, h, `${p}.cq`), crossK[l]!, crossV[l]!, cross), `${p}.co`);
      y = this.ffn(dctx, y, p);
    }
    if (c.preNorm) y = this.ln(dctx, y, "dec.lnF");
    const head = this.weights.get("lm_head.weight") ?? this.w("shared.weight");
    const logits = ops.linear(dctx, y, head, { bias: this.weights.get("flb") });
    return {
      S,
      enc: { ctx: ectx, dispatches: ectx.dispatches, step, ids, pos, kc, vc },
      dec: { ctx: dctx, dispatches: dctx.dispatches, logits },
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

  /** Feeds the target token at position t (0 = the decoder start token); resolves to the next token's logits. */
  async step(token: number, t: number): Promise<Float32Array> {
    const pair = this.current;
    if (!pair) throw new Error("seq2seq: encode a source first");
    if (t >= this.maxTarget) throw new Error(`seq2seq: target longer than ${this.maxTarget} tokens`);
    const be = this.backend;
    be.write(this.tok, 0, Uint32Array.of(token));
    be.write(this.posBuf, 0, Uint32Array.of(t + this.cfg.posOffset));
    be.write(this.decStep, 0, packStep({ pos0: t, lastRow: 0, seed: 0, temperature: 0 }));
    be.submit(pair.dec.dispatches);
    return new Float32Array(await be.readback(physical(pair.dec.logits.buf), 0, this.cfg.vocab * 4));
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
    for (const b of [this.decStep, this.crossStep, this.tok, this.posBuf, ...this.selfK, ...this.selfV]) this.backend.free(b);
  }
}
