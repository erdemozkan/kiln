import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";
import { DecoderSampler, type Seq2SeqConfig, type StepOptions } from "./seq2seq";

/** T5 / Flan-T5 / mT5 specifics beside the shared encoder-decoder fields. */
export interface T5Config extends Seq2SeqConfig {
  /** Width of one attention head (inner width is heads × dKv, which need not equal d_model). */
  dKv: number;
  buckets: number;
  maxDistance: number;
  /** Feed-forward: gated GELU (wi_0, wi_1; T5 v1.1, Flan-T5, mT5) or ReLU (wi; T5 v1.0). */
  gated: boolean;
  /** Output layer tied to the shared embedding (inputs then scaled by d_model^-0.5). */
  tied: boolean;
  eps: number;
}

/** The config of a T5 checkpoint (model_type t5 or mt5). */
export function t5ConfigFromHF(c: Record<string, unknown>): T5Config {
  const type = String(c.model_type);
  if (type !== "t5" && type !== "mt5") throw new UnsupportedModelError(`"${type}" is not a T5 model`);
  const n = (k: string, d?: number) => {
    const v = c[k] ?? d;
    if (typeof v !== "number") throw new UnsupportedModelError(`config.json is missing ${k}`);
    return v;
  };
  const proj = String(c.feed_forward_proj ?? "relu");
  if (proj !== "relu" && proj !== "gated-gelu") throw new UnsupportedModelError(`feed_forward_proj "${proj}" is not supported`);
  const layers = n("num_layers");
  const heads = n("num_heads");
  return {
    family: "t5",
    modelType: type,
    dModel: n("d_model"),
    encLayers: layers,
    decLayers: n("num_decoder_layers", layers),
    encHeads: heads,
    decHeads: heads,
    encFfn: n("d_ff"),
    decFfn: n("d_ff"),
    vocab: n("vocab_size"),
    // Relative positions have no hard limit; n_positions is the trained length.
    maxPositions: n("n_positions", 512),
    preNorm: true,
    layernormEmbedding: false,
    positions: "learned",
    posOffset: 0,
    embedScale: 1,
    act: "gelu_tanh",
    pad: n("pad_token_id", 0),
    eos: n("eos_token_id", 1),
    decoderStart: n("decoder_start_token_id", 0),
    raw: c,
    dKv: n("d_kv"),
    buckets: n("relative_attention_num_buckets", 32),
    maxDistance: n("relative_attention_max_distance", 128),
    gated: proj === "gated-gelu",
    tied: c.tie_word_embeddings !== false,
    eps: n("layer_norm_epsilon", 1e-6),
  };
}

/** Load-time edit for tied T5: the d_model^-0.5 output scaling folded into the final decoder norm. */
export function t5Transform(cfg: T5Config): ((name: string, values: Float32Array) => void) | undefined {
  if (!cfg.tied) return undefined;
  const s = cfg.dModel ** -0.5;
  return (name, v) => {
    if (name === "dec.lnF.weight") for (let i = 0; i < v.length; i++) v[i]! *= s;
  };
}

/** T5 tensor names in one naming ("enc.l3.q.weight", "dec.l0.cq.weight", "enc.rel"…), or undefined. */
export function t5Name(name: string): string | undefined {
  if (name === "shared.weight" || name === "encoder.embed_tokens.weight" || name === "decoder.embed_tokens.weight") return "shared.weight";
  if (name === "lm_head.weight") return "lm_head.weight";
  let m = /^(encoder|decoder)\.final_layer_norm\.weight$/.exec(name);
  if (m) return `${m[1] === "encoder" ? "enc" : "dec"}.lnF.weight`;
  m = /^(encoder|decoder)\.block\.(\d+)\.layer\.(\d)\.(.+)\.weight$/.exec(name);
  if (!m) return undefined;
  const side = m[1] === "encoder" ? "enc" : "dec";
  const [l, sub, rest] = [m[2]!, Number(m[3]), m[4]!];
  const ffnLayer = side === "enc" ? 1 : 2;
  if (rest === "SelfAttention.relative_attention_bias") return `${side}.rel`;
  const attn = /^(SelfAttention|EncDecAttention)\.([qkvo])$/.exec(rest);
  if (attn) return `${side}.l${l}.${attn[1] === "EncDecAttention" ? "c" : ""}${attn[2]}.weight`;
  if (rest === "layer_norm") return `${side}.l${l}.${sub === 0 ? "ln1" : sub === ffnLayer ? "ln2" : "lnc"}.weight`;
  const ffn = /^DenseReluDense\.(wi|wi_0|wi_1|wo)$/.exec(rest);
  if (ffn) return `${side}.l${l}.${ffn[1]}.weight`;
  return undefined;
}

interface Pair {
  S: number;
  enc: { ctx: Ctx; dispatches: readonly Dispatch[]; step: Buf; ids: Buf; kc: Buf; vc: Buf };
  dec: { ctx: Ctx; dispatches: readonly Dispatch[]; logits: Tensor };
  crossK: Buf[];
  crossV: Buf[];
}

/**
 * T5: pre-RMSNorm blocks without biases, a learned relative position bias (one table per stack, in
 * its first layer, added to every self-attention score), unscaled attention, and a ReLU or gated
 * GELU feed-forward. Same encode-once / decode-a-step-at-a-time scheme as Seq2SeqModel.
 */
export class T5Model {
  private readonly pairs = new Map<number, Pair>();
  private readonly pool: ActivationPool;
  private readonly decStep: Buf;
  private readonly crossStep: Buf;
  private readonly tok: Buf;
  private readonly selfK: Buf[] = [];
  private readonly selfV: Buf[] = [];
  private current: Pair | undefined;
  private readonly sampler: DecoderSampler;
  readonly maxTarget: number;

  constructor(
    readonly backend: Backend,
    readonly cfg: T5Config,
    private readonly weights: Map<string, Tensor>,
    opts: { maxTarget?: number } = {},
  ) {
    this.maxTarget = Math.min(opts.maxTarget ?? 1024, 4096);
    this.pool = new ActivationPool(backend, "t5");
    this.decStep = backend.alloc(STEP_BYTES, "storage", "t5/step");
    this.crossStep = backend.alloc(STEP_BYTES, "storage", "t5/cross-step");
    backend.write(this.crossStep, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: 0x7fffffff }));
    this.tok = backend.alloc(16, "storage", "t5/token");
    this.sampler = new DecoderSampler(backend, cfg.vocab);
    const inner = cfg.decHeads * cfg.dKv;
    for (let l = 0; l < cfg.decLayers; l++) {
      this.selfK.push(backend.alloc(this.maxTarget * inner * 4, "storage", "t5/self-k"));
      this.selfV.push(backend.alloc(this.maxTarget * inner * 4, "storage", "t5/self-v"));
    }
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`T5 weight "${name}" was not loaded`);
    return t;
  }

  private rms(ctx: Ctx, x: Tensor, name: string): Tensor {
    return ops.rmsnorm(ctx, x, this.w(name), this.cfg.eps);
  }

  /** x += FFN(norm(x)). */
  private ffn(ctx: Ctx, x: Tensor, p: string): void {
    const h = this.rms(ctx, x, `${p}.ln2.weight`);
    let mid: Tensor;
    if (this.cfg.gated) mid = ops.siluMul(ctx, ops.linear(ctx, h, this.w(`${p}.wi_0.weight`)), ops.linear(ctx, h, this.w(`${p}.wi_1.weight`)), "gelu");
    else {
      mid = ops.linear(ctx, h, this.w(`${p}.wi.weight`));
      ops.act(ctx, mid, "relu");
    }
    ops.linear(ctx, mid, this.w(`${p}.wo.weight`), { residual: x });
  }

  private build(S: number): Pair {
    const c = this.cfg;
    const be = this.backend;
    const inner = c.encHeads * c.dKv;
    const step = be.alloc(STEP_BYTES, "storage", "t5/enc-step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: S }));
    const ectx = new Ctx(be, step, `t5/enc S${S}`, this.pool);
    const ids = ectx.buffer(S * 4, "t5/src");
    const kc = be.alloc(S * inner * 4, "storage", "t5/enc-k");
    const vc = be.alloc(S * inner * 4, "storage", "t5/enc-v");
    const x = ops.embed(ectx, this.w("shared.weight"), ids, S);
    const encBias = { table: this.w("enc.rel"), buckets: c.buckets, maxDistance: c.maxDistance, bidirectional: true };
    const eshape = { T: S, H: c.encHeads, Hkv: c.encHeads, D: c.dKv, maxSeq: S, kv: "f32" as const, scale: 1, relBias: encBias };
    for (let l = 0; l < c.encLayers; l++) {
      const p = `enc.l${l}`;
      const h = this.rms(ectx, x, `${p}.ln1.weight`);
      const q = ops.linear(ectx, h, this.w(`${p}.q.weight`));
      ops.kvWrite(ectx, ops.linear(ectx, h, this.w(`${p}.k.weight`)), ops.linear(ectx, h, this.w(`${p}.v.weight`)), kc, vc, S, S, "f32");
      ops.linear(ectx, ops.attention(ectx, q, kc, vc, eshape), this.w(`${p}.o.weight`), { residual: x });
      this.ffn(ectx, x, p);
    }
    const encOut = this.rms(ectx, x, "enc.lnF.weight");
    const crossK: Buf[] = [];
    const crossV: Buf[] = [];
    for (let l = 0; l < c.decLayers; l++) {
      const k = be.alloc(S * inner * 4, "storage", "t5/cross-k");
      const v = be.alloc(S * inner * 4, "storage", "t5/cross-v");
      ops.kvWrite(ectx, ops.linear(ectx, encOut, this.w(`dec.l${l}.ck.weight`)), ops.linear(ectx, encOut, this.w(`dec.l${l}.cv.weight`)), k, v, S, S, "f32");
      crossK.push(k);
      crossV.push(v);
    }

    const dctx = new Ctx(be, this.decStep, `t5/dec S${S}`, this.pool);
    const y = ops.embed(dctx, this.w("shared.weight"), this.tok, 1);
    const decBias = { table: this.w("dec.rel"), buckets: c.buckets, maxDistance: c.maxDistance, bidirectional: false };
    const self = { T: 1, H: c.decHeads, Hkv: c.decHeads, D: c.dKv, maxSeq: this.maxTarget, kv: "f32" as const, scale: 1, relBias: decBias };
    const cross = { T: 1, H: c.decHeads, Hkv: c.decHeads, D: c.dKv, maxSeq: S, kv: "f32" as const, scale: 1, step: this.crossStep };
    for (let l = 0; l < c.decLayers; l++) {
      const p = `dec.l${l}`;
      let h = this.rms(dctx, y, `${p}.ln1.weight`);
      const q = ops.linear(dctx, h, this.w(`${p}.q.weight`));
      ops.kvWrite(dctx, ops.linear(dctx, h, this.w(`${p}.k.weight`)), ops.linear(dctx, h, this.w(`${p}.v.weight`)), this.selfK[l]!, this.selfV[l]!, 1, this.maxTarget, "f32");
      ops.linear(dctx, ops.attention(dctx, q, this.selfK[l]!, this.selfV[l]!, self), this.w(`${p}.o.weight`), { residual: y });
      h = this.rms(dctx, y, `${p}.lnc.weight`);
      ops.linear(dctx, ops.attention(dctx, ops.linear(dctx, h, this.w(`${p}.cq.weight`)), crossK[l]!, crossV[l]!, cross), this.w(`${p}.co.weight`), { residual: y });
      this.ffn(dctx, y, p);
    }
    // Tied T5 rescales the decoder output by d_model^-0.5 before the shared embedding: folded into
    // dec.lnF.weight at load (see T5_TIED_SCALE).
    const out = this.rms(dctx, y, "dec.lnF.weight");
    const head = c.tied ? this.w("shared.weight") : this.w("lm_head.weight");
    const logits = ops.linear(dctx, out, head);
    ops.sample(dctx, logits, this.sampler.out, this.sampler.seen, this.sampler.allowed);
    return { S, enc: { ctx: ectx, dispatches: ectx.dispatches, step, ids, kc, vc }, dec: { ctx: dctx, dispatches: dctx.dispatches, logits }, crossK, crossV };
  }

  async encode(ids: readonly number[]): Promise<void> {
    const S = ids.length;
    if (!S) throw new Error("T5: empty source");
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
    this.backend.write(pair.enc.ids, 0, Uint32Array.from(ids));
    this.backend.submit(pair.enc.dispatches);
    this.current = pair;
  }

  async step(token: number, t: number): Promise<Float32Array> {
    const pair = this.current;
    if (!pair) throw new Error("T5: encode a source first");
    if (t >= this.maxTarget) throw new Error(`T5: target longer than ${this.maxTarget} tokens`);
    const be = this.backend;
    be.write(this.tok, 0, Uint32Array.of(token));
    be.write(this.decStep, 0, packStep({ pos0: t, lastRow: 0, seed: 0, temperature: 0 }));
    be.submit(pair.dec.dispatches);
    return new Float32Array(await be.readback(physical(pair.dec.logits.buf), 0, this.cfg.vocab * 4));
  }

  /** Feeds the token at position t and returns the greedy next token, chosen on the backend under `opts`. */
  async next(token: number, t: number, opts: StepOptions = {}): Promise<number> {
    const pair = this.current;
    if (!pair) throw new Error("T5: encode a source first");
    if (t >= this.maxTarget) throw new Error(`T5: target longer than ${this.maxTarget} tokens`);
    const be = this.backend;
    const rep = opts.repetitionPenalty ?? 1;
    this.sampler.prepare(opts.banned ?? [], rep !== 1 ? opts.seen : undefined);
    be.write(this.tok, 0, Uint32Array.of(token));
    be.write(this.decStep, 0, packStep({ pos0: t, lastRow: 0, seed: 0, temperature: 0, repetitionPenalty: rep }));
    be.submit(pair.dec.dispatches);
    return this.sampler.read();
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
    for (const b of [this.decStep, this.crossStep, this.tok, ...this.selfK, ...this.selfV]) this.backend.free(b);
  }
}
