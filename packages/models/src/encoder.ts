import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError, configNumber } from "./config";

export type EncoderFamily = "bert" | "distilbert" | "roberta";

/** A BERT-style encoder (BERT, DistilBERT, RoBERTa, and models built on them: MiniLM, BGE, GTE…). */
export interface EncoderConfig {
  family: EncoderFamily;
  modelType: string;
  architectures: string[];
  hidden: number;
  layers: number;
  heads: number;
  intermediate: number;
  vocab: number;
  maxPositions: number;
  /** Rows of the token-type table (0: none, as DistilBERT). */
  typeVocab: number;
  /** Position of the first token: 0, or RoBERTa's padding_idx + 1. */
  posOffset: number;
  eps: number;
  /** "gelu" (erf) or "gelu_tanh". */
  act: "gelu" | "gelu_tanh";
  id2label: Record<number, string> | undefined;
  /** Multi-label classifiers (sigmoid per label) rather than softmax. */
  multiLabel: boolean;
}

const num = (c: Record<string, unknown>, ...keys: string[]): number => {
  for (const k of keys) if (typeof c[k] === "number") return configNumber(k, c[k]);
  throw new UnsupportedModelError(`config.json is missing ${keys.join(" / ")}`);
};

function activation(name: unknown): EncoderConfig["act"] {
  if (name === undefined || name === "gelu") return "gelu";
  if (name === "gelu_new" || name === "gelu_pytorch_tanh" || name === "gelu_fast") return "gelu_tanh";
  throw new UnsupportedModelError(`activation "${String(name)}" is not supported`);
}

/** The encoder config from a transformers config.json (model_type bert, distilbert or roberta). */
export function encoderConfigFromHF(c: Record<string, unknown>): EncoderConfig {
  const type = String(c.model_type);
  const id2label = c.id2label ? Object.fromEntries(Object.entries(c.id2label as Record<string, string>).map(([k, v]) => [Number(k), v])) : undefined;
  const base = {
    modelType: type,
    architectures: (c.architectures as string[] | undefined) ?? [],
    vocab: num(c, "vocab_size"),
    id2label,
    multiLabel: c.problem_type === "multi_label_classification",
  };
  if (type === "distilbert") {
    return {
      ...base,
      family: "distilbert",
      hidden: num(c, "dim"),
      layers: num(c, "n_layers"),
      heads: num(c, "n_heads"),
      intermediate: num(c, "hidden_dim"),
      maxPositions: num(c, "max_position_embeddings"),
      typeVocab: 0,
      posOffset: 0,
      eps: 1e-12,
      act: activation(c.activation),
    };
  }
  if (type === "bert" || type === "roberta") {
    if (c.position_embedding_type !== undefined && c.position_embedding_type !== "absolute") {
      throw new UnsupportedModelError(`position_embedding_type "${String(c.position_embedding_type)}" is not supported`);
    }
    const roberta = type === "roberta";
    return {
      ...base,
      family: roberta ? "roberta" : "bert",
      hidden: num(c, "hidden_size"),
      layers: num(c, "num_hidden_layers"),
      heads: num(c, "num_attention_heads"),
      intermediate: num(c, "intermediate_size"),
      maxPositions: num(c, "max_position_embeddings") - (roberta ? Number(c.pad_token_id ?? 1) + 1 : 0),
      typeVocab: Number(c.type_vocab_size ?? 2),
      posOffset: roberta ? Number(c.pad_token_id ?? 1) + 1 : 0,
      eps: Number(c.layer_norm_eps ?? 1e-12),
      act: activation(c.hidden_act),
    };
  }
  throw new UnsupportedModelError(`encoder model_type "${type}" is not supported (bert, distilbert and roberta are)`);
}

/**
 * The encoder body's tensors under one naming for every family ("emb.word", "l3.q.weight",
 * "l3.ln2.bias"…), or undefined for anything else (task heads, pooler, position_ids). Old
 * checkpoints' LayerNorm gamma / beta read as weight / bias.
 */
export function encoderName(name: string): string | undefined {
  const n = name.replace(/^(bert|roberta|distilbert)\./, "").replace(/\.gamma$/, ".weight").replace(/\.beta$/, ".bias");
  const emb: Record<string, string> = {
    "embeddings.word_embeddings.weight": "emb.word",
    "embeddings.position_embeddings.weight": "emb.pos",
    "embeddings.token_type_embeddings.weight": "emb.type",
    "embeddings.LayerNorm.weight": "emb.ln.weight",
    "embeddings.LayerNorm.bias": "emb.ln.bias",
  };
  if (emb[n]) return emb[n];
  let m = /^encoder\.layer\.(\d+)\.(.+)\.(weight|bias)$/.exec(n);
  if (m) {
    const part: Record<string, string> = {
      "attention.self.query": "q",
      "attention.self.key": "k",
      "attention.self.value": "v",
      "attention.output.dense": "o",
      "attention.output.LayerNorm": "ln1",
      "intermediate.dense": "up",
      "output.dense": "down",
      "output.LayerNorm": "ln2",
    };
    return part[m[2]!] ? `l${m[1]}.${part[m[2]!]}.${m[3]}` : undefined;
  }
  m = /^transformer\.layer\.(\d+)\.(.+)\.(weight|bias)$/.exec(n);
  if (m) {
    const part: Record<string, string> = {
      "attention.q_lin": "q",
      "attention.k_lin": "k",
      "attention.v_lin": "v",
      "attention.out_lin": "o",
      sa_layer_norm: "ln1",
      "ffn.lin1": "up",
      "ffn.lin2": "down",
      output_layer_norm: "ln2",
    };
    return part[m[2]!] ? `l${m[1]}.${part[m[2]!]}.${m[3]}` : undefined;
  }
  return undefined;
}

/** A masked-LM head's weights (decoder undefined: tied to the word embeddings). */
/** Rows of the masked-LM head per batch: [32, vocab] logits stay well under storage-binding limits. */
const MLM_ROWS = 32;

export interface MlmHead {
  dense: Float32Array;
  denseBias?: Float32Array;
  lnW: Float32Array;
  lnB: Float32Array;
  eps: number;
  /** Exact (erf) GELU, else tanh-approximated. */
  exact: boolean;
  decoder?: Float32Array;
  bias?: Float32Array;
}

/** Body matrices that may be stored quantized (embedding tables, norms and biases stay f32). */
export const isEncoderMatrix = (name: string): boolean => /^l\d+\.(q|k|v|o|up|down)\.weight$/.test(name);

interface Plan {
  ctx: Ctx;
  dispatches: readonly Dispatch[];
  step: Buf;
  ids: Buf;
  types: Buf | undefined;
  pos: Buf;
  kc: Buf;
  vc: Buf;
  out: Tensor;
}

/**
 * A post-LayerNorm transformer encoder: token + position (+ token type) embeddings and LayerNorm,
 * then per layer x = LN(x + attn(x)), x = LN(x + ffn(x)), with full bidirectional attention. Runs one
 * sequence at a time with no padding; a plan is recorded per sequence length and reused. Attention
 * is the text model's, over a 'cache' of one slot per token and a span covering all of them (as
 * the vision tower does).
 */
export class EncoderModel {
  private readonly plans = new Map<number, Plan>();
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: EncoderConfig,
    private readonly weights: Map<string, Tensor>,
  ) {
    if (cfg.hidden % cfg.heads) throw new UnsupportedModelError(`hidden size ${cfg.hidden} is not divisible by ${cfg.heads} heads`);
    this.pool = new ActivationPool(backend, "encoder");
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`encoder weight "${name}" was not loaded`);
    return t;
  }

  private build(T: number): Plan {
    const c = this.cfg;
    const be = this.backend;
    const D = c.hidden / c.heads;
    const step = be.alloc(STEP_BYTES, "storage", "encoder/step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: T }));
    const ctx = new Ctx(be, step, `encoder/T${T}`, this.pool);
    const ids = ctx.buffer(T * 4, "encoder/ids");
    const pos = ctx.buffer(T * 4, "encoder/pos");
    const types = c.typeVocab && this.weights.has("emb.type") ? ctx.buffer(T * 4, "encoder/types") : undefined;
    const kc = be.alloc(T * c.hidden * 4, "storage", "encoder/k");
    const vc = be.alloc(T * c.hidden * 4, "storage", "encoder/v");
    let x = ops.embed(ctx, this.w("emb.word"), ids, T);
    ops.addInPlace(ctx, x, ops.embed(ctx, this.w("emb.pos"), pos, T));
    if (types) ops.addInPlace(ctx, x, ops.embed(ctx, this.w("emb.type"), types, T));
    x = ops.layernorm(ctx, x, this.w("emb.ln.weight"), this.w("emb.ln.bias"), c.eps);
    const shape = { T, H: c.heads, Hkv: c.heads, D, maxSeq: T, kv: "f32" as const, scale: D ** -0.5, span: true };
    for (let l = 0; l < c.layers; l++) {
      const p = `l${l}.`;
      const lin = (h: Tensor, n: string, residual?: Tensor) => ops.linear(ctx, h, this.w(`${p}${n}.weight`), { bias: this.w(`${p}${n}.bias`), residual });
      const q = lin(x, "q");
      ops.kvWrite(ctx, lin(x, "k"), lin(x, "v"), kc, vc, T, T, "f32");
      lin(ops.attention(ctx, q, kc, vc, shape), "o", x);
      x = ops.layernorm(ctx, x, this.w(p + "ln1.weight"), this.w(p + "ln1.bias"), c.eps);
      const up = lin(x, "up");
      ops.gelu(ctx, up, c.act === "gelu");
      lin(up, "down", x);
      x = ops.layernorm(ctx, x, this.w(p + "ln2.weight"), this.w(p + "ln2.bias"), c.eps);
    }
    return { ctx, dispatches: ctx.dispatches, step, ids, types, pos, kc, vc, out: x };
  }

  /** Longest sequence the position table allows. */
  get maxLength(): number {
    return this.cfg.maxPositions;
  }

  /** Final hidden states [T, hidden] of one sequence (token ids and, for BERT pairs, segment ids). */
  async forward(ids: readonly number[], typeIds?: readonly number[]): Promise<Float32Array> {
    return (await this.forwardMany([{ ids, typeIds }]))[0]!;
  }

  /**
   * Final hidden states of several sequences. Every forward pass is queued before any result is read:
   * each one's output is copied aside as it finishes, so the GPU runs them back to back instead of
   * waiting on a readback between inputs (16 short sentences: one wait instead of sixteen).
   */
  async forwardMany(items: readonly { ids: readonly number[]; typeIds?: readonly number[] }[]): Promise<Float32Array[]> {
    const be = this.backend;
    const H = this.cfg.hidden;
    const outs: { buf: Buf; bytes: number }[] = [];
    try {
      for (const { ids, typeIds } of items) {
        const T = ids.length;
        if (!T) throw new Error("encoder: empty input");
        if (T > this.maxLength) throw new Error(`encoder: ${T} tokens; the model takes at most ${this.maxLength}`);
        let plan = this.plans.get(T);
        if (!plan) {
          plan = this.build(T);
          this.plans.set(T, plan);
        } else {
          // Most recently used last, for the trim below.
          this.plans.delete(T);
          this.plans.set(T, plan);
        }
        be.write(plan.ids, 0, Uint32Array.from(ids));
        be.write(plan.pos, 0, Uint32Array.from({ length: T }, (_, i) => i + this.cfg.posOffset));
        if (plan.types) be.write(plan.types, 0, Uint32Array.from(typeIds ?? new Array<number>(T).fill(0)));
        be.submit(plan.dispatches);
        const bytes = T * H * 4;
        const buf = be.alloc(bytes, "storage", "encoder/out");
        be.copy(physical(plan.out.buf), 0, buf, 0, bytes);
        outs.push({ buf, bytes });
      }
      // All copies are queued: map them together rather than waiting on each in turn.
      return await Promise.all(outs.map(async (o) => new Float32Array(await be.readback(o.buf, 0, o.bytes))));
    } finally {
      for (const o of outs) be.free(o.buf);
      // A few lengths stay recorded, trimmed only once nothing queued still uses them.
      while (this.plans.size > 8) {
        const [k, old] = this.plans.entries().next().value!;
        this.dropPlan(old);
        this.plans.delete(k);
      }
    }
  }

  /** The masked-LM head: its weights (uploaded on first use) and a recorded plan per row count. */
  private mlm: { step: Buf; owned: Tensor[]; dense: Tensor; denseBias?: Tensor; lnW: Tensor; lnB: Tensor; dec: Tensor; bias?: Tensor; plans: Map<number, { ctx: Ctx; x: Buf; logits: Tensor; dispatches: readonly Dispatch[] }> } | undefined;

  /**
   * A masked-LM head on hidden rows x [R, H], on the backend: dense → GELU → LayerNorm → vocabulary
   * logits through the decoder (the word embedding table when tied). Weights are uploaded on first
   * use. Rows run in batches of up to MLM_ROWS through one matmul each, read back once per batch.
   */
  async maskLogits(x: Float32Array, head: MlmHead): Promise<Float32Array> {
    const be = this.backend;
    const H = this.cfg.hidden;
    const R = x.length / H;
    if (!Number.isInteger(R) || R < 1) throw new Error(`maskLogits: ${x.length} values is not a whole number of ${H}-wide rows`);
    if (!this.mlm) {
      const owned: Tensor[] = [];
      const up = (v: Float32Array, shape: number[]): Tensor => {
        const buf = be.alloc(v.byteLength, "storage", "encoder/mlm");
        be.write(buf, 0, v);
        const t = { shape, dtype: "f32" as const, buf };
        owned.push(t);
        return t;
      };
      const V = head.bias?.length ?? (head.decoder ? head.decoder.length / H : this.cfg.vocab);
      const step = be.alloc(STEP_BYTES, "storage", "encoder/mlm-step");
      be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
      this.mlm = {
        step,
        owned,
        dense: up(head.dense, [H, H]),
        denseBias: head.denseBias ? up(head.denseBias, [H]) : undefined,
        lnW: up(head.lnW, [H]),
        lnB: up(head.lnB, [H]),
        dec: head.decoder ? up(head.decoder, [V, H]) : this.w("emb.word"),
        bias: head.bias ? up(head.bias, [V]) : undefined,
        plans: new Map(),
      };
    }
    const m = this.mlm;
    const V = m.dec.shape[0]!;
    const out = new Float32Array(R * V);
    for (let r0 = 0; r0 < R; r0 += MLM_ROWS) {
      const r = Math.min(MLM_ROWS, R - r0);
      // Row counts round up to a power of two, so a few plans serve every length.
      const P = Math.min(MLM_ROWS, 2 ** Math.ceil(Math.log2(r)));
      let plan = m.plans.get(P);
      if (!plan) {
        const ctx = new Ctx(be, m.step, `encoder/mlm${P}`, this.pool);
        const xb = ctx.buffer(P * H * 4, "encoder/mlm-x");
        const h = ops.linear(ctx, { shape: [P, H], dtype: "f32", buf: xb }, m.dense, { bias: m.denseBias });
        ops.gelu(ctx, h, head.exact);
        const n = ops.layernorm(ctx, h, m.lnW, m.lnB, head.eps);
        const logits = ops.linear(ctx, n, m.dec, { bias: m.bias });
        plan = { ctx, x: xb, logits, dispatches: ctx.dispatches };
        m.plans.set(P, plan);
      }
      const rows = new Float32Array(P * H);
      rows.set(x.subarray(r0 * H, (r0 + r) * H));
      be.write(plan.x, 0, rows);
      be.submit(plan.dispatches);
      out.set(new Float32Array(await be.readback(physical(plan.logits.buf), 0, r * V * 4)), r0 * V);
    }
    return out;
  }

  private dropPlan(p: Plan): void {
    p.ctx.destroy();
    for (const b of [p.step, p.kc, p.vc]) this.backend.free(b);
  }

  /** GPU bytes the weights hold. */
  get weightBytes(): number {
    let n = 0;
    for (const t of this.weights.values()) n += t.buf.size + (t.scales?.size ?? 0) + (t.mins?.size ?? 0);
    return n;
  }

  destroy(): void {
    for (const p of this.plans.values()) this.dropPlan(p);
    this.plans.clear();
    if (this.mlm) {
      for (const p of this.mlm.plans.values()) p.ctx.destroy();
      this.backend.free(this.mlm.step);
      for (const t of this.mlm.owned) this.backend.free(t.buf);
      this.mlm = undefined;
    }
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    this.pool.destroy();
  }
}
