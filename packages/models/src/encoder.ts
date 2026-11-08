import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

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
  for (const k of keys) if (typeof c[k] === "number") return c[k] as number;
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
    const T = ids.length;
    if (!T) throw new Error("encoder: empty input");
    if (T > this.maxLength) throw new Error(`encoder: ${T} tokens; the model takes at most ${this.maxLength}`);
    let plan = this.plans.get(T);
    if (!plan) {
      // A few lengths stay recorded; the oldest goes first.
      if (this.plans.size >= 8) {
        const [k, old] = this.plans.entries().next().value!;
        this.dropPlan(old);
        this.plans.delete(k);
      }
      plan = this.build(T);
      this.plans.set(T, plan);
    }
    const be = this.backend;
    be.write(plan.ids, 0, Uint32Array.from(ids));
    be.write(plan.pos, 0, Uint32Array.from({ length: T }, (_, i) => i + this.cfg.posOffset));
    if (plan.types) be.write(plan.types, 0, Uint32Array.from(typeIds ?? new Array<number>(T).fill(0)));
    be.submit(plan.dispatches);
    return new Float32Array(await be.readback(physical(plan.out.buf), 0, T * this.cfg.hidden * 4));
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
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    this.pool.destroy();
  }
}
