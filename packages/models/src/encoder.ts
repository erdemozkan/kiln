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

const align = (n: number) => Math.ceil(n / BATCH_ALIGN) * BATCH_ALIGN;

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
  /** A padded batch's sequence lengths (B > 1 only). */
  lens: Buf | undefined;
  out: Tensor;
}

/** Rows a padded batch may hold, and the share of them that must be real tokens. */
const BATCH_ROWS = 2048;
const BATCH_FILL = 0.7;
/** Batch rows per sequence round up to this, so nearby lengths share a plan. */
const BATCH_ALIGN = 8;

/**
 * A post-LayerNorm transformer encoder: token + position (+ token type) embeddings and LayerNorm,
 * then per layer x = LN(x + attn(x)), x = LN(x + ffn(x)), with full bidirectional attention. Runs one
 * sequence at a time with no padding; a plan is recorded per sequence length and reused. Attention
 * is the text model's, over a 'cache' of one slot per token and a span covering all of them (as
 * the vision tower does).
 */
export class EncoderModel {
  private readonly plans = new Map<string, Plan>();
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

  /** A plan for one sequence of L tokens (B = 1), or a padded batch of B sequences of L rows each. */
  private build(B: number, L: number): Plan {
    const c = this.cfg;
    const be = this.backend;
    const D = c.hidden / c.heads;
    const T = B * L;
    const step = be.alloc(STEP_BYTES, "storage", "encoder/step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: T }));
    const ctx = new Ctx(be, step, B > 1 ? `encoder/B${B}xL${L}` : `encoder/T${T}`, this.pool);
    const lens = B > 1 ? be.alloc(B * 4, "storage", "encoder/lens") : undefined;
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
      const att = lens ? ops.attentionBatched(ctx, q, kc, vc, { B, L, H: c.heads, D, scale: D ** -0.5, lens }) : ops.attention(ctx, q, kc, vc, shape);
      lin(att, "o", x);
      x = ops.layernorm(ctx, x, this.w(p + "ln1.weight"), this.w(p + "ln1.bias"), c.eps);
      const up = lin(x, "up");
      ops.gelu(ctx, up, c.act === "gelu");
      lin(up, "down", x);
      x = ops.layernorm(ctx, x, this.w(p + "ln2.weight"), this.w(p + "ln2.bias"), c.eps);
    }
    return { ctx, dispatches: ctx.dispatches, step, ids, types, pos, kc, vc, lens, out: x };
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
   * waiting on a readback between inputs. On the GPU, inputs of similar length also run together as
   * one padded batch (16 short sentences: one pass of 16 × L rows instead of 16 passes; every matmul
   * then has 16 times the rows), each sequence's attention seeing only its own tokens.
   */
  async forwardMany(items: readonly { ids: readonly number[]; typeIds?: readonly number[] }[]): Promise<Float32Array[]> {
    const be = this.backend;
    const H = this.cfg.hidden;
    const outs: { buf: Buf; bytes: number }[] = new Array(items.length);
    try {
      for (const group of this.batches(items)) {
        if (group.length === 1) {
          const i = group[0]!;
          const plan = this.run(items[i]!.ids, items[i]!.typeIds);
          const bytes = items[i]!.ids.length * H * 4;
          const buf = be.alloc(bytes, "storage", "encoder/out");
          be.copy(physical(plan.out.buf), 0, buf, 0, bytes);
          outs[i] = { buf, bytes };
          continue;
        }
        const { plan, L } = this.runBatch(group.map((i) => items[i]!));
        group.forEach((i, b) => {
          const bytes = items[i]!.ids.length * H * 4;
          const buf = be.alloc(bytes, "storage", "encoder/out");
          be.copy(physical(plan.out.buf), b * L * H * 4, buf, 0, bytes);
          outs[i] = { buf, bytes };
        });
      }
      // All copies are queued: map them together rather than waiting on each in turn.
      return await Promise.all(outs.map(async (o) => new Float32Array(await be.readback(o.buf, 0, o.bytes))));
    } finally {
      for (const o of outs) if (o) be.free(o.buf);
      this.trim();
    }
  }

  /**
   * Input indices grouped for padded batches: by length, each group as long as its padded rows stay
   * within BATCH_ROWS and at least BATCH_FILL of them are real tokens. Groups of one run alone, as
   * does everything where the batched attention is not available (the CPU backend).
   */
  private batches(items: readonly { ids: readonly number[] }[]): number[][] {
    const D = this.cfg.hidden / this.cfg.heads;
    const order = items.map((_, i) => i);
    if (items.length < 2 || !ops.attentionBatchable(this.backend, D)) return order.map((i) => [i]);
    order.sort((a, b) => items[a]!.ids.length - items[b]!.ids.length);
    const groups: number[][] = [];
    let g: number[] = [];
    let real = 0;
    for (const i of order) {
      const n = items[i]!.ids.length;
      const L = align(n);
      if (g.length && ((g.length + 1) * L > BATCH_ROWS || real + n < BATCH_FILL * (g.length + 1) * L)) {
        groups.push(g);
        g = [];
        real = 0;
      }
      g.push(i);
      real += n;
    }
    if (g.length) groups.push(g);
    return groups;
  }

  /** Writes a group's inputs into a padded batch and queues its forward pass. */
  private runBatch(group: readonly { ids: readonly number[]; typeIds?: readonly number[] }[]): { plan: Plan; L: number } {
    const be = this.backend;
    const B = group.length;
    const L = align(Math.max(...group.map((x) => x.ids.length)));
    for (const x of group) if (x.ids.length > this.maxLength) throw new Error(`encoder: ${x.ids.length} tokens; the model takes at most ${this.maxLength}`);
    const plan = this.plan(B, L);
    const ids = new Uint32Array(B * L);
    const pos = new Uint32Array(B * L);
    const types = new Uint32Array(B * L);
    const lens = new Uint32Array(B);
    group.forEach((x, b) => {
      if (!x.ids.length) throw new Error("encoder: empty input");
      ids.set(x.ids, b * L);
      if (x.typeIds) types.set(x.typeIds, b * L);
      lens[b] = x.ids.length;
      // Padding rows take the last real position, so position tables shorter than L still serve.
      for (let t = 0; t < L; t++) pos[b * L + t] = Math.min(t, x.ids.length - 1) + this.cfg.posOffset;
    });
    be.write(plan.ids, 0, ids);
    be.write(plan.pos, 0, pos);
    if (plan.types) be.write(plan.types, 0, types);
    be.write(plan.lens!, 0, lens);
    be.submit(plan.dispatches);
    return { plan, L };
  }

  /** The recorded plan for B × L, built on first use; most recently used last, for the trim. */
  private plan(B: number, L: number): Plan {
    const key = B > 1 ? `${B}x${L}` : `${L}`;
    let plan = this.plans.get(key);
    if (!plan) plan = this.build(B, L);
    else this.plans.delete(key);
    this.plans.set(key, plan);
    return plan;
  }

  /** Writes one sequence's inputs and queues its forward pass; the plan's `out` holds the hidden states once it runs. */
  private run(ids: readonly number[], typeIds?: readonly number[]): Plan {
    const be = this.backend;
    const T = ids.length;
    if (!T) throw new Error("encoder: empty input");
    if (T > this.maxLength) throw new Error(`encoder: ${T} tokens; the model takes at most ${this.maxLength}`);
    const plan = this.plan(1, T);
    be.write(plan.ids, 0, Uint32Array.from(ids));
    be.write(plan.pos, 0, Uint32Array.from({ length: T }, (_, i) => i + this.cfg.posOffset));
    if (plan.types) be.write(plan.types, 0, Uint32Array.from(typeIds ?? new Array<number>(T).fill(0)));
    be.submit(plan.dispatches);
    return plan;
  }

  /** A few lengths stay recorded, trimmed only once nothing queued still uses them. */
  private trim(): void {
    while (this.plans.size > 8) {
      const [k, old] = this.plans.entries().next().value!;
      this.dropPlan(old);
      this.plans.delete(k);
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
    const V = this.mlmHead(head).dec.shape[0]!;
    const out = new Float32Array(R * V);
    for (let r0 = 0; r0 < R; r0 += MLM_ROWS) {
      const r = Math.min(MLM_ROWS, R - r0);
      const plan = this.mlmPlan(r, head);
      const rows = new Float32Array(plan.P * H);
      rows.set(x.subarray(r0 * H, (r0 + r) * H));
      be.write(plan.x, 0, rows);
      be.submit(plan.dispatches);
      out.set(new Float32Array(await be.readback(physical(plan.logits.buf), 0, r * V * 4)), r0 * V);
    }
    return out;
  }

  /**
   * Masked-LM logits [rows, vocab] at chosen rows of each sequence, all on the backend: the body runs,
   * its chosen hidden rows are copied straight into the head's input, and only the logits come back,
   * read once for every input together (fill-mask: one wait instead of two, and no hidden states read).
   */
  async maskedLogits(items: readonly { ids: readonly number[]; typeIds?: readonly number[]; rows: readonly number[] }[], head: MlmHead): Promise<Float32Array[]> {
    const be = this.backend;
    const H = this.cfg.hidden;
    const V = this.mlmHead(head).dec.shape[0]!;
    const outs: { buf: Buf; bytes: number }[] = [];
    try {
      for (const { ids, typeIds, rows } of items) {
        for (const r of rows) if (!Number.isInteger(r) || r < 0 || r >= ids.length) throw new Error(`maskedLogits: row ${r} is outside a ${ids.length}-token input`);
        const body = this.run(ids, typeIds);
        const bytes = rows.length * V * 4;
        const buf = be.alloc(Math.max(bytes, 4), "storage", "encoder/mlm-out");
        outs.push({ buf, bytes });
        for (let r0 = 0; r0 < rows.length; r0 += MLM_ROWS) {
          const r = Math.min(MLM_ROWS, rows.length - r0);
          const plan = this.mlmPlan(r, head);
          for (let i = 0; i < r; i++) be.copy(physical(body.out.buf), rows[r0 + i]! * H * 4, plan.x, i * H * 4, H * 4);
          be.submit(plan.dispatches);
          be.copy(physical(plan.logits.buf), 0, buf, r0 * V * 4, r * V * 4);
        }
      }
      return await Promise.all(outs.map(async (o) => (o.bytes ? new Float32Array(await be.readback(o.buf, 0, o.bytes)) : new Float32Array(0))));
    } finally {
      for (const o of outs) be.free(o.buf);
      this.trim();
    }
  }

  /** The head's weights, uploaded on first use. */
  private mlmHead(head: MlmHead): NonNullable<EncoderModel["mlm"]> {
    if (this.mlm) return this.mlm;
    const be = this.backend;
    const H = this.cfg.hidden;
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
    return this.mlm;
  }

  /** The head's plan for r rows (row counts round up to a power of two, so a few plans serve every length). */
  private mlmPlan(r: number, head: MlmHead): { P: number; x: Buf; logits: Tensor; dispatches: readonly Dispatch[] } {
    const m = this.mlmHead(head);
    const H = this.cfg.hidden;
    const P = Math.min(MLM_ROWS, 2 ** Math.ceil(Math.log2(r)));
    let plan = m.plans.get(P);
    if (!plan) {
      const ctx = new Ctx(this.backend, m.step, `encoder/mlm${P}`, this.pool);
      const xb = ctx.buffer(P * H * 4, "encoder/mlm-x");
      const h = ops.linear(ctx, { shape: [P, H], dtype: "f32", buf: xb }, m.dense, { bias: m.denseBias });
      ops.gelu(ctx, h, head.exact);
      const n = ops.layernorm(ctx, h, m.lnW, m.lnB, head.eps);
      const logits = ops.linear(ctx, n, m.dec, { bias: m.bias });
      plan = { ctx, x: xb, logits, dispatches: ctx.dispatches };
      m.plans.set(P, plan);
    }
    return { P, ...plan };
  }

  private dropPlan(p: Plan): void {
    p.ctx.destroy();
    for (const b of [p.step, p.kc, p.vc, p.lens]) if (b) this.backend.free(b);
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
