import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** CLIP's text tower (CLIPTextConfig, with its defaults for fields a config leaves out). */
export interface ClipTextConfig {
  vocab: number;
  hidden: number;
  intermediate: number;
  layers: number;
  heads: number;
  maxPositions: number;
  eps: number;
  act: "gelu" | "quick_gelu";
  /** Pooling: the first `eosId` token, or (legacy configs with eos_token_id 2) the highest id. */
  eosId: number;
  projectionDim: number;
}

export function clipTextConfigFromHF(config: Record<string, unknown>): ClipTextConfig {
  const t = { ...((config.text_config as Record<string, unknown> | undefined) ?? {}) };
  const n = (k: string, d: number) => Number(t[k] ?? d);
  const act = String(t.hidden_act ?? "quick_gelu");
  if (act !== "gelu" && act !== "quick_gelu") throw new UnsupportedModelError(`CLIP text activation "${act}" is not supported`);
  return {
    vocab: n("vocab_size", 49408),
    hidden: n("hidden_size", 512),
    intermediate: n("intermediate_size", 2048),
    layers: n("num_hidden_layers", 12),
    heads: n("num_attention_heads", 8),
    maxPositions: n("max_position_embeddings", 77),
    eps: n("layer_norm_eps", 1e-5),
    act,
    eosId: n("eos_token_id", 2),
    projectionDim: Number(config.projection_dim ?? t.projection_dim ?? 512),
  };
}

/** Kiln's name for a CLIP text tensor, or undefined for anything else. */
export function clipTextName(name: string): string | undefined {
  let m = /^text_model\.encoder\.layers\.(\d+)\.(.+)\.(weight|bias)$/.exec(name);
  if (m) {
    const part: Record<string, string> = {
      "self_attn.q_proj": "q",
      "self_attn.k_proj": "k",
      "self_attn.v_proj": "v",
      "self_attn.out_proj": "o",
      "mlp.fc1": "up",
      "mlp.fc2": "down",
      layer_norm1: "ln1",
      layer_norm2: "ln2",
    };
    const p = part[m[2]!];
    return p ? `t${m[1]}.${p}.${m[3]}` : undefined;
  }
  m = /^text_model\.final_layer_norm\.(weight|bias)$/.exec(name);
  if (m) return `tlnF.${m[1]}`;
  if (name === "text_model.embeddings.token_embedding.weight") return "tok.weight";
  if (name === "text_model.embeddings.position_embedding.weight") return "tpos.weight";
  return undefined;
}

/** The position CLIP pools a token sequence at: the first eos, or for legacy configs the argmax id. */
export function clipPoolIndex(ids: readonly number[], eosId: number): number {
  if (eosId === 2) {
    let best = 0;
    for (let i = 1; i < ids.length; i++) if (ids[i]! > ids[best]!) best = i;
    return best;
  }
  const i = ids.indexOf(eosId);
  return i < 0 ? ids.length - 1 : i;
}

/**
 * CLIP's text transformer: token + position embeddings → pre-norm layers with causal attention →
 * final LayerNorm. A batch of sequences runs as one pass, each padded to the longest (padding comes
 * after a sequence's pooled token, which causal attention keeps it from touching). One plan per
 * batch shape (a few kept).
 */
export class ClipTextModel {
  /** Sequences per pass; longer batches run in chunks. */
  static readonly MAX_BATCH = 32;
  private readonly plans = new Map<string, { dispatches: readonly Dispatch[]; ids: Buf; pos: Buf; out: Tensor; ctx: Ctx }>();
  private readonly step: Buf;
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: ClipTextConfig,
    private readonly weights: Map<string, Tensor>,
  ) {
    this.step = backend.alloc(STEP_BYTES, "storage", "clip-text/step");
    backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
    this.pool = new ActivationPool(backend, "clip-text");
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`CLIP text weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string, residual?: Tensor): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
  }

  private build(B: number, T: number) {
    const c = this.cfg;
    const D = c.hidden / c.heads;
    const N = B * T;
    const ctx = new Ctx(this.backend, this.step, `clip-text/${B}x${T}`, this.pool);
    const ids = ctx.buffer(N * 4, "clip-text/ids");
    const pos = ctx.buffer(N * 4, "clip-text/pos");
    const x = ops.embed(ctx, this.w("tok.weight"), ids, N);
    ops.addInPlace(ctx, x, ops.embed(ctx, this.w("tpos.weight"), pos, N));
    const shape = { T, H: c.heads, D, scale: D ** -0.5 };
    for (let l = 0; l < c.layers; l++) {
      const p = `t${l}`;
      const h = ops.layernorm(ctx, x, this.w(`${p}.ln1.weight`), this.w(`${p}.ln1.bias`), c.eps);
      const a = ops.blockAttention(ctx, this.lin(ctx, h, `${p}.q`), this.lin(ctx, h, `${p}.k`), this.lin(ctx, h, `${p}.v`), shape);
      this.lin(ctx, a, `${p}.o`, x);
      const h2 = ops.layernorm(ctx, x, this.w(`${p}.ln2.weight`), this.w(`${p}.ln2.bias`), c.eps);
      const up = this.lin(ctx, h2, `${p}.up`);
      ops.act(ctx, up, c.act);
      this.lin(ctx, up, `${p}.down`, x);
    }
    const out = ops.layernorm(ctx, x, this.w("tlnF.weight"), this.w("tlnF.bias"), c.eps);
    return { dispatches: ctx.dispatches, ids, pos, out, ctx };
  }

  private plan(B: number, T: number) {
    const key = `${B}x${T}`;
    let plan = this.plans.get(key);
    if (!plan) {
      if (this.plans.size >= 6) {
        const [k, old] = this.plans.entries().next().value!;
        old.ctx.destroy();
        this.plans.delete(k);
      }
      plan = this.build(B, T);
      this.plans.set(key, plan);
    }
    return plan;
  }

  /** The final-normed hidden state [hidden] of each sequence at its row `rows[i]`, in one pass per chunk. */
  async forwardBatch(seqs: readonly (readonly number[])[], rows: readonly number[]): Promise<Float32Array[]> {
    const H = this.cfg.hidden;
    const res: Float32Array[] = [];
    for (let at = 0; at < seqs.length; at += ClipTextModel.MAX_BATCH) {
      const chunk = seqs.slice(at, at + ClipTextModel.MAX_BATCH);
      const B = chunk.length;
      const T = Math.max(...chunk.map((s) => s.length));
      if (!chunk.every((s) => s.length) || T > this.cfg.maxPositions) throw new Error(`CLIP text: sequences take 1 to ${this.cfg.maxPositions} tokens`);
      const plan = this.plan(B, T);
      const ids = new Uint32Array(B * T);
      const pos = new Uint32Array(B * T);
      chunk.forEach((s, b) => {
        ids.set(s, b * T);
        ids.fill(s[s.length - 1]!, b * T + s.length, (b + 1) * T);
        for (let t = 0; t < T; t++) pos[b * T + t] = t;
      });
      const be = this.backend;
      be.write(plan.ids, 0, ids);
      be.write(plan.pos, 0, pos);
      be.submit(plan.dispatches);
      const out = new Float32Array(await be.readback(physical(plan.out.buf), 0, B * T * H * 4));
      chunk.forEach((_, b) => {
        const r = (b * T + rows[at + b]!) * H;
        res.push(out.slice(r, r + H));
      });
    }
    return res;
  }

  /** The final-normed hidden state [hidden] at row `row` of the token sequence `ids`. */
  async forward(ids: readonly number[], row: number): Promise<Float32Array> {
    return (await this.forwardBatch([ids], [row]))[0]!;
  }

  destroy(): void {
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    for (const p of this.plans.values()) p.ctx.destroy();
    this.pool.destroy();
    this.backend.free(this.step);
  }
}
