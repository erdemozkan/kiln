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
 * final LayerNorm. forward() returns the normalized hidden state of one row (the pooled token).
 * One plan per sequence length (a few kept).
 */
export class ClipTextModel {
  private readonly plans = new Map<number, { dispatches: readonly Dispatch[]; ids: Buf; pos: Buf; out: Tensor; ctx: Ctx }>();
  private readonly step: Buf;
  private readonly kc: Buf;
  private readonly vc: Buf;
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: ClipTextConfig,
    private readonly weights: Map<string, Tensor>,
  ) {
    this.step = backend.alloc(STEP_BYTES, "storage", "clip-text/step");
    // No span: every row sees keys up to its own position (causal).
    backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
    this.kc = backend.alloc(cfg.maxPositions * cfg.hidden * 4, "storage", "clip-text/k");
    this.vc = backend.alloc(cfg.maxPositions * cfg.hidden * 4, "storage", "clip-text/v");
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

  private build(T: number) {
    const c = this.cfg;
    const D = c.hidden / c.heads;
    const ctx = new Ctx(this.backend, this.step, `clip-text/T${T}`, this.pool);
    const ids = ctx.buffer(T * 4, "clip-text/ids");
    const pos = ctx.buffer(T * 4, "clip-text/pos");
    const x = ops.embed(ctx, this.w("tok.weight"), ids, T);
    ops.addInPlace(ctx, x, ops.embed(ctx, this.w("tpos.weight"), pos, T));
    const shape = { T, H: c.heads, Hkv: c.heads, D, maxSeq: c.maxPositions, kv: "f32" as const, scale: D ** -0.5 };
    for (let l = 0; l < c.layers; l++) {
      const p = `t${l}`;
      const h = ops.layernorm(ctx, x, this.w(`${p}.ln1.weight`), this.w(`${p}.ln1.bias`), c.eps);
      const q = this.lin(ctx, h, `${p}.q`);
      ops.kvWrite(ctx, this.lin(ctx, h, `${p}.k`), this.lin(ctx, h, `${p}.v`), this.kc, this.vc, T, c.maxPositions, "f32");
      this.lin(ctx, ops.attention(ctx, q, this.kc, this.vc, shape), `${p}.o`, x);
      const h2 = ops.layernorm(ctx, x, this.w(`${p}.ln2.weight`), this.w(`${p}.ln2.bias`), c.eps);
      const up = this.lin(ctx, h2, `${p}.up`);
      ops.act(ctx, up, c.act);
      this.lin(ctx, up, `${p}.down`, x);
    }
    const out = ops.layernorm(ctx, x, this.w("tlnF.weight"), this.w("tlnF.bias"), c.eps);
    return { dispatches: ctx.dispatches, ids, pos, out, ctx };
  }

  /** The final-normed hidden state [hidden] at row `row` of the token sequence `ids`. */
  async forward(ids: readonly number[], row: number): Promise<Float32Array> {
    const T = ids.length;
    if (!T || T > this.cfg.maxPositions) throw new Error(`CLIP text: ${T} tokens; the model takes 1 to ${this.cfg.maxPositions}`);
    let plan = this.plans.get(T);
    if (!plan) {
      if (this.plans.size >= 6) {
        const [k, old] = this.plans.entries().next().value!;
        old.ctx.destroy();
        this.plans.delete(k);
      }
      plan = this.build(T);
      this.plans.set(T, plan);
    }
    const be = this.backend;
    be.write(plan.ids, 0, Uint32Array.from(ids));
    be.write(plan.pos, 0, Uint32Array.from({ length: T }, (_, i) => i));
    be.submit(plan.dispatches);
    const H = this.cfg.hidden;
    return new Float32Array(await be.readback(physical(plan.out.buf), row * H * 4, H * 4));
  }

  destroy(): void {
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    for (const p of this.plans.values()) p.ctx.destroy();
    this.pool.destroy();
    for (const b of [this.step, this.kc, this.vc]) this.backend.free(b);
  }
}
