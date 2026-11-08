import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops, type Activation } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** A wav2vec2-style speech encoder (wav2vec2, HuBERT): a convolutional front end on raw 16 kHz samples, then a transformer. */
export interface Wav2Vec2Config {
  modelType: string;
  architectures: string[];
  hidden: number;
  layers: number;
  heads: number;
  intermediate: number;
  eps: number;
  act: Activation;
  /** The front end: one entry per conv layer. */
  conv: { dim: number; kernel: number; stride: number }[];
  convBias: boolean;
  convAct: Activation;
  /** "group": GroupNorm after the first conv layer only; "layer": LayerNorm after every conv layer. */
  featNorm: "group" | "layer";
  /** Pre-LayerNorm transformer layers (do_stable_layer_norm), else post-LayerNorm. */
  stable: boolean;
  posKernel: number;
  posGroups: number;
  vocab: number;
  id2label: Record<number, string> | undefined;
  /** Sequence classifiers average every layer's output with learned weights. */
  weightedLayerSum: boolean;
  raw: Record<string, unknown>;
}

const num = (c: Record<string, unknown>, ...keys: string[]): number => {
  for (const k of keys) if (typeof c[k] === "number") return c[k] as number;
  throw new UnsupportedModelError(`config.json is missing ${keys.join(" / ")}`);
};

function activation(name: unknown): Activation {
  if (name === undefined || name === "gelu") return "gelu";
  if (name === "gelu_new" || name === "gelu_pytorch_tanh" || name === "gelu_fast") return "gelu_tanh";
  if (name === "relu" || name === "silu") return name;
  if (name === "swish") return "silu";
  throw new UnsupportedModelError(`activation "${String(name)}" is not supported`);
}

export const WAV2VEC2_TYPES = new Set(["wav2vec2", "hubert"]);

/** The config from a transformers config.json (model_type wav2vec2 or hubert). */
export function wav2vec2ConfigFromHF(c: Record<string, unknown>): Wav2Vec2Config {
  const type = String(c.model_type);
  if (!WAV2VEC2_TYPES.has(type)) throw new UnsupportedModelError(`speech model_type "${type}" is not supported (wav2vec2 and hubert are)`);
  const dims = c.conv_dim as number[];
  const kernels = c.conv_kernel as number[];
  const strides = c.conv_stride as number[];
  if (!Array.isArray(dims) || dims.length !== kernels?.length || dims.length !== strides?.length) throw new UnsupportedModelError("config.json has no conv_dim / conv_kernel / conv_stride");
  const norm = String(c.feat_extract_norm ?? "group");
  if (norm !== "group" && norm !== "layer") throw new UnsupportedModelError(`feat_extract_norm "${norm}" is not supported`);
  if (c.position_embeddings_type !== undefined && c.position_embeddings_type !== "conv") throw new UnsupportedModelError(`position_embeddings_type "${String(c.position_embeddings_type)}" is not supported`);
  if (c.add_adapter) throw new UnsupportedModelError("wav2vec2 adapters are not supported");
  const id2label = c.id2label ? Object.fromEntries(Object.entries(c.id2label as Record<string, string>).map(([k, v]) => [Number(k), v])) : undefined;
  return {
    modelType: type,
    architectures: (c.architectures as string[] | undefined) ?? [],
    hidden: num(c, "hidden_size"),
    layers: num(c, "num_hidden_layers"),
    heads: num(c, "num_attention_heads"),
    intermediate: num(c, "intermediate_size"),
    eps: Number(c.layer_norm_eps ?? 1e-5),
    act: activation(c.hidden_act),
    conv: dims.map((dim, i) => ({ dim, kernel: kernels[i]!, stride: strides[i]! })),
    convBias: !!c.conv_bias,
    convAct: activation(c.feat_extract_activation),
    featNorm: norm,
    stable: !!c.do_stable_layer_norm,
    posKernel: num(c, "num_conv_pos_embeddings"),
    posGroups: num(c, "num_conv_pos_embedding_groups"),
    vocab: Number(c.vocab_size ?? 0),
    id2label,
    weightedLayerSum: !!c.use_weighted_layer_sum,
    raw: c,
  };
}

/**
 * The backend tensors under one naming ("fe.0.norm.weight", "l3.q.weight", "head.weight"…), or
 * undefined for everything loaded separately or not at all: the conv weights (rearranged for
 * im2col by the loader), the positional conv's weight-normed parts, the classifier and the
 * pre-training heads.
 */
export function wav2vec2Name(name: string): string | undefined {
  const n = name.replace(/^(wav2vec2|hubert)\./, "");
  let m = /^feature_extractor\.conv_layers\.(\d+)\.(conv|layer_norm)\.(weight|bias)$/.exec(n);
  if (m) return m[2] === "conv" ? (m[3] === "bias" ? `fe.${m[1]}.bias` : undefined) : `fe.${m[1]}.norm.${m[3]}`;
  m = /^feature_projection\.(layer_norm|projection)\.(weight|bias)$/.exec(n);
  if (m) return m[1] === "layer_norm" ? `proj.ln.${m[2]}` : `proj.${m[2]}`;
  m = /^encoder\.layer_norm\.(weight|bias)$/.exec(n);
  if (m) return `enc.ln.${m[1]}`;
  if (n === "encoder.pos_conv_embed.conv.bias") return "pos.bias";
  m = /^encoder\.layers\.(\d+)\.(.+)\.(weight|bias)$/.exec(n);
  if (m) {
    const part: Record<string, string> = {
      "attention.q_proj": "q",
      "attention.k_proj": "k",
      "attention.v_proj": "v",
      "attention.out_proj": "o",
      layer_norm: "ln1",
      "feed_forward.intermediate_dense": "up",
      "feed_forward.output_dense": "down",
      final_layer_norm: "ln2",
    };
    return part[m[2]!] ? `l${m[1]}.${part[m[2]!]}.${m[3]}` : undefined;
  }
  m = /^lm_head\.(weight|bias)$/.exec(n);
  if (m) return `head.${m[1]}`;
  return undefined;
}

/** Body matrices that may be stored quantized. */
export const isWav2Vec2Matrix = (name: string): boolean => /^l\d+\.(q|k|v|o|up|down)\.weight$/.test(name);

/** Samples the front end turns into `frames` frames, or the frames from `samples` samples. */
export function wav2vec2Frames(cfg: Pick<Wav2Vec2Config, "conv">, samples: number): number {
  let t = samples;
  for (const l of cfg.conv) t = Math.floor((t - l.kernel) / l.stride) + 1;
  return t;
}

export interface Wav2Vec2Options {
  /** What the plan ends in: CTC logits [T, vocab] through lm_head, or the hidden states [T, hidden] a classifier pools. */
  head: "ctc" | "hidden";
  /** Softmaxed layer weights (embedding output first) when the hidden states are a weighted sum of every layer's. */
  layerWeights?: number[];
}

interface Plan {
  ctx: Ctx;
  dispatches: readonly Dispatch[];
  step: Buf;
  input: Buf;
  kc: Buf;
  vc: Buf;
  out: Tensor;
  frames: number;
}

/**
 * wav2vec2 / HuBERT on the backend. The conv front end runs as im2col + matmul per layer (GroupNorm
 * or LayerNorm, GELU), then feature projection, the grouped positional convolution, and post- or
 * pre-LayerNorm transformer layers with full attention. A plan is recorded per input length (the
 * models take no padding) and the last few are kept.
 */
export class Wav2Vec2Model {
  private readonly plans = new Map<number, Plan>();
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: Wav2Vec2Config,
    private readonly weights: Map<string, Tensor>,
    readonly opts: Wav2Vec2Options,
  ) {
    if (cfg.hidden % cfg.heads) throw new UnsupportedModelError(`hidden size ${cfg.hidden} is not divisible by ${cfg.heads} heads`);
    if (opts.layerWeights && opts.layerWeights.length !== cfg.layers + 1) throw new Error(`wav2vec2: ${opts.layerWeights.length} layer weights for ${cfg.layers} layers`);
    this.pool = new ActivationPool(backend, "wav2vec2");
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`wav2vec2 weight "${name}" was not loaded`);
    return t;
  }

  private opt(name: string): Tensor | undefined {
    return this.weights.get(name);
  }

  /** Fewest samples the front end accepts (one output frame). */
  get minSamples(): number {
    let n = 1;
    for (let i = this.cfg.conv.length - 1; i >= 0; i--) n = (n - 1) * this.cfg.conv[i]!.stride + this.cfg.conv[i]!.kernel;
    return n;
  }

  private build(N: number): Plan {
    const c = this.cfg;
    const be = this.backend;
    const T = wav2vec2Frames(c, N);
    const step = be.alloc(STEP_BYTES, "storage", "wav2vec2/step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: T }));
    const ctx = new Ctx(be, step, `wav2vec2/N${N}`, this.pool);
    const input = ctx.buffer(N * 4, "wav2vec2/samples");
    let x: Tensor = { shape: [N, 1], dtype: "f32", buf: input };
    c.conv.forEach((l, i) => {
      const y = ops.linear(ctx, ops.im2col1d(ctx, x, l.kernel, l.stride, 0), this.w(`fe.${i}.weight`), { bias: this.opt(`fe.${i}.bias`) });
      let h = y;
      if (c.featNorm === "layer") h = ops.layernorm(ctx, y, this.w(`fe.${i}.norm.weight`), this.w(`fe.${i}.norm.bias`), 1e-5);
      else if (i === 0) h = ops.channelNorm(ctx, y, this.w("fe.0.norm.weight"), this.w("fe.0.norm.bias"), 1e-5);
      ops.act(ctx, h, c.convAct);
      x = h;
    });
    if (this.opt("proj.ln.weight")) x = ops.layernorm(ctx, x, this.w("proj.ln.weight"), this.w("proj.ln.bias"), c.eps);
    x = ops.linear(ctx, x, this.w("proj.weight"), { bias: this.w("proj.bias") });
    // Positional convolution (padding k / 2; an even kernel's extra last frame is dropped), GELU, added.
    const pos = ops.groupConv1d(ctx, x, this.w("pos.weight"), this.w("pos.bias"), { k: c.posKernel, pad: Math.floor(c.posKernel / 2), groups: c.posGroups, tout: T });
    ops.gelu(ctx, pos, true);
    ops.addInPlace(ctx, x, pos);
    const ln = (h: Tensor, n: string) => ops.layernorm(ctx, h, this.w(`${n}.weight`), this.w(`${n}.bias`), c.eps);
    if (!c.stable) x = ln(x, "enc.ln");
    const lw = this.opts.layerWeights;
    let acc: Tensor | undefined;
    if (lw) {
      acc = ctx.alloc([T, c.hidden]);
      ops.scaleAdd(ctx, acc, x, lw[0]!, false);
    }
    const kc = be.alloc(T * c.hidden * 4, "storage", "wav2vec2/k");
    const vc = be.alloc(T * c.hidden * 4, "storage", "wav2vec2/v");
    const D = c.hidden / c.heads;
    const shape = { T, H: c.heads, Hkv: c.heads, D, maxSeq: T, kv: "f32" as const, scale: D ** -0.5, span: true };
    for (let l = 0; l < c.layers; l++) {
      const p = `l${l}.`;
      const lin = (h: Tensor, n: string, residual?: Tensor) => ops.linear(ctx, h, this.w(`${p}${n}.weight`), { bias: this.w(`${p}${n}.bias`), residual });
      const attn = (h: Tensor) => {
        const q = lin(h, "q");
        ops.kvWrite(ctx, lin(h, "k"), lin(h, "v"), kc, vc, T, T, "f32");
        return ops.attention(ctx, q, kc, vc, shape);
      };
      const ffn = (h: Tensor) => {
        const up = lin(h, "up");
        ops.act(ctx, up, c.act);
        return up;
      };
      if (c.stable) {
        lin(attn(ln(x, p + "ln1")), "o", x);
        lin(ffn(ln(x, p + "ln2")), "down", x);
      } else {
        lin(attn(x), "o", x);
        x = ln(x, p + "ln1");
        lin(ffn(x), "down", x);
        x = ln(x, p + "ln2");
      }
      if (c.stable && l === c.layers - 1) x = ln(x, "enc.ln");
      if (acc) ops.scaleAdd(ctx, acc, x, lw![l + 1]!, true);
    }
    const out = this.opts.head === "ctc" ? ops.linear(ctx, x, this.w("head.weight"), { bias: this.w("head.bias") }) : (acc ?? x);
    return { ctx, dispatches: ctx.dispatches, step, input, kc, vc, out, frames: T };
  }

  /** The plan's output for raw samples: { data, frames } with data [frames, vocab] (CTC) or [frames, hidden]. */
  async forward(samples: Float32Array): Promise<{ data: Float32Array; frames: number }> {
    const N = samples.length;
    if (N < this.minSamples) throw new Error(`wav2vec2: ${N} samples; the model needs at least ${this.minSamples}`);
    let plan = this.plans.get(N);
    if (plan) this.plans.delete(N);
    else {
      // Plans hold every activation of their length, so few are kept.
      while (this.plans.size >= 2) {
        const [k, old] = this.plans.entries().next().value!;
        this.dropPlan(old);
        this.plans.delete(k);
      }
      plan = this.build(N);
    }
    this.plans.set(N, plan);
    const be = this.backend;
    be.write(plan.input, 0, samples);
    be.submit(plan.dispatches);
    const width = plan.out.shape[1]!;
    const data = new Float32Array(await be.readback(physical(plan.out.buf), 0, plan.frames * width * 4));
    return { data, frames: plan.frames };
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
