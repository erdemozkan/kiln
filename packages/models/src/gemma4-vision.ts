import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError, configNumber } from "./config";
import type { ImagePixels } from "./vision";

/**
 * Gemma 4's vision tower: a ViT over 16 × 16 patches at the image's own aspect ratio, with one
 * learned position embedding per column and per row and rotary positions on both axes, whose patches
 * are averaged in 3 × 3 blocks into tokens and projected to the text width.
 */
export interface Gemma4VisionConfig {
  hidden: number;
  heads: number;
  layers: number;
  intermediate: number;
  patchSize: number;
  /** Patches per side averaged into one token (3: nine patches a token). */
  poolSize: number;
  /** Rows of each axis's position table (10240). */
  posSize: number;
  /** Width of the tokens the projection makes: the text model's. */
  projectionDim: number;
  eps: number;
  ropeTheta: number;
  /** Most tokens the processor resizes an image to (280). */
  maxTokens: number;
}

/** Token budgets the model was trained with. */
export const GEMMA4_IMAGE_TOKENS = [70, 140, 280, 560, 1120] as const;

/** The projections of a layer whose input and output the tower clips (see `Gemma4VisionTower`). */
export const GEMMA4_CLIPPED = ["attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down"] as const;

/** A clipped projection's limits: its input's lowest and highest value, then its output's. */
export type Gemma4Clip = readonly [number, number, number, number];

const num = (o: Record<string, unknown>, k: string, d: number): number => (o[k] === undefined || o[k] === null ? d : configNumber(k, o[k]));

function check(c: Gemma4VisionConfig): Gemma4VisionConfig {
  const D = c.hidden / c.heads;
  if (!Number.isInteger(D) || D % 4) throw new UnsupportedModelError(`Gemma 4 vision: heads of ${D} do not split into column and row pairs`);
  for (const [what, n] of [["patch", 3 * c.patchSize ** 2], ["hidden", c.hidden], ["MLP", c.intermediate]] as const)
    if (n % 32) throw new UnsupportedModelError(`Gemma 4 vision: a ${what} width of ${n} is not a multiple of 32`);
  return c;
}

/** From a transformers checkpoint's config.json (`vision_config`) and its processor's image settings. */
export function gemma4VisionConfig(config: Record<string, unknown>, processor?: Record<string, unknown>): Gemma4VisionConfig {
  const v = config.vision_config as Record<string, unknown> | undefined;
  if (!v) throw new UnsupportedModelError("this checkpoint has no vision_config");
  if (v.standardize) throw new UnsupportedModelError("Gemma 4 vision: standardized features are not supported");
  const text = (config.text_config ?? {}) as Record<string, unknown>;
  const rope = (v.rope_parameters ?? {}) as Record<string, unknown>;
  const hidden = num(v, "hidden_size", 768);
  return check({
    hidden,
    heads: num(v, "num_attention_heads", 12),
    layers: num(v, "num_hidden_layers", 16),
    intermediate: num(v, "intermediate_size", 3072),
    patchSize: num(v, "patch_size", 16),
    poolSize: num(v, "pooling_kernel_size", 3),
    posSize: num(v, "position_embedding_size", 10240),
    projectionDim: num(text, "hidden_size", hidden),
    eps: num(v, "rms_norm_eps", 1e-6),
    ropeTheta: num(rope, "rope_theta", 100),
    maxTokens: num(processor ?? {}, "max_soft_tokens", num(config, "vision_soft_tokens_per_image", 280)),
  });
}

/** From an mmproj GGUF's metadata (`clip.vision.projector_type` "gemma4v"); `posRows` is the position table's rows per axis. */
export function gemma4VisionConfigFromGGUF(meta: Map<string, unknown>, posRows: number): Gemma4VisionConfig {
  const n = (k: string, d?: number): number => {
    const v = meta.get(k);
    if (typeof v === "number") return v;
    if (d === undefined) throw new UnsupportedModelError(`mmproj: missing ${k}`);
    return d;
  };
  return check({
    hidden: n("clip.vision.embedding_length"),
    heads: n("clip.vision.attention.head_count"),
    layers: n("clip.vision.block_count"),
    intermediate: n("clip.vision.feed_forward_length"),
    patchSize: n("clip.vision.patch_size"),
    poolSize: 3,
    posSize: posRows,
    projectionDim: n("clip.vision.projection_dim"),
    eps: n("clip.vision.attention.layer_norm_epsilon", 1e-6),
    ropeTheta: 100,
    maxTokens: 280,
  });
}

/**
 * The token grid an image of width × height is resized to, as the processor picks it: the largest
 * size of the same aspect ratio whose patches fit `maxTokens` tokens, each side rounded down to
 * whole tokens (48 pixels). A side that would round to nothing is one token, and the other as long
 * as the aspect ratio asks, up to the budget.
 */
export function gemma4ImageGrid(width: number, height: number, c: Gemma4VisionConfig, maxTokens = c.maxTokens): { rows: number; cols: number } {
  if (!(width > 0 && height > 0)) throw new Error("image has no pixels");
  const side = c.poolSize * c.patchSize;
  const factor = Math.sqrt((maxTokens * side * side) / (width * height));
  let rows = Math.floor((factor * height) / side);
  let cols = Math.floor((factor * width) / side);
  if (rows === 0 && cols === 0) throw new Error("image cannot be resized to a whole token");
  if (rows === 0) {
    rows = 1;
    cols = Math.min(Math.floor(width / height), maxTokens);
  } else if (cols === 0) {
    cols = 1;
    rows = Math.min(Math.floor(height / width), maxTokens);
  }
  return { rows, cols };
}

/**
 * Patch rows of an image whose sides are whole patches: [patches, 3 · p · p], patches in raster
 * order, each channel-major (channel, row, column), pixels scaled to −1…1.
 */
export function gemma4Patches(c: Gemma4VisionConfig, img: ImagePixels): Float32Array {
  const P = c.patchSize;
  const [gh, gw] = [img.height / P, img.width / P];
  if (!Number.isInteger(gh) || !Number.isInteger(gw)) throw new Error(`image is ${img.width}×${img.height}, not whole patches of ${P}`);
  const ch = img.data.length / (img.width * img.height);
  if (ch !== 3 && ch !== 4) throw new Error("image data must be RGB or RGBA bytes");
  const K = 3 * P * P;
  const out = new Float32Array(gh * gw * K);
  for (let py = 0; py < gh; py++)
    for (let px = 0; px < gw; px++) {
      const o = (py * gw + px) * K;
      for (let k = 0; k < 3; k++)
        for (let y = 0; y < P; y++)
          for (let x = 0; x < P; x++) out[o + (k * P + y) * P + x] = 2 * (img.data[((py * P + y) * img.width + px * P + x) * ch + k]! / 255 - 0.5);
    }
  return out;
}

/** Position embeddings of a gh × gw grid from `table` [2, posSize, hidden]: a patch's column's row of the first half plus its row's of the second. */
export function gemma4PositionEmbeddings(table: Float32Array, c: Gemma4VisionConfig, gh: number, gw: number): Float32Array {
  const H = c.hidden;
  if (gh > c.posSize || gw > c.posSize) throw new Error(`a ${gw}×${gh} grid is past the ${c.posSize} positions of the table`);
  const out = new Float32Array(gh * gw * H);
  for (let py = 0; py < gh; py++)
    for (let px = 0; px < gw; px++) {
      const o = (py * gw + px) * H;
      const [xs, ys] = [px * H, (c.posSize + py) * H];
      for (let j = 0; j < H; j++) out[o + j] = table[xs + j]! + table[ys + j]!;
    }
  return out;
}

/**
 * Rotary angles [patches, D / 2] of a gh × gw grid for a head of two groups: the first half of a
 * head turns with the patch's column, the second with its row, both at the same D / 4 frequencies.
 */
export function gemma4RotaryAngles(c: Gemma4VisionConfig, gh: number, gw: number): Float32Array {
  const quarter = c.hidden / c.heads / 4;
  const out = new Float32Array(gh * gw * quarter * 2);
  for (let py = 0; py < gh; py++)
    for (let px = 0; px < gw; px++) {
      const o = (py * gw + px) * quarter * 2;
      for (let i = 0; i < quarter; i++) {
        const f = 1 / c.ropeTheta ** (i / quarter);
        out[o + i] = px * f;
        out[o + quarter + i] = py * f;
      }
    }
  return out;
}

interface Plan {
  dispatches: readonly Dispatch[];
  patches: Buf;
  pos: Buf;
  angles: Buf;
  features: Tensor;
  ctx: Ctx;
  used: number;
}

/**
 * The tower on the GPU. Weights use llama.cpp's mmproj names (the loader renames a transformers
 * checkpoint's): `v.patch_embd.weight` as one [hidden, 3 · p · p] matrix over channel-major patches,
 * `v.blk.N.*` and `mm.input_projection.weight`. Every projection of a layer clips its input and its
 * output to limits stored with the weights (`clips`, by `v.blk.N.<projection>`). The position table
 * stays on the CPU. Attention is the text model's over a 'cache' of one slot per patch with a span
 * over all of them (see `VisionTower`).
 */
export class Gemma4VisionTower {
  readonly kind = "gemma4";
  /** No fixed size: the engine resizes each image to its own grid (see `grid`). */
  readonly processing = undefined;
  readonly bidirectional = false;
  private readonly plans = new Map<string, Plan>();
  private clock = 0;
  private readonly step: Buf;
  private readonly kc: Buf;
  private readonly vc: Buf;
  private readonly ones: Tensor;
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: Gemma4VisionConfig,
    private readonly weights: Map<string, Tensor>,
    private readonly clips: ReadonlyMap<string, Gemma4Clip>,
    private readonly posTable: Float32Array,
    /** Most tokens an image is resized to; a feature slot holds this many. */
    readonly maxTokens: number,
  ) {
    if (posTable.length !== 2 * cfg.posSize * cfg.hidden) throw new Error("Gemma 4 vision: the position table does not fit the config");
    const n = maxTokens * cfg.poolSize ** 2;
    this.step = backend.alloc(STEP_BYTES, "storage", "vision/step");
    this.kc = backend.alloc(n * cfg.hidden * 4, "storage", "vision/k");
    this.vc = backend.alloc(n * cfg.hidden * 4, "storage", "vision/v");
    const ones = backend.alloc(cfg.hidden * 4, "storage", "vision/ones");
    backend.write(ones, 0, new Float32Array(cfg.hidden).fill(1));
    this.ones = { shape: [cfg.hidden], dtype: "f32", buf: ones };
    this.pool = new ActivationPool(backend, "vision");
  }

  /** Feature rows a slot holds (an image may use fewer). */
  get tokensPerImage(): number {
    return this.maxTokens;
  }

  /** As `tokensPerImage`: the engine's unit for slots. */
  get tokens(): number {
    return this.maxTokens;
  }

  /** The token grid an image of this size gets, and the pixel size to resize it to. */
  grid(width: number, height: number): { rows: number; cols: number; width: number; height: number } {
    const g = gemma4ImageGrid(width, height, this.cfg, this.maxTokens);
    const f = this.cfg.patchSize * this.cfg.poolSize;
    return { ...g, width: g.cols * f, height: g.rows * f };
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`vision weight "${name}" was not loaded`);
    return t;
  }

  private build(gh: number, gw: number): Plan {
    const c = this.cfg;
    const n = gh * gw;
    const D = c.hidden / c.heads;
    const K = 3 * c.patchSize ** 2;
    const ctx = new Ctx(this.backend, this.step, "vision", this.pool);
    const patches = ctx.buffer(n * K * 4, "vision/patches");
    const pos = ctx.buffer(n * c.hidden * 4, "vision/pos");
    const anglesBuf = ctx.buffer(n * (D / 2) * 4, "vision/angles");
    const angles: Tensor = { shape: [n, D / 2], dtype: "f32", buf: anglesBuf };
    const norm = (t: Tensor, name: string, dim?: number) => ops.rmsnorm(ctx, t, this.w(name), c.eps, dim);
    // Inputs that share limits (q, k and v; gate and up) are clipped once.
    const clipped = new Map<string, Tensor>();
    const clipIn = (t: Tensor, at: string, key: string): Tensor => {
      const lim = this.clips.get(at);
      if (!lim || (lim[0] === -Infinity && lim[1] === Infinity)) return t;
      const id = `${key}|${lim[0]}|${lim[1]}`;
      let y = clipped.get(id);
      if (!y) clipped.set(id, (y = ops.clamp(ctx, t, lim[0], lim[1])));
      return y;
    };
    const linear = (t: Tensor, at: string, key: string): Tensor => {
      const y = ops.linear(ctx, clipIn(t, at, key), this.w(`${at}.weight`));
      const lim = this.clips.get(at);
      return !lim || (lim[2] === -Infinity && lim[3] === Infinity) ? y : ops.clamp(ctx, y, lim[2], lim[3]);
    };
    const x = ops.linear(ctx, { shape: [n, K], dtype: "f32", buf: patches }, this.w("v.patch_embd.weight"));
    ops.addInPlace(ctx, x, { shape: [n, c.hidden], dtype: "f32", buf: pos });
    const shape = { T: n, H: c.heads, Hkv: c.heads, D, maxSeq: n, kv: "f32" as const, scale: 1, span: true };
    const headOnes: Tensor = { shape: [D], dtype: "f32", buf: this.ones.buf };
    for (let l = 0; l < c.layers; l++) {
      const p = `v.blk.${l}.`;
      const h = norm(x, p + "ln1.weight");
      const q = norm(linear(h, p + "attn_q", `${l}a`), p + "attn_q_norm.weight", D);
      const k = norm(linear(h, p + "attn_k", `${l}a`), p + "attn_k_norm.weight", D);
      // Values are normalised per head too, without a learned weight.
      const v = ops.rmsnorm(ctx, linear(h, p + "attn_v", `${l}a`), headOnes, c.eps, D);
      ops.ropeTable(ctx, q, angles, c.heads, D, 2);
      ops.ropeTable(ctx, k, angles, c.heads, D, 2);
      ops.kvWrite(ctx, k, v, this.kc, this.vc, n, n, "f32");
      const a = ops.attention(ctx, q, this.kc, this.vc, shape);
      ops.addInPlace(ctx, x, norm(linear(a, p + "attn_out", `${l}o`), p + "attn_post_norm.weight"));
      const h2 = norm(x, p + "ln2.weight");
      const m = ops.siluMul(ctx, linear(h2, p + "ffn_gate", `${l}m`), linear(h2, p + "ffn_up", `${l}m`), "gelu");
      ops.addInPlace(ctx, x, norm(linear(m, p + "ffn_down", `${l}d`), p + "ffn_post_norm.weight"));
    }
    const pooled = ops.avgPool(ctx, x, gw, c.poolSize, gh);
    // The model scales the pooled patches by √hidden before a norm without a weight, which undoes
    // the scale but for eps: the same as eps / hidden without it.
    const normed = ops.rmsnorm(ctx, pooled, this.ones, c.eps / c.hidden);
    const features = ops.linear(ctx, normed, this.w("mm.input_projection.weight"));
    return { dispatches: ctx.dispatches, patches, pos, angles: anglesBuf, features, ctx, used: 0 };
  }

  /**
   * Runs the tower on an image already resized to `grid`'s size and copies its rows · cols features
   * into `dst` at byte offset `dstOffset`; returns the token grid. Plans are kept for the three most
   * recent grid sizes. The work goes out a layer at a time, as `VisionTower.encode` does.
   */
  async encode(img: ImagePixels, dst: Buf, dstOffset: number): Promise<{ rows: number; cols: number }> {
    const c = this.cfg;
    const [gh, gw] = [img.height / c.patchSize, img.width / c.patchSize];
    const [rows, cols] = [gh / c.poolSize, gw / c.poolSize];
    if (!Number.isInteger(rows) || !Number.isInteger(cols)) throw new Error(`image is ${img.width}×${img.height}, not whole tokens of ${c.patchSize * c.poolSize} pixels`);
    if (rows * cols > this.maxTokens) throw new Error(`image makes ${rows * cols} tokens, but at most ${this.maxTokens} fit`);
    const key = `${gh}x${gw}`;
    let plan = this.plans.get(key);
    if (!plan) {
      if (this.plans.size >= 3) {
        const [oldest] = [...this.plans].sort((a, b) => a[1].used - b[1].used)[0]!;
        this.plans.get(oldest)!.ctx.destroy();
        this.plans.delete(oldest);
      }
      plan = this.build(gh, gw);
      this.plans.set(key, plan);
    }
    plan.used = ++this.clock;
    const n = gh * gw;
    this.backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: n }));
    this.backend.write(plan.patches, 0, gemma4Patches(c, img));
    this.backend.write(plan.pos, 0, gemma4PositionEmbeddings(this.posTable, c, gh, gw));
    this.backend.write(plan.angles, 0, gemma4RotaryAngles(c, gh, gw));
    const d = plan.dispatches;
    const per = Math.max(1, Math.ceil(d.length / (c.layers + 2)));
    let pending: Promise<ArrayBuffer> | undefined;
    for (let i = 0; i < d.length; i += per) {
      this.backend.submit(d.slice(i, i + per));
      const done = this.backend.readback(this.step, 0, 4);
      if (pending) await pending;
      pending = done;
    }
    await pending;
    this.backend.copy(physical(plan.features.buf), 0, dst, dstOffset, rows * cols * c.projectionDim * 4);
    return { rows, cols };
  }

  /** GPU bytes the tower's weights hold. */
  get weightBytes(): number {
    let n = 0;
    for (const t of this.weights.values()) n += t.buf.size + (t.scales?.size ?? 0) + (t.mins?.size ?? 0);
    return n;
  }

  destroy(): void {
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    for (const p of this.plans.values()) p.ctx.destroy();
    this.plans.clear();
    this.pool.destroy();
    for (const b of [this.step, this.kc, this.vc, this.ones.buf]) this.backend.free(b);
  }
}

/**
 * A transformers checkpoint's vision tensor name as the tower's (mmproj) name, or undefined for
 * other tensors. A clipped projection's limits come out as `<projection>.input_min` and so on.
 */
export function gemma4VisionName(name: string): string | undefined {
  if (/^(?:model\.)?embed_vision\.embedding_projection\.weight$/.test(name)) return "mm.input_projection.weight";
  const m = /^(?:model\.)?vision_tower\.(.+)$/.exec(name);
  if (!m) return undefined;
  const n = m[1]!;
  if (n === "patch_embedder.input_proj.weight") return "v.patch_embd.weight";
  if (n === "patch_embedder.position_embedding_table") return "v.position_embd.weight";
  const blk = /^encoder\.layers\.(\d+)\.(.+)$/.exec(n);
  if (!blk) return undefined;
  const norms: Record<string, string> = {
    "input_layernorm.weight": "ln1.weight",
    "post_attention_layernorm.weight": "attn_post_norm.weight",
    "pre_feedforward_layernorm.weight": "ln2.weight",
    "post_feedforward_layernorm.weight": "ffn_post_norm.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
  };
  const rest = blk[2]!;
  if (norms[rest]) return `v.blk.${blk[1]}.${norms[rest]}`;
  const lin = /^(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.(linear\.weight|input_min|input_max|output_min|output_max)$/.exec(rest);
  if (!lin) return undefined;
  const to: Record<string, string> = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_out",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
  };
  return `v.blk.${blk[1]}.${to[lin[1]!]}.${lin[2] === "linear.weight" ? "weight" : lin[2]}`;
}
