import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError, configNumber } from "./config";
import type { ImagePixels } from "./vision";

/**
 * Qwen 3.5's vision tower (Qwen3-VL's): a ViT over 16 × 16 patches at the image's own aspect ratio,
 * with learned position embeddings resampled to the patch grid and rotary positions on rows and
 * columns, then a merger that joins each 2 × 2 block of patches into one token of the text width.
 */
export interface QwenVisionConfig {
  hidden: number;
  heads: number;
  layers: number;
  intermediate: number;
  patchSize: number;
  /** Patches per side joined into one token (2: four patches a token). */
  mergeSize: number;
  /** Side of the square grid the learned position embeddings were trained on (48). */
  posGrid: number;
  /** Width of the tokens the merger makes: the text model's. */
  projectionDim: number;
  eps: number;
  mean: [number, number, number];
  std: [number, number, number];
  /** Fewest and most tokens the processor resizes an image to (`size` in pixels over a token's). */
  minTokens: number;
  maxTokens: number;
}

const THETA = 10000;

/**
 * Most tokens Kiln gives a pair of video frames (the tower joins two frames into one set of
 * tokens). The processor's own budget, up to 768 a pair, is more than a browser's context holds for
 * a clip of any length.
 */
export const QWEN_VIDEO_PAIR_TOKENS = 64;

/** A number from a config, or `d` where the key is absent. */
const num = (o: Record<string, unknown>, k: string, d: number): number => (o[k] === undefined || o[k] === null ? d : configNumber(k, o[k]));

function check(c: QwenVisionConfig): QwenVisionConfig {
  const D = c.hidden / c.heads;
  if (!Number.isInteger(D) || D % 4) throw new UnsupportedModelError(`Qwen vision: heads of ${D} do not split into row and column pairs`);
  for (const [what, n] of [["patch", 3 * c.patchSize ** 2], ["hidden", c.hidden], ["MLP", c.intermediate], ["merger", c.hidden * c.mergeSize ** 2]] as const)
    if (n % 32) throw new UnsupportedModelError(`Qwen vision: a ${what} width of ${n} is not a multiple of 32`);
  if (!Number.isInteger(c.posGrid)) throw new UnsupportedModelError("Qwen vision: the position embeddings are not a square grid");
  return c;
}

/** From a transformers checkpoint's config.json (`vision_config`) and preprocessor_config.json. */
export function qwenVisionConfig(config: Record<string, unknown>, preprocessor?: Record<string, unknown>): QwenVisionConfig {
  const v = config.vision_config as Record<string, unknown> | undefined;
  if (!v) throw new UnsupportedModelError("this checkpoint has no vision_config");
  if (Array.isArray(v.deepstack_visual_indexes) && v.deepstack_visual_indexes.length) throw new UnsupportedModelError("Qwen vision: deepstack features are not supported");
  const patchSize = num(v, "patch_size", 16);
  const mergeSize = num(v, "spatial_merge_size", 2);
  const token = (patchSize * mergeSize) ** 2;
  const size = (preprocessor?.size ?? {}) as Record<string, unknown>;
  const triple = (x: unknown, d: number): [number, number, number] => (Array.isArray(x) && x.length === 3 ? (x.map(Number) as [number, number, number]) : [d, d, d]);
  return check({
    hidden: num(v, "hidden_size", 768),
    heads: num(v, "num_heads", 12),
    layers: num(v, "depth", 12),
    intermediate: num(v, "intermediate_size", 3072),
    patchSize,
    mergeSize,
    posGrid: Math.sqrt(num(v, "num_position_embeddings", 2304)),
    projectionDim: num(v, "out_hidden_size", 1024),
    eps: 1e-6,
    mean: triple(preprocessor?.image_mean, 0.5),
    std: triple(preprocessor?.image_std, 0.5),
    minTokens: Math.max(1, Math.round(num(size, "shortest_edge", 65536) / token)),
    maxTokens: Math.round(num(size, "longest_edge", 16777216) / token),
  });
}

/** From llama.cpp's mmproj GGUF (`clip.projector_type` "qwen3vl_merger"); `posRows` is the position table's length. */
export function qwenVisionConfigFromGGUF(meta: Map<string, unknown>, posRows: number): QwenVisionConfig {
  const n = (k: string, d?: number): number => {
    const v = meta.get(k);
    if (typeof v === "number") return v;
    if (d === undefined) throw new UnsupportedModelError(`mmproj: missing ${k}`);
    return d;
  };
  const deep = meta.get("clip.vision.is_deepstack_layers");
  if (Array.isArray(deep) && deep.some(Boolean)) throw new UnsupportedModelError("Qwen vision: deepstack features are not supported");
  const triple = (k: string): [number, number, number] => {
    const v = meta.get(k);
    return Array.isArray(v) && v.length === 3 ? (v.map(Number) as [number, number, number]) : [0.5, 0.5, 0.5];
  };
  return check({
    hidden: n("clip.vision.embedding_length"),
    heads: n("clip.vision.attention.head_count"),
    layers: n("clip.vision.block_count"),
    intermediate: n("clip.vision.feed_forward_length"),
    patchSize: n("clip.vision.patch_size"),
    mergeSize: n("clip.vision.spatial_merge_size", 2),
    posGrid: Math.sqrt(posRows),
    projectionDim: n("clip.vision.projection_dim"),
    eps: n("clip.vision.attention.layer_norm_epsilon", 1e-6),
    mean: triple("clip.vision.image_mean"),
    std: triple("clip.vision.image_std"),
    minTokens: 64,
    maxTokens: 16384,
  });
}

/**
 * The token grid an image of width × height is resized to, as the processor's `smart_resize` picks
 * it: each side rounded to whole tokens (32 pixels), then scaled down or up, keeping the aspect
 * ratio, until the tokens are between `minTokens` and `maxTokens`.
 */
export function qwenImageGrid(width: number, height: number, c: QwenVisionConfig, maxTokens = c.maxTokens): { rows: number; cols: number } {
  if (!(width > 0 && height > 0)) throw new Error("image has no pixels");
  if (Math.max(width, height) / Math.min(width, height) > 200) throw new Error("image is more than 200 times longer than wide");
  const f = c.patchSize * c.mergeSize;
  const max = Math.max(maxTokens, 1) * f * f;
  const min = Math.min(c.minTokens, maxTokens) * f * f;
  let h = Math.max(f, Math.round(height / f) * f);
  let w = Math.max(f, Math.round(width / f) * f);
  if (h * w > max) {
    const beta = Math.sqrt((height * width) / max);
    h = Math.max(f, Math.floor(height / beta / f) * f);
    w = Math.max(f, Math.floor(width / beta / f) * f);
  } else if (h * w < min) {
    const beta = Math.sqrt(min / (height * width));
    h = Math.ceil((height * beta) / f) * f;
    w = Math.ceil((width * beta) / f) * f;
  }
  return { rows: h / f, cols: w / f };
}

/**
 * The patch at row r of the tower's input, as [patch row, patch column]: the patches of each merge
 * block are consecutive (block by block in raster order, then raster inside the block), so the
 * merger's 2 × 2 join is a reshape.
 */
function patchAt(r: number, gw: number, m: number): [number, number] {
  const per = m * m;
  const block = Math.floor(r / per);
  const inner = r % per;
  const bw = gw / m;
  return [Math.floor(block / bw) * m + Math.floor(inner / m), (block % bw) * m + (inner % m)];
}

/**
 * Patch rows [patches, 3 · p · p] of an image already resized to the grid: each patch's values in
 * (channel, row, column) order, as the convolution weight flattens, normalised with the config's
 * mean and std, in merge-block order. With `second` (a video's next frame, the same size), each row
 * holds the patch of both frames, one after the other: [patches, 2 · 3 · p · p].
 */
export function qwenPatches(c: QwenVisionConfig, img: ImagePixels, second?: ImagePixels): Float32Array {
  const P = c.patchSize;
  const [gh, gw] = [img.height / P, img.width / P];
  if (!Number.isInteger(gh / c.mergeSize) || !Number.isInteger(gw / c.mergeSize)) throw new Error(`image is ${img.width}×${img.height}, not whole tokens of ${P * c.mergeSize} pixels`);
  const ch = img.data.length / (img.width * img.height);
  if (ch !== 3 && ch !== 4) throw new Error("image data must be RGB or RGBA bytes");
  if (second && (second.width !== img.width || second.height !== img.height || second.data.length !== img.data.length)) throw new Error("a video's frames must all have the same size");
  const K = 3 * P * P;
  const frames = second ? [img, second] : [img];
  const row = K * frames.length;
  const out = new Float32Array(gh * gw * row);
  for (let r = 0; r < gh * gw; r++) {
    const [py, px] = patchAt(r, gw, c.mergeSize);
    frames.forEach((f, t) => {
      const o = r * row + t * K;
      for (let k = 0; k < 3; k++)
        for (let y = 0; y < P; y++)
          for (let x = 0; x < P; x++) {
            const v = f.data[((py * P + y) * f.width + px * P + x) * ch + k]! / 255;
            out[o + (k * P + y) * P + x] = (v - c.mean[k]!) / c.std[k]!;
          }
    });
  }
  return out;
}

/**
 * The learned position embeddings `table` [G · G, hidden] resampled to a grid of gh × gw patches
 * (bilinear, corners aligned, as the model interpolates them), in merge-block order.
 */
export function qwenPositionEmbeddings(table: Float32Array, c: QwenVisionConfig, gh: number, gw: number): Float32Array {
  const G = c.posGrid;
  const hidden = c.hidden;
  const out = new Float32Array(gh * gw * hidden);
  const at = (i: number, n: number) => (n === 1 ? 0 : (i * (G - 1)) / (n - 1));
  for (let r = 0; r < gh * gw; r++) {
    const [py, px] = patchAt(r, gw, c.mergeSize);
    const [fy, fx] = [at(py, gh), at(px, gw)];
    const [y0, x0] = [Math.floor(fy), Math.floor(fx)];
    const [y1, x1] = [Math.min(y0 + 1, G - 1), Math.min(x0 + 1, G - 1)];
    const [dy, dx] = [fy - y0, fx - x0];
    const o = r * hidden;
    for (const [y, x, w] of [[y0, x0, (1 - dy) * (1 - dx)], [y0, x1, (1 - dy) * dx], [y1, x0, dy * (1 - dx)], [y1, x1, dy * dx]] as const) {
      if (w === 0) continue;
      const s = (y * G + x) * hidden;
      for (let j = 0; j < hidden; j++) out[o + j] += w * table[s + j]!;
    }
  }
  return out;
}

/**
 * Rotary angles [patches, D / 2] of a gh × gw grid, for rotate-half pairs: the first half of the
 * pairs turns with the patch's row, the second with its column, both at the same D / 4 frequencies.
 */
export function qwenRotaryAngles(c: QwenVisionConfig, gh: number, gw: number): Float32Array {
  const quarter = c.hidden / c.heads / 4;
  const out = new Float32Array(gh * gw * quarter * 2);
  for (let r = 0; r < gh * gw; r++) {
    const [py, px] = patchAt(r, gw, c.mergeSize);
    for (let i = 0; i < quarter; i++) {
      const f = 1 / THETA ** (i / quarter);
      out[r * quarter * 2 + i] = py * f;
      out[r * quarter * 2 + quarter + i] = px * f;
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
 * checkpoint's): `v.patch_embd.weight` as one [hidden, 3 · p · p] matrix (the two frames of the
 * model's video convolution added, since a still image is both frames; `v.patch_embd.pair.weight`
 * [hidden, 2 · 3 · p · p] keeps them apart, for a video's pairs of frames), `v.blk.N.*` with q, k and v
 * as one `attn_qkv`, `v.post_ln` (the merger's norm) and `mm.0` / `mm.2` (its two layers). The
 * position table stays on the CPU, where it is resampled per image size. Attention is the text
 * model's over a 'cache' of one slot per patch with a span over all of them (see `VisionTower`).
 */
export class QwenVisionTower {
  readonly kind = "qwen";
  /** No fixed size: the engine resizes each image to its own grid (see `grid`). */
  readonly processing = undefined;
  readonly bidirectional = false;
  private readonly plans = new Map<string, Plan>();
  private clock = 0;
  private readonly step: Buf;
  private readonly kc: Buf;
  private readonly vc: Buf;
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: QwenVisionConfig,
    private readonly weights: Map<string, Tensor>,
    private readonly posTable: Float32Array,
    /** Most tokens an image is resized to; a feature slot holds this many. */
    readonly maxTokens: number,
  ) {
    if (posTable.length !== cfg.posGrid ** 2 * cfg.hidden) throw new Error("Qwen vision: the position table does not fit the config");
    const n = maxTokens * cfg.mergeSize ** 2;
    this.step = backend.alloc(STEP_BYTES, "storage", "vision/step");
    this.kc = backend.alloc(n * cfg.hidden * 4, "storage", "vision/k");
    this.vc = backend.alloc(n * cfg.hidden * 4, "storage", "vision/v");
    this.pool = new ActivationPool(backend, "vision");
  }

  /** Feature rows a slot holds (an image may use fewer). */
  get tokensPerImage(): number {
    return this.maxTokens;
  }

  /** Whether the tower has the video convolution's two frames apart, to take a video's pairs of frames. */
  get takesVideo(): boolean {
    return this.weights.has("v.patch_embd.pair.weight");
  }

  /** As `tokensPerImage`: the engine's unit for slots. */
  get tokens(): number {
    return this.maxTokens;
  }

  /** The token grid an image of this size gets (at most `maxTokens` tokens), and the pixel size to resize it to. */
  grid(width: number, height: number, maxTokens = this.maxTokens): { rows: number; cols: number; width: number; height: number } {
    const g = qwenImageGrid(width, height, this.cfg, Math.min(maxTokens, this.maxTokens));
    const f = this.cfg.patchSize * this.cfg.mergeSize;
    return { ...g, width: g.cols * f, height: g.rows * f };
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`vision weight "${name}" was not loaded`);
    return t;
  }

  private build(gh: number, gw: number, pair: boolean): Plan {
    const c = this.cfg;
    const n = gh * gw;
    const D = c.hidden / c.heads;
    const K = (pair ? 2 : 1) * 3 * c.patchSize ** 2;
    const ctx = new Ctx(this.backend, this.step, "vision", this.pool);
    const patches = ctx.buffer(n * K * 4, "vision/patches");
    const pos = ctx.buffer(n * c.hidden * 4, "vision/pos");
    const anglesBuf = ctx.buffer(n * (D / 2) * 4, "vision/angles");
    const angles: Tensor = { shape: [n, D / 2], dtype: "f32", buf: anglesBuf };
    const x = ops.linear(ctx, { shape: [n, K], dtype: "f32", buf: patches }, this.w(pair ? "v.patch_embd.pair.weight" : "v.patch_embd.weight"), { bias: this.w("v.patch_embd.bias") });
    ops.addInPlace(ctx, x, { shape: [n, c.hidden], dtype: "f32", buf: pos });
    const shape = { T: n, H: c.heads, Hkv: c.heads, D, maxSeq: n, kv: "f32" as const, scale: D ** -0.5, span: true };
    for (let l = 0; l < c.layers; l++) {
      const p = `v.blk.${l}.`;
      const h = ops.layernorm(ctx, x, this.w(p + "ln1.weight"), this.w(p + "ln1.bias"), c.eps);
      const qkv = ops.linear(ctx, h, this.w(p + "attn_qkv.weight"), { bias: this.w(p + "attn_qkv.bias") });
      const [q, k, v] = [0, 1, 2].map((i) => {
        const t = ctx.alloc([n, c.hidden]);
        ops.cols(ctx, t, qkv, { n: c.hidden, srcStart: i * c.hidden });
        return t;
      }) as [Tensor, Tensor, Tensor];
      ops.ropeTable(ctx, q, angles, c.heads, D);
      ops.ropeTable(ctx, k, angles, c.heads, D);
      ops.kvWrite(ctx, k, v, this.kc, this.vc, n, n, "f32");
      const a = ops.attention(ctx, q, this.kc, this.vc, shape);
      ops.linear(ctx, a, this.w(p + "attn_out.weight"), { bias: this.w(p + "attn_out.bias"), residual: x });
      const h2 = ops.layernorm(ctx, x, this.w(p + "ln2.weight"), this.w(p + "ln2.bias"), c.eps);
      const up = ops.linear(ctx, h2, this.w(p + "ffn_up.weight"), { bias: this.w(p + "ffn_up.bias") });
      ops.gelu(ctx, up);
      ops.linear(ctx, up, this.w(p + "ffn_down.weight"), { bias: this.w(p + "ffn_down.bias"), residual: x });
    }
    const normed = ops.layernorm(ctx, x, this.w("v.post_ln.weight"), this.w("v.post_ln.bias"), c.eps);
    // The patches of a merge block are consecutive rows: joining them is a reshape.
    const per = c.mergeSize ** 2;
    const joined: Tensor = { ...normed, shape: [n / per, c.hidden * per] };
    const mid = ops.linear(ctx, joined, this.w("mm.0.weight"), { bias: this.w("mm.0.bias") });
    ops.gelu(ctx, mid, true);
    const features = ops.linear(ctx, mid, this.w("mm.2.weight"), { bias: this.w("mm.2.bias") });
    return { dispatches: ctx.dispatches, patches, pos, angles: anglesBuf, features, ctx, used: 0 };
  }

  /**
   * Runs the tower on an image already resized to `grid`'s size and copies its rows · cols features
   * into `dst` at byte offset `dstOffset`; returns the token grid. Plans are kept for the three most
   * recent grid sizes. The work goes out a layer at a time, as `VisionTower.encode` does. With
   * `second`, the two are consecutive frames of a video, joined by the model's video convolution.
   */
  async encode(img: ImagePixels, dst: Buf, dstOffset: number, second?: ImagePixels): Promise<{ rows: number; cols: number }> {
    const c = this.cfg;
    const [gh, gw] = [img.height / c.patchSize, img.width / c.patchSize];
    const [rows, cols] = [gh / c.mergeSize, gw / c.mergeSize];
    if (!Number.isInteger(rows) || !Number.isInteger(cols)) throw new Error(`image is ${img.width}×${img.height}, not whole tokens of ${c.patchSize * c.mergeSize} pixels`);
    if (rows * cols > this.maxTokens) throw new Error(`image makes ${rows * cols} tokens, but at most ${this.maxTokens} fit`);
    const key = `${gh}x${gw}${second ? "/pair" : ""}`;
    let plan = this.plans.get(key);
    if (!plan) {
      if (this.plans.size >= 3) {
        const [oldest] = [...this.plans].sort((a, b) => a[1].used - b[1].used)[0]!;
        this.plans.get(oldest)!.ctx.destroy();
        this.plans.delete(oldest);
      }
      plan = this.build(gh, gw, !!second);
      this.plans.set(key, plan);
    }
    plan.used = ++this.clock;
    const n = gh * gw;
    this.backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: n }));
    this.backend.write(plan.patches, 0, qwenPatches(c, img, second));
    this.backend.write(plan.pos, 0, qwenPositionEmbeddings(this.posTable, c, gh, gw));
    this.backend.write(plan.angles, 0, qwenRotaryAngles(c, gh, gw));
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
    for (const b of [this.step, this.kc, this.vc]) this.backend.free(b);
  }
}

/** A transformers checkpoint's vision tensor name as the tower's (mmproj) name, or undefined for other tensors. */
export function qwenVisionName(name: string): string | undefined {
  const m = /^(?:model\.)?visual\.(.+)$/.exec(name);
  if (!m) return undefined;
  const n = m[1]!;
  const blk = /^blocks\.(\d+)\.(.+)$/.exec(n);
  if (blk) {
    const map: Record<string, string> = {
      "norm1.weight": "ln1.weight", "norm1.bias": "ln1.bias", "norm2.weight": "ln2.weight", "norm2.bias": "ln2.bias",
      "attn.qkv.weight": "attn_qkv.weight", "attn.qkv.bias": "attn_qkv.bias", "attn.proj.weight": "attn_out.weight", "attn.proj.bias": "attn_out.bias",
      "mlp.linear_fc1.weight": "ffn_up.weight", "mlp.linear_fc1.bias": "ffn_up.bias", "mlp.linear_fc2.weight": "ffn_down.weight", "mlp.linear_fc2.bias": "ffn_down.bias",
    };
    const to = map[blk[2]!];
    return to ? `v.blk.${blk[1]}.${to}` : undefined;
  }
  const map: Record<string, string> = {
    "patch_embed.proj.weight": "v.patch_embd.weight", "patch_embed.proj.bias": "v.patch_embd.bias", "pos_embed.weight": "v.position_embd.weight",
    "merger.norm.weight": "v.post_ln.weight", "merger.norm.bias": "v.post_ln.bias",
    "merger.linear_fc1.weight": "mm.0.weight", "merger.linear_fc1.bias": "mm.0.bias", "merger.linear_fc2.weight": "mm.2.weight", "merger.linear_fc2.bias": "mm.2.bias",
  };
  return map[n];
}
