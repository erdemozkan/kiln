import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** Segment Anything (transformers' SamModel): a windowed ViT image encoder, a prompt encoder and a two-way mask decoder. */
export interface SamConfig {
  hidden: number;
  layers: number;
  heads: number;
  mlp: number;
  /** Side of the attention windows; the global layers attend over the whole grid. */
  window: number;
  globalLayers: number[];
  imageSize: number;
  patchSize: number;
  /** Channels of the image embedding (the neck's output). */
  channels: number;
  eps: number;
  qkvBias: boolean;
  dec: { hidden: number; heads: number; mlp: number; layers: number; downsample: number; masks: number; eps: number };
}

export function samConfigFromHF(config: Record<string, unknown>): SamConfig {
  if (config.model_type !== "sam") throw new UnsupportedModelError(`model type "${String(config.model_type)}" is not SAM`);
  const v = (config.vision_config as Record<string, unknown> | undefined) ?? {};
  const d = (config.mask_decoder_config as Record<string, unknown> | undefined) ?? {};
  const n = (o: Record<string, unknown>, k: string, def: number) => Number(o[k] ?? def);
  if (v.use_rel_pos === false || v.use_abs_pos === false) throw new UnsupportedModelError("SAM without relative and absolute position embeddings is not supported");
  if ((v.hidden_act ?? "gelu") !== "gelu" || (d.hidden_act ?? "relu") !== "relu") throw new UnsupportedModelError("SAM with other activations is not supported");
  const cfg: SamConfig = {
    hidden: n(v, "hidden_size", 768),
    layers: n(v, "num_hidden_layers", 12),
    heads: n(v, "num_attention_heads", 12),
    mlp: n(v, "mlp_dim", 3072),
    window: n(v, "window_size", 14),
    globalLayers: ((v.global_attn_indexes as number[] | undefined) ?? [2, 5, 8, 11]).map(Number),
    imageSize: n(v, "image_size", 1024),
    patchSize: n(v, "patch_size", 16),
    channels: n(v, "output_channels", 256),
    eps: n(v, "layer_norm_eps", 1e-6),
    qkvBias: v.qkv_bias !== false,
    dec: {
      hidden: n(d, "hidden_size", 256),
      heads: n(d, "num_attention_heads", 8),
      mlp: n(d, "mlp_dim", 2048),
      layers: n(d, "num_hidden_layers", 2),
      downsample: n(d, "attention_downsample_rate", 2),
      masks: n(d, "num_multimask_outputs", 3) + 1,
      eps: n(d, "layer_norm_eps", 1e-6),
    },
  };
  if (n(d, "iou_head_depth", 3) !== 3) throw new UnsupportedModelError("SAM's IoU head must have 3 layers");
  return cfg;
}

/** Kiln's name for a vision-encoder tensor loaded as a weight (matrices that may be quantized, and norms), or undefined for those loaded otherwise. */
export function samName(name: string): string | undefined {
  const m = /^vision_encoder\.layers\.(\d+)\.(.+)\.(weight|bias)$/.exec(name);
  if (!m) return undefined;
  const part: Record<string, string> = { layer_norm1: "ln1", layer_norm2: "ln2", "attn.qkv": "qkv", "attn.proj": "o", "mlp.lin1": "up", "mlp.lin2": "down" };
  const p = part[m[2]!];
  return p ? `l${m[1]}.${p}.${m[3]}` : undefined;
}

/** The grid of image tokens: imageSize / patchSize on a side. */
export const samGrid = (c: SamConfig): number => c.imageSize / c.patchSize;

/** One pixel-value image [3, imageSize, imageSize] as patch rows [grid², 3·p·p] (channel, then row, then column within the patch), as the patch convolution's weight is laid out. */
export function samPatchRows(c: SamConfig, px: Float32Array): Float32Array {
  const p = c.patchSize;
  const n = c.imageSize;
  const g = n / p;
  const K = 3 * p * p;
  const out = new Float32Array(g * g * K);
  for (let gy = 0; gy < g; gy++)
    for (let gx = 0; gx < g; gx++) {
      const o = (gy * g + gx) * K;
      for (let ch = 0; ch < 3; ch++)
        for (let ky = 0; ky < p; ky++) {
          const src = (ch * n + gy * p + ky) * n + gx * p;
          out.set(px.subarray(src, src + p), o + (ch * p + ky) * p);
        }
    }
  return out;
}

/**
 * transformers' SamPositionalEmbedding of points already normalized to [0, 1]: x and y mapped to
 * [-1, 1], times the random matrix gaussian [2, F], times 2π; then the F sines and the F cosines.
 */
export function samPositional(gaussian: Float32Array, x: number, y: number): Float32Array {
  const F = gaussian.length / 2;
  const f = Math.fround;
  const out = new Float32Array(2 * F);
  const cx = f(f(2 * f(x)) - 1);
  const cy = f(f(2 * f(y)) - 1);
  for (let i = 0; i < F; i++) {
    const v = f(f(2 * Math.PI) * f(f(cx * gaussian[i]!) + f(cy * gaussian[F + i]!)));
    out[i] = Math.sin(v);
    out[F + i] = Math.cos(v);
  }
  return out;
}

/** The positional embedding of every cell of the image embedding's grid [side², 2F], cell centers in raster order. */
export function samImagePositional(gaussian: Float32Array, side: number): Float32Array {
  const F = gaussian.length / 2;
  const out = new Float32Array(side * side * 2 * F);
  for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) out.set(samPositional(gaussian, Math.fround((x + 0.5) / side), Math.fround((y + 0.5) / side)), (y * side + x) * 2 * F);
  return out;
}

/** Prompts in the model's input frame (pixels of the resized image, before padding to imageSize). */
export interface SamPrompt {
  /** Points [x, y] with labels: 1 the object, 0 not the object, -1 background (no point). */
  points?: { x: number; y: number; label: number }[];
  /** Boxes [x1, y1, x2, y2]. */
  boxes?: [number, number, number, number][];
}

/** The prompt encoder's weights the CPU needs: SamPromptEncoder's embeddings. */
export interface SamPromptWeights {
  gaussian: Float32Array;
  /** point_embed.0 … 3: negative point, positive point, box top-left, box bottom-right. */
  pointEmbed: Float32Array[];
  notAPoint: Float32Array;
}

/**
 * The sparse prompt embeddings [n, hidden] as SamPromptEncoder makes them: each point shifted to its
 * pixel's center and normalized by the input size, its positional embedding plus its label's
 * embedding (a background point is the not-a-point embedding alone); without boxes a padding point
 * (the not-a-point embedding) closes the points; then each box's two corners.
 */
export function samSparsePrompt(w: SamPromptWeights, prompt: SamPrompt, imageSize: number): Float32Array[] {
  const rows: Float32Array[] = [];
  const at = (x: number, y: number) => samPositional(w.gaussian, Math.fround(Math.fround(x + 0.5) / imageSize), Math.fround(Math.fround(y + 0.5) / imageSize));
  const plus = (a: Float32Array, b: Float32Array) => a.map((v, i) => v + b[i]!);
  const points = prompt.points ?? [];
  if (points.length) {
    for (const p of points) {
      if (p.label === 1 || p.label === 0) rows.push(plus(at(p.x, p.y), w.pointEmbed[p.label]!));
      else if (p.label === -1) rows.push(Float32Array.from(w.notAPoint));
      else throw new Error(`SAM: a point's label is 1 (object), 0 (not the object) or -1 (background), got ${p.label}`);
    }
    if (!prompt.boxes?.length) rows.push(Float32Array.from(w.notAPoint));
  }
  for (const [x1, y1, x2, y2] of prompt.boxes ?? []) {
    rows.push(plus(at(x1, y1), w.pointEmbed[2]!));
    rows.push(plus(at(x2, y2), w.pointEmbed[3]!));
  }
  if (!rows.length) throw new Error("SAM: give at least one point or box");
  return rows;
}

interface Plan {
  dispatches: readonly Dispatch[];
  input: Buf;
  ctx: Ctx;
  owned: Buf[];
}

interface DecoderPlan extends Plan {
  masks: Tensor;
  iou: Tensor;
}

/**
 * SAM on Kiln's kernels. The image encoder runs once per image and leaves the image embedding
 * [grid², channels] on the device (its own activation pool, so decoding does not overwrite it); the
 * mask decoder then runs per prompt, one plan per token count, and reads back only the low-resolution
 * mask logits and the IoU scores.
 */
export class SamModel {
  private encoder: (Plan & { out: Tensor; cuts: number[] }) | undefined;
  private readonly decoders = new Map<number, DecoderPlan>();
  private readonly encPool: ActivationPool;
  private readonly decPool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: SamConfig,
    private readonly weights: Map<string, Tensor>,
    /** The IoU token and the mask tokens [1 + masks, hidden], which open every decoder input. */
    private readonly outputTokens: Float32Array,
  ) {
    this.encPool = new ActivationPool(backend, "sam/enc");
    this.decPool = new ActivationPool(backend, "sam/dec");
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`sam weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string, residual?: Tensor): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
  }

  private ln(ctx: Ctx, x: Tensor, p: string, eps: number): Tensor {
    return ops.layernorm(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), eps);
  }

  private step(owned: Buf[], spanHi: number, label: string): Buf {
    const step = this.backend.alloc(STEP_BYTES, "storage", label);
    owned.push(step);
    this.backend.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi }));
    return step;
  }

  private buildEncoder(): Plan & { out: Tensor; cuts: number[] } {
    const c = this.cfg;
    const be = this.backend;
    const g = samGrid(c);
    const T = g * g;
    const D = c.hidden / c.heads;
    const owned: Buf[] = [];
    const ctx = new Ctx(be, this.step(owned, T, "sam/enc-step"), "sam/encoder", this.encPool);
    const K = 3 * c.patchSize * c.patchSize;
    const input = ctx.buffer(T * K * 4, "sam/patches");
    let x = ops.linear(ctx, { shape: [T, K], dtype: "f32", buf: input }, this.w("patch.weight"), {});
    ops.addInPlace(ctx, x, this.w("embed.add"));
    // Where each layer ends: the encoder goes to the GPU a layer per command buffer, since all of it in
    // one runs past the GPU watchdog (seconds) on integrated graphics.
    const cuts: number[] = [];
    for (let l = 0; l < c.layers; l++) {
      cuts.push(ctx.size);
      const p = `l${l}`;
      const win = c.globalLayers.includes(l) ? g : c.window;
      const h = this.ln(ctx, x, `${p}.ln1`, c.eps);
      const qkv = this.lin(ctx, h, `${p}.qkv`);
      const s = ops.samWindow(ctx, qkv, this.w(`${p}.rh`), this.w(`${p}.rw`), this.w(`${p}.qkv.bias`), { G: g, W: g, win, heads: c.heads, D });
      const a = ops.attentionBatched(ctx, s.q, s.k.buf, s.v.buf, { B: s.B, L: s.L, H: c.heads, D: s.DP, scale: 1, lens: this.w(`lens.${win}`).buf });
      this.lin(ctx, ops.samUnwindow(ctx, a, { G: g, W: g, win, heads: c.heads, D, L: s.L, DP: s.DP, nwx: s.nwx }), `${p}.o`, x);
      const up = this.lin(ctx, this.ln(ctx, x, `${p}.ln2`, c.eps), `${p}.up`);
      ops.gelu(ctx, up, true);
      this.lin(ctx, up, `${p}.down`, x);
    }
    // Neck: 1×1 convolution, LayerNorm over channels, 3×3 convolution, LayerNorm.
    x = this.ln(ctx, ops.linear(ctx, x, this.w("neck.conv1.weight"), {}), "neck.ln1", 1e-6);
    x = ops.conv2d(ctx, x, this.w("neck.conv2.weight"), this.w("neck.zero"), { H: g, W: g, k: 3, stride: 1, pad: 1 }).y;
    const out = this.ln(ctx, x, "neck.ln2", 1e-6);
    return { dispatches: ctx.dispatches, input, out, ctx, owned, cuts: [...cuts.slice(1), ctx.size] };
  }

  /** SamAttention: projections to the inner width, multi-head attention of qIn's rows over kIn's, the output projection (added into `residual` when given). */
  private attn(ctx: Ctx, p: string, qIn: Tensor, kIn: Tensor, vIn: Tensor, residual?: Tensor): Tensor {
    const d = this.cfg.dec;
    const q = this.lin(ctx, qIn, `${p}.q_proj`);
    const k = this.lin(ctx, kIn, `${p}.k_proj`);
    const v = this.lin(ctx, vIn, `${p}.v_proj`);
    const inner = q.shape[1]!;
    const D = inner / d.heads;
    const a = ops.attention(ctx, q, k.buf, v.buf, { T: q.shape[0]!, H: d.heads, Hkv: d.heads, D, maxSeq: k.shape[0]!, kv: "f32", scale: D ** -0.5, span: true });
    return this.lin(ctx, a, `${p}.out_proj`, residual);
  }

  /** SamFeedForward with three layers: ReLU after the first two. */
  private ff(ctx: Ctx, x: Tensor, p: string): Tensor {
    const h = this.lin(ctx, x, `${p}.proj_in`);
    ops.act(ctx, h, "relu");
    const h2 = this.lin(ctx, h, `${p}.layers.0`);
    ops.act(ctx, h2, "relu");
    return this.lin(ctx, h2, `${p}.proj_out`);
  }

  private buildDecoder(tokens: number, image: Tensor): DecoderPlan {
    const c = this.cfg;
    const d = c.dec;
    const be = this.backend;
    const g = samGrid(c);
    const owned: Buf[] = [];
    // Every attention here sees all its keys: the span covers the image's cells (and so every token).
    const ctx = new Ctx(be, this.step(owned, g * g, "sam/dec-step"), `sam/decoder${tokens}`, this.decPool);
    const input = ctx.buffer(tokens * d.hidden * 4, "sam/tokens");
    const pe: Tensor = { shape: [tokens, d.hidden], dtype: "f32", buf: input };
    const imagePe = this.w("dec.image_pe");
    const t = "dec.transformer";
    // The image embedding plus the dense prompt (no mask: no_mask_embed in every cell).
    let keys = ops.add(ctx, image, this.w("dec.no_mask"));
    let queries = pe;
    for (let l = 0; l < d.layers; l++) {
      const p = `${t}.layers.${l}`;
      if (l === 0) {
        // The first layer's self-attention replaces the tokens (no residual, no positional embedding).
        queries = this.ln(ctx, this.attn(ctx, `${p}.self_attn`, queries, queries, queries), `${p}.layer_norm1`, d.eps);
      } else {
        const qpe = ops.add(ctx, queries, pe);
        this.attn(ctx, `${p}.self_attn`, qpe, qpe, queries, queries);
        queries = this.ln(ctx, queries, `${p}.layer_norm1`, d.eps);
      }
      let qpe = ops.add(ctx, queries, pe);
      let kpe = ops.add(ctx, keys, imagePe);
      this.attn(ctx, `${p}.cross_attn_token_to_image`, qpe, kpe, keys, queries);
      queries = this.ln(ctx, queries, `${p}.layer_norm2`, d.eps);
      const up = this.lin(ctx, queries, `${p}.mlp.lin1`);
      ops.act(ctx, up, "relu");
      this.lin(ctx, up, `${p}.mlp.lin2`, queries);
      queries = this.ln(ctx, queries, `${p}.layer_norm3`, d.eps);
      qpe = ops.add(ctx, queries, pe);
      kpe = ops.add(ctx, keys, imagePe);
      this.attn(ctx, `${p}.cross_attn_image_to_token`, kpe, qpe, queries, keys);
      keys = this.ln(ctx, keys, `${p}.layer_norm4`, d.eps);
    }
    const qpe = ops.add(ctx, queries, pe);
    const kpe = ops.add(ctx, keys, imagePe);
    this.attn(ctx, `${t}.final_attn_token_to_image`, qpe, kpe, keys, queries);
    // nn.LayerNorm's default eps.
    queries = this.ln(ctx, queries, `${t}.layer_norm_final_attn`, 1e-5);
    // Upscaling: two 2×2 transposed convolutions with a channel LayerNorm and GELUs, to 4× the grid.
    const u1 = ops.convTranspose(ctx, keys, this.w("dec.upscale1.weight"), this.w("dec.upscale1.bias"), { H: g, W: g, f: 2 });
    const n1 = this.ln(ctx, u1.y, "dec.upscale_layer_norm", 1e-6);
    ops.gelu(ctx, n1, true);
    const u2 = ops.convTranspose(ctx, n1, this.w("dec.upscale2.weight"), this.w("dec.upscale2.bias"), { H: u1.H, W: u1.W, f: 2 });
    ops.gelu(ctx, u2.y, true);
    // Each mask token through its own hypernetwork MLP (run on every token row, its own row kept).
    const hyper = ops.pickRows(
      ctx,
      Array.from({ length: d.masks }, (_, i) => this.ff(ctx, queries, `dec.output_hypernetworks_mlps.${i}`)),
      Array.from({ length: d.masks }, (_, i) => 1 + i),
    );
    const masks = ops.linear(ctx, u2.y, hyper, {});
    const iou = this.ff(ctx, queries, "dec.iou_prediction_head");
    return { dispatches: ctx.dispatches, input, masks, iou, ctx, owned };
  }

  /** Encodes pixel values [3, imageSize, imageSize] (resized, normalized, padded): the image embedding stays on the device for `decode`. */
  encode(pixels: Float32Array): void {
    const c = this.cfg;
    if (pixels.length !== 3 * c.imageSize * c.imageSize) throw new Error(`sam: expected 3 × ${c.imageSize} × ${c.imageSize} pixel values, got ${pixels.length}`);
    this.encoder ??= this.buildEncoder();
    this.backend.write(this.encoder.input, 0, samPatchRows(c, pixels));
    const d = this.encoder.dispatches;
    let from = 0;
    for (const to of this.encoder.cuts) {
      this.backend.submit(d.slice(from, to));
      from = to;
    }
  }

  /** The image embedding [grid², channels] of the last `encode`, read back (for checks). */
  async imageEmbedding(): Promise<Float32Array> {
    if (!this.encoder) throw new Error("sam: no image encoded yet");
    const n = samGrid(this.cfg) ** 2 * this.cfg.channels;
    return new Float32Array(await this.backend.readback(physical(this.encoder.out.buf), 0, n * 4));
  }

  /** Runs the mask decoder for sparse prompt embeddings on the last encoded image; returns its plan, with the logits not yet read. */
  private runDecoder(sparse: readonly Float32Array[]): DecoderPlan {
    if (!this.encoder) throw new Error("sam: encode an image first");
    const d = this.cfg.dec;
    const tokens = 1 + d.masks + sparse.length;
    let plan = this.decoders.get(tokens);
    if (!plan) {
      if (this.decoders.size >= 4) {
        const [k, old] = this.decoders.entries().next().value!;
        this.free(old);
        this.decoders.delete(k);
      }
      // The encoder's output buffer itself: it lives in the encoder's pool, which decoding never touches.
      const image: Tensor = { shape: this.encoder.out.shape, dtype: "f32", buf: physical(this.encoder.out.buf) };
      plan = this.buildDecoder(tokens, image);
      this.decoders.set(tokens, plan);
    }
    const input = new Float32Array(tokens * d.hidden);
    input.set(this.outputTokens, 0);
    sparse.forEach((r, i) => input.set(r, (1 + d.masks + i) * d.hidden));
    this.backend.write(plan.input, 0, input);
    this.backend.submit(plan.dispatches);
    return plan;
  }

  /**
   * The mask decoder for sparse prompt embeddings (`samSparsePrompt`) on the last encoded image:
   * low-resolution mask logits [masks, (4·grid)²] (all mask tokens, the single-mask one first) and
   * their IoU predictions [masks].
   */
  async decode(sparse: readonly Float32Array[]): Promise<{ masks: Float32Array[]; iou: Float32Array; side: number }> {
    const d = this.cfg.dec;
    const plan = this.runDecoder(sparse);
    const side = 4 * samGrid(this.cfg);
    const [m, iou] = await Promise.all([this.backend.readback(physical(plan.masks.buf), 0, side * side * d.masks * 4), this.backend.readback(physical(plan.iou.buf), 0, d.masks * 4)]);
    const flat = new Float32Array(m);
    const masks = Array.from({ length: d.masks }, (_, i) => {
      const out = new Float32Array(side * side);
      for (let j = 0; j < out.length; j++) out[j] = flat[j * d.masks + i]!;
      return out;
    });
    return { masks, iou: new Float32Array(iou), side };
  }

  /**
   * As `decode`, with SamProcessor's post-processing on the device: masks first … first + count − 1
   * resized to the input size, cropped to the resized image (cw × ch), resized to width × height and
   * thresholded, read back as bytes (255 inside), with all the IoU predictions.
   */
  async segment(sparse: readonly Float32Array[], o: { first: number; count: number; cw: number; ch: number; width: number; height: number; threshold: number }): Promise<{ masks: Uint8Array[]; iou: Float32Array }> {
    const d = this.cfg.dec;
    const plan = this.runDecoder(sparse);
    const side = 4 * samGrid(this.cfg);
    const key = `${o.first}/${o.count}/${o.cw}x${o.ch}/${o.width}x${o.height}/${o.threshold}`;
    if (this.post?.key !== key) {
      if (this.post) this.free(this.post);
      const owned: Buf[] = [];
      // Its own pool: the decoder's IoU output, read after this runs, lives in the decoder's.
      const ctx = new Ctx(this.backend, this.step(owned, 1, "sam/post-step"), "sam/post");
      const input = ctx.buffer(side * side * d.masks * 4, "sam/low");
      const out = ops.samMasks(ctx, { shape: [side * side, d.masks], dtype: "f32", buf: input }, { side, first: o.first, count: o.count, inSize: this.cfg.imageSize, cw: o.cw, ch: o.ch, W: o.width, H: o.height, threshold: o.threshold });
      this.post = { key, dispatches: ctx.dispatches, input, out, ctx, owned };
    }
    const post = this.post;
    this.backend.copy(physical(plan.masks.buf), 0, post.input, 0, side * side * d.masks * 4);
    this.backend.submit(post.dispatches);
    const words = post.out.shape[1]!;
    const [bytes, iou] = await Promise.all([this.backend.readback(physical(post.out.buf), 0, o.count * words * 4), this.backend.readback(physical(plan.iou.buf), 0, d.masks * 4)]);
    const all = new Uint8Array(bytes);
    const n = o.width * o.height;
    return { masks: Array.from({ length: o.count }, (_, k) => all.slice(k * words * 4, k * words * 4 + n)), iou: new Float32Array(iou) };
  }

  /** The post-processing plan for the last image size and mask choice. */
  private post: (Plan & { key: string; out: Tensor }) | undefined;

  private free(p: Plan): void {
    p.ctx.destroy();
    for (const b of p.owned) this.backend.free(b);
  }

  destroy(): void {
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    if (this.encoder) this.free(this.encoder);
    if (this.post) this.free(this.post);
    for (const p of this.decoders.values()) this.free(p);
    this.encPool.destroy();
    this.decPool.destroy();
  }
}
