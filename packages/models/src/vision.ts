import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/**
 * A SigLIP vision tower and its multimodal projector: Gemma 3's (as llama.cpp's mmproj GGUF describes
 * it) or Idefics3's (SmolVLM, from its safetensors).
 */
export interface VisionConfig {
  /**
   * "gemma3": post-norm → 4 × 4 average pool → RMSNorm → projection; image tokens attend both ways.
   * "idefics3": post-norm → pixel shuffle (scaleFactor² neighbours into one token) → projection;
   * image tokens are causal like text.
   */
  projector: "gemma3" | "idefics3";
  /** Patches per side merged into one image token. */
  scaleFactor: number;
  imageSize: number;
  patchSize: number;
  hidden: number;
  intermediate: number;
  layers: number;
  heads: number;
  eps: number;
  /** Text model width the projector maps to. */
  projectionDim: number;
  /** Image tokens per side after pooling (16: 256 tokens per image). */
  tokensPerSide: number;
  mean: [number, number, number];
  std: [number, number, number];
}

/** The vision config of a Gemma 3 mmproj GGUF (`clip` architecture, projector type gemma3). */
export function visionConfigFromGGUF(meta: Map<string, unknown>): VisionConfig {
  if (meta.get("general.architecture") !== "clip") throw new UnsupportedModelError(`mmproj: architecture "${String(meta.get("general.architecture"))}" is not clip`);
  const type = meta.get("clip.projector_type");
  if (type !== "gemma3") throw new UnsupportedModelError(`mmproj: projector type "${String(type)}" is not supported (gemma3 only)`);
  const num = (k: string, fallback?: number) => {
    const v = meta.get(`clip.vision.${k}`) ?? fallback;
    if (v === undefined) throw new UnsupportedModelError(`mmproj: metadata is missing "clip.vision.${k}"`);
    return Number(v);
  };
  const triple = (k: string): [number, number, number] => {
    const v = meta.get(`clip.vision.${k}`) as number[] | undefined;
    return v && v.length === 3 ? [v[0]!, v[1]!, v[2]!] : [0.5, 0.5, 0.5];
  };
  const imageSize = num("image_size");
  const patchSize = num("patch_size");
  return {
    projector: "gemma3",
    scaleFactor: 4,
    imageSize,
    patchSize,
    hidden: num("embedding_length"),
    intermediate: num("feed_forward_length"),
    layers: num("block_count"),
    heads: num("attention.head_count"),
    eps: num("attention.layer_norm_epsilon", 1e-6),
    projectionDim: num("projection_dim"),
    // Gemma 3 pools the patch grid 4 × 4 into 256 tokens.
    tokensPerSide: imageSize / patchSize / 4,
    mean: triple("image_mean"),
    std: triple("image_std"),
  };
}

/**
 * The vision config of an Idefics3 checkpoint (SmolVLM): `vision_config`, the pixel-shuffle
 * `scale_factor`, the text width, and the image processor's mean and std.
 */
export function visionConfigFromIdefics3(config: Record<string, unknown>, preprocessor?: Record<string, unknown>): VisionConfig {
  const v = config.vision_config as Record<string, unknown> | undefined;
  const t = config.text_config as Record<string, unknown> | undefined;
  if (!v || !t) throw new UnsupportedModelError("Idefics3: config.json needs vision_config and text_config");
  const act = String(v.hidden_act ?? "gelu_pytorch_tanh");
  if (act !== "gelu_pytorch_tanh") throw new UnsupportedModelError(`Idefics3 vision activation "${act}" is not supported (gelu_pytorch_tanh)`);
  const n = (c: Record<string, unknown>, k: string, d?: number) => {
    const x = c[k] ?? d;
    if (typeof x !== "number") throw new UnsupportedModelError(`Idefics3: config is missing "${k}"`);
    return x;
  };
  const imageSize = n(v, "image_size", 224);
  const patchSize = n(v, "patch_size", 32);
  const scaleFactor = n(config, "scale_factor", 2);
  const side = imageSize / patchSize;
  if (!Number.isInteger(side) || side % scaleFactor) throw new UnsupportedModelError(`Idefics3: ${side} patches per side do not split by scale_factor ${scaleFactor}`);
  const triple = (k: string): [number, number, number] => {
    const x = preprocessor?.[k] as number[] | undefined;
    return x && x.length === 3 ? [x[0]!, x[1]!, x[2]!] : [0.5, 0.5, 0.5];
  };
  return {
    projector: "idefics3",
    scaleFactor,
    imageSize,
    patchSize,
    hidden: n(v, "hidden_size", 768),
    intermediate: n(v, "intermediate_size", 3072),
    layers: n(v, "num_hidden_layers", 12),
    heads: n(v, "num_attention_heads", 12),
    eps: n(v, "layer_norm_eps", 1e-6),
    projectionDim: n(t, "hidden_size"),
    tokensPerSide: side / scaleFactor,
    mean: triple("image_mean"),
    std: triple("image_std"),
  };
}

/**
 * The order patches go through an Idefics3 tower: raster patch index per row. The tower treats patches
 * as a set (full attention, position embeddings row by row), so feeding them grouped by output token
 * — token (h', w') in raster order, then its s × s neighbours (dh, dw) — makes the post-norm rows,
 * read as [tokens, s² · hidden], exactly the connector's pixel shuffle. Gemma 3: raster order.
 */
export function patchOrder(c: VisionConfig): Int32Array {
  const side = c.imageSize / c.patchSize;
  const order = new Int32Array(side * side);
  if (c.projector !== "idefics3") return order.map((_, i) => i);
  const s = c.scaleFactor;
  const per = side / s;
  let r = 0;
  for (let th = 0; th < per; th++)
    for (let tw = 0; tw < per; tw++)
      for (let dh = 0; dh < s; dh++) for (let dw = 0; dw < s; dw++) order[r++] = (th * s + dh) * side + tw * s + dw;
  return order;
}

/** Idefics3's image processing (SmolVLM): how an image becomes tiles and prompt tokens. */
export interface Idefics3Processing {
  /** Side the image's longest edge is first resized to (`size.longest_edge`; 2048 for SmolVLM-256M). */
  longestEdge: number;
  /** Tile side (`max_image_size.longest_edge`), the tower's image size. */
  tileSize: number;
  /** Tiles plus a global view (`do_image_splitting`), or one square view of the whole image. */
  split: boolean;
  /** PIL resampling filter (1: Lanczos). */
  resample: number;
  /** The `<image>` token the template writes and the processor repeats per image token. */
  imageTokenId: number;
}

export function idefics3Processing(config: Record<string, unknown>, preprocessor: Record<string, unknown> | undefined, vision: VisionConfig): Idefics3Processing {
  const edge = (k: string, d: number) => Number((preprocessor?.[k] as { longest_edge?: number } | undefined)?.longest_edge ?? d);
  const tileSize = edge("max_image_size", vision.imageSize);
  if (tileSize !== vision.imageSize) throw new UnsupportedModelError(`Idefics3: tiles of ${tileSize} for a ${vision.imageSize} vision tower are not supported`);
  return {
    longestEdge: edge("size", 4 * tileSize),
    tileSize,
    split: preprocessor?.do_image_splitting !== false,
    resample: Number(preprocessor?.resample ?? 1),
    imageTokenId: Number(config.image_token_id ?? 49190),
  };
}

const IDEFICS3_BLOCK: Record<string, string> = {
  "self_attn.q_proj": "attn_q",
  "self_attn.k_proj": "attn_k",
  "self_attn.v_proj": "attn_v",
  "self_attn.out_proj": "attn_out",
  layer_norm1: "ln1",
  layer_norm2: "ln2",
  "mlp.fc1": "ffn_up",
  "mlp.fc2": "ffn_down",
};

/**
 * An Idefics3 (SmolVLM) checkpoint's vision and connector tensors under the mmproj names
 * `VisionTower` reads; undefined for the language model's.
 */
export function idefics3VisionName(name: string): string | undefined {
  const n = name.replace(/^model\./, "");
  const m = n.match(/^vision_model\.encoder\.layers\.(\d+)\.(.+)\.(weight|bias)$/);
  if (m) {
    const mapped = IDEFICS3_BLOCK[m[2]!];
    return mapped ? `v.blk.${m[1]}.${mapped}.${m[3]}` : undefined;
  }
  const global: Record<string, string> = {
    "vision_model.embeddings.patch_embedding.weight": "v.patch_embd.weight",
    "vision_model.embeddings.patch_embedding.bias": "v.patch_embd.bias",
    "vision_model.embeddings.position_embedding.weight": "v.position_embd.weight",
    "vision_model.post_layernorm.weight": "v.post_ln.weight",
    "vision_model.post_layernorm.bias": "v.post_ln.bias",
    "connector.modality_projection.proj.weight": "mm.input_projection.weight",
  };
  return global[n];
}

/** An Idefics3 (SmolVLM) checkpoint's language-model tensors under Llama's names; undefined for the rest. */
export function idefics3TextName(name: string): string | undefined {
  if (name.startsWith("model.text_model.")) return `model.${name.slice("model.text_model.".length)}`;
  return name === "lm_head.weight" ? name : undefined;
}

/** Patch features padded to a multiple of 32 (3 · 14 · 14 = 588 → 608) for the matmul kernels. */
export const patchK = (c: VisionConfig): number => Math.ceil((3 * c.patchSize * c.patchSize) / 32) * 32;

/** An image as RGBA (or RGB) bytes, already `imageSize` square. */
export interface ImagePixels {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray;
}

/**
 * Patch rows [patches, patchK] from an image, in `patchOrder`: each patch's values in (channel, row,
 * column) order, as the convolution weight [hidden, 3, p, p] flattens, normalised with the config's
 * mean and std.
 */
export function imagePatches(c: VisionConfig, img: ImagePixels): Float32Array {
  const S = c.imageSize;
  const P = c.patchSize;
  if (img.width !== S || img.height !== S) throw new Error(`image is ${img.width}×${img.height}; the vision tower takes ${S}×${S}`);
  const ch = img.data.length / (S * S);
  if (ch !== 3 && ch !== 4) throw new Error("image data must be RGB or RGBA bytes");
  const side = S / P;
  const K = patchK(c);
  const out = new Float32Array(side * side * K);
  const order = patchOrder(c);
  for (let r = 0; r < order.length; r++) {
    const py = Math.floor(order[r]! / side);
    const px = order[r]! % side;
    {
      const o = r * K;
      for (let k = 0; k < 3; k++)
        for (let y = 0; y < P; y++)
          for (let x = 0; x < P; x++) {
            const v = img.data[((py * P + y) * S + px * P + x) * ch + k]! / 255;
            out[o + (k * P + y) * P + x] = (v - c.mean[k]!) / c.std[k]!;
          }
    }
  }
  return out;
}

/**
 * SigLIP encoder plus a projector: patches → pre-norm transformer layers with full attention → post
 * LayerNorm → Gemma 3's 4 × 4 average pool, RMSNorm and projection, or Idefics3's pixel shuffle (free:
 * see `patchOrder`) and projection to the text width. Weights use llama.cpp's mmproj names (the
 * loader renames Idefics3's); the patch weight comes padded to `patchK`, the projection as
 * [projectionDim, input width], and Idefics3's position embeddings in `patchOrder`. Attention is the text model's: k and v go
 * through a 'cache' of one slot per patch, and a bidirectional span over every patch makes it full.
 */
export class VisionTower {
  private plan: { dispatches: readonly Dispatch[]; patches: Buf; features: Tensor; ctx: Ctx } | undefined;
  private readonly step: Buf;
  private readonly kc: Buf;
  private readonly vc: Buf;
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: VisionConfig,
    private readonly weights: Map<string, Tensor>,
    /** Idefics3's image processing; the engine splits images into tiles with it. */
    readonly processing?: Idefics3Processing,
  ) {
    const n = this.patches;
    this.step = backend.alloc(STEP_BYTES, "storage", "vision/step");
    backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: n }));
    this.kc = backend.alloc(n * cfg.hidden * 4, "storage", "vision/k");
    this.vc = backend.alloc(n * cfg.hidden * 4, "storage", "vision/v");
    this.pool = new ActivationPool(backend, "vision");
  }

  /** Feature rows one image can need: Gemma 3's 256, or Idefics3's tokens for the most tiles plus the global view. */
  get tokensPerImage(): number {
    const p = this.processing;
    if (!p?.split) return this.tokens;
    return (Math.ceil(p.longestEdge / p.tileSize) ** 2 + 1) * this.tokens;
  }

  /** Whether an image's tokens attend to each other both ways (Gemma 3) or causally (Idefics3). */
  get bidirectional(): boolean {
    return this.cfg.projector === "gemma3";
  }

  get patches(): number {
    return (this.cfg.imageSize / this.cfg.patchSize) ** 2;
  }

  /** Image tokens per image (256 for Gemma 3), or per tile for Idefics3 (64 for SmolVLM-256M). */
  get tokens(): number {
    return this.cfg.tokensPerSide ** 2;
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`vision weight "${name}" was not loaded`);
    return t;
  }

  private build() {
    const c = this.cfg;
    const n = this.patches;
    const D = c.hidden / c.heads;
    const ctx = new Ctx(this.backend, this.step, "vision", this.pool);
    const patches = ctx.buffer(n * patchK(c) * 4, "vision/patches");
    let x = ops.linear(ctx, { shape: [n, patchK(c)], dtype: "f32", buf: patches }, this.w("v.patch_embd.weight"), { bias: this.w("v.patch_embd.bias") });
    ops.addInPlace(ctx, x, this.w("v.position_embd.weight"));
    // Every patch sees every patch: span attention, 0.88 s per layer on the Intel Mac against 2.76 s
    // with the text prefill's tiled kernel.
    const shape = { T: n, H: c.heads, Hkv: c.heads, D, maxSeq: n, kv: "f32" as const, scale: D ** -0.5, span: true };
    for (let l = 0; l < c.layers; l++) {
      const p = `v.blk.${l}.`;
      const h = ops.layernorm(ctx, x, this.w(p + "ln1.weight"), this.w(p + "ln1.bias"), c.eps);
      const q = ops.linear(ctx, h, this.w(p + "attn_q.weight"), this.w(p + "attn_q.bias"));
      const k = ops.linear(ctx, h, this.w(p + "attn_k.weight"), this.w(p + "attn_k.bias"));
      const v = ops.linear(ctx, h, this.w(p + "attn_v.weight"), this.w(p + "attn_v.bias"));
      ops.kvWrite(ctx, k, v, this.kc, this.vc, n, n, "f32");
      const a = ops.attention(ctx, q, this.kc, this.vc, shape);
      ops.linear(ctx, a, this.w(p + "attn_out.weight"), { bias: this.w(p + "attn_out.bias"), residual: x });
      const h2 = ops.layernorm(ctx, x, this.w(p + "ln2.weight"), this.w(p + "ln2.bias"), c.eps);
      const up = ops.linear(ctx, h2, this.w(p + "ffn_up.weight"), this.w(p + "ffn_up.bias"));
      ops.gelu(ctx, up);
      ops.linear(ctx, up, this.w(p + "ffn_down.weight"), { bias: this.w(p + "ffn_down.bias"), residual: x });
    }
    x = ops.layernorm(ctx, x, this.w("v.post_ln.weight"), this.w("v.post_ln.bias"), c.eps);
    if (c.projector === "idefics3") {
      const shuffled: Tensor = { ...x, shape: [this.tokens, c.hidden * c.scaleFactor ** 2] };
      const features = ops.linear(ctx, shuffled, this.w("mm.input_projection.weight"));
      return { dispatches: ctx.dispatches, patches, features, ctx };
    }
    const pooled = ops.avgPool(ctx, x, c.imageSize / c.patchSize, c.imageSize / c.patchSize / c.tokensPerSide);
    // Gemma's RMSNorm weight is stored as 1 + w in the GGUF.
    const normed = ops.rmsnorm(ctx, pooled, this.w("mm.soft_emb_norm.weight"), c.eps);
    const features = ops.linear(ctx, normed, this.w("mm.input_projection.weight"));
    return { dispatches: ctx.dispatches, patches, features, ctx };
  }

  /**
   * Runs the tower on one image and copies its `tokens` × projectionDim features into `dst` at byte
   * offset `dstOffset`. The work goes out a layer at a time, at most two layers queued, so the GPU is
   * never handed seconds of work at once (a watchdog may reset it) and the page stays responsive, but
   * always has the next layer waiting (waiting on each layer left it idle 0.3 s of SmolVLM's 1.3 s
   * per tile on the Intel Mac).
   */
  async encode(img: ImagePixels, dst: Buf, dstOffset: number): Promise<void> {
    this.plan ??= this.build();
    this.backend.write(this.plan.patches, 0, imagePatches(this.cfg, img));
    const d = this.plan.dispatches;
    const per = Math.max(1, Math.ceil(d.length / (this.cfg.layers + 2)));
    let pending: Promise<ArrayBuffer> | undefined;
    for (let i = 0; i < d.length; i += per) {
      this.backend.submit(d.slice(i, i + per));
      const done = this.backend.readback(this.step, 0, 4);
      if (pending) await pending;
      pending = done;
    }
    await pending;
    this.backend.copy(physical(this.plan.features.buf), 0, dst, dstOffset, this.tokens * this.cfg.projectionDim * 4);
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
    this.plan?.ctx.destroy();
    this.pool.destroy();
    for (const b of [this.step, this.kc, this.vc]) this.backend.free(b);
  }
}
