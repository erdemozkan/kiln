import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** Gemma 3's vision tower (SigLIP) and multimodal projector, as llama.cpp's mmproj GGUF describes them. */
export interface VisionConfig {
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

/** Patch features padded to a multiple of 32 (3 · 14 · 14 = 588 → 608) for the matmul kernels. */
export const patchK = (c: VisionConfig): number => Math.ceil((3 * c.patchSize * c.patchSize) / 32) * 32;

/** An image as RGBA (or RGB) bytes, already `imageSize` square. */
export interface ImagePixels {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray;
}

/**
 * Patch rows [patches, patchK] from an image: each patch's values in (channel, row, column) order,
 * as the convolution weight [hidden, 3, p, p] flattens, normalised with the config's mean and std.
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
  for (let py = 0; py < side; py++)
    for (let px = 0; px < side; px++) {
      const o = (py * side + px) * K;
      for (let k = 0; k < 3; k++)
        for (let y = 0; y < P; y++)
          for (let x = 0; x < P; x++) {
            const v = img.data[((py * P + y) * S + px * P + x) * ch + k]! / 255;
            out[o + (k * P + y) * P + x] = (v - c.mean[k]!) / c.std[k]!;
          }
    }
  return out;
}

/**
 * SigLIP encoder plus Gemma 3's projector: patches → 27 pre-norm transformer layers with full
 * attention → post LayerNorm → 4 × 4 average pool → RMSNorm → projection to the text width. Weights
 * use llama.cpp's mmproj names; the patch weight comes padded to `patchK` and the projection
 * transposed to [projectionDim, hidden] (see the loader). Attention is the text model's: k and v go
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
  ) {
    const n = this.patches;
    this.step = backend.alloc(STEP_BYTES, "storage", "vision/step");
    backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: n }));
    this.kc = backend.alloc(n * cfg.hidden * 4, "storage", "vision/k");
    this.vc = backend.alloc(n * cfg.hidden * 4, "storage", "vision/v");
    this.pool = new ActivationPool(backend, "vision");
  }

  get patches(): number {
    return (this.cfg.imageSize / this.cfg.patchSize) ** 2;
  }

  /** Image tokens per image (256 for Gemma 3). */
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
    const shape = { T: n, H: c.heads, Hkv: c.heads, D, maxSeq: n, kv: "f32" as const, scale: D ** -0.5 };
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
    const pooled = ops.avgPool(ctx, x, c.imageSize / c.patchSize, c.imageSize / c.patchSize / c.tokensPerSide);
    // Gemma's RMSNorm weight is stored as 1 + w in the GGUF.
    const normed = ops.rmsnorm(ctx, pooled, this.w("mm.soft_emb_norm.weight"), c.eps);
    const features = ops.linear(ctx, normed, this.w("mm.input_projection.weight"));
    return { dispatches: ctx.dispatches, patches, features, ctx };
  }

  /**
   * Runs the tower on one image and copies its `tokens` × projectionDim features into `dst` at byte
   * offset `dstOffset`. The work goes out a layer at a time, waiting for each, so the GPU is never
   * handed seconds of work at once (a watchdog may reset it) and the page stays responsive.
   */
  async encode(img: ImagePixels, dst: Buf, dstOffset: number): Promise<void> {
    this.plan ??= this.build();
    this.backend.write(this.plan.patches, 0, imagePatches(this.cfg, img));
    const d = this.plan.dispatches;
    const per = Math.max(1, Math.ceil(d.length / (this.cfg.layers + 2)));
    for (let i = 0; i < d.length; i += per) {
      this.backend.submit(d.slice(i, i + per));
      await this.backend.readback(this.step, 0, 4);
    }
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
