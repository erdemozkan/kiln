import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** Vision transformers from transformers' configs: ViT, DeiT (with its distillation token), CLIP's vision tower, YOLOS. */
export interface VitConfig {
  modelType: string;
  /** CLIP: LayerNorm right after the embeddings ("pre_layrnorm"), and none over every token at the end. */
  preLn: boolean;
  finalLn: boolean;
  architectures: string[];
  hidden: number;
  layers: number;
  heads: number;
  intermediate: number;
  imageSize: number;
  /** The image size the position embeddings were trained at, [height, width] (YOLOS: not square). */
  imageHW: [number, number];
  patchSize: number;
  channels: number;
  eps: number;
  act: "gelu" | "gelu_tanh" | "quick_gelu" | "relu";
  /** Tokens before the patches: CLS, DeiT's distillation token, YOLOS's detection tokens (placed first). */
  prefix: number;
  /** YOLOS: detection tokens (rows 0 … detTokens - 1, ahead of CLS; full attention is order-free). */
  detTokens: number;
  /** YOLOS: position embeddings added again after every layer but the last. */
  midPos: boolean;
  labels: string[];
}

export const VIT_TYPES = new Set(["vit", "deit", "clip", "clip_vision_model", "yolos", "dinov2"]);

/** CLIPVisionConfig's defaults, for configs that leave fields out. */
const CLIP_VISION_DEFAULTS = { hidden_size: 768, intermediate_size: 3072, num_hidden_layers: 12, num_attention_heads: 12, image_size: 224, patch_size: 32, hidden_act: "quick_gelu", layer_norm_eps: 1e-5 };

export function vitConfigFromHF(config: Record<string, unknown>): VitConfig {
  const outer = String(config.model_type ?? "");
  if (!VIT_TYPES.has(outer)) throw new UnsupportedModelError(`vision model type "${outer}" is not supported (ViT, DeiT, CLIP, YOLOS, DINOv2)`);
  if (outer === "dinov2" && config.use_swiglu_ffn === true) throw new UnsupportedModelError("DINOv2 with a SwiGLU MLP is not supported");
  const clip = outer.startsWith("clip");
  const c: Record<string, unknown> = clip ? { ...CLIP_VISION_DEFAULTS, ...((outer === "clip" ? config.vision_config : config) as Record<string, unknown>) } : config;
  const type = clip ? "clip" : outer;
  const num = (k: string, d?: number) => {
    const v = c[k] ?? d;
    if (v === undefined) throw new UnsupportedModelError(`config.json is missing "${k}"`);
    return Number(v);
  };
  const actName = String(c.hidden_act ?? "gelu");
  const act = actName === "gelu" ? "gelu" : actName === "quick_gelu" ? "quick_gelu" : actName === "relu" ? "relu" : actName.startsWith("gelu") ? "gelu_tanh" : undefined;
  if (!act) throw new UnsupportedModelError(`activation "${actName}" is not supported`);
  if (num("num_channels", 3) !== 3) throw new UnsupportedModelError("only 3-channel images are supported");
  const id2label = (c.id2label as Record<string, string> | undefined) ?? {};
  const labels: string[] = [];
  for (const [k, v] of Object.entries(id2label)) labels[Number(k)] = v;
  const size = c.image_size ?? 224;
  const imageHW: [number, number] = Array.isArray(size) ? [Number(size[0]), Number(size[1])] : [Number(size), Number(size)];
  const detTokens = type === "yolos" ? num("num_detection_tokens", 100) : 0;
  return {
    modelType: type,
    preLn: clip,
    finalLn: !clip,
    architectures: (config.architectures as string[] | undefined) ?? [],
    hidden: num("hidden_size"),
    layers: num("num_hidden_layers"),
    heads: num("num_attention_heads"),
    // DINOv2 gives an MLP ratio instead.
    intermediate: type === "dinov2" ? num("hidden_size") * num("mlp_ratio", 4) : num("intermediate_size"),
    imageSize: imageHW[0],
    imageHW,
    patchSize: num("patch_size", 16),
    channels: 3,
    eps: num("layer_norm_eps", 1e-12),
    act,
    prefix: type === "deit" ? 2 : 1 + detTokens,
    detTokens,
    midPos: type === "yolos" && c.use_mid_position_embeddings === true,
    labels,
  };
}

/** Kiln's name for a ViT / DeiT / CLIP vision tensor (matrices, biases, norms), or undefined for those loaded otherwise. */
export function vitName(name: string): string | undefined {
  let m = /^vision_model\.(.+)$/.exec(name);
  if (m) {
    const n = m[1]!;
    m = /^encoder\.layers\.(\d+)\.(.+)\.(weight|bias)$/.exec(n);
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
      return p ? `l${m[1]}.${p}.${m[3]}` : undefined;
    }
    m = /^pre_layrnorm\.(weight|bias)$/.exec(n);
    return m ? `lnPre.${m[1]}` : undefined;
  }
  if (/^(text_model|text_projection|visual_projection|logit_scale)/.test(name)) return undefined;
  const n = name.replace(/^(vit|deit|dinov2)\./, "");
  m = /^encoder\.layer\.(\d+)\.(.+)\.(weight|bias)$/.exec(n);
  if (m) {
    const part: Record<string, string> = {
      "attention.attention.query": "q",
      "attention.attention.key": "k",
      "attention.attention.value": "v",
      "attention.output.dense": "o",
      "intermediate.dense": "up",
      "output.dense": "down",
      layernorm_before: "ln1",
      layernorm_after: "ln2",
      // DINOv2 (its output projections are scaled by the layer scales when loaded, not here).
      norm1: "ln1",
      norm2: "ln2",
      "mlp.fc1": "up",
    };
    const p = part[m[2]!];
    return p ? `l${m[1]}.${p}.${m[3]}` : undefined;
  }
  m = /^layernorm\.(weight|bias)$/.exec(n);
  if (m) return `lnF.${m[1]}`;
  // YOLOS: the class and box MLPs.
  m = /^(class_labels_classifier|bbox_predictor)\.layers\.(\d+)\.(weight|bias)$/.exec(n);
  if (m) return `det.${m[1] === "bbox_predictor" ? "box" : "cls"}.${m[2]}.${m[3]}`;
  return undefined;
}

/** Patch features padded to a multiple of 32 for the matmul kernels. */
export const vitPatchK = (c: Pick<VitConfig, "patchSize" | "channels">): number => Math.ceil((c.channels * c.patchSize * c.patchSize) / 32) * 32;

/** Patch grid [rows, columns] of an image (a convolution with stride = kernel drops the remainder). */
export const vitGrid = (c: Pick<VitConfig, "patchSize">, width: number, height: number): [number, number] => [Math.floor(height / c.patchSize), Math.floor(width / c.patchSize)];

/**
 * Rows [prefix + patches, patchK] from pixel values [3, height, width]: `prefix` zero rows (the CLS,
 * distillation and detection tokens come from the addend), then each patch in (channel, row,
 * column) order as the convolution weight [hidden, 3, p, p] flattens.
 */
export function vitPatchRows(c: VitConfig, pixels: Float32Array, width = c.imageSize, height = width): Float32Array {
  const P = c.patchSize;
  const [gh, gw] = vitGrid(c, width, height);
  const K = vitPatchK(c);
  const n = width * height;
  const out = new Float32Array((c.prefix + gh * gw) * K);
  for (let py = 0; py < gh; py++)
    for (let px = 0; px < gw; px++) {
      const o = (c.prefix + py * gw + px) * K;
      for (let ch = 0; ch < 3; ch++)
        for (let y = 0; y < P; y++)
          for (let x = 0; x < P; x++) out[o + (ch * P + y) * P + x] = pixels[ch * n + (py * P + y) * width + px * P + x]!;
    }
  return out;
}

/**
 * Bicubic resize of a [C, h, w] grid to [C, oh, ow] as torch's interpolate(mode="bicubic",
 * align_corners=False) does it (A = -0.75, edge indices clamped). YOLOS fits its position
 * embeddings to the image's patch grid with it.
 */
export function bicubicResize(src: Float32Array, C: number, h: number, w: number, oh: number, ow: number): Float32Array {
  const A = -0.75;
  const cubic1 = (x: number) => ((A + 2) * x - (A + 3)) * x * x + 1;
  const cubic2 = (x: number) => ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
  const taps = (inSize: number, outSize: number) =>
    Array.from({ length: outSize }, (_, i) => {
      const real = (i + 0.5) * (inSize / outSize) - 0.5;
      const i0 = Math.floor(real);
      const t = real - i0;
      const wts = [cubic2(t + 1), cubic1(t), cubic1(1 - t), cubic2(2 - t)];
      const idx = [-1, 0, 1, 2].map((d) => Math.min(Math.max(i0 + d, 0), inSize - 1));
      return { idx, wts };
    });
  const ty = taps(h, oh);
  const tx = taps(w, ow);
  const out = new Float32Array(C * oh * ow);
  for (let c = 0; c < C; c++)
    for (let y = 0; y < oh; y++)
      for (let x = 0; x < ow; x++) {
        let s = 0;
        for (let a = 0; a < 4; a++) {
          let r = 0;
          for (let b = 0; b < 4; b++) r += tx[x]!.wts[b]! * src[(c * h + ty[y]!.idx[a]!) * w + tx[x]!.idx[b]!]!;
          s += ty[y]!.wts[a]! * r;
        }
        out[(c * oh + y) * ow + x] = s;
      }
  return out;
}

/** The per-grid addends of a model whose position embeddings follow the image size. */
export interface VitEmbeddings {
  /** [tokens, hidden]: position embeddings plus the prefix tokens and the patch bias. */
  add: Float32Array;
  /** YOLOS: [tokens, hidden] added after each layer but the last. */
  mid?: Float32Array[];
}

interface VitPlan {
  dispatches: readonly Dispatch[];
  input: Buf;
  out: Tensor;
  det?: { logits: Tensor; boxes: Tensor };
  ctx: Ctx;
  owned: Buf[];
  tokens: number;
}

/**
 * A pre-norm vision transformer encoder: patch rows → one matmul (the patch convolution, weight
 * "patch.weight" [hidden, patchK]) plus a [tokens, hidden] addend (CLS / distillation / detection
 * tokens, position embeddings and the patch bias) → layers of LayerNorm, full attention, LayerNorm,
 * MLP → final LayerNorm, and for YOLOS the class and box MLPs on the detection tokens. One plan per
 * patch grid (a few kept): the addend is the "embed.add" weight at the trained size, or comes from
 * `embeddings` for models that take any size.
 */
export class VitModel {
  private readonly plans = new Map<string, VitPlan>();
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: VitConfig,
    private readonly weights: Map<string, Tensor>,
    private readonly embeddings?: (gh: number, gw: number) => VitEmbeddings,
  ) {
    this.pool = new ActivationPool(backend, "vit");
  }

  /** Tokens at the trained image size. */
  get tokens(): number {
    const [gh, gw] = vitGrid(this.cfg, this.cfg.imageHW[1], this.cfg.imageHW[0]);
    return this.cfg.prefix + gh * gw;
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`vit weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string, residual?: Tensor): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
  }

  private mlp(ctx: Ctx, x: Tensor, p: string): Tensor {
    let h = x;
    for (let i = 0; this.weights.has(`${p}.${i}.weight`); i++) {
      if (i) ops.act(ctx, h, "relu");
      h = this.lin(ctx, h, `${p}.${i}`);
    }
    return h;
  }

  private build(gh: number, gw: number): VitPlan {
    const c = this.cfg;
    const be = this.backend;
    const T = c.prefix + gh * gw;
    const D = c.hidden / c.heads;
    const owned: Buf[] = [];
    const own = (bytes: number, label: string) => {
      const b = be.alloc(bytes, "storage", label);
      owned.push(b);
      return b;
    };
    const step = own(STEP_BYTES, "vit/step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: T }));
    const kc = own(T * c.hidden * 4, "vit/k");
    const vc = own(T * c.hidden * 4, "vit/v");
    const tensor = (data: Float32Array, label: string): Tensor => {
      const buf = own(data.byteLength, label);
      be.write(buf, 0, data);
      return { shape: [T, c.hidden], dtype: "f32", buf };
    };
    let add: Tensor;
    let mid: Tensor[] = [];
    if (this.embeddings) {
      const e = this.embeddings(gh, gw);
      add = tensor(e.add, "vit/embed");
      mid = (e.mid ?? []).map((m, i) => tensor(m, `vit/mid${i}`));
    } else {
      add = this.w("embed.add");
      if (add.shape[0] !== T) throw new Error(`vit: this model takes ${c.imageSize} × ${c.imageSize} images`);
    }
    const ctx = new Ctx(be, step, `vit/${gh}x${gw}`, this.pool);
    const input = ctx.buffer(T * vitPatchK(c) * 4, "vit/patches");
    let x = ops.linear(ctx, { shape: [T, vitPatchK(c)], dtype: "f32", buf: input }, this.w("patch.weight"), {});
    ops.addInPlace(ctx, x, add);
    if (c.preLn) x = ops.layernorm(ctx, x, this.w("lnPre.weight"), this.w("lnPre.bias"), c.eps);
    const shape = { T, H: c.heads, Hkv: c.heads, D, maxSeq: T, kv: "f32" as const, scale: D ** -0.5, span: true };
    for (let l = 0; l < c.layers; l++) {
      const p = `l${l}`;
      const h = ops.layernorm(ctx, x, this.w(`${p}.ln1.weight`), this.w(`${p}.ln1.bias`), c.eps);
      const q = this.lin(ctx, h, `${p}.q`);
      ops.kvWrite(ctx, this.lin(ctx, h, `${p}.k`), this.lin(ctx, h, `${p}.v`), kc, vc, T, T, "f32");
      this.lin(ctx, ops.attention(ctx, q, kc, vc, shape), `${p}.o`, x);
      const h2 = ops.layernorm(ctx, x, this.w(`${p}.ln2.weight`), this.w(`${p}.ln2.bias`), c.eps);
      const up = this.lin(ctx, h2, `${p}.up`);
      ops.act(ctx, up, c.act);
      this.lin(ctx, up, `${p}.down`, x);
      if (mid[l]) ops.addInPlace(ctx, x, mid[l]!);
    }
    const out = c.finalLn ? ops.layernorm(ctx, x, this.w("lnF.weight"), this.w("lnF.bias"), c.eps) : x;
    let det: VitPlan["det"];
    if (c.detTokens) {
      // The detection tokens are the first rows: a view of them is the same buffer with fewer rows.
      const tokens: Tensor = { shape: [c.detTokens, c.hidden], dtype: "f32", buf: out.buf };
      det = { logits: this.mlp(ctx, tokens, "det.cls"), boxes: this.mlp(ctx, tokens, "det.box") };
    }
    return { dispatches: ctx.dispatches, input, out, det, ctx, owned, tokens: T };
  }

  private run(pixels: Float32Array, width: number, height: number): VitPlan {
    const c = this.cfg;
    if (pixels.length !== 3 * width * height) throw new Error(`vit: expected 3 × ${height} × ${width} pixel values, got ${pixels.length}`);
    const [gh, gw] = vitGrid(c, width, height);
    if (!gh || !gw) throw new Error(`vit: a ${width} × ${height} image is smaller than one ${c.patchSize}-pixel patch`);
    const key = `${gh}x${gw}`;
    let plan = this.plans.get(key);
    if (!plan) {
      if (this.plans.size >= 3) {
        const [k, old] = this.plans.entries().next().value!;
        this.free(old);
        this.plans.delete(k);
      }
      plan = this.build(gh, gw);
      this.plans.set(key, plan);
    }
    this.backend.write(plan.input, 0, vitPatchRows(c, pixels, width, height));
    this.backend.submit(plan.dispatches);
    return plan;
  }

  /** Last hidden state [tokens, hidden] for pixel values [3, height, width] (default: the trained square size). */
  async forward(pixels: Float32Array, width = this.cfg.imageSize, height = width): Promise<Float32Array> {
    // With heads, the planner may reuse the hidden state's memory once they have read it.
    if (this.cfg.detTokens) throw new Error("vit: a detection model's hidden state is not kept; use detect()");
    const plan = this.run(pixels, width, height);
    return new Float32Array(await this.backend.readback(physical(plan.out.buf), 0, plan.tokens * this.cfg.hidden * 4));
  }

  /** YOLOS: class logits [detTokens, classes + 1] and boxes [detTokens, 4] (center x, y, width, height before the sigmoid). */
  async detect(pixels: Float32Array, width: number, height: number): Promise<{ logits: Float32Array; boxes: Float32Array; classes: number }> {
    const plan = this.run(pixels, width, height);
    const d = plan.det;
    if (!d) throw new Error("this model has no detection heads");
    const n = this.cfg.detTokens;
    const classes = d.logits.shape[1]!;
    const [logits, boxes] = await Promise.all([
      this.backend.readback(physical(d.logits.buf), 0, n * classes * 4),
      this.backend.readback(physical(d.boxes.buf), 0, n * 4 * 4),
    ]);
    return { logits: new Float32Array(logits), boxes: new Float32Array(boxes), classes };
  }

  private free(p: VitPlan): void {
    p.ctx.destroy();
    for (const b of p.owned) this.backend.free(b);
  }

  destroy(): void {
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    for (const p of this.plans.values()) this.free(p);
    this.pool.destroy();
  }
}
