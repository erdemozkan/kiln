import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** DETR (DetrConfig) with a timm ResNet backbone. */
export interface DetrConfig {
  dModel: number;
  encLayers: number;
  decLayers: number;
  encHeads: number;
  decHeads: number;
  queries: number;
  /** Bottleneck blocks per ResNet stage. */
  stages: number[];
  labels: string[];
}

const RESNET_STAGES: Record<string, number[]> = { resnet50: [3, 4, 6, 3], resnet101: [3, 4, 23, 3] };

export function detrConfigFromHF(config: Record<string, unknown>): DetrConfig {
  if (config.model_type !== "detr") throw new UnsupportedModelError(`model type "${String(config.model_type)}" is not DETR`);
  const backbone = String(config.backbone ?? "resnet50");
  const stages = RESNET_STAGES[backbone];
  if (config.use_timm_backbone === false || !stages) throw new UnsupportedModelError(`DETR backbone "${backbone}" is not supported (timm resnet50, resnet101)`);
  if (config.dilation === true) throw new UnsupportedModelError("DETR with a dilated backbone (DC5) is not supported");
  if ((config.position_embedding_type ?? "sine") !== "sine") throw new UnsupportedModelError("only DETR's sine position embeddings are supported");
  if ((config.activation_function ?? "relu") !== "relu") throw new UnsupportedModelError(`DETR activation "${String(config.activation_function)}" is not supported`);
  const n = (k: string, d: number) => Number(config[k] ?? d);
  const labels: string[] = [];
  for (const [k, v] of Object.entries((config.id2label as Record<string, string> | undefined) ?? {})) labels[Number(k)] = v;
  return {
    dModel: n("d_model", 256),
    encLayers: n("encoder_layers", 6),
    decLayers: n("decoder_layers", 6),
    encHeads: n("encoder_attention_heads", 8),
    decHeads: n("decoder_attention_heads", 8),
    queries: n("num_queries", 100),
    stages,
    labels,
  };
}

/**
 * Kiln's name for a DETR transformer or head tensor, or undefined for the backbone, the input
 * projection and the query embeddings (loaded otherwise).
 */
export function detrName(name: string): string | undefined {
  let m = /^model\.(encoder|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$/.exec(name);
  if (m) {
    const part: Record<string, string> = {
      "self_attn.q_proj": "q",
      "self_attn.k_proj": "k",
      "self_attn.v_proj": "v",
      "self_attn.out_proj": "o",
      "encoder_attn.q_proj": "cq",
      "encoder_attn.k_proj": "ck",
      "encoder_attn.v_proj": "cv",
      "encoder_attn.out_proj": "co",
      fc1: "up",
      fc2: "down",
      self_attn_layer_norm: "ln1",
      encoder_attn_layer_norm: "lnC",
      final_layer_norm: "ln2",
    };
    const p = part[m[3]!];
    return p ? `${m[1] === "encoder" ? "e" : "d"}${m[2]}.${p}.${m[4]}` : undefined;
  }
  m = /^model\.decoder\.layernorm\.(weight|bias)$/.exec(name);
  if (m) return `dlnF.${m[1]}`;
  if ((m = /^class_labels_classifier\.(weight|bias)$/.exec(name))) return `det.cls.0.${m[1]}`;
  if ((m = /^bbox_predictor\.layers\.(\d+)\.(weight|bias)$/.exec(name))) return `det.box.${m[1]}.${m[2]}`;
  return undefined;
}

/**
 * DETR's sine position embeddings for an h × w feature map with no padding, [h·w, dModel] in raster
 * order: per pixel the y half then the x half, each sin / cos pairs of the normalized cumulative
 * position (as DetrSinePositionEmbedding computes it in float32, normalize=True, scale 2π).
 */
export function detrSinePositions(h: number, w: number, dModel: number): Float32Array {
  const f = Math.fround;
  const half = dModel / 2;
  const dimT = Float32Array.from({ length: half }, (_, i) => f(10000 ** f((2 * Math.floor(i / 2)) / half)));
  const scale = f(2 * Math.PI);
  const embed = (n: number) => Float32Array.from({ length: n }, (_, i) => f(f((i + 1) / f(n + 1e-6)) * scale));
  const ye = embed(h);
  const xe = embed(w);
  const out = new Float32Array(h * w * dModel);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * dModel;
      for (let i = 0; i < half; i++) {
        const a = f(ye[y]! / dimT[i]!);
        const b = f(xe[x]! / dimT[i]!);
        out[o + i] = i % 2 ? Math.cos(a) : Math.sin(a);
        out[o + half + i] = i % 2 ? Math.cos(b) : Math.sin(b);
      }
    }
  return out;
}

/** Input channels of the first convolution: RGB padded to 4 for the conv kernel's vec4 loads. */
export const DETR_IN_CHANNELS = 4;

/**
 * A convolution weight [O, C, k, k] with its frozen batch norm folded in, as the conv kernel takes
 * it: w [O, K] tap-major with channels padded to `cp` and K to a multiple of 32, and the bias.
 */
export function foldConvBn(w: Float32Array, shape: readonly number[], bn: { weight: Float32Array; bias: Float32Array; mean: Float32Array; var: Float32Array }, cp = shape[1]!, eps = 1e-5): { w: Float32Array; b: Float32Array; K: number } {
  const [O, C, kh, kw] = shape as [number, number, number, number];
  const taps = kh * kw;
  const K = Math.ceil((taps * cp) / 32) * 32;
  const out = new Float32Array(O * K);
  const b = new Float32Array(O);
  for (let o = 0; o < O; o++) {
    const s = Math.fround(bn.weight[o]! / Math.sqrt(bn.var[o]! + eps));
    b[o] = bn.bias[o]! - bn.mean[o]! * s;
    for (let c = 0; c < C; c++) for (let t = 0; t < taps; t++) out[o * K + t * cp + c] = w[(o * C + c) * taps + t]! * s;
  }
  return { w: out, b, K };
}

interface DetrPlan {
  dispatches: readonly Dispatch[];
  input: Buf;
  logits: Tensor;
  boxes: Tensor;
  ctx: Ctx;
  owned: Buf[];
}

/**
 * DETR: ResNet backbone (convolutions with folded batch norm, weights "bb.*" [O, K] and biases) →
 * 1×1 input projection → post-norm encoder over the feature map with sine position embeddings
 * added to queries and keys → post-norm decoder of `queries` learned query embeddings (self- and
 * cross-attention, the encoder keys again with positions) → final LayerNorm → class and box MLPs.
 * One plan per image size (two kept).
 */
export class DetrModel {
  private readonly plans = new Map<string, DetrPlan>();
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: DetrConfig,
    private readonly weights: Map<string, Tensor>,
  ) {
    this.pool = new ActivationPool(backend, "detr");
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`DETR weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string, residual?: Tensor): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
  }

  private ln(ctx: Ctx, x: Tensor, p: string): Tensor {
    return ops.layernorm(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), 1e-5);
  }

  private build(H: number, W: number): DetrPlan {
    const c = this.cfg;
    const be = this.backend;
    const owned: Buf[] = [];
    const own = (data: Float32Array | Uint32Array, label: string) => {
      const b = be.alloc(data.byteLength, "storage", label);
      be.write(b, 0, data);
      owned.push(b);
      return b;
    };
    // Every attention here sees all its keys: a span covering every position.
    const step = own(new Uint32Array(STEP_BYTES / 4), "detr/step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: 0x7fffffff }));
    const ctx = new Ctx(be, step, `detr/${H}x${W}`, this.pool);
    const input = ctx.buffer(H * W * DETR_IN_CHANNELS * 4, "detr/pixels");

    // Backbone.
    const conv = (x: { y: Tensor; H: number; W: number }, p: string, k: number, stride: number, relu: boolean, residual?: Tensor) =>
      ops.conv2d(ctx, x.y, this.w(`${p}.weight`), this.w(`${p}.bias`), { H: x.H, W: x.W, k, stride, pad: (k - 1) / 2, relu, residual });
    let x = conv({ y: { shape: [H * W, DETR_IN_CHANNELS], dtype: "f32", buf: input }, H, W }, "bb.conv1", 7, 2, true);
    x = ops.maxPool2d(ctx, x.y, { H: x.H, W: x.W, k: 3, stride: 2, pad: 1 });
    c.stages.forEach((blocks, s) => {
      for (let b = 0; b < blocks; b++) {
        const p = `bb.layer${s + 1}.${b}`;
        const stride = b === 0 && s > 0 ? 2 : 1;
        const identity = b === 0 ? conv(x, `${p}.down`, 1, stride, false).y : x.y;
        const h1 = conv(x, `${p}.conv1`, 1, 1, true);
        const h2 = conv(h1, `${p}.conv2`, 3, stride, true);
        x = conv(h2, `${p}.conv3`, 1, 1, true, identity);
      }
    });

    // Transformer.
    const S = x.H * x.W;
    const D = c.dModel;
    const pos: Tensor = { shape: [S, D], dtype: "f32", buf: own(detrSinePositions(x.H, x.W, D), "detr/pos") };
    const Q = c.queries;
    const qpos = this.w("query.pos");
    const rows = Math.max(S, Q);
    const kc = own(new Float32Array(rows * D), "detr/k");
    const vc = own(new Float32Array(rows * D), "detr/v");
    const attend = (q: Tensor, k: Tensor, v: Tensor, T: number, keys: number, heads: number) => {
      ops.kvWrite(ctx, k, v, kc, vc, keys, keys, "f32");
      const hd = D / heads;
      return ops.attention(ctx, q, kc, vc, { T, H: heads, Hkv: heads, D: hd, maxSeq: keys, kv: "f32", scale: hd ** -0.5, span: true });
    };
    let h = this.lin(ctx, x.y, "proj");
    for (let l = 0; l < c.encLayers; l++) {
      const p = `e${l}`;
      const qk = ops.add(ctx, h, pos);
      this.lin(ctx, attend(this.lin(ctx, qk, `${p}.q`), this.lin(ctx, qk, `${p}.k`), this.lin(ctx, h, `${p}.v`), S, S, c.encHeads), `${p}.o`, h);
      h = this.ln(ctx, h, `${p}.ln1`);
      const up = this.lin(ctx, h, `${p}.up`);
      ops.act(ctx, up, "relu");
      this.lin(ctx, up, `${p}.down`, h);
      h = this.ln(ctx, h, `${p}.ln2`);
    }
    const memory = h;
    const memoryPos = ops.add(ctx, memory, pos);
    // The decoder starts from zeros.
    let t = ctx.alloc([Q, D]);
    ops.scaleAdd(ctx, t, qpos, 0, false);
    for (let l = 0; l < c.decLayers; l++) {
      const p = `d${l}`;
      const qk = ops.add(ctx, t, qpos);
      this.lin(ctx, attend(this.lin(ctx, qk, `${p}.q`), this.lin(ctx, qk, `${p}.k`), this.lin(ctx, t, `${p}.v`), Q, Q, c.decHeads), `${p}.o`, t);
      t = this.ln(ctx, t, `${p}.ln1`);
      const q = this.lin(ctx, ops.add(ctx, t, qpos), `${p}.cq`);
      this.lin(ctx, attend(q, this.lin(ctx, memoryPos, `${p}.ck`), this.lin(ctx, memory, `${p}.cv`), Q, S, c.decHeads), `${p}.co`, t);
      t = this.ln(ctx, t, `${p}.lnC`);
      const up = this.lin(ctx, t, `${p}.up`);
      ops.act(ctx, up, "relu");
      this.lin(ctx, up, `${p}.down`, t);
      t = this.ln(ctx, t, `${p}.ln2`);
    }
    const out = this.ln(ctx, t, "dlnF");
    const logits = this.lin(ctx, out, "det.cls.0");
    let boxes = out;
    for (let i = 0; this.weights.has(`det.box.${i}.weight`); i++) {
      if (i) ops.act(ctx, boxes, "relu");
      boxes = this.lin(ctx, boxes, `det.box.${i}`);
    }
    return { dispatches: ctx.dispatches, input, logits, boxes, ctx, owned };
  }

  /**
   * Class logits [queries, classes + 1] and boxes [queries, 4] (center x, y, width, height before the
   * sigmoid) for pixel values [3, height, width].
   */
  async detect(pixels: Float32Array, width: number, height: number): Promise<{ logits: Float32Array; boxes: Float32Array; classes: number }> {
    const n = width * height;
    if (pixels.length !== 3 * n) throw new Error(`DETR: expected 3 × ${height} × ${width} pixel values, got ${pixels.length}`);
    if (width < 32 || height < 32) throw new Error(`DETR: a ${width} × ${height} image is smaller than the backbone's 32-pixel stride`);
    const key = `${height}x${width}`;
    let plan = this.plans.get(key);
    if (!plan) {
      if (this.plans.size >= 2) {
        const [k, old] = this.plans.entries().next().value!;
        this.free(old);
        this.plans.delete(k);
      }
      plan = this.build(height, width);
      this.plans.set(key, plan);
    }
    const nhwc = new Float32Array(n * DETR_IN_CHANNELS);
    for (let ch = 0; ch < 3; ch++) for (let i = 0; i < n; i++) nhwc[i * DETR_IN_CHANNELS + ch] = pixels[ch * n + i]!;
    const be = this.backend;
    be.write(plan.input, 0, nhwc);
    be.submit(plan.dispatches);
    const Q = this.cfg.queries;
    const classes = plan.logits.shape[1]!;
    const [logits, boxes] = await Promise.all([be.readback(physical(plan.logits.buf), 0, Q * classes * 4), be.readback(physical(plan.boxes.buf), 0, Q * 4 * 4)]);
    return { logits: new Float32Array(logits), boxes: new Float32Array(boxes), classes };
  }

  private free(p: DetrPlan): void {
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
