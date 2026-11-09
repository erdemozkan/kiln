import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** SegFormer (SegformerConfig): a Mix Transformer encoder of four stages and the all-MLP decode head. */
export interface SegformerConfig {
  hidden: number[];
  depths: number[];
  heads: number[];
  srRatios: number[];
  mlpRatios: number[];
  patchSizes: number[];
  strides: number[];
  decoderHidden: number;
  labels: string[];
}

export function segformerConfigFromHF(config: Record<string, unknown>): SegformerConfig {
  if (config.model_type !== "segformer") throw new UnsupportedModelError(`model type "${String(config.model_type)}" is not SegFormer`);
  if (!((config.architectures as string[] | undefined) ?? []).includes("SegformerForSemanticSegmentation")) {
    throw new UnsupportedModelError("only SegformerForSemanticSegmentation checkpoints are supported");
  }
  if ((config.hidden_act ?? "gelu") !== "gelu") throw new UnsupportedModelError(`SegFormer activation "${String(config.hidden_act)}" is not supported`);
  if (config.reshape_last_stage === false) throw new UnsupportedModelError("SegFormer with reshape_last_stage=false is not supported");
  const arr = (k: string, d: number[]) => ((config[k] as number[] | undefined) ?? d).map(Number);
  const labels: string[] = [];
  for (const [k, v] of Object.entries((config.id2label as Record<string, string> | undefined) ?? {})) labels[Number(k)] = v;
  return {
    hidden: arr("hidden_sizes", [32, 64, 160, 256]),
    depths: arr("depths", [2, 2, 2, 2]),
    heads: arr("num_attention_heads", [1, 2, 5, 8]),
    srRatios: arr("sr_ratios", [8, 4, 2, 1]),
    mlpRatios: arr("mlp_ratios", [4, 4, 4, 4]),
    patchSizes: arr("patch_sizes", [7, 3, 3, 3]),
    strides: arr("strides", [4, 2, 2, 2]),
    decoderHidden: Number(config.decoder_hidden_size ?? 256),
    labels,
  };
}

/** Kiln's name for a SegFormer matrix, bias or norm loaded as a weight, or undefined for the convolutions (loaded otherwise). */
export function segformerName(name: string): string | undefined {
  let m = /^segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)\.(weight|bias)$/.exec(name);
  if (m) {
    const part: Record<string, string> = {
      layer_norm_1: "ln1",
      layer_norm_2: "ln2",
      "attention.self.query": "q",
      "attention.self.key": "k",
      "attention.self.value": "v",
      "attention.self.layer_norm": "srLn",
      "attention.output.dense": "o",
      "mlp.dense1": "up",
      "mlp.dense2": "down",
    };
    const p = part[m[3]!];
    return p ? `b${m[1]}.${m[2]}.${p}.${m[4]}` : undefined;
  }
  if ((m = /^segformer\.encoder\.patch_embeddings\.(\d+)\.layer_norm\.(weight|bias)$/.exec(name))) return `s${m[1]}.patchLn.${m[2]}`;
  if ((m = /^segformer\.encoder\.layer_norm\.(\d+)\.(weight|bias)$/.exec(name))) return `s${m[1]}.ln.${m[2]}`;
  if ((m = /^decode_head\.linear_c\.(\d+)\.proj\.(weight|bias)$/.exec(name))) return `dec.c${m[1]}.${m[2]}`;
  return undefined;
}

/** Input channels of the first patch convolution: RGB padded to 4. */
export const SEGFORMER_IN_CHANNELS = 4;

interface Plan {
  dispatches: readonly Dispatch[];
  input: Buf;
  logits: Tensor;
  H: number;
  W: number;
  ctx: Ctx;
  step: Buf;
  kc: Buf;
  vc: Buf;
}

/**
 * SegFormer on the GPU. Each stage: overlapping patch convolution + LayerNorm, then blocks of
 * efficient self-attention (keys and values from a strided "sequence reduction" convolution +
 * LayerNorm when sr > 1) and Mix-FFN (dense, depthwise 3×3, GELU, dense), then LayerNorm. The decode
 * head projects each stage to the decoder width, resizes to the first stage's size, and fuses them
 * with one 1×1 convolution per stage (the concatenation, split; batch norm folded; ReLU) before the
 * classifier. `segment` resizes the class scores to the requested size and takes each pixel's best
 * class in one kernel. All LayerNorms use torch's default eps (1e-5), as transformers' code does.
 */
export class SegformerModel {
  private readonly plans = new Map<string, Plan>();
  private readonly argmax = new Map<string, { dispatches: readonly Dispatch[]; out: Tensor; ctx: Ctx }>();
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: SegformerConfig,
    private readonly weights: Map<string, Tensor>,
  ) {
    this.pool = new ActivationPool(backend, "segformer");
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`SegFormer weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string, residual?: Tensor): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
  }

  private ln(ctx: Ctx, x: Tensor, p: string): Tensor {
    return ops.layernorm(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), 1e-5);
  }

  private build(H: number, W: number): Plan {
    const c = this.cfg;
    const be = this.backend;
    const step = be.alloc(STEP_BYTES, "storage", "segformer/step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: 0x7fffffff }));
    const ctx = new Ctx(be, step, `segformer/${H}x${W}`, this.pool);
    const input = ctx.buffer(H * W * SEGFORMER_IN_CHANNELS * 4, "segformer/pixels");
    // Largest key/value set: every stage's reduced token count times its width.
    let kvMax = 0;
    {
      let [h, w] = [H, W];
      c.hidden.forEach((C, s) => {
        const k = c.patchSizes[s]!;
        h = ops.convOut(h, k, c.strides[s]!, Math.floor(k / 2));
        w = ops.convOut(w, k, c.strides[s]!, Math.floor(k / 2));
        const r = c.srRatios[s]!;
        kvMax = Math.max(kvMax, (r > 1 ? ops.convOut(h, r, r, 0) * ops.convOut(w, r, r, 0) : h * w) * C);
      });
    }
    const kc = be.alloc(kvMax * 4, "storage", "segformer/k");
    const vc = be.alloc(kvMax * 4, "storage", "segformer/v");

    let x = { y: { shape: [H * W, SEGFORMER_IN_CHANNELS], dtype: "f32", buf: input } as Tensor, H, W };
    const feats: { y: Tensor; H: number; W: number }[] = [];
    c.hidden.forEach((C, s) => {
      const k = c.patchSizes[s]!;
      const e = ops.conv2d(ctx, x.y, this.w(`s${s}.patch.weight`), this.w(`s${s}.patch.bias`), { H: x.H, W: x.W, k, stride: c.strides[s]!, pad: Math.floor(k / 2) });
      let h = this.ln(ctx, e.y, `s${s}.patchLn`);
      const N = e.H * e.W;
      const heads = c.heads[s]!;
      const D = C / heads;
      const r = c.srRatios[s]!;
      for (let j = 0; j < c.depths[s]!; j++) {
        const p = `b${s}.${j}`;
        const n1 = this.ln(ctx, h, `${p}.ln1`);
        const q = this.lin(ctx, n1, `${p}.q`);
        let kv = n1;
        let S = N;
        if (r > 1) {
          const red = ops.conv2d(ctx, n1, this.w(`${p}.sr.weight`), this.w(`${p}.sr.bias`), { H: e.H, W: e.W, k: r, stride: r, pad: 0 });
          kv = this.ln(ctx, red.y, `${p}.srLn`);
          S = red.H * red.W;
        }
        ops.kvWrite(ctx, this.lin(ctx, kv, `${p}.k`), this.lin(ctx, kv, `${p}.v`), kc, vc, S, S, "f32");
        const a = ops.attention(ctx, q, kc, vc, { T: N, H: heads, Hkv: heads, D, maxSeq: S, kv: "f32", scale: D ** -0.5, span: true });
        this.lin(ctx, a, `${p}.o`, h);
        const up = this.lin(ctx, this.ln(ctx, h, `${p}.ln2`), `${p}.up`);
        const mixed = ops.dwConv(ctx, up, this.w(`${p}.dw.weight`), this.w(`${p}.dw.bias`), { H: e.H, W: e.W });
        ops.act(ctx, mixed, "gelu");
        this.lin(ctx, mixed, `${p}.down`, h);
      }
      h = this.ln(ctx, h, `s${s}.ln`);
      x = { y: h, H: e.H, W: e.W };
      feats.push(x);
    });

    // Decode head: the concatenation (last stage first) as one 1×1 convolution per stage, summed.
    const [H0, W0] = [feats[0]!.H, feats[0]!.W];
    let fused: Tensor | undefined;
    const n = feats.length;
    for (let k = 0; k < n; k++) {
      const s = n - 1 - k;
      const f = feats[s]!;
      let proj = this.lin(ctx, f.y, `dec.c${s}`);
      if (f.H !== H0 || f.W !== W0) proj = ops.resizeBilinear(ctx, proj, { H: f.H, W: f.W, Ho: H0, Wo: W0, alignCorners: false });
      fused = ops.conv2d(ctx, proj, this.w(`dec.fuse.p${k}.weight`), this.w(`dec.fuse.p${k}.bias`), { H: H0, W: W0, k: 1, stride: 1, pad: 0, residual: fused, relu: k === n - 1 }).y;
    }
    const logits = this.lin(ctx, fused!, "dec.cls");
    return { dispatches: ctx.dispatches, input, logits, H: H0, W: W0, ctx, step, kc, vc };
  }

  /** Each pixel's class at outWidth × outHeight (scores resized to it) for pixel values [3, height, width]. */
  async segment(pixels: Float32Array, width: number, height: number, outWidth: number, outHeight: number): Promise<Uint32Array> {
    const n = width * height;
    if (pixels.length !== 3 * n) throw new Error(`SegFormer: expected 3 × ${height} × ${width} pixel values, got ${pixels.length}`);
    const key = `${height}x${width}`;
    let plan = this.plans.get(key);
    if (!plan) {
      this.clear();
      plan = this.build(height, width);
      this.plans.set(key, plan);
    }
    const akey = `${key}->${outHeight}x${outWidth}`;
    let am = this.argmax.get(akey);
    if (!am) {
      for (const a of this.argmax.values()) a.ctx.destroy();
      this.argmax.clear();
      // Its own pool: the shared one could place its output on the scores it reads.
      const ctx = new Ctx(this.backend, plan.step, `segformer/argmax`);
      const logits: Tensor = { shape: plan.logits.shape, dtype: "f32", buf: physical(plan.logits.buf) };
      const out = ops.resizeArgmax(ctx, logits, { H: plan.H, W: plan.W, Ho: outHeight, Wo: outWidth });
      am = { dispatches: ctx.dispatches, out, ctx };
      this.argmax.set(akey, am);
    }
    const nhwc = new Float32Array(n * SEGFORMER_IN_CHANNELS);
    for (let ch = 0; ch < 3; ch++) for (let i = 0; i < n; i++) nhwc[i * SEGFORMER_IN_CHANNELS + ch] = pixels[ch * n + i]!;
    const be = this.backend;
    be.write(plan.input, 0, nhwc);
    be.submit(plan.dispatches);
    be.submit(am.dispatches);
    return new Uint32Array(await be.readback(physical(am.out.buf), 0, outWidth * outHeight * 4));
  }

  private clear(): void {
    for (const a of this.argmax.values()) a.ctx.destroy();
    this.argmax.clear();
    for (const p of this.plans.values()) {
      p.ctx.destroy();
      for (const b of [p.step, p.kc, p.vc]) this.backend.free(b);
    }
    this.plans.clear();
  }

  destroy(): void {
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    this.clear();
    this.pool.destroy();
  }
}
