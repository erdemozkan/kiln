import { ActivationPool, Ctx, STEP_BYTES, lastDim, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** An image of channels-last features on the GPU. */
export interface ConvImage {
  y: Tensor;
  H: number;
  W: number;
}

/** Input channels of a convnet's first convolution: RGB padded to 4 for the conv kernel. */
export const CONVNET_IN_CHANNELS = 4;

/** ResNet (transformers' ResNetConfig, ResNetForImageClassification). */
export interface ResNetConfig {
  kind: "resnet";
  embedding: number;
  hidden: number[];
  depths: number[];
  bottleneck: boolean;
  downsampleInFirstStage: boolean;
  downsampleInBottleneck: boolean;
}

/** ConvNeXt (transformers' ConvNextConfig, ConvNextForImageClassification). */
export interface ConvNextConfig {
  kind: "convnext";
  hidden: number[];
  depths: number[];
  patch: number;
  /** The pooled output's LayerNorm (the stages' own norms use 1e-6). */
  eps: number;
}

export type ConvNetConfig = ResNetConfig | ConvNextConfig;

export function convNetConfigFromHF(config: Record<string, unknown>): ConvNetConfig {
  const arr = (k: string) => (config[k] as number[] | undefined)?.map(Number);
  if (config.model_type === "resnet") {
    if ((config.hidden_act ?? "relu") !== "relu") throw new UnsupportedModelError(`ResNet activation "${String(config.hidden_act)}" is not supported`);
    return {
      kind: "resnet",
      embedding: Number(config.embedding_size ?? 64),
      hidden: arr("hidden_sizes") ?? [256, 512, 1024, 2048],
      depths: arr("depths") ?? [3, 4, 6, 3],
      bottleneck: (config.layer_type ?? "bottleneck") === "bottleneck",
      downsampleInFirstStage: config.downsample_in_first_stage === true,
      downsampleInBottleneck: config.downsample_in_bottleneck === true,
    };
  }
  if (config.model_type === "convnext") {
    if ((config.hidden_act ?? "gelu") !== "gelu") throw new UnsupportedModelError(`ConvNeXt activation "${String(config.hidden_act)}" is not supported`);
    return { kind: "convnext", hidden: arr("hidden_sizes") ?? [96, 192, 384, 768], depths: arr("depths") ?? [3, 3, 9, 3], patch: Number(config.patch_size ?? 4), eps: Number(config.layer_norm_eps ?? 1e-12) };
  }
  throw new UnsupportedModelError(`model type "${String(config.model_type)}" is not a supported convnet (ResNet, ConvNeXt)`);
}

/**
 * Every convolution a convnet loads, by module path, with what the loader needs: "conv" (ResNet's
 * conv + batch norm, folded), "conv2" (a plain conv with bias), "dw" (depthwise k × k), plus the
 * input channels the first convolution is padded to.
 */
export function convNetConvs(c: ConvNetConfig): { path: string; to: string; kind: "convbn" | "conv" | "dw"; cp?: number }[] {
  const out: { path: string; to: string; kind: "convbn" | "conv" | "dw"; cp?: number }[] = [];
  if (c.kind === "resnet") {
    out.push({ path: "resnet.embedder.embedder", to: "stem", kind: "convbn", cp: CONVNET_IN_CHANNELS });
    let cin = c.embedding;
    c.hidden.forEach((cout, s) => {
      for (let l = 0; l < c.depths[s]!; l++) {
        const p = `resnet.encoder.stages.${s}.layers.${l}`;
        const stride = l === 0 && (s > 0 || c.downsampleInFirstStage) ? 2 : 1;
        if (l === 0 && (cin !== cout || stride !== 1)) out.push({ path: `${p}.shortcut`, to: `s${s}.${l}.short`, kind: "convbn" });
        const n = c.bottleneck ? 3 : 2;
        for (let i = 0; i < n; i++) out.push({ path: `${p}.layer.${i}`, to: `s${s}.${l}.c${i}`, kind: "convbn" });
        cin = cout;
      }
    });
  } else {
    out.push({ path: "convnext.embeddings.patch_embeddings", to: "stem", kind: "conv", cp: CONVNET_IN_CHANNELS });
    c.hidden.forEach((C, s) => {
      if (s > 0) out.push({ path: `convnext.encoder.stages.${s}.downsampling_layer.1`, to: `s${s}.down`, kind: "conv" });
      for (let l = 0; l < c.depths[s]!; l++) out.push({ path: `convnext.encoder.stages.${s}.layers.${l}.dwconv`, to: `s${s}.${l}.dw`, kind: "dw" });
      void C;
    });
  }
  return out;
}

/** Kiln's name for a ConvNeXt matrix, bias or norm loaded as a weight (the convolutions load otherwise). */
export function convNextName(name: string): string | undefined {
  let m = /^convnext\.encoder\.stages\.(\d+)\.layers\.(\d+)\.(layernorm|pwconv1|pwconv2)\.(weight|bias)$/.exec(name);
  if (m) return `s${m[1]}.${m[2]}.${{ layernorm: "ln", pwconv1: "up", pwconv2: "down" }[m[3]!]}.${m[4]}`;
  if ((m = /^convnext\.encoder\.stages\.(\d+)\.downsampling_layer\.0\.(weight|bias)$/.exec(name))) return `s${m[1]}.downLn.${m[2]}`;
  if ((m = /^convnext\.embeddings\.layernorm\.(weight|bias)$/.exec(name))) return `stemLn.${m[1]}`;
  return undefined;
}

/**
 * A convnet's feature extractor on the GPU (the classifier's pooling and head run on the CPU over
 * its output). ResNet: stem 7×7 / 2 + max pool, bottleneck or basic blocks (batch norm folded,
 * shortcut projections where the shape changes, the residual and ReLU in the last conv's epilogue).
 * ConvNeXt: patchify convolution + LayerNorm, per stage a LayerNorm + 2×2 / 2 downsampling, blocks of
 * depthwise 7×7, LayerNorm, MLP (GELU), layer scale (folded into the second matrix) and residual.
 * One plan per input size.
 */
export class ConvNetModel {
  private readonly plans = new Map<string, { dispatches: readonly Dispatch[]; input: Buf; out: ConvImage; ctx: Ctx; step: Buf }>();
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: ConvNetConfig,
    private readonly weights: Map<string, Tensor>,
  ) {
    this.pool = new ActivationPool(backend, cfg.kind);
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`${this.cfg.kind} weight "${name}" was not loaded`);
    return t;
  }

  private build(H: number, W: number) {
    const be = this.backend;
    const step = be.alloc(STEP_BYTES, "storage", `${this.cfg.kind}/step`);
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
    const ctx = new Ctx(be, step, `${this.cfg.kind}/${H}x${W}`, this.pool);
    const input = ctx.buffer(H * W * CONVNET_IN_CHANNELS * 4, `${this.cfg.kind}/pixels`);
    const conv = (x: ConvImage, p: string, k: number, stride: number, o: { relu?: boolean; residual?: Tensor; pad?: number } = {}) =>
      ops.conv2d(ctx, x.y, this.w(`${p}.weight`), this.w(`${p}.bias`), { H: x.H, W: x.W, k, stride, pad: o.pad ?? (k - 1) / 2, relu: o.relu, residual: o.residual });
    const ln = (x: Tensor, p: string) => ops.layernorm(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), 1e-6);
    let x: ConvImage = { y: { shape: [H * W, CONVNET_IN_CHANNELS], dtype: "f32", buf: input }, H, W };
    const c = this.cfg;
    if (c.kind === "resnet") {
      x = conv(x, "stem", 7, 2, { relu: true });
      x = ops.maxPool2d(ctx, x.y, { H: x.H, W: x.W, k: 3, stride: 2, pad: 1 });
      c.hidden.forEach((_, s) => {
        for (let l = 0; l < c.depths[s]!; l++) {
          const p = `s${s}.${l}`;
          const stride = l === 0 && (s > 0 || c.downsampleInFirstStage) ? 2 : 1;
          const shortcut = this.weights.has(`${p}.short.weight`) ? conv(x, `${p}.short`, 1, stride).y : x.y;
          if (c.bottleneck) {
            const [s1, s2] = c.downsampleInBottleneck ? [stride, 1] : [1, stride];
            const h1 = conv(x, `${p}.c0`, 1, s1, { relu: true });
            const h2 = conv(h1, `${p}.c1`, 3, s2, { relu: true });
            x = conv(h2, `${p}.c2`, 1, 1, { relu: true, residual: shortcut });
          } else {
            const h1 = conv(x, `${p}.c0`, 3, stride, { relu: true });
            x = conv(h1, `${p}.c1`, 3, 1, { relu: true, residual: shortcut });
          }
        }
      });
    } else {
      x = conv(x, "stem", c.patch, c.patch, { pad: 0 });
      x = { ...x, y: ln(x.y, "stemLn") };
      c.hidden.forEach((C, s) => {
        if (s > 0) x = conv({ ...x, y: ln(x.y, `s${s}.downLn`) }, `s${s}.down`, 2, 2, { pad: 0 });
        for (let l = 0; l < c.depths[s]!; l++) {
          const p = `s${s}.${l}`;
          const d = ops.dwConv(ctx, x.y, this.w(`${p}.dw.weight`), this.w(`${p}.dw.bias`), { H: x.H, W: x.W, k: 7 });
          const up = ops.linear(ctx, ln(d, `${p}.ln`), this.w(`${p}.up.weight`), { bias: this.w(`${p}.up.bias`) });
          ops.act(ctx, up, "gelu");
          ops.linear(ctx, up, this.w(`${p}.down.weight`), { bias: this.w(`${p}.down.bias`), residual: x.y });
        }
        void C;
      });
    }
    return { dispatches: ctx.dispatches, input, out: x, ctx, step };
  }

  /** The last stage's features [rows, channels] for pixel values [3, height, width]. */
  async features(pixels: Float32Array, width: number, height: number): Promise<{ data: Float32Array; rows: number; channels: number }> {
    const n = width * height;
    if (pixels.length !== 3 * n) throw new Error(`${this.cfg.kind}: expected 3 × ${height} × ${width} pixel values, got ${pixels.length}`);
    const key = `${height}x${width}`;
    let plan = this.plans.get(key);
    if (!plan) {
      for (const p of this.plans.values()) this.free(p);
      this.plans.clear();
      plan = this.build(height, width);
      this.plans.set(key, plan);
    }
    const nhwc = new Float32Array(n * CONVNET_IN_CHANNELS);
    for (let ch = 0; ch < 3; ch++) for (let i = 0; i < n; i++) nhwc[i * CONVNET_IN_CHANNELS + ch] = pixels[ch * n + i]!;
    this.backend.write(plan.input, 0, nhwc);
    this.backend.submit(plan.dispatches);
    const channels = lastDim(plan.out.y);
    const rows = plan.out.H * plan.out.W;
    return { data: new Float32Array(await this.backend.readback(physical(plan.out.y.buf), 0, rows * channels * 4)), rows, channels };
  }

  private free(p: { ctx: Ctx; step: Buf }): void {
    p.ctx.destroy();
    this.backend.free(p.step);
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
