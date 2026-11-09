import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** Vision transformers from transformers' configs: ViT, DeiT (with its distillation token). */
export interface VitConfig {
  modelType: string;
  architectures: string[];
  hidden: number;
  layers: number;
  heads: number;
  intermediate: number;
  imageSize: number;
  patchSize: number;
  channels: number;
  eps: number;
  act: "gelu" | "gelu_tanh" | "quick_gelu" | "relu";
  /** Tokens before the patches: CLS, and DeiT's distillation token. */
  prefix: number;
  labels: string[];
}

export const VIT_TYPES = new Set(["vit", "deit"]);

export function vitConfigFromHF(c: Record<string, unknown>): VitConfig {
  const type = String(c.model_type ?? "");
  if (!VIT_TYPES.has(type)) throw new UnsupportedModelError(`vision model type "${type}" is not supported (ViT, DeiT)`);
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
  return {
    modelType: type,
    architectures: (c.architectures as string[] | undefined) ?? [],
    hidden: num("hidden_size"),
    layers: num("num_hidden_layers"),
    heads: num("num_attention_heads"),
    intermediate: num("intermediate_size"),
    imageSize: num("image_size", 224),
    patchSize: num("patch_size", 16),
    channels: 3,
    eps: num("layer_norm_eps", 1e-12),
    act,
    prefix: type === "deit" ? 2 : 1,
    labels,
  };
}

/** Kiln's name for a ViT / DeiT tensor (matrices, biases, norms), or undefined for those loaded otherwise. */
export function vitName(name: string): string | undefined {
  const n = name.replace(/^(vit|deit)\./, "");
  let m = /^encoder\.layer\.(\d+)\.(.+)\.(weight|bias)$/.exec(n);
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
    };
    const p = part[m[2]!];
    return p ? `l${m[1]}.${p}.${m[3]}` : undefined;
  }
  m = /^layernorm\.(weight|bias)$/.exec(n);
  if (m) return `lnF.${m[1]}`;
  return undefined;
}

/** Patch features padded to a multiple of 32 for the matmul kernels. */
export const vitPatchK = (c: Pick<VitConfig, "patchSize" | "channels">): number => Math.ceil((c.channels * c.patchSize * c.patchSize) / 32) * 32;

/**
 * Rows [prefix + patches, patchK] from pixel values [3, size, size]: `prefix` zero rows (the CLS and
 * distillation tokens come from the constant addend), then each patch in (channel, row, column)
 * order as the convolution weight [hidden, 3, p, p] flattens.
 */
export function vitPatchRows(c: VitConfig, pixels: Float32Array, size = c.imageSize): Float32Array {
  const P = c.patchSize;
  const side = size / P;
  const K = vitPatchK(c);
  const n = size * size;
  const out = new Float32Array((c.prefix + side * side) * K);
  for (let py = 0; py < side; py++)
    for (let px = 0; px < side; px++) {
      const o = (c.prefix + py * side + px) * K;
      for (let ch = 0; ch < 3; ch++)
        for (let y = 0; y < P; y++)
          for (let x = 0; x < P; x++) out[o + (ch * P + y) * P + x] = pixels[ch * n + (py * P + y) * size + px * P + x]!;
    }
  return out;
}

/**
 * A pre-norm vision transformer encoder: patch rows → one matmul (the patch convolution, weight
 * "patch.weight" [hidden, patchK]) plus a constant [tokens, hidden] ("embed.add": CLS / distillation
 * tokens, position embeddings and the patch bias) → layers of LayerNorm, full attention, LayerNorm,
 * MLP → final LayerNorm. forward() returns the last hidden state [tokens, hidden].
 */
export class VitModel {
  private plan: { dispatches: readonly Dispatch[]; input: Buf; out: Tensor; ctx: Ctx } | undefined;
  private readonly step: Buf;
  private readonly kc: Buf;
  private readonly vc: Buf;
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: VitConfig,
    private readonly weights: Map<string, Tensor>,
  ) {
    const T = this.tokens;
    this.step = backend.alloc(STEP_BYTES, "storage", "vit/step");
    backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: T }));
    this.kc = backend.alloc(T * cfg.hidden * 4, "storage", "vit/k");
    this.vc = backend.alloc(T * cfg.hidden * 4, "storage", "vit/v");
    this.pool = new ActivationPool(backend, "vit");
  }

  get tokens(): number {
    return this.cfg.prefix + (this.cfg.imageSize / this.cfg.patchSize) ** 2;
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`vit weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string, residual?: Tensor): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
  }

  private build() {
    const c = this.cfg;
    const T = this.tokens;
    const D = c.hidden / c.heads;
    const ctx = new Ctx(this.backend, this.step, "vit", this.pool);
    const input = ctx.buffer(T * vitPatchK(c) * 4, "vit/patches");
    let x = ops.linear(ctx, { shape: [T, vitPatchK(c)], dtype: "f32", buf: input }, this.w("patch.weight"), {});
    ops.addInPlace(ctx, x, this.w("embed.add"));
    const shape = { T, H: c.heads, Hkv: c.heads, D, maxSeq: T, kv: "f32" as const, scale: D ** -0.5, span: true };
    for (let l = 0; l < c.layers; l++) {
      const p = `l${l}`;
      const h = ops.layernorm(ctx, x, this.w(`${p}.ln1.weight`), this.w(`${p}.ln1.bias`), c.eps);
      const q = this.lin(ctx, h, `${p}.q`);
      ops.kvWrite(ctx, this.lin(ctx, h, `${p}.k`), this.lin(ctx, h, `${p}.v`), this.kc, this.vc, T, T, "f32");
      this.lin(ctx, ops.attention(ctx, q, this.kc, this.vc, shape), `${p}.o`, x);
      const h2 = ops.layernorm(ctx, x, this.w(`${p}.ln2.weight`), this.w(`${p}.ln2.bias`), c.eps);
      const up = this.lin(ctx, h2, `${p}.up`);
      ops.act(ctx, up, c.act);
      this.lin(ctx, up, `${p}.down`, x);
    }
    const out = ops.layernorm(ctx, x, this.w("lnF.weight"), this.w("lnF.bias"), c.eps);
    return { dispatches: ctx.dispatches, input, out, ctx };
  }

  /** Last hidden state [tokens, hidden] for pixel values [3, imageSize, imageSize]. */
  async forward(pixels: Float32Array): Promise<Float32Array> {
    const c = this.cfg;
    if (pixels.length !== 3 * c.imageSize * c.imageSize) throw new Error(`vit: expected 3 × ${c.imageSize} × ${c.imageSize} pixel values, got ${pixels.length}`);
    this.plan ??= this.build();
    this.backend.write(this.plan.input, 0, vitPatchRows(c, pixels));
    this.backend.submit(this.plan.dispatches);
    return new Float32Array(await this.backend.readback(physical(this.plan.out.buf), 0, this.tokens * c.hidden * 4));
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
