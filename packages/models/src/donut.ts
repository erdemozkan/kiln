import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";

/** Donut's Swin encoder (DonutSwinConfig): Swin v1 without a final LayerNorm. */
export interface DonutSwinConfig {
  /** The image size the processor pads to, [height, width]. */
  imageHW: [number, number];
  patchSize: number;
  embedDim: number;
  depths: number[];
  heads: number[];
  window: number;
  eps: number;
}

export function donutSwinConfigFromHF(enc: Record<string, unknown>): DonutSwinConfig {
  if (enc.model_type !== "donut-swin") throw new UnsupportedModelError(`VisionEncoderDecoder encoder "${String(enc.model_type)}" is not supported (vit, donut-swin)`);
  if (enc.use_absolute_embeddings === true) throw new UnsupportedModelError("Donut with absolute position embeddings is not supported");
  if (enc.hidden_act !== undefined && enc.hidden_act !== "gelu") throw new UnsupportedModelError(`Donut activation "${String(enc.hidden_act)}" is not supported`);
  if (Number(enc.num_channels ?? 3) !== 3) throw new UnsupportedModelError("only 3-channel images are supported");
  const size = enc.image_size ?? 224;
  const imageHW: [number, number] = Array.isArray(size) ? [Number(size[0]), Number(size[1])] : [Number(size), Number(size)];
  const c: DonutSwinConfig = {
    imageHW,
    patchSize: Number(enc.patch_size ?? 4),
    embedDim: Number(enc.embed_dim ?? 96),
    depths: ((enc.depths as number[] | undefined) ?? [2, 2, 6, 2]).map(Number),
    heads: ((enc.num_heads as number[] | undefined) ?? [3, 6, 12, 24]).map(Number),
    window: Number(enc.window_size ?? 7),
    eps: Number(enc.layer_norm_eps ?? 1e-5),
  };
  // Each stage's grid must split into whole windows larger than one (transformers pads otherwise).
  let [h, w] = donutGrid(c);
  c.depths.forEach((_, s) => {
    if (h % c.window || w % c.window || Math.min(h, w) <= c.window) throw new UnsupportedModelError(`Donut: a ${h} × ${w} grid at stage ${s + 1} does not split into ${c.window}-token windows`);
    if (s < c.depths.length - 1) {
      if (h % 2 || w % 2) throw new UnsupportedModelError(`Donut: a ${h} × ${w} grid cannot be merged`);
      [h, w] = [h / 2, w / 2];
    }
  });
  return c;
}

/** The patch grid [rows, columns] at the first stage. */
export const donutGrid = (c: DonutSwinConfig): [number, number] => [Math.floor(c.imageHW[0] / c.patchSize), Math.floor(c.imageHW[1] / c.patchSize)];

/** Width of the encoder's output (the last stage's). */
export const donutHidden = (c: DonutSwinConfig): number => c.embedDim * 2 ** (c.depths.length - 1);

/** Patch rows padded to a multiple of 32 for the matmul kernels. */
export const donutPatchK = (c: DonutSwinConfig): number => Math.ceil((3 * c.patchSize * c.patchSize) / 32) * 32;

/**
 * Kiln's name for a Donut encoder tensor (without the VisionEncoderDecoder's "encoder." prefix):
 * blocks "s{stage}.b{block}.{q,k,v,o,up,down,ln1,ln2}", patch merging "m{stage}.{ln,red}", the
 * embedding LayerNorm "embNorm". The patch convolution and position-bias tables load otherwise.
 */
export function donutSwinName(name: string): string | undefined {
  let m = /^encoder\.layers\.(\d+)\.blocks\.(\d+)\.(.+)\.(weight|bias)$/.exec(name);
  if (m) {
    const part: Record<string, string> = {
      "attention.self.query": "q",
      "attention.self.key": "k",
      "attention.self.value": "v",
      "attention.output.dense": "o",
      "intermediate.dense": "up",
      "output.dense": "down",
      layernorm_before: "ln1",
      layernorm_after: "ln2",
    };
    const p = part[m[3]!];
    return p ? `s${m[1]}.b${m[2]}.${p}.${m[4]}` : undefined;
  }
  m = /^encoder\.layers\.(\d+)\.downsample\.(norm|reduction)\.(weight|bias)$/.exec(name);
  if (m) return `m${m[1]}.${m[2] === "norm" ? "ln" : "red"}.${m[3]}`;
  m = /^embeddings\.norm\.(weight|bias)$/.exec(name);
  return m ? `embNorm.${m[1]}` : undefined;
}

/**
 * A Swin relative-position table [(2w − 1)², heads] expanded to [heads, w², w²]: entry (i, j) of a
 * window is the row for their offset (Δy + w − 1)·(2w − 1) + (Δx + w − 1), as transformers indexes it.
 */
export function swinRelativeBias(table: Float32Array, heads: number, win: number): Float32Array {
  const N = win * win;
  const out = new Float32Array(heads * N * N);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      const r = (Math.floor(i / win) - Math.floor(j / win) + win - 1) * (2 * win - 1) + ((i % win) - (j % win) + win - 1);
      for (let h = 0; h < heads; h++) out[(h * N + i) * N + j] = table[r * heads + h]!;
    }
  return out;
}

/**
 * Patch rows [rows · cols, patchK] from pixel values [3, height, width]: each patch in (channel, row,
 * column) order as the convolution weight [embed, 3, p, p] flattens, zero-padded to patchK.
 */
export function donutPatchRows(c: DonutSwinConfig, pixels: Float32Array): Float32Array {
  const P = c.patchSize;
  const [H, W] = c.imageHW;
  const [gh, gw] = donutGrid(c);
  const K = donutPatchK(c);
  const out = new Float32Array(gh * gw * K);
  const n = H * W;
  for (let py = 0; py < gh; py++)
    for (let px = 0; px < gw; px++) {
      const o = (py * gw + px) * K;
      for (let ch = 0; ch < 3; ch++)
        for (let y = 0; y < P; y++) {
          const src = ch * n + (py * P + y) * W + px * P;
          out.set(pixels.subarray(src, src + P), o + (ch * P + y) * P);
        }
    }
  return out;
}

/**
 * Donut's Swin encoder as one GPU plan at the processor's image size: the patch convolution as a
 * matmul ("patch.weight" [embed, patchK], "patch.bias"), LayerNorm; per stage, pre-norm blocks of
 * shifted-window attention (every second block shifted by half a window; "s.b.bias" the expanded
 * position bias) and an exact-GELU MLP; between stages, patch merging (2 × 2 gather, LayerNorm,
 * bias-free reduction). The output is the last stage's tokens [rows · cols, hidden], row-major.
 */
export class DonutSwinModel {
  private plan?: { dispatches: readonly Dispatch[]; input: Buf; out: Tensor; ctx: Ctx; step: Buf };
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    readonly cfg: DonutSwinConfig,
    private readonly weights: Map<string, Tensor>,
  ) {
    this.pool = new ActivationPool(backend, "donut");
  }

  /** Output tokens and the grid they form, [tokens, rows, columns]. */
  get outGrid(): [number, number, number] {
    const [gh, gw] = donutGrid(this.cfg);
    const f = 2 ** (this.cfg.depths.length - 1);
    return [(gh / f) * (gw / f), gh / f, gw / f];
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`donut weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string, residual?: Tensor): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
  }

  private ln(ctx: Ctx, x: Tensor, p: string): Tensor {
    return ops.layernorm(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), this.cfg.eps);
  }

  private build() {
    const c = this.cfg;
    const be = this.backend;
    let [H, W] = donutGrid(c);
    const step = be.alloc(STEP_BYTES, "storage", "donut/step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
    const ctx = new Ctx(be, step, "donut", this.pool);
    const K = donutPatchK(c);
    const input = ctx.buffer(H * W * K * 4, "donut/patches");
    let x = this.ln(ctx, this.lin(ctx, { shape: [H * W, K], dtype: "f32", buf: input }, "patch"), "embNorm");
    c.depths.forEach((depth, s) => {
      for (let b = 0; b < depth; b++) {
        const p = `s${s}.b${b}`;
        const h = this.ln(ctx, x, `${p}.ln1`);
        const shift = b % 2 ? Math.floor(c.window / 2) : 0;
        const a = ops.swinV1Attention(ctx, this.lin(ctx, h, `${p}.q`), this.lin(ctx, h, `${p}.k`), this.lin(ctx, h, `${p}.v`), this.w(`${p}.bias`), { H, W, heads: c.heads[s]!, win: c.window, shift });
        this.lin(ctx, a, `${p}.o`, x);
        const up = this.lin(ctx, this.ln(ctx, x, `${p}.ln2`), `${p}.up`);
        ops.act(ctx, up, "gelu");
        this.lin(ctx, up, `${p}.down`, x);
      }
      if (s < c.depths.length - 1) {
        x = this.lin(ctx, this.ln(ctx, ops.patchMerge(ctx, x, { H, W }), `m${s}.ln`), `m${s}.red`);
        [H, W] = [H / 2, W / 2];
      }
    });
    return { dispatches: ctx.dispatches, input, out: x, ctx, step };
  }

  /** The last stage's tokens [tokens, hidden] for pixel values [3, height, width] at the processor's size. */
  async forward(pixels: Float32Array): Promise<Float32Array> {
    const [H, W] = this.cfg.imageHW;
    if (pixels.length !== 3 * H * W) throw new Error(`donut: expected 3 × ${H} × ${W} pixel values, got ${pixels.length}`);
    this.plan ??= this.build();
    const be = this.backend;
    be.write(this.plan.input, 0, donutPatchRows(this.cfg, pixels));
    // Seconds of GPU work in one command buffer can trip the GPU watchdog: about a block per submit,
    // each waited on once the next is queued.
    const d = this.plan.dispatches;
    const per = Math.max(1, Math.ceil(d.length / (this.cfg.depths.reduce((a, b) => a + b, 0) + this.cfg.depths.length)));
    let pending: Promise<ArrayBuffer> | undefined;
    for (let i = 0; i < d.length; i += per) {
      be.submit(d.slice(i, i + per));
      const done = be.readback(this.plan.step, 0, 4);
      if (pending) await pending;
      pending = done;
    }
    await pending;
    const n = this.outGrid[0] * donutHidden(this.cfg);
    return new Float32Array(await this.backend.readback(physical(this.plan.out.buf), 0, n * 4));
  }

  destroy(): void {
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    if (this.plan) {
      this.plan.ctx.destroy();
      this.backend.free(this.plan.step);
    }
    this.pool.destroy();
  }
}
