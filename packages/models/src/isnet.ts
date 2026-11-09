import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";

/**
 * The operations IS-Net (BRIA's RMBG-1.4, briarmbg.py) is written in, so one description of the
 * network can both list its convolutions (for loading) and build the GPU plan.
 */
export interface IsNetOps<T> {
  /**
   * 3×3 convolution of the channel concatenation of `inputs` (BN folded in and ReLU after when `bn`),
   * module path `path` ("…/conv_s1" + "…/bn_s1" with `bn`, else the conv itself).
   */
  conv(path: string, inputs: T[], out: number, o: { dil?: number; stride?: number; bn: boolean }): T;
  /** 2×2 max pooling, stride 2, ceil_mode. */
  pool(x: T): T;
  /** Bilinear resize (align_corners=False) to the size of `like`. */
  up(x: T, like: T): T;
  add(a: T, b: T): T;
}

/** RSU-L (L = 4…7, pooling between levels) or RSU-4F (dilations 1, 2, 4, 8 at one size) of briarmbg.py. */
function rsu<T>(f: IsNetOps<T>, path: string, inputs: T[], L: number, mid: number, out: number, flat: boolean): T {
  const c = (name: string, xs: T[], n: number, dil = 1) => f.conv(`${path}.${name}`, xs, n, { dil, bn: true });
  const hxin = c("rebnconvin", inputs, out);
  const enc: T[] = [c("rebnconv1", [hxin], mid)];
  for (let i = 2; i < L; i++) {
    const prev = enc[enc.length - 1]!;
    enc.push(c(`rebnconv${i}`, [flat ? prev : f.pool(prev)], mid, flat ? 2 ** (i - 1) : 1));
  }
  const deepest = c(`rebnconv${L}`, [enc[enc.length - 1]!], mid, flat ? 2 ** (L - 1) : 2);
  let d = deepest;
  for (let i = L - 1; i >= 1; i--) {
    const skip = enc[i - 1]!;
    const x = flat || i === L - 1 ? d : f.up(d, skip);
    d = c(`rebnconv${i}d`, [x, skip], i === 1 ? out : mid, flat ? 2 ** (i - 1) : 1);
  }
  return f.add(d, hxin);
}

/** BriaRMBG's forward to the first side output (the matte logits, before the upsampling and sigmoid). */
export function isnet<T>(f: IsNetOps<T>, x: T): T {
  const hxin = f.conv("conv_in", [x], 64, { stride: 2, bn: false });
  const enc: [string, number, number, number, boolean][] = [
    ["stage1", 7, 32, 64, false],
    ["stage2", 6, 32, 128, false],
    ["stage3", 5, 64, 256, false],
    ["stage4", 4, 128, 512, false],
    ["stage5", 4, 256, 512, true],
    ["stage6", 4, 256, 512, true],
  ];
  const hx: T[] = [];
  let h = hxin;
  enc.forEach(([p, L, mid, out, flat], i) => {
    h = rsu(f, p, [i ? f.pool(h) : h], L, mid, out, flat);
    hx.push(h);
  });
  const dec: [string, number, number, number, boolean][] = [
    ["stage5d", 4, 256, 512, true],
    ["stage4d", 4, 128, 256, false],
    ["stage3d", 5, 64, 128, false],
    ["stage2d", 6, 32, 64, false],
    ["stage1d", 7, 16, 64, false],
  ];
  let d: T = hx[5]!;
  dec.forEach(([p, L, mid, out, flat], i) => {
    const skip = hx[4 - i]!;
    d = rsu(f, p, [f.up(d, skip), skip], L, mid, out, flat);
  });
  return f.conv("side1", [d], 1, { bn: false });
}

/** Each convolution's module path, input channel split (one entry per concatenated input) and kind. */
export function isnetConvs(inChannels = 3): Map<string, { splits: number[]; bn: boolean }> {
  type S = { C: number };
  const convs = new Map<string, { splits: number[]; bn: boolean }>();
  isnet<S>(
    {
      conv: (path, inputs, out, o) => {
        convs.set(path, { splits: inputs.map((t) => t.C), bn: o.bn });
        return { C: out };
      },
      pool: (x) => x,
      up: (x) => x,
      add: (a) => a,
    },
    { C: inChannels },
  );
  return convs;
}

/** Input channels of conv_in: RGB padded to 4 for the conv kernel. */
export const ISNET_IN_CHANNELS = 4;

interface Img {
  y: Tensor;
  H: number;
  W: number;
}

/**
 * BRIA RMBG-1.4 (IS-Net) on the GPU: every convolution through the conv kernel with its batch norm
 * folded in, a concatenation becoming one convolution per input summed in the epilogue (weights
 * "<path>.p<i>"), and the first side output upsampled to the input size. One plan per input size.
 */
export class IsNetModel {
  private readonly plans = new Map<string, { dispatches: readonly Dispatch[]; input: Buf; out: Tensor; ctx: Ctx; step: Buf }>();
  private readonly pool: ActivationPool;

  constructor(
    readonly backend: Backend,
    private readonly weights: Map<string, Tensor>,
  ) {
    this.pool = new ActivationPool(backend, "isnet");
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`RMBG weight "${name}" was not loaded`);
    return t;
  }

  private build(H: number, W: number) {
    const be = this.backend;
    const step = be.alloc(STEP_BYTES, "storage", "isnet/step");
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
    const ctx = new Ctx(be, step, `isnet/${H}x${W}`, this.pool);
    const input = ctx.buffer(H * W * ISNET_IN_CHANNELS * 4, "isnet/pixels");
    const x: Img = { y: { shape: [H * W, ISNET_IN_CHANNELS], dtype: "f32", buf: input }, H, W };
    const side = isnet<Img>(
      {
        conv: (path, inputs, _out, o) => {
          const dil = o.dil ?? 1;
          let r: Img | undefined;
          inputs.forEach((t, i) => {
            const last = i === inputs.length - 1;
            r = ops.conv2d(ctx, t.y, this.w(`${path}.p${i}.weight`), this.w(`${path}.p${i}.bias`), { H: t.H, W: t.W, k: 3, stride: o.stride ?? 1, pad: dil, dilation: dil, relu: last && o.bn, residual: r?.y });
          });
          return r!;
        },
        pool: (t) => ops.maxPool2d(ctx, t.y, { H: t.H, W: t.W, k: 2, stride: 2, pad: 0, ceil: true }),
        up: (t, like) => (t.H === like.H && t.W === like.W ? t : { y: ops.resizeBilinear(ctx, t.y, { H: t.H, W: t.W, Ho: like.H, Wo: like.W, alignCorners: false }), H: like.H, W: like.W }),
        add: (a, b) => {
          ops.addInPlace(ctx, a.y, b.y);
          return a;
        },
      },
      x,
    );
    const out = ops.resizeBilinear(ctx, side.y, { H: side.H, W: side.W, Ho: H, Wo: W, alignCorners: false });
    return { dispatches: ctx.dispatches, input, out, ctx, step };
  }

  /** Matte logits [height, width] (sigmoid gives the foreground probability) for pixel values [3, height, width]. */
  async predict(pixels: Float32Array, width: number, height: number): Promise<Float32Array> {
    const n = width * height;
    if (pixels.length !== 3 * n) throw new Error(`RMBG: expected 3 × ${height} × ${width} pixel values, got ${pixels.length}`);
    const key = `${height}x${width}`;
    let plan = this.plans.get(key);
    if (!plan) {
      for (const p of this.plans.values()) this.free(p);
      this.plans.clear();
      plan = this.build(height, width);
      this.plans.set(key, plan);
    }
    const nhwc = new Float32Array(n * ISNET_IN_CHANNELS);
    for (let ch = 0; ch < 3; ch++) for (let i = 0; i < n; i++) nhwc[i * ISNET_IN_CHANNELS + ch] = pixels[ch * n + i]!;
    this.backend.write(plan.input, 0, nhwc);
    this.backend.submit(plan.dispatches);
    return new Float32Array(await this.backend.readback(physical(plan.out.buf), 0, n * 4));
  }

  private free(p: { ctx: Ctx; step: Buf }): void {
    p.ctx.destroy();
    this.backend.free(p.step);
  }

  destroy(): void {
    for (const t of this.weights.values()) this.backend.free(t.buf);
    for (const p of this.plans.values()) this.free(p);
    this.pool.destroy();
  }
}
