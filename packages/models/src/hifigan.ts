import type { Ctx, Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";

/** A HiFi-GAN generator's shape (VITS's decoder, SpeechT5HifiGan). */
export interface HifiGanConfig {
  upsampleRates: number[];
  upsampleKernels: number[];
  resblockKernels: number[];
  resblockDilations: number[][];
  leakySlope: number;
}

/** A stride-1 1-D convolution of [T, C] with "same" padding for its dilation, on the conv kernel. */
export function conv1d(ctx: Ctx, w: (name: string) => Tensor, x: Tensor, p: string, k: number, o: { dil?: number; relu?: boolean; leakyIn?: number; tanh?: boolean; residual?: Tensor } = {}): Tensor {
  const T = x.shape[0]!;
  const dil = o.dil ?? 1;
  return ops.conv2d(ctx, x, w(`${p}.weight`), w(`${p}.bias`), { H: 1, W: T, k, kh: 1, stride: 1, pad: (k * dil - dil) / 2, padH: 0, dilation: dil, relu: o.relu, leakyIn: o.leakyIn, tanh: o.tanh, residual: o.residual }).y;
}

/**
 * HiFi-GAN from frames [T, C] to samples [T · Π rates, 1] (before the tanh, applied by the caller):
 * conv_pre (7), then per upsampler LeakyReLU + transposed convolution and the mean of the
 * multi-receptive-field residual blocks (each: per dilation LeakyReLU + conv, LeakyReLU + conv,
 * plus the input), then LeakyReLU (0.01) and conv_post (7). Weights under `prefix`: pre, up{i},
 * rb{n}.c1_{j} / c2_{j}, post.
 */
export function hifiGan(ctx: Ctx, w: (name: string) => Tensor, c: HifiGanConfig, x: Tensor, prefix: string): Tensor {
  let h = conv1d(ctx, w, x, `${prefix}.pre`, 7);
  const nk = c.resblockKernels.length;
  c.upsampleRates.forEach((s, i) => {
    const k = c.upsampleKernels[i]!;
    h = ops.convTranspose1d(ctx, h, w(`${prefix}.up${i}.weight`), w(`${prefix}.up${i}.bias`), { s, pad: (k - s) / 2, leakyIn: c.leakySlope });
    const sum = ctx.alloc(h.shape);
    c.resblockKernels.forEach((kr, r) => {
      const p = `${prefix}.rb${i * nk + r}`;
      let y = h;
      c.resblockDilations[r]!.forEach((d, j) => {
        const t = conv1d(ctx, w, y, `${p}.c1_${j}`, kr, { dil: d, leakyIn: c.leakySlope });
        y = conv1d(ctx, w, t, `${p}.c2_${j}`, kr, { leakyIn: c.leakySlope, residual: y });
      });
      if (r === 0) ops.scaleAdd(ctx, sum, y, 1, false);
      else ops.addInPlace(ctx, sum, y);
    });
    const avg = ctx.alloc(h.shape);
    ops.scaleAdd(ctx, avg, sum, 1 / nk, false);
    h = avg;
  });
  return conv1d(ctx, w, h, `${prefix}.post`, 7, { leakyIn: 0.01 });
}
