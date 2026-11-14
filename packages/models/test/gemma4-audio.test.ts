import { describe, expect, it } from "vitest";
import { type Tensor } from "@kiln/core";
import { CpuBackend } from "@kiln/kernels";
import { gemma4AudioFeatures } from "@kiln/loader";
import {
  GEMMA4_AUDIO_CLIPPED,
  Gemma4AudioTower,
  gemma4AudioConfig,
  gemma4AudioConfigFromGGUF,
  gemma4AudioName,
  gemma4AudioPositions,
  gemma4AudioSubsample,
  gemma4AudioTokens,
  type Gemma4AudioConfig,
  type Gemma4AudioConvs,
} from "../src/gemma4-audio";
import type { Gemma4Clip } from "../src/gemma4-vision";

const config = {
  text_config: { hidden_size: 1536 },
  audio_config: { attention_chunk_size: 12, attention_context_left: 13, attention_context_right: 0, attention_logit_cap: 50, conv_kernel_size: 5, hidden_size: 1024, num_attention_heads: 8, num_hidden_layers: 12, output_proj_dims: 1536, residual_weight: 0.5, rms_norm_eps: 1e-6, subsampling_conv_channels: [128, 32] },
};

function rand(n: number, seed: number, scale: number): Float32Array {
  let s = seed >>> 0;
  return Float32Array.from({ length: n }, () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s / 2 ** 32 - 0.5) * 2 * scale;
  });
}

type Weights = Map<string, { shape: number[]; data: Float32Array }>;

/**
 * The tower as the transformers model computes it, in float64, from mel features: the convolutions
 * with their padding, and the attention in blocks of `chunk` queries over a context of the block
 * and the rows before it, the relative term through the model's pad-and-reshape shift, with the
 * mask that lets a query see itself and the `contextLeft − 2` rows before it.
 */
function reference(c: Gemma4AudioConfig, w: Weights, clips: Map<string, Gemma4Clip>, convs: Gemma4AudioConvs, features: Float32Array, frames: number): Float64Array {
  const H = c.hidden;
  const D = H / c.heads;
  const W = (name: string) => w.get(name)!;
  const rms = (x: Float64Array, weight?: Float32Array): Float64Array => {
    let ss = 0;
    for (const v of x) ss += v * v;
    const r = (ss / x.length + c.eps) ** -0.5;
    return x.map((v, i) => v * r * (weight ? weight[i]! : 1));
  };
  const matvec = (name: string, x: ArrayLike<number>, lim?: Gemma4Clip): Float64Array => {
    const { shape, data } = W(name);
    const [N, K] = shape as [number, number];
    const y = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < K; j++) s += data[i * K + j]! * (lim ? Math.min(Math.max(x[j]!, lim[0]), lim[1]) : x[j]!);
      y[i] = lim ? Math.min(Math.max(s, lim[2]), lim[3]) : s;
    }
    return y;
  };
  const clipped = (at: string, x: ArrayLike<number>) => matvec(`${at}.weight`, x, clips.get(at)!);
  const silu = (v: number) => v / (1 + Math.exp(-v));
  // Subsampling: Conv2d(kernel 3, stride 2, padding 1) on [channel, time, frequency], LayerNorm over channels, ReLU.
  const conv = (x: Float64Array[][], cin: number, wt: Float32Array, nw: Float32Array): Float64Array[][] => {
    const [T, F, cout] = [x.length, x[0]!.length, nw.length];
    const out: Float64Array[][] = [];
    for (let t = 0; t < Math.floor((T + 2 - 3) / 2) + 1; t++) {
      out.push([]);
      for (let f = 0; f < Math.floor((F + 2 - 3) / 2) + 1; f++) {
        const y = new Float64Array(cout);
        for (let o = 0; o < cout; o++)
          for (let i = 0; i < cin; i++)
            for (let a = 0; a < 3; a++)
              for (let b = 0; b < 3; b++) {
                const [ti, fi] = [2 * t + a - 1, 2 * f + b - 1];
                if (ti >= 0 && ti < T && fi >= 0 && fi < F) y[o] += wt[((o * cin + i) * 3 + a) * 3 + b]! * x[ti]![fi]![i]!;
              }
        const mean = y.reduce((p, v) => p + v, 0) / cout;
        const variance = y.reduce((p, v) => p + (v - mean) ** 2, 0) / cout;
        out[t]!.push(y.map((v, o) => Math.max(0, ((v - mean) / Math.sqrt(variance + c.eps)) * nw[o]!)));
      }
    }
    return out;
  };
  const mel = Array.from({ length: frames }, (_, t) => Array.from({ length: c.mels }, (_, f) => Float64Array.of(features[t * c.mels + f]!)));
  const sub = conv(conv(mel, 1, convs.w[0], convs.norm[0]), c.convChannels[0], convs.w[1], convs.norm[1]);
  let xs = sub.map((row) => matvec("a.input_projection.weight", row.flatMap((v) => [...v])));
  const T = xs.length;
  // Position embeddings for positions contextLeft − 1 … 0, as the model orders them.
  const R = c.contextLeft;
  const half = H / 2;
  const posRow = (pos: number) => Float64Array.from({ length: H }, (_, i) => (i < half ? Math.sin(pos * Math.exp((-i * Math.log(10000)) / (half - 1))) : Math.cos(pos * Math.exp((-(i - half) * Math.log(10000)) / (half - 1)))));
  const [chunk, past] = [12, R - 1];
  const context = chunk + past;
  for (let l = 0; l < c.layers; l++) {
    const p = `a.blk.${l}.`;
    const ff = (x: Float64Array, s: string) => {
      const up = clipped(`${p}ffn_up${s}`, rms(x, W(`${p}ffn_norm${s}.weight`).data)).map(silu);
      const down = rms(clipped(`${p}ffn_down${s}`, up), W(`${p}ffn_post_norm${s}.weight`).data);
      return x.map((v, i) => v + c.residualWeight * down[i]!);
    };
    xs = xs.map((x) => ff(x, ""));
    const hs = xs.map((x) => rms(x, W(p + "attn_pre_norm.weight").data));
    const scale = W(p + "per_dim_scale.weight").data;
    const qs = hs.map((h) => clipped(p + "attn_q", h).map((v, i) => v * (D ** -0.5 / Math.log(2)) * scale[i % D]!));
    const ks = hs.map((h) => clipped(p + "attn_k", h).map((v) => (v * Math.log(1 + Math.E)) / Math.log(2)));
    const vs = hs.map((h) => clipped(p + "attn_v", h));
    const rel = Array.from({ length: R }, (_, j) => matvec(p + "attn_k_rel.weight", posRow(R - 1 - j)));
    const blocks = Math.ceil(T / chunk);
    const attn = xs.map(() => new Float64Array(H));
    for (let b = 0; b < blocks; b++)
      for (let h = 0; h < c.heads; h++) {
        // Relative scores [chunk, R], padded to context + 1 columns, read back as [chunk, context].
        const flat = new Float64Array(chunk * (context + 1));
        for (let i = 0; i < chunk; i++) {
          const t = b * chunk + i;
          if (t >= T) continue;
          for (let j = 0; j < R; j++) for (let d = 0; d < D; d++) flat[i * (context + 1) + j] += qs[t]![h * D + d]! * rel[j]![h * D + d]!;
        }
        for (let i = 0; i < chunk; i++) {
          const t = b * chunk + i;
          if (t >= T) continue;
          const sc: number[] = [];
          const key: number[] = [];
          for (let col = 0; col < context; col++) {
            const k = b * chunk - past + col;
            if (k < 0 || k >= T || t - k < 0 || t - k >= past) continue;
            let s = flat[i * context + col]!;
            for (let d = 0; d < D; d++) s += qs[t]![h * D + d]! * ks[k]![h * D + d]!;
            sc.push(c.logitCap * Math.tanh(s / c.logitCap));
            key.push(k);
          }
          const m = Math.max(...sc);
          const e = sc.map((s) => Math.exp(s - m));
          const z = e.reduce((p0, v) => p0 + v, 0);
          key.forEach((k, j) => {
            for (let d = 0; d < D; d++) attn[t]![h * D + d] += (e[j]! / z) * vs[k]![h * D + d]!;
          });
        }
      }
    xs = xs.map((x, t) => {
      const o = rms(clipped(p + "attn_out", attn[t]!), W(p + "attn_post_norm.weight").data);
      return x.map((v, i) => v + o[i]!);
    });
    // Convolution module.
    const gated = xs.map((x) => {
      const s = clipped(p + "conv_pw1", rms(x, W(p + "conv_norm.weight").data));
      return Float64Array.from({ length: H }, (_, i) => s[i]! / (1 + Math.exp(-s[H + i]!)));
    });
    const dw = W(p + "conv_dw.weight").data;
    const K = c.convKernel;
    xs = xs.map((x, t) => {
      const y = new Float64Array(H);
      for (let ch = 0; ch < H; ch++) for (let j = 0; j < K; j++) if (t - (K - 1) + j >= 0) y[ch] += dw[ch * K + j]! * gated[t - (K - 1) + j]![ch]!;
      const e = clipped(p + "conv_pw2", rms(y, W(p + "norm_conv.weight").data).map(silu));
      return x.map((v, i) => v + e[i]!);
    });
    xs = xs.map((x) => rms(ff(x, "_1"), W(p + "ln2.weight").data));
  }
  const bias = W("a.pre_encode.out.bias").data;
  const out = new Float64Array(T * c.projectionDim);
  xs.forEach((x, t) => out.set(matvec("mm.a.input_projection.weight", rms(matvec("a.pre_encode.out.weight", x).map((v, i) => v + bias[i]!))), t * c.projectionDim));
  return out;
}

describe("Gemma 4 audio", () => {
  const c = gemma4AudioConfig(config, { audio_seq_length: 750, feature_extractor: { feature_size: 128 } });

  it("reads the config", () => {
    expect(c).toMatchObject({ hidden: 1024, heads: 8, layers: 12, mels: 128, convChannels: [128, 32], contextLeft: 13, logitCap: 50, convKernel: 5, residualWeight: 0.5, outputDim: 1536, projectionDim: 1536, maxTokens: 750 });
    expect(() => gemma4AudioConfig({})).toThrow(/audio_config/);
    expect(() => gemma4AudioConfig({ audio_config: { ...config.audio_config, attention_context_right: 2 } })).toThrow(/later rows/);
    const meta = new Map<string, unknown>([["clip.audio.embedding_length", 1024], ["clip.audio.attention.head_count", 8], ["clip.audio.block_count", 12], ["clip.audio.num_mel_bins", 128], ["clip.audio.projection_dim", 1536]]);
    expect(gemma4AudioConfigFromGGUF(meta, [128, 32], 5, 1536)).toEqual(c);
  });

  it("counts a token per 40 ms, as the processor does", () => {
    // 30 s: 2999 frames, 1500 rows after the first convolution, 750 tokens.
    expect(gemma4AudioTokens(2999, c)).toBe(750);
    expect(gemma4AudioTokens(5000, c)).toBe(750);
    expect(gemma4AudioTokens(31, c)).toBe(8);
    expect(gemma4AudioTokens(1, c)).toBe(1);
    expect(gemma4AudioTokens(0, c)).toBe(0);
  });

  it("makes the extractor's log-mel features", () => {
    const n = 5000;
    const x = Float32Array.from({ length: n }, (_, t) => 0.5 * Math.sin((2 * Math.PI * 440 * t) / 16000) + 0.25 * Math.sin((2 * Math.PI * 3000 * t) / 16000 + 1) + 0.1 * Math.sin((2 * Math.PI * 97 * t) / 16000));
    const { features, frames } = gemma4AudioFeatures(x);
    // Frames that lie wholly inside the audio: floor((n + 160 − 321) / 160) + 1.
    expect(frames).toBe(31);
    expect(features.length).toBe(31 * 128);
    // Values from a direct DFT of the same frames (|X| through the HTK filters, log(x + 0.001)).
    for (const [f, m, want] of [[0, 0, -6.907755374908447], [0, 7, 0.8292293548583984], [0, 60, -1.6107534170150757], [3, 7, 1.3510290384292603], [3, 60, -6.310640335083008], [3, 127, -6.886599063873291], [29, 7, 1.3545300960540771], [29, 100, -6.3600687980651855]] as const)
      expect(features[f * 128 + m]).toBeCloseTo(want, 3);
    expect(gemma4AudioFeatures(new Float32Array(100)).frames).toBe(0);
    // Audio past 30 s is dropped.
    expect(gemma4AudioFeatures(new Float32Array(500000)).frames).toBe(2999);
  });

  it("embeds distances with sines then cosines", () => {
    const p = gemma4AudioPositions({ ...c, hidden: 8, contextLeft: 3 });
    expect(p.length).toBe(16);
    expect([...p.subarray(0, 8)]).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    expect(p[8]).toBeCloseTo(Math.sin(1), 6);
    expect(p[8 + 3]).toBeCloseTo(Math.sin(1 / 10000), 9);
    expect(p[8 + 4]).toBeCloseTo(Math.cos(1), 6);
  });

  it("matches the reference forward pass on a small tower", async () => {
    const s: Gemma4AudioConfig = { hidden: 64, heads: 2, layers: 2, mels: 16, convChannels: [8, 8], contextLeft: 13, logitCap: 2, convKernel: 5, residualWeight: 0.5, outputDim: 64, projectionDim: 32, eps: 1e-6, maxTokens: 40 };
    const D = s.hidden / s.heads;
    const w: Weights = new Map();
    let seed = 21;
    const mat = (name: string, n: number, k: number, scale = 0.3) => w.set(name, { shape: [n, k], data: rand(n * k, seed++, scale) });
    const vec = (name: string, n: number) => w.set(name, { shape: [n], data: rand(n, seed++, 0.4).map((v) => 1 + v) });
    const clips = new Map<string, Gemma4Clip>();
    mat("a.input_projection.weight", s.hidden, (s.mels / 4) * s.convChannels[1]);
    for (let l = 0; l < s.layers; l++) {
      const p = `a.blk.${l}.`;
      for (const n of ["ffn_norm", "ffn_post_norm", "ffn_norm_1", "ffn_post_norm_1", "attn_pre_norm", "attn_post_norm", "norm_conv", "conv_norm", "ln2"]) vec(`${p}${n}.weight`, s.hidden);
      for (const n of ["ffn_up", "ffn_up_1"]) mat(`${p}${n}.weight`, 4 * s.hidden, s.hidden);
      for (const n of ["ffn_down", "ffn_down_1"]) mat(`${p}${n}.weight`, s.hidden, 4 * s.hidden);
      for (const n of ["attn_q", "attn_k", "attn_v", "attn_out", "attn_k_rel", "conv_pw2"]) mat(`${p}${n}.weight`, s.hidden, s.hidden, n === "attn_q" || n === "attn_k" ? 1.5 : 0.3);
      mat(`${p}conv_pw1.weight`, 2 * s.hidden, s.hidden);
      mat(`${p}conv_dw.weight`, s.hidden, s.convKernel, 0.6);
      w.set(`${p}per_dim_scale.weight`, { shape: [D], data: rand(D, seed++, 0.5).map((v) => 1 + v) });
      // Limits tight enough to bite; the second layer's attention output has none.
      for (const n of GEMMA4_AUDIO_CLIPPED) clips.set(p + n, l === 1 && n === "attn_out" ? [-Infinity, Infinity, -Infinity, Infinity] : [-1.2, 1.4, -0.9 - 0.1 * l, 1.1]);
    }
    mat("a.pre_encode.out.weight", s.outputDim, s.hidden);
    w.set("a.pre_encode.out.bias", { shape: [s.outputDim], data: rand(s.outputDim, seed++, 0.3) });
    mat("mm.a.input_projection.weight", s.projectionDim, s.outputDim);
    const convs: Gemma4AudioConvs = { w: [rand(8 * 1 * 9, 3, 0.8), rand(8 * 8 * 9, 4, 0.5)], norm: [rand(8, 5, 0.4).map((v) => 1 + v), rand(8, 6, 0.4).map((v) => 1 + v)] };

    const be = new CpuBackend({ threads: 1 });
    const gpu = new Map<string, Tensor>();
    for (const [name, { shape, data }] of w) {
      const buf = be.alloc(data.byteLength, "storage", name);
      be.write(buf, 0, data);
      gpu.set(name, { shape, dtype: "f32", buf });
    }
    const tower = new Gemma4AudioTower(be, s, gpu, clips, convs);
    const dst = be.alloc(s.maxTokens * s.projectionDim * 4, "storage", "features");
    // 33 tokens: three attention blocks of the model, the last partly filled; then a clip shorter than the window.
    for (const frames of [130, 19]) {
      const features = rand(frames * s.mels, frames, 3);
      const sub = gemma4AudioSubsample(s, convs, features, frames);
      expect(sub.tokens).toBe(gemma4AudioTokens(frames, s));
      const want = reference(s, w, clips, convs, features, frames);
      const T = await tower.encode(features, frames, dst, 0);
      expect(T).toBe(sub.tokens);
      expect(want.length).toBe(T * s.projectionDim);
      const got = new Float32Array(await be.readback(dst, 0, want.length * 4));
      const diff = got.reduce((m, v, i) => Math.max(m, Math.abs(v - want[i]!)), 0);
      expect(diff).toBeLessThan(2e-3);
      expect(Math.max(...want.map(Math.abs))).toBeGreaterThan(0.3);
    }
    await expect(tower.encode(new Float32Array(0), 0, dst, 0)).rejects.toThrow(/too short/);
    tower.destroy();
  });

  it("names a checkpoint's tensors as the mmproj file does", () => {
    const n = (s: string) => gemma4AudioName(`model.audio_tower.${s}`);
    expect(n("layers.3.feed_forward1.ffw_layer_1.linear.weight")).toBe("a.blk.3.ffn_up.weight");
    expect(n("layers.3.feed_forward2.ffw_layer_2.output_max")).toBe("a.blk.3.ffn_down_1.output_max");
    expect(n("layers.3.feed_forward2.pre_layer_norm.weight")).toBe("a.blk.3.ffn_norm_1.weight");
    expect(n("layers.0.self_attn.post.linear.weight")).toBe("a.blk.0.attn_out.weight");
    expect(n("layers.0.self_attn.q_proj.input_min")).toBe("a.blk.0.attn_q.input_min");
    expect(n("layers.0.self_attn.relative_k_proj.weight")).toBe("a.blk.0.attn_k_rel.weight");
    expect(n("layers.0.self_attn.per_dim_scale")).toBe("a.blk.0.per_dim_scale.weight");
    expect(n("layers.11.lconv1d.linear_start.linear.weight")).toBe("a.blk.11.conv_pw1.weight");
    expect(n("layers.11.lconv1d.depthwise_conv1d.weight")).toBe("a.blk.11.conv_dw.weight");
    expect(n("layers.11.lconv1d.conv_norm.weight")).toBe("a.blk.11.norm_conv.weight");
    expect(n("layers.11.lconv1d.pre_layer_norm.weight")).toBe("a.blk.11.conv_norm.weight");
    expect(n("layers.11.norm_out.weight")).toBe("a.blk.11.ln2.weight");
    expect(n("subsample_conv_projection.layer1.conv.weight")).toBe("a.conv1d.1.weight");
    expect(n("subsample_conv_projection.input_proj_linear.weight")).toBe("a.input_projection.weight");
    expect(n("output_proj.bias")).toBe("a.pre_encode.out.bias");
    expect(gemma4AudioName("model.embed_audio.embedding_projection.weight")).toBe("mm.a.input_projection.weight");
    expect(gemma4AudioName("model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight")).toBeUndefined();
  });
});
