import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";
import { conv1d } from "./hifigan";

/** Kokoro (StyleTTS 2 with an iSTFTNet decoder), from its config.json. */
export interface KokoroConfig {
  vocab: Record<string, number>;
  nToken: number;
  hidden: number;
  styleDim: number;
  nLayer: number;
  maxDur: number;
  textKernel: number;
  albert: { hidden: number; heads: number; intermediate: number; maxPositions: number; layers: number; embedding: number };
  istft: { upsampleRates: number[]; upsampleKernels: number[]; resblockKernels: number[]; resblockDilations: number[][]; initialChannel: number; nFft: number; hop: number };
}

export function kokoroConfigFromHF(c: Record<string, unknown>): KokoroConfig {
  const i = c.istftnet as Record<string, unknown> | undefined;
  const pl = c.plbert as Record<string, unknown> | undefined;
  if (!i || !pl || !c.vocab) throw new UnsupportedModelError("not a Kokoro config (istftnet, plbert and vocab expected)");
  const cfg: KokoroConfig = {
    vocab: c.vocab as Record<string, number>,
    nToken: Number(c.n_token),
    hidden: Number(c.hidden_dim),
    styleDim: Number(c.style_dim),
    nLayer: Number(c.n_layer),
    maxDur: Number(c.max_dur),
    textKernel: Number(c.text_encoder_kernel_size),
    albert: {
      hidden: Number(pl.hidden_size),
      heads: Number(pl.num_attention_heads),
      intermediate: Number(pl.intermediate_size),
      maxPositions: Number(pl.max_position_embeddings),
      layers: Number(pl.num_hidden_layers),
      embedding: Number(pl.embedding_size ?? 128),
    },
    istft: {
      upsampleRates: (i.upsample_rates as number[]).map(Number),
      upsampleKernels: (i.upsample_kernel_sizes as number[]).map(Number),
      resblockKernels: (i.resblock_kernel_sizes as number[]).map(Number),
      resblockDilations: (i.resblock_dilation_sizes as number[][]).map((d) => d.map(Number)),
      initialChannel: Number(i.upsample_initial_channel),
      nFft: Number(i.gen_istft_n_fft),
      hop: Number(i.gen_istft_hop_size),
    },
  };
  if (cfg.istft.upsampleRates.length !== 2) throw new UnsupportedModelError("Kokoro with other than two upsamplers is not supported");
  if (cfg.hidden !== 512 || cfg.styleDim !== 128) throw new UnsupportedModelError("Kokoro with other widths than 512 / style 128 is not supported");
  return cfg;
}

/** Channels padded to a multiple of 4 for the conv kernel. */
export const pad4 = (c: number) => Math.ceil(c / 4) * 4;

/**
 * A style-conditioned norm's parameters: fc [2C, style] maps the style to (γ, β); the GPU reads
 * 1 + γ and β (padded to Cp with zeros) from "{name}.w" / "{name}.b", written per utterance.
 * `half`: which half of the 256-value reference style it reads (decoder: 0, predictor: 1).
 */
export interface KokoroStyleNorm {
  name: string;
  w: Float32Array;
  b: Float32Array;
  C: number;
  Cp: number;
  half: 0 | 1;
}

/** A plan with its input buffers and outputs. */
interface Plan {
  dispatches: readonly Dispatch[];
  ctx: Ctx;
  step: Buf;
  inputs: Record<string, Buf>;
  outputs: Record<string, Tensor>;
}

/**
 * Kokoro's networks on the GPU, as three plans:
 * - text (per token count T): ALBERT (PL-BERT) and its projection, the duration encoder (three
 *   bidirectional LSTMs with AdaLayerNorm), the duration LSTM and projection, and the text encoder
 *   (embedding, three conv + LayerNorm + LeakyReLU, an LSTM) → duration logits, d [T, 640], t_en [T, 512];
 * - prosody (per frame count F): the shared LSTM over the aligned d, then the F0 and N stacks
 *   (AdaIN residual blocks, one upsampling 2×) → F0 and N curves [2F];
 * - decoder (per F): the AdaIN residual encoder and decoder blocks over the aligned t_en with the
 *   downsampled curves, then the iSTFTNet generator (the harmonic source's spectrum in, through
 *   the noise convolutions and blocks) → conv_post's [L, n_fft + 2] (log-magnitudes, then phases).
 * Style-dependent norms read per-utterance buffers (see setStyle).
 */
export class KokoroModel {
  private readonly pool: ActivationPool;
  private readonly plans = new Map<string, Plan>();

  constructor(
    readonly backend: Backend,
    readonly cfg: KokoroConfig,
    private readonly weights: Map<string, Tensor>,
    private readonly norms: KokoroStyleNorm[],
  ) {
    this.pool = new ActivationPool(backend, "kokoro");
    for (const n of norms) {
      for (const k of ["w", "b"]) {
        const buf = backend.alloc(n.Cp * 4, "storage", `kokoro/${n.name}.${k}`);
        weights.set(`${n.name}.${k}`, { shape: [n.Cp], dtype: "f32", buf } as Tensor);
      }
    }
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`kokoro weight "${name}" was not loaded`);
    return t;
  }

  private lin(ctx: Ctx, x: Tensor, p: string, residual?: Tensor): Tensor {
    return ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.weights.get(`${p}.bias`), residual });
  }

  private ln(ctx: Ctx, x: Tensor, p: string, eps: number): Tensor {
    return ops.layernorm(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), eps);
  }

  /** Writes every style-conditioned norm's 1 + γ and β for a reference style [256]. */
  setStyle(style: Float32Array): void {
    const S = this.cfg.styleDim;
    for (const n of this.norms) {
      const s = style.subarray(n.half * S, (n.half + 1) * S);
      const w = new Float32Array(n.Cp);
      const b = new Float32Array(n.Cp);
      for (let o = 0; o < 2 * n.C; o++) {
        let acc = n.b[o]!;
        for (let i = 0; i < S; i++) acc += n.w[o * S + i]! * s[i]!;
        if (o < n.C) w[o] = 1 + acc;
        else b[o - n.C] = acc;
      }
      this.backend.write(this.w(`${n.name}.w`).buf, 0, w);
      this.backend.write(this.w(`${n.name}.b`).buf, 0, b);
    }
  }

  /** A bidirectional LSTM layer ("{p}.wih" [8H, In] with "{p}.b", "{p}.whhT"). */
  private lstm(ctx: Ctx, x: Tensor, p: string): Tensor {
    const pre = ops.linear(ctx, x, this.w(`${p}.wih`), { bias: this.w(`${p}.b`) });
    return ops.lstm(ctx, pre, this.w(`${p}.whhT`));
  }

  /** Columns of several tensors side by side in a new [T, width] tensor; the rest filled with copies of `fill`'s first column (finite, met by zero weights). */
  private concat(ctx: Ctx, T: number, width: number, parts: Tensor[], fill: Tensor): Tensor {
    const y = ctx.alloc([T, width]);
    let at = 0;
    for (const p of parts) {
      const n = p.shape[p.shape.length - 1]!;
      ops.cols(ctx, y, p, { n, dstStart: at });
      at += n;
    }
    for (; at < width; at++) ops.cols(ctx, y, fill, { n: 1, dstStart: at });
    return y;
  }

  /** AdaIN: InstanceNorm over time per channel, scaled and shifted by the style ("{p}.w" / "{p}.b"). */
  private adain(ctx: Ctx, x: Tensor, p: string): Tensor {
    return ops.channelNorm(ctx, x, this.w(`${p}.w`), this.w(`${p}.b`), 1e-5);
  }

  /**
   * StyleTTS 2's AdainResBlk1d on x [T, Cin]: AdaIN, LeakyReLU 0.2, (2× depthwise transposed conv),
   * conv 3, AdaIN, LeakyReLU, conv 3; plus the shortcut (nearest 2× then 1×1 conv when learned),
   * the sum over √2.
   */
  private adainRes(ctx: Ctx, x: Tensor, p: string, o: { upsample: boolean; learned: boolean }): Tensor {
    const T = x.shape[0]!;
    let h = this.adain(ctx, x, `${p}.n1`);
    let leakyIn: number | undefined = 0.2;
    if (o.upsample) {
      h = ops.dwConvTranspose1d(ctx, h, this.w(`${p}.pool.weight`), this.w(`${p}.pool.bias`), { s: 2, pad: 1, tout: 2 * T, leakyIn: 0.2 });
      leakyIn = undefined;
    }
    h = conv1d(ctx, (n) => this.w(n), h, `${p}.conv1`, 3, { leakyIn });
    h = this.adain(ctx, h, `${p}.n2`);
    let y: Tensor;
    if (!o.upsample) {
      const sc = o.learned ? conv1d(ctx, (n) => this.w(n), x, `${p}.sc`, 1) : x;
      y = conv1d(ctx, (n) => this.w(n), h, `${p}.conv2`, 3, { leakyIn: 0.2, residual: sc });
    } else {
      if (!o.learned) throw new Error("kokoro: an upsampling block without a learned shortcut");
      const r = conv1d(ctx, (n) => this.w(n), h, `${p}.conv2`, 3, { leakyIn: 0.2 });
      // 1×1 conv and nearest upsampling commute: the shortcut at T rows, upsampled into the sum.
      const sc = conv1d(ctx, (n) => this.w(n), x, `${p}.sc`, 1);
      y = ops.nearestAdd(ctx, sc, r, { n: 1, H: 1, W: T, Ho: 1, Wo: 2 * T });
    }
    const out = ctx.alloc(y.shape);
    ops.scaleAdd(ctx, out, y, Math.fround(1 / Math.SQRT2), false);
    return out;
  }

  /** The generator's AdaINResBlock1: per dilation, AdaIN, Snake, conv; AdaIN, Snake, conv; plus the input. */
  private adainRes1(ctx: Ctx, x: Tensor, p: string, k: number, dil: number[]): Tensor {
    let y = x;
    dil.forEach((d, j) => {
      const a = this.adain(ctx, y, `${p}.n1_${j}`);
      ops.snake(ctx, a, this.w(`${p}.a1_${j}`));
      const b = this.adain(ctx, conv1d(ctx, (n) => this.w(n), a, `${p}.c1_${j}`, k, { dil: d }), `${p}.n2_${j}`);
      ops.snake(ctx, b, this.w(`${p}.a2_${j}`));
      y = conv1d(ctx, (n) => this.w(n), b, `${p}.c2_${j}`, k, { residual: y });
    });
    return y;
  }

  private plan(key: string, build: () => Omit<Plan, "dispatches"> & { ctx: Ctx }): Plan {
    let p = this.plans.get(key);
    if (!p) {
      if (this.plans.size >= 9) {
        const [k, old] = this.plans.entries().next().value!;
        old.ctx.destroy();
        this.backend.free(old.step);
        this.plans.delete(k);
      }
      const b = build();
      p = { ...b, dispatches: b.ctx.dispatches };
      this.plans.set(key, p);
    }
    return p;
  }

  private newCtx(label: string, T: number): { ctx: Ctx; step: Buf } {
    const step = this.backend.alloc(STEP_BYTES, "storage", `kokoro/${label}-step`);
    this.backend.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0, spanLo: 0, spanHi: T }));
    return { ctx: new Ctx(this.backend, step, `kokoro/${label}`, this.pool), step };
  }

  private buildText(T: number) {
    const c = this.cfg;
    const a = c.albert;
    const { ctx, step } = this.newCtx(`text T${T}`, T);
    const ids = ctx.buffer(T * 4, "kokoro/ids");
    const pos = ctx.buffer(T * 4, "kokoro/pos");
    const types = ctx.buffer(T * 4, "kokoro/types");
    const style = ctx.buffer(T * c.styleDim * 4, "kokoro/style-rows");
    const sRows: Tensor = { shape: [T, c.styleDim], dtype: "f32", buf: style };
    // ALBERT: 128-wide embeddings, LayerNorm, mapped to the hidden width, one post-norm layer applied `layers` times.
    let x = ops.embed(ctx, this.w("a.word"), ids, T);
    ops.addInPlace(ctx, x, ops.embed(ctx, this.w("a.pos"), pos, T));
    ops.addInPlace(ctx, x, ops.embed(ctx, this.w("a.type"), types, T));
    x = this.lin(ctx, this.ln(ctx, x, "a.lnE", 1e-12), "a.map");
    const D = a.hidden / a.heads;
    const kc = ctx.buffer(T * a.hidden * 4, "kokoro/k");
    const vc = ctx.buffer(T * a.hidden * 4, "kokoro/v");
    const shape = { T, H: a.heads, Hkv: a.heads, D, maxSeq: T, kv: "f32" as const, scale: D ** -0.5, span: true };
    for (let l = 0; l < a.layers; l++) {
      const q = this.lin(ctx, x, "a.q");
      ops.kvWrite(ctx, this.lin(ctx, x, "a.k"), this.lin(ctx, x, "a.v"), kc, vc, T, T, "f32");
      this.lin(ctx, ops.attention(ctx, q, kc, vc, shape), "a.o", x);
      x = this.ln(ctx, x, "a.ln1", 1e-12);
      const up = this.lin(ctx, x, "a.ffn");
      ops.act(ctx, up, "gelu_tanh");
      this.lin(ctx, up, "a.ffo", x);
      x = this.ln(ctx, x, "a.ln2", 1e-12);
    }
    const den = this.lin(ctx, x, "benc");
    // Duration encoder: [x, s] → LSTM → AdaLayerNorm, three times; d keeps the style columns.
    let d = this.concat(ctx, T, c.hidden + c.styleDim, [den, sRows], den);
    for (let l = 0; l < c.nLayer; l++) {
      const h = this.lstm(ctx, d, `de${l}`);
      const n = ops.layernorm(ctx, h, this.w(`de${l}.ln.w`), this.w(`de${l}.ln.b`), 1e-5);
      d = this.concat(ctx, T, c.hidden + c.styleDim, [n, sRows], n);
    }
    const dur = this.lin(ctx, this.lstm(ctx, d, "plstm"), "dproj");
    // Text encoder.
    let t = ops.embed(ctx, this.w("te.emb"), ids, T);
    for (let l = 0; l < c.nLayer; l++) {
      t = conv1d(ctx, (n) => this.w(n), t, `te.c${l}`, c.textKernel, { leakyIn: l ? 0.2 : undefined });
      t = this.ln(ctx, t, `te.ln${l}`, 1e-5);
    }
    ops.leakyRelu(ctx, t, 0.2);
    const ten = this.lstm(ctx, t, "te.lstm");
    // d's last use is a read, so the planner may reuse its memory: a copy written last is kept.
    const dOut = ctx.alloc(d.shape);
    ops.scaleAdd(ctx, dOut, d, 1, false);
    return { ctx, step, inputs: { ids, pos, types, style }, outputs: { dur, d: dOut, ten } };
  }

  /** Duration logits [T, maxDur], d [T, hidden + style] and t_en [T, hidden] for token ids (style set first). */
  async text(ids: readonly number[]): Promise<{ dur: Float32Array; d: Float32Array; ten: Float32Array }> {
    const T = ids.length;
    const c = this.cfg;
    if (T > c.albert.maxPositions) throw new Error(`kokoro: ${T} tokens; the model takes at most ${c.albert.maxPositions}`);
    const p = this.plan(`text${T}`, () => this.buildText(T));
    const be = this.backend;
    be.write(p.inputs.ids!, 0, Uint32Array.from(ids));
    be.write(p.inputs.pos!, 0, Uint32Array.from({ length: T }, (_, i) => i));
    be.write(p.inputs.types!, 0, new Uint32Array(T));
    be.write(p.inputs.style!, 0, this.styleRows);
    be.submit(p.dispatches);
    const [dur, d, ten] = await Promise.all(["dur", "d", "ten"].map((k) => this.read(p.outputs[k]!)));
    return { dur: dur!, d: d!, ten: ten! };
  }

  /** The predictor's style [style] repeated per token row; set by the caller with setStyleRows. */
  private styleRows = new Float32Array(0);

  /** The predictor half of the style repeated for T rows (the duration encoder's concatenations). */
  setStyleRows(style: Float32Array, T: number): void {
    const S = this.cfg.styleDim;
    const rows = new Float32Array(T * S);
    for (let t = 0; t < T; t++) rows.set(style.subarray(S, 2 * S), t * S);
    this.styleRows = rows;
  }

  private async read(t: Tensor): Promise<Float32Array> {
    return new Float32Array(await this.backend.readback(physical(t.buf), 0, t.shape.reduce((a, b) => a * b, 1) * 4));
  }

  private buildProsody(F: number) {
    const c = this.cfg;
    const { ctx, step } = this.newCtx(`prosody F${F}`, F);
    const en = ctx.buffer(F * (c.hidden + c.styleDim) * 4, "kokoro/en");
    const x = this.lstm(ctx, { shape: [F, c.hidden + c.styleDim], dtype: "f32", buf: en }, "shared");
    const curve = (p: string) => {
      let h = this.adainRes(ctx, x, `${p}0`, { upsample: false, learned: false });
      h = this.adainRes(ctx, h, `${p}1`, { upsample: true, learned: true });
      h = this.adainRes(ctx, h, `${p}2`, { upsample: false, learned: false });
      return conv1d(ctx, (n) => this.w(n), h, `${p}proj`, 1);
    };
    return { ctx, step, inputs: { en }, outputs: { f0: curve("f0."), n: curve("n.") } };
  }

  /** F0 and N curves [2F] from the aligned duration-encoder output en [F, hidden + style]. */
  async prosody(en: Float32Array, F: number): Promise<{ f0: Float32Array; n: Float32Array }> {
    const p = this.plan(`prosody${F}`, () => this.buildProsody(F));
    this.backend.write(p.inputs.en!, 0, en);
    this.backend.submit(p.dispatches);
    const [f0, n] = await Promise.all([this.read(p.outputs.f0!), this.read(p.outputs.n!)]);
    return { f0: f0!, n: n! };
  }

  /** Rows of the harmonic source's spectrum the generator takes for F frames. */
  sourceFrames(F: number): number {
    const r = this.cfg.istft.upsampleRates;
    return 2 * F * r[0]! * r[1]! + 1;
  }

  private buildDecoder(F: number) {
    const c = this.cfg;
    const g = c.istft;
    const { ctx, step } = this.newCtx(`decoder F${F}`, F);
    const asrBuf = ctx.buffer(F * c.hidden * 4, "kokoro/asr");
    const f0Buf = ctx.buffer(F * 4, "kokoro/f0c");
    const nBuf = ctx.buffer(F * 4, "kokoro/nc");
    const L = this.sourceFrames(F);
    const harW = pad4(g.nFft + 2);
    const harBuf = ctx.buffer(L * harW * 4, "kokoro/har");
    const asr: Tensor = { shape: [F, c.hidden], dtype: "f32", buf: asrBuf };
    const f0: Tensor = { shape: [F, 1], dtype: "f32", buf: f0Buf };
    const nc: Tensor = { shape: [F, 1], dtype: "f32", buf: nBuf };
    const W = (n: string) => this.w(n);
    let x = this.adainRes(ctx, this.concat(ctx, F, pad4(c.hidden + 2), [asr, f0, nc], f0), "dec.enc", { upsample: false, learned: true });
    const asrRes = conv1d(ctx, W, asr, "dec.asrres", 1);
    for (let i = 0; i < 4; i++) {
      const up = i === 3;
      x = this.adainRes(ctx, this.concat(ctx, F, pad4(1024 + 64 + 2), [x, asrRes, f0, nc], f0), `dec.d${i}`, { upsample: up, learned: true });
    }
    // Generator.
    const har: Tensor = { shape: [L, harW], dtype: "f32", buf: harBuf };
    const nk = g.resblockKernels.length;
    g.upsampleRates.forEach((u, i) => {
      const last = i === g.upsampleRates.length - 1;
      const stride = last ? 1 : g.upsampleRates.slice(i + 1).reduce((a, b) => a * b, 1);
      const k = last ? 1 : 2 * stride;
      let src = ops.conv2d(ctx, har, W(`g.nc${i}.weight`), W(`g.nc${i}.bias`), { H: 1, W: L, k, kh: 1, stride, pad: last ? 0 : Math.floor((stride + 1) / 2), padH: 0 }).y;
      src = this.adainRes1(ctx, src, `g.nr${i}`, last ? 11 : 7, [1, 3, 5]);
      const K = g.upsampleKernels[i]!;
      x = ops.convTranspose1d(ctx, x, W(`g.up${i}.weight`), W(`g.up${i}.bias`), { s: u, pad: (K - u) / 2, leakyIn: 0.1 });
      if (last) x = ops.padReflectLeft(ctx, x, 1);
      ops.addInPlace(ctx, x, src);
      const sum = ctx.alloc(x.shape);
      g.resblockKernels.forEach((kr, r) => {
        const y = this.adainRes1(ctx, x, `g.rb${i * nk + r}`, kr, g.resblockDilations[r]!);
        if (r === 0) ops.scaleAdd(ctx, sum, y, 1, false);
        else ops.addInPlace(ctx, sum, y);
      });
      const avg = ctx.alloc(x.shape);
      ops.scaleAdd(ctx, avg, sum, 1 / nk, false);
      x = avg;
    });
    const post = conv1d(ctx, W, x, "g.post", 7, { leakyIn: 0.01 });
    return { ctx, step, inputs: { asr: asrBuf, f0: f0Buf, n: nBuf, har: harBuf }, outputs: { post } };
  }

  /**
   * conv_post's output [L, n_fft + 2] from the aligned t_en asr [F, hidden], the downsampled F0 and
   * N curves [F] and the harmonic source's spectrum har [L, pad4(n_fft + 2)] (magnitudes, phases).
   */
  async decode(asr: Float32Array, f0: Float32Array, n: Float32Array, har: Float32Array, F: number): Promise<Float32Array> {
    const p = this.plan(`decoder${F}`, () => this.buildDecoder(F));
    const be = this.backend;
    be.write(p.inputs.asr!, 0, asr);
    be.write(p.inputs.f0!, 0, f0);
    be.write(p.inputs.n!, 0, n);
    be.write(p.inputs.har!, 0, har);
    // The generator is seconds of work for a long sentence: submitted in slices, each waited on once the next is queued.
    const d = p.dispatches;
    const per = Math.max(1, Math.ceil(d.length / 16));
    let pending: Promise<ArrayBuffer> | undefined;
    for (let i = 0; i < d.length; i += per) {
      be.submit(d.slice(i, i + per));
      const done = be.readback(p.step, 0, 4);
      if (pending) await pending;
      pending = done;
    }
    await pending;
    return this.read(p.outputs.post!);
  }

  destroy(): void {
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    for (const p of this.plans.values()) {
      p.ctx.destroy();
      this.backend.free(p.step);
    }
    this.pool.destroy();
  }
}
