// Kokoro text-to-speech (hexgrad/Kokoro-82M: StyleTTS 2 with an iSTFTNet decoder). The networks run on
// the GPU (see KokoroModel); durations, the harmonic source and its STFT, and the inverse STFT run here.
import type { Tensor } from "@kiln/core";
import { fileUrl, type ModelRef } from "@kiln/loader";
import { KokoroModel, convMatrix, kokoroConfigFromHF, pad4, type KokoroConfig, type KokoroStyleNorm } from "@kiln/models";
import { openBackend } from "./load";
import type { AudioLoadOptions } from "./audio";
import { TtsEngine, tensorReader, type TextToSpeechOptions } from "./tts";

/** Where voices load from by name: onnx-community's export holds them as raw float32 [510, 256]. */
export const KOKORO_VOICES = "onnx-community/Kokoro-82M-v1.0-ONNX";

/** torch.round: halves to even. */
const roundEven = (x: number) => {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r;
};

/** Frames per token from the duration logits [T, maxDur]: Σ sigmoid / speed, rounded (halves to even), at least 1. */
export function kokoroDurations(logits: Float32Array, T: number, maxDur: number, speed = 1): number[] {
  const f = Math.fround;
  return Array.from({ length: T }, (_, t) => {
    let s = 0;
    for (let j = 0; j < maxDur; j++) s = f(s + f(1 / (1 + Math.exp(-logits[t * maxDur + j]!))));
    return Math.max(1, roundEven(f(s / speed)));
  });
}

/** Rows of x [T, C] repeated per their durations: [Σ durations, C]. */
export function repeatRows(x: Float32Array, C: number, durations: readonly number[]): Float32Array {
  const F = durations.reduce((a, b) => a + b, 0);
  const out = new Float32Array(F * C);
  let at = 0;
  durations.forEach((n, t) => {
    for (let i = 0; i < n; i++, at++) out.set(x.subarray(t * C, (t + 1) * C), at * C);
  });
  return out;
}

/** Conv1d(1, 1, 3, stride 2, padding 1) of a curve: [⌈n / 2⌉]. */
export function strideConv(x: Float32Array, w: Float32Array, b: number): Float32Array {
  const n = Math.floor((x.length + 2 - 3) / 2) + 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = b;
    for (let k = 0; k < 3; k++) {
      const j = 2 * i - 1 + k;
      if (j >= 0 && j < x.length) s += w[k]! * x[j]!;
    }
    out[i] = s;
  }
  return out;
}

/** mulberry32 with Box-Muller normals, for repeatable noise. */
function rng(seed: number) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  return { uniform, normal };
}

/**
 * torch's upsample_linear1d (align_corners=False, scale_factor given) of x [n] in float32: output
 * floor(n · scale), source coordinate (1 / scale) · (i + 0.5) − 0.5 in float, clamped at 0.
 */
function interpLinear(x: Float32Array, scale: number): Float32Array {
  const f = Math.fround;
  const n = x.length;
  const m = Math.floor(n * scale);
  const r = f(1 / scale);
  const out = new Float32Array(m);
  for (let i = 0; i < m; i++) {
    const src = Math.max(f(f(r * f(i + 0.5)) - 0.5), 0);
    const i0 = Math.floor(src);
    const l1 = f(src - i0);
    const l0 = f(1 - l1);
    const i1 = i0 < n - 1 ? i0 + 1 : i0;
    out[i] = f(f(l0 * x[i0]!) + f(l1 * x[i1]!));
  }
  return out;
}

/**
 * The generator's harmonic source (SourceModuleHnNSF with SineGen, 8 harmonics), in float32 as
 * torch computes it: the F0 curve upsampled (nearest) to samples; per harmonic, the phase increment
 * f·h / sr (mod 1) plus a random initial phase (not the fundamental's); integrated at the curve's
 * rate (linear down- and upsampling around the cumulative sum, which torch accumulates in double);
 * sines times 0.1 where voiced (F0 > 10), plus Gaussian noise (0.003 voiced, 0.1 / 3 unvoiced);
 * merged by the linear layer and tanh. `noise` undefined leaves out the random phases and the noise.
 */
export function harmonicSource(f0: Float32Array, upsample: number, lin: { w: Float32Array; b: number }, sr: number, noise: { uniform(): number; normal(): number } | undefined): Float32Array {
  const f = Math.fround;
  const H = lin.w.length;
  const n = f0.length * upsample;
  const sum = new Float32Array(n);
  const pi = f(Math.PI);
  for (let h = 0; h < H; h++) {
    const rad = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = f(f(f0[Math.floor(i / upsample)]! * (h + 1)) / sr);
      rad[i] = f(v - Math.floor(v));
    }
    if (h > 0 && noise) rad[0] = f(rad[0]! + f(noise.uniform()));
    const down = interpLinear(rad, 1 / upsample);
    let acc = 0;
    for (let i = 0; i < down.length; i++) {
      acc += down[i]!;
      down[i] = f(f(f(f(acc) * 2) * pi) * upsample);
    }
    const phase = interpLinear(down, upsample);
    for (let i = 0; i < n; i++) {
      const voiced = f0[Math.floor(i / upsample)]! > 10;
      let s = voiced ? f(f(Math.sin(phase[i]!)) * f(0.1)) : 0;
      if (noise) s = f(s + f(f(voiced ? 0.003 : 0.1 / 3) * f(noise.normal())));
      sum[i] = f(sum[i]! + f(lin.w[h]! * s));
    }
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.tanh(f(sum[i]! + lin.b));
  return out;
}

/** torch's periodic Hann window. */
const hann = (n: number) => Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));

/**
 * torch.stft (center, reflect padding, periodic Hann) of x: per frame its magnitudes then phases,
 * [frames, pad4(nFft + 2)] (the last columns zero).
 */
export function stftMagPhase(x: Float32Array, nFft: number, hop: number): Float32Array {
  const half = nFft / 2;
  const n = x.length;
  const frames = Math.floor(n / hop) + 1;
  const bins = half + 1;
  const width = pad4(nFft + 2);
  const win = hann(nFft);
  const at = (i: number) => {
    const j = i - half;
    return x[j < 0 ? -j : j >= n ? 2 * (n - 1) - j : j]!;
  };
  const out = new Float32Array(frames * width);
  for (let f = 0; f < frames; f++) {
    for (let k = 0; k < bins; k++) {
      let re = 0;
      let im = 0;
      for (let t = 0; t < nFft; t++) {
        const v = at(f * hop + t) * win[t]!;
        const a = (-2 * Math.PI * k * t) / nFft;
        re += v * Math.cos(a);
        im += v * Math.sin(a);
      }
      out[f * width + k] = Math.hypot(re, im);
      out[f * width + bins + k] = Math.atan2(im, re);
    }
  }
  return out;
}

/**
 * The generator's output to samples: magnitudes exp(y[:, :bins]) and phases sin(y[:, bins:]) of
 * y [frames, nFft + 2], through torch.istft (center, periodic Hann, divided by the window's squared
 * overlap): hop · (frames − 1) samples.
 */
export function istft(y: Float32Array, frames: number, nFft: number, hop: number): Float32Array {
  const bins = nFft / 2 + 1;
  const C = nFft + 2;
  const win = hann(nFft);
  const total = nFft + hop * (frames - 1);
  const sig = new Float64Array(total);
  const env = new Float64Array(total);
  const frame = new Float64Array(nFft);
  for (let f = 0; f < frames; f++) {
    const re: number[] = [];
    const im: number[] = [];
    for (let k = 0; k < bins; k++) {
      const m = Math.exp(y[f * C + k]!);
      const p = Math.sin(y[f * C + bins + k]!);
      re.push(m * Math.cos(p));
      im.push(m * Math.sin(p));
    }
    // irfft: the DC and Nyquist bins' imaginary parts are ignored.
    for (let t = 0; t < nFft; t++) {
      let s = re[0]! + re[bins - 1]! * (t % 2 ? -1 : 1);
      for (let k = 1; k < bins - 1; k++) {
        const a = (2 * Math.PI * k * t) / nFft;
        s += 2 * (re[k]! * Math.cos(a) - im[k]! * Math.sin(a));
      }
      frame[t] = s / nFft;
    }
    for (let t = 0; t < nFft; t++) {
      sig[f * hop + t]! += frame[t]! * win[t]!;
      env[f * hop + t]! += win[t]! ** 2;
    }
  }
  const half = nFft / 2;
  const n = hop * (frames - 1);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const e = env[i + half]!;
    out[i] = e > 1e-11 ? sig[i + half]! / e : sig[i + half]!;
  }
  return out;
}

/** Loads Kokoro from its repo (config.json and the kokoro-*.pth checkpoint). */
export async function createKokoro(modelId: string, ref: ModelRef, config: Record<string, unknown>, weightFile: string, opts: AudioLoadOptions): Promise<TtsEngine> {
  const cfg = kokoroConfigFromHF(config);
  const ep = opts.endpoint;
  const r = await tensorReader(modelId, ref, { names: [weightFile], format: "pytorch" }, ep);
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const weights = new Map<string, Tensor>();
  const norms: KokoroStyleNorm[] = [];
  const up = (name: string, shape: number[], data: Float32Array) => {
    const buf = backend.alloc(Math.max(data.byteLength, 4), "storage", `kokoro/${name}`);
    backend.write(buf, 0, data);
    weights.set(name, { shape, dtype: "f32", buf } as Tensor);
  };
  try {
    const plain = async (to: string, from: string) => {
      const t = await r.need(from);
      up(to, t.shape, t.data);
    };
    const linear = async (to: string, from: string) => {
      await plain(`${to}.weight`, `${from}.weight`);
      await plain(`${to}.bias`, `${from}.bias`);
    };
    /** A Conv1d [O, C, k] (weight-normed or not) as the conv kernel's [O, K], input channels padded to cp. */
    const conv = async (to: string, from: string, cp?: number, bias = true) => {
      const w = await r.weightOf(from);
      const [O, C, k] = w.shape as [number, number, number];
      const m = convMatrix(w.data, [O, C, 1, k], cp ?? pad4(C));
      up(`${to}.weight`, [O, m.K], m.w);
      up(`${to}.bias`, [O], bias ? (await r.need(`${from}.bias`)).data : new Float32Array(O));
    };
    const styleNorm = async (name: string, from: string, C: number, half: 0 | 1) => {
      const [w, b] = [await r.need(`${from}.weight`), await r.need(`${from}.bias`)];
      norms.push({ name, w: w.data, b: b.data, C, Cp: pad4(C), half });
    };
    /** A bidirectional LSTM: [W_ih; W_ih_reverse] [8H, In], both biases summed, W_hh transposed per direction. */
    const lstm = async (to: string, from: string) => {
      const g = (n: string) => r.need(`${from}.${n}`);
      const [wf, wr, hf, hr, bif, bhf, bir, bhr] = await Promise.all(["weight_ih_l0", "weight_ih_l0_reverse", "weight_hh_l0", "weight_hh_l0_reverse", "bias_ih_l0", "bias_hh_l0", "bias_ih_l0_reverse", "bias_hh_l0_reverse"].map(g));
      const [G, In] = wf!.shape as [number, number];
      const H = G / 4;
      const wih = new Float32Array(2 * G * In);
      wih.set(wf!.data);
      wih.set(wr!.data, G * In);
      up(`${to}.wih`, [2 * G, In], wih);
      const b = new Float32Array(2 * G);
      for (let i = 0; i < G; i++) {
        b[i] = bif!.data[i]! + bhf!.data[i]!;
        b[G + i] = bir!.data[i]! + bhr!.data[i]!;
      }
      up(`${to}.b`, [2 * G], b);
      const whhT = new Float32Array(2 * H * G);
      [hf!, hr!].forEach((w, d) => {
        for (let row = 0; row < G; row++) for (let k = 0; k < H; k++) whhT[(d * H + k) * G + row] = w.data[row * H + k]!;
      });
      up(`${to}.whhT`, [2, H, G], whhT);
    };
    /** StyleTTS 2's AdainResBlk1d. */
    const block = async (to: string, from: string, cin: number, cout: number, o: { upsample?: boolean; half: 0 | 1 }) => {
      const cp = pad4(cin);
      await conv(`${to}.conv1`, `${from}.conv1`, cp);
      await conv(`${to}.conv2`, `${from}.conv2`);
      if (cin !== cout) await conv(`${to}.sc`, `${from}.conv1x1`, cp, false);
      if (o.upsample) {
        const w = await r.weightOf(`${from}.pool`);
        const pw = new Float32Array(3 * cp);
        for (let c = 0; c < cin; c++) for (let k = 0; k < 3; k++) pw[k * cp + c] = w.data[c * 3 + k]!;
        up(`${to}.pool.weight`, [3, cp], pw);
        const pb = new Float32Array(cp);
        pb.set((await r.need(`${from}.pool.bias`)).data);
        up(`${to}.pool.bias`, [cp], pb);
      }
      await styleNorm(`${to}.n1`, `${from}.norm1.fc`, cin, o.half);
      await styleNorm(`${to}.n2`, `${from}.norm2.fc`, cout, o.half);
    };
    /** The generator's AdaINResBlock1 (three dilations, Snake). */
    const block1 = async (to: string, from: string, C: number, n: number) => {
      for (let j = 0; j < n; j++) {
        await conv(`${to}.c1_${j}`, `${from}.convs1.${j}`);
        await conv(`${to}.c2_${j}`, `${from}.convs2.${j}`);
        await plain(`${to}.a1_${j}`, `${from}.alpha1.${j}`);
        await plain(`${to}.a2_${j}`, `${from}.alpha2.${j}`);
        await styleNorm(`${to}.n1_${j}`, `${from}.adain1.${j}.fc`, C, 0);
        await styleNorm(`${to}.n2_${j}`, `${from}.adain2.${j}.fc`, C, 0);
      }
    };

    // ALBERT (one shared layer).
    const B = "bert.module.";
    const L = `${B}encoder.albert_layer_groups.0.albert_layers.0.`;
    await plain("a.word", `${B}embeddings.word_embeddings.weight`);
    await plain("a.pos", `${B}embeddings.position_embeddings.weight`);
    await plain("a.type", `${B}embeddings.token_type_embeddings.weight`);
    await linear("a.lnE", `${B}embeddings.LayerNorm`);
    await linear("a.map", `${B}encoder.embedding_hidden_mapping_in`);
    for (const [to, from] of [["a.q", "attention.query"], ["a.k", "attention.key"], ["a.v", "attention.value"], ["a.o", "attention.dense"], ["a.ln1", "attention.LayerNorm"], ["a.ffn", "ffn"], ["a.ffo", "ffn_output"], ["a.ln2", "full_layer_layer_norm"]]) await linear(to!, L + from);
    await linear("benc", "bert_encoder.module");
    // Prosody predictor.
    const P = "predictor.module.";
    for (let l = 0; l < cfg.nLayer; l++) {
      await lstm(`de${l}`, `${P}text_encoder.lstms.${2 * l}`);
      await styleNorm(`de${l}.ln`, `${P}text_encoder.lstms.${2 * l + 1}.fc`, cfg.hidden, 1);
    }
    await lstm("plstm", `${P}lstm`);
    await linear("dproj", `${P}duration_proj.linear_layer`);
    await lstm("shared", `${P}shared`);
    for (const [to, from] of [["f0.", "F0"], ["n.", "N"]] as const) {
      await block(`${to}0`, `${P}${from}.0`, 512, 512, { half: 1 });
      await block(`${to}1`, `${P}${from}.1`, 512, 256, { upsample: true, half: 1 });
      await block(`${to}2`, `${P}${from}.2`, 256, 256, { half: 1 });
      await conv(`${to}proj`, `${P}${from}_proj`);
    }
    // Text encoder.
    const T = "text_encoder.module.";
    await plain("te.emb", `${T}embedding.weight`);
    for (let l = 0; l < cfg.nLayer; l++) {
      await conv(`te.c${l}`, `${T}cnn.${l}.0`);
      await plain(`te.ln${l}.weight`, `${T}cnn.${l}.1.gamma`);
      await plain(`te.ln${l}.bias`, `${T}cnn.${l}.1.beta`);
    }
    await lstm("te.lstm", `${T}lstm`);
    // Decoder and generator.
    const D = "decoder.module.";
    await block("dec.enc", `${D}encode`, cfg.hidden + 2, 1024, { half: 0 });
    for (let i = 0; i < 4; i++) await block(`dec.d${i}`, `${D}decode.${i}`, 1024 + 64 + 2, i === 3 ? 512 : 1024, { upsample: i === 3, half: 0 });
    await conv("dec.asrres", `${D}asr_res.0`);
    const G = `${D}generator.`;
    const g = cfg.istft;
    const nk = g.resblockKernels.length;
    for (let i = 0; i < g.upsampleRates.length; i++) {
      const C = g.initialChannel / 2 ** (i + 1);
      const w = await r.weightOf(`${G}ups.${i}`);
      const [Cin, O, K] = w.shape as [number, number, number];
      const m = new Float32Array(K * O * Cin);
      for (let c = 0; c < Cin; c++) for (let o = 0; o < O; o++) for (let t = 0; t < K; t++) m[(t * O + o) * Cin + c] = w.data[(c * O + o) * K + t]!;
      up(`g.up${i}.weight`, [K, O, Cin], m);
      up(`g.up${i}.bias`, [O], (await r.need(`${G}ups.${i}.bias`)).data);
      await conv(`g.nc${i}`, `${G}noise_convs.${i}`, pad4(g.nFft + 2));
      await block1(`g.nr${i}`, `${G}noise_res.${i}`, C, 3);
      for (let j = 0; j < nk; j++) await block1(`g.rb${i * nk + j}`, `${G}resblocks.${i * nk + j}`, C, g.resblockDilations[j]!.length);
    }
    await conv("g.post", `${G}conv_post`);
    // On the CPU: the curves' stride-2 convolutions and the source's linear layer.
    const f0Conv = await r.weightOf(`${D}F0_conv`);
    const nConv = await r.weightOf(`${D}N_conv`);
    const f0b = (await r.need(`${D}F0_conv.bias`)).data[0]!;
    const nb = (await r.need(`${D}N_conv.bias`)).data[0]!;
    const srcLin = { w: (await r.need(`${G}m_source.l_linear.weight`)).data, b: (await r.need(`${G}m_source.l_linear.bias`)).data[0]! };
    const model = new KokoroModel(backend, cfg, weights, norms);
    return kokoroEngine(ref, cfg, model, { f0Conv: f0Conv.data, f0b, nConv: nConv.data, nb, srcLin }, opts);
  } finally {
    r.close();
  }
}

function kokoroEngine(ref: ModelRef, cfg: KokoroConfig, model: KokoroModel, cpu: { f0Conv: Float32Array; f0b: number; nConv: Float32Array; nb: number; srcLin: { w: Float32Array; b: number } }, opts: AudioLoadOptions): TtsEngine {
  const voices = new Map<string, Float32Array>();
  const voiceRepo = parseVoiceRepo(ref);
  const voiceOf = async (v: string | Float32Array | undefined): Promise<Float32Array> => {
    if (v instanceof Float32Array) return v;
    const name = v ?? "af_heart";
    let pack = voices.get(name);
    if (!pack) {
      const res = await fetch(fileUrl(voiceRepo, `voices/${name}.bin`, opts.endpoint));
      if (!res.ok) throw new Error(`Kokoro voice "${name}": HTTP ${res.status} (voices are ${KOKORO_VOICES}'s voices/*.bin)`);
      pack = new Float32Array(await res.arrayBuffer());
      voices.set(name, pack);
    }
    return pack;
  };
  const sr = 24000;
  const g = cfg.istft;
  const upsample = g.upsampleRates.reduce((a, b) => a * b, 1) * g.hop;
  const run = async (phonemes: string, o: TextToSpeechOptions): Promise<Float32Array> => {
    const ids = [...phonemes].map((ch) => cfg.vocab[ch]).filter((i): i is number => i !== undefined);
    if (!ids.length) throw new Error("Kokoro: no phonemes the model knows (it reads phonemes, not text; see the `phonemize` option)");
    if (ids.length + 2 > cfg.albert.maxPositions) throw new Error(`Kokoro: ${ids.length} phonemes; the model takes at most ${cfg.albert.maxPositions - 2} at a time`);
    const pack = await voiceOf(o.voice);
    const rows = pack.length / (2 * cfg.styleDim);
    // Python's KPipeline: the voice row for len(phonemes) − 1.
    const row = Math.min(Math.max([...phonemes].length - 1, 0), rows - 1);
    const style = pack.subarray(row * 2 * cfg.styleDim, (row + 1) * 2 * cfg.styleDim);
    const tokens = [0, ...ids, 0];
    const T = tokens.length;
    model.setStyle(style);
    model.setStyleRows(style, T);
    const { dur, d, ten } = await model.text(tokens);
    const durations = kokoroDurations(dur, T, cfg.maxDur, o.speed ?? 1);
    const F = durations.reduce((a, b) => a + b, 0);
    const en = repeatRows(d, cfg.hidden + cfg.styleDim, durations);
    const asr = repeatRows(ten, cfg.hidden, durations);
    const { f0, n } = await model.prosody(en, F);
    const noise = o.noise_scale === 0 ? undefined : rng(o.seed ?? Math.floor(Math.random() * 2 ** 32));
    const source = harmonicSource(f0, upsample, cpu.srcLin, sr, noise);
    const har = stftMagPhase(source, g.nFft, g.hop);
    const post = await model.decode(asr, strideConv(f0, cpu.f0Conv, cpu.f0b), strideConv(n, cpu.nConv, cpu.nb), har, F);
    return istft(post, model.sourceFrames(F), g.nFft, g.hop);
  };
  return new TtsEngine(model.backend, { family: "kokoro", layers: cfg.albert.layers, hidden: cfg.hidden, samplingRate: sr }, run, () => model.destroy());
}

/** The voices' repo for a Kokoro repo: its own when it is the ONNX export, else onnx-community's. */
function parseVoiceRepo(ref: ModelRef): ModelRef {
  return /Kokoro-82M-v1\.0-ONNX$/.test(ref.repo) ? ref : { ...ref, repo: KOKORO_VOICES, revision: "main" };
}

/** config.json of a Kokoro repo, when it is one (no model_type; istftnet and plbert sections). */
export function isKokoroConfig(c: Record<string, unknown> | undefined): boolean {
  return !!c && !c.model_type && !!c.istftnet && !!c.plbert;
}

