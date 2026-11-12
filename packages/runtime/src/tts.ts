// Text-to-speech: VITS (MMS-TTS) and SpeechT5, with transformers' VitsTokenizer and transformers.js's output shape
// ({ audio, sampling_rate }).
import type { Backend, Tensor } from "@kiln/core";
import { fetchBytes, fetchJSON, fileUrl, listRepoFiles, parseModelId, readPytorchHeader, readSafetensorsHeader, toF32, weightFiles, type ByteSource, type ModelRef, type TensorEntry } from "@kiln/loader";
import { Seq2SeqModel, SpeechT5Model, UnsupportedModelError, VitsModel, VitsTokenizer, convMatrix, speecht5ConfigFromHF, speecht5Name, speecht5Positional, vitsConfigFromHF, type HifiGanConfig } from "@kiln/models";
import { CharTokenizer } from "@kiln/tokenizers";
import { openBackend, openParsed } from "./load";
import type { AudioLoadOptions } from "./audio";

export interface TextToSpeechOptions {
  /** VITS: scale of the sampled latent noise (default: the model's noise_scale, 0.667 for MMS). 0 is deterministic. Kokoro: 0 leaves out the source's random phases and noise. */
  noise_scale?: number;
  /** VITS: scale of the duration predictor's noise (default: noise_scale_duration, 0.8). */
  noise_scale_duration?: number;
  /** VITS: faster above 1, slower below (divides the durations; default the config's, 1). */
  speaking_rate?: number;
  /** Seed for the noise (VITS, Kokoro) or the prenet dropout (SpeechT5), for repeatable output (default: random, as transformers samples it). */
  seed?: number;
  /** SpeechT5: the speaker's x-vector (512 values), required. */
  speaker_embeddings?: Float32Array;
  /** SpeechT5: stop once a step's stop probabilities sum to this (default 0.5). */
  threshold?: number;
  /** SpeechT5: the speech prenet's dropout, "sample" (default, as transformers) or "keep" (deterministic). */
  prenet_dropout?: "sample" | "keep";
  /** Kokoro: a voice name (default "af_heart") or a voice's style table (510 × 256 float32 values). */
  voice?: string | Float32Array;
  /** Kokoro: faster above 1, slower below (default 1). */
  speed?: number;
}

/** A loaded text-to-speech model (VITS or SpeechT5 with its vocoder). */
export class TtsEngine {
  constructor(
    readonly backend: Backend,
    readonly info: { family: "vits" | "speecht5" | "kokoro"; layers: number; hidden: number; samplingRate: number },
    private readonly run: (text: string, o: TextToSpeechOptions) => Promise<Float32Array>,
    private readonly free: () => void,
  ) {}

  /** Speech for a text: { audio (mono samples in −1…1), sampling_rate }. */
  async synthesize(text: string, o: TextToSpeechOptions = {}): Promise<{ audio: Float32Array; sampling_rate: number }> {
    return { audio: await this.run(text, o), sampling_rate: this.info.samplingRate };
  }

  destroy(): void {
    this.free();
  }
}

/** Opens a repo's weight files and reads tensors from them as f32, by name. */
export async function tensorReader(modelId: string, ref: ModelRef, names: { names: string[]; format: "safetensors" | "pytorch" }, ep: string | undefined) {
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  const close = () => {
    for (const s of sources) s.close?.();
  };
  try {
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name, { source, entry: t });
    }
  } catch (e) {
    close();
    throw e;
  }
  const read = async (name: string) => {
    const t = located.get(name);
    if (!t) return undefined;
    const count = t.entry.shape.reduce((a, b) => a * b, 1);
    return { shape: t.entry.shape, data: toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count) };
  };
  const need = async (name: string) => {
    const t = await read(name);
    if (!t) throw new Error(`${modelId}: "${name}" is missing`);
    return t;
  };
  // A weight-normed layer's weight, from weight_g / weight_v or the parametrizations' original0 / original1.
  const weightOf = async (p: string) => {
    const plain = await read(`${p}.weight`);
    if (plain) return plain;
    const g = (await read(`${p}.weight_g`)) ?? (await need(`${p}.parametrizations.weight.original0`));
    const v = (await read(`${p}.weight_v`)) ?? (await need(`${p}.parametrizations.weight.original1`));
    const O = v.shape[0]!;
    const per = v.data.length / O;
    const out = new Float32Array(v.data.length);
    for (let o = 0; o < O; o++) {
      let n = 0;
      for (let i = 0; i < per; i++) n += v.data[o * per + i]! ** 2;
      const s = g.data[o]! / Math.sqrt(n);
      for (let i = 0; i < per; i++) out[o * per + i] = v.data[o * per + i]! * s;
    }
    return { shape: v.shape, data: out };
  };
  return { names: [...located.keys()], read, need, weightOf, close };
}

type Reader = Awaited<ReturnType<typeof tensorReader>>;

/** Uploads f32 tensors to a weight map. */
function uploader(backend: Backend, weights: Map<string, Tensor>, label: string) {
  return (name: string, shape: number[], data: Float32Array) => {
    const buf = backend.alloc(data.byteLength, "storage", `${label}/${name}`);
    backend.write(buf, 0, data);
    weights.set(name, { shape, dtype: "f32", buf } as Tensor);
  };
}

/** A HiFi-GAN generator's weights (conv_pre, upsampler.i, resblocks.n.convs1/2.j, conv_post) under Kiln's names `${to}.pre` … */
async function loadHifiGan(r: Reader, upload: ReturnType<typeof uploader>, c: HifiGanConfig, from: string, to: string, postBias: boolean): Promise<void> {
  const conv = async (src: string, dst: string, bias = true) => {
    const w = await r.weightOf(src);
    const [O, C, k] = w.shape as [number, number, number];
    const m = convMatrix(w.data, [O, C, 1, k]);
    upload(`${dst}.weight`, [O, m.K], m.w);
    const b = bias ? await r.read(`${src}.bias`) : undefined;
    upload(`${dst}.bias`, [O], b ? b.data : new Float32Array(O));
  };
  await conv(`${from}conv_pre`, `${to}.pre`);
  for (let i = 0; i < c.upsampleRates.length; i++) {
    const w = await r.weightOf(`${from}upsampler.${i}`);
    const [C, O, K] = w.shape as [number, number, number];
    const m = new Float32Array(K * O * C);
    for (let ch = 0; ch < C; ch++) for (let o = 0; o < O; o++) for (let k = 0; k < K; k++) m[(k * O + o) * C + ch] = w.data[(ch * O + o) * K + k]!;
    upload(`${to}.up${i}.weight`, [K, O, C], m);
    upload(`${to}.up${i}.bias`, [O], (await r.need(`${from}upsampler.${i}.bias`)).data);
  }
  const nr = c.upsampleRates.length * c.resblockKernels.length;
  for (let n = 0; n < nr; n++) {
    const dil = c.resblockDilations[n % c.resblockKernels.length]!;
    for (let j = 0; j < dil.length; j++) {
      await conv(`${from}resblocks.${n}.convs1.${j}`, `${to}.rb${n}.c1_${j}`);
      await conv(`${from}resblocks.${n}.convs2.${j}`, `${to}.rb${n}.c2_${j}`);
    }
  }
  await conv(`${from}conv_post`, `${to}.post`, postBias);
}

/**
 * Loads a VITS checkpoint. Everything is read and reshaped here: the text encoder's matrices and
 * kernel-3 convolutions, the WaveNet layers with their weight norm folded in (g · v / ‖v‖ per output
 * channel), the flow's output projection negated (the GPU adds it), the transposed convolutions as
 * [K, O, C], and the duration predictor kept on the CPU.
 */
export async function createTts(modelId: string, ref: ModelRef, config: Record<string, unknown>, names: { names: string[]; format: "safetensors" | "pytorch" }, opts: AudioLoadOptions): Promise<TtsEngine> {
  const cfg = vitsConfigFromHF(config);
  const ep = opts.endpoint;
  const [vocab, tokCfg] = await Promise.all([fetchJSON<Record<string, number>>(fileUrl(ref, "vocab.json", ep)), fetchJSON<Record<string, unknown>>(fileUrl(ref, "tokenizer_config.json", ep), true)]);
  if (!vocab) throw new Error(`${modelId} has no vocab.json`);
  const tokenizer = new VitsTokenizer(vocab, { addBlank: tokCfg?.add_blank !== false, normalize: tokCfg?.normalize !== false, phonemize: tokCfg?.phonemize === true, isUroman: tokCfg?.is_uroman === true });
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const r = await tensorReader(modelId, ref, names, ep);
  try {
    const { read, need, weightOf } = r;
    const weights = new Map<string, Tensor>();
    const upload = uploader(backend, weights, "vits");
    /** A Conv1d [O, C, k] as the conv kernel's [O, K] (k = 1: a plain matrix [O, C]). */
    const conv = async (from: string, to: string, o: { bias?: boolean; negate?: boolean; asLinear?: boolean } = {}) => {
      const w = await weightOf(from);
      const [O, C, k] = w.shape as [number, number, number];
      const sign = o.negate ? -1 : 1;
      if (o.asLinear) {
        if (k !== 1) throw new Error(`${modelId}: ${from} is not a 1×1 convolution`);
        upload(`${to}.weight`, [O, C], w.data.map((v) => v * sign));
      } else {
        const m = convMatrix(w.data, [O, C, 1, k]);
        upload(`${to}.weight`, [O, m.K], m.w.map((v) => v * sign));
      }
      const b = o.bias === false ? undefined : await read(`${from}.bias`);
      upload(`${to}.bias`, [O], b ? b.data.map((v) => v * sign) : new Float32Array(O));
    };
    const linear = async (from: string, to: string) => {
      const w = await need(`${from}.weight`);
      upload(`${to}.weight`, w.shape, w.data);
      upload(`${to}.bias`, [w.shape[0]!], (await read(`${from}.bias`))?.data ?? new Float32Array(w.shape[0]!));
    };
    const vec = async (from: string, to: string) => {
      const t = await need(from);
      upload(to, t.shape.length > 1 ? [t.shape[t.shape.length - 2]!, t.shape[t.shape.length - 1]!] : t.shape, t.data);
    };

    // Text encoder.
    const te = "text_encoder.";
    await vec(te + "embed_tokens.weight", "te.emb.weight");
    for (let l = 0; l < cfg.layers; l++) {
      const p = `${te}encoder.layers.${l}.`;
      for (const [from, to] of [["attention.q_proj", "q"], ["attention.k_proj", "k"], ["attention.v_proj", "v"], ["attention.out_proj", "o"]] as const) await linear(p + from, `te.l${l}.${to}`);
      await vec(p + "attention.emb_rel_k", `te.l${l}.ek`);
      await vec(p + "attention.emb_rel_v", `te.l${l}.ev`);
      for (const [from, to] of [["layer_norm", "ln1"], ["final_layer_norm", "ln2"]] as const) {
        await vec(`${p}${from}.weight`, `te.l${l}.${to}.weight`);
        await vec(`${p}${from}.bias`, `te.l${l}.${to}.bias`);
      }
      await conv(p + "feed_forward.conv_1", `te.l${l}.ff1`);
      await conv(p + "feed_forward.conv_2", `te.l${l}.ff2`);
    }
    await conv(te + "project", "te.proj", { asLinear: true });

    // Flow.
    for (let f = 0; f < cfg.flows; f++) {
      const p = `flow.flows.${f}.`;
      await conv(p + "conv_pre", `fl${f}.pre`, { asLinear: true });
      for (let j = 0; j < cfg.wavenetLayers; j++) {
        await conv(`${p}wavenet.in_layers.${j}`, `fl${f}.in${j}`);
        await conv(`${p}wavenet.res_skip_layers.${j}`, `fl${f}.rs${j}`, { asLinear: true });
      }
      await conv(p + "conv_post", `fl${f}.post`, { asLinear: true, negate: true });
    }

    // HiFi-GAN.
    await loadHifiGan(r, upload, cfg, "decoder.", "dec", false);

    // The duration predictor stays on the CPU.
    const cpu = new Map<string, Float32Array>();
    for (const name of r.names) {
      if (name.startsWith("duration_predictor.") && !name.startsWith("duration_predictor.post_")) cpu.set(name.slice("duration_predictor.".length), (await need(name)).data);
    }
    const cpuWeight = (n: string) => {
      const t = cpu.get(n);
      if (!t) throw new Error(`${modelId}: "duration_predictor.${n}" is missing`);
      return t;
    };
    const model = new VitsModel(backend, cfg, weights, cpuWeight);
    return new TtsEngine(
      backend,
      { family: "vits", layers: cfg.layers, hidden: cfg.hidden, samplingRate: cfg.samplingRate },
      (text, o) => model.synthesize(tokenizer.encode(text), { noiseScale: o.noise_scale, noiseScaleDuration: o.noise_scale_duration, speakingRate: o.speaking_rate, seed: o.seed }),
      () => model.destroy(),
    );
  } finally {
    r.close();
  }
}

/** SpeechT5 load options: the vocoder repo (default microsoft/speecht5_hifigan). */
export interface SpeechT5LoadOptions extends AudioLoadOptions {
  vocoder?: string;
  /** Longest speech, in decoder steps (two frames each; default the model's max_speech_positions). */
  maxSteps?: number;
}

/**
 * Loads SpeechT5ForTextToSpeech and its HiFi-GAN vocoder: the encoder and decoder layers under the
 * seq2seq names (the encoder's scaled positional table precomputed), the speech prenet and output
 * heads kept on the CPU, the postnet's batch norms folded into its convolutions.
 */
export async function createSpeechT5(modelId: string, ref: ModelRef, config: Record<string, unknown>, names: { names: string[]; format: "safetensors" | "pytorch" }, opts: SpeechT5LoadOptions): Promise<TtsEngine> {
  const cfg = speecht5ConfigFromHF(config);
  const ep = opts.endpoint;
  const vocId = opts.vocoder ?? "microsoft/speecht5_hifigan";
  const vref = parseModelId(vocId, undefined);
  const [spm, vconfig, vnames] = await Promise.all([
    fetchBytes(fileUrl(ref, "spm_char.model", ep)).then((b) => b!),
    fetchJSON<Record<string, unknown>>(fileUrl(vref, "config.json", ep)),
    listRepoFiles(vref, ep).then((files) => weightFiles(vref, ep, files)),
  ]);
  const tokenizer = new CharTokenizer(spm);
  if (vconfig?.model_type !== "hifigan") throw new UnsupportedModelError(`${vocId} is not a SpeechT5 HiFi-GAN vocoder`);
  const arr = (k: string, d: number[]) => ((vconfig[k] as number[] | undefined) ?? d).map(Number);
  const voc = {
    upsampleRates: arr("upsample_rates", [4, 4, 4, 4]),
    upsampleKernels: arr("upsample_kernel_sizes", [8, 8, 8, 8]),
    resblockKernels: arr("resblock_kernel_sizes", [3, 7, 11]),
    resblockDilations: ((vconfig.resblock_dilation_sizes as number[][] | undefined) ?? [[1, 3, 5], [1, 3, 5], [1, 3, 5]]).map((d) => d.map(Number)),
    leakySlope: Number(vconfig.leaky_relu_slope ?? 0.1),
    samplingRate: Number(vconfig.sampling_rate ?? 16000),
  };
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const r = await tensorReader(modelId, ref, names, ep);
  const vr = await tensorReader(vocId, vref, vnames, ep).catch((e) => {
    r.close();
    throw e;
  });
  try {
    const d = cfg.seq.dModel;
    const seqWeights = new Map<string, Tensor>();
    const up = uploader(backend, seqWeights, "speecht5");
    const cpu = new Map<string, Float32Array>();
    for (const name of r.names) {
      const to = speecht5Name(name);
      if (to) {
        const t = await r.need(name);
        up(to, t.shape, t.data);
      } else if (name.startsWith("speecht5.decoder.prenet.") || /^speech_decoder_postnet\.(feat|prob)_out\./.test(name)) cpu.set(name, (await r.need(name)).data);
    }
    // The encoder's positions: alpha · the sinusoidal table, looked up by position.
    const alpha = (await r.need("speecht5.encoder.prenet.encode_positions.alpha")).data[0]!;
    const P = cfg.seq.maxPositions;
    const pe = new Float32Array(P * d);
    for (let i = 0; i < P; i++) pe.set(speecht5Positional(i, d).map((v) => Math.fround(alpha * v)), i * d);
    up("enc.pos", [P, d], pe);

    // Postnet: Conv1d (no bias) + BatchNorm1d folded.
    const weights = new Map<string, Tensor>();
    const upload = uploader(backend, weights, "speecht5");
    for (let i = 0; i < cfg.postnetLayers; i++) {
      const p = `speech_decoder_postnet.layers.${i}.`;
      const w = await r.need(`${p}conv.weight`);
      const [O, C, k] = w.shape as [number, number, number];
      const [g, beta, mean, varr] = await Promise.all(["weight", "bias", "running_mean", "running_var"].map((n) => r.need(`${p}batch_norm.${n}`)));
      const scale = Float32Array.from(g!.data, (v, o) => v / Math.sqrt(varr!.data[o]! + 1e-5));
      const folded = w.data.map((v, j) => v * scale[Math.floor(j / (C * k))]!);
      const m = convMatrix(folded, [O, C, 1, k]);
      upload(`post.l${i}.weight`, [O, m.K], m.w);
      upload(`post.l${i}.bias`, [O], Float32Array.from(beta!.data, (b, o) => b - mean!.data[o]! * scale[o]!));
    }
    await loadHifiGan(vr, upload, voc, "", "voc", true);
    const norm = vconfig.normalize_before === false ? undefined : { mean: (await vr.need("mean")).data, scale: (await vr.need("scale")).data };
    const cpuWeight = (n: string) => {
      const t = cpu.get(n);
      if (!t) throw new Error(`${modelId}: "${n}" is missing`);
      return t;
    };
    const seq = new Seq2SeqModel(backend, cfg.seq, seqWeights, { maxTarget: opts.maxSteps ?? cfg.maxSpeechPositions });
    const model = new SpeechT5Model(backend, cfg, voc, seq, weights, cpuWeight, norm);
    return new TtsEngine(
      backend,
      { family: "speecht5", layers: cfg.seq.encLayers, hidden: d, samplingRate: voc.samplingRate },
      (text, o) => {
        if (!o.speaker_embeddings) throw new Error("SpeechT5 needs speaker_embeddings (a 512-value x-vector)");
        return model.synthesize(tokenizer.encode(text), o.speaker_embeddings, { threshold: o.threshold, prenetDropout: o.prenet_dropout, seed: o.seed });
      },
      () => model.destroy(),
    );
  } finally {
    r.close();
    vr.close();
  }
}
