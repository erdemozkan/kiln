// Text-to-speech: VITS (MMS-TTS), with transformers' VitsTokenizer and transformers.js's output shape
// ({ audio, sampling_rate }).
import type { Backend, Tensor } from "@kiln/core";
import { fetchJSON, fileUrl, readPytorchHeader, readSafetensorsHeader, toF32, type ByteSource, type ModelRef, type TensorEntry } from "@kiln/loader";
import { VitsModel, VitsTokenizer, convMatrix, vitsConfigFromHF, type VitsConfig } from "@kiln/models";
import { openBackend, openParsed } from "./load";
import type { AudioLoadOptions } from "./audio";

export interface TextToSpeechOptions {
  /** Scale of the sampled latent noise (default: the model's noise_scale, 0.667 for MMS). 0 is deterministic. */
  noise_scale?: number;
  /** Scale of the duration predictor's noise (default: noise_scale_duration, 0.8). */
  noise_scale_duration?: number;
  /** Faster above 1, slower below (divides the durations; default the config's, 1). */
  speaking_rate?: number;
  /** Seed for the noise, for repeatable output (default: random, as transformers samples it). */
  seed?: number;
}

export class TtsEngine {
  constructor(
    readonly backend: Backend,
    readonly cfg: VitsConfig,
    readonly model: VitsModel,
    readonly tokenizer: VitsTokenizer,
  ) {}

  /** Speech for a text: { audio (mono samples in −1…1), sampling_rate }. */
  async synthesize(text: string, o: TextToSpeechOptions = {}): Promise<{ audio: Float32Array; sampling_rate: number }> {
    const audio = await this.model.synthesize(this.tokenizer.encode(text), { noiseScale: o.noise_scale, noiseScaleDuration: o.noise_scale_duration, speakingRate: o.speaking_rate, seed: o.seed });
    return { audio, sampling_rate: this.cfg.samplingRate };
  }

  destroy(): void {
    this.model.destroy();
  }
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
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name, { source, entry: t });
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
    const weights = new Map<string, Tensor>();
    const upload = (name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `vits/${name}`);
      backend.write(buf, 0, data);
      weights.set(name, { shape, dtype: "f32", buf } as Tensor);
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
    await conv("decoder.conv_pre", "dec.pre");
    for (let i = 0; i < cfg.upsampleRates.length; i++) {
      const w = await weightOf(`decoder.upsampler.${i}`);
      const [C, O, K] = w.shape as [number, number, number];
      const m = new Float32Array(K * O * C);
      for (let c = 0; c < C; c++) for (let o = 0; o < O; o++) for (let k = 0; k < K; k++) m[(k * O + o) * C + c] = w.data[(c * O + o) * K + k]!;
      upload(`dec.up${i}.weight`, [K, O, C], m);
      upload(`dec.up${i}.bias`, [O], (await need(`decoder.upsampler.${i}.bias`)).data);
    }
    const nr = cfg.upsampleRates.length * cfg.resblockKernels.length;
    for (let r = 0; r < nr; r++) {
      const dil = cfg.resblockDilations[r % cfg.resblockKernels.length]!;
      for (let j = 0; j < dil.length; j++) {
        await conv(`decoder.resblocks.${r}.convs1.${j}`, `dec.rb${r}.c1_${j}`);
        await conv(`decoder.resblocks.${r}.convs2.${j}`, `dec.rb${r}.c2_${j}`);
      }
    }
    await conv("decoder.conv_post", "dec.post", { bias: false });

    // The duration predictor stays on the CPU.
    const cpu = new Map<string, Float32Array>();
    for (const name of located.keys()) {
      if (name.startsWith("duration_predictor.") && !name.startsWith("duration_predictor.post_")) cpu.set(name.slice("duration_predictor.".length), (await need(name)).data);
    }
    const cpuWeight = (n: string) => {
      const t = cpu.get(n);
      if (!t) throw new Error(`${modelId}: "duration_predictor.${n}" is missing`);
      return t;
    };
    return new TtsEngine(backend, cfg, new VitsModel(backend, cfg, weights, cpuWeight), tokenizer);
  } finally {
    for (const s of sources) s.close?.();
  }
}
