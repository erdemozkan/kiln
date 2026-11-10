// Speech encoders on raw audio (wav2vec2, HuBERT): CTC speech recognition and audio classification,
// with transformers.js's pre- and post-processing.
import type { Backend, Tensor } from "@kiln/core";
import { fetchJSON, fileUrl, listRepoFiles, parseModelId, readPytorchHeader, readSafetensorsHeader, toF32, weightFiles, type ByteSource, type TensorEntry } from "@kiln/loader";
import { Wav2Vec2Model, wav2vec2ConfigFromHF, wav2vec2Name, type Wav2Vec2Config } from "@kiln/models";
import { ctcTokenizer, type CtcTokenizer } from "@kiln/tokenizers";
import { openBackend, openParsed } from "./load";
import { loadWeights, type LoadProgress, type WeightFile } from "./weights";
import { createSpeechT5, createTts, type SpeechT5LoadOptions, type TtsEngine } from "./tts";
import { createKokoro, isKokoroConfig } from "./kokoro";

export interface AudioLoadOptions {
  /** Matrix storage: "f32" (default), "q8" or "q4". */
  quant?: "f32" | "q8" | "q4";
  device?: "auto" | "gpu" | "cpu";
  cpuThreads?: number;
  revision?: string;
  endpoint?: string;
  backend?: Backend;
  onProgress?: (p: LoadProgress) => void;
}

export interface AudioClassificationOptions {
  /** Labels returned, best first (default 5; 0 or negative: all). */
  top_k?: number;
}

/** Zero mean, unit variance (variance + 1e-7), as Wav2Vec2FeatureExtractor's do_normalize. */
export function normalizeAudio(x: Float32Array): Float32Array {
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i]!;
  mean /= x.length || 1;
  let v = 0;
  for (let i = 0; i < x.length; i++) v += (x[i]! - mean) ** 2;
  const inv = 1 / Math.sqrt(v / (x.length || 1) + 1e-7);
  return x.map((s) => (s - mean) * inv);
}

/** The highest-scoring id of each row of [rows, width]. */
export function argmaxRows(data: Float32Array, rows: number, width: number): Int32Array {
  const out = new Int32Array(rows);
  for (let r = 0; r < rows; r++) {
    let best = 0;
    for (let i = 1; i < width; i++) if (data[r * width + i]! > data[r * width + best]!) best = i;
    out[r] = best;
  }
  return out;
}

export interface ClassifierHead {
  projector: Float32Array;
  projectorBias: Float32Array;
  classifier: Float32Array;
  classifierBias: Float32Array;
}

/** Mean over frames, projector, classifier: a sequence classifier's logits from hidden states [frames, hidden]. */
export function classifyHidden(h: Float32Array, frames: number, hidden: number, head: ClassifierHead): Float32Array {
  // The projector is affine, so averaging first gives the same result as projecting every frame.
  const mean = new Float32Array(hidden);
  for (let t = 0; t < frames; t++) for (let i = 0; i < hidden; i++) mean[i] += h[t * hidden + i]!;
  for (let i = 0; i < hidden; i++) mean[i] /= frames;
  const affine = (x: Float32Array, w: Float32Array, b: Float32Array) => {
    const out = new Float32Array(b.length);
    for (let o = 0; o < b.length; o++) {
      let s = b[o]!;
      for (let i = 0; i < x.length; i++) s += w[o * x.length + i]! * x[i]!;
      out[o] = s;
    }
    return out;
  };
  return affine(affine(mean, head.projector, head.projectorBias), head.classifier, head.classifierBias);
}

/** A loaded speech encoder with its task head. */
export class AudioEngine {
  constructor(
    readonly backend: Backend,
    readonly cfg: Wav2Vec2Config,
    readonly model: Wav2Vec2Model,
    /** The CTC decoder (speech recognition models). */
    readonly tokenizer: CtcTokenizer | undefined,
    /** The pooled classifier (audio classification models). */
    readonly classifier: ClassifierHead | undefined,
    /** Inputs are normalized to zero mean and unit variance first. */
    readonly normalize: boolean,
    readonly samplingRate: number,
  ) {}

  private input(samples: Float32Array): Float32Array {
    return this.normalize ? normalizeAudio(samples) : samples;
  }

  /** CTC speech recognition on 16 kHz mono samples: frame-wise best tokens, merged and decoded. */
  async transcribe(samples: Float32Array): Promise<{ text: string }> {
    if (!this.tokenizer) throw new Error(`${this.cfg.architectures[0] ?? this.cfg.modelType} has no CTC head`);
    const { data, frames } = await this.model.forward(this.input(samples));
    return { text: this.tokenizer.decode(argmaxRows(data, frames, data.length / frames)) };
  }

  /** Audio classification: softmax over the labels, the top_k best. */
  async classify(samples: Float32Array, opts: AudioClassificationOptions = {}): Promise<{ label: string; score: number }[]> {
    const head = this.classifier;
    if (!head) throw new Error(`${this.cfg.architectures[0] ?? this.cfg.modelType} has no classification head`);
    const { data, frames } = await this.model.forward(this.input(samples));
    const logits = classifyHidden(data, frames, this.cfg.hidden, head);
    const max = Math.max(...logits);
    const exp = Array.from(logits, (v) => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    const ranked = exp.map((e, i) => ({ label: this.cfg.id2label?.[i] ?? `LABEL_${i}`, score: e / sum })).sort((a, b) => b.score - a.score);
    const k = opts.top_k ?? 5;
    return k > 0 ? ranked.slice(0, k) : ranked;
  }

  destroy(): void {
    this.model.destroy();
  }
}

/** Positional conv weight [O, C/groups, k] from weight norm (g, v) or a plain weight, stored [k, C/groups, O] for groupConv1d. */
export function positionalConvWeight(shape: readonly number[], v: Float32Array, g?: { shape: readonly number[]; data: Float32Array }): Float32Array {
  const [O, Cg, k] = shape as [number, number, number];
  const w = v.slice();
  if (g) {
    // weight_norm(dim=2) (wav2vec2): one norm per tap over [O, Cg]; dim=0: one per output channel.
    const dim = g.shape.findIndex((s, i) => s > 1 && s === shape[i]);
    if (dim === 2) {
      for (let j = 0; j < k; j++) {
        let n = 0;
        for (let i = 0; i < O * Cg; i++) n += v[i * k + j]! ** 2;
        const f = g.data[j]! / Math.sqrt(n);
        for (let i = 0; i < O * Cg; i++) w[i * k + j] = v[i * k + j]! * f;
      }
    } else if (dim === 0) {
      for (let o = 0; o < O; o++) {
        let n = 0;
        for (let i = 0; i < Cg * k; i++) n += v[o * Cg * k + i]! ** 2;
        const f = g.data[o]! / Math.sqrt(n);
        for (let i = 0; i < Cg * k; i++) w[o * Cg * k + i] = v[o * Cg * k + i]! * f;
      }
    } else throw new Error(`positional conv: weight norm over shape [${g.shape}] is not supported`);
  }
  const out = new Float32Array(w.length);
  for (let o = 0; o < O; o++) for (let c = 0; c < Cg; c++) for (let j = 0; j < k; j++) out[(j * Cg + c) * O + o] = w[(o * Cg + c) * k + j]!;
  return out;
}

/** Conv weight [O, C, k] → matmul rows [O, row] tap-major (row = k·C rounded up to 32), im2col1d's layout. */
export function convRows(shape: readonly number[], raw: Float32Array): { data: Float32Array; row: number } {
  const [O, C, k] = shape as [number, number, number];
  const row = Math.ceil((k * C) / 32) * 32;
  const data = new Float32Array(O * row);
  for (let o = 0; o < O; o++) for (let c = 0; c < C; c++) for (let j = 0; j < k; j++) data[o * row + j * C + c] = raw[(o * C + c) * k + j]!;
  return { data, row };
}

/** Loads a wav2vec2 / HuBERT model with a CTC or sequence-classification head. */
export async function createAudio(modelId: string, opts: AudioLoadOptions & Pick<SpeechT5LoadOptions, "vocoder" | "maxSteps"> = {}): Promise<AudioEngine | TtsEngine> {
  const ref = parseModelId(modelId, opts.revision);
  if (ref.gguf) throw new Error("speech models load from safetensors or PyTorch repos, not GGUF files");
  const ep = opts.endpoint;
  const listing = listRepoFiles(ref, ep);
  const has = async (f: string) => {
    const files = await listing;
    return !files || files.has(f);
  };
  const optional = <T>(f: string) => has(f).then((y) => (y ? fetchJSON<T>(fileUrl(ref, f, ep), true) : undefined));
  const [config, prep, tokJson, vocab, tokCfg, names] = await Promise.all([
    fetchJSON<Record<string, unknown>>(fileUrl(ref, "config.json", ep)),
    optional<Record<string, unknown>>("preprocessor_config.json"),
    optional<Parameters<typeof ctcTokenizer>[0]["tokenizerJson"]>("tokenizer.json"),
    optional<Record<string, number>>("vocab.json"),
    optional<Record<string, unknown>>("tokenizer_config.json"),
    // Kokoro's checkpoint is a .pth, which weightFiles does not look for.
    listing.then((files) => weightFiles(ref, ep, files)).catch((e: unknown) => e as Error),
  ]);
  // Text-to-speech models share the speech channel.
  if (isKokoroConfig(config)) {
    const files = await listing;
    const pth = files ? [...files].filter((f) => /^kokoro.*\.pth$/.test(f)).sort()[0] : "kokoro-v1_0.pth";
    if (!pth) throw new Error(`${modelId}: no kokoro-*.pth checkpoint`);
    return createKokoro(modelId, ref, config!, pth, opts);
  }
  if (names instanceof Error) throw names;
  if (config!.model_type === "vits") return createTts(modelId, ref, config!, names, opts);
  if (config!.model_type === "speecht5") return createSpeechT5(modelId, ref, config!, names, opts);
  const cfg = wav2vec2ConfigFromHF(config!);
  const arch = cfg.architectures.join(" ");
  const classify = /ForSequenceClassification|ForAudioClassification/.test(arch);
  const ctc = !classify && (/ForCTC/.test(arch) || !!vocab || !!tokJson);
  if (!classify && !ctc) throw new Error(`${modelId} (${arch || cfg.modelType}) has neither a CTC nor a classification head`);
  const tokenizer = ctc ? ctcTokenizer({ tokenizerJson: tokJson, vocab, config: tokCfg }) : undefined;

  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    const files: WeightFile[] = [];
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name.replace(/^(wav2vec2|hubert)\./, ""), { source, entry: t });
      files.push({ name, source, tensors: header.tensors.filter((t) => wav2vec2Name(t.name) !== undefined), rename: wav2vec2Name });
    }
    const read = async (name: string): Promise<{ shape: number[]; data: Float32Array } | undefined> => {
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
    const weights = await loadWeights(backend, files, opts.quant ?? "f32", opts.onProgress);
    const upload = (name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `wav2vec2/${name}`);
      backend.write(buf, 0, data);
      weights.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    for (let i = 0; i < cfg.conv.length; i++) {
      const t = await need(`feature_extractor.conv_layers.${i}.conv.weight`);
      const { data, row } = convRows(t.shape, t.data);
      upload(`fe.${i}.weight`, [t.shape[0]!, row], data);
    }
    const p = "encoder.pos_conv_embed.conv.";
    const v = (await read(p + "weight_v")) ?? (await read(p + "parametrizations.weight.original1")) ?? (await need(p + "weight"));
    const g = (await read(p + "weight_g")) ?? (await read(p + "parametrizations.weight.original0"));
    upload("pos.weight", [cfg.posKernel, v.shape[1]!, v.shape[0]!], positionalConvWeight(v.shape, v.data, g));

    let classifier: ClassifierHead | undefined;
    let layerWeights: number[] | undefined;
    if (classify) {
      const [pw, pb, cw, cb] = await Promise.all(["projector.weight", "projector.bias", "classifier.weight", "classifier.bias"].map(need));
      classifier = { projector: pw!.data, projectorBias: pb!.data, classifier: cw!.data, classifierBias: cb!.data };
      if (cfg.weightedLayerSum) {
        const lw = Array.from((await need("layer_weights")).data);
        const max = Math.max(...lw);
        const e = lw.map((x) => Math.exp(x - max));
        const s = e.reduce((a, b) => a + b, 0);
        layerWeights = e.map((x) => x / s);
      }
    }
    const model = new Wav2Vec2Model(backend, cfg, weights, { head: classify ? "hidden" : "ctc", layerWeights });
    const normalize = prep?.do_normalize !== undefined ? !!prep.do_normalize : !!tokCfg?.do_normalize;
    return new AudioEngine(backend, cfg, model, tokenizer, classifier, normalize, Number(prep?.sampling_rate ?? 16000));
  } finally {
    for (const s of sources) s.close?.();
  }
}
