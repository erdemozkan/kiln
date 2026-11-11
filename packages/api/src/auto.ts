// transformers.js's AutoModel family and AutoProcessor on Kiln's engines. The models run in Kiln's
// worker; tensors in and out have transformers.js's names and shapes (int64 ids, float32 outputs,
// [batch, tokens, …]). Processors run on the calling thread.
import { fetchJSON, fileUrl, listRepoFiles, normalizeAudio, parseModelId, preprocessImage, readImage, whisperFeatures, type ImageInput, type ImageProcessorConfig, type RgbImage } from "@kiln/loader";
import { KilnEncoder, KilnModel, KilnVision } from "./client";
import { PreTrainedTokenizer, RawImage, envEndpoint } from "./compat";
import { encoderQuant, toGenerateOptions, toLoadOptions, type PipelineOptions, type Streamer } from "./pipeline";
import type { EncoderHead, RawOutputs, VisionHead } from "./protocol";
import { Tensor } from "./tensor";

/** Loading options, as transformers.js takes them (`dtype`, `device`, `progress_callback`, `revision`) plus Kiln's. */
export type ModelOptions = PipelineOptions;

type Config = Record<string, unknown>;
type Inputs = Record<string, unknown>;

const ENCODERS = new Set(["bert", "distilbert", "roberta"]);
const VISION = new Set(["vit", "deit", "dinov2", "clip", "resnet", "convnext"]);

async function readConfig(id: string, options: ModelOptions): Promise<Config> {
  const ref = parseModelId(id, options.revision);
  if (ref.gguf) return { model_type: "gguf" };
  return (await fetchJSON<Config>(fileUrl(ref, "config.json", options.endpoint ?? envEndpoint())))!;
}

function withEndpoint(options: ModelOptions): ModelOptions {
  return options.endpoint === undefined && envEndpoint() ? { ...options, endpoint: envEndpoint() } : options;
}

/** Token id rows from `input_ids` (and `attention_mask`, `token_type_ids`): each row without its padding. */
function idRows(inputs: Inputs): { rows: { ids: number[]; typeIds?: number[] }[]; keep: number[][]; batch: number; length: number } {
  const ids = inputs.input_ids;
  if (!(ids instanceof Tensor)) throw new Error("expected `input_ids` (a Tensor, as the tokenizer returns it)");
  const [batch, length] = ids.dims.length === 1 ? [1, ids.dims[0]!] : [ids.dims[0]!, ids.dims[1]!];
  const num = (t: unknown) => (t instanceof Tensor ? Array.from(t.data as ArrayLike<number | bigint>, Number) : undefined);
  const all = num(ids)!;
  const mask = num(inputs.attention_mask);
  const types = num(inputs.token_type_ids);
  const rows: { ids: number[]; typeIds?: number[] }[] = [];
  const keep: number[][] = [];
  for (let b = 0; b < batch; b++) {
    const k: number[] = [];
    for (let t = 0; t < length; t++) if (!mask || mask[b * length + t]) k.push(t);
    keep.push(k);
    rows.push({ ids: k.map((t) => all[b * length + t]!), ...(types ? { typeIds: k.map((t) => types[b * length + t]!) } : {}) });
  }
  return { rows, keep, batch, length };
}

/** Outputs with one row per token, by head (the rest have one value per input). */
const PER_TOKEN: Record<string, readonly string[]> = { base: ["last_hidden_state"], token: ["logits"], qa: ["start_logits", "end_logits"], mlm: ["logits"] };

/**
 * Per-input outputs as batch tensors: per-token outputs [batch, length, …] with each row's values at
 * its tokens' positions (zeros at padding, where transformers.js would hold the masked rows); the
 * others stacked [batch, …].
 */
function stack(outs: RawOutputs[], perToken: readonly string[], keep?: number[][], length = 0): Record<string, Tensor> {
  const result: Record<string, Tensor> = {};
  for (const name of Object.keys(outs[0] ?? {})) {
    const first = outs[0]![name]!;
    if (keep && perToken.includes(name)) {
      const inner = first.dims.slice(1);
      const w = inner.reduce((a, b) => a * b, 1);
      const data = new Float32Array(outs.length * length * w);
      outs.forEach((o, b) => {
        const d = o[name]!.data;
        keep[b]!.forEach((t, j) => data.set(d.subarray(j * w, (j + 1) * w), (b * length + t) * w));
      });
      result[name] = new Tensor("float32", data, [outs.length, length, ...inner]);
    } else {
      const w = first.data.length;
      const data = new Float32Array(outs.length * w);
      outs.forEach((o, b) => data.set(o[name]!.data, b * w));
      result[name] = new Tensor("float32", data, [outs.length, ...first.dims]);
    }
  }
  return result;
}

/** Pixel values [batch, 3, height, width] split into images for the worker. */
function images(inputs: Inputs): { data: Float32Array; width: number; height: number }[] {
  const px = inputs.pixel_values;
  if (!(px instanceof Tensor)) throw new Error("expected `pixel_values` (a Tensor, as the image processor returns it)");
  const dims = px.dims.length === 3 ? [1, ...px.dims] : px.dims;
  const [n, c, h, w] = dims as [number, number, number, number];
  if (c !== 3) throw new Error(`pixel_values must be [batch, 3, height, width], got [${px.dims.join(", ")}]`);
  const data = px.data as Float32Array;
  return Array.from({ length: n }, (_, i) => ({ data: data.slice(i * 3 * h * w, (i + 1) * 3 * h * w), width: w, height: h }));
}

const normalized = (v: Float32Array) => {
  const n = Math.max(Math.hypot(...v), 1e-12);
  return v.map((x) => x / n);
};

function stackRows(rows: Float32Array[]): Tensor {
  const w = rows[0]?.length ?? 0;
  const data = new Float32Array(rows.length * w);
  rows.forEach((r, i) => data.set(r, i * w));
  return new Tensor("float32", data, [rows.length, w]);
}

/**
 * transformers.js's PreTrainedModel: call it with the tokenizer's or processor's output for its
 * outputs (`last_hidden_state`, `logits`, …), as Tensors.
 */
export class PreTrainedModel {
  /** The model's config.json. */
  readonly config: Config;
  protected constructor(
    config: Config,
    /** The Kiln handle the model runs in. */
    readonly kiln: KilnEncoder | KilnVision | KilnModel,
    protected readonly head: string,
  ) {
    this.config = config;
    const call = (inputs: Inputs) => this.forward(inputs);
    Object.setPrototypeOf(call, Object.getPrototypeOf(this));
    return Object.assign(call, this) as unknown as PreTrainedModel;
  }

  /** The forward pass on named inputs (`input_ids`, `attention_mask`, `token_type_ids`, `pixel_values`). */
  async forward(inputs: Inputs): Promise<Record<string, Tensor>> {
    const k = this.kiln;
    if (k instanceof KilnEncoder) {
      const { rows, keep, length } = idRows(inputs);
      return stack(await k.forward(rows, this.head as EncoderHead), PER_TOKEN[this.head] ?? [], keep, length);
    }
    if (k instanceof KilnVision) {
      if (this.head === "clip-text" || this.head === "clip-image" || this.head === "clip") return this.clip(k, inputs);
      const out = await k.forward({ images: images(inputs) }, this.head as VisionHead);
      return stack(out.images, []);
    }
    throw new Error("Kiln runs causal language models through generate(); call model.generate({ ...inputs, max_new_tokens })");
  }

  /** CLIPModel / CLIPTextModelWithProjection / CLIPVisionModelWithProjection outputs. */
  private async clip(k: KilnVision, inputs: Inputs): Promise<Record<string, Tensor>> {
    const wantText = this.head !== "clip-image";
    const wantImage = this.head !== "clip-text";
    const texts = wantText && inputs.input_ids ? idRows(inputs).rows.map((r) => r.ids) : undefined;
    const imgs = wantImage && inputs.pixel_values ? images(inputs) : undefined;
    if (wantText && !texts && this.head === "clip-text") throw new Error("expected `input_ids`");
    if (wantImage && !imgs && this.head === "clip-image") throw new Error("expected `pixel_values`");
    const out = await k.forward({ images: imgs, texts }, "clip");
    const imageEmbeds = out.images.map((o) => o.image_embeds!.data);
    if (this.head === "clip-text") return { text_embeds: stackRows(out.texts!) };
    if (this.head === "clip-image") return { image_embeds: stackRows(imageEmbeds) };
    // CLIPModel: normalized embeddings and their scaled similarities.
    const ti = (out.texts ?? []).map(normalized);
    const ii = imageEmbeds.map(normalized);
    const result: Record<string, Tensor> = {};
    if (ti.length) result.text_embeds = stackRows(ti);
    if (ii.length) result.image_embeds = stackRows(ii);
    if (ti.length && ii.length) {
      const per = new Float32Array(ii.length * ti.length);
      const perText = new Float32Array(ti.length * ii.length);
      ii.forEach((a, i) =>
        ti.forEach((b, j) => {
          let d = 0;
          for (let x = 0; x < a.length; x++) d += a[x]! * b[x]!;
          per[i * ti.length + j] = perText[j * ii.length + i] = out.scale! * d;
        }),
      );
      result.logits_per_image = new Tensor("float32", per, [ii.length, ti.length]);
      result.logits_per_text = new Tensor("float32", perText, [ti.length, ii.length]);
    }
    return result;
  }

  /** Unloads the model and stops its worker. */
  dispose(): Promise<void> {
    return this.kiln.unload();
  }

  /** Loads `id` for this class's outputs (see the Auto classes for which models each takes). */
  static async load(id: string, options: ModelOptions, head: string, kind: "encoder" | "vision" | "causal" | "auto"): Promise<PreTrainedModel> {
    const opts = withEndpoint(options);
    const config = await readConfig(id, opts);
    const type = String(config.model_type);
    if (kind === "auto") kind = ENCODERS.has(type) ? "encoder" : VISION.has(type) ? "vision" : "causal";
    const { progress_callback, dtype, quant, device, cpuThreads, revision, endpoint, worker } = opts;
    const onProgress = progress_callback ? toLoadOptions(id, { progress_callback }).onProgress : undefined;
    const dev = device === "webgpu" || device === "gpu" ? "gpu" : device === "wasm" || device === "cpu" ? "cpu" : "auto";
    const q = quant === "q4" || quant === "q8" || quant === "f32" ? quant : encoderQuant(dtype);
    let model: PreTrainedModel;
    if (kind === "encoder") {
      if (!ENCODERS.has(type)) throw new Error(`${id}: model_type "${type}" is not an encoder Kiln runs (${[...ENCODERS].join(", ")})`);
      model = new PreTrainedModel(config, await KilnEncoder.load(id, { quant: q, device: dev, cpuThreads, revision, endpoint, worker, onProgress }), head);
    } else if (kind === "vision") {
      if (!VISION.has(type)) throw new Error(`${id}: model_type "${type}" is not an image model AutoModel runs in Kiln (${[...VISION].join(", ")})`);
      if (head.startsWith("clip") && type !== "clip") throw new Error(`${id} is not a CLIP model`);
      if (type === "clip" && head === "base") head = "clip";
      model = new PreTrainedModel(config, await KilnVision.load(id, { quant: q, device: dev, cpuThreads, revision, endpoint, worker, onProgress }), head);
    } else {
      if (head !== "causal") throw new Error(`${id}: model_type "${type}" is a language model; load it with AutoModelForCausalLM`);
      const [m, generation] = await Promise.all([
        KilnModel.load(id, toLoadOptions(id, opts)),
        type === "gguf" ? undefined : fetchJSON<Config>(fileUrl(parseModelId(id, revision), "generation_config.json", endpoint ?? envEndpoint()), true),
      ]);
      model = new PreTrainedCausalLM(config, m, generation ?? {});
    }
    progress_callback?.({ status: "ready", task: head as never, model: id });
    return model;
  }
}

/** transformers.js's generate() options (the ones Kiln runs). */
export interface GenerateArgs {
  input_ids?: Tensor;
  inputs?: Tensor;
  attention_mask?: Tensor;
  max_new_tokens?: number;
  max_length?: number;
  do_sample?: boolean;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  streamer?: Streamer;
  generation_config?: Config;
  [key: string]: unknown;
}

/**
 * A causal language model: `generate({ ...inputs, max_new_tokens })` returns the sequences (prompt
 * then new tokens, the stop token included when one ended the reply) as int64 [batch, length],
 * shorter rows padded at the end. Defaults come from the repo's generation_config.json, as in
 * transformers.js (so 20 tokens in all, and at least one new one, unless `max_new_tokens` or
 * `max_length` says otherwise).
 */
export class PreTrainedCausalLM extends PreTrainedModel {
  constructor(
    config: Config,
    kiln: KilnModel,
    /** The repo's generation_config.json. */
    readonly generation_config: Config,
  ) {
    super(config, kiln, "causal");
  }

  async generate(args: GenerateArgs = {}): Promise<Tensor> {
    const { input_ids, inputs, attention_mask, streamer, generation_config, ...rest } = args;
    const g = { ...this.generation_config, ...(generation_config ?? {}), ...rest } as Config;
    const { rows, batch, length } = idRows({ input_ids: input_ids ?? inputs, attention_mask });
    const all = Array.from(((input_ids ?? inputs) as Tensor).data as ArrayLike<number | bigint>, Number);
    const eos = g.eos_token_id ?? this.config.eos_token_id;
    const pad = Number(g.pad_token_id ?? this.config.pad_token_id ?? (Array.isArray(eos) ? eos[0] : eos) ?? 0);
    const seqs: number[][] = [];
    for (let b = 0; b < batch; b++) {
      const maxNew = g.max_new_tokens !== undefined && g.max_new_tokens !== null ? Number(g.max_new_tokens) : Math.max(1, Number(g.max_length ?? 20) - length);
      const opts = toGenerateOptions({
        max_new_tokens: maxNew,
        do_sample: !!g.do_sample,
        temperature: g.temperature as number | undefined,
        top_k: g.top_k as number | undefined,
        top_p: g.top_p as number | undefined,
        repetition_penalty: g.repetition_penalty as number | undefined,
      });
      const r = maxNew > 0 ? await (this.kiln as KilnModel).generateIds(rows[b]!.ids, opts, (_id, text) => streamer?.put(text)) : { ids: [], stopId: undefined };
      seqs.push([...all.slice(b * length, (b + 1) * length), ...r.ids, ...(r.stopId !== undefined ? [r.stopId] : [])]);
    }
    streamer?.end();
    const longest = Math.max(...seqs.map((s) => s.length));
    const data = BigInt64Array.from(seqs.flatMap((s) => [...s, ...new Array<number>(longest - s.length).fill(pad)]).map(BigInt));
    return new Tensor("int64", data, [batch, longest]);
  }
}

type ModelClass = { from_pretrained(id: string, options?: ModelOptions): Promise<PreTrainedModel & ((inputs: Inputs) => Promise<Record<string, Tensor>>)> };
const autoClass = (head: string, kind: "encoder" | "vision" | "causal" | "auto"): ModelClass => ({
  from_pretrained: (id: string, options: ModelOptions = {}) => PreTrainedModel.load(id, options, head, kind) as never,
});

/** BERT / DistilBERT / RoBERTa: { last_hidden_state }; ViT / DeiT / DINOv2: { last_hidden_state, pooler_output? }; CLIP: CLIPModel's outputs. */
export const AutoModel = autoClass("base", "auto");
export const AutoModelForSequenceClassification = autoClass("sequence", "encoder");
export const AutoModelForTokenClassification = autoClass("token", "encoder");
export const AutoModelForQuestionAnswering = autoClass("qa", "encoder");
export const AutoModelForMaskedLM = autoClass("mlm", "encoder");
export const AutoModelForImageClassification = autoClass("classification", "vision");
export const CLIPModel = autoClass("clip", "vision");
export const CLIPTextModelWithProjection = autoClass("clip-text", "vision");
export const CLIPVisionModelWithProjection = autoClass("clip-image", "vision");
/** Llama, Mistral, Qwen2/3, SmolLM, Granite, Gemma 3 … (safetensors or GGUF): `generate()`. */
export const AutoModelForCausalLM = {
  from_pretrained: (id: string, options: ModelOptions = {}) => PreTrainedModel.load(id, options, "causal", "causal") as Promise<PreTrainedCausalLM>,
};

/** preprocessor_config.json as the processors read it. */
type ProcessorConfig = ImageProcessorConfig & {
  processor_class?: string;
  feature_size?: number;
  sampling_rate?: number;
  return_attention_mask?: boolean;
  padding_value?: number;
  [key: string]: unknown;
};

/** Audio feature extractors (some wav2vec2 repos do not name their type): a sampling rate and no image size. */
const isAudio = (c: ProcessorConfig) => /WhisperFeatureExtractor|Wav2Vec2FeatureExtractor/.test(String(c.feature_extractor_type ?? "")) || (c.sampling_rate !== undefined && c.size === undefined && c.crop_size === undefined);

/**
 * transformers.js's Processor / image processor / feature extractor, from the repo's
 * preprocessor_config.json: images to `{ pixel_values, original_sizes, reshaped_input_sizes }`
 * (transformers' resize, crop, rescale and normalize, with PIL's resampling); Whisper audio to
 * `{ input_features }` [1, mels, 3000]; wav2vec2 / HuBERT audio to `{ input_values }` (and
 * `attention_mask`). `tokenizer` is the repo's tokenizer, when it has one.
 */
export class Processor {
  tokenizer: PreTrainedTokenizer | undefined;
  constructor(readonly config: ProcessorConfig) {
    const call = (input: unknown, ...args: unknown[]) => this._call(input, ...args);
    Object.setPrototypeOf(call, Object.getPrototypeOf(this));
    return Object.assign(call, this) as unknown as Processor;
  }

  static async from_pretrained(id: string, options: ModelOptions = {}): Promise<Processor & ((input: unknown, ...args: unknown[]) => Promise<Record<string, unknown>>)> {
    const ref = parseModelId(id, options.revision);
    const ep = options.endpoint ?? envEndpoint();
    const files = await listRepoFiles(ref, ep);
    const has = (f: string) => !files || files.has(f);
    const [config, tokenizer] = await Promise.all([
      has("preprocessor_config.json") ? fetchJSON<ProcessorConfig>(fileUrl(ref, "preprocessor_config.json", ep), true) : undefined,
      has("tokenizer.json") || has("vocab.txt") ? PreTrainedTokenizer.from_pretrained(id, { revision: options.revision, endpoint: ep }).catch(() => undefined) : undefined,
    ]);
    if (!config && !tokenizer) throw new Error(`${id} has no preprocessor_config.json or tokenizer`);
    const p = new this(config ?? {}) as Processor & ((input: unknown, ...args: unknown[]) => Promise<Record<string, unknown>>);
    p.tokenizer = tokenizer;
    return p;
  }

  async _call(input: unknown, ...args: unknown[]): Promise<Record<string, unknown>> {
    const c = this.config;
    if (isAudio(c)) return this.audio(input as Float32Array);
    if (!c.size && !c.crop_size && this.tokenizer) return (this.tokenizer as unknown as (t: unknown, o?: unknown) => Record<string, unknown>)(input, args[0]);
    return this.images(input);
  }

  private async images(input: unknown): Promise<Record<string, unknown>> {
    const list = (Array.isArray(input) ? input : [input]) as (RawImage | ImageInput)[];
    const imgs: RgbImage[] = await Promise.all(list.map(async (x) => (x instanceof RawImage ? (x.channels < 3 ? x.rgb() : x) : readImage(x as ImageInput))));
    const outs = imgs.map((img) => preprocessImage(img, this.config));
    const { width: w, height: h } = outs[0]!;
    if (outs.some((o) => o.width !== w || o.height !== h)) throw new Error("the images came out at different sizes; process them one at a time");
    const data = new Float32Array(outs.length * 3 * h * w);
    outs.forEach((o, i) => data.set(o.data, i * 3 * h * w));
    return {
      pixel_values: new Tensor("float32", data, [outs.length, 3, h, w]),
      original_sizes: imgs.map((i) => [i.height, i.width]),
      reshaped_input_sizes: outs.map(() => [h, w]),
    };
  }

  private audio(samples: Float32Array): Record<string, unknown> {
    const c = this.config;
    if (/Whisper/.test(String(c.feature_extractor_type)) || (c.feature_size ?? 1) > 1) {
      const mels = Number(c.feature_size ?? 80);
      const tm = whisperFeatures(samples, { mels, sampleRate: c.sampling_rate, nFft: c.n_fft as number | undefined, hop: c.hop_length as number | undefined, chunkSeconds: c.chunk_length as number | undefined });
      const frames = tm.length / mels;
      const out = new Float32Array(tm.length);
      for (let t = 0; t < frames; t++) for (let m = 0; m < mels; m++) out[m * frames + t] = tm[t * mels + m]!;
      return { input_features: new Tensor("float32", out, [1, mels, frames]) };
    }
    const x = c.do_normalize ? normalizeAudio(samples) : samples;
    const result: Record<string, unknown> = { input_values: new Tensor("float32", x, [1, x.length]) };
    // transformers.js returns the mask whether or not the config asks for it.
    result.attention_mask = new Tensor("int64", new BigInt64Array(x.length).fill(1n), [1, x.length]);
    return result;
  }
}

/** transformers.js's AutoProcessor (also AutoImageProcessor and AutoFeatureExtractor): see `Processor`. */
export const AutoProcessor = { from_pretrained: (id: string, options?: ModelOptions) => Processor.from_pretrained(id, options) };
export const AutoImageProcessor = AutoProcessor;
export const AutoFeatureExtractor = AutoProcessor;
