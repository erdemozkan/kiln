// A transformers.js-shaped front end over `load` / `chat` / `complete`, so code written for
// transformers.js's text-generation pipeline runs on Kiln with little more than a new import.
import type { LoadProgress } from "@kiln/runtime";
import type { ChatMessage } from "@kiln/tokenizers";
import { KilnVision, KilnAudio, KilnEncoder, KilnModel, KilnSeq2Seq, type GenerateOptions, type LoadOptions, type TextStream } from "./client";
import { fetchJSON, fileUrl, parseModelId, readAudio, type AudioInput } from "@kiln/loader";
import type { VisionInfo, AudioInfo, EncoderInfo, EncoderTask, ModelInfo, Seq2SeqInfo, Seq2SeqTask } from "./protocol";

/** Task names, with transformers.js's aliases. */
export type PipelineTask =
  | "text-generation"
  | "feature-extraction"
  | "embeddings"
  | "text-classification"
  | "sentiment-analysis"
  | "token-classification"
  | "ner"
  | "question-answering"
  | "fill-mask"
  | "zero-shot-classification"
  | "summarization"
  | "translation"
  | "text2text-generation"
  | "automatic-speech-recognition"
  | "audio-classification"
  | "image-classification"
  | "image-feature-extraction"
  | "zero-shot-image-classification"
  | "object-detection"
  | "depth-estimation";

const SEQ2SEQ = new Set<string>(["summarization", "translation", "text2text-generation", "automatic-speech-recognition"]);

const ALIASES: Partial<Record<PipelineTask, PipelineTask>> = { embeddings: "feature-extraction", "sentiment-analysis": "text-classification", ner: "token-classification" };

/** Models used when `pipeline(task)` is given none (original Hub repos, not ONNX exports). */
export const DEFAULT_MODELS: Record<string, string> = {
  "text-generation": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "feature-extraction": "sentence-transformers/all-MiniLM-L6-v2",
  "text-classification": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "token-classification": "dslim/bert-base-NER",
  "question-answering": "distilbert/distilbert-base-cased-distilled-squad",
  "fill-mask": "google-bert/bert-base-uncased",
  "zero-shot-classification": "typeform/distilbert-base-uncased-mnli",
  summarization: "sshleifer/distilbart-cnn-6-6",
  translation: "Helsinki-NLP/opus-mt-en-de",
  "text2text-generation": "google/flan-t5-small",
  "automatic-speech-recognition": "openai/whisper-tiny.en",
  "audio-classification": "superb/wav2vec2-base-superb-ks",
  "image-classification": "google/vit-base-patch16-224",
  "image-feature-extraction": "google/vit-base-patch16-224-in21k",
  "zero-shot-image-classification": "openai/clip-vit-base-patch32",
  "object-detection": "facebook/detr-resnet-50",
  "depth-estimation": "depth-anything/Depth-Anything-V2-Small-hf",
};

const VISION = new Set<string>(["image-classification", "image-feature-extraction", "zero-shot-image-classification", "object-detection", "depth-estimation"]);

/** Speech encoders run on raw samples (CTC or a classifier) rather than as encoder-decoders. */
const AUDIO_ENCODERS = new Set(["wav2vec2", "hubert"]);

/** transformers.js's names, plus any of Kiln's own load options. */
export interface PipelineOptions extends Omit<LoadOptions, "device" | "onProgress" | "quant"> {
  /** "webgpu" (or "gpu"), "wasm" (or "cpu"), or "auto" (default): WebGPU where the browser has it, else the CPU. */
  device?: "auto" | "webgpu" | "gpu" | "wasm" | "cpu";
  /**
   * Weight format. Kiln quantizes at load, from the repo's safetensors or GGUF: 4-bit names ("q4",
   * "q4f16", "bnb4", "int4") load as q4, 8-bit ones ("q8", "int8", "uint8") as q8, "fp16" / "fp32" as
   * f32, "auto" (default) by model size.
   */
  dtype?: string;
  /** Kiln's own spelling of `dtype`; wins over it. */
  quant?: LoadOptions["quant"];
  progress_callback?: (info: ProgressInfo) => void;
}

/** Download progress in transformers.js's shape (`progress` in percent). */
export type ProgressInfo =
  | { status: "initiate" | "done"; name: string; file: string }
  | { status: "progress"; name: string; file: string; progress: number; loaded: number; total: number }
  | { status: "ready"; task: PipelineTask; model: string };

export interface TextGenerationOptions extends Omit<GenerateOptions, "maxTokens" | "temperature" | "topK" | "topP" | "repetitionPenalty"> {
  /** Tokens to generate at most (default: Kiln's, until the model ends its turn or the context fills). */
  max_new_tokens?: number;
  /** false (default): greedy. true: sample, with `temperature` (default 1), `top_k` (50) and `top_p` (1). */
  do_sample?: boolean;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  /** Prepend the prompt to `generated_text` for string inputs (default true, as transformers.js). */
  return_full_text?: boolean;
  /** Receives the text as it is generated (a `TextStreamer`, or anything with `put` / `end`). */
  streamer?: Streamer;
}

/** Text streamer interface: Kiln streams decoded text, not token ids. */
export interface Streamer {
  put(text: string): void;
  end(): void;
}

/** One result: the completion (string input) or the conversation with the reply appended (chat input). */
export type TextGenerationSingle = { generated_text: string | ChatMessage[] };
export type TextGenerationOutput = TextGenerationSingle[];

export interface TextStreamerOptions {
  /** transformers.js would print the prompt first unless this is set; Kiln streams only new text. */
  skip_prompt?: boolean;
  /** Special tokens are never streamed; accepted for compatibility. */
  skip_special_tokens?: boolean;
  /** Called with each piece of new text. */
  callback_function?: (text: string) => void;
}

/**
 * Streams generated text to `callback_function`, as transformers.js's TextStreamer does. The first
 * argument (a tokenizer there) is accepted and ignored: Kiln decodes in its worker.
 */
export class TextStreamer implements Streamer {
  private readonly callback: (text: string) => void;
  constructor(_tokenizer?: unknown, opts: TextStreamerOptions = {}) {
    this.callback = opts.callback_function ?? (() => {});
  }
  put(text: string): void {
    if (text) this.callback(text);
  }
  end(): void {}
}

type ChatInput = ChatMessage[];
type Input = string | ChatInput;

/** A loaded text-generation pipeline: call it like transformers.js's. */
export interface TextGenerationPipeline {
  (input: Input, options?: TextGenerationOptions): Promise<TextGenerationOutput>;
  (inputs: Input[], options?: TextGenerationOptions): Promise<TextGenerationOutput[]>;
  readonly task: PipelineTask;
  readonly model: KilnModel;
  /** Model details (family, layers, backend, context); Kiln's `ModelInfo`. */
  readonly info: ModelInfo;
  /** Placeholder for `new TextStreamer(generator.tokenizer, …)`: the tokenizer itself runs in Kiln's worker. */
  readonly tokenizer: object;
  /** Unloads the model and stops its worker. */
  dispose(): Promise<void>;
}

const DTYPES: Record<string, NonNullable<LoadOptions["quant"]>> = {
  auto: "auto",
  q4: "q4",
  q4f16: "q4",
  bnb4: "q4",
  int4: "q4",
  q8: "q8",
  int8: "q8",
  uint8: "q8",
  fp16: "f32",
  fp32: "f32",
  f32: "f32",
};

/** Maps transformers.js load options to Kiln's. */
export function toLoadOptions(model: string, opts: PipelineOptions = {}): LoadOptions {
  const { device, dtype, quant, progress_callback, ...rest } = opts;
  let q = quant;
  if (q === undefined && dtype !== undefined) {
    q = DTYPES[dtype];
    if (!q) throw new Error(`Kiln: unsupported dtype "${dtype}" (use ${Object.keys(DTYPES).join(", ")})`);
  }
  const dev = device === "webgpu" || device === "gpu" ? "gpu" : device === "wasm" || device === "cpu" ? "cpu" : "auto";
  const out: LoadOptions = { ...rest, device: dev };
  if (q) out.quant = q;
  if (progress_callback) {
    const seen = new Set<string>();
    out.onProgress = (p: LoadProgress) => {
      if (!seen.has(p.file)) {
        seen.add(p.file);
        progress_callback({ status: "initiate", name: model, file: p.file });
      }
      const progress = p.total ? (p.loaded / p.total) * 100 : 0;
      progress_callback({ status: "progress", name: model, file: p.file, progress, loaded: p.loaded, total: p.total });
      if (p.total && p.loaded >= p.total) progress_callback({ status: "done", name: model, file: p.file });
    };
  }
  return out;
}

/** Maps transformers.js generation options to Kiln's. */
export function toGenerateOptions(opts: TextGenerationOptions = {}): GenerateOptions {
  const { max_new_tokens, do_sample, temperature, top_k, top_p, repetition_penalty, return_full_text: _r, streamer: _s, ...rest } = opts;
  const out: GenerateOptions = { ...rest };
  if (max_new_tokens !== undefined) out.maxTokens = max_new_tokens;
  if (do_sample) {
    out.temperature = temperature ?? 1;
    out.topK = top_k ?? 50;
    out.topP = top_p ?? 1;
  } else out.temperature = 0;
  if (repetition_penalty !== undefined) out.repetitionPenalty = repetition_penalty;
  return out;
}

const isChat = (x: unknown): x is ChatInput =>
  Array.isArray(x) && x.length > 0 && typeof x[0] === "object" && x[0] !== null && "role" in x[0];

/** Generates for one input (string completion or chat), feeding `streamer`. */
async function generateOne(
  model: Pick<KilnModel, "chat" | "complete">,
  input: Input,
  opts: TextGenerationOptions,
): Promise<TextGenerationOutput> {
  const gen = toGenerateOptions(opts);
  const stream: TextStream = typeof input === "string" ? model.complete(input, gen) : model.chat(input, gen);
  let text = "";
  try {
    for await (const piece of stream) {
      text += piece;
      opts.streamer?.put(piece);
    }
  } finally {
    opts.streamer?.end();
  }
  if (typeof input === "string") return [{ generated_text: opts.return_full_text === false ? text : input + text }];
  return [{ generated_text: [...input, { role: "assistant", content: text }] }];
}

/** Runs a pipeline call on a loaded model: one input, or an array of inputs one after another. */
export async function runTextGeneration(
  model: Pick<KilnModel, "chat" | "complete">,
  input: Input | Input[],
  opts: TextGenerationOptions = {},
): Promise<TextGenerationOutput | TextGenerationOutput[]> {
  if (typeof input === "string" || isChat(input)) return generateOne(model, input, opts);
  if (!Array.isArray(input)) throw new Error("Kiln: text-generation takes a string, a chat (messages), or an array of them");
  const out: TextGenerationOutput[] = [];
  for (const one of input) out.push(await generateOne(model, one as Input, opts));
  return out;
}

/** A float tensor, as transformers.js returns embeddings: row-major `data` of shape `dims`. */
export class Tensor {
  constructor(
    readonly type: "float32",
    readonly data: Float32Array,
    readonly dims: number[],
  ) {}

  get size(): number {
    return this.data.length;
  }

  /** Nested arrays of numbers, outermost dimension first. */
  tolist(): unknown[] {
    const rec = (offset: number, d: number): unknown[] => {
      const n = this.dims[d]!;
      if (d === this.dims.length - 1) return Array.from(this.data.subarray(offset, offset + n));
      const stride = this.dims.slice(d + 1).reduce((a, b) => a * b, 1);
      return Array.from({ length: n }, (_, i) => rec(offset + i * stride, d + 1));
    };
    return rec(0, 0);
  }
}

/** Cosine similarity of two vectors (transformers.js's `cos_sim`). */
export function cos_sim(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** What every encoder pipeline carries besides being callable. */
interface EncoderPipelineBase {
  readonly task: PipelineTask;
  readonly model: KilnEncoder;
  readonly info: EncoderInfo;
  /** Placeholder, as for text generation: the tokenizer runs in Kiln's worker. */
  readonly tokenizer: object;
  dispose(): Promise<void>;
}

export interface Label {
  label: string;
  score: number;
}
export interface TokenEntity {
  entity: string;
  score: number;
  index: number;
  word: string;
}
export interface EntityGroup {
  entity_group: string;
  score: number;
  word: string;
}
export interface Answer {
  answer: string;
  score: number;
}
export interface MaskFill {
  score: number;
  token: number;
  token_str: string;
  sequence: string;
}
export interface ZeroShotResult {
  sequence: string;
  labels: string[];
  scores: number[];
}

export interface FeatureExtractionPipeline extends EncoderPipelineBase {
  /** Last hidden states [batch, tokens, hidden], or [batch, hidden] with pooling "mean" or "cls". */
  (texts: string | string[], options?: { pooling?: "none" | "mean" | "cls" | "first_token"; normalize?: boolean }): Promise<Tensor>;
}
export interface TextClassificationPipeline extends EncoderPipelineBase {
  (texts: string | string[], options?: { top_k?: number | null }): Promise<Label[] | Label[][]>;
}
export interface TokenClassificationPipeline extends EncoderPipelineBase {
  (texts: string | string[], options?: { ignore_labels?: string[]; aggregation_strategy?: "none" | "simple" }): Promise<(TokenEntity | EntityGroup)[] | (TokenEntity | EntityGroup)[][]>;
}
export interface QuestionAnsweringPipeline extends EncoderPipelineBase {
  (question: string | string[], context: string | string[], options?: { top_k?: number }): Promise<Answer | Answer[] | (Answer | Answer[])[]>;
}
export interface FillMaskPipeline extends EncoderPipelineBase {
  (texts: string | string[], options?: { top_k?: number }): Promise<MaskFill[] | MaskFill[][]>;
}
export interface ZeroShotClassificationPipeline extends EncoderPipelineBase {
  (texts: string | string[], candidate_labels: string | string[], options?: { hypothesis_template?: string; multi_label?: boolean }): Promise<ZeroShotResult | ZeroShotResult[]>;
}

/** transformers' generation settings for summarization, translation and text2text (greedy; `num_beams` is ignored, as in transformers.js). */
export interface Text2TextOptions {
  /** Tokens to generate at most (default 256). */
  max_new_tokens?: number;
  min_length?: number;
  no_repeat_ngram_size?: number;
  repetition_penalty?: number;
  forced_bos_token_id?: number;
  forced_eos_token_id?: number;
  /** Multilingual translation models (NLLB): source and target language codes, e.g. "eng_Latn", "fra_Latn". */
  src_lang?: string;
  tgt_lang?: string;
}

interface Seq2SeqPipelineBase {
  readonly task: PipelineTask;
  readonly model: KilnSeq2Seq;
  readonly info: Seq2SeqInfo;
  readonly tokenizer: object;
  dispose(): Promise<void>;
}
export interface SummarizationPipeline extends Seq2SeqPipelineBase {
  (texts: string | string[], options?: Text2TextOptions): Promise<{ summary_text: string }[]>;
}
export interface TranslationPipeline extends Seq2SeqPipelineBase {
  (texts: string | string[], options?: Text2TextOptions): Promise<{ translation_text: string }[]>;
}
/** Speech recognition options (Whisper; transformers.js's names). */
export interface SpeechRecognitionOptions {
  /** Spoken language for multilingual models: a code ("fr") or English name ("french"); default English. */
  language?: string;
  /** "transcribe" (default) or "translate" (into English). */
  task?: "transcribe" | "translate";
  max_new_tokens?: number;
  /**
   * Window length in seconds for chunked transcription: overlapping windows, their texts merged where
   * they agree. Default none, except that audio past 30 s is chunked in 30 s windows.
   */
  chunk_length_s?: number;
  /** Overlap on each side of a window in seconds (default chunk_length_s / 6). */
  stride_length_s?: number;
  /**
   * true: also return `chunks`, segments with [start, end] times in seconds. "word": `chunks` are words
   * with [start, end] times, aligned through the model's cross-attention (Whisper only).
   */
  return_timestamps?: boolean | "word";
}
/** A transcribed segment (with `return_timestamps: true`), or a word (with `return_timestamps: "word"`). */
export interface SpeechChunk {
  timestamp: [number | null, number | null];
  text: string;
}
export interface SpeechRecognitionPipeline {
  readonly task: PipelineTask;
  /** Whisper runs as an encoder-decoder, wav2vec2 / HuBERT (CTC) as a speech encoder. */
  readonly model: KilnSeq2Seq | KilnAudio;
  readonly info: Seq2SeqInfo | AudioInfo;
  readonly tokenizer: object;
  dispose(): Promise<void>;
  /** 16 kHz samples, a URL, a Blob / File, or encoded audio bytes (WAV anywhere; other formats on a page's main thread). */
  (audio: AudioInput, options?: SpeechRecognitionOptions): Promise<{ text: string; chunks?: SpeechChunk[] }>;
  (audio: AudioInput[], options?: SpeechRecognitionOptions): Promise<{ text: string; chunks?: SpeechChunk[] }[]>;
}
export interface AudioClassificationOptions {
  /** Labels returned, best first (default 5). */
  top_k?: number;
}
export interface AudioClassificationPipeline {
  readonly task: PipelineTask;
  readonly model: KilnAudio;
  readonly info: AudioInfo;
  dispose(): Promise<void>;
  (audio: AudioInput, options?: AudioClassificationOptions): Promise<{ label: string; score: number }[]>;
  (audio: AudioInput[], options?: AudioClassificationOptions): Promise<{ label: string; score: number }[][]>;
}
/**
 * An image: a URL (relative to the page), a Blob / File, encoded bytes, an ImageBitmap / ImageData,
 * an <img> / <canvas> element, or RGB(A) pixels { width, height, data } (transformers.js's RawImage
 * with 3 or 4 channels fits).
 */
export type ImageInput = string | URL | Blob | ArrayBuffer | Uint8Array | ImageBitmap | ImageData | CanvasImageSource | { width: number; height: number; data: Uint8Array | Uint8ClampedArray };
export interface ImageClassificationOptions {
  /** Labels returned, best first (default 5; 0 or negative: all). */
  top_k?: number;
}
export interface ImageClassificationPipeline {
  readonly task: PipelineTask;
  readonly model: KilnVision;
  readonly info: VisionInfo;
  dispose(): Promise<void>;
  (image: ImageInput, options?: ImageClassificationOptions): Promise<{ label: string; score: number }[]>;
  (images: ImageInput[], options?: ImageClassificationOptions): Promise<{ label: string; score: number }[][]>;
}
export interface ImageFeatureExtractionOptions {
  /** The pooled output [1, hidden] instead of the last hidden state [1, tokens, hidden]. */
  pool?: boolean;
}
export interface ImageFeatureExtractionPipeline {
  readonly task: PipelineTask;
  readonly model: KilnVision;
  readonly info: VisionInfo;
  dispose(): Promise<void>;
  (image: ImageInput, options?: ImageFeatureExtractionOptions): Promise<Tensor>;
  (images: ImageInput[], options?: ImageFeatureExtractionOptions): Promise<Tensor[]>;
}

export interface ZeroShotImageClassificationOptions {
  /** Turns each label into the text the model reads (default "This is a photo of {}"). */
  hypothesis_template?: string;
}
export interface ZeroShotImageClassificationPipeline {
  readonly task: PipelineTask;
  readonly model: KilnVision;
  readonly info: VisionInfo;
  dispose(): Promise<void>;
  (image: ImageInput, candidate_labels: string[], options?: ZeroShotImageClassificationOptions): Promise<{ score: number; label: string }[]>;
  (images: ImageInput[], candidate_labels: string[], options?: ZeroShotImageClassificationOptions): Promise<{ score: number; label: string }[][]>;
}

export interface ObjectDetectionOptions {
  /** Least score a detection needs (default 0.9). */
  threshold?: number;
  /** Boxes as fractions of the image instead of whole pixels (default false). */
  percentage?: boolean;
}
export interface ObjectDetectionResult {
  score: number;
  label: string;
  box: { xmin: number; ymin: number; xmax: number; ymax: number };
}
export interface ObjectDetectionPipeline {
  readonly task: PipelineTask;
  readonly model: KilnVision;
  readonly info: VisionInfo;
  dispose(): Promise<void>;
  (image: ImageInput, options?: ObjectDetectionOptions): Promise<ObjectDetectionResult[]>;
  (images: ImageInput[], options?: ObjectDetectionOptions): Promise<ObjectDetectionResult[][]>;
}

/** A depth map: `predicted_depth` [height, width] at the original image size, and `depth`, the map as 0…255 grey bytes. */
export interface DepthEstimationResult {
  predicted_depth: Tensor;
  depth: { width: number; height: number; channels: 1; data: Uint8Array };
}
export interface DepthEstimationPipeline {
  readonly task: PipelineTask;
  readonly model: KilnVision;
  readonly info: VisionInfo;
  dispose(): Promise<void>;
  (image: ImageInput): Promise<DepthEstimationResult>;
  (images: ImageInput[]): Promise<DepthEstimationResult[]>;
}

/** An image in a form the worker can receive and decode: relative URLs resolved, DOM images drawn to bitmaps. */
async function toWorkerImage(x: ImageInput): Promise<unknown> {
  if (typeof x === "string") return typeof location !== "undefined" ? new URL(x, location.href).href : x;
  if (x instanceof URL) return x.href;
  if (x instanceof Blob || x instanceof ArrayBuffer || x instanceof Uint8Array) return x;
  if (typeof ImageBitmap !== "undefined" && x instanceof ImageBitmap) return x;
  if (typeof ImageData !== "undefined" && x instanceof ImageData) return x;
  if ("data" in x && "width" in x) {
    const d = x.data as Uint8Array;
    const ch = d.length / (x.width * x.height);
    if (ch !== 3 && ch !== 4) throw new Error(`Kiln: images need 3 or 4 channels, got ${ch}`);
    return { width: x.width, height: x.height, data: d };
  }
  return createImageBitmap(x as ImageBitmapSource, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
}

export interface Text2TextGenerationPipeline extends Seq2SeqPipelineBase {
  (texts: string | string[], options?: Text2TextOptions): Promise<{ generated_text: string }[]>;
}

/** Encoder dtypes: 8-bit names load as q8, 4-bit as q4, float and "auto" as f32 (encoders are small). */
function encoderQuant(dtype: string | undefined): "f32" | "q8" | "q4" {
  if (dtype === undefined || dtype === "auto" || dtype === "fp32" || dtype === "fp16" || dtype === "f32") return "f32";
  const q = DTYPES[dtype];
  if (!q) throw new Error(`Kiln: unsupported dtype "${dtype}" (use ${Object.keys(DTYPES).join(", ")})`);
  return q === "q4" ? "q4" : "q8";
}

/**
 * transformers.js-style entry point: "text-generation" for decoder LLMs (safetensors or GGUF), and for
 * encoder models (BERT, DistilBERT, RoBERTa from safetensors) "feature-extraction" ("embeddings"),
 * "text-classification" ("sentiment-analysis"), "token-classification" ("ner"),
 * "question-answering", "fill-mask" and "zero-shot-classification". Without a model id, the task's
 * entry in `DEFAULT_MODELS` loads.
 *
 *   const generator = await pipeline("text-generation", "HuggingFaceTB/SmolLM2-135M-Instruct", { dtype: "q4" });
 *   const extractor = await pipeline("feature-extraction", "sentence-transformers/all-MiniLM-L6-v2");
 *   const output = await extractor(["Hello", "World"], { pooling: "mean", normalize: true });
 */
export async function pipeline(task: "text-generation", model?: string, options?: PipelineOptions): Promise<TextGenerationPipeline>;
export async function pipeline(task: "feature-extraction" | "embeddings", model?: string, options?: PipelineOptions): Promise<FeatureExtractionPipeline>;
export async function pipeline(task: "text-classification" | "sentiment-analysis", model?: string, options?: PipelineOptions): Promise<TextClassificationPipeline>;
export async function pipeline(task: "token-classification" | "ner", model?: string, options?: PipelineOptions): Promise<TokenClassificationPipeline>;
export async function pipeline(task: "question-answering", model?: string, options?: PipelineOptions): Promise<QuestionAnsweringPipeline>;
export async function pipeline(task: "fill-mask", model?: string, options?: PipelineOptions): Promise<FillMaskPipeline>;
export async function pipeline(task: "zero-shot-classification", model?: string, options?: PipelineOptions): Promise<ZeroShotClassificationPipeline>;
export async function pipeline(task: "summarization", model?: string, options?: PipelineOptions): Promise<SummarizationPipeline>;
export async function pipeline(task: "translation", model?: string, options?: PipelineOptions): Promise<TranslationPipeline>;
export async function pipeline(task: "text2text-generation", model?: string, options?: PipelineOptions): Promise<Text2TextGenerationPipeline>;
export async function pipeline(task: "automatic-speech-recognition", model?: string, options?: PipelineOptions): Promise<SpeechRecognitionPipeline>;
export async function pipeline(task: "audio-classification", model?: string, options?: PipelineOptions): Promise<AudioClassificationPipeline>;
export async function pipeline(task: "image-classification", model?: string, options?: PipelineOptions): Promise<ImageClassificationPipeline>;
export async function pipeline(task: "image-feature-extraction", model?: string, options?: PipelineOptions): Promise<ImageFeatureExtractionPipeline>;
export async function pipeline(task: "zero-shot-image-classification", model?: string, options?: PipelineOptions): Promise<ZeroShotImageClassificationPipeline>;
export async function pipeline(task: "object-detection", model?: string, options?: PipelineOptions): Promise<ObjectDetectionPipeline>;
export async function pipeline(task: "depth-estimation", model?: string, options?: PipelineOptions): Promise<DepthEstimationPipeline>;
export async function pipeline(task: PipelineTask, model?: string, options: PipelineOptions = {}): Promise<unknown> {
  const t = ALIASES[task] ?? task;
  const id = model ?? DEFAULT_MODELS[t];
  if (!id) {
    throw new Error(`Kiln: unsupported pipeline task "${task}" (Kiln runs ${Object.keys(DEFAULT_MODELS).map((k) => `"${k}"`).join(", ")})`);
  }
  if (t === "text-generation") {
    const m = await KilnModel.load(id, toLoadOptions(id, options));
    options.progress_callback?.({ status: "ready", task, model: id });
    const call = (input: Input | Input[], opts?: TextGenerationOptions) => runTextGeneration(m, input, opts);
    return Object.assign(call, { task, model: m, info: m.info, tokenizer: {}, dispose: () => m.unload() });
  }
  const { progress_callback, dtype, quant, device, cpuThreads, revision, endpoint, worker } = options;
  const progress = progress_callback ? toLoadOptions(id, { progress_callback }).onProgress : undefined;
  const dev = device === "webgpu" || device === "gpu" ? "gpu" : device === "wasm" || device === "cpu" ? "cpu" : "auto";
  const q = quant === "q4" || quant === "q8" || quant === "f32" ? quant : encoderQuant(dtype);
  if (VISION.has(t)) {
    const m = await KilnVision.load(id, { quant: q, device: dev, cpuThreads, revision, endpoint, worker, onProgress: progress });
    progress_callback?.({ status: "ready", task, model: id });
    const zeroShot = t === "zero-shot-image-classification";
    const one = async (x: ImageInput, a?: unknown, b?: object) => {
      const img = await toWorkerImage(x);
      if (zeroShot) return m.run<unknown>("zero-shot-image-classification", img, a, { ...b });
      const r = await m.run<unknown>(t as "image-classification" | "image-feature-extraction" | "object-detection" | "depth-estimation", img, { ...(a as object) });
      if (t === "image-feature-extraction") {
        const f = r as { data: Float32Array; dims: number[] };
        return new Tensor("float32", f.data, f.dims);
      }
      if (t === "depth-estimation") {
        const d = r as { predicted_depth: { data: Float32Array; dims: number[] }; depth: DepthEstimationResult["depth"] };
        return { predicted_depth: new Tensor("float32", d.predicted_depth.data, d.predicted_depth.dims), depth: d.depth };
      }
      return r;
    };
    const call = async (images: ImageInput | ImageInput[], a?: unknown, b?: object) => {
      if (!Array.isArray(images)) return one(images, a, b);
      const out: unknown[] = [];
      for (const x of images) out.push(await one(x, a, b));
      return out;
    };
    return Object.assign(call, { task, model: m, info: m.info, tokenizer: {}, dispose: () => m.unload() });
  }
  // Whisper or a CTC model? The config says.
  const speechEncoder =
    t === "audio-classification" ||
    (t === "automatic-speech-recognition" &&
      AUDIO_ENCODERS.has(String((await fetchJSON<Record<string, unknown>>(fileUrl(parseModelId(id, revision), "config.json", endpoint)))?.model_type)));
  if (speechEncoder) {
    const m = await KilnAudio.load(id, { quant: q, device: dev, cpuThreads, revision, endpoint, worker, onProgress: progress });
    progress_callback?.({ status: "ready", task, model: id });
    const audioTask = t as "automatic-speech-recognition" | "audio-classification";
    // Decoded here: the Web Audio API is on the main thread, not in the worker.
    const one = async (x: AudioInput, opts?: object) => m.run(audioTask, await readAudio(x, 16000), { ...opts });
    const call = async (audio: AudioInput | AudioInput[], opts?: object) => {
      if (!Array.isArray(audio)) return one(audio, opts);
      const out: unknown[] = [];
      for (const x of audio) out.push(await one(x, opts));
      return out;
    };
    return Object.assign(call, { task, model: m, info: m.info, tokenizer: {}, dispose: () => m.unload() });
  }
  if (SEQ2SEQ.has(t)) {
    const m = await KilnSeq2Seq.load(id, { quant: q, device: dev, cpuThreads, revision, endpoint, worker, onProgress: progress });
    progress_callback?.({ status: "ready", task, model: id });
    const call =
      t === "automatic-speech-recognition"
        ? async (audio: AudioInput | AudioInput[], opts?: SpeechRecognitionOptions) => {
            // Decoded here: the Web Audio API is on the main thread, not in the worker.
            const one = async (x: AudioInput) => (await m.run("automatic-speech-recognition", await readAudio(x, 16000), { ...opts })) as unknown as { text: string };
            if (Array.isArray(audio)) {
              const out: { text: string }[] = [];
              for (const x of audio) out.push(await one(x));
              return out;
            }
            return one(audio);
          }
        : (texts: string | string[], opts?: Text2TextOptions) => m.run(t as Seq2SeqTask, texts, { ...opts });
    return Object.assign(call, { task, model: m, info: m.info, tokenizer: {}, dispose: () => m.unload() });
  }
  const m = await KilnEncoder.load(id, { quant: q, device: dev, cpuThreads, revision, endpoint, worker, onProgress: progress });
  progress_callback?.({ status: "ready", task, model: id });
  const run = (...args: unknown[]) => m.run(t as EncoderTask, ...args);
  const call =
    t === "feature-extraction"
      ? async (...args: unknown[]) => {
          const r = (await run(...args)) as { data: Float32Array; dims: number[] };
          return new Tensor("float32", r.data, r.dims);
        }
      : run;
  return Object.assign(call, { task, model: m, info: m.info, tokenizer: {}, dispose: () => m.unload() });
}
