import type { Profile } from "@kiln/core";
import type { GenerateOptions, GenerateStats, LoadProgress, Quant } from "@kiln/runtime";
import type { ChatMessage } from "@kiln/tokenizers";

export type WireGenerateOptions = Omit<GenerateOptions, "signal">;

export interface WireLoadOptions {
  quant?: Quant;
  /** Context length in tokens (default: the model's, up to 8192 tokens and 512 MiB of KV cache). */
  maxSeq?: number;
  revision?: string;
  endpoint?: string;
  /**
   * Keep the last prompt's K/V in OPFS and restore it on the next load (default true), so a page
   * reload does not prefill a shared prefix such as the system prompt again. A few MB per model.
   */
  promptCache?: boolean;
  /**
   * Gemma 3's vision tower, as llama.cpp's mmproj GGUF: a file in the model's repo ("mmproj-F16.gguf")
   * or a full id. Messages may then hold `{ type: "image", image }` parts (Blob, ImageBitmap, ImageData).
   */
  mmproj?: string;
  /** Images whose features stay on the GPU at once, with `mmproj` (default 4). */
  imageSlots?: number;
  /** "auto" (default): WebGPU, or the CPU when the browser has none; "gpu" or "cpu" to choose. */
  device?: "auto" | "gpu" | "cpu";
  /** Threads for CPU matmuls (default hardwareConcurrency, at most 4; more than one needs a cross-origin isolated page). */
  cpuThreads?: number;
  /** Longest GPU work in one submit, in estimated ms (default 50). */
  maxSubmitMs?: number;
  /**
   * GPU bytes for the K/V of earlier prompts (other conversations, other system prompts), so going
   * back to one does not prefill it again. Default 256 MiB; 0 keeps only the live cache.
   */
  prefixCacheBytes?: number;
}

export interface ModelInfo {
  model: string;
  family: string;
  layers: number;
  hiddenSize: number;
  vocabSize: number;
  /** Where the model runs: "webgpu", or "cpu" (the fallback without WebGPU, or `device: "cpu"`). */
  backend: string;
  /** On the CPU backend, the threads large matmuls are split across (1 without cross-origin isolation). */
  cpuThreads?: number;
  /** True when chat messages may hold images (Gemma 3 loaded with `mmproj`). */
  vision: boolean;
  /** Context length this load allocated, in tokens. */
  maxSeq: number;
  /** Longest context the model was trained for, if its files say; pass a larger `maxSeq` to use more. */
  contextLength?: number;
  chatFormat: string | undefined;
  backendFeatures: string[];
  /** Tokens of K/V restored from the prompt cache at load (0 if none). */
  restoredTokens: number;
  /** Whether the browser agreed to keep cached weights (see `LoadOptions.persist`); set by the client once the browser answers. */
  persistent?: boolean;
}

/** Load options of an encoder model (feature extraction, classification, NER, QA, fill-mask, zero-shot). */
export interface WireEncoderOptions {
  /** Matrix storage: "f32" (default), "q8" or "q4". */
  quant?: "f32" | "q8" | "q4";
  device?: "auto" | "gpu" | "cpu";
  cpuThreads?: number;
  revision?: string;
  endpoint?: string;
}

export type EncoderTask = "feature-extraction" | "text-classification" | "token-classification" | "question-answering" | "fill-mask" | "zero-shot-classification";
export type Seq2SeqTask = "summarization" | "translation" | "text2text-generation" | "automatic-speech-recognition";

/** Load options of an encoder-decoder model (summarization, translation, text2text). */
export interface WireSeq2SeqOptions extends WireEncoderOptions {
  /** Longest output in tokens (default the model's position limit). */
  maxTarget?: number;
}

export interface Seq2SeqInfo {
  model: string;
  /** "bart", "marian", "m2m100", "t5" or "whisper". */
  family: string;
  encoderLayers: number;
  decoderLayers: number;
  dModel: number;
  vocabSize: number;
  backend: string;
  cpuThreads?: number;
}

/** Tasks of speech encoders on raw audio (wav2vec2, HuBERT). */
export type AudioTask = "automatic-speech-recognition" | "audio-classification" | "text-to-audio";

/** Load options of a speech model; `vocoder` picks SpeechT5's vocoder repo (default microsoft/speecht5_hifigan). */
export interface WireAudioOptions extends WireEncoderOptions {
  vocoder?: string;
}

export interface AudioInfo {
  model: string;
  /** "wav2vec2", "hubert", "vits" or "speecht5". */
  family: string;
  /** "ctc" (speech recognition), "classification", or "tts" (text-to-speech). */
  head: "ctc" | "classification" | "tts";
  layers: number;
  hiddenSize: number;
  backend: string;
  cpuThreads?: number;
  labels?: Record<number, string>;
}

/** Tasks of image models (ViT, DeiT, CLIP). */
export type VisionTask = "image-classification" | "image-feature-extraction" | "zero-shot-image-classification" | "object-detection" | "depth-estimation" | "image-segmentation" | "background-removal" | "image-to-image" | "image-to-text";

export interface VisionInfo {
  model: string;
  /** "vit", "deit" or "clip". */
  family: string;
  layers: number;
  hiddenSize: number;
  imageSize: number;
  backend: string;
  cpuThreads?: number;
  labels?: Record<number, string>;
}

export interface EncoderInfo {
  model: string;
  /** "bert", "distilbert" or "roberta". */
  family: string;
  layers: number;
  hiddenSize: number;
  vocabSize: number;
  /** Longest input in tokens; longer inputs are truncated. */
  maxLength: number;
  backend: string;
  cpuThreads?: number;
  /** Class labels of a fine-tuned head, by id. */
  labels?: Record<number, string>;
}

export type Request =
  | { id: number; type: "load"; model: string; opts: WireLoadOptions }
  | { id: number; type: "loadEncoder"; model: string; opts: WireEncoderOptions }
  | { id: number; type: "loadSeq2Seq"; model: string; opts: WireSeq2SeqOptions }
  | { id: number; type: "loadAudio"; model: string; opts: WireAudioOptions }
  | { id: number; type: "task"; task: EncoderTask | Seq2SeqTask; args: unknown[] }
  | { id: number; type: "audioTask"; task: AudioTask; args: unknown[] }
  | { id: number; type: "loadVision"; model: string; opts: WireEncoderOptions }
  | { id: number; type: "visionTask"; task: VisionTask; args: unknown[] }
  | { id: number; type: "chat"; messages: ChatMessage[]; opts: WireGenerateOptions }
  | { id: number; type: "complete"; prompt: string; opts: WireGenerateOptions }
  | { id: number; type: "abort"; target: number }
  | { id: number; type: "profile"; on: boolean }
  | { id: number; type: "unload" };

export type Response =
  | { id: number; type: "progress"; progress: LoadProgress }
  | { id: number; type: "loaded"; info: ModelInfo }
  | { id: number; type: "encoderLoaded"; info: EncoderInfo }
  | { id: number; type: "seq2seqLoaded"; info: Seq2SeqInfo }
  | { id: number; type: "audioLoaded"; info: AudioInfo }
  | { id: number; type: "visionLoaded"; info: VisionInfo }
  | { id: number; type: "result"; result: unknown }
  | { id: number; type: "delta"; text: string }
  | { id: number; type: "done"; stats: GenerateStats | undefined }
  | { id: number; type: "profiled"; profile: Profile | undefined }
  | { id: number; type: "error"; message: string };
