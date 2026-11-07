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

export type Request =
  | { id: number; type: "load"; model: string; opts: WireLoadOptions }
  | { id: number; type: "chat"; messages: ChatMessage[]; opts: WireGenerateOptions }
  | { id: number; type: "complete"; prompt: string; opts: WireGenerateOptions }
  | { id: number; type: "abort"; target: number }
  | { id: number; type: "unload" };

export type Response =
  | { id: number; type: "progress"; progress: LoadProgress }
  | { id: number; type: "loaded"; info: ModelInfo }
  | { id: number; type: "delta"; text: string }
  | { id: number; type: "done"; stats: GenerateStats | undefined }
  | { id: number; type: "error"; message: string };
