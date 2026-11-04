import type { GenerateOptions, GenerateStats, LoadProgress, Quant } from "@kiln/runtime";
import type { ChatMessage } from "@kiln/tokenizers";

export type WireGenerateOptions = Omit<GenerateOptions, "signal">;

export interface WireLoadOptions {
  quant?: Quant;
  maxSeq?: number;
  revision?: string;
  endpoint?: string;
  /**
   * Keep the last prompt's K/V in OPFS and restore it on the next load (default true), so a page
   * reload does not prefill a shared prefix such as the system prompt again. A few MB per model.
   */
  promptCache?: boolean;
}

export interface ModelInfo {
  model: string;
  family: string;
  layers: number;
  hiddenSize: number;
  vocabSize: number;
  maxSeq: number;
  chatFormat: string | undefined;
  backendFeatures: string[];
  /** Tokens of K/V restored from the prompt cache at load (0 if none). */
  restoredTokens: number;
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
