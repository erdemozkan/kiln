import type { GenerateOptions, GenerateStats, LoadProgress, Quant } from "@kiln/runtime";
import type { ChatMessage } from "@kiln/tokenizers";

export type WireGenerateOptions = Omit<GenerateOptions, "signal">;

export interface WireLoadOptions {
  quant?: Quant;
  maxSeq?: number;
  revision?: string;
  endpoint?: string;
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
