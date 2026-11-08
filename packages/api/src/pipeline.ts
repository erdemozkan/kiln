// A transformers.js-shaped front end over `load` / `chat` / `complete`, so code written for
// transformers.js's text-generation pipeline runs on Kiln with little more than a new import.
import type { LoadProgress } from "@kiln/runtime";
import type { ChatMessage } from "@kiln/tokenizers";
import { KilnModel, type GenerateOptions, type LoadOptions, type TextStream } from "./client";
import type { ModelInfo } from "./protocol";

export type PipelineTask = "text-generation";

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

/**
 * transformers.js-style entry point. Only "text-generation" (decoder LLMs from safetensors or GGUF):
 *
 *   const generator = await pipeline("text-generation", "HuggingFaceTB/SmolLM2-135M-Instruct", { dtype: "q4" });
 *   const [out] = await generator([{ role: "user", content: "Hi!" }], { max_new_tokens: 64 });
 *   out.generated_text.at(-1).content;
 */
export async function pipeline(task: PipelineTask, model: string, options: PipelineOptions = {}): Promise<TextGenerationPipeline> {
  if (task !== "text-generation") throw new Error(`Kiln: unsupported pipeline task "${task}" (Kiln runs "text-generation")`);
  const m = await KilnModel.load(model, toLoadOptions(model, options));
  options.progress_callback?.({ status: "ready", task, model });
  const call = (input: Input | Input[], opts?: TextGenerationOptions) => runTextGeneration(m, input, opts);
  return Object.assign(call, {
    task,
    model: m,
    info: m.info,
    tokenizer: {},
    dispose: () => m.unload(),
  }) as unknown as TextGenerationPipeline;
}
