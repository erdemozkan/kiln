import { Template, TemplateError } from "./jinja";
import type { Tokenizer } from "./tokenizer";

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  /** Calls the assistant made in this turn (as `parseToolCalls` returns them). */
  tool_calls?: ToolCall[];
  /** For a `tool` message: the id of the call it answers (Mistral templates require one). */
  tool_call_id?: string;
  /** For a `tool` message: the function that produced it. */
  name?: string;
}

/** A function the model may call, in the OpenAI / transformers shape: a JSON Schema for its arguments. */
export interface ToolSpec {
  type: "function";
  function: { name: string; description?: string; parameters?: Record<string, unknown> };
}

export interface ToolCall {
  id?: string;
  type: "function";
  function: { name: string; arguments: Record<string, unknown> };
}

/** A document for grounded answers (templates such as Granite's list them in the prompt). */
export type ChatDocument = Record<string, unknown>;

export interface RenderOptions {
  /** Qwen3-style reasoning models: false appends an empty think block so the model answers directly. */
  enableThinking?: boolean;
  /** Functions the model may call. Only models whose chat template reads `tools` see them. */
  tools?: readonly ToolSpec[];
  /** Documents to ground the answer in. Only models whose chat template reads `documents` see them. */
  documents?: readonly ChatDocument[];
}

/**
 * Turns messages into a prompt. Models that ship a `chat_template` run it (see `templateFormat`); the
 * built-in chatml and Llama 3 formats are the fallback for models without one.
 */
export interface ChatFormat {
  readonly name: string;
  render(messages: readonly ChatMessage[], opts?: RenderOptions): string;
  /** Token strings that end an assistant turn. */
  readonly stop: readonly string[];
}

/** The built-in formats know no tool or document syntax; quietly dropping them would mislead the model. */
function plainOnly(name: string, messages: readonly ChatMessage[], opts: RenderOptions): void {
  if (opts.tools?.length || opts.documents?.length || messages.some((m) => m.role === "tool" || m.tool_calls?.length)) {
    throw new Error(`tools and documents need the model's own chat template; this model uses the built-in ${name} format`);
  }
}

export const chatml: ChatFormat = {
  name: "chatml",
  stop: ["<|im_end|>", "<|endoftext|>"],
  render(messages, opts = {}) {
    plainOnly("chatml", messages, opts);
    let s = messages.map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join("");
    s += "<|im_start|>assistant\n";
    if (opts.enableThinking === false) s += "<think>\n\n</think>\n\n";
    return s;
  },
};

export const llama3: ChatFormat = {
  name: "llama3",
  stop: ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"],
  render(messages, opts = {}) {
    plainOnly("llama3", messages, opts);
    let s = "<|begin_of_text|>";
    for (const m of messages) s += `<|start_header_id|>${m.role}<|end_header_id|>\n\n${m.content.trim()}<|eot_id|>`;
    return s + "<|start_header_id|>assistant<|end_header_id|>\n\n";
  },
};

/** A model's Jinja chat template, from `tokenizer_config.json`, `chat_template.jinja` or GGUF metadata. */
export interface ChatTemplate {
  source: string;
  bosToken?: string;
  eosToken?: string;
}

/** Turn-ending tokens that, when a template writes them, also end generation. */
const TURN_ENDS = ["<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<|end_of_text|>", "<|endoftext|>", "<|eom_id|>", "</s>"];

/** A chat format that runs the model's own template, rendering the prompt for the assistant's next turn. */
export function templateFormat(t: ChatTemplate): ChatFormat {
  const tpl = new Template(t.source);
  const stop = [...new Set([...(t.eosToken ? [t.eosToken] : []), ...TURN_ENDS.filter((s) => t.source.includes(s))])];
  // A template that never names them would drop tools or documents without a trace.
  const reads = (name: string) => new RegExp(`\\b${name}\\b`).test(t.source);
  const readsTools = reads("tools");
  const readsDocuments = reads("documents");
  return {
    name: "chat_template",
    stop,
    render(messages, opts = {}) {
      if (opts.tools?.length && !readsTools) throw new Error("this model's chat template has no tool support (it never reads `tools`)");
      if (opts.documents?.length && !readsDocuments) throw new Error("this model's chat template has no document support (it never reads `documents`)");
      return tpl.render({
        messages,
        add_generation_prompt: true,
        bos_token: t.bosToken ?? "",
        eos_token: t.eosToken ?? "",
        enable_thinking: opts.enableThinking,
        // None rather than undefined when absent, as transformers passes them: `tools is not none`
        // must be false then.
        tools: opts.tools?.length ? opts.tools : null,
        documents: opts.documents?.length ? opts.documents : null,
      });
    },
  };
}

export function detectChatFormat(tok: Tokenizer): ChatFormat {
  if (tok.chatTemplate) {
    try {
      return templateFormat(tok.chatTemplate);
    } catch (err) {
      if (!(err instanceof TemplateError)) throw err;
      console.warn(`[kiln] the model's chat template uses unsupported Jinja (${err.message}); falling back to a built-in format`);
    }
  }
  if (tok.tokenId("<|im_start|>") !== undefined) return chatml;
  if (tok.tokenId("<|start_header_id|>") !== undefined) return llama3;
  throw new Error("this model has no chat template, and its tokenizer matches no built-in format (chatml, llama3)");
}

const TOOL_CALL = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;

/**
 * Splits a reply into text and tool calls in the `<tool_call>{"name": …, "arguments": …}</tool_call>`
 * format that Qwen2.5, Qwen3, SmolLM3 and Hermes-style templates ask for. A block that is not valid JSON
 * with a string `name` stays in the text. Formats whose markers are special tokens (Mistral's
 * `[TOOL_CALLS]`, Granite's `<|tool_call|>`) are dropped when decoding and are not recognised.
 */
export function parseToolCalls(text: string): { content: string; toolCalls: ToolCall[] } {
  const toolCalls: ToolCall[] = [];
  const content = text.replace(TOOL_CALL, (block, body: string) => {
    let v: unknown;
    try {
      v = JSON.parse(body);
    } catch {
      return block;
    }
    const call = v as { name?: unknown; arguments?: unknown; parameters?: unknown };
    if (!call || typeof call !== "object" || typeof call.name !== "string") return block;
    let args = call.arguments ?? call.parameters ?? {};
    // Some models write the arguments as a JSON string, as the OpenAI wire format does.
    if (typeof args === "string") {
      try {
        args = JSON.parse(args);
      } catch {
        return block;
      }
    }
    if (!args || typeof args !== "object" || Array.isArray(args)) return block;
    toolCalls.push({ type: "function", function: { name: call.name, arguments: args as Record<string, unknown> } });
    return "";
  });
  return { content: toolCalls.length ? content.trim() : text, toolCalls };
}
