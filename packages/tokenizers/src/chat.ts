import { Template, TemplateError } from "./jinja";
import type { Tokenizer } from "./tokenizer";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface RenderOptions {
  /** Qwen3-style reasoning models: false appends an empty think block so the model answers directly. */
  enableThinking?: boolean;
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

export const chatml: ChatFormat = {
  name: "chatml",
  stop: ["<|im_end|>", "<|endoftext|>"],
  render(messages, opts = {}) {
    let s = messages.map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join("");
    s += "<|im_start|>assistant\n";
    if (opts.enableThinking === false) s += "<think>\n\n</think>\n\n";
    return s;
  },
};

export const llama3: ChatFormat = {
  name: "llama3",
  stop: ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"],
  render(messages) {
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
  return {
    name: "chat_template",
    stop,
    render(messages, opts = {}) {
      return tpl.render({
        messages,
        add_generation_prompt: true,
        bos_token: t.bosToken ?? "",
        eos_token: t.eosToken ?? "",
        enable_thinking: opts.enableThinking,
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
