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
 * Built-in chat formats. A Jinja subset that runs the model's own `chat_template` is on the roadmap;
 * these cover the families the MVP supports and are tested against their reference output.
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

export function detectChatFormat(tok: Tokenizer): ChatFormat {
  if (tok.tokenId("<|im_start|>") !== undefined) return chatml;
  if (tok.tokenId("<|start_header_id|>") !== undefined) return llama3;
  throw new Error("could not detect a chat format for this tokenizer (supported: chatml, llama3)");
}
