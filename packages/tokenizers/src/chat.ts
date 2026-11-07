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
  /** True when the template asks for tool calls as `<tool_call>{…}</tool_call>` (what `toolChoice` constrains). */
  readonly toolCallTags?: boolean;
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
    toolCallTags: readsTools && t.source.includes("<tool_call>"),
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
 * Special tokens that mark tool calls. The decoder keeps them in the text (it drops other special
 * tokens), so `parseToolCalls` can find the calls of templates that use them.
 */
export const TOOL_CALL_MARKERS: readonly string[] = ["[TOOL_CALLS]", "[CALL_ID]", "[ARGS]", "<|tool_call|>"];

/** One call from a parsed `{"name": …, "arguments": …}` value, or undefined when it is not one. */
function toCall(v: unknown): ToolCall | undefined {
  const call = v as { name?: unknown; arguments?: unknown; parameters?: unknown; id?: unknown };
  if (!call || typeof call !== "object" || typeof call.name !== "string") return undefined;
  let args = call.arguments ?? call.parameters ?? {};
  // Some models write the arguments as a JSON string, as the OpenAI wire format does.
  if (typeof args === "string") {
    try {
      args = JSON.parse(args);
    } catch {
      return undefined;
    }
  }
  if (!args || typeof args !== "object" || Array.isArray(args)) return undefined;
  const out: ToolCall = { type: "function", function: { name: call.name, arguments: args as Record<string, unknown> } };
  if (typeof call.id === "string") out.id = call.id;
  return out;
}

/** The end of the JSON object or array that starts at `s[i]`, or -1 when it does not close. */
function jsonEnd(s: string, i: number): number {
  let depth = 0;
  let str = false;
  for (let j = i; j < s.length; j++) {
    const c = s[j];
    if (str) {
      if (c === "\\") j++;
      else if (c === '"') str = false;
    } else if (c === '"') str = true;
    else if (c === "{" || c === "[") depth++;
    else if (c === "}" || c === "]") {
      if (--depth === 0) return j + 1;
    }
  }
  return -1;
}

/** Reads a JSON value at `s[i]` (after spaces): `[value, end]`, or undefined. */
function readJSON(s: string, i: number): [unknown, number] | undefined {
  while (/\s/.test(s[i] ?? "")) i++;
  if (s[i] !== "{" && s[i] !== "[") return undefined;
  const end = jsonEnd(s, i);
  if (end < 0) return undefined;
  try {
    return [JSON.parse(s.slice(i, end)), end];
  } catch {
    return undefined;
  }
}

/** Calls from a JSON list (or one object) of `{"name", "arguments"}`; undefined if any is not a call. */
function callList(v: unknown): ToolCall[] | undefined {
  const calls = (Array.isArray(v) ? v : [v]).map(toCall);
  return calls.length && calls.every((c) => c) ? (calls as ToolCall[]) : undefined;
}

/** Mistral templates require 9-character alphanumeric call ids; models often write none. */
function mistralId(): string {
  const a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  return Array.from({ length: 9 }, () => a[Math.floor(Math.random() * a.length)]).join("");
}

/**
 * Mistral's calls after `[TOOL_CALLS]`: a JSON list (v3, v7), or `name[ARGS]{…}` per call (v11 and
 * later, with an optional `[CALL_ID]id` before `[ARGS]`). Returns the calls and where they end.
 */
function mistralCalls(s: string, i: number): [ToolCall[], number] | undefined {
  const json = readJSON(s, i);
  if (json) {
    const calls = callList(json[0]);
    return calls && [calls, json[1]];
  }
  const calls: ToolCall[] = [];
  const named = /\s*([\w.-]+)\s*(?:\[CALL_ID\]\s*(\w+)\s*)?\[ARGS\]/y;
  for (;;) {
    named.lastIndex = i;
    const m = named.exec(s);
    const args = m && readJSON(s, named.lastIndex);
    const call = args && toCall({ name: m[1], arguments: args[0], id: m[2] });
    if (!call) break;
    calls.push(call);
    i = args[1];
    const next = /\s*\[TOOL_CALLS\]/y;
    next.lastIndex = i;
    if (!next.test(s)) break;
    i = next.lastIndex;
  }
  return calls.length ? [calls, i] : undefined;
}

/**
 * Splits a reply into text and tool calls. It reads the formats that chat templates ask for:
 * `<tool_call>{"name": …, "arguments": …}</tool_call>` blocks (Qwen2.5, Qwen3, SmolLM3, Hermes),
 * Granite's `<|tool_call|>[{…}, …]`, and Mistral's `[TOOL_CALLS]`, either as a JSON list or as
 * `name[ARGS]{…}`. Mistral calls without an id get a random 9-character one, which its templates
 * need when the calls go back into the history. Anything that is not valid JSON with a string `name`
 * (and object arguments) stays in the text.
 */
export function parseToolCalls(text: string): { content: string; toolCalls: ToolCall[] } {
  const toolCalls: ToolCall[] = [];
  let content = text.replace(TOOL_CALL, (block, body: string) => {
    let v: unknown;
    try {
      v = JSON.parse(body);
    } catch {
      return block;
    }
    const call = toCall(v);
    if (!call) return block;
    toolCalls.push(call);
    return "";
  });
  // Marker formats: each marker starts a run of calls; what does not parse stays as text.
  for (const marker of ["[TOOL_CALLS]", "<|tool_call|>"]) {
    let from = 0;
    for (let at = content.indexOf(marker, from); at >= 0; at = content.indexOf(marker, from)) {
      const i = at + marker.length;
      let got: [ToolCall[], number] | undefined;
      if (marker === "[TOOL_CALLS]") {
        got = mistralCalls(content, i);
        for (const c of got?.[0] ?? []) c.id ??= mistralId();
      } else {
        const json = readJSON(content, i);
        const calls = json && callList(json[0]);
        got = calls && [calls, json[1]];
      }
      if (!got) {
        from = i;
        continue;
      }
      toolCalls.push(...got[0]);
      content = content.slice(0, at) + content.slice(got[1]);
      from = at;
    }
  }
  return { content: toolCalls.length ? content.trim() : text, toolCalls };
}
