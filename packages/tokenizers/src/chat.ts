import { Template, TemplateError } from "./jinja";
import type { Tokenizer } from "./tokenizer";

/** A piece of a message: text, or an image (for models that take images, such as Gemma 3 with its vision tower). */
export type ContentPart = { type: "text"; text: string } | { type: "image"; image: unknown } | { type: "audio"; audio: unknown } | { type: "video"; video: unknown };

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  /** Text, or a list of text and image parts (images where the model takes them). */
  content: string | ContentPart[];
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
  /** True when the template asks for them as Granite's `<|tool_call|>[{…}, …]`. */
  readonly toolCallList?: boolean;
}

/** The built-in formats know no tool or document syntax; quietly dropping them would mislead the model. */
function plainOnly(name: string, messages: readonly ChatMessage[], opts: RenderOptions): void {
  if (opts.tools?.length || opts.documents?.length || messages.some((m) => m.role === "tool" || m.tool_calls?.length)) {
    throw new Error(`tools and documents need the model's own chat template; this model uses the built-in ${name} format`);
  }
}

/** A message's text: the string, or its text parts joined (image parts have no text). */
export function textOf(content: ChatMessage["content"]): string {
  return typeof content === "string" ? content : content.map((p) => (p.type === "text" ? p.text : "")).join("");
}

export const chatml: ChatFormat = {
  name: "chatml",
  stop: ["<|im_end|>", "<|endoftext|>"],
  render(messages, opts = {}) {
    plainOnly("chatml", messages, opts);
    let s = messages.map((m) => `<|im_start|>${m.role}\n${textOf(m.content)}<|im_end|>\n`).join("");
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
    for (const m of messages) s += `<|start_header_id|>${m.role}<|end_header_id|>\n\n${textOf(m.content).trim()}<|eot_id|>`;
    return s + "<|start_header_id|>assistant<|end_header_id|>\n\n";
  },
};

/**
 * FunctionGemma ends a call by opening the function's response, which is the caller's to write:
 * the token ends the model's turn.
 */
const FUNCTION_RESPONSE = "<start_function_response>";

/** A model's Jinja chat template, from `tokenizer_config.json`, `chat_template.jinja` or GGUF metadata. */
export interface ChatTemplate {
  source: string;
  bosToken?: string;
  eosToken?: string;
}

/** Turn-ending tokens that, when a template writes them, also end generation. */
const TURN_ENDS = ["<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<|end_of_text|>", "<|endoftext|>", "<|eom_id|>", "</s>", FUNCTION_RESPONSE];

/** A chat format that runs the model's own template, rendering the prompt for the assistant's next turn. */
export function templateFormat(t: ChatTemplate): ChatFormat {
  const tpl = new Template(t.source);
  const stop = [...new Set([...(t.eosToken ? [t.eosToken] : []), ...TURN_ENDS.filter((s) => t.source.includes(s))])];
  // A template that never names them would drop tools or documents without a trace.
  const reads = (name: string) => new RegExp(`\\b${name}\\b`).test(t.source);
  const readsTools = reads("tools");
  const readsDocuments = reads("documents");
  const functionCalls = t.source.includes("<start_function_call>");
  return {
    name: "chat_template",
    stop,
    toolCallTags: readsTools && t.source.includes("<tool_call>"),
    toolCallList: readsTools && !t.source.includes("<tool_call>") && t.source.includes("<|tool_call|>"),
    render(messages, opts = {}) {
      // FunctionGemma's template writes a result field by field when it is a mapping; a JSON string would go in whole, as one quoted value.
      if (functionCalls) messages = messages.map((m) => (m.role === "tool" && typeof m.content === "string" ? { ...m, content: (jsonObject(m.content) ?? m.content) as never } : m));
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

function jsonObject(text: string): Record<string, unknown> | undefined {
  try {
    const v: unknown = JSON.parse(text);
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Hermes models (Hermes 2 Pro, Hermes 3) ship a plain ChatML template and take their tools in the
 * system message, in the wording their makers trained them with: function signatures between
 * `<tools>` tags, calls as `<tool_call>{…}</tool_call>`, results as `<tool_response>` in a `tool` turn.
 */
const HERMES_TOOLS = (tools: readonly ToolSpec[]) =>
  `You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> ${tools.map((t) => JSON.stringify(t)).join(" ")} </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"name": {"title": "Name", "type": "string"}, "arguments": {"title": "Arguments", "type": "object"}}, "required": ["name", "arguments"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{"name": <function-name>, "arguments": <args-dict>}\n</tool_call>`;

/** `format` (a ChatML template without tool support) with Hermes' tool convention on top. */
export function hermesToolFormat(format: ChatFormat): ChatFormat {
  const text = (m: ChatMessage) => (typeof m.content === "string" ? m.content : m.content.map((p) => (p.type === "text" ? p.text : "")).join(""));
  return {
    ...format,
    toolCallTags: true,
    render(messages, opts = {}) {
      const uses = !!opts.tools?.length || messages.some((m) => m.role === "tool" || m.tool_calls?.length);
      if (!uses) return format.render(messages, opts);
      const out: ChatMessage[] = [];
      const first = messages[0]?.role === "system" ? messages[0] : undefined;
      // The caller's own instructions come first, then the tool wording.
      if (opts.tools?.length) out.push({ role: "system", content: `${first ? `${text(first)}\n\n` : ""}${HERMES_TOOLS(opts.tools)}` });
      else if (first) out.push(first);
      for (const m of first ? messages.slice(1) : messages) {
        if (m.role === "tool") {
          const body = `<tool_response>\n${text(m)}\n</tool_response>`;
          const prev = out[out.length - 1];
          // Several results in a row are one tool turn.
          if (prev?.role === "tool") prev.content = `${prev.content as string}\n${body}`;
          else out.push({ role: "tool", content: body });
        } else if (m.tool_calls?.length) {
          const calls = m.tool_calls.map((c) => `<tool_call>\n${JSON.stringify({ name: c.function.name, arguments: c.function.arguments })}\n</tool_call>`).join("\n");
          out.push({ role: "assistant", content: [text(m), calls].filter(Boolean).join("\n") });
        } else out.push(m);
      }
      return format.render(out, { ...opts, tools: undefined });
    },
  };
}

export function detectChatFormat(tok: Tokenizer): ChatFormat {
  if (tok.chatTemplate) {
    try {
      const format = templateFormat(tok.chatTemplate);
      // A ChatML template that never reads `tools`, in a vocabulary with Hermes' call tokens.
      const hermes = !/\btools\b/.test(tok.chatTemplate.source) && tok.chatTemplate.source.includes("<|im_start|>") && tok.tokenId("<tool_call>") !== undefined && tok.tokenId("<tool_response>") !== undefined;
      return hermes ? hermesToolFormat(format) : format;
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
export const TOOL_CALL_MARKERS: readonly string[] = ["[TOOL_CALLS]", "[CALL_ID]", "[ARGS]", "<|tool_call|>", "<start_function_call>", "<end_function_call>", "<escape>"];

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

const ESCAPE = "<escape>";

/**
 * FunctionGemma's argument syntax at `s[i]`: `<escape>text<escape>`, a number, true / false / null,
 * `[…]`, or `{key:value,…}` (keys bare, or escaped inside nested values). Returns the value and
 * where it ends, or undefined if `s` does not hold one there.
 */
function gemmaValue(s: string, i: number): [unknown, number] | undefined {
  if (s.startsWith(ESCAPE, i)) {
    const end = s.indexOf(ESCAPE, i + ESCAPE.length);
    return end < 0 ? undefined : [s.slice(i + ESCAPE.length, end), end + ESCAPE.length];
  }
  if (s[i] === "{") {
    const out: Record<string, unknown> = {};
    i++;
    while (s[i] !== "}") {
      let key: string;
      if (s.startsWith(ESCAPE, i)) {
        const k = gemmaValue(s, i);
        if (!k) return undefined;
        [key, i] = k as [string, number];
      } else {
        const colon = s.indexOf(":", i);
        if (colon < 0) return undefined;
        key = s.slice(i, colon).trim();
        i = colon;
      }
      if (!key || s[i] !== ":") return undefined;
      const v = gemmaValue(s, i + 1);
      if (!v) return undefined;
      out[key] = v[0];
      i = v[1];
      if (s[i] === ",") i++;
      else if (s[i] !== "}") return undefined;
    }
    return [out, i + 1];
  }
  if (s[i] === "[") {
    const out: unknown[] = [];
    i++;
    while (s[i] !== "]") {
      const v = gemmaValue(s, i);
      if (!v) return undefined;
      out.push(v[0]);
      i = v[1];
      if (s[i] === ",") i++;
      else if (s[i] !== "]") return undefined;
    }
    return [out, i + 1];
  }
  const word = /^(true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?=[,}\]]|$)/.exec(s.slice(i, i + 40));
  if (word) return [JSON.parse(word[0]), i + word[0].length];
  // A bare word: a string the model forgot to escape. It ends at the next separator.
  const bare = /^[^,}\]<]+/.exec(s.slice(i));
  return bare ? [bare[0].trim(), i + bare[0].length] : undefined;
}

/** FunctionGemma's calls: `<start_function_call>call:name{…}<end_function_call>`, one after another. */
function gemmaCalls(text: string): { content: string; toolCalls: ToolCall[] } {
  const toolCalls: ToolCall[] = [];
  let content = "";
  let at = 0;
  for (;;) {
    const open = text.indexOf("<start_function_call>", at);
    if (open < 0) break;
    const head = /^call:([\w.-]+)/.exec(text.slice(open + 21, open + 121));
    const args = head ? gemmaValue(text, open + 21 + head[0].length) : undefined;
    if (!head || !args || typeof args[0] !== "object" || Array.isArray(args[0])) {
      // Not a call after all: it stays text.
      content += text.slice(at, open + 21);
      at = open + 21;
      continue;
    }
    content += text.slice(at, open);
    toolCalls.push({ type: "function", function: { name: head[1]!, arguments: args[0] as Record<string, unknown> } });
    at = text.startsWith("<end_function_call>", args[1]) ? args[1] + 19 : args[1];
  }
  content += text.slice(at);
  return { content: toolCalls.length ? content.trim() : text, toolCalls };
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
  if (text.includes("<start_function_call>")) return gemmaCalls(text);
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
