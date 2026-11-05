import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { chatml, llama3, parseToolCalls, templateFormat, type ChatMessage, type ToolSpec } from "../src/chat";

interface Fixture {
  conversations: Record<string, ChatMessage[]>;
  templates: { name: string; template: string; cases: { messages: string; vars: Record<string, unknown>; text?: string }[] }[];
}

const fx = JSON.parse(readFileSync(fileURLToPath(new URL("./fixtures/chat-templates.json", import.meta.url)), "utf8")) as Fixture;
const qwen = fx.templates.find((t) => t.name === "qwen2.5")!;
const mistral = fx.templates.find((t) => t.name === "mistral-v0.3")!;
const tools = qwen.cases.find((c) => c.vars.tools)!.vars.tools as ToolSpec[];

describe("chat formats with tools", () => {
  it("passes tools to the model's template the way transformers does", () => {
    const f = templateFormat({ source: qwen.template, bosToken: "<s>", eosToken: "<|im_end|>" });
    for (const c of qwen.cases) {
      const render = f.render(fx.conversations[c.messages]!, { tools: c.vars.tools as ToolSpec[] | undefined });
      expect(render, `${c.messages} tools=${!!c.vars.tools}`).toBe(c.text);
    }
  });

  it("passes None when there are no tools, so `tools is not none` is false", () => {
    // An undefined variable is not none in Jinja, so a template that skips the `is defined` check sees tools.
    const probe = templateFormat({ source: "{{ 'tools' if tools is not none else 'none' }}/{{ 'docs' if documents is not none else 'none' }}" });
    expect(probe.render([])).toBe("none/none");
    expect(probe.render([], { tools: [], documents: [] })).toBe("none/none");
    expect(probe.render([], { tools, documents: [{ text: "x" }] })).toBe("tools/docs");
    // Mistral lists [AVAILABLE_TOOLS] whenever tools is not none.
    const f = templateFormat({ source: mistral.template, bosToken: "<s>", eosToken: "</s>" });
    const plain = mistral.cases.find((c) => c.messages === "single" && !c.vars.tools)!;
    expect(f.render(fx.conversations.single!)).toBe(plain.text);
    expect(f.render(fx.conversations.single!, { tools: [] })).toBe(plain.text);
    expect(f.render(fx.conversations.single!, { tools })).toContain("[AVAILABLE_TOOLS]");
  });

  it("templates that never read tools or documents refuse them rather than drop them", () => {
    const smol = templateFormat({ source: fx.templates.find((t) => t.name === "smollm2")!.template });
    const msgs = fx.conversations.single!;
    expect(() => smol.render(msgs, { tools })).toThrow(/no tool support/);
    expect(() => smol.render(msgs, { documents: [{ text: "x" }] })).toThrow(/no document support/);
    expect(() => templateFormat({ source: qwen.template }).render(msgs, { documents: [{ text: "x" }] })).toThrow(/no document support/);
    expect(smol.render(msgs, { tools: [] })).toBe(smol.render(msgs));
  });

  it("built-in formats refuse tools rather than drop them", () => {
    const msgs: ChatMessage[] = [{ role: "user", content: "hi" }];
    for (const f of [chatml, llama3]) {
      expect(() => f.render(msgs, { tools })).toThrow(/chat template/);
      expect(() => f.render([...msgs, { role: "tool", content: "18" }])).toThrow(/chat template/);
      expect(f.render(msgs, { tools: [] })).toContain("hi");
    }
  });
});

describe("parseToolCalls", () => {
  it("reads <tool_call> blocks and leaves the rest as text", () => {
    const r = parseToolCalls('Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n<tool_call>{"name": "get_time", "arguments": "{\\"city\\": \\"Rome\\"}"}</tool_call>');
    expect(r.content).toBe("Let me check.");
    expect(r.toolCalls).toEqual([
      { type: "function", function: { name: "get_weather", arguments: { city: "Paris" } } },
      { type: "function", function: { name: "get_time", arguments: { city: "Rome" } } },
    ]);
  });

  it("keeps blocks that are not valid calls as text", () => {
    for (const text of ["<tool_call>{not json}</tool_call>", '<tool_call>{"arguments": {}}</tool_call>', '<tool_call>{"name": "f", "arguments": [1]}</tool_call>', '<tool_call>{"name": "f"', "plain answer"]) {
      expect(parseToolCalls(text)).toEqual({ content: text, toolCalls: [] });
    }
    expect(parseToolCalls('<tool_call>{"name": "f"}</tool_call>').toolCalls[0]!.function.arguments).toEqual({});
  });

  it("round-trips: parsed calls render back into the template's history format", () => {
    const reply = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>';
    const { content, toolCalls } = parseToolCalls(reply);
    const f = templateFormat({ source: qwen.template });
    const prompt = f.render([{ role: "user", content: "What's the weather in Paris?" }, { role: "assistant", content, tool_calls: toolCalls }], { tools });
    expect(prompt).toContain(`<|im_start|>assistant\n${reply}<|im_end|>`);
  });
});
