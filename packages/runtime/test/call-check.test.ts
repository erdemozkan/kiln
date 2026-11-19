import { describe, expect, it } from "vitest";
import { CpuBackend } from "@kiln/kernels";
import { LlamaModel } from "@kiln/models";
import type { ChatFormat, ToolSpec } from "@kiln/tokenizers";
import { Engine, type GenerateOptions } from "../src/engine";
import type { CallCheck } from "../src/tool-calls";
import { TINY, tinyTokenizer, tinyWeights } from "./tiny-model";

const tool = (parameters: Record<string, unknown> = { type: "object", properties: { q: { type: "string" } } }): ToolSpec => ({ type: "function", function: { name: "f", parameters } });

/** An engine whose template writes calls as `tags` says, with these special tokens. */
function engine(added: Record<string, number>, tags: "tags" | "list" | "none" = "tags") {
  const be = new CpuBackend({ threads: 1 });
  const e = new Engine(be, new LlamaModel(be, TINY, tinyWeights(be), { maxSeq: 64, kvDtype: "f32" }), tinyTokenizer(added));
  const format: ChatFormat = { name: "test", stop: [], toolCallTags: tags === "tags", toolCallList: tags === "list", render: () => "a" };
  Object.defineProperty(e, "chatFormat", { value: format });
  const p = e as unknown as { callCheck(o: GenerateOptions): CallCheck | undefined; grammars: WeakMap<object, unknown> };
  return { e, check: (o: GenerateOptions) => p.callCheck(o), compiled: (c: CallCheck) => p.grammars.has(c.format) };
}

describe("Engine.callCheck", () => {
  it("takes the template's marker, closing token and reasoning tokens", () => {
    const { check, compiled } = engine({ "<tool_call>": 90, "</tool_call>": 91, "<think>": 92, "</think>": 93 });
    const c = check({ tools: [tool()] })!;
    expect(c).toMatchObject({ trigger: 90, close: 91, think: { open: 92, close: 93 }, context: 64, format: { type: "tool_call", syntax: "tags_inner" } });
    // Compiled once here, for every call of the reply.
    expect(compiled(c)).toBe(true);
    // Without a closing token the grammar writes the tags itself.
    expect(engine({ "<tool_call>": 90 }).check({ tools: [tool()] })).toMatchObject({ trigger: 90, close: undefined, think: undefined, format: { syntax: "tags_body" } });
    expect(engine({ "<|tool_call|>": 94 }, "list").check({ tools: [tool()] })).toMatchObject({ trigger: 94, close: undefined, format: { syntax: "list_body" } });
  });

  it("checks nothing it cannot", () => {
    const { check } = engine({ "<tool_call>": 90 });
    expect(check({})).toBeUndefined();
    expect(check({ tools: [tool()], checkToolCalls: false })).toBeUndefined();
    expect(check({ tools: [tool()], toolChoice: "required" })).toBeUndefined();
    expect(check({ tools: [tool()], format: { type: "json" } })).toBeUndefined();
    // A marker that is not one token, or a template whose call format is not known.
    expect(engine({}).check({ tools: [tool()] })).toBeUndefined();
    expect(engine({ "<tool_call>": 90 }, "none").check({ tools: [tool()] })).toBeUndefined();
  });

  it("leaves a reply unchecked when a tool's schema cannot be compiled, rather than failing at the call", async () => {
    const { e, check } = engine({ "<tool_call>": 90 });
    const tools = [tool({ type: "object", properties: { tags: { type: "array", items: { type: "string" }, uniqueItems: true } } })];
    expect(check({ tools })).toBeUndefined();
    const out: string[] = [];
    for await (const d of e.complete("ab", { tools, maxTokens: 3, temperature: 0 })) out.push(d.text);
    expect(e.lastStats).toMatchObject({ generatedTokens: 3, stopReason: "maxTokens" });
  });
});
