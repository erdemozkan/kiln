import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { chatml, hermesToolFormat, llama3, parseToolCalls, templateFormat, type ChatMessage, type ToolSpec } from "../src/chat";

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

describe("FunctionGemma calls", () => {
  it("reads call:name{…} with escaped strings, numbers, lists and nested values", () => {
    const r = parseToolCalls('<start_function_call>call:get_date_time{place:<escape>New York, USA<escape>}<end_function_call>');
    expect(r).toEqual({ content: "", toolCalls: [{ type: "function", function: { name: "get_date_time", arguments: { place: "New York, USA" } } }] });
    const many = parseToolCalls(
      'Sure.<start_function_call>call:set_light{brightness:50,on:true,room:<escape>kitchen<escape>,tags:[<escape>a}b<escape>,2],opts:{<escape>k<escape>:null,x:-1.5e2}}<end_function_call><start_function_call>call:noargs{}<end_function_call>',
    );
    expect(many.content).toBe("Sure.");
    expect(many.toolCalls.map((c) => c.function)).toEqual([
      { name: "set_light", arguments: { brightness: 50, on: true, room: "kitchen", tags: ["a}b", 2], opts: { k: null, x: -150 } } },
      { name: "noargs", arguments: {} },
    ]);
    // The end marker may be missing (the reply stops at the response marker), and a string may come unescaped.
    expect(parseToolCalls("<start_function_call>call:calculate{expression:2 + 2}").toolCalls[0]!.function).toEqual({ name: "calculate", arguments: { expression: "2 + 2" } });
    expect(parseToolCalls("<start_function_call>call:f{a:<escape>never closed").toolCalls).toEqual([]);
    expect(parseToolCalls("<start_function_call>hello").content).toBe("<start_function_call>hello");
  });

  it("ends the turn at the response marker and hands results to the template as mappings", () => {
    const source = "{{ bos_token }}{% for m in messages %}{% if m['role'] == 'tool' %}<start_function_response>response:{{ m['name'] }}{% if m['content'] is mapping %}{ {%- for k, v in m['content'] | dictsort %}{{ k }}:{{ v }},{% endfor -%} }{% else %}RAW{% endif %}<end_function_response>{% else %}{{ m['content'] }}<end_of_turn>{% endif %}{% endfor %}{% if tools %}<start_function_call>{% endif %}";
    const f = templateFormat({ source, bosToken: "<bos>", eosToken: "<eos>" });
    expect(f.stop).toContain("<start_function_response>");
    const p = f.render([{ role: "user", content: "hi" }, { role: "tool", name: "f", content: '{"b":2,"a":1}' }, { role: "tool", name: "g", content: "plain" }]);
    expect(p).toBe("<bos>hi<end_of_turn><start_function_response>response:f{a:1,b:2,}<end_function_response><start_function_response>response:gRAW<end_function_response>");
  });
});

describe("Hermes tool format", () => {
  // Hermes 3's template: plain ChatML, any role.
  const source = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
  const f = hermesToolFormat(templateFormat({ source, eosToken: "<|im_end|>" }));
  const spec: ToolSpec = { type: "function", function: { name: "get_time", description: "The time in a city", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } };

  it("puts the tools in the system message, after the caller's own", () => {
    const p = f.render([{ role: "system", content: "Be brief." }, { role: "user", content: "Time in Rome?" }], { tools: [spec] });
    expect(p.startsWith("<|im_start|>system\nBe brief.\n\nYou are a function calling AI model.")).toBe(true);
    expect(p).toContain(`<tools> ${JSON.stringify(spec)} </tools>`);
    expect(p).toContain('<tool_call>\n{"name": <function-name>, "arguments": <args-dict>}\n</tool_call><|im_end|>\n<|im_start|>user\nTime in Rome?<|im_end|>\n<|im_start|>assistant\n');
    expect(f.toolCallTags).toBe(true);
    // Without tools it is the template alone.
    expect(f.render([{ role: "user", content: "Hi" }])).toBe("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
  });

  it("writes earlier calls and their results the way the model was trained on", () => {
    const call = (city: string) => ({ type: "function" as const, function: { name: "get_time", arguments: { city } } });
    const p = f.render(
      [
        { role: "user", content: "Time in Rome and Oslo?" },
        { role: "assistant", content: "", tool_calls: [call("Rome"), call("Oslo")] },
        { role: "tool", name: "get_time", content: '{"time":"10:00"}' },
        { role: "tool", name: "get_time", content: '{"time":"10:00"}' },
      ],
      { tools: [spec] },
    );
    expect(p).toContain('<|im_start|>assistant\n<tool_call>\n{"name":"get_time","arguments":{"city":"Rome"}}\n</tool_call>\n<tool_call>\n{"name":"get_time","arguments":{"city":"Oslo"}}\n</tool_call><|im_end|>\n');
    expect(p).toContain('<|im_start|>tool\n<tool_response>\n{"time":"10:00"}\n</tool_response>\n<tool_response>\n{"time":"10:00"}\n</tool_response><|im_end|>\n<|im_start|>assistant\n');
    const made = parseToolCalls('<tool_call>\n{"arguments": {"city": "Rome"}, "name": "get_time"}\n</tool_call>');
    expect(made.toolCalls).toEqual([call("Rome")]);
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

  it("reads Mistral's [TOOL_CALLS] list and round-trips it through the v0.3 template", () => {
    const reply = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Rome", "unit": "celsius"}, "id": "callA1234"}, {"name": "get_time", "arguments": {"city": "Rome"}, "id": "callB5678"}]';
    const { content, toolCalls } = parseToolCalls(reply);
    expect(content).toBe("");
    const conv = fx.conversations["tool-parallel"]!;
    expect(toolCalls).toEqual(conv[2]!.tool_calls);
    const f = templateFormat({ source: mistral.template, bosToken: "<s>", eosToken: "</s>" });
    const want = mistral.cases.find((c) => c.messages === "tool-parallel" && c.vars.tools)!.text;
    expect(f.render(conv.map((m, i) => (i === 2 ? { ...m, content, tool_calls: toolCalls } : m)), { tools })).toBe(want);
    // Without ids, each call gets the 9-character one the template insists on.
    const bare = parseToolCalls('Sure. [TOOL_CALLS][{"name": "get_time", "arguments": {"city": "Rome"}}]');
    expect(bare.content).toBe("Sure.");
    expect(bare.toolCalls[0]!.id).toMatch(/^[A-Za-z0-9]{9}$/);
  });

  it("reads Mistral's name[ARGS]{…} calls, with or without [CALL_ID]", () => {
    const r = parseToolCalls('[TOOL_CALLS]get_weather[ARGS]{"city": "Paris", "note": "a ] } \\" b"}[TOOL_CALLS]get_time[CALL_ID]abc123XYZ[ARGS]{"city": "Rome"}');
    expect(r.content).toBe("");
    expect(r.toolCalls.map((c) => [c.function.name, c.function.arguments])).toEqual([
      ["get_weather", { city: "Paris", note: 'a ] } " b' }],
      ["get_time", { city: "Rome" }],
    ]);
    expect(r.toolCalls[1]!.id).toBe("abc123XYZ");
  });

  it("reads Granite's <|tool_call|> list", () => {
    const r = parseToolCalls('<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "get_time", "arguments": {"city": "Paris"}}]');
    expect(r).toEqual({
      content: "",
      toolCalls: [
        { type: "function", function: { name: "get_weather", arguments: { city: "Paris" } } },
        { type: "function", function: { name: "get_time", arguments: { city: "Paris" } } },
      ],
    });
  });

  it("leaves markers that are not followed by calls in the text", () => {
    for (const text of ["[TOOL_CALLS] I cannot call tools.", '<|tool_call|>[{"x": 1}]', "[TOOL_CALLS][{\"name\": \"f\"", "[TOOL_CALLS]f[ARGS]oops", '<|tool_call|>[{"name": "f", "arguments": [1]}]']) {
      expect(parseToolCalls(text)).toEqual({ content: text, toolCalls: [] });
    }
  });

  it("round-trips: parsed calls render back into the template's history format", () => {
    const reply = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>';
    const { content, toolCalls } = parseToolCalls(reply);
    const f = templateFormat({ source: qwen.template });
    const prompt = f.render([{ role: "user", content: "What's the weather in Paris?" }, { role: "assistant", content, tool_calls: toolCalls }], { tools });
    expect(prompt).toContain(`<|im_start|>assistant\n${reply}<|im_end|>`);
  });
});
