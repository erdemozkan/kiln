import { describe, expect, it } from "vitest";
import type { ChatMessage } from "@kiln/tokenizers";
import { runTools, type RunnableTool } from "../src/tools";
import type { GenerateOptions, KilnModel } from "../src/client";

/** A model that gives `replies` in turn and records what it was asked. */
function scripted(replies: string[]) {
  const asked: { messages: ChatMessage[]; opts: GenerateOptions }[] = [];
  const llm = {
    chat(messages: ChatMessage[], opts: GenerateOptions = {}) {
      asked.push({ messages: [...messages], opts });
      const text = replies[asked.length - 1] ?? "done";
      return (async function* () {
        yield text.slice(0, 3);
        yield text.slice(3);
      })();
    },
  } as unknown as Pick<KilnModel, "chat">;
  return { llm, asked };
}

const call = (name: string, args: unknown) => `<tool_call>\n${JSON.stringify({ name, arguments: args })}\n</tool_call>`;
const weather: RunnableTool = {
  type: "function",
  function: { name: "get_weather", description: "Weather in a city", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } },
  execute: ({ city }) => ({ city, celsius: city === "Paris" ? 18 : 25 }),
};

describe("runTools", () => {
  it("runs the calls, passes the answers back and returns the final reply", async () => {
    const { llm, asked } = scripted([call("get_weather", { city: "Paris" }), "It is 18 °C in Paris."]);
    const seen: string[] = [];
    const r = await runTools(llm, [{ role: "user", content: "Weather in Paris?" }], { tools: [weather], toolChoice: "required", temperature: 0, onText: (t) => seen.push(t) });
    expect(r.content).toBe("It is 18 °C in Paris.");
    expect(r.unfinished).toBe(false);
    expect(r.steps).toEqual([{ call: { type: "function", function: { name: "get_weather", arguments: { city: "Paris" } } }, result: { city: "Paris", celsius: 18 } }]);
    expect(r.messages.map((m) => m.role)).toEqual(["user", "assistant", "tool", "assistant"]);
    expect(r.messages[2]).toEqual({ role: "tool", name: "get_weather", content: '{"city":"Paris","celsius":18}' });
    expect(seen.join("")).toContain("18 °C");
    // The model sees the specs without the functions; toolChoice holds for the first reply only.
    expect(asked[0]!.opts.tools).toEqual([{ type: "function", function: weather.function }]);
    expect(asked[0]!.opts.toolChoice).toBe("required");
    expect(asked[1]!.opts.toolChoice).toBeUndefined();
    expect(asked[1]!.opts.temperature).toBe(0);
    expect(asked[1]!.messages.length).toBe(3);
  });

  it("answers a failing or unknown tool with an error the model can read", async () => {
    const boom: RunnableTool = { type: "function", function: { name: "boom" }, execute: () => Promise.reject(new Error("no network")) };
    const { llm } = scripted([call("boom", {}) + call("nope", {}), "Sorry."]);
    const r = await runTools(llm, [{ role: "user", content: "go" }], { tools: [boom] });
    expect(r.steps.map((s) => s.error)).toEqual(["no network", 'no tool named "nope"']);
    expect(r.messages[2]!.content).toBe('{"error":"no network"}');
    expect(r.content).toBe("Sorry.");
  });

  it("stops after maxRounds replies", async () => {
    const { llm, asked } = scripted(Array.from({ length: 9 }, () => call("get_weather", { city: "Rome" })));
    const r = await runTools(llm, [{ role: "user", content: "go" }], { tools: [weather], maxRounds: 2 });
    expect(asked.length).toBe(2);
    expect(r.unfinished).toBe(true);
    expect(r.steps.length).toBe(2);
    await expect(runTools(llm, [], { tools: [] })).rejects.toThrow(/at least one tool/);
  });
});
