import { describe, expect, it } from "vitest";
import { tasks, type TaskModel } from "../src/tasks";

/** A model that replies `reply` and scores choices by `logprobs`. */
function fake(reply: string, logprobs: (choices: string[]) => number[] = (c) => c.map(() => 0)) {
  const seen: { messages: unknown[]; opts: unknown }[] = [];
  const llm: TaskModel = {
    chat: (messages, opts) => {
      seen.push({ messages, opts });
      return Object.assign((async function* () { yield reply; })(), { stats: Promise.resolve(undefined) });
    },
    scoreChoices: async (_m, choices) => logprobs(choices),
  };
  return { t: tasks(llm), seen };
}

describe("LLM tasks", () => {
  it("classify returns the labels by the model's probabilities", async () => {
    const { t } = fake("", () => [Math.log(0.2), Math.log(0.7), Math.log(0.1)]);
    const r = await t.classify("my card was charged twice", ["bug", "billing", "other"]);
    expect(r.label).toBe("billing");
    expect(r.labels).toEqual(["billing", "bug", "other"]);
    expect(r.scores[0]).toBeCloseTo(0.7, 6);
    expect(r.scores.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  it("multi_label scores each label as yes / no", async () => {
    const { t } = fake("", () => [Math.log(0.9), Math.log(0.3)]);
    const r = await t.classify("x", ["a", "b"], { multi_label: true });
    expect(r.scores[0]).toBeCloseTo(0.75, 6);
  });

  it("ner keeps mentions found in the text, typed by score", async () => {
    const text = "Aspirin was given to Ms. Kaur. Later, aspirin again.";
    let call = 0;
    const replies = [
      JSON.stringify({ entities: [{ text: "Aspirin", type: "drug" }, { text: "Ibuprofen", type: "drug" }] }),
      JSON.stringify({ mentions: ["aspirin", "Kaur"] }),
      JSON.stringify({ mentions: ["Ms. Kaur"] }),
    ];
    const llm: TaskModel = {
      chat: () => Object.assign((async function* () { yield replies[call++]!; })(), { stats: Promise.resolve(undefined) }),
      scoreChoices: async (m) => (String(m[1]!.content).includes('"Ms. Kaur"') ? [Math.log(0.1), Math.log(0.9), -9] : [Math.log(0.8), Math.log(0.2), -9]),
    };
    const r = await tasks(llm).ner(text, ["drug", "person"]);
    expect(r.map((e) => [e.entity_group, text.slice(e.start, e.end), +e.score.toFixed(2)])).toEqual([
      ["drug", "Aspirin", 0.8],
      ["person", "Ms. Kaur", 0.9],
      ["drug", "aspirin", 0.8],
    ]);
    expect(r.dropped).toBe(1);
  });

  it("answer copies a span when the context answers, else null", async () => {
    const ctx = "Mount Ardel rises to 3,412 metres.";
    const yes = async (c: string[]) => (c[0] === "yes" ? [Math.log(0.9), Math.log(0.1)] : [0]);
    const no = async (c: string[]) => (c[0] === "yes" ? [Math.log(0.2), Math.log(0.8)] : [0]);
    expect(await fake(JSON.stringify({ answer: "3,412 metres" }), (c) => c.map(() => 0)).t.answer("How tall?", ctx).then((r) => r.answer)).toBe("3,412 metres");
    const mk = (reply: string, score: (c: string[]) => Promise<number[]>) => tasks({ chat: () => Object.assign((async function* () { yield reply; })(), { stats: Promise.resolve(undefined) }), scoreChoices: async (_m, c) => score(c) });
    const a = await mk(JSON.stringify({ answer: "3,412 metres" }), yes).answer("How tall?", ctx);
    expect(a).toEqual({ answer: "3,412 metres", score: expect.closeTo(0.9, 6), start: 21, end: 33 });
    expect((await mk(JSON.stringify({ answer: "Bell" }), no).answer("Who?", ctx)).answer).toBeNull();
    expect((await mk(JSON.stringify({ answer: "Bell" }), yes).answer("Who?", ctx)).answer).toBeNull();
  });
});
