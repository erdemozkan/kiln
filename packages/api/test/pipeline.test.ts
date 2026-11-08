import { describe, expect, it } from "vitest";
import type { GenerateOptions, TextStream } from "../src/client";
import { runTextGeneration, TextStreamer, toGenerateOptions, toLoadOptions, type ProgressInfo } from "../src/pipeline";

/** A model that replies "Paris." in two pieces and records what it was asked. */
function fakeModel() {
  const calls: { kind: string; input: unknown; opts: GenerateOptions }[] = [];
  const stream = (): TextStream => ({
    stats: Promise.resolve(undefined),
    async *[Symbol.asyncIterator]() {
      yield "Par";
      yield "is.";
    },
  });
  return {
    calls,
    chat: (input: unknown, opts: GenerateOptions = {}) => (calls.push({ kind: "chat", input, opts }), stream()),
    complete: (input: string, opts: GenerateOptions = {}) => (calls.push({ kind: "complete", input, opts }), stream()),
  };
}

describe("pipeline option mapping", () => {
  it("maps dtype and device to quant and device", () => {
    expect(toLoadOptions("m", { dtype: "q4f16", device: "webgpu" })).toMatchObject({ quant: "q4", device: "gpu" });
    expect(toLoadOptions("m", { dtype: "int8", device: "wasm" })).toMatchObject({ quant: "q8", device: "cpu" });
    expect(toLoadOptions("m", { dtype: "fp16" })).toMatchObject({ quant: "f32", device: "auto" });
    expect(toLoadOptions("m", { dtype: "fp16", quant: "q4" }).quant).toBe("q4");
    expect(toLoadOptions("m", {}).quant).toBeUndefined();
    expect(() => toLoadOptions("m", { dtype: "q2" })).toThrow(/unsupported dtype/);
  });

  it("reports progress as transformers.js does", () => {
    const seen: ProgressInfo[] = [];
    const opts = toLoadOptions("org/m", { progress_callback: (p) => seen.push(p) });
    opts.onProgress!({ file: "model.safetensors", loaded: 50, total: 200 } as never);
    opts.onProgress!({ file: "model.safetensors", loaded: 200, total: 200 } as never);
    expect(seen.map((p) => p.status)).toEqual(["initiate", "progress", "progress", "done"]);
    expect(seen[1]).toMatchObject({ name: "org/m", file: "model.safetensors", progress: 25, loaded: 50, total: 200 });
  });

  it("is greedy unless do_sample, with transformers' sampling defaults", () => {
    expect(toGenerateOptions({ max_new_tokens: 8 })).toEqual({ maxTokens: 8, temperature: 0 });
    expect(toGenerateOptions({ do_sample: true })).toEqual({ temperature: 1, topK: 50, topP: 1 });
    expect(toGenerateOptions({ do_sample: true, temperature: 0.6, top_k: 0, top_p: 0.9, repetition_penalty: 1.1 })).toEqual({
      temperature: 0.6,
      topK: 0,
      topP: 0.9,
      repetitionPenalty: 1.1,
    });
    // Kiln's own options pass through; streamer and return_full_text do not reach the model.
    const g = toGenerateOptions({ format: { type: "json" }, streamer: new TextStreamer(), return_full_text: false });
    expect(g).toEqual({ format: { type: "json" }, temperature: 0 });
  });
});

describe("text-generation calls", () => {
  it("completes a string, with the prompt unless return_full_text is false", async () => {
    const m = fakeModel();
    expect(await runTextGeneration(m, "The capital of France is ")).toEqual([{ generated_text: "The capital of France is Paris." }]);
    expect(await runTextGeneration(m, "Q: ", { return_full_text: false })).toEqual([{ generated_text: "Paris." }]);
    expect(m.calls.map((c) => c.kind)).toEqual(["complete", "complete"]);
  });

  it("chats, appending the reply to the messages", async () => {
    const m = fakeModel();
    const messages = [{ role: "user" as const, content: "Capital of France?" }];
    const out = await runTextGeneration(m, messages, { max_new_tokens: 4 });
    expect(out).toEqual([{ generated_text: [...messages, { role: "assistant", content: "Paris." }] }]);
    expect(m.calls[0]).toMatchObject({ kind: "chat", opts: { maxTokens: 4, temperature: 0 } });
  });

  it("runs an array of inputs one after another", async () => {
    const m = fakeModel();
    const out = await runTextGeneration(m, ["a ", [{ role: "user", content: "b" }]]);
    expect(out).toHaveLength(2);
    expect((out as { generated_text: unknown }[][])[0]![0]!.generated_text).toBe("a Paris.");
    expect(m.calls.map((c) => c.kind)).toEqual(["complete", "chat"]);
  });

  it("streams pieces to a TextStreamer", async () => {
    const pieces: string[] = [];
    const streamer = new TextStreamer({}, { skip_prompt: true, callback_function: (t) => pieces.push(t) });
    await runTextGeneration(fakeModel(), "x", { streamer });
    expect(pieces).toEqual(["Par", "is."]);
  });
});

describe("encoder pipeline helpers", () => {
  it("Tensor.tolist nests by dims", async () => {
    const { Tensor } = await import("../src/pipeline");
    const t = new Tensor("float32", Float32Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
    expect(t.tolist()).toEqual([[1, 2, 3], [4, 5, 6]]);
    expect(new Tensor("float32", Float32Array.from([1, 2, 3, 4]), [1, 2, 2]).tolist()).toEqual([[[1, 2], [3, 4]]]);
    expect(t.size).toBe(6);
  });

  it("cos_sim", async () => {
    const { cos_sim } = await import("../src/pipeline");
    expect(cos_sim([1, 0], [1, 0])).toBeCloseTo(1);
    expect(cos_sim([1, 0], [0, 1])).toBeCloseTo(0);
    expect(cos_sim([1, 2], [-1, -2])).toBeCloseTo(-1);
  });
});
