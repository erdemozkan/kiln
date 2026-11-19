import { describe, expect, it } from "vitest";
import type { GenerateOptions, GenerateStats } from "../src/engine";
import { checkedToolCalls, type CallCheck, type PhaseEnd } from "../src/tool-calls";

const TRIGGER = 90;
const EOS = 0;
const CHECK: CallCheck = { trigger: TRIGGER, format: { type: "tool_call", syntax: "tags_body", tools: [{ name: "f" }] } };

/**
 * A model that replays `script` after the prompt: it stops at a stop token (as the engine does,
 * without yielding it), and under a format it ends once CLOSE has been written.
 */
const CLOSE = 99;
/** `context`: the engine's maxSeq, which a prompt must stay under and generation stops at. */
function fake(script: number[], context = Infinity) {
  const calls: { prompt: number[]; opts: GenerateOptions }[] = [];
  let last: PhaseEnd = {};
  const promptLength = 2;
  async function* run(ids: readonly number[], opts: GenerateOptions): AsyncGenerator<number> {
    if (ids.length >= context) throw new Error(`prompt has ${ids.length} tokens but the context is ${context}`);
    calls.push({ prompt: [...ids], opts });
    const given = ids.length;
    const stop = new Set([EOS, ...(opts.stopTokenIds ?? [])]);
    const stats: GenerateStats = { promptTokens: ids.length, cachedTokens: 0, generatedTokens: 0, prefillMs: 1, decodeMs: 2, ttftMs: 1, speculation: { passes: 1, drafted: 3, accepted: 2 } };
    let at = ids.length - promptLength;
    let stopId: number | undefined;
    const requested = opts.maxTokens ?? 512;
    const max = Math.min(requested, context - ids.length);
    try {
      for (;;) {
        if (stats.generatedTokens >= max) {
          stats.stopReason = max < requested ? "context" : "maxTokens";
          break;
        }
        const id = script[at++] ?? EOS;
        if (stop.has(id)) {
          stats.stopReason = "stop";
          stopId = id;
          break;
        }
        stats.generatedTokens++;
        yield id;
        if (opts.format && id === CLOSE) {
          stats.stopReason = "format";
          break;
        }
      }
      // The engine reads its prompt again when the call ends: it must not have grown meanwhile.
      if (ids.length !== given) throw new Error("the prompt changed during the call");
    } finally {
      // As the engine does, also when the caller stops reading.
      last = { stats: { ...stats, stopReason: stats.stopReason ?? "abort" }, stopId };
    }
  }
  return { run, end: () => last, calls };
}

async function reply(script: number[], opts: GenerateOptions = {}, check: CallCheck = CHECK, stopAfter = Infinity) {
  const m = fake(script, check.context);
  const out: number[] = [];
  let stats: GenerateStats | undefined;
  let stopId: number | undefined;
  for await (const id of checkedToolCalls([1, 2], opts, check, m.run, m.end, (s, id) => ((stats = s), (stopId = id)))) {
    out.push(id);
    if (out.length >= stopAfter) break;
  }
  return { out, stats, stopId, calls: m.calls };
}

describe("checked tool calls", () => {
  it("is one free generation when the model calls nothing", async () => {
    const r = await reply([5, 6, 7]);
    expect(r.out).toEqual([5, 6, 7]);
    expect(r.calls.length).toBe(1);
    expect(r.calls[0]!.opts.format).toBeUndefined();
    expect(r.calls[0]!.opts.stopTokenIds).toEqual([TRIGGER]);
    expect(r.stats).toMatchObject({ generatedTokens: 3, stopReason: "stop" });
    expect(r.stopId).toBe(EOS);
  });

  it("generates each call under the grammar, after the marker, and goes on freely", async () => {
    const r = await reply([5, TRIGGER, 10, 11, CLOSE, TRIGGER, 12, CLOSE]);
    expect(r.out).toEqual([5, TRIGGER, 10, 11, CLOSE, TRIGGER, 12, CLOSE]);
    expect(r.calls.map((c) => !!c.opts.format)).toEqual([false, true, false, true, false]);
    // The call continues from the prompt, the text so far and the marker.
    expect(r.calls[1]!.prompt).toEqual([1, 2, 5, TRIGGER]);
    expect(r.calls[1]!.opts.stopTokenIds).toBeUndefined();
    expect(r.calls[2]!.prompt).toEqual([1, 2, 5, TRIGGER, 10, 11, CLOSE]);
    expect(r.stats).toMatchObject({ generatedTokens: 8, promptTokens: 2, stopReason: "stop" });
    expect(r.stopId).toBe(EOS);
  });

  it("shares maxTokens across the phases", async () => {
    const r = await reply([5, TRIGGER, 10, 11, 12, 13, CLOSE], { maxTokens: 4 });
    expect(r.out).toEqual([5, TRIGGER, 10, 11]);
    expect(r.calls[1]!.opts.maxTokens).toBe(2);
    expect(r.stats).toMatchObject({ generatedTokens: 4, stopReason: "maxTokens" });
    // No room for a call after the marker: the marker is left out.
    const cut = await reply([5, 6, TRIGGER, 10, CLOSE], { maxTokens: 3 });
    expect(cut.out).toEqual([5, 6]);
    expect(cut.stats?.stopReason).toBe("maxTokens");
    expect(cut.stopId).toBeUndefined();
  });

  it("writes the closing token itself when the template has one", async () => {
    const m = fake([5, TRIGGER, 10, CLOSE, 98, 7]);
    const out: number[] = [];
    let stats: GenerateStats | undefined;
    for await (const id of checkedToolCalls([1, 2], {}, { ...CHECK, close: 98 }, m.run, m.end, (s) => (stats = s))) out.push(id);
    expect(out).toEqual([5, TRIGGER, 10, CLOSE, 98, 7]);
    expect(m.calls[2]!.prompt).toEqual([1, 2, 5, TRIGGER, 10, CLOSE, 98]);
    expect(stats?.generatedTokens).toBe(6);
  });

  it("leaves a marker inside a reasoning block as text", async () => {
    const think = { open: 70, close: 71 };
    const run = async (script: number[], prompt = [1, 2]) => {
      const m = fake(script);
      const out: number[] = [];
      for await (const id of checkedToolCalls(prompt, {}, { ...CHECK, think }, m.run, m.end, () => {})) out.push(id);
      return { out, formats: m.calls.map((c) => !!c.opts.format) };
    };
    const r = await run([70, 5, TRIGGER, 6, 71, TRIGGER, 10, CLOSE]);
    expect(r.out).toEqual([70, 5, TRIGGER, 6, 71, TRIGGER, 10, CLOSE]);
    expect(r.formats).toEqual([false, false, true, false]);
    // The prompt can open the block itself.
    const p = await run([TRIGGER, 71, 7], [1, 70]);
    expect(p.out).toEqual([TRIGGER, 71, 7]);
    expect(p.formats).toEqual([false, false]);
  });

  it("keeps the caller's stop tokens in both phases", async () => {
    const r = await reply([5, 44, 6], { stopTokenIds: [44] });
    expect(r.out).toEqual([5]);
    expect(r.stopId).toBe(44);
    expect(r.calls[0]!.opts.stopTokenIds).toEqual([44, TRIGGER]);
  });

  it("ends the reply at the context's end instead of starting a phase that cannot fit", async () => {
    // Prompt 2 + text 1: the marker and a call need two more positions than the context's 4 leave.
    const r = await reply([5, TRIGGER, 10, CLOSE], {}, { ...CHECK, context: 4 });
    expect(r.out).toEqual([5]);
    expect(r.stats).toMatchObject({ generatedTokens: 1, stopReason: "context" });
    expect(r.stopId).toBeUndefined();
    // A call that ends with its closing token at the last position: no free phase follows.
    const c = await reply([5, TRIGGER, 10, CLOSE, 7], {}, { ...CHECK, close: 98, context: 7 });
    expect(c.out).toEqual([5, TRIGGER, 10, CLOSE, 98]);
    expect(c.stats).toMatchObject({ generatedTokens: 5, stopReason: "context" });
    // A call cut by the context is a reply cut by the context.
    const d = await reply([5, TRIGGER, 10, 11, 12, CLOSE], {}, { ...CHECK, context: 6 });
    expect(d.out).toEqual([5, TRIGGER, 10, 11]);
    expect(d.stats).toMatchObject({ generatedTokens: 4, stopReason: "context" });
  });

  it("reports the phase the consumer stopped reading in", async () => {
    const r = await reply([5, TRIGGER, 10, 11, 12, CLOSE], {}, CHECK, 4);
    expect(r.out).toEqual([5, TRIGGER, 10, 11]);
    expect(r.stats).toMatchObject({ generatedTokens: 4, stopReason: "abort" });
    expect(r.stopId).toBeUndefined();
    // Stopped at the marker the driver wrote itself.
    const m = await reply([5, TRIGGER, 10, CLOSE], {}, CHECK, 2);
    expect(m.stats).toMatchObject({ generatedTokens: 2, stopReason: "abort" });
  });

  it("adds up speculation over the phases", async () => {
    const r = await reply([5, TRIGGER, 10, CLOSE, 6]);
    expect(r.calls.length).toBe(3);
    expect(r.stats?.speculation).toEqual({ passes: 3, drafted: 9, accepted: 6 });
  });
});
