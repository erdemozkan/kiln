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
function fake(script: number[]) {
  const calls: { prompt: number[]; opts: GenerateOptions }[] = [];
  let last: PhaseEnd = {};
  const promptLength = 2;
  async function* run(ids: readonly number[], opts: GenerateOptions): AsyncGenerator<number> {
    calls.push({ prompt: [...ids], opts });
    const given = ids.length;
    const stop = new Set([EOS, ...(opts.stopTokenIds ?? [])]);
    const stats: GenerateStats = { promptTokens: ids.length, cachedTokens: 0, generatedTokens: 0, prefillMs: 1, decodeMs: 2, ttftMs: 1 };
    let at = ids.length - promptLength;
    let stopId: number | undefined;
    for (;;) {
      if (stats.generatedTokens >= (opts.maxTokens ?? 512)) {
        stats.stopReason = "maxTokens";
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
    last = { stats, stopId };
  }
  return { run, end: () => last, calls };
}

async function reply(script: number[], opts: GenerateOptions = {}) {
  const m = fake(script);
  const out: number[] = [];
  let stats: GenerateStats | undefined;
  let stopId: number | undefined;
  for await (const id of checkedToolCalls([1, 2], opts, CHECK, m.run, m.end, (s, id) => ((stats = s), (stopId = id)))) out.push(id);
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

  it("keeps the caller's stop tokens in both phases", async () => {
    const r = await reply([5, 44, 6], { stopTokenIds: [44] });
    expect(r.out).toEqual([5]);
    expect(r.stopId).toBe(44);
    expect(r.calls[0]!.opts.stopTokenIds).toEqual([44, TRIGGER]);
  });
});
