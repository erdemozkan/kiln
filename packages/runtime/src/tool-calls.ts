// Tool calls the model makes on its own (toolChoice "auto"), checked against the tools' schemas:
// text is generated freely until the model writes its template's call marker, the call itself is
// generated under a grammar, and free generation resumes after it.
import type { ResponseFormat } from "@kiln/grammar";
import type { GenerateOptions, GenerateStats } from "./engine";

export interface CallCheck {
  /** The token that opens a call (`<tool_call>`, or Granite's `<|tool_call|>`). */
  trigger: number;
  /** What must follow it. */
  format: ResponseFormat;
  /**
   * The token that closes a call, written after `format` is complete. Special tokens are not in a
   * grammar's vocabulary: left to the grammar, `</tool_call>` would be spelled out in ordinary tokens.
   */
  close?: number;
  /**
   * The tokens around a reasoning block (`<think>`, `</think>`). A marker written inside one is
   * the model thinking about a call, not making it: it stays text.
   */
  think?: { open: number; close: number };
}

/** What one generation call left behind: its stats and the stop token that ended it, if one did. */
export interface PhaseEnd {
  stats?: GenerateStats;
  stopId?: number;
}

function merge(total: GenerateStats | undefined, s: GenerateStats | undefined, extra: number): GenerateStats | undefined {
  if (!s) return total;
  if (!total) return { ...s, generatedTokens: s.generatedTokens + extra };
  return {
    ...total,
    generatedTokens: total.generatedTokens + s.generatedTokens + extra,
    // A later phase's prefill (the one or two tokens the cache lacks) is decode time of the reply.
    decodeMs: total.decodeMs + s.prefillMs + s.decodeMs,
    maskMs: total.maskMs === undefined && s.maskMs === undefined ? undefined : (total.maskMs ?? 0) + (s.maskMs ?? 0),
    submits: (total.submits ?? 0) + (s.submits ?? 0),
    stopReason: s.stopReason,
  };
}

/**
 * Yields the reply's tokens, markers included. `run` generates from prompt ids (the engine's
 * `generateTokens`; each call continues the last one's cache), `end` reads what the call just
 * finished left behind, and `done` receives the stats of the whole reply.
 */
export async function* checkedToolCalls(
  prompt: readonly number[],
  opts: GenerateOptions,
  check: CallCheck,
  run: (ids: readonly number[], opts: GenerateOptions) => AsyncGenerator<number>,
  end: () => PhaseEnd,
  done: (stats: GenerateStats | undefined, stopId: number | undefined) => void,
): AsyncGenerator<number> {
  // Each call gets a copy: the engine keeps its prompt until the call ends, and `ids` grows meanwhile.
  const ids = [...prompt];
  const free: GenerateOptions = { ...opts, tools: undefined, toolChoice: undefined, format: undefined, stopTokenIds: [...(opts.stopTokenIds ?? []), check.trigger] };
  const call: GenerateOptions = { ...opts, tools: undefined, toolChoice: undefined, format: check.format };
  let left = opts.maxTokens ?? 512;
  let total: GenerateStats | undefined;
  let stopId: number | undefined;
  // A prompt may itself end inside a reasoning block (templates that open one for the model).
  const think = check.think;
  let thinking = !!think && prompt.lastIndexOf(think.open) > prompt.lastIndexOf(think.close);
  try {
    for (;;) {
      let n = 0;
      for await (const id of run([...ids], { ...free, maxTokens: left })) {
        ids.push(id);
        n++;
        if (think && (id === think.open || id === think.close)) thinking = id === think.open;
        yield id;
      }
      left -= n;
      const e = end();
      const opened = e.stopId === check.trigger && e.stats?.stopReason === "stop";
      if (opened && thinking && left >= 2) {
        total = merge(total, e.stats, 1);
        ids.push(check.trigger);
        left--;
        yield check.trigger;
        continue;
      }
      // The marker needs room for a call after it; without it the reply ends as one cut short would.
      if (!opened || left < 2) {
        total = merge(total, e.stats, 0);
        stopId = opened ? undefined : e.stopId;
        if (opened && total) total.stopReason = "maxTokens";
        return;
      }
      total = merge(total, e.stats, 1);
      ids.push(check.trigger);
      left--;
      yield check.trigger;
      n = 0;
      for await (const id of run([...ids], { ...call, maxTokens: left })) {
        ids.push(id);
        n++;
        yield id;
      }
      left -= n;
      const c = end();
      total = merge(total, c.stats, 0);
      stopId = c.stopId;
      // After a complete call the model may end its turn, write text, or call again.
      if (c.stats?.stopReason !== "format") return;
      if (check.close !== undefined && left >= 1) {
        ids.push(check.close);
        left--;
        if (total) total.generatedTokens++;
        yield check.close;
      }
      if (left < 1) {
        if (total) total.stopReason = "maxTokens";
        return;
      }
    }
  } finally {
    done(total, stopId);
  }
}
