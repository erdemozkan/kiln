import { describe, expect, it } from "vitest";
import { CpuBackend } from "@kiln/kernels";
import { LlamaModel } from "@kiln/models";
import { Engine } from "../src/engine";
import { TINY, tinyTokenizer, tinyWeights } from "./tiny-model";

const engine = () => {
  const be = new CpuBackend({ threads: 1 });
  return new Engine(be, new LlamaModel(be, TINY, tinyWeights(be), { maxSeq: 128, kvDtype: "f32" }), tinyTokenizer());
};
const prompt = (n: number, k = 0) => Array.from({ length: n }, (_, i) => 1 + ((i * 13 + k) % 90));

async function run(e: Engine, ids: number[], opts: { maxTokens?: number; signal?: AbortSignal } = {}) {
  const out: number[] = [];
  for await (const id of e.generateTokens(ids, { maxTokens: 4, temperature: 0, ...opts })) out.push(id);
  return out;
}

describe("Engine calls", () => {
  it("skip a prompt whose call was aborted before it started, and leave the cache as it was", async () => {
    const e = engine();
    const a = prompt(40);
    const reply = await run(e, a);
    const aborted = new AbortController();
    aborted.abort();
    expect(await run(e, prompt(40, 5), { signal: aborted.signal })).toEqual([]);
    expect(e.lastStats).toMatchObject({ generatedTokens: 0, cachedTokens: 0, submits: 0, stopReason: "abort" });
    // The next turn still resumes the first call's prompt and reply.
    await run(e, [...a, ...reply.slice(0, 3), 7]);
    expect(e.lastStats?.cachedTokens).toBe(a.length + 3);
  });

  it("stay usable after a failure while the cache is being prepared", async () => {
    const e = engine();
    await run(e, prompt(40));
    // Saving the live cache before another prompt overwrites it fails.
    const m = e.model as unknown as { saveKv(n: number): unknown };
    const saveKv = m.saveKv;
    m.saveKv = () => {
      throw new Error("out of memory");
    };
    await expect(run(e, prompt(40, 5))).rejects.toThrow(/out of memory/);
    m.saveKv = saveKv;
    const b = prompt(20, 9);
    expect(await run(e, b)).toEqual(await run(engine(), b));
    expect(e.lastStats?.cachedTokens).toBe(0);
  });
});
