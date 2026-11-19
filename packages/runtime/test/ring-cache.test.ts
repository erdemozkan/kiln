import { describe, expect, it } from "vitest";
import { CpuBackend } from "@kiln/kernels";
import { LlamaModel, type LlamaConfig } from "@kiln/models";
import { Engine } from "../src/engine";
import { TINY, rand, tinyTokenizer, tinyWeights } from "./tiny-model";

// Both layers slide over 8 positions: rings of max(256, 8 + 256) = 264 slots.
const CFG: LlamaConfig = { ...TINY, slidingWindow: 8, slidingLayers: [true, true] };

const engine = (be: CpuBackend) => new Engine(be, new LlamaModel(be, CFG, tinyWeights(be, CFG), { maxSeq: 1024, kvDtype: "f32" }), tinyTokenizer());

async function run(e: Engine, ids: number[], maxTokens: number, reuseCache = true) {
  const out: number[] = [];
  for await (const id of e.generateTokens(ids, { maxTokens, temperature: 0, reuseCache, draftTokens: 0 })) out.push(id);
  return { out, cached: e.lastStats!.cachedTokens };
}

describe("ring caches", () => {
  it("do not resume a prefix that an earlier, longer call's later positions overwrote", async () => {
    const be = new CpuBackend({ threads: 1 });
    const e = engine(be);
    const A = Array.from(rand(500, 99, 1), (v) => 1 + Math.floor((v + 1) * 47));
    // A writes up to position 520; B resumes A's first 400 tokens and ends near 410, so the ring
    // slots of positions 410…519 still hold A's, which overwrote positions 146…255.
    await run(e, A, 20);
    expect((await run(e, [...A.slice(0, 400), 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 2)).cached).toBe(400);
    const C = [...A.slice(0, 200), 20, 21, 22, 23, 24, 25, 26, 27, 28, 29];
    const c = await run(e, C, 8);
    const ref = await run(engine(be), C, 8, false);
    expect(c.cached).toBe(0);
    expect(c.out).toEqual(ref.out);
  }, 120_000);

  it("still resume a prefix the ring holds", async () => {
    const be = new CpuBackend({ threads: 1 });
    const e = engine(be);
    const A = Array.from(rand(300, 5, 1), (v) => 1 + Math.floor((v + 1) * 47));
    await run(e, A, 4);
    const B = [...A.slice(0, 280), 7, 8, 9];
    const b = await run(e, B, 6);
    expect(b.cached).toBe(280);
    expect(b.out).toEqual((await run(engine(be), B, 6, false)).out);
  }, 120_000);
});
