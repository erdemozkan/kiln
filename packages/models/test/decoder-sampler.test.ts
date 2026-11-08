import { describe, expect, it } from "vitest";
import { CpuBackend } from "@kiln/kernels";
import { DecoderSampler } from "../src/seq2seq";

describe("DecoderSampler masks", () => {
  it("bans exactly the given ids and clears the flag when nothing is banned", async () => {
    const be = new CpuBackend({ threads: 1 });
    const s = new DecoderSampler(be, 70);
    s.prepare([3, 40, 69, 70, -1]);
    const m = new Uint32Array(await be.readback(s.allowed, 0, 4 * 4));
    expect(m[0]).toBe(1);
    const allowed = (i: number) => ((m[1 + (i >>> 5)]! >>> (i & 31)) & 1) === 1;
    expect([3, 40, 69].map(allowed)).toEqual([false, false, false]);
    expect([0, 4, 39, 41, 68].map(allowed)).toEqual([true, true, true, true, true]);
    s.prepare([]);
    expect(new Uint32Array(await be.readback(s.allowed, 0, 4))[0]).toBe(0);
  });

  it("marks seen ids for the repetition penalty", async () => {
    const be = new CpuBackend({ threads: 1 });
    const s = new DecoderSampler(be, 64);
    s.prepare([], [1, 33, 33]);
    const m = new Uint32Array(await be.readback(s.seen, 0, 8));
    expect(Array.from(m)).toEqual([2, 2]);
  });
});
