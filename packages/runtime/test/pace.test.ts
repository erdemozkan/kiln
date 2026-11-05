import { describe, expect, it } from "vitest";
import { SubmitPacer } from "../src/pace";

describe("SubmitPacer", () => {
  it("cuts work over budget into pieces, from the learned cost", () => {
    const p = new SubmitPacer(0.1);
    expect(p.pieces(400, 400, 50)).toBe(1); // 40 ms
    expect(p.pieces(1200, 400, 50)).toBe(3); // 120 ms
    expect(p.pieces(1e7, 400, 50)).toBe(400); // never more pieces than dispatches
    for (let i = 0; i < 20; i++) p.observe(12, 1200); // a fast GPU: 0.01 ms per unit
    expect(p.pieces(1200, 400, 50)).toBe(1);
  });

  it("learns a slower GPU quickly and a faster one cautiously", () => {
    const slow = new SubmitPacer(0.01);
    slow.observe(100, 1000); // 0.1 per unit
    const fast = new SubmitPacer(0.1);
    fast.observe(10, 1000); // 0.01 per unit
    // One slow sample moves the estimate half way (to 0.055), one fast sample a fifth of the way (0.082).
    expect(slow.pieces(1000, 1000, 10)).toBe(6);
    expect(fast.pieces(1000, 1000, 10)).toBe(9);
  });

  it("ignores empty or nonsense measurements", () => {
    const p = new SubmitPacer(0.1);
    p.observe(0, 100);
    p.observe(10, 0);
    p.observe(NaN, 100);
    expect(p.pieces(1000, 1000, 50)).toBe(2);
  });

  it("splits in order into nearly equal runs", () => {
    expect(SubmitPacer.split([1, 2, 3, 4, 5, 6, 7], 3)).toEqual([[1, 2], [3, 4], [5, 6, 7]]);
    expect(SubmitPacer.split([1, 2], 5)).toEqual([[1], [2]]);
    expect(SubmitPacer.split([1, 2, 3], 1)).toEqual([[1, 2, 3]]);
  });
});
