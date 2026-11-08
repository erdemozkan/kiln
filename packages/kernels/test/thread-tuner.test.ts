import { describe, expect, it } from "vitest";
import { ThreadTuner } from "../src/simd";

/** Runs `calls` matmuls of one shape whose single-thread and threaded times are given (ms). */
function drive(t: ThreadTuner, key: string, calls: number, oneMs: number, manyMs: number): boolean[] {
  const ways: boolean[] = [];
  for (let i = 0; i < calls; i++) {
    const threaded = t.choose(key);
    ways.push(threaded);
    t.record(key, threaded, threaded ? manyMs : oneMs);
  }
  return ways;
}

describe("ThreadTuner", () => {
  it("times each way twice, then keeps the faster", () => {
    const t = new ThreadTuner();
    const ways = drive(t, "a", 4, 1, 3);
    expect(ways.filter(Boolean)).toHaveLength(2);
    expect(t.choices).toEqual({ a: "one" });
    expect(drive(t, "a", 10, 1, 3).every((w) => !w)).toBe(true);

    const u = new ThreadTuner();
    drive(u, "b", 4, 3, 1);
    expect(u.choices).toEqual({ b: "threads" });
  });

  it("re-times the slower way less often while it stays slower", () => {
    const t = new ThreadTuner();
    drive(t, "a", 4, 1, 3);
    const ways = drive(t, "a", 16 + 32 + 64, 1, 3);
    // Probes at the 16th call, then 32 later, then 64 later.
    expect(ways.map((w, i) => (w ? i : -1)).filter((i) => i >= 0)).toEqual([15, 47, 111]);
  });

  it("switches when conditions change", () => {
    const t = new ThreadTuner();
    drive(t, "a", 4, 1, 3); // threads slow (a background page, say)
    expect(t.choices).toEqual({ a: "one" });
    drive(t, "a", 200, 3, 1); // now the helpers win: picked up at the next probes
    expect(t.choices).toEqual({ a: "threads" });
  });

  it("decides per shape", () => {
    const t = new ThreadTuner();
    drive(t, "decode", 4, 1, 2);
    drive(t, "prefill", 4, 8, 3);
    expect(t.choices).toEqual({ decode: "one", prefill: "threads" });
  });
});
