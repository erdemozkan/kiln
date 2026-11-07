import { describe, expect, it } from "vitest";
import { PrefixStore, commonPrefix } from "../src/prefix-store";

const seq = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i);

describe("PrefixStore", () => {
  it("finds the saved prefix sharing the most tokens, one short of the prompt", () => {
    const freed: string[] = [];
    const s = new PrefixStore<string>(1000, (d) => freed.push(d));
    s.add([1, 2, 3, 4], 4, 10, () => "a");
    s.add([1, 2, 9, 9, 9], 5, 10, () => "b");
    expect(s.best([1, 2, 3, 4, 5], 0, () => 0)).toMatchObject({ entry: { data: "a" }, length: 4 });
    // The prompt's last token is always prefilled, so a full match reuses all but it.
    expect(s.best([1, 2, 3, 4], 0, () => 0)?.length).toBe(3);
    expect(s.best([1, 2, 9, 9, 9, 7], 0, () => 0)?.entry.data).toBe("b");
    // Only when it beats what the live cache already shares.
    expect(s.best([1, 2, 3, 4, 5], 4, () => 0)).toBeUndefined();
    // Ring caches: a prefix whose resume point is below minReuseStart(written) is unusable.
    expect(s.best([1, 2, 3, 4, 5], 0, (w) => (w === 4 ? 5 : 0))?.entry.data).toBe("b");
    expect(freed).toEqual([]);
  });

  it("drops prefixes a new one extends, skips ones already covered", () => {
    const freed: string[] = [];
    const s = new PrefixStore<string>(1000, (d) => freed.push(d));
    s.add([1, 2], 2, 10, () => "short");
    expect(s.add([1, 2, 3], 3, 10, () => "long")).toBe(true);
    expect(freed).toEqual(["short"]);
    expect(s.add([1, 2], 2, 10, () => "again")).toBe(false);
    expect(s.covers([1, 2, 3])).toBe(true);
    expect(s.size).toBe(1);
  });

  it("evicts the least recently used to stay within the budget", () => {
    const freed: string[] = [];
    const s = new PrefixStore<string>(30, (d) => freed.push(d));
    s.add(seq(5, 100), 5, 10, () => "a");
    s.add(seq(5, 200), 5, 10, () => "b");
    s.add(seq(5, 300), 5, 10, () => "c");
    s.best([...seq(5, 100), 1], 0, () => 0); // touch a
    s.add(seq(5, 400), 5, 10, () => "d");
    expect(freed).toEqual(["b"]);
    expect(s.bytes).toBe(30);
    // Larger than the whole budget: not kept, nothing evicted for it.
    expect(s.add(seq(5, 500), 5, 31, () => "huge")).toBe(false);
    expect(freed).toEqual(["b"]);
    s.clear();
    expect(freed.sort()).toEqual(["a", "b", "c", "d"]);
  });

  it("commonPrefix", () => {
    expect(commonPrefix([1, 2, 3], [1, 2, 4])).toBe(2);
    expect(commonPrefix([], [1])).toBe(0);
    expect(commonPrefix([5, 6], [5, 6])).toBe(2);
  });
});

describe("imageSpans", () => {
  it("finds runs of ids past the vocabulary (image tokens) as [start, end)", async () => {
    const { imageSpans } = await import("../src/engine");
    expect(imageSpans([1, 2, 100, 101, 102, 3, 105, 4], 100)).toEqual([[2, 5], [6, 7]]);
    expect(imageSpans([1, 2, 3], 100)).toEqual([]);
    expect(imageSpans([200, 201], 100)).toEqual([[0, 2]]);
  });
});
