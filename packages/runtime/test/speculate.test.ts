import { describe, expect, it } from "vitest";
import { DraftGate, acceptedLength, lookupDraft } from "../src/speculate";

describe("lookupDraft", () => {
  it("proposes what followed the latest earlier occurrence of the suffix", () => {
    //            0  1  2  3  4  5  6  7  8  9
    const toks = [1, 2, 3, 4, 5, 9, 2, 3, 7, 1, 2, 3];
    // Suffix [1, 2, 3] occurs at 0, followed by 4, 5, 9.
    expect(lookupDraft(toks, 3)).toEqual([4, 5, 9]);
    expect(lookupDraft(toks, 10)).toEqual([4, 5, 9, 2, 3, 7, 1, 2, 3]);
  });

  it("prefers the longest matching suffix, then the most recent occurrence", () => {
    // [2, 3] occurs at 1 (then 4) and at 6 (then 7); [1, 2, 3] only at 0.
    expect(lookupDraft([1, 2, 3, 4, 0, 0, 2, 3, 7, 1, 2, 3], 1)).toEqual([4]);
    // Only [2, 3] matches; its latest occurrence is followed by 7.
    expect(lookupDraft([1, 2, 3, 4, 0, 0, 2, 3, 7, 8, 2, 3], 1)).toEqual([7]);
  });

  it("returns nothing without a match of at least minNgram tokens", () => {
    expect(lookupDraft([1, 2, 3, 4, 5], 4)).toEqual([]);
    expect(lookupDraft([5, 1, 6, 1], 4)).toEqual([]); // a single repeated token is too weak a cue
    expect(lookupDraft([5, 1, 6, 1], 4, 3, 1)).toEqual([6, 1]);
    expect(lookupDraft([], 4)).toEqual([]);
    expect(lookupDraft([1, 2, 1, 2], 0)).toEqual([]);
  });

  it("can draft from an overlapping repetition", () => {
    // Suffix [7, 7, 7] also starts at 1, followed by the suffix's own last tokens.
    expect(lookupDraft([7, 7, 7, 7, 7], 4)).toEqual([7]);
  });
});

describe("acceptedLength", () => {
  it("keeps the draft up to the first token the model chose differently", () => {
    expect(acceptedLength([4, 5, 6], [4, 5, 6, 9])).toBe(3);
    expect(acceptedLength([4, 5, 6], [4, 8, 6, 9])).toBe(1);
    expect(acceptedLength([4, 5, 6], [3, 5, 6, 9])).toBe(0);
    expect(acceptedLength([], [3])).toBe(0);
  });
});

describe("DraftGate", () => {
  it("measures plain steps first, then tries one pass", () => {
    const g = new DraftGate();
    expect(g.worth(7)).toBe(false);
    g.plain(96, 4, true);
    expect(g.worth(7)).toBe(true);
  });

  it("keeps verifying while passes pay for themselves", () => {
    const g = new DraftGate();
    g.plain(96, 4, false); // 24 ms a token
    g.verified(83, 7, 7); // 8 tokens for 3.5 tokens' time
    expect(g.worth(7)).toBe(true);
    expect(g.worth(1)).toBe(false); // 2 tokens cannot pay for 3.5
  });

  it("stops verifying when drafts are rejected, and tries again later", () => {
    const g = new DraftGate();
    g.plain(96, 4, false);
    for (let i = 0; i < 4; i++) g.verified(83, 7, 0);
    expect(g.worth(7)).toBe(false);
    let skips = 0;
    while (!g.worth(7) && skips < 100) {
      g.plain(96, 4, true);
      skips++;
    }
    // About 8 plain chunks (32 tokens) before the next try.
    expect(skips).toBeGreaterThan(4);
    expect(skips).toBeLessThan(16);
  });
});
