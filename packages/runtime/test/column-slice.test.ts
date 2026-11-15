import { describe, expect, it } from "vitest";
import { columnSlice } from "../src/weights";

describe("columnSlice", () => {
  const rows = (n: number, whole: number) => Uint8Array.from({ length: n * whole }, (_, i) => (i * 37 + 11) & 255);
  const slow = (bytes: Uint8Array, n: number, whole: number, part: number, g: number) => {
    const out = new Uint8Array(n * part);
    for (let i = 0; i < n; i++) out.set(bytes.subarray(i * whole + g * part, i * whole + (g + 1) * part), i * part);
    return out;
  };

  it("cuts out one group of columns, word by word or byte by byte", () => {
    // Q5_0 groups of 8 blocks (176 bytes, whole words) and Q8_0-like groups of 34 bytes (not).
    for (const [part, groups] of [[176, 35], [34, 6], [8, 3]] as const) {
      const whole = part * groups;
      const n = 7;
      const bytes = rows(n, whole);
      for (const g of [0, 1, groups - 1]) expect([...columnSlice(bytes, n, whole, part, g)]).toEqual([...slow(bytes, n, whole, part, g)]);
      // A view that does not start on a word boundary takes the byte path and gives the same.
      const padded = new Uint8Array(bytes.length + 1);
      padded.set(bytes, 1);
      const view = padded.subarray(1);
      expect([...columnSlice(view, n, whole, part, 1)]).toEqual([...slow(bytes, n, whole, part, 1)]);
    }
  });
});
