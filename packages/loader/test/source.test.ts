import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpSource } from "../src";

const ok = (bytes: number[]) => new Response(Uint8Array.from(bytes), { status: 206, headers: { "Content-Range": `bytes 0-${bytes.length - 1}/10` } });

describe("HttpSource", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("retries network errors and server errors, then reads", async () => {
    const fetch = vi.fn()
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockResolvedValueOnce(new Response(null, { status: 503 }))
      .mockResolvedValueOnce(ok([1, 2, 3]));
    vi.stubGlobal("fetch", fetch);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const src = new HttpSource("https://example.test/f", "f", 1);
    expect(Array.from(await src.read(0, 3))).toEqual([1, 2, 3]);
    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it("does not retry a 404", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(null, { status: 404 }));
    vi.stubGlobal("fetch", fetch);
    await expect(new HttpSource("https://example.test/f", "f", 1).read(0, 3)).rejects.toThrow(/HTTP 404/);
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});
