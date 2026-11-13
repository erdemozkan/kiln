import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpSource, accessToken, fetchBytes, fileUrl, httpError, hubFetch, parseModelId, sendsToken, setAccessToken } from "../src";

const TOKEN = "hf_testOnlyNotARealToken";

afterEach(() => {
  setAccessToken(undefined);
  vi.unstubAllGlobals();
});

/** A fetch that records each request's URL and Authorization header. */
function recorder(status = 206) {
  const seen: [string, string | undefined][] = [];
  vi.stubGlobal("fetch", async (url: string, init?: { headers?: Record<string, string> }) => {
    seen.push([url, init?.headers?.Authorization]);
    return new Response(Uint8Array.from([1, 2, 3, 4]), { status, headers: { "Content-Range": "bytes 0-3/4" } });
  });
  return seen;
}

describe("access token", () => {
  it("goes to the Hub over https and nowhere else", () => {
    expect(sendsToken("https://huggingface.co/org/model/resolve/main/config.json")).toBe(false); // none set
    setAccessToken(TOKEN);
    expect(accessToken()).toBe(TOKEN);
    expect(sendsToken("https://huggingface.co/org/model/resolve/main/config.json")).toBe(true);
    expect(sendsToken(fileUrl(parseModelId("org/model"), "model.safetensors"))).toBe(true);
    for (const url of [
      "http://huggingface.co/org/model/resolve/main/config.json", // not https
      "https://huggingface.co:8443/org/model/resolve/main/config.json", // another server on the name
      "https://hf-mirror.com/org/model/resolve/main/config.json", // a mirror given as endpoint
      "https://huggingface.co.example.com/org/model/resolve/main/config.json",
      "https://example.com/huggingface.co/model.gguf",
      "https://huggingface.co@example.com/model.gguf", // the Hub's name as a user name
      "https://cdn-lfs.huggingface.co/repos/ab/cd", // storage: its links are signed, they take no token
      "https://user-space.hf.space/model.gguf",
      "kiln-local://model/config.json",
      "not a url",
    ]) {
      expect(sendsToken(url), url).toBe(false);
    }
    setAccessToken(null);
    expect(sendsToken("https://huggingface.co/org/model/resolve/main/config.json")).toBe(false);
  });

  it("is added to requests for repo files, ranges kept", async () => {
    const seen = recorder();
    setAccessToken(TOKEN);
    await hubFetch("https://huggingface.co/org/model/resolve/main/a.json");
    await hubFetch("https://mirror.example/org/model/resolve/main/a.json", { headers: { Range: "bytes=0-3" } });
    expect(await new HttpSource("https://huggingface.co/org/model/resolve/main/w.bin").read(0, 4)).toEqual(Uint8Array.from([1, 2, 3, 4]));
    expect(await new HttpSource("https://files.example/w.gguf").read(0, 4)).toEqual(Uint8Array.from([1, 2, 3, 4]));
    expect(seen.map(([, auth]) => auth)).toEqual([`Bearer ${TOKEN}`, undefined, `Bearer ${TOKEN}`, undefined]);
    setAccessToken(undefined);
    await hubFetch("https://huggingface.co/org/model/resolve/main/a.json");
    expect(seen.at(-1)![1]).toBeUndefined();
  });

  it("refuses values that are not one header value", () => {
    for (const bad of ["a b", "line\nbreak", "tab\there", "x\r\nX-Other: 1", "é", "a".repeat(513), 5 as never]) expect(() => setAccessToken(bad), String(bad).slice(0, 12)).toThrow(/token/);
    expect(accessToken()).toBeUndefined();
  });

  it("says what to do when the Hub refuses, without the token in the message", async () => {
    const hub = "https://huggingface.co/meta/gated/resolve/main/config.json";
    expect(httpError(hub, 401)).toMatch(/gated or private model: set an access token with setAccessToken/);
    expect(httpError(hub, 404)).toBe(`${hub}: HTTP 404`);
    expect(httpError("https://mirror.example/a", 401)).toMatch(/gated or private model\?/);
    setAccessToken(TOKEN);
    expect(httpError(hub, 403)).toMatch(/does not open this model/);
    expect(httpError(hub, 403)).not.toContain(TOKEN);
    recorder(401);
    await expect(fetchBytes(hub)).rejects.toThrow(/does not open this model/);
    await expect(new HttpSource(hub).read(0, 4)).rejects.toThrow(/does not open this model/);
  });
});
