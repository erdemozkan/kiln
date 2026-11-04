import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { listRepoFiles, parseModelId, safetensorsFiles } from "../src";

const ref = parseModelId("org/model");
const requests: string[] = [];

/** Answers each URL from `routes` by path suffix; anything else is a 404. */
function serve(routes: Record<string, unknown>) {
  vi.stubGlobal("fetch", async (url: string) => {
    requests.push(url);
    const hit = Object.entries(routes).find(([suffix]) => url.endsWith(suffix));
    if (!hit) return { ok: false, status: 404, json: async () => ({}) };
    if (hit[1] instanceof Error) throw hit[1];
    return { ok: true, status: 200, json: async () => hit[1] };
  });
}

beforeEach(() => {
  requests.length = 0;
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("listRepoFiles", () => {
  it("reads the file names from the Hub API", async () => {
    serve({ "/api/models/org/model/revision/main?expand%5B%5D=siblings": { siblings: [{ rfilename: "config.json" }, { rfilename: "model.safetensors" }] } });
    expect(await listRepoFiles(ref)).toEqual(new Set(["config.json", "model.safetensors"]));
    expect(requests).toEqual(["https://huggingface.co/api/models/org/model/revision/main?expand%5B%5D=siblings"]);
  });

  it("is undefined where the endpoint has no API or the request fails", async () => {
    serve({});
    expect(await listRepoFiles(ref, "http://localhost:8080")).toBeUndefined();
    serve({ "siblings": new TypeError("Failed to fetch") });
    expect(await listRepoFiles(ref)).toBeUndefined();
  });
});

describe("safetensorsFiles", () => {
  const index = { weight_map: { a: "model-00002-of-00002.safetensors", b: "model-00001-of-00002.safetensors", c: "model-00001-of-00002.safetensors" } };

  it("does not request the index when the listing has none", async () => {
    serve({});
    expect(await safetensorsFiles(ref, undefined, new Set(["model.safetensors"]))).toEqual(["model.safetensors"]);
    expect(requests).toEqual([]);
  });

  it("reads the shards from the index when the listing has it", async () => {
    serve({ "model.safetensors.index.json": index });
    expect(await safetensorsFiles(ref, undefined, new Set(["model.safetensors.index.json"]))).toEqual(["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]);
  });

  it("probes for the index without a listing", async () => {
    serve({});
    expect(await safetensorsFiles(ref)).toEqual(["model.safetensors"]);
    expect(requests).toHaveLength(1);
  });
});
