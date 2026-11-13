import { describe, expect, it } from "vitest";
import { CachedHttpSource, adoptLocalModel, cacheKey, fetchJSON, fetchText, fileUrl, isLocalModel, listRepoFiles, localModelFiles, parseModelId, registerLocalModel, releaseLocalModel, weightFiles } from "../src";

const file = (text: string, name: string, lastModified = 1) => new File([text], name, { lastModified });

describe("local models", () => {
  it("serves a folder's files like a repo's", async () => {
    const id = registerLocalModel(
      [
        ["my-model/config.json", file('{"model_type":"bert"}', "config.json")],
        ["my-model/tokenizer.json", file("{}", "tokenizer.json")],
        ["my-model/model.safetensors", file("0123456789", "model.safetensors")],
        ["my-model/onnx/model.onnx", file("x", "model.onnx")],
      ],
      "my model",
    );
    expect(isLocalModel(id)).toBe(true);
    expect(id).toMatch(/^@local\/my-model-[a-z0-9]+$/);
    const ref = parseModelId(id);
    expect(ref.gguf).toBeUndefined();
    // The shared leading folder is dropped; deeper folders stay.
    expect(await listRepoFiles(ref)).toEqual(new Set(["config.json", "tokenizer.json", "model.safetensors", "onnx/model.onnx"]));
    expect(await fetchJSON(fileUrl(ref, "config.json"))).toEqual({ model_type: "bert" });
    // A mirror endpoint and a revision do not apply to local files.
    expect(await fetchText(fileUrl({ ...ref, revision: "v2" }, "tokenizer.json", "https://mirror.example"))).toBe("{}");
    expect(await fetchJSON(fileUrl(ref, "generation_config.json"), true)).toBeUndefined();
    await expect(fetchJSON(fileUrl(ref, "generation_config.json"))).rejects.toThrow("the model folder has no generation_config.json");
    expect(await weightFiles(ref, undefined, await listRepoFiles(ref))).toEqual({ format: "safetensors", names: ["model.safetensors"] });
    const source = await CachedHttpSource.open(fileUrl(ref, "model.safetensors"), cacheKey(ref, "model.safetensors"));
    expect(await source.size()).toBe(10);
    expect(new TextDecoder().decode(await source.read(3, 4))).toBe("3456");
    await expect(CachedHttpSource.open(fileUrl(ref, "missing.bin"), "k")).rejects.toThrow("the model folder has no missing.bin");
  });

  it("gives a lone .gguf the id of that file", () => {
    const id = registerLocalModel([["SmolLM2-Q4_K_M.gguf", file("gguf", "SmolLM2-Q4_K_M.gguf")]], "SmolLM2-Q4_K_M");
    const ref = parseModelId(id);
    expect(ref.gguf).toBe("SmolLM2-Q4_K_M.gguf");
    expect(ref.repo).toMatch(/^@local\/SmolLM2-Q4_K_M-/);
    // A projector beside the model does not make the choice ambiguous.
    const two = registerLocalModel([
      ["m/model-Q4_0.gguf", file("a", "model-Q4_0.gguf")],
      ["m/mmproj-F16.gguf", file("b", "mmproj-F16.gguf")],
    ]);
    expect(parseModelId(two).gguf).toBe("model-Q4_0.gguf");
  });

  it("changes the id, and so every cache key, when a file changes", () => {
    const a = registerLocalModel([["config.json", file("{}", "config.json", 1)]], "m");
    const same = registerLocalModel([["config.json", file("{}", "config.json", 1)]], "m");
    const touched = registerLocalModel([["config.json", file("{}", "config.json", 2)]], "m");
    const grown = registerLocalModel([["config.json", file("{ }", "config.json", 1)]], "m");
    expect(same).toBe(a);
    expect(touched).not.toBe(a);
    expect(grown).not.toBe(a);
    expect(cacheKey(parseModelId(a), "weights", "https://mirror.example")).toBe(`${a}/weights`);
  });

  it("hands its files to another thread, and fails clearly once released", async () => {
    const id = registerLocalModel([["config.json", file('{"a":1}', "config.json")]], "handoff");
    const files = [...localModelFiles(id)!];
    releaseLocalModel(id);
    await expect(fetchJSON(fileUrl(parseModelId(id), "config.json"))).rejects.toThrow("is not registered in this page");
    adoptLocalModel(id, files);
    expect(await fetchJSON(fileUrl(parseModelId(id), "config.json"))).toEqual({ a: 1 });
  });

  it("rejects paths that climb out and an empty pick", () => {
    expect(() => registerLocalModel([])).toThrow("no files to load");
    const id = registerLocalModel([
      ["config.json", file("{}", "config.json")],
      ["../secret", file("x", "secret")],
    ]);
    expect([...localModelFiles(id)!.keys()]).toEqual(["config.json"]);
    expect(() => fileUrl(parseModelId(id), "../secret")).toThrow("is not a file path inside a repo");
  });
});
