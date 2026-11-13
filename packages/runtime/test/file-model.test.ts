import { describe, expect, it } from "vitest";
import { CpuBackend } from "@kiln/kernels";
import { createEngine, isFileModel, modelKey, modelName } from "../src/load";

/** A File as the browser hands one over (Node's Blob has no name or time). */
function file(bytes: Uint8Array, name: string, lastModified: number): Blob {
  return Object.assign(new Blob([bytes]), { name, lastModified });
}

describe("file models are told from Hub ids", () => {
  it("knows a picked file and a URL from a repo id", () => {
    expect(isFileModel("org/repo")).toBe(false);
    expect(isFileModel("org/repo/model.gguf")).toBe(false);
    expect(isFileModel("https://example.com/models/m.gguf")).toBe(true);
    expect(isFileModel(new Blob([]))).toBe(true);
  });

  it("shows an id or URL as given and a file by its name", () => {
    expect(modelName("org/repo")).toBe("org/repo");
    expect(modelName("https://example.com/a/m.gguf")).toBe("https://example.com/a/m.gguf");
    expect(modelName(file(new Uint8Array(4), "tiny.gguf", 7))).toBe("tiny.gguf");
    expect(modelName(new Blob([]))).toBe("model.gguf");
  });

  it("keys caches by the file itself, and a URL by a short hash", () => {
    expect(modelKey("org/repo")).toBe("org/repo");
    expect(modelKey(file(new Uint8Array(4), "tiny.gguf", 7))).toBe("file|tiny.gguf|4|7");
    // Same name, different bytes or time: a different file.
    expect(modelKey(file(new Uint8Array(5), "tiny.gguf", 7))).not.toBe(modelKey(file(new Uint8Array(4), "tiny.gguf", 7)));
    expect(modelKey(file(new Uint8Array(4), "tiny.gguf", 8))).not.toBe(modelKey(file(new Uint8Array(4), "tiny.gguf", 7)));
    const long = `https://example.com/${"a/".repeat(300)}my%20model.gguf?token=1`;
    const key = modelKey(long);
    expect(key).toMatch(/^url\|[0-9a-f]{16}\|my model\.gguf$/);
    expect(modelKey(`${long}2`)).not.toBe(key);
    expect(modelKey(long)).toBe(key);
  });
});

describe("loading a file model", () => {
  const backend = new CpuBackend();

  it("refuses a URL that is not a .gguf file, before any request", async () => {
    await expect(createEngine("https://example.com/org/repo/model.safetensors", { backend })).rejects.toThrow("only a .gguf file can be loaded from a URL");
  });

  it("says what is wrong with a picked file that is not GGUF", async () => {
    const zip = file(new TextEncoder().encode("PK\u0003\u0004 not a model, padded so the header read has bytes to look at ........"), "weights.safetensors", 1);
    await expect(createEngine(zip, { backend })).rejects.toThrow(/^weights\.safetensors: .*\(only \.gguf files can be loaded from a file\)$/);
  });
});
