import { describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fetchText, fileUrl, listRepoFiles, parseModelId } from "@kiln/loader";
import { env } from "../src/compat";
import { isDiskPath, localModel, resolveModel } from "../src/local";

const file = (name: string, path = "") => Object.assign(new File(["x"], name, { lastModified: 1 }), path ? { webkitRelativePath: path } : {});
const fileHandle = (name: string) => ({ kind: "file" as const, name, getFile: async () => file(name) });
const dirHandle = (name: string, entries: unknown[]): unknown => ({
  kind: "directory" as const,
  name,
  async *values() {
    yield* entries;
  },
});
const names = async (id: string) => [...(await listRepoFiles(parseModelId(id)))!].sort();

describe("localModel", () => {
  it("reads the files of a folder input", async () => {
    const id = await localModel([file("config.json", "bert-tiny/config.json"), file("model.safetensors", "bert-tiny/model.safetensors"), file("vocab.txt", "bert-tiny/vocab.txt")]);
    expect(id).toMatch(/^@local\/bert-tiny-/);
    expect(await names(id)).toEqual(["config.json", "model.safetensors", "vocab.txt"]);
  });

  it("walks a directory handle, skipping hidden entries", async () => {
    const dir = dirHandle("whisper-tiny", [fileHandle("config.json"), fileHandle(".DS_Store"), dirHandle(".git", [fileHandle("HEAD")]), dirHandle("onnx", [fileHandle("encoder.onnx")])]);
    const id = await localModel(dir as FileSystemHandle);
    expect(id).toMatch(/^@local\/whisper-tiny-/);
    expect(await names(id)).toEqual(["config.json", "onnx/encoder.onnx"]);
  });

  it("takes one .gguf as a File, a handle or a map", async () => {
    const a = await localModel(file("tiny-Q4_K_M.gguf"));
    expect(a).toMatch(/^@local\/tiny-Q4_K_M-[a-z0-9]+\/tiny-Q4_K_M\.gguf$/);
    expect(await localModel(fileHandle("tiny-Q4_K_M.gguf") as unknown as FileSystemHandle)).toBe(a);
    expect(await localModel({ "tiny-Q4_K_M.gguf": file("tiny-Q4_K_M.gguf") }, { name: "tiny-Q4_K_M" })).toBe(a);
    expect(await localModel(new Map([["tiny-Q4_K_M.gguf", file("tiny-Q4_K_M.gguf")]]), { name: "other" })).not.toBe(a);
  });
});

describe("localModel in Node.js", () => {
  const folder = () => {
    const root = mkdtempSync(join(tmpdir(), "kiln-local-"));
    const dir = join(root, "bert-tiny");
    mkdirSync(join(dir, "1_Pooling"), { recursive: true });
    mkdirSync(join(dir, ".git"));
    writeFileSync(join(dir, "config.json"), '{"model_type":"bert"}');
    writeFileSync(join(dir, "1_Pooling", "config.json"), "{}");
    writeFileSync(join(dir, ".git", "HEAD"), "x");
    writeFileSync(join(root, "tiny-Q8_0.gguf"), "GGUF");
    symlinkSync(join(dir, "config.json"), join(dir, "linked.json"));
    return { root, dir };
  };

  it("reads a folder on disk in place, following links and skipping hidden entries", async () => {
    const { root, dir } = folder();
    try {
      const id = await localModel(dir);
      expect(id).toMatch(/^@local\/bert-tiny-/);
      expect(await names(id)).toEqual(["1_Pooling/config.json", "config.json", "linked.json"]);
      const ref = parseModelId(id);
      expect(await fetchText(fileUrl(ref, "config.json"))).toBe('{"model_type":"bert"}');
      expect(await fetchText(fileUrl(ref, "linked.json"))).toBe('{"model_type":"bert"}');
      // The same files give the same id; an edited file a new one.
      expect(await localModel(dir)).toBe(id);
      writeFileSync(join(dir, "config.json"), '{"model_type":"bert","x":1}');
      expect(await localModel(dir)).not.toBe(id);
    } finally {
      rmSync(root, { recursive: true });
    }
  });

  it("takes one .gguf path, and refuses other files and missing paths", async () => {
    const { root, dir } = folder();
    try {
      expect(await localModel(join(root, "tiny-Q8_0.gguf"))).toMatch(/^@local\/tiny-Q8_0-[a-z0-9]+\/tiny-Q8_0\.gguf$/);
      await expect(localModel(join(dir, "config.json"))).rejects.toThrow(/is a file/);
      await expect(localModel(join(root, "nope"))).rejects.toThrow(/does not exist/);
    } finally {
      rmSync(root, { recursive: true });
    }
  });

  it("resolves paths, and repo names under env.localModelPath when allowed", async () => {
    const { root, dir } = folder();
    const saved = { allow: env.allowLocalModels, path: env.localModelPath };
    try {
      expect(await resolveModel(dir)).toMatch(/^@local\/bert-tiny-/);
      expect(await resolveModel("Xenova/bert-tiny")).toBe("Xenova/bert-tiny");
      expect(await resolveModel("bert-tiny")).toBe("bert-tiny");
      env.allowLocalModels = true;
      env.localModelPath = root;
      expect(await resolveModel("bert-tiny")).toMatch(/^@local\/bert-tiny-/);
      expect(await resolveModel("Xenova/bert-tiny")).toBe("Xenova/bert-tiny");
    } finally {
      env.allowLocalModels = saved.allow;
      env.localModelPath = saved.path;
      rmSync(root, { recursive: true });
    }
  });

  it("tells disk paths from Hub ids", () => {
    for (const p of ["./m", "../m", "/m", "~/m", "C:\\m", "file:///m", "."]) expect(isDiskPath(p), p).toBe(true);
    for (const p of ["org/repo", "repo", "@local/x", "https://x/y.gguf", "org/repo/file.gguf"]) expect(isDiskPath(p), p).toBe(false);
  });
});
