import { describe, expect, it } from "vitest";
import { listRepoFiles, parseModelId } from "@kiln/loader";
import { localModel } from "../src/local";

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
