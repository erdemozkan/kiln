import { describe, expect, it } from "vitest";
import { MemorySource, fileUrl, parseModelId, readSafetensorsHeader, readWav, resample, safeRepoPath } from "../src";

/** A safetensors file whose header is `header`, followed by `dataBytes` zero bytes. */
function safetensors(header: object, dataBytes: number): MemorySource {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + json.length + dataBytes);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  return new MemorySource(out, "test.safetensors");
}

describe("hostile weight files fail cleanly", () => {
  it("accepts a well-formed safetensors file", async () => {
    const h = await readSafetensorsHeader(safetensors({ w: { dtype: "F32", shape: [2, 3], data_offsets: [0, 24] } }, 24));
    expect(h.tensors[0]).toMatchObject({ name: "w", byteLength: 24 });
  });

  it("rejects offsets past the file, lengths that do not match the shape, and bad offsets", async () => {
    await expect(readSafetensorsHeader(safetensors({ w: { dtype: "F32", shape: [2, 3], data_offsets: [0, 24] } }, 8))).rejects.toThrow(/past the end/);
    await expect(readSafetensorsHeader(safetensors({ w: { dtype: "F32", shape: [2, 3], data_offsets: [0, 8] } }, 24))).rejects.toThrow(/its shape and type need/);
    await expect(readSafetensorsHeader(safetensors({ w: { dtype: "F32", shape: [2], data_offsets: [8, 0] } }, 24))).rejects.toThrow(/invalid data_offsets/);
    await expect(readSafetensorsHeader(safetensors({ w: { dtype: "F32", shape: [-1], data_offsets: [0, 4] } }, 24))).rejects.toThrow(/invalid shape/);
    await expect(readSafetensorsHeader(safetensors({ w: { dtype: "constructor", shape: [1], data_offsets: [0, 4] } }, 24))).rejects.toThrow(/not supported/);
  });

  it("keeps repo file names inside the repo", () => {
    const ref = parseModelId("org/repo");
    expect(fileUrl(ref, "1_Pooling/config.json")).toBe("https://huggingface.co/org/repo/resolve/main/1_Pooling/config.json");
    expect(fileUrl(ref, "model 01.safetensors")).toBe("https://huggingface.co/org/repo/resolve/main/model%2001.safetensors");
    for (const bad of ["../../other/repo/x", "/etc/x", "a/../b", "x?download=1", "a#b", "a\\b", ""]) expect(() => safeRepoPath(bad), bad).toThrow(/not a file path/);
  });

  it("rejects WAV headers with absurd rates and caps resampling", () => {
    const wav = new Uint8Array(44 + 8);
    const dv = new DataView(wav.buffer);
    const tag = (o: number, s: string) => [...s].forEach((c, i) => (wav[o + i] = c.charCodeAt(0)));
    tag(0, "RIFF");
    tag(8, "WAVE");
    tag(12, "fmt ");
    dv.setUint32(16, 16, true);
    dv.setUint16(20, 1, true);
    dv.setUint16(22, 1, true);
    dv.setUint32(24, 1, true); // 1 Hz
    dv.setUint16(34, 16, true);
    tag(36, "data");
    dv.setUint32(40, 8, true);
    expect(() => readWav(wav)).toThrow(/not supported/);
    dv.setUint32(24, 16000, true);
    expect(readWav(wav).samples.length).toBe(4);
    expect(() => resample(new Float32Array(100000), 1, 16000)).toThrow(/too many/);
  });
});
