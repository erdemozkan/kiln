import { describe, expect, it } from "vitest";
import { MemorySource, readPytorchHeader, toF32 } from "../src";

// torch.save of { "a.weight": arange(6)/4 as [2, 3] f32, "b.bias": [1.5, -2, 0.25, 8] f16, "ids": int64,
// "view": arange(10)[4:8], "tied": the same storage as a.weight }, in both formats (PyTorch 1.7).
const ZIP = "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";
const LEGACY = "gAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAmNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApxAClScQEoWAgAAABhLndlaWdodHECY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQMoKFgHAAAAc3RvcmFnZXEEY3RvcmNoCkZsb2F0U3RvcmFnZQpxBVgPAAAAMTQwMzg0OTA4MzE5MzQ0cQZYAwAAAGNwdXEHSwZOdHEIUUsASwJLA4ZxCUsDSwGGcQqJaAApUnELdHEMUnENWAYAAABiLmJpYXNxDmgDKChoBGN0b3JjaApIYWxmU3RvcmFnZQpxD1gPAAAAMTQwMzg0OTA4MzE5NDI0cRBoB0sETnRxEVFLAEsEhXESSwGFcROJaAApUnEUdHEVUnEWWAMAAABpZHNxF2gDKChoBGN0b3JjaApMb25nU3RvcmFnZQpxGFgPAAAAMTQwMzg0ODgzMTk4MjI0cRloB0sDTnRxGlFLAEsDhXEbSwGFcRyJaAApUnEddHEeUnEfWAQAAAB2aWV3cSBoAygoaARoBVgPAAAAMTQwMzg0OTA4MzI3NDQwcSFoB0sKTnRxIlFLBEsEhXEjSwGFcSSJaAApUnEldHEmUnEnWAQAAAB0aWVkcShoDXUugAJdcQAoWA8AAAAxNDAzODQ4ODMxOTgyMjRxAVgPAAAAMTQwMzg0OTA4MzE5MzQ0cQJYDwAAADE0MDM4NDkwODMxOTQyNHEDWA8AAAAxNDAzODQ5MDgzMjc0NDBxBGUuAwAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAAGAAAAAAAAAAAAAAAAAIA+AAAAPwAAQD8AAIA/AACgPwQAAAAAAAAAAD4AwAA0AEgKAAAAAAAAAAAAAAAAAIA/AAAAQAAAQEAAAIBAAACgQAAAwEAAAOBAAAAAQQAAEEE=";

const bytes = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

describe.each([["zip", ZIP], ["legacy", LEGACY]])("pytorch_model.bin (%s)", (_, b64) => {
  it("reads float tensors as byte ranges, skipping integer ones", async () => {
    const data = bytes(b64);
    const src = new MemorySource(data);
    const { tensors } = await readPytorchHeader(src);
    const byName = new Map(tensors.map((t) => [t.name, t]));
    expect([...byName.keys()].sort()).toEqual(["a.weight", "b.bias", "tied", "view"]);
    const values = async (name: string) => {
      const t = byName.get(name)!;
      const n = t.shape.reduce((a, b) => a * b, 1);
      return Array.from(toF32(t.type as "F32", await src.read(t.offset, t.byteLength), n));
    };
    expect(byName.get("a.weight")!.shape).toEqual([2, 3]);
    expect(await values("a.weight")).toEqual([0, 0.25, 0.5, 0.75, 1, 1.25]);
    expect(byName.get("b.bias")!.type).toBe("F16");
    expect(await values("b.bias")).toEqual([1.5, -2, 0.25, 8]);
    expect(await values("view")).toEqual([4, 5, 6, 7]);
    expect(byName.get("tied")!.offset).toBe(byName.get("a.weight")!.offset);
  });
});

/** A zip holding one stored entry, archive/data.pkl. */
function zipOf(pkl: Uint8Array): Uint8Array {
  const name = new TextEncoder().encode("archive/data.pkl");
  const local = new Uint8Array(30 + name.length);
  const lv = new DataView(local.buffer);
  lv.setUint32(0, 0x04034b50, true);
  lv.setUint32(18, pkl.length, true);
  lv.setUint32(22, pkl.length, true);
  lv.setUint16(26, name.length, true);
  local.set(name, 30);
  const central = new Uint8Array(46 + name.length);
  const cv = new DataView(central.buffer);
  cv.setUint32(0, 0x02014b50, true);
  cv.setUint32(20, pkl.length, true);
  cv.setUint32(24, pkl.length, true);
  cv.setUint16(28, name.length, true);
  central.set(name, 46);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, 1, true);
  ev.setUint16(10, 1, true);
  ev.setUint32(12, central.length, true);
  ev.setUint32(16, local.length + pkl.length, true);
  const out = new Uint8Array(local.length + pkl.length + central.length + end.length);
  out.set(local, 0);
  out.set(pkl, local.length);
  out.set(central, local.length + pkl.length);
  out.set(end, local.length + pkl.length + central.length);
  return out;
}

describe("hostile pytorch_model.bin", () => {
  it("walks a dict named from many places once, not once per path", async () => {
    // d0 = {}; d1 = {"a": d0, "b": d0}; … d40 = {"a": d39, "b": d39}: 300 bytes, 2^40 paths.
    const ops: number[] = [0x80, 2, 0x7d, 0x71, 0];
    for (let i = 1; i <= 40; i++) {
      ops.push(0x7d, 0x71, i);
      for (const key of ["a", "b"]) ops.push(0x58, 1, 0, 0, 0, key.charCodeAt(0), 0x68, i - 1, 0x73);
    }
    ops.push(0x2e);
    const t0 = performance.now();
    const header = await readPytorchHeader(new MemorySource(zipOf(Uint8Array.from(ops)))).catch((e: unknown) => e);
    expect(performance.now() - t0).toBeLessThan(1000);
    expect(header instanceof Error ? [] : (header as { tensors: unknown[] }).tensors).toEqual([]);
  });
});
