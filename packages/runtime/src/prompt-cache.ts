import { readCacheFile, writeCacheFile } from "@kiln/loader";
import type { CacheSnapshot } from "./engine";

/** Bump when the K/V layout or anything that changes cached values (kernels, RoPE) changes. */
const VERSION = 1;
const MAGIC = 0x6b6c6e31; // "kln1"

/**
 * Where a model's prompt cache lives. Everything that changes the cached K/V is in the key: the
 * model and its revision, how the weights are stored, and f16 math.
 */
export function promptCacheName(model: string, o: { revision?: string; quant?: string; f16Math?: boolean }): string {
  return `${encodeURIComponent(`${model}@${o.revision ?? "main"}/${o.quant ?? "q4"}${o.f16Math ? "/f16math" : ""}/v${VERSION}`)}.kv`;
}

/** Layout: magic, header length (u32 LE each), JSON header, then the K/V bytes. */
export async function savePromptCache(name: string, s: CacheSnapshot): Promise<boolean> {
  const header = new TextEncoder().encode(JSON.stringify({ tokens: s.tokens, kvDtype: s.kvDtype, layers: s.layers, rowBytes: s.rowBytes }));
  const head = new Uint8Array(8);
  const dv = new DataView(head.buffer);
  dv.setUint32(0, MAGIC, true);
  dv.setUint32(4, header.length, true);
  return writeCacheFile(name, [head, header, s.data]);
}

/** The snapshot saved under `name`, or undefined if there is none or it does not parse. */
export async function loadPromptCache(name: string): Promise<CacheSnapshot | undefined> {
  const bytes = await readCacheFile(name);
  if (!bytes || bytes.length < 8) return undefined;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== MAGIC) return undefined;
  const len = dv.getUint32(4, true);
  if (8 + len > bytes.length) return undefined;
  try {
    const h = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + len))) as Omit<CacheSnapshot, "data">;
    const data = bytes.subarray(8 + len);
    if (!Array.isArray(h.tokens) || data.length !== h.layers * 2 * h.tokens.length * h.rowBytes) return undefined;
    return { ...h, data };
  } catch {
    return undefined;
  }
}
