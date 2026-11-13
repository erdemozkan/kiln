import { endpointPrefix, readCacheBlob, removeCacheFile, writeCacheFile } from "@kiln/loader";
import type { CacheShape, CacheSnapshot } from "./engine";

/**
 * Bump when the K/V layout or anything that changes cached values (kernels, RoPE) changes.
 * 2: sliding layers store min(n, slots) ring rows, so the data length is in the header.
 * 3: the file is a log of segments, one per save, so a turn appends its own rows only.
 */
const VERSION = 3;
const MAGIC = 0x6b6c6e33; // "kln3"
const SEGMENT = 0x6b736567; // "kseg"
/** Longest JSON header read from a file (a segment's lists its tokens). */
const MAX_JSON = 64 * 2 ** 20;

/**
 * Where a model's prompt cache lives. Everything that changes the cached K/V is in the key: the
 * server, the model and its revision, how the weights are stored, and f16 math.
 */
export function promptCacheName(model: string, o: { revision?: string; quant?: string; f16Math?: boolean; endpoint?: string }): string {
  return `${encodeURIComponent(`${endpointPrefix(o.endpoint)}${model}@${o.revision ?? "main"}/${o.quant ?? "q4"}${o.f16Math ? "/f16math" : ""}/v${VERSION}`)}.kv`;
}

/** The names `name` had under earlier layouts: files nothing reads any more. */
export function oldPromptCacheNames(name: string): string[] {
  const suffix = `${encodeURIComponent(`/v${VERSION}`)}.kv`;
  if (!name.endsWith(suffix)) return [];
  return Array.from({ length: VERSION - 1 }, (_, i) => `${name.slice(0, -suffix.length)}${encodeURIComponent(`/v${i + 1}`)}.kv`);
}

/** What a prompt cache file holds, as the writer needs it to append: its tokens, and where each segment ends. */
export interface PromptCacheState {
  shape: CacheShape;
  tokens: number[];
  /** Per segment in file order: the token count it reaches, and the file offset just past it. */
  segments: { to: number; end: number }[];
}

/** The engine, as far as saving needs it (see `Engine.cacheTokens`, `cacheShape`, `exportCacheFrom`). */
export interface PromptCacheSource {
  readonly cacheTokens: readonly number[];
  readonly cacheShape: CacheShape;
  exportCacheFrom(from: number): Promise<{ tokens: number[]; parts: Uint8Array[] }>;
}

const sameShape = (a: CacheShape, b: CacheShape) => a.kvDtype === b.kvDtype && a.layers === b.layers && a.rowBytes === b.rowBytes && a.slots.length === b.slots.length && a.slots.every((s, i) => s === b.slots[i]);

/** First position of from…to-1 a layer of `slots` slots still holds once `to` are written (as `LlamaModel.readKvRange`). */
const rangeStart = (slots: number, from: number, to: number) => Math.max(from, to - slots);
const rangeBytes = (shape: CacheShape, from: number, to: number) => shape.slots.reduce((t, s) => t + 2 * (to - rangeStart(s, from, to)) * shape.rowBytes, 0);

function framed(magic: number, json: unknown): Uint8Array[] {
  const body = new TextEncoder().encode(JSON.stringify(json));
  const head = new Uint8Array(8);
  const dv = new DataView(head.buffer);
  dv.setUint32(0, magic, true);
  dv.setUint32(4, body.length, true);
  return [head, body];
}

/**
 * Brings the file up to the engine's cached tokens and returns its new state, or undefined when the
 * write failed (the next save then starts the file over).
 *
 * Layout: magic, header length (u32 LE each), JSON header (the cache's shape), then segments. A
 * segment is its magic, a header length, a JSON header (the positions it covers, their tokens, its
 * data length) and those positions' K/V as `LlamaModel.readKvRange` returns them. With `prev` (the
 * state the last save or load returned) only what changed is written: segments that end inside the
 * tokens both share stay, the file is cut after them, and one segment with the rest follows. A chat
 * turn so costs its own tokens' rows, not the whole conversation's again.
 */
export async function savePromptCache(name: string, e: PromptCacheSource, prev?: PromptCacheState): Promise<PromptCacheState | undefined> {
  const shape = e.cacheShape;
  const now = e.cacheTokens;
  const old = prev && sameShape(prev.shape, shape) ? prev : undefined;
  let common = 0;
  if (old) while (common < old.tokens.length && common < now.length && old.tokens[common] === now[common]) common++;
  // Nothing changed: no GPU readback, no write.
  if (old && common === old.tokens.length && common === now.length) return old;
  const kept = old ? old.segments.filter((s) => s.to <= common) : [];
  const from = kept.at(-1)?.to ?? 0;
  const at = kept.at(-1)?.end ?? 0;
  // Taken in the same task as `now`, so both describe the same cache.
  const { tokens, parts } = now.length > from ? await e.exportCacheFrom(from) : { tokens: now.slice(0, from), parts: [] };
  const out: Uint8Array[] = at === 0 ? framed(MAGIC, shape) : [];
  if (tokens.length > from) {
    const bytes = parts.reduce((n, p) => n + p.length, 0);
    if (bytes !== rangeBytes(shape, from, tokens.length)) return undefined;
    out.push(...framed(SEGMENT, { from, to: tokens.length, tokens: tokens.slice(from), bytes }), ...parts);
  }
  if (!(await writeCacheFile(name, out, at))) return undefined;
  const end = at + out.reduce((n, p) => n + p.length, 0);
  return { shape, tokens, segments: tokens.length > from ? [...kept, { to: tokens.length, end }] : kept };
}

/** A u32-framed JSON block at `at`: its value and the offset just past it, or undefined if it is not there. */
async function readFramed(blob: Blob, at: number, magic: number): Promise<{ json: unknown; end: number } | undefined> {
  if (at + 8 > blob.size) return undefined;
  const dv = new DataView(await blob.slice(at, at + 8).arrayBuffer());
  const len = dv.getUint32(4, true);
  if (dv.getUint32(0, true) !== magic || len > MAX_JSON || at + 8 + len > blob.size) return undefined;
  try {
    return { json: JSON.parse(await blob.slice(at + 8, at + 8 + len).text()), end: at + 8 + len };
  } catch {
    return undefined;
  }
}

const count = (v: unknown, max: number): v is number => typeof v === "number" && Number.isInteger(v) && v >= 1 && v <= max;

function shapeOf(json: unknown): CacheShape | undefined {
  const h = json as Partial<CacheShape> | null;
  if (!h || typeof h.kvDtype !== "string" || !count(h.layers, 4096) || !count(h.rowBytes, 2 ** 24) || !Array.isArray(h.slots) || h.slots.length !== h.layers) return undefined;
  if (!h.slots.every((s) => count(s, 2 ** 24))) return undefined;
  return { kvDtype: h.kvDtype, layers: h.layers, rowBytes: h.rowBytes, slots: h.slots };
}

/**
 * The cache saved under `name`: the snapshot to hand `Engine.importCache` (which decides whether it
 * fits the model) and the state to hand the next `savePromptCache`. Undefined if there is no file
 * or it does not parse. Segments are read up to the first one that is cut short or does not follow
 * the one before, so a write that was interrupted costs that segment, not the file.
 */
export async function loadPromptCache(name: string): Promise<{ snapshot: CacheSnapshot; state: PromptCacheState } | undefined> {
  // Files of earlier layouts for this model would otherwise stay on disk for good.
  await Promise.all(oldPromptCacheNames(name).map(removeCacheFile));
  const blob = await readCacheBlob(name);
  if (!blob) return undefined;
  const head = await readFramed(blob, 0, MAGIC);
  const shape = head && shapeOf(head.json);
  if (!head || !shape) return undefined;
  const tokens: number[] = [];
  const segments: { from: number; to: number; data: number; end: number }[] = [];
  for (let at = head.end; ; ) {
    const seg = await readFramed(blob, at, SEGMENT);
    const h = seg?.json as { from?: unknown; to?: unknown; tokens?: unknown; bytes?: unknown } | undefined;
    if (!seg || !h || h.from !== tokens.length || !count(h.to, 2 ** 24) || h.to <= tokens.length) break;
    if (!Array.isArray(h.tokens) || h.tokens.length !== h.to - tokens.length || !h.tokens.every((t) => typeof t === "number")) break;
    const bytes = rangeBytes(shape, tokens.length, h.to);
    if (h.bytes !== bytes || seg.end + bytes > blob.size) break;
    segments.push({ from: tokens.length, to: h.to, data: seg.end, end: seg.end + bytes });
    for (const t of h.tokens as number[]) tokens.push(t);
    at = seg.end + bytes;
  }
  if (!segments.length) return undefined;
  // The snapshot in `LlamaModel.readKv`'s layout: per layer its first min(n, slots) K rows, then as
  // many V rows. Every row of it was stored by some segment, so it is no larger than the file.
  const n = tokens.length;
  const row = shape.rowBytes;
  const total = shape.slots.reduce((t, s) => t + 2 * Math.min(n, s) * row, 0);
  if (total > blob.size) return undefined;
  const data = new Uint8Array(total);
  for (const seg of segments) {
    // One segment in memory at a time; later ones overwrite the ring slots earlier ones filled.
    const bytes = new Uint8Array(await blob.slice(seg.data, seg.end).arrayBuffer());
    if (bytes.length !== seg.end - seg.data) return undefined;
    let src = 0;
    let base = 0;
    for (const slots of shape.slots) {
      const held = Math.min(n, slots);
      const start = rangeStart(slots, seg.from, seg.to);
      const rows = seg.to - start;
      const first = start % slots;
      const run = Math.min(rows, slots - first);
      // K rows, then V rows; a range that runs off the ring's end goes on at slot 0.
      for (const part of [base, base + held * row]) {
        data.set(bytes.subarray(src, src + run * row), part + first * row);
        data.set(bytes.subarray(src + run * row, src + rows * row), part);
        src += rows * row;
      }
      base += 2 * held * row;
    }
  }
  return { snapshot: { tokens: [...tokens], kvDtype: shape.kvDtype, layers: shape.layers, rowBytes: shape.rowBytes, data }, state: { shape, tokens, segments: segments.map((s) => ({ to: s.to, end: s.end })) } };
}
