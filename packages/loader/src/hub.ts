/** Hugging Face Hub resolution. A model id is either "org/repo" (safetensors) or "org/repo/file.gguf". */
import { readCacheFile, writeCacheFile } from "./opfs";

export interface HubOptions {
  revision?: string;
  endpoint?: string;
}

export interface ModelRef {
  repo: string;
  revision: string;
  /** Set for single-file GGUF models. */
  gguf?: string;
}

export function parseModelId(id: string, revision = "main"): ModelRef {
  const parts = id.split("/");
  if (parts.length < 2) throw new Error(`"${id}" is not a Hub model id (expected org/repo or org/repo/file.gguf)`);
  const repo = `${parts[0]}/${parts[1]}`;
  const rest = parts.slice(2).join("/");
  if (rest && !rest.endsWith(".gguf")) throw new Error(`"${id}": only .gguf files can be addressed directly`);
  return { repo, revision, gguf: rest || undefined };
}

export function fileUrl(ref: ModelRef, file: string, endpoint = "https://huggingface.co"): string {
  return `${endpoint}/${ref.repo}/resolve/${encodeURIComponent(ref.revision)}/${safeRepoPath(file).split("/").map(encodeURIComponent).join("/")}`;
}

/**
 * Throws unless `file` is a plain path inside a repo: names come from index files the repo itself
 * supplies (`weight_map`), which must not climb out of the repo's folder on the server.
 */
export function safeRepoPath(file: string): string {
  if (!file || file.startsWith("/") || /\\|[?#]|[\x00-\x1f]/.test(file) || file.split("/").some((seg) => seg === ".." || seg === "." || seg === "")) {
    throw new Error(`"${file}" is not a file path inside a repo`);
  }
  return file;
}

/**
 * Cache key that pins the file to its repo, revision and server. Files from the Hub keep the plain
 * key; a mirror's are prefixed with its origin, so one server's copy never serves for another's.
 */
export function cacheKey(ref: ModelRef, file: string, endpoint?: string): string {
  return `${endpointPrefix(endpoint)}${ref.repo}@${ref.revision}/${file}`;
}

/** "" for the Hub, else the endpoint's origin and path followed by "|". */
export function endpointPrefix(endpoint?: string): string {
  if (!endpoint) return "";
  const e = endpoint.replace(/\/+$/, "");
  return e === "https://huggingface.co" ? "" : `${e}|`;
}

const HUB_MAGIC = 0x6b6c6e68; // "klnh"
const UTF8 = new TextDecoder();

/** A small repo file as the server answered it: its bytes, or 404. */
interface HubFile {
  status: number;
  bytes?: Uint8Array;
}

/** The cache file for a repo URL: its cache key (repo@revision/file, as weights use) where the URL has that form. */
function hubEntryName(url: string): string {
  const m = /^(.*?)\/([^/]+\/[^/]+)\/resolve\/([^/]+)\/(.+)$/.exec(url);
  let key = url;
  try {
    if (m) key = `${endpointPrefix(m[1])}${m[2]}@${decodeURIComponent(m[3]!)}/${decodeURIComponent(m[4]!)}`;
  } catch {
    // Not percent-encoded as fileUrl writes it: the URL itself is the key.
  }
  return `${encodeURIComponent(key)}.hub`;
}

/** Layout: magic, status, body length (u32 LE each), then the body. Anything else is a miss. */
async function readHubEntry(name: string): Promise<HubFile | undefined> {
  const b = await readCacheFile(name);
  if (!b || b.length < 12) return undefined;
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  if (dv.getUint32(0, true) !== HUB_MAGIC || dv.getUint32(8, true) !== b.length - 12) return undefined;
  const status = dv.getUint32(4, true);
  return status === 200 ? { status, bytes: b.subarray(12) } : status === 404 ? { status } : undefined;
}

function writeHubEntry(name: string, f: HubFile): Promise<boolean> {
  const head = new Uint8Array(12);
  const dv = new DataView(head.buffer);
  dv.setUint32(0, HUB_MAGIC, true);
  dv.setUint32(4, f.status, true);
  dv.setUint32(8, f.bytes?.length ?? 0, true);
  return writeCacheFile(name, f.bytes ? [head, f.bytes] : [head]);
}

async function download(url: string): Promise<HubFile> {
  const res = await fetch(url);
  return res.ok ? { status: 200, bytes: new Uint8Array(await res.arrayBuffer()) } : { status: res.status };
}

const sameBytes = (a?: Uint8Array, b?: Uint8Array) => a?.length === b?.length && (!a || !b || a.every((v, i) => v === b[i]));

/** Entries already compared with the server in this session. */
const checked = new Set<string>();

/**
 * A small repo file (config, tokenizer, the file listing), cache first: a stored answer is returned
 * without waiting for the network, then compared with the server's once per session in the background
 * and replaced if it changed, so an updated repo shows from the following load on. A 404 is stored
 * too when `keep404` (a file that may not exist is asked for on every load).
 */
async function hubFile(url: string, keep404: boolean, name = hubEntryName(url)): Promise<HubFile> {
  const keeps = (f: HubFile) => f.status === 200 || (keep404 && f.status === 404);
  const hit = await readHubEntry(name);
  if (hit && keeps(hit)) {
    if (!checked.has(name)) {
      checked.add(name);
      void download(url)
        .then((now) => (keeps(now) && (now.status !== hit.status || !sameBytes(now.bytes, hit.bytes)) ? writeHubEntry(name, now) : false))
        .catch(() => false);
    }
    return hit;
  }
  const got = await download(url);
  if (keeps(got)) {
    checked.add(name);
    void writeHubEntry(name, got);
  }
  return got;
}

/** The bytes of a repo file, undefined for a missing one when `optional`. Cached; see `hubFile`. */
export async function fetchBytes(url: string, optional = false): Promise<Uint8Array | undefined> {
  const f = await hubFile(url, optional);
  if (f.status === 404 && optional) return undefined;
  if (!f.bytes) throw new Error(`${url}: HTTP ${f.status}${f.status === 401 ? " (gated or private model?)" : ""}`);
  return f.bytes;
}

export async function fetchJSON<T>(url: string, optional = false): Promise<T | undefined> {
  const text = await fetchText(url, optional);
  if (text === undefined) return undefined;
  try {
    return JSON.parse(text) as T;
  } catch (err) {
    throw new Error(`${url}: not JSON (${err instanceof Error ? err.message : String(err)})`);
  }
}

export async function fetchText(url: string, optional = false): Promise<string | undefined> {
  const bytes = await fetchBytes(url, optional);
  return bytes && UTF8.decode(bytes);
}

/**
 * The repo's file names at this revision, from the Hub API. Undefined when the endpoint has no such
 * API (a mirror or a static server) or refuses it; callers then probe for optional files instead.
 */
export async function listRepoFiles(ref: ModelRef, endpoint = "https://huggingface.co"): Promise<Set<string> | undefined> {
  try {
    const url = `${endpoint}/api/models/${ref.repo}/revision/${encodeURIComponent(ref.revision)}?expand%5B%5D=siblings`;
    // "?" cannot be in a repo path, so the listing's entry never collides with a file's.
    const f = await hubFile(url, false, `${encodeURIComponent(cacheKey(ref, "?siblings", endpoint))}.hub`);
    if (!f.bytes) return undefined;
    const { siblings } = JSON.parse(UTF8.decode(f.bytes)) as { siblings?: { rfilename: string }[] };
    return siblings ? new Set(siblings.map((s) => s.rfilename)) : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Lists safetensors shards from the index file, or the single-file default. With the repo's file
 * listing, the index is only requested when it exists.
 */
export async function safetensorsFiles(ref: ModelRef, endpoint?: string, files?: Set<string>): Promise<string[]> {
  const name = "model.safetensors.index.json";
  const index = files && !files.has(name) ? undefined : await fetchJSON<{ weight_map: Record<string, string> }>(fileUrl(ref, name, endpoint), true);
  return index ? [...new Set(Object.values(index.weight_map))].sort() : ["model.safetensors"];
}

/**
 * The weight files of a transformers repo: safetensors when it has them (sharded or not), otherwise
 * PyTorch's pytorch_model.bin (sharded or not). Without a file listing, safetensors is assumed.
 */
export async function weightFiles(ref: ModelRef, endpoint?: string, files?: Set<string>): Promise<{ format: "safetensors" | "pytorch"; names: string[] }> {
  const hasSt = !files || files.has("model.safetensors") || files.has("model.safetensors.index.json");
  if (hasSt) return { format: "safetensors", names: await safetensorsFiles(ref, endpoint, files) };
  if (files.has("pytorch_model.bin.index.json")) {
    const index = await fetchJSON<{ weight_map: Record<string, string> }>(fileUrl(ref, "pytorch_model.bin.index.json", endpoint));
    return { format: "pytorch", names: [...new Set(Object.values(index!.weight_map))].sort() };
  }
  if (files.has("pytorch_model.bin")) return { format: "pytorch", names: ["pytorch_model.bin"] };
  throw new Error(`${ref.repo} has no safetensors or pytorch_model.bin weights`);
}
