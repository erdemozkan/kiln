/** Hugging Face Hub resolution. A model id is either "org/repo" (safetensors) or "org/repo/file.gguf". */

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
  return `${endpoint}/${ref.repo}/resolve/${encodeURIComponent(ref.revision)}/${file}`;
}

/** Cache key that pins the file to its repo and revision. */
export function cacheKey(ref: ModelRef, file: string): string {
  return `${ref.repo}@${ref.revision}/${file}`;
}

export async function fetchJSON<T>(url: string, optional = false): Promise<T | undefined> {
  const res = await fetch(url);
  if (res.status === 404 && optional) return undefined;
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}${res.status === 401 ? " (gated or private model?)" : ""}`);
  return (await res.json()) as T;
}

export async function fetchText(url: string, optional = false): Promise<string | undefined> {
  const res = await fetch(url);
  if (res.status === 404 && optional) return undefined;
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}${res.status === 401 ? " (gated or private model?)" : ""}`);
  return res.text();
}

/**
 * The repo's file names at this revision, from the Hub API. Undefined when the endpoint has no such
 * API (a mirror or a static server) or refuses it; callers then probe for optional files instead.
 */
export async function listRepoFiles(ref: ModelRef, endpoint = "https://huggingface.co"): Promise<Set<string> | undefined> {
  try {
    const res = await fetch(`${endpoint}/api/models/${ref.repo}/revision/${encodeURIComponent(ref.revision)}?expand%5B%5D=siblings`);
    if (!res.ok) return undefined;
    const { siblings } = (await res.json()) as { siblings?: { rfilename: string }[] };
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
