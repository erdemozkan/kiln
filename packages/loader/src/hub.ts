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

/** Lists safetensors shards from the index file, or the single-file default. */
export async function safetensorsFiles(ref: ModelRef, endpoint?: string): Promise<string[]> {
  const index = await fetchJSON<{ weight_map: Record<string, string> }>(fileUrl(ref, "model.safetensors.index.json", endpoint), true);
  return index ? [...new Set(Object.values(index.weight_map))].sort() : ["model.safetensors"];
}
