/**
 * Models on the user's own disk: a picked folder (config, tokenizer and weights, laid out as a Hub
 * repo is) or a single .gguf. The files are registered under an id of the form "@local/<name>", which
 * loads wherever a Hub id does; "@" cannot start a Hub organisation, so the two never collide. Files
 * are read in place from their Blobs: nothing is uploaded, and nothing is copied into the cache.
 */
const PREFIX = "@local/";
const SCHEME = "kiln-local://";

const registry = new Map<string, Map<string, Blob>>();

/** True for an id (or a repo) that names registered local files. */
export function isLocalModel(id: string): boolean {
  return id.startsWith(PREFIX);
}

/** cyrb53: 53 bits are plenty to tell two versions of a folder apart. */
function hash(text: string): string {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 2654435761);
    h2 = Math.imul(h2 ^ c, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
}

/** Paths as a repo has them: forward slashes, and without the folder(s) every file shares. */
function normalize(files: Iterable<readonly [string, Blob]>): Map<string, Blob> {
  let entries = [...files].map(([path, blob]) => [path.replace(/\\/g, "/").replace(/^\/+/, "").split("/"), blob] as const).filter(([p]) => p.every((s) => s && s !== "." && s !== ".."));
  // A picked folder's files arrive as "folder/config.json": the shared leading folders are not part of the repo.
  while (entries.length && entries.every(([p]) => p.length > 1 && p[0] === entries[0]![0][0])) entries = entries.map(([p, b]) => [p.slice(1), b] as const);
  return new Map(entries.map(([p, b]) => [p.join("/"), b]));
}

/**
 * Registers local files as a model and returns its id. The id carries a fingerprint of the files
 * (paths, sizes and modification times), so caches keyed by it (converted weights, the prompt cache)
 * are not reused once a file changes. A folder holding one .gguf and no config.json yields the id of
 * that file, as "org/repo/file.gguf" does on the Hub.
 */
export function registerLocalModel(files: Iterable<readonly [string, Blob]>, name = "model"): string {
  const map = normalize(files);
  if (!map.size) throw new Error("no files to load: pick a model folder or a .gguf file");
  const print = [...map].map(([p, b]) => `${p}|${b.size}|${(b as Partial<File>).lastModified ?? 0}`).sort().join("\n");
  const slug = `${name.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "model"}-${hash(print)}`;
  registry.set(slug, map);
  const repo = PREFIX + slug;
  const ggufs = [...map.keys()].filter((p) => /\.gguf$/i.test(p) && !/mmproj/i.test(p));
  return !map.has("config.json") && ggufs.length === 1 ? `${repo}/${ggufs[0]}` : repo;
}

/** The repo part of a local id ("@local/name", without a .gguf path). */
export function localRepo(id: string): string {
  return id.split("/").slice(0, 2).join("/");
}

/** The files registered for a local id, for handing them to another thread. */
export function localModelFiles(id: string): Map<string, Blob> | undefined {
  return registry.get(localRepo(id).slice(PREFIX.length));
}

/** Registers files under an id another thread made with `registerLocalModel`. */
export function adoptLocalModel(id: string, files: Iterable<readonly [string, Blob]>): void {
  registry.set(localRepo(id).slice(PREFIX.length), new Map(files));
}

/** Drops a local model's files from this thread's registry (a loaded model keeps what it holds). */
export function releaseLocalModel(id: string): void {
  registry.delete(localRepo(id).slice(PREFIX.length));
}

/** The address of a local repo's file, which `localBlob` resolves. */
export function localUrl(repo: string, file: string): string {
  return `${SCHEME}${repo.slice(PREFIX.length)}/${file}`;
}

export function isLocalUrl(url: string): boolean {
  return url.startsWith(SCHEME);
}

/** The Blob behind a local URL; undefined when the folder has no such file. Throws for a model that is not registered. */
export function localBlob(url: string): Blob | undefined {
  const rest = url.slice(SCHEME.length);
  const cut = rest.indexOf("/");
  const files = registry.get(rest.slice(0, cut));
  if (!files) throw new Error(`the local model "${rest.slice(0, cut)}" is not registered in this page (local ids do not survive a reload: register the files again)`);
  return files.get(rest.slice(cut + 1));
}

/** The file names of a local repo. */
export function localListing(repo: string): Set<string> | undefined {
  const files = registry.get(repo.slice(PREFIX.length));
  return files && new Set(files.keys());
}
