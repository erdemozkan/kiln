// Models from the user's own disk: `localModel()` turns picked files into an id that `load`,
// `pipeline` and the Auto classes accept like a Hub id.
import { registerLocalModel, type DiskFile, type LocalFile } from "@kiln/loader";
import { env } from "./compat";

/** The slice of the File System Access API used here (Chromium; not in every TypeScript lib). */
interface FileHandle {
  kind: "file";
  name: string;
  getFile(): Promise<File>;
}
interface DirectoryHandle {
  kind: "directory";
  name: string;
  values(): AsyncIterable<FileHandle | DirectoryHandle>;
}

/**
 * What `localModel` takes: the files of an `<input type="file">` (with `webkitdirectory` for a
 * folder, or several files picked together), a directory or file handle from `showDirectoryPicker()`
 * / `showOpenFilePicker()` or a drop, one `.gguf` `File`, a map from repo path to `Blob`, or in
 * Node.js the path of a model folder or of one .gguf file.
 */
export type LocalModelInput = FileList | readonly File[] | Blob | FileSystemHandle | Map<string, Blob> | Record<string, Blob> | string;

/** Folders deeper than this are not part of a model repo (and a wrong pick, such as a home folder, must end). */
const MAX_DEPTH = 4;
const MAX_FILES = 5000;

async function walk(dir: DirectoryHandle, prefix: string, depth: number, out: [string, LocalFile][]): Promise<void> {
  for await (const entry of dir.values()) {
    if (entry.name.startsWith(".")) continue;
    if (entry.kind === "file") {
      out.push([prefix + entry.name, await entry.getFile()]);
      if (out.length > MAX_FILES) throw new Error(`"${dir.name}" holds more than ${MAX_FILES} files: pick the model's own folder`);
    } else if (depth < MAX_DEPTH) await walk(entry, `${prefix}${entry.name}/`, depth + 1, out);
  }
}

const isHandle = (v: unknown): v is FileHandle | DirectoryHandle => typeof v === "object" && v !== null && "kind" in v && ((v as { kind: unknown }).kind === "file" || (v as { kind: unknown }).kind === "directory");

/**
 * Registers a model from the user's disk and resolves to its id ("@local/…"), which loads wherever a
 * Hub id does: `load(id)`, `pipeline(task, id)`, `AutoTokenizer.from_pretrained(id)`. A folder is read
 * as a Hub repo is laid out (config.json, the tokenizer files and the weights at its top level); a
 * single .gguf needs nothing beside it. Files are read in place when the model loads: nothing is
 * uploaded or copied. The id lasts for this page: after a reload the files are registered again.
 * `remember: true` keeps a handle for that (see `savedLocalModels`).
 */
export async function localModel(input: LocalModelInput, opts: { name?: string; remember?: boolean } = {}): Promise<string> {
  let files: [string, LocalFile][];
  let name = opts.name;
  if (typeof input === "string") {
    if (opts.remember) throw new Error("localModel: a path on disk needs no remembering; pass it again");
    const disk = diskModel(input);
    return registerLocalModel(disk.files, opts.name ?? disk.name);
  }
  if (opts.remember) {
    if (!isHandle(input)) throw new Error("localModel: only a file or directory handle (showOpenFilePicker, showDirectoryPicker, a drop) can be remembered; an <input>'s files cannot be reopened after a reload");
    await saveHandle(opts.name ?? input.name, { handle: input, name: opts.name });
  }
  if (isHandle(input)) {
    name ??= input.name.replace(/\.gguf$/i, "");
    if (input.kind === "file") files = [[input.name, await input.getFile()]];
    else await walk(input, "", 0, (files = []));
  } else if (input instanceof Blob) {
    const file = (input as Partial<File>).name || "model.gguf";
    name ??= file.replace(/\.gguf$/i, "");
    files = [[file, input]];
  } else if (input instanceof Map) {
    files = [...input];
  } else if (typeof (input as FileList).length === "number") {
    const list = Array.from(input as ArrayLike<File>);
    // A folder input names each file "folder/sub/file"; files picked one by one have only their names.
    files = list.map((f) => [f.webkitRelativePath || f.name, f]);
    const first = list[0]?.webkitRelativePath?.split("/")[0];
    name ??= first || (list.length === 1 ? list[0]!.name.replace(/\.gguf$/i, "") : undefined);
  } else {
    files = Object.entries(input as Record<string, Blob>);
  }
  return registerLocalModel(files, name);
}

/** The slice of Node's fs and path modules used to read a model folder. */
interface NodeFs {
  statSync(path: string, opts?: { throwIfNoEntry?: boolean }): { isDirectory(): boolean; isFile(): boolean; size: number; mtimeMs: number } | undefined;
  readdirSync(path: string, opts: { withFileTypes: true }): { name: string; isDirectory(): boolean; isFile(): boolean }[];
}
interface NodePath {
  resolve(...parts: string[]): string;
  join(...parts: string[]): string;
  basename(path: string, ext?: string): string;
}
interface NodeOs {
  homedir(): string;
}

function nodeModules(): { fs: NodeFs; path: NodePath; os: NodeOs } | undefined {
  const get = (globalThis as { process?: { getBuiltinModule?(id: string): unknown } }).process?.getBuiltinModule;
  if (!get) return undefined;
  return { fs: get("node:fs") as NodeFs, path: get("node:path") as NodePath, os: get("node:os") as NodeOs };
}

/** A model folder or a .gguf on disk (Node.js), as files read in place. */
function diskModel(where: string): { files: [string, DiskFile][]; name: string } {
  const node = nodeModules();
  if (!node) throw new Error(`localModel: "${where}" is a path on disk, which only Node.js can read; in a browser, pass picked files or a handle`);
  const { fs, path, os } = node;
  const full = path.resolve(where.startsWith("file:") ? decodeURIComponent(new URL(where).pathname) : where.replace(/^~(?=$|[\\/])/, os.homedir()));
  const stat = fs.statSync(full, { throwIfNoEntry: false });
  if (!stat) throw new Error(`localModel: ${full} does not exist`);
  const disk = (p: string, s: { size: number; mtimeMs: number }): DiskFile => ({ path: p, size: s.size, lastModified: Math.round(s.mtimeMs) });
  if (stat.isFile()) {
    const file = path.basename(full);
    if (!/\.gguf$/i.test(file)) throw new Error(`localModel: ${full} is a file; give a model folder, or a .gguf file`);
    return { files: [[file, disk(full, stat)]], name: file.replace(/\.gguf$/i, "") };
  }
  const files: [string, DiskFile][] = [];
  const walkDisk = (dir: string, prefix: string, depth: number) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith(".")) continue;
      const p = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (depth < MAX_DEPTH) walkDisk(p, `${prefix}${entry.name}/`, depth + 1);
        continue;
      }
      // A symbolic link (as in a Hugging Face cache snapshot) is followed to its file.
      const s = fs.statSync(p, { throwIfNoEntry: false });
      if (!s?.isFile()) continue;
      files.push([prefix + entry.name, disk(p, s)]);
      if (files.length > MAX_FILES) throw new Error(`localModel: ${full} holds more than ${MAX_FILES} files: give the model's own folder`);
    }
  };
  walkDisk(full, "", 0);
  if (!files.length) throw new Error(`localModel: ${full} has no files`);
  return { files, name: path.basename(full) };
}

/** True for a string that names a place on disk rather than a Hub repo: "./x", "../x", "/x", "~/x", "C:\\x", "file:…". */
export function isDiskPath(id: string): boolean {
  return /^(\.{1,2}([\\/]|$)|[\\/]|~([\\/]|$)|[A-Za-z]:[\\/]|file:)/.test(id);
}

/**
 * The id to load a string as. In Node.js a path ("./models/bert", "/data/llama.gguf") becomes a local
 * model id; so does a repo name found under `env.localModelPath` when `env.allowLocalModels` is set,
 * as transformers.js does. Anything else (a Hub id, a URL, an "@local/…" id) is returned as it is.
 */
export async function resolveModel(id: string): Promise<string> {
  if (!nodeModules()) return id;
  if (isDiskPath(id)) return localModel(id);
  if (env.allowLocalModels && !/^(@local\/|https?:)/i.test(id)) {
    const { fs, path } = nodeModules()!;
    const dir = path.resolve(env.localModelPath, id);
    const stat = fs.statSync(dir, { throwIfNoEntry: false });
    if (stat?.isDirectory() || (stat?.isFile() && /\.gguf$/i.test(id))) return localModel(dir);
  }
  return id;
}

/** A local model the user picked earlier and asked to be remembered. */
export interface SavedLocalModel {
  /** The name it was saved under: `opts.name`, or the file's or folder's own. */
  name: string;
  kind: "file" | "directory";
  /**
   * Registers the files again and resolves to the model's id, or to undefined when the user does not
   * allow reading them. After a reload the browser asks for that permission once more, which it only
   * does from a click or a key press: call this from one.
   */
  open(): Promise<string | undefined>;
  /** Forgets the handle; the files and any caches stay. */
  forget(): Promise<void>;
}

type Permission = "granted" | "denied" | "prompt";
type SavedHandle = (FileHandle | DirectoryHandle) & {
  queryPermission?(d: { mode: "read" }): Promise<Permission>;
  requestPermission?(d: { mode: "read" }): Promise<Permission>;
};

const DB = "kiln-local-models";
const STORE = "handles";

/** Runs one request against the handle store. Handles are structured-cloneable, which is what makes IndexedDB the place for them. */
function idb<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    const factory = (globalThis as { indexedDB?: IDBFactory }).indexedDB;
    if (!factory) return reject(new Error("this browser has no IndexedDB to remember a local model in"));
    const open = factory.open(DB, 1);
    open.onupgradeneeded = () => open.result.createObjectStore(STORE);
    open.onerror = () => reject(open.error);
    open.onsuccess = () => {
      const db = open.result;
      const req = run(db.transaction(STORE, mode).objectStore(STORE));
      req.onsuccess = () => (db.close(), resolve(req.result));
      req.onerror = () => (db.close(), reject(req.error));
    };
  });
}

/** A stored handle, with the name the caller gave it (if any): reopening with the same name yields the same id. */
interface SavedRecord {
  handle: SavedHandle;
  name?: string;
}

async function saveHandle(key: string, record: SavedRecord): Promise<void> {
  await idb("readwrite", (s) => s.put(record, key));
}

/**
 * The local models remembered with `localModel(handle, { remember: true })`, by name. Empty where the
 * browser has no IndexedDB. Handles come from the File System Access API (Chrome and Edge); other
 * browsers have nothing to remember, and the user picks the files again.
 */
export async function savedLocalModels(): Promise<SavedLocalModel[]> {
  let names: string[];
  let records: SavedRecord[];
  try {
    names = (await idb("readonly", (s) => s.getAllKeys())) as string[];
    records = (await idb("readonly", (s) => s.getAll())) as SavedRecord[];
  } catch {
    return [];
  }
  return names.map((name, i) => {
    const { handle, name: given } = records[i]!;
    return {
      name,
      kind: handle.kind,
      async open() {
        let state = (await handle.queryPermission?.({ mode: "read" })) ?? "granted";
        if (state === "prompt") state = (await handle.requestPermission?.({ mode: "read" })) ?? "denied";
        return state === "granted" ? localModel(handle as unknown as FileSystemHandle, { name: given }) : undefined;
      },
      async forget() {
        await idb("readwrite", (s) => s.delete(name));
      },
    };
  });
}
