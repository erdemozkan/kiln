// Models from the user's own disk: `localModel()` turns picked files into an id that `load`,
// `pipeline` and the Auto classes accept like a Hub id.
import { registerLocalModel } from "@kiln/loader";

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
 * / `showOpenFilePicker()` or a drop, one `.gguf` `File`, or a map from repo path to `Blob`.
 */
export type LocalModelInput = FileList | readonly File[] | Blob | FileSystemHandle | Map<string, Blob> | Record<string, Blob>;

/** Folders deeper than this are not part of a model repo (and a wrong pick, such as a home folder, must end). */
const MAX_DEPTH = 4;
const MAX_FILES = 5000;

async function walk(dir: DirectoryHandle, prefix: string, depth: number, out: [string, Blob][]): Promise<void> {
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
  let files: [string, Blob][];
  let name = opts.name;
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
