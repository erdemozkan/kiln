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
 */
export async function localModel(input: LocalModelInput, opts: { name?: string } = {}): Promise<string> {
  let files: [string, Blob][];
  let name = opts.name;
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
