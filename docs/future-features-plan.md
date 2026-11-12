# Future features plan

Ideas that are not built yet. Each entry says what it is, why it is worth doing, and roughly how.

## Load models from local files

**What.** Let Kiln load a model without a Hub-style server:

1. **A file the user picks in the browser** (a file input or drag and drop), most useful for a single `.gguf`.
2. **A direct URL** to one model file, with no `org/repo/resolve/…` path layout needed.
3. **A folder on disk in Node.js**, the equivalent of transformers.js's `env.localModelPath`.

**Why.** Today a model must be reachable over HTTP in the Hub's path layout, from the Hub, a mirror or
your own static server (`endpoint`). transformers.js can read a folder from disk in Node.js and a
same-origin folder in the browser, but has no official loader for a user-picked file, and it needs
ONNX files. A file picker would put Kiln ahead: people already have GGUF files on their machines, and
opening one in a web page with nothing uploaded fits the reason Kiln exists.

**How, roughly.**

- The loader reads every format through one interface (`ByteSource`: `size`, `read(offset, length)`),
  and a `BlobSource` already exists, so a `File` can be read in ranges without loading it all into
  memory.
- `load` and `pipeline` would accept a source in place of a model id: a `File`/`Blob`, a URL, or for a
  multi-file model a map of file names to either. A GGUF carries its own config and tokenizer, so one
  file is enough; a safetensors model also needs `config.json` and its tokenizer files.
- The file has to reach the model worker: a `File` can be posted to it as it is.
- Caches need a key that is not a repo name. For a picked file, the name, size and last-modified time
  would do for the converted-weight and prompt caches; the download cache does not apply.
- Node.js folder loading needs a file-system `ByteSource` and a check of what else in the engine
  assumes a browser, so it is the largest of the three.

**Open questions.**

- Should a picked file be copied into the browser's storage so the model reloads without picking it
  again, or only its converted weights?
- Which multi-file layout to accept for safetensors models: a picked folder, several files, or both.
