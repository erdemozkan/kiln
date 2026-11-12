# Future features plan

Ideas that are not built yet. Each entry says what it is, why it is worth doing, and roughly how.

## Version 0.2.0

What 0.2.0 already holds on `main` (see the CHANGELOG's "Unreleased" section): the converted-weight
cache, offline loads, the hardening and performance fixes, and the new README and logo. What is left
before and around the release, in the order suggested:

1. **Re-measure cached loading against transformers.js.** The benchmarks page still lists it under
   "where Kiln is slower", with numbers from two different sessions (1.8 s against 0.9 s, then Kiln
   alone at 1.0–1.9 s). One session with both libraries settles whether that line stays.
2. **Release prep.** Bump the version to 0.2.0, date the CHANGELOG section, rebuild the package and
   the site, and run the pre-publish checks in RELEASE.md. Publishing to npm, making the repository
   public and enabling Pages are done by hand after that.
3. **After publishing.** Swap the README's static badges for live ones (npm version, CI), and confirm
   the install line and the CDN link work.
4. **Bring the other entry pages in line with the README.** The package README (the page npm shows)
   and the docs site still open with Hub-first wording.
5. **Housekeeping.** Delete the merged remote branches, and decide whether commits should use a
   GitHub noreply email.

## Later

### More architectures

The largest gap against transformers.js, which runs any ONNX graph while Kiln implements each model
family itself. Pick the next families by what people ask for.

### Load models from local files

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
