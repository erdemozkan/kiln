# Future features plan

Ideas that are not built yet. Each entry says what it is, why it is worth doing, and roughly how.

## After 0.2.0

Version 0.2.0 is published on npm (`kiln-webgpu`), the repository is public, and the docs site and
playground are live on GitHub Pages. Left over from that release:

1. **Versioned copies on the Pages site.** It serves only the latest build of `main`
   (`kiln-webgpu.js`); npm and jsDelivr are the versioned route.

## Later

### More architectures

The largest gap against transformers.js, which runs any ONNX graph while Kiln implements each model
family itself. Pick the next families by what people ask for.

### More ways to load local models

Done, unreleased: `load` takes a `.gguf` `File` the user picked, and the URL of one `.gguf` on any
server. Still open:

1. **A folder on disk in Node.js**, the equivalent of transformers.js's `env.localModelPath`. Needs a
   file-system `ByteSource` and a check of what else in the engine assumes a browser.
2. **Safetensors models from picked files.** They need `config.json` and the tokenizer files beside
   the weights, so the input would be a picked folder or several files.
3. **`pipeline()` with a file.** Only `load` accepts one today.
4. **Remembering a picked file.** It has to be picked again after a reload; copying it into the
   browser's storage (or keeping its file handle) would avoid that.
