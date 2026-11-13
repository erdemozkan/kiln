# Future features plan

Ideas that are not built yet. Each entry says what it is, why it is worth doing, and roughly how.

## After 0.2.0

Version 0.2.0 is published on npm (`kiln-webgpu`), the repository is public, and the docs site and
playground are live on GitHub Pages. Versioned copies on the Pages site are scripted
(`scripts/pages-versions.mjs`, run by the deploy workflow) and take effect with the next deploy.

## Later

### More architectures

The largest gap against transformers.js, which runs any ONNX graph while Kiln implements each model
family itself. Pick the next families by what people ask for.

### More ways to load local models

Done, unreleased: `load` takes a `.gguf` `File` the user picked, and the URL of one `.gguf` on any
server; `localModel` registers a picked folder (any supported format) under an id that `load`,
`pipeline` and the Auto classes accept (both also take the picked files directly), and can remember a picked handle across reloads. Still open:

1. **A folder on disk in Node.js**, the equivalent of transformers.js's `env.localModelPath`. Needs a
   file-system `ByteSource` and a check of what else in the engine assumes a browser.
2. **Remembering without handles.** Safari and Firefox have no file handles to keep; copying a picked
   file into the browser's storage would cover them, at the cost of a second copy on disk.
