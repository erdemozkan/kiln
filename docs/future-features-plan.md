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

Phi-3, 3.5 and Phi-4 mini are done (unreleased); Phi-4 mini needed rope on part of each head, which
the rope kernels now take. Qwen 3.5's text model is done too (unreleased), from safetensors: its
linear-attention layers (a gated delta net) run in their own kernels, and the engine snapshots their
recurrent state; GGUF files, the prompt cache across reloads, speculative decoding and adapters
work too. Open for it: images (its vision tower, and rotary positions on three axes in the language
model: a token's position is then no longer its index), GGUF files whose projections are merged,
and the chunked
form of the recurrence if prefill of the linear layers ever dominates (it is 14% now). Gemma 4 uses
the same partial rotary but needs K/V shared between layers, per-layer input embeddings and two
head sizes. Open in the Phi family: LongRoPE past the original 4096 tokens (the long
factors); and a run of the safetensors checkpoints, whose name mapping is unit-tested but which were
not loaded here (7.6 GB each).

### More ways to load local models

Done, unreleased: `load` takes a `.gguf` `File` the user picked, and the URL of one `.gguf` on any
server; `localModel` registers a picked folder (any supported format) under an id that `load`,
`pipeline` and the Auto classes accept (both also take the picked files directly), and can remember a picked handle across reloads. Still open:

1. **A folder on disk in Node.js**, the equivalent of transformers.js's `env.localModelPath`. Needs a
   file-system `ByteSource` and a check of what else in the engine assumes a browser.
2. **Remembering without handles.** Safari and Firefox have no file handles to keep; copying a picked
   file into the browser's storage would cover them, at the cost of a second copy on disk.
