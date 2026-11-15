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
work too, and so do images (its vision tower, and rotary positions on three axes in the language
model: a token's position is then no longer its index). Open for it: video, and the chunked
form of the recurrence if prefill of the linear layers ever dominates (it is 14% now). Gemma 4's
text model, its images and its audio are done too (E2B from GGUF with its mmproj file, run; see the
changelog). Open for it: video, the mixture-of-experts size, adapters, and the
safetensors checkpoint (10 GB, not run). Open in the Phi family: LongRoPE past the original 4096 tokens (the long
factors). Phi-4 mini's safetensors checkpoint (7.7 GB) was loaded and answers as the GGUF does.

### More ways to load local models

Done, unreleased: `load` takes a `.gguf` `File` the user picked, and the URL of one `.gguf` on any
server; `localModel` registers a picked folder (any supported format) under an id that `load`,
`pipeline` and the Auto classes accept (both also take the picked files directly), and can remember a picked handle across reloads. Still open:

1. **A folder on disk in Node.js**, the equivalent of transformers.js's `env.localModelPath`. Needs a
   file-system `ByteSource` and a check of what else in the engine assumes a browser.
2. **Remembering without handles.** Safari and Firefox have no file handles to keep; copying a picked
   file into the browser's storage would cover them, at the cost of a second copy on disk.

### A 4-bit KV cache: tried, not merged

The local branch `q4-cache` holds a 4-bit cache after TurboQuant (arXiv:2504.19874): each head
turned by a Hadamard rotation, then 4-bit Lloyd-Max codes with a scale per 32 values (5 bits a
value, against 8 for i8 and 16 for f16), with GPU and CPU kernels and their checks. It keeps Gemma 4
E2B's answers, whose keys and values are normalised, but breaks Qwen3 0.6B. Qwen3's keys have a few
channels around a hundred times the rest and nearly the same at every position; with 4-bit keys
most of the first layer's attention weight goes to the wrong tokens (92% of it on that model's real
cache), and centring the keys first (free: the same vector off every key leaves the softmax alone)
only brings that to 61%. 8-bit keys with 4-bit values keep attention as the i8 cache does, but at
only about 17% less memory than it. What could make 4-bit keys work is quantizing them per channel
over blocks of positions (KIVI's way), which needs a different layout and a running store; worth it
only if long contexts become the limit.
