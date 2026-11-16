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
changelog), and so are its video and LoRA adapters. Open for it: the mixture-of-experts size, and the
safetensors checkpoint (10 GB, not run). Open in the Phi family: LongRoPE past the original 4096 tokens (the long
factors). Segment Anything (SAM, SlimSAM-50) is done (unreleased); open for it: SlimSAM-77, whose width of
168 needs padding to a multiple of 32, a mask as a prompt, and the relative position bias added
inside the attention kernel rather than through widened heads (the global layers' attention would
take about a third of its work now). Phi-4 mini's safetensors checkpoint (7.7 GB) was loaded and answers as the GGUF does.

### More ways to load local models

Done, unreleased: `load` takes a `.gguf` `File` the user picked, and the URL of one `.gguf` on any
server; `localModel` registers a picked folder (any supported format) under an id that `load`,
`pipeline` and the Auto classes accept (both also take the picked files directly), and can remember a picked handle across reloads. In Node.js a path loads a folder or a .gguf from disk, and `env.localModelPath` works as in transformers.js. Still open:

1. **A download cache in Node.js.** Hub files are fetched again on every run there (no OPFS); a
   folder under the user's cache directory would do, as transformers.js keeps one.
2. **Copies in OPFS instead of IndexedDB.** `remember: "copy"` keeps picked files in IndexedDB,
   which works in every browser; not tried with files of several GB in Safari.

### A 4-bit KV cache: values only

A cache with 4-bit keys and values, after TurboQuant (arXiv:2504.19874), kept Gemma 4 E2B's answers
but broke Qwen3 0.6B. Qwen3's keys have a few channels around a hundred times the rest and nearly
the same at every position; with 4-bit keys most of the first layer's attention weight goes to the
wrong tokens (92% of it on that model's real cache). Centring the keys first (free: the same vector
off every key leaves the softmax alone) only brings that to 61%, and KIVI-style keys (per channel
over blocks of 32 positions) did no better on that cache (72%). So `kvDtype: "q4"` keeps keys as i8
and stores only values in 4 bits, which keeps attention as i8 does. Saving more would need 4-bit
keys that survive such channels.
