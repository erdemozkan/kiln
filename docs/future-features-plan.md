# Future features plan

Ideas that are not built yet. Each entry says what it is, why it is worth doing, and roughly how.

## After 0.2.0

Version 0.2.0 is published on npm (`kiln-webgpu`), the repository is public, and the docs site and
playground are live on GitHub Pages. Versioned copies on the Pages site are scripted
(`scripts/pages-versions.mjs`, run by the deploy workflow) and take effect with the next deploy.

## Next, in order

Three features, built one after another: pictures from text, mixture-of-experts language models,
then tool calls in chat.

### 1. Pictures from text (`text-to-image`)

Neither Kiln nor transformers.js makes images from a prompt; a one-step model makes it a demo that
runs in a second or two on a laptop GPU.

**Model: SDXS-512-DreamShaper** (`IDKiro/sdxs-512-dreamshaper`, OpenRAIL++, 512 × 512 in one
step, no classifier-free guidance). In diffusers' layout, safetensors, about 1.8 GB in fp32 (half
that in fp16):

- Text encoder: CLIP ViT-L/14's text model (12 layers, 768 wide, quick GELU), 77 tokens, the last
  hidden state (not the pooled one) as the UNet's context. Kiln's CLIP text tower and tokenizer
  already run it.
- UNet (`UNet2DConditionModel`, 316M parameters): 3 levels of 320, 640 and 1280 channels, one
  ResNet block per level, no middle block; cross-attention transformers (8 heads: diffusers' old
  `attention_head_dim` names the head count) on the two lower levels down and the two upper up,
  each a GroupNorm, a 1×1 projection in, self-attention, cross-attention on the text, a GEGLU
  feed-forward, a projection out. Latents 64 × 64 × 4.
- Decoder: TAESD (`AutoencoderTiny`, 2.4M parameters): 3×3 convolutions with ReLU, nearest
  upsampling ×2 three times, latents scaled by `latent_magnitude` 3 and shifted by 0.5.
- Scheduler: one Euler step at timestep 999 (`trailing` spacing, `scaled_linear` betas, epsilon
  prediction): noise z goes in as is, and the result is z · √(σ² + 1) − σ · ε, σ that timestep's.

Kernels: most are there (conv2d, GroupNorm, LayerNorm, attention, GEGLU through `gatedAct`, nearest
upsampling, `cols` for the skip connections' concatenation). New: the sinusoidal timestep
embedding (on the CPU, one row), and tanh-clamped output in the decoder (conv2d has `tanh`).

API: `pipeline("text-to-image")` → `(prompt, { seed, width, height })` returning a `RawImage`;
fixed seed gives the same picture on CPU and GPU within rounding. Python gets it through the bridge
unchanged.

Checks: each part against diffusers in float32 on the same latents (the text encoder's hidden state,
the UNet's noise prediction, the decoder's pixels), on a machine with PyTorch (the MacBook Air; not
on the Intel Mac). GPU runs only in Chrome or on macOS 15 and later.

Later: 1024 × 1024 (SDXS-1024 is not released), image-to-image from a starting picture, more
steps for other distilled models (SD-Turbo's licence is non-commercial).

### 2. Mixture-of-experts language models

gpt-oss-20b, Qwen3-30B-A3B and Gemma 4's MoE size: a router picks a few experts per token, so the
model answers like a large one at the speed of a small one, but needs 12–20 GB of memory.

Done (unreleased): Qwen3-MoE (safetensors and GGUF) and Granite-MoE, matching transformers' greedy
tokens; see the changelog. Open: gpt-oss (its own attention sinks and activation),
Gemma 4's MoE size, and memory for the 30B sizes.

### 3. Tool calls in chat

A chat model asks for a function (`get_weather("Paris")`), the caller runs it and passes the
answer back. Kiln's grammar-constrained output makes the call's JSON valid.

Done (unreleased): `runTools` runs the loop, and calls under "auto" are checked against the tools'
schemas, on the CPU and the GPU, from JavaScript and Python (`run_tools`); see the changelog. Open:
Mistral's call format (its smallest model, 7B, needs more than the CPU backend's 4 GiB and has not
been run on a GPU), and a run of the thinking-block rule on a reasoning model (unit-tested only).

Built-in skills (unreleased): `calculate`, `get_date_time`, `search_wikipedia`, `read_web_page`; see
the changelog. Open: a run with a model large enough for two-step questions (1.5B and up), the
skills from Python, and more skills: a code runner, search over the user's own documents, and Kiln's
own models as skills (make a picture, look at a picture, speak and listen). After those, the user's
aim: a skill that operates other applications. A web page cannot reach outside itself, so that
needs a browser extension (for web apps) or Kiln in Node.js or Python driving the desktop.

Found on the way: in Node.js a download that fails starts again from zero and shows no progress
(the browser's resumes); on a slow link a 1 GB model never arrived.

## Later

### More architectures

The largest gap against transformers.js, which runs any ONNX graph while Kiln implements each model
family itself. Pick the next families by what people ask for.

Phi-3, 3.5 and Phi-4 mini are done (unreleased); Phi-4 mini needed rope on part of each head, which
the rope kernels now take. Qwen 3.5's text model is done too (unreleased), from safetensors: its
linear-attention layers (a gated delta net) run in their own kernels, and the engine snapshots their
recurrent state; GGUF files, the prompt cache across reloads, speculative decoding and adapters
work too, and so do images (its vision tower, and rotary positions on three axes in the language
model: a token's position is then no longer its index). Video is done too (frames in pairs of up to 64 tokens). Open for it: a larger budget per pair, and the chunked
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

### A Python library: what is left

The package in python/ (`kiln_webgpu`) drives Kiln in a Node.js process over pipes, and Node runs
the GPU through Dawn's `webgpu` package. Left to do:

- **Publishing** to PyPI (the name `kiln-webgpu` is free there; `kiln` is taken).
- **A download cache in Node.js**: every new process fetches a Hub model again.
- **Image and audio decoding** without Pillow or a WAV file: Kiln's decoders are the browser's.
- **Requests in parallel** and an asyncio interface; the bridge runs one request at a time.
- **Not needing Node installed**: shipping a Node binary in the wheel, or a native port (the WGSL
  kernels run unchanged under wgpu-py, but the loaders, plans, engine and tokenizers would be
  rewritten in Python, a second code base to keep correct).

### A 4-bit KV cache: values only

A cache with 4-bit keys and values, after TurboQuant (arXiv:2504.19874), kept Gemma 4 E2B's answers
but broke Qwen3 0.6B. Qwen3's keys have a few channels around a hundred times the rest and nearly
the same at every position; with 4-bit keys most of the first layer's attention weight goes to the
wrong tokens (92% of it on that model's real cache). Centring the keys first (free: the same vector
off every key leaves the softmax alone) only brings that to 61%, and KIVI-style keys (per channel
over blocks of 32 positions) did no better on that cache (72%). So `kvDtype: "q4"` keeps keys as i8
and stores only values in 4 bits, which keeps attention as i8 does. Saving more would need 4-bit
keys that survive such channels.
