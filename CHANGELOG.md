# Changelog

## 0.1.0 (2026-11-08)

First release of `kiln-webgpu` (Kiln.js): LLM inference in the browser on WebGPU, loading Hugging
Face safetensors and GGUF files directly.

**Packaging.** One ES module with the model worker's code inside (started from a blob URL), so it
loads from a CDN with no bundler, and from any bundler or Vite's dev server with no worker file to
serve; types need no `@webgpu/types` (TypeScript 5.4 to 7). The in-process engine is
`kiln-webgpu/engine`.

**transformers.js-style pipeline.** `pipeline("text-generation", id, { dtype, device,
progress_callback })` and `TextStreamer`, taking transformers.js's option names (`max_new_tokens`,
`do_sample`, `top_k`, `top_p`, `repetition_penalty`, `return_full_text`) and returning its output
shapes (`[{ generated_text }]`, the messages with the reply appended for chats).

**Encoder tasks.** BERT, DistilBERT and RoBERTa models from safetensors (WordPiece tokenizers from
tokenizer.json or vocab.txt, RoBERTa's byte-level BPE) for `feature-extraction`,
`text-classification`, `token-classification`, `question-answering`, `fill-mask` and
`zero-shot-classification`, with transformers.js's output shapes, `Tensor` and `cos_sim`; scores
equal to Python transformers' to six decimals. Several inputs (a batch of sentences, zero-shot
labels) run back to back with one wait; the masked-LM head runs on the GPU.

**Summarization, translation and text2text.** BART / DistilBART, Marian (OPUS-MT), M2M100 /
NLLB and T5 / Flan-T5 encoder-decoders for `summarization`, `translation` and
`text2text-generation` (greedy, with the model's generation settings and language codes), with
SentencePiece tokenizers (Unigram and BPE, the Precompiled normalizer, .spm files), T5's relative
position bias in attention, and `pytorch_model.bin` checkpoints (zip and legacy) wherever a repo has
no safetensors. Uncached range reads retry network and server errors.

**Speech recognition.** Whisper (`automatic-speech-recognition`) from the original repos: log-mel
features as transformers computes them, convolutions as matmuls, transformers.js's decoding prompt
and token suppression, `language` / `task` options, audio from samples, URLs, Blobs or bytes;
segment timestamps (`return_timestamps: true`, with transformers' timestamp rule and seek loop) and
overlapping `chunk_length_s` / `stride_length_s` windows merged as transformers.js merges them (by
default for audio past 30 s); word timestamps (`return_timestamps: "word"`) from the alignment heads'
cross-attention, recorded on the GPU during decoding, with transformers.js's dynamic time warping
and word grouping.

**Speech encoders.** wav2vec2 and HuBERT from safetensors or `pytorch_model.bin`: CTC
`automatic-speech-recognition` (the pipeline picks Whisper or CTC from the config) and
`audio-classification` (sequence classifiers, including the weighted sum over layers), both
front-end variants (GroupNorm, post-LayerNorm base models; LayerNorm, pre-LayerNorm large models),
the grouped positional convolution as one kernel.

**Images.** ViT and DeiT (including the distilled model's two heads) from safetensors or
`pytorch_model.bin`: `image-classification` and `image-feature-extraction`, with transformers' image
preprocessing: PIL's resampling filters reimplemented with its fixed-point arithmetic, so resized
pixels equal PIL's byte for byte, plus shortest-edge and ConvNeXt resizing, center crop, and float32
rescale and normalize as numpy does them. `quick_gelu` activation. CLIP (`openai/clip-*`):
`zero-shot-image-classification` with transformers.js's hypothesis template and softmax over labels,
the vision tower on the ViT code (pre-LayerNorm, post-LayerNorm of the pooled token), a causal text
tower that encodes all labels in one batch (block-causal attention kernel) alongside the image,
CLIP's tokenizer (lowercasing, whitespace-dropping split, `</w>` word ends), text embeddings
cached per text; `image-feature-extraction` on CLIP returns the image embedding. YOLOS
(`hustvl/yolos-*`): `object-detection` with transformers.js's post-processing, at the image's own
aspect ratio (position embeddings fitted to each patch grid by torch's bicubic interpolation),
detection heads on the GPU. DETR (`facebook/detr-resnet-50`, `-101`, the `object-detection`
default): a ResNet backbone on a new implicit-GEMM 2-D convolution kernel (bias, residual and ReLU
fused; frozen batch norms folded in when loading) and max pooling, DETR's encoder and decoder with
sine position embeddings. `startProfile` / `stopProfile` work for every model type, not only
language models. Depth Anything (`depth-anything/Depth-Anything-V2-*-hf`): `depth-estimation`
with transformers.js's output (`predicted_depth` at the image size, `depth` as grey bytes),
DINOv2 encoder with position embeddings fitted to each grid, DPT neck and head on new transposed
convolution and bilinear resize kernels; DPT's aspect-keeping resize to multiples of 14. DETR
panoptic (`facebook/detr-resnet-50-panoptic`): `image-segmentation` with transformers.js's panoptic
post-processing, the mask head run only for the queries kept (GroupNorm, nearest-upsample-and-add
and box attention kernels; the conv kernel takes a batch of images). RMBG-1.4 (`briaai/RMBG-1.4`):
`background-removal` (RGBA with the matte as alpha) and `image-segmentation` (the matte), IS-Net
from one architecture description, concatenations as split convolutions; dilated convolutions,
ceil-mode pooling, a direct kernel for convolutions with at most four outputs; `resizeImage` takes
one-channel images. SegFormer (`nvidia/segformer-*`): semantic `image-segmentation` with
transformers.js's output (a mask per class present), a depthwise convolution kernel, and the score
resize and per-pixel best class fused into one kernel. Swin2SR (`caidas/swin2SR-classical-sr-x*`):
`image-to-image` super-resolution with the width padded to a multiple of 32, a Swin V2
shifted-window cosine attention kernel, pixel shuffle and a leaky-ReLU conv epilogue; LayerNorm
takes many short rows 8 per workgroup. Classifiers: ResNet (`microsoft/resnet-*`), ConvNeXt
(`facebook/convnext-*`, 7×7 depthwise convolution) and DINOv2 (`facebook/dinov2-*-imagenet1k-*`,
on the ViT code). Fixed: an integer `size` with `crop_pct` (ConvNeXt-style processors) now resizes
and crops as transformers does; ConvNeXt at 384 and above resizes to a square. Image captioning
(`image-to-text`): VisionEncoderDecoder with a ViT encoder and a GPT-2 decoder (cross-attention
from the image's hidden states, greedy decoding on the GPU).

**Loading.** Reads ahead of a download in progress are fetched on their own instead of waiting
(a `pytorch_model.bin` keeps its zip directory at the end, so large checkpoints showed no progress
until fully downloaded). 2-D tensors saved as transposed views load where the weights are read on
the CPU. An unsupported tensor no longer makes the cache re-download the file.

**Models.** Llama (including Llama 3 rope scaling), Mistral, Qwen2, Qwen3, SmolLM, Granite 3 and 4
(350M to 2B; for Granite 4, the attention-only models) and Gemma 3 (270M to 4B and larger; images
with the GGUF mmproj), from safetensors (F32/F16/BF16) or GGUF
(F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K). Weights are held as q4, q8 or f32; the
default `quant: "auto"` keeps each GGUF tensor at its own precision, every type but Q6_K exactly
(affine q4/q8 planes for Q4_1, Q4_K, Q5_1 and Q5_K).

**Images.** Gemma 3 4B and larger take images with llama.cpp's mmproj file (`load(id, { mmproj })`,
then `{ type: "image", image }` message parts): SigLIP vision tower, 4 × 4 pooling projector, and
bidirectional attention among each image's 256 tokens. The tower's attention runs as a
register-tiled kernel with the softmax folded in (about 46 s per image on an Intel Iris Plus 650).

**API.** `load(id)` runs the model in a module worker; `chat` and `complete` return async iterators
of text, with `stats` (timings, token counts, `stopReason`). The model's own Jinja chat template,
tools and documents, parsed tool calls (`<tool_call>`, Mistral's `[TOOL_CALLS]`, Granite's
`<|tool_call|>`) and `toolChoice`. Structured output: JSON, JSON Schema (with
`pattern`, common `format`s, numeric bounds and `multipleOf`), regular expressions, GBNF grammars or a fixed choice,
enforced by a grammar mask at every token. Draft tokens (speculative decoding by
prompt lookup) on by default, also under a grammar. `startProfile` / `stopProfile` report the time
per kernel and shape (GPU timestamp queries on WebGPU, wall clock on the CPU). Without WebGPU the
CPU fallback runs matmuls in WASM SIMD, split across helper threads on cross-origin isolated pages where timing each matmul shape both ways shows the split is faster.

**Loading.** Weights stream into GPU buffers without holding the model in CPU memory. The download is
the cache: files land in OPFS as 1 GiB segments, interrupted downloads resume, dropped connections
retry, and `listCache` / `clearCache` manage it. `load` asks the browser to keep the cache
(`navigator.storage.persist()`); `info.persistent` gives the answer. Conversion to q4 and q8 runs on the GPU for F32/F16/BF16 safetensors and every GGUF block type
Kiln reads, with the CPU converter's exact bits. The
last prompt's K/V survives page reloads, and earlier prefixes stay on the GPU (`prefixCacheBytes`,
256 MiB by default), so conversations used in turn each prefill only their new turn; overlapping
calls queue.

**CPU fallback.** Without WebGPU (or with `device: "cpu"`), the same engine runs on CPU versions of
every kernel: matmuls and attention in WASM SIMD (TypeScript where SIMD is missing), the rest in
TypeScript; q4/q8 matmuls are integer dot products with x as int16 per block (normalised and quantized in WASM), and large ones use
several threads on cross-origin isolated pages. About 37 tok/s for SmolLM2-135M on an Intel Mac
(32 with top-k / top-p sampling),
with the GPU's greedy tokens; meant for small models. `info.backend` reports which
backend runs.

**Context.** The default context is the model's trained context, up to 8192 tokens and 512 MiB of
KV cache; sliding-window layers keep a ring of window + 256 slots, so Gemma 3 4B gets 8192 tokens in
305 MiB. Prompts that do not fit say how to make room.

**Speed** (Gemma 3 4B Q4_0). Intel Iris Plus 650: about 5.5 tok/s decode, a 1,138-token prompt in
66 s, a cached load in about 10 s. Apple M3: 24.6 tok/s decode, about 179 tok/s prefill, a cached
load in 2.4 s. Decode uses a block-per-lane GEMV with several rows per thread on large matrices,
prefill a tiled GEMM and tiled attention, and GPU work is cut into short submits so the page stays
responsive.

**Checks.** 197 unit tests; a kernel check page (817 checks on the Intel GPU, 763 on the CPU backend) compares every kernel with a CPU reference; Gemma 3 1B matches a NumPy reference token for
token in q8 and q4.
