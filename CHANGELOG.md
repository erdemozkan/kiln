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
from the image's hidden states, greedy decoding on the GPU). Text-to-speech (`text-to-speech`,
`text-to-audio`): VITS / MMS-TTS with VitsTokenizer, the stochastic duration predictor on the CPU,
the flow and HiFi-GAN vocoder on the GPU (relative attention, transposed 1-D convolution, gated
tanh and column-copy kernels; the conv kernel takes 1-D shapes and LeakyReLU on its input);
`seed`, `noise_scale`, `noise_scale_duration`, `speaking_rate`. SpeechT5 text-to-speech (`microsoft/speecht5_tts` with
the `microsoft/speecht5_hifigan` vocoder, or `vocoder`): `speaker_embeddings` (a Float32Array or a
URL), the character SentencePiece tokenizer, the encoder on the seq2seq code with a clamped
relative-position bias, the speech prenet on the CPU with its always-on dropout (`seed`, or
`prenet_dropout: "keep"`), Python's stop rule (`threshold`), the postnet with folded batch norm and
a tanh epilogue on the conv kernel, and the HiFi-GAN builder shared with VITS. Florence-2 (`microsoft/Florence-2-base`,
`-large` and their `-ft` versions) in `image-to-text`: `task` takes its task tokens (`<CAPTION>`,
`<DETAILED_CAPTION>`, `<OD>`, `<OCR>`, `<OCR_WITH_REGION>`, `<CAPTION_TO_PHRASE_GROUNDING>`,
`<REFERRING_EXPRESSION_SEGMENTATION>`, …) and `text_input` their text; results carry the processor's
parsed answer (boxes, quad boxes, polygons in pixels). The DaViT tower runs on the GPU with new
window-attention and channel-attention kernels, its image tokens go ahead of the prompt's in the
BART encoder, generation applies the model's processors, and the processor's tokens are added to
Microsoft's tokenizer. SmolVLM (Idefics3: `HuggingFaceTB/SmolVLM-256M-Instruct`, `-500M-Instruct`) takes
images in chat like Gemma 3, with no extra file: Idefics3's processor (PIL Lanczos, 512 × 512 tiles
plus a global view, row and column tags; `imageSplitting: false` for one view), the SigLIP tower
with the pixel shuffle folded into the patch order, causal image tokens, and a feature slot per
tile. The vision tower keeps one layer queued ahead (0.3 s less per SmolVLM tile), and the PIL
resizer reads memory in order (the same bytes; SmolVLM's preprocessing of a 640 × 480 photo took
370 ms instead of 575). Fixed: a prompt cache restored at load could match a new image's tokens to an old
image's K/V; restored prefixes now stop at the first image token. OWL-ViT (`google/owlvit-*`) and
OWLv2 (`google/owlv2-*`): `zero-shot-object-detection` with Python's post-processing (each patch's
best label, OWLv2's boxes on the padded square), `threshold`, `top_k` and `percentage`; the heads run
in the image tower's plan (a patch-times-class-token kernel), labels meet the patches on the CPU and
the last four images are kept, so new labels on an image skip the tower; OWLv2's processor (grey
padding, scipy's Gaussian blur and linear zoom, identical float32 pixels); tokenizers built from
vocab.json and merges.txt when a repo has no tokenizer.json. Donut
(`naver-clova-ix/donut-base-finetuned-docvqa`): `document-question-answering` with transformers'
prompt and answer extraction (`max_new_tokens`), the Swin encoder on new shifted-window attention
(relative-position bias, wrap-around mask) and patch-merging kernels, submitted a block at a time,
the mBART decoder on the seq2seq code, DonutImageProcessor's resize, thumbnail and centered padding;
the last two pages' encodings are kept. Donut's other fine-tunes (receipts with
`donut-base-finetuned-cord-v2`, document classes with `-rvlcdip`) run in `image-to-text` with their task token
(`prompt` to override) and `json`, DonutProcessor's `token2json` ported. `resizeImage` takes PIL's `box`, and `reduceImage` /
`resizeImageReducing` are PIL's Image.reduce and `reducing_gap`, byte for byte. Image parts in chat combine with `format` (JSON Schema): a vision model reads a document into a given shape (SmolVLM-500M: 4 of 4 replies valid against 0 of 4 unconstrained). Kokoro
(`hexgrad/Kokoro-82M`, from its .pth): `text-to-speech` with `voice` (the ONNX export's voices
by name, or a style table), `speed`, `seed` / `noise_scale: 0`, and kokoro-js's text front end
(`kokoroNormalize`, `kokoroPhonemes`) around a phonemizer the caller passes (`phonemize`) or
phonemes as input (`phonemes: true`); ALBERT, bidirectional LSTMs, style-conditioned norms and the
iSTFTNet generator on the GPU in three plans, with new LSTM, Snake, depthwise transposed
convolution, LeakyReLU and reflection-pad kernels; the harmonic source and the (inverse) STFT on
the CPU. PyTorch checkpoints holding a dict of state dicts load with prefixed names.

**transformers.js's lower-level API.** `AutoTokenizer` / `PreTrainedTokenizer` (callable, int64
tensors, padding and truncation, `encode`, `decode`, `batch_decode`, `tokenize`, special tokens,
`apply_chat_template`), `RawImage`, `read_audio` and `env` (`remoteHost`); pipelines carry the
model's tokenizer and take a `RawImage`. Ids equal to transformers.js 4.3.2's on 31 repos. Tokenizer
fixes found on the way: added tokens' `lstrip` / `rstrip` (Phi-3.5's chat prompts), whitespace
splitting before Metaspace (T5, XLM-R), the "▁" before an added token (NLLB), and Mistral v0.3's
prefix space.

**Site.** `site/`: a landing page, docs, a migration guide from transformers.js, a benchmark
page charting the 31 measured task / model pairs against transformers.js, and eight runnable
examples (chat, text and invoice images to JSON, retrieval, LLM tasks, LoRA switching, Whisper
with word timestamps, zero-shot detection). Built in CI; published by the manual Pages workflow
with the playground under it.

**One model, many tasks.** `tasks(llm)`: `classify` with your own labels (the model's
probabilities, from `llm.scoreChoices`, which scores whole replies in verify passes), `extract` to a
JSON schema, `ner` with your own entity types (offsets, nothing outside the text), extractive
`answer` with a scored answerability check, and `summarize`.

**Retrieval.** `Retriever`: sentence-aware chunking by the embedder's tokens, sentence-transformers
pooling and prompts read from the repo, exact cosine search with filters, `ask(llm, question)` with
the passages in a stable, cacheable system prompt, and `export()` / `fromSnapshot` to store an
index. Scores within 5e-7 of transformers.js embeddings (MiniLM, BGE).

**LoRA adapters.** `loadAdapter(name, repo)` reads PEFT LoRA adapters (safetensors; per-module
rank and alpha, rsLoRA) for the decoder LLMs; `setAdapter(name)` or a call's `adapter` switches.
Updates run beside the base weights in slots sized for the largest rank, each adapter laid out on
the GPU at load, so switching is a GPU copy (4–8 ms on SmolLM2-135M). Greedy output matches a NumPy
reference with the adapters merged in.

**AutoModel and AutoProcessor.** transformers.js's model classes on Kiln's engines: `AutoModel`
and the sequence-classification, token-classification, question-answering and masked-LM heads for
BERT, DistilBERT and RoBERTa; `AutoModel` and `AutoModelForImageClassification` for ViT, DeiT,
DINOv2, ResNet and ConvNeXt; CLIP's `CLIPModel` and projection models; `AutoModelForCausalLM` with
`generate()` (ids in, ids out, generation_config.json defaults, streamers). `AutoProcessor` /
`AutoImageProcessor` / `AutoFeatureExtractor` for images, Whisper and wav2vec2. Outputs within
1e-4 of transformers.js's on the same inputs; greedy generation identical. `read_audio` decodes as
transformers.js does.

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

**Images.** Gemma 3 4B and larger take images with llama.cpp's mmproj file (`load(id, { mmproj })`, or from the transformers checkpoint with no extra file, `vision: false` to skip the tower,
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
