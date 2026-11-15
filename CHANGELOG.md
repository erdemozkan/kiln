# Changelog

## Unreleased

**GGUF from a file or a URL.** `load` takes a `.gguf` `File` (or `Blob`) the user picked, read in
place, and the URL of one `.gguf` on any server, downloaded into the cache like a Hub file. The
prompt cache works for both (a file is keyed by its name, size and modification time, a URL by a hash
of itself). The playground has a file picker. Other formats still need their repo layout.

**Models from a folder on the user's disk.** `localModel(files)` registers a picked folder (a folder
`<input>`'s files, a directory handle, or a map of paths to Blobs) and returns an id, "@local/…", that
loads wherever a Hub id does: `load`, `pipeline`, the Auto classes. The folder is laid out as a Hub
repo is, so safetensors and pytorch_model.bin models of every supported family load from disk, not
only GGUF. Files are read in place; the id carries a fingerprint of them, so converted weights and
the prompt cache are reused until a file changes. The playground has a folder picker, and loads a
picked file or folder as soon as it is chosen.

**Remembering a picked file or folder.** `localModel(handle, { remember: true })` keeps the handle
(from `showOpenFilePicker()`, `showDirectoryPicker()` or a drop) in IndexedDB, and `savedLocalModels()`
lists what is kept: each entry reopens to the same id after a reload, so its converted weights and
prompt cache are found again, or is forgotten. Chrome and Edge have such handles; elsewhere the user
picks again. The playground lists remembered models as buttons.

**`pipeline()` and `load()` take picked files.** Beside a model id, `pipeline(task, model)` accepts
what `localModel` does (a .gguf `File`, a folder input's files, a file or directory handle), and
"text-generation" the URL of one .gguf; `load` accepts a folder's files and handles as well as a
.gguf `File`. A task that needs a repo says so when given the URL of one file.

**Released versions on the Pages site.** A deploy now also publishes every released version at its
own address (`…/kiln/0.2.0/kiln-webgpu.js`, listed in `…/kiln/versions.json`), fetched from npm and
checked against the registry's checksum, so a page that imports the library from the site can pin a
version instead of following `main`.

**Phi-3.** The `phi3` family loads from safetensors and GGUF: Phi-3 mini and medium, and the LongRoPE
variants (Phi-3.5 mini, Phi-3 128k) up to their original 4096-token context, with the short rotary
factors and the attention factor applied. Their stacked `qkv_proj` and `gate_up_proj` are the stacked
matmuls Kiln already runs.

**Phi-4 mini, and rotary embedding on part of a head.** Phi-4 mini rotates the first three quarters
of each attention head and leaves the rest without position. The rope kernels (plain, fused with the
cache write, and fused with q/k norm) now take the number of rotated elements, from the number of
frequencies they are given, so Phi-4 mini loads from GGUF and safetensors (GGUF Q4_K_M checked: 4.3
tokens/s, facts recalled from a 442-token prompt). Qwen 3.5 and Gemma 4 use the same scheme on their
full-attention layers.

**Qwen 3.5 (text).** The `qwen3_5` family loads from its safetensors repos (Qwen/Qwen3.5-0.8B
checked; the larger sizes share the code). Three of every four layers are linear attention, a gated
delta net: a causal convolution over time, then a recurrent state per head, then a gated norm. They
run in new kernels (`causal_conv`, `gated_delta`, `head_slices`, `sigmoid_mul`) on the GPU and the
CPU; the others are attention with q/k norm, rotary on the first quarter of each head, and a gate
on the output that `q_proj` computes beside the query. The 0.8B model answers correctly at 20 to 23
tokens/s and reads a prompt at 120 to 150 tokens/s; a cached load takes 5.7 s.

A recurrent state cannot be cut back to an earlier position the way keys and values can, so the
engine keeps snapshots of it (with the attention layers' keys and values) after each prompt and 16
tokens before its end, up to 256 MiB of them, and a call resumes from the longest one its prompt
starts with: a follow-up chat turn reused 425 of 463 tokens. For these models the prefix store is
off and prefill runs in exact sizes (padding rows would enter the state). Speculative decoding
works: the state is copied before a draft is checked, and when part of the draft is rejected it is
put back and the accepted tokens are run again. LoRA adapters may update linear attention's
projections (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`) as well as the usual
ones. Not yet: images, which need the model's vision tower and positions on three axes in the
language model's rotary kernels. With Kiln's own 4-bit conversion of the safetensors weights the
0.8B model makes slips it does not make with `quant: "q8"` or from a Q4_K_M GGUF (17 + 26 came out
as 33).

**Qwen 3.5 from GGUF.** llama.cpp's files load as well (unsloth's 0.8B, 2B and 9B Q4_K_M checked: 15
and 6 to 7.5 tokens/s): its tensor names for linear attention, the sizes from the `ssm.*` metadata,
the decay rates it stores as −exp(A_log), the `qwen35` pre-tokenizer, and value heads in the tiled
order its converter writes when there are more of them than key heads. Files whose converter merged
linear attention's projections into `attn_qkv_gate` and `ssm_beta_alpha` load too: each half is a
byte range of the merged tensor, listed under its usual name (covered by tests on made-up tensor
lists; no published file with the merge was found to try).

**Prompt cache for Qwen 3.5.** The cache file holds the attention layers' keys and values as for
any model; a small file beside it (`….kv.state`) holds the two newest recurrent snapshots, rewritten
when they change, and both are restored at load, so a conversation resumes after a reload.

**Gemma 4 (text).** The E2B and E4B sizes load, from llama.cpp's GGUF files and from the
transformers checkpoints. Beyond Gemma 3 they have: heads of 512 in the full-attention layers beside
256 in the sliding ones (a new attention kernel, since the others stop at 256), with rotary on a
quarter of those heads; values normalised like queries and keys; the last 20 of 35 layers reading
the keys and values of an earlier layer instead of making their own, so only 15 layers hold a cache
a prompt chunk whose output is not needed stops after them, and the last chunk runs the layers
after them on its final token alone; a second embedding table that gives
every layer its own 256 values per token; a scalar on each layer's output; and logits capped with
tanh before sampling. The second table has 2.3 billion values (1.6 GB of the 3 GB file), too many
for one GPU buffer: it is read once and stored as one table per layer, and each layer looks its rows
up on the GPU, so decode steps still chain there. The model code matches a float64 reference of the
transformers forward pass within 2e-3 (a prompt at once, in chunks, and token by token), and the new
kernel passes its checks on the GPU. On unsloth's gemma-4-E2B-it Q4_0 file (Intel Iris GPU): sums
and capitals right, four facts recalled from a 302-token prompt, a number recalled from the start
of an 839-token prompt (past the 512-token sliding window), the next turn reusing 334 of 352 tokens;
5 to 6 tokens/s, 6.5 s to prefill 839 tokens (3.0 s for 302), and about 50 s of the load for the per-layer table
(it is kept at its 5-bit precision as 8-bit planes, so the model takes roughly 4 GB on the GPU by
its sizes; not measured). Not done: the mixture-of-experts size (26B-A4B), LoRA adapters, the
i8 cache (heads past 256), and a run of the safetensors checkpoint (10 GB), whose loading shares
the GGUF path's code but was not tried.

**Gemma 4 E4B, run.** unsloth's gemma-4-E4B-it Q4_0 file (4.8 GB) with its mmproj-F16, on the same
Intel Iris GPU: the sum and the capital right, the number recalled from the start of an 839-token
prompt, a red circle, a blue square and "KILN 4273" read off a picture, and the Kennedy clip
transcribed word for word, punctuation included. It decodes at 3.2 to 3.8 tokens/s with both towers loaded and prefills
839 tokens in 25 s; with the text model alone, 3.9 to 4.4 tokens/s and 18 to 20 s for 834 tokens
(7 s for 297: about 43 tokens/s at either length), after a first call that takes 11 to 14 s to
warm up. E2B prefills the same prompt in 6.5 s. That gap is the model's size: 24 layers of E4B
take every prompt token, against 15 of E2B, and each multiplies about 2.5 times as many weights,
some 4.1 times the work per token for a prefill 3 to 4 times as long. Fewer, larger GPU submits
(`maxSubmitMs` 400 instead of 50: a tenth as many) change nothing, so the time is the GPU's. All 1031 kernel checks pass on the GPU.

**Images for Gemma 4.** Chat messages may hold images: from a GGUF with its mmproj file
(`mmproj: "mmproj-F16.gguf"`; of its 986 MB only the vision tensors, 369 MB, are read, the audio
tower is left alone) or from the transformers checkpoint, whose files hold the tower (`vision:
false` leaves it out). The tower takes each image at its own shape, resized to whole tokens of
48 × 48 pixels within a budget of `maxImageTokens` (280 by default; 70, 140, 560 and 1120 are the
others the model knows): a ViT over 16 × 16 patches with one learned position embedding per column
and per row, rotary positions with half of each head on each axis, and every projection clipping
its input and output to limits stored with the weights; patches are then averaged 3 × 3 into
tokens and projected to the text width. In the language model an image token is its feature row,
and in each layer's own embedding table it counts as the padding token. New in the kernels: a
clamp, rotary tables with a head in groups, and average pooling of a grid that is not square. The
tower matches a float64 reference of the transformers forward pass within 2e-3 on a small
configuration (two image shapes), and the text model with image tokens matches its reference too;
all 1026 kernel checks pass on the GPU. On unsloth's gemma-4-E2B-it Q4_0 file with mmproj-F16
(Intel Iris GPU): a red circle and a blue square named with their sides, "KILN 4273" read off a
sign, a tall picture's shapes named top to bottom, a photo of a cat described, a follow-up turn
reusing 292 of 309 tokens, and two images in one message told apart. A 640 × 480 picture is 266
tokens; the tower takes about 6 s and the prefill 2 s. The budgets of 560 and 1120 tokens work too: at each
of the three, a 1200 × 900 page of five lines and a reference code in small print was read exactly
(the tower takes roughly 7, 17 and 48 s for 280, 560 and 1120 tokens, from the totals). The largest
needed RMSNorm over more rows than one dispatch dimension holds (its patches times the heads), which
now continue in a second. Not run: the checkpoint's own tower (the 10 GB safetensors).

**Audio for Gemma 4.** Chat messages may hold audio: `{ type: "audio", audio }` parts with mono
16 kHz samples as a Float32Array (`readAudio` makes them from a file or URL), from the same mmproj
file as the images (`audio: false` leaves the tower out; `audioSlots`, default 2, is how many clips
stay on the GPU). A clip becomes a token per 40 ms, up to 750 for 30 s; longer audio goes in as
consecutive pieces of 30 s, each a run of audio tokens of its own and a slot of its own, so a
recording past a minute needs a larger `audioSlots` (the error says so).
The features are the model's log-mel spectrogram (128 HTK bands of a 512-point FFT over 20 ms
frames every 10 ms). The tower halves time twice with two strided convolutions, which run on the
CPU, then twelve conformer layers on the GPU: two feed-forward blocks at half weight around an
attention over the row itself and the eleven before it with a relative position term (a new
kernel), and a causal depthwise convolution with a gated input. Like the vision tower's, its
projections clip their input and output. The tower matches a float64 reference of the transformers
forward pass within 2e-3 on a small configuration, and on the real weights it matches the model's
official ONNX audio encoder (onnx-community/gemma-4-E2B-it-ONNX) on 3 s of speech: a cosine of
1.0000 on every token at f32, 0.9999 at q8. That comparison found what reading the code had not:
in the mmproj file a layer's `conv_norm` is the norm before the convolution module and `norm_conv`
the one inside it, the reverse of what the names suggest; with them swapped the model still
answered about the audio but misheard words. All 1030 kernel checks pass on the GPU. On unsloth's
gemma-4-E2B-it Q4_0 file with mmproj-F16 (Intel Iris GPU): an 11 s clip of Kennedy and a 13 s clip
of King transcribed word for word, a follow-up turn reusing 313 of 339 tokens, and two clips in one
message told apart; roughly 2 s for the tower on 11 s of audio (from the totals) and 2.5 to 3.7 s of prefill. A plain
440 Hz tone gets the answer that no audio was provided. Recordings of 35 s and 48 s (the two clips
end to end) are transcribed through the cut at 30 s, the second reusing the first's 757 tokens;
asked how often one sentence occurs in the 35 s, the model said once where it is twice. Not done:
the checkpoint's own tower (the 10 GB safetensors, whose loading is written but was not run).

**Images for Qwen 3.5.** Chat messages may hold images: from the transformers checkpoint, whose
files hold the vision tower (`vision: false` leaves it out), or from a GGUF with its mmproj file
(`mmproj: "mmproj-F16.gguf"`). The tower takes each image at its own shape, resized to whole tokens
of 32 × 32 pixels and to at most `maxImageTokens` of them (default 576, a 768 × 768 picture): a ViT
with position embeddings resampled to the image's patch grid, rotary positions on rows and columns,
and a merger that joins each 2 × 2 patches into a token. In the language model an image's tokens
share one time position and count rows and columns from it, and the text after continues past the
image's longer side; the step block carries this to the rotary kernels, so a prefill step is all
text or lies inside one image. Checked on the 0.8B (safetensors and GGUF) and 2B (GGUF) models on
an Intel Iris GPU and the CPU backend, which agree: a photo described, shapes, colours and sides
named, three lines of small text and large text on a sign read exactly, two images told apart, a
follow-up question reusing the first turn. A 640 × 416 picture is 260 tokens and takes about 3 s to
prefill on the 0.8B model. Large letters alone on a blank white picture are often misread, more so
the larger the picture; the same letters on a photo are read at every size tried, so this is the
model, not the code. Not compared number for number with another implementation. Video is not
supported.

**Live transcription.** `transcribeMicrophone(asr, { onUpdate })` transcribes the microphone as it
is spoken, and `StreamingTranscriber` does the same for audio fed in pieces from anywhere
(`feed(samples, sampleRate)`, `finish()`). Whisper hears fixed windows, so the audio since the last
settled word is transcribed again on each pass (every `stepSeconds`, default 1) with word times, and
a word is committed once two passes in a row agree on it; the rest is reported as tentative. Past
`maxWindowSeconds` (default 20) the audio the committed words cover is dropped. Windows without
sound are not transcribed, since Whisper invents words for silence. Checked with whisper-tiny.en on
a recording fed at speaking speed, twice with a pause between: the same words as transcribing the
file whole, settling a few seconds behind the audio on an Intel Iris GPU. Punctuation and capitals
are those of the pass that committed a word, so they can differ from a whole-file transcript. The
microphone capture itself (an AudioWorklet) was not run here: the test environment has no microphone.

**Hardening against hostile model repos.** Three ways a crafted repo could freeze the tab are closed.
The check on regexes from `tokenizer.json` and adapter configs now also refuses patterns that are
slow without a repeated group (many choices or many open-ended repetitions in a row; one such pattern
took 41 s on 40 characters). Reading `pytorch_model.bin` walks a dict named from many places once,
not once per path. Chat templates are charged for the work inside filters, methods, `in` and
printing, and for the strings `replace`, `join`, `indent` and `tojson` build, not only per statement.
Phi-3's LongRoPE values are range-checked, the PyTorch zip directory and pickle are size-capped, and
the Pages deploy pins its actions to commits and copies only a plain file out of an npm tarball.

**Failed loads free their GPU memory.** An encoder, seq2seq, audio or vision model whose load fails
no longer leaves its device and buffers behind (the LLM loader already cleaned up). The prompt cache
is no longer read back from the GPU after a call that left it unchanged.

**The prompt cache is appended to, not rewritten.** Its file is now a log of segments, one per save:
after a chat turn only the new tokens' K/V is read back from the GPU and added, where the whole
conversation's was read and written again each turn. When the tokens part ways with what is saved
(an edited message), the file is cut at the last segment both share and continues from there. A
write cut short costs that segment, not the file. Loading reads one segment at a time. Files of the
old layout are removed at the model's next load.

**Chat templates: chained comparisons and argument unpacking.** `a < b < c` now means what it does
in Python (it was read as `(a < b) < c`), and calls take `*list` and `**mapping`, which templates
using them failed on with `unexpected "*"`.

**JSON schema: `allOf` in full.** Its members, their `$ref`s and the keywords beside it are folded
into one schema (properties and `required` joined, types and enums intersected, the tighter bound
kept), and members with `anyOf` are multiplied out. Before, more than one entry was refused, and a
single entry dropped the keywords written beside it. What cannot be combined exactly (two different
patterns) is refused by name.

**8-bit KV cache.** `load(model, { kvDtype: "f8" })` keeps keys and values as 8-bit floats (OCP FP8
E4M3, four to a word, the format llama.cpp 1.0 added), half the memory of the default f16: within the
same 512 MiB budget Qwen3-0.6B's default context goes from 4,608 to 8,192 tokens, and the prompt
cache file halves. It costs some fidelity: log-probabilities of 10 to 15 token answers moved by 0.3
to 2.7 nats (1 to 5%) on Qwen3-0.6B, SmolLM2-135M and Phi-3.5-mini, where f16 moves them by 0.02;
greedy answers to a 470-token question were unchanged on all three. On eight reading questions with four answer
choices each, it changed which answer ranked first in 1 of 8 on both SmolLM2-135M and Qwen3-0.6B
(scores moved by 0.5 and 1.2 nats on average), where f16 changed none: use it where memory matters
more than close calls. Decode speed is within 10% of f16. The default stays f16. On the CPU backend an f8 cache takes the TypeScript attention path, not the WASM one.

**Access tokens for gated and private models.** `setAccessToken(token)` makes requests for repo
files carry a Hugging Face access token, so gated repos (the original Llama and Gemma ones) and
private ones load. The token is sent to `https://huggingface.co` only: not to a mirror, an
`endpoint`, or the URL of one .gguf (the restriction llama.cpp 1.0 put on `HF_TOKEN`). It is kept in
memory, handed to the workers, and never written to the cache. A refusal by the Hub now says what
to do (set a token, or accept the model's terms).

**Whisper's log-mel features about five times faster.** A 30-second window took about 430 ms on
the CPU and now takes 75 to 100 ms: the transform of each frame uses half-length complex FFTs of the
real samples, with the radix-2, 4 and 5 butterflies written out and nothing allocated per frame. The
features are the same to rounding. (llama.cpp moved this step into its compute graph; here the CPU
version is fast enough that a GPU kernel would save little.)

**A second 8-bit KV cache, `kvDtype: "i8"`.** Each head of each position is stored as signed bytes
with one scale: a byte c stands for c·|c|·scale, so codes are spaced by the square root, fine near
zero and under 1% apart at the head's largest value. The same memory as `"f8"` plus a word per head.
On the eight reading questions it moved scores by 0.09 nats on SmolLM2-135M and 0.54 on Qwen3-0.6B
(f8: 0.48 and 1.21) and changed no first choice on either (f8: one of eight on each), at f16's
speed. Evenly spaced bytes were tried first and were worse than f8 on Qwen3 (2.3 nats): its heads
have a few very large values, which use up an even scale. Of the two, i8 is the one to pick.

**Phi-4 mini from safetensors.** A tiny random Phi-4 mini checkpoint in that layout loads and gives
the same scores on the GPU and the CPU backend. The real checkpoint (microsoft/Phi-4-mini-instruct, 7.7 GB) loads too: facts recalled from a
232-token prompt at about 5 tokens/s on the Intel GPU, the next turn reusing 284 of 296 tokens.

**Fewer requests on a first download.** The request that downloads a weight file also tells its
size, so the separate one-byte request that asked for it is gone, and the shards of a split
safetensors model are opened together instead of one after another.

**A download closed just short of its end now finishes.** A GGUF whose last bytes the model does not
read (bartowski's SmolLM2-135M Q4_K_M, for one) was never marked complete, so up to 32 MiB of it was
downloaded again on every load. A source closed with at most 64 MiB left runs to the end first; one
with more left records exactly how far it got. Cached loads of that file went from 2.7 s to 1.8 s.

**Cached loading against transformers.js, re-measured.** Now about even: 1.6 s against 1.7 s to
ready, 1.9 s against 2.5 s to the first tokens (SmolLM2-135M, q4, one session).

## 0.2.0 (2026-11-12)

**Converted-weight cache.** Float weights loaded as q4 or q8 are written to OPFS as converted (read
back from the GPU after the load), and later loads send those bytes to the GPU without reading the
original file: about half the cached load time (SmolLM2-135M 1.0–1.9 s against 2.0–3.4 s, SmolVLM-256M
3.0 s against 6.1 s, same greedy replies), for a third to a half of the download in extra disk and one
slower first load. Vision towers of SmolVLM and Gemma 3 are cached the same way. Skipped where it
would not be smaller (f32 loads, a GGUF's own quantized tensors); `weightCache: false` turns it off.
Readback staging buffers over 4 MiB are no longer pooled.

**Offline loads.** Config, tokenizer and file-listing requests (and Kokoro's voices) are cached in
OPFS and answered from there, then compared with the server in the background once per session, so a
cached model loads with the server unreachable and an updated repo shows from the following load on.
`fetchBytes` joins `fetchJSON` and `fetchText` in the loader.

**Hardening.** Parsing of untrusted model repos is bounded (chat templates, tokenizer regexes, token
ids, config sizes, tensor entries, repo paths); cache keys include the server; retrieved passages are
fenced off from instructions; CI actions and CDN snippets are pinned.

**Performance.** Workers are terminated and GPU memory freed when a load fails; large results are
transferred, not copied; the worker is minified (241 KB gzipped in all); mel filters are sparse;
`scoreChoices` and the masked-LM head run batched on the GPU.

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
