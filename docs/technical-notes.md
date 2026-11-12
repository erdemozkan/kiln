# Kiln.js: technical notes

The design, measurements and per-model notes behind Kiln. The short introduction is the [README](../README.md).

WebGPU-native LLM inference for the browser, with a CPU (WASM SIMD) fallback. Loads Hugging Face
safetensors or GGUF directly — no ONNX export, no ahead-of-time compile step. Published as
`kiln-webgpu`, one file that also works straight from a CDN, with a transformers.js-style pipeline:

```ts
import { pipeline } from "kiln-webgpu"; // or "https://cdn.jsdelivr.net/npm/kiln-webgpu@0.1.0"

const generator = await pipeline("text-generation", "HuggingFaceTB/SmolLM2-135M-Instruct", { dtype: "q4" });
const [out] = await generator([{ role: "user", content: "Hi!" }], { max_new_tokens: 128 });
```

The site in [`site/`](../site) has the docs, runnable examples, a migration guide from transformers.js
and the benchmarks against it: `npm run dev:site` serves it locally, and the manual Pages workflow
publishes it.

Kiln's own API streams text:

```ts
import { load } from "kiln-webgpu";

const llm = await load("HuggingFaceTB/SmolLM2-135M-Instruct", { quant: "q4", onProgress: console.log });
for await (const text of llm.chat([{ role: "user", content: "Hi!" }], { temperature: 0.7 })) render(text);
```

The model runs in a module worker; the main thread only sees an async iterator.

Structured output is a generate option:

```ts
const schema = { type: "object", properties: { city: { type: "string" }, population: { type: "integer" } }, required: ["city", "population"] };
const json = llm.chat(messages, { format: { type: "json", schema } }); // or { type: "json" }, { type: "choice", choices: ["yes", "no"] },
// { type: "regex", pattern: "\\d{3}-\\d{4}" }, or { type: "grammar", grammar: 'root ::= "yes" | "no"' } (GBNF)
```

## Design

| Package | Role |
|---|---|
| `@kiln/api` | Main-thread client, worker RPC, streaming `TextStream`, transformers.js-style `pipeline` |
| `@kiln/runtime` | `Engine`: chunked prefill, pipelined decode, weight streaming |
| `@kiln/models` | Code-defined architectures (Llama / Mistral / Qwen2 / Qwen3 / SmolLM / Granite / Gemma 3 / SmolVLM), HF + GGUF config mapping |
| `@kiln/tokenizers` | Byte-level and SentencePiece-style BPE from `tokenizer.json` or GGUF metadata, streaming UTF-8 decode, a Jinja interpreter for chat templates |
| `@kiln/grammar` | JSON / JSON Schema / choice grammars as a byte-level pushdown matcher, token masks over a vocabulary trie |
| `@kiln/kernels` | WGSL kernels, op wrappers, CPU reference implementations |
| `@kiln/core` | Backend HAL interface, tensors, uniform packing, `Ctx` plan recorder |
| `@kiln/backend-webgpu` | WebGPU backend: explicit bind group layouts, pipeline cache, staging pool, error reporting |
| `@kiln/loader` | Byte sources (HTTP range, OPFS cache), safetensors/GGUF parsers, dtype conversion, q4 and q8 quantization |

Key ideas:

- **Compile once, replay many.** `LlamaModel.buildPlan(T)` records every dispatch for a T-token forward
  pass. Positions, the sampled row, seed and sampling settings live in a small step buffer, so one plan
  serves every decode step.
- **The decode loop runs on the GPU.** Sampling runs on the GPU, and a one-thread `advance` kernel feeds
  the sampled id back as the next input, appends it to a history buffer, marks it for the repetition
  penalty and moves the position forward. So `stepsPerSubmit` decode steps (default 4) go out in one
  submit and come back in one 16-byte readback, with two batches in flight.
- **No submit hogs the GPU.** The GPU only switches to other work, such as drawing the screen, between
  command buffers, and on a slow integrated GPU one 128-token prefill chunk can take most of a second;
  that froze the display on the development Mac and once tripped macOS's GPU hang watchdog. So the
  engine cuts work into submits of at most `maxSubmitMs` (default 50) of estimated GPU time, from costs
  it measures as it runs and a slow-GPU guess before that. On SmolLM2-135M this made 36–39 submits per
  call instead of 12, with the same decode speed and time to first token.
- **Few, fat dispatches.** At load, q/k/v and gate/up weights are stacked into one matrix each. A decode
  layer is 7 dispatches: RMSNorm fused into the QKV matmul, RoPE fused with the KV-cache write, split-K
  attention + combine, o-proj with the residual add fused in, RMSNorm + gate/up + SiLU in one matmul, and
  down-proj with the residual add. 214 dispatches per token instead of 515.
- **Tuned on the user's GPU while downloading.** Each matmul shape is timed with 8, 16 and 32 lanes per
  weight row on scratch buffers while the weights download (the GPU is idle then); the choice is cached
  per device in OPFS, so later loads skip it.
- **Intermediates are planned, not freed by hand.** Ops allocate placeholders; when a plan is recorded,
  each intermediate's lifetime is the span of dispatches that bind it, and the planner packs them
  largest-first into buffers whose earlier tenants are dead. One pool serves every plan (decode and each
  prefill bucket), since they run one after another on the queue. Prefill attention keeps its scores in
  workgroup memory (online softmax), so no intermediate scales with the context length.
- **Prefill attention is tiled over GQA heads and tokens.** One workgroup takes a tile of query rows,
  every (token, head) pair that shares a kv head, stages each block of keys and then values in workgroup
  memory once for the whole tile, and keeps a vec4 of output per thread with the softmax folded into the
  value pass. All plans of
  SmolLM2-135M together (T = 1, 16, 64, 128) hold 1.8 MiB of intermediates, against 19.3 MiB before.
- **Grammars mask tokens on the GPU sampler.** A JSON Schema compiles to a byte-level pushdown matcher.
  Each step walks a prefix tree of every token's bytes from the current state, pruning a subtree at its
  first rejected byte, and caches the resulting bitmask by state, since generation keeps revisiting the
  same grammar positions. Inside a string with no length bounds, every token of plain bytes (no quote,
  backslash or control character) is allowed whatever follows, so that state's mask is a precomputed
  bitmask plus a walk over the few other tokens: 20–29× less CPU per token on string-heavy schemas
  (0.08 ms on SmolLM2's 49k vocabulary, 0.6 ms on Qwen's 152k). The sampler skips masked ids before top-k / top-p, so truncation applies among
  allowed tokens only. The forward pass needs only the last sampled id, so it is submitted before the
  mask is computed and the CPU work overlaps it.
- **Speculative decoding by prompt lookup (`draftTokens`, default 7).** When the last tokens appeared
  earlier in the prompt or the output, the tokens that followed are guessed and checked in one forward
  pass that samples every row. The draft stands up to the first token the model chose differently.
  Each row draws the random numbers a single-token step at its position would, so sampling with a
  temperature verifies too. K/V of rejected rows is overwritten by the next pass before anything reads
  it, so nothing is rolled back. A verify pass costs more than a decode step (3.5 steps for up to 8
  rows on the Intel GPU, less on an M3), so the engine measures both as it goes and only checks a
  draft that is likely to pay; a check that does not pay backs drafting off (8, 16, 32, then 64
  skipped drafts before the next try). Between passes it decodes as usual, two submits in flight.
  With JSON or schema output, drafts are cut where the grammar rejects them and each row samples
  under its own mask. Measured on SmolLM2-135M, output identical in every case: repeating a paragraph
  2.0× faster on the Intel GPU and 2.2× on an M3; editing code 1.3×; summarising 1.2×; a list schema
  1.4× on the M3; sampled free text 0.96× at worst. Off with `draftTokens: 0`, and not combined with
  the repetition penalty.
- **The prompt prefix is reused, across reloads too.** The engine remembers which tokens' K/V are in
  the cache and only prefills what changed, so a multi-turn chat prefills just the new turn. The
  worker also keeps the last call's K/V in OPFS (`promptCache`, on by default; about 23 KB per token
  on SmolLM2-135M) and restores it at load, so after a page reload a shared system prompt is not
  prefilled again: with a 214-token system prompt, time to first token fell from 1231 to 325 ms on
  the Intel GPU, with the same reply. The file is keyed by model, revision, weight format and f16
  math, written after the reply goes out, and a half-written file is ignored.
- **Several conversations, each with its own cached prefix.** Before a call overwrites the cache, the
  engine copies the K/V it holds into a GPU buffer (GPU to GPU), keeping up to `prefixCacheBytes`
  (default 256 MiB, least recently used dropped first, a saved prefix dropped when a longer one
  extends it). Each call starts from whichever prefix, live or saved, shares the most tokens with its
  prompt, so two chats (or two system prompts) used in turn each prefill only their new turn. Calls
  that overlap queue in the worker instead of failing. Continuing a single chat copies nothing. Gemma
  3 270M Q8_0, two conversations with 650-token system prompts in turn: going back to the first
  restored 660 tokens and gave its first token in 208 ms against 1,846 ms for a full prefill, with
  the same reply, its sliding layers' rings already wrapped; SmolLM2-135M likewise (189 against 366 ms).
- **1× memory while loading.** Weights stream row-chunk by row-chunk from the network into GPU buffers,
  quantized to q4 or q8 on the way if requested. Nothing holds the whole model in CPU memory. F32/F16/BF16
  tensors and every GGUF block type are uploaded raw and converted to q4 or q8 by a compute kernel, bit
  for bit what the CPU converter gives: each rounding is decided by an exact integer comparison on the
  f32 mantissas, not by the GPU's division (`setGpuConvert(false)` keeps it on the CPU). On the
  Intel Mac, cached SmolLM2-360M loads went from 8.3 to 3.5 s (BF16 safetensors) and from 6.6 to 1.4 s
  (Q4_K_M GGUF, mostly Q5_0: llama.cpp's fallback for rows that are not a multiple of 256), and as q8
  from 15.7 to 2.9 s (BF16), with the same replies.
- **Images for Gemma 3.** A SigLIP vision tower and Gemma 3's projector run on the same kernels as
  the text model plus LayerNorm, GELU and 4 × 4 pooling, and attention takes a bidirectional span from
  the step buffer: queries inside it see every key of the span (all 4096 patches in the vision tower,
  each image's 256 tokens in the prompt). See Models for the check against a NumPy reference.
- **Span attention.** When every query sees the same keys, as all 4096 patches of the vision tower
  do, attention runs as a register-tiled GEMM with the softmax folded in: a 128-thread workgroup takes
  32 query rows of one head and walks the keys 64 at a time, each thread holding a 4 × 4 block of
  scores built from Q and K chunks staged in workgroup memory (as the matmul kernels do). Row maxima
  go through workgroup memory so the threads of a row share one running max, each thread exponentiates
  only its own scores, and the probabilities and values are staged 16 keys at a time for the weighted
  sum. Nothing of the 4096 × 4096 score matrix reaches global memory. On the Intel Mac a vision layer's
  attention takes 0.88 s, against 2.76 s with the tiled prefill kernel (2.05 s at its best tile), and
  an image encodes in about 46 s instead of 90; the features are unchanged (cosine 0.9998 to the
  reference). Eight rows per thread was slower (about 4 s: its registers spill on this GPU).
- **A profiler.** `await llm.startProfile()`, any calls, then `await llm.stopProfile()` gives the
  time per kernel and shape (`gemm/q4 3072x576`, `attention/split/f16`, ...), most expensive first.
  On WebGPU each dispatch runs in a compute pass of its own with timestamp queries (the device must
  offer `timestamp-query`); on the CPU backend each kernel is timed with the wall clock. The bench
  page (`/bench.html`) shows one for an uncached call. SmolLM2-135M on the Intel Mac, a 39-token
  prompt and 32 tokens: 601 ms of GPU time, led by the fused MLP GEMV (137 ms over 930 calls) and
  decode attention (110 ms); on the CPU backend 3.4 s, led by attention in TypeScript (875 ms).
- **A CPU fallback.** Every kernel the models use also exists in TypeScript, attached to its WGSL spec
  with the same bindings and uniform layout, and a `CpuBackend` runs them on the model's worker
  thread. `load` uses it when the browser has no WebGPU or no adapter (`device: "auto"`, the
  default), or when asked (`device: "cpu"`); `info.backend` says which ran. The whole engine works
  unchanged on it: grammars, saved prefixes, the prompt cache. The kernel check page runs every check
  against it too (`/kernels.html?backend=cpu`, 763 of 763 passing; the f16 matmul variants need
  shader-f16), and Node tests cover it in CI. Matmuls run in WASM SIMD (`packages/kernels/src/simd.wat`,
  4 KB compiled): every CPU buffer lives in one WebAssembly memory, so the kernels read weights,
  activations and scratch in place. q4 and q8 matmuls take integer dot products
  (`i32x4.dot_i16x8_s`) of the codes with x quantized to int16 per 32-value block (a scale per
  block, so each value is off by at most 1/65534 of its block's largest; the min term of affine
  weights uses exact float block sums): 7.5–8.4 GMAC/s for q4 and 7.5–10 for q8 on one thread of the
  Intel Mac, against 4.4–4.9 with f32x4 dot products from the codes and 0.24–0.42 in TypeScript
  (which remains the path without WASM SIMD). Prefill builds plans of the exact prompt
  length instead of padded buckets, and draft tokens are off by default (a checked row costs as much as
  a decoded one). SmolLM2-135M gives the GPU's tokens in q4 and q8, decoding at 18 tok/s on the Intel
  Mac (it was 1–2 with TypeScript matmuls), with a 42-token prompt in 0.4 s. On a cross-origin
  isolated page (COOP/COEP headers, as the playground's dev server sends) the heap is a shared
  memory and large matmuls (a million multiply-adds and up) split their rows across helper workers
  running the same module (`load(id, { cpuThreads })`, default the browser's hardwareConcurrency up
  to 4); the model's thread posts each job in a control block and waits on `Atomics.wait`. Results
  are bit-identical to one thread. On the Intel Mac (2 cores, 4 threads): a 64 × 3072 × 576 matmul
  27 → 12 ms in f32 and 27 → 16 ms in q4; SmolLM2-135M's first token 1.77 → 1.36 s and decode 16.7 →
  17.7 tok/s (same tokens; decode's small matmuls stay on one thread, where waking helpers would
  cost more than they save). Without isolation, or with the engine on a page's main thread, it
  stays on one thread. Whether a split pays is measured, not assumed: per matmul shape the CPU backend
  times both ways twice, then runs the faster and times the slower again now and then (every 16
  calls, doubling to 1024 while it stays slower; `ThreadTuner`). In a browser reporting 2 cores with
  the page in the background, waking the helper had made SmolLM2-135M decode at 8.6 tok/s on two
  threads against 55 on one; tuned, two threads give 57 tok/s with a 0.33 s first token (prompt
  matmuls keep the split, decode's drop it). Prefill matmuls take four x rows per pass over a q4 or q8 weight block
  (`gemm4_q4i`, `gemm4_q8i`: each block's codes widened once for the four, the same operations per
  row as one at a time): the 64 × 3072 × 576 matmul takes 11.6 ms in q4 and 13.0 ms in q8 on one
  thread, against 25 and 24 ms with one row at a time in f32x4. Attention runs in WASM too (`att_scores`, `att_wsum` for f32 and f16 caches: queries
  against a run of cached keys, ring wraparound included, f16 widened by an exact
  shift-and-rescale), with masking and the softmax in TypeScript: 875 → 139 ms of a profiled
  SmolLM2-135M call. With these, SmolLM2-135M on the CPU backend: first token 0.76 s and 36.7 tok/s
  on one thread, 0.63 s and 37.7 tok/s on four, the GPU's 64 greedy tokens exactly. The CPU kernel
  checks allow the integer dot products an error of 2e-4 of the outputs' size (the worst check is at
  43% of that). Sampling computes the adjusted logits once and runs the GPU kernel's 24-step top-k / top-p
  bisection over only the ids still between its bounds: per token on SmolLM2-135M's 49,152 logits,
  greedy 1.6 → 0.6 ms and top-k 40 with top-p 0.9 28 → 4.4 ms (decode 19.5 → 32 tok/s, same
  tokens). Splitting decode's small matmuls across threads was measured too and made them slower.
  The RMS norm in front of a q4/q8 matmul and the int16 quantization of its x rows run in WASM
  (`quant_x`), and a matmul with no epilogue writes y in place: that TypeScript was 5 ms of every
  25 ms forward pass, and decode over 96 tokens went from 32–35 to 35–38.5 tok/s (first token
  0.76–0.89 → 0.67–0.75 s, same 64 greedy tokens as the GPU). Attention takes all the query heads of a KV head per call
  (`att_scores`, `att_wsum`), so each cache row is read and widened from f16 once per group instead
  of once per head: for SmolLM2-135M's 9 heads over 3 KV heads, 196 → 91 µs per layer at 100
  positions and 1078 → 468 µs at 500. Other architectures give the GPU's 48 greedy tokens on the CPU
  too, on one thread of the Intel Mac (GPU in brackets): Gemma 3 270M-it q8 21 tok/s (30), Qwen2.5-0.5B-Instruct
  q4 16.5 tok/s (30, its biased q/k/v included), Qwen3-0.6B q4 13.5 tok/s (28, per-head q/k norms).
- **GGUF weights held exactly.** Q4_1 and Q4_K blocks are code·scale + min per 32 values with 4-bit
  codes, and Q5_x with 5-bit codes, so Kiln's q4 and q8 planes plus one f32 min per block hold them
  as llama.cpp dequantizes them, where they used to be dequantized and rounded a second time. The GPU
  writes these planes straight from the raw blocks (the CPU's bits, checked over 7.7M codes per
  type), and the matmul, GEMV, GEMM and embedding kernels take an affine variant: GEMV folds the
  mins in as min × (sum of the x block), once per block for all of a thread's rows. With the new
  `quant: "auto"` default, granite-4.0-350m-Q4_K_M.gguf gave the same 40 greedy tokens as `"f32"`
  (which holds the dequantized values themselves), where the old all-q4 load diverged at token 8. The cost
  is bytes: 301 MB of weights against 236 MB, since its Q6_K tensors (the head among them) now take 8
  bits, and decode at 24.1 against 27.9 tok/s; loading took 2.0 s against 3.4 s, with nothing left to
  round. `quant: "q4"` keeps the old behaviour.
- **Why a reply ended.** `stats.stopReason` is `"stop"` (an end-of-turn or stop token),
  `"maxTokens"`, `"context"` (the reply filled the context before maxTokens), `"format"` (the
  structured value is complete) or `"abort"` (the signal fired or the caller stopped reading), the
  same with and without draft tokens.
- **Context sized to the model.** The default `maxSeq` is the model's trained context
  (`max_position_embeddings`, GGUF `context_length`), at most 8192 tokens and 512 MiB of KV cache,
  where it was 2048 for every model. `info.contextLength` tells an app how far a larger `maxSeq` can
  go, and a prompt that does not fit says so. Sliding-window layers (Gemma 3: 29 of 34 in 4B, window
  1024) keep a ring of window + 256 slots (room for an image's 256 tokens prefilled at once), key j in slot j % slots, instead of a row per position:
  Gemma 3 4B gets 8192 tokens in 305 MiB, where a full cache per layer held 3840 tokens in 510 MB.
  Same greedy tokens as the full cache on a 1,518-token prompt (the ring wraps after 1,152), same
  prefill time, and the decode step does not grow with the slots (attention 4.0 ms at position 256
  with 3840, 4.8 with 2048). Reusing a cached prefix stays exact across the ring: a second question
  on the same 1,500 tokens reused 1,494 of them (6.6 s against 88.9 s) with the same tokens as a
  fresh prefill, and a prefix too far behind the ring's oldest key is prefilled from the start.
- **Converted weights are cached too.** After a load converts float weights to q4 or q8, the planes,
  scales and f32 tensors are read back from the GPU and written to OPFS beside the download; the next
  load writes those bytes straight to the GPU and never reads the original file. Reading the 257 MB
  BF16 file was three quarters of a cached SmolLM2-135M load; the converted weights are 80 MB. In one
  session on the Intel Mac (which ran slower that day than for the 1.8 s quoted below), cached loads
  took 1.0–1.9 s with the entry against 2.0–3.4 s without, and SmolVLM-256M (its vision tower's q8
  weights are cached the same way) 3.0 s against 6.1 s, with identical greedy replies, to an image
  question too. The load that writes the entry is slower once (SmolVLM: 8.8 s against 6.1 s). An entry
  is only kept where it is at most three quarters of its source, so f32 loads and a GGUF's own
  quantized tensors are read from the original as before; it is named by repo, revision, server and
  setting, stamped with the tensor list it was made from, and made again when either changes, when
  it cannot be read back, or when a write to it fell short. `load(id, { weightCache: false })` keeps
  only the download; `listCache()` lists the entries as `converted|…`.
- **Config, tokenizer and file listing are cached too.** The small files a load reads (config.json,
  tokenizer.json, the repo's file listing, a missing optional file's 404, Kokoro's voices) are kept in
  OPFS and answered from there without waiting for the network, so a cached model loads with the server
  unreachable (checked by loading SmolLM2-135M through a local mirror and stopping the mirror). Once
  per session each is compared with the server's in the background and replaced if it changed, so an
  updated repo shows from the following load on. On the Intel Mac's connection the five requests of a
  SmolLM2-135M load took 180–250 ms from the network and 40–60 ms from the cache; whole loads
  alternated 2.0–2.9 s without and 2.1–2.8 s with, a difference inside the run-to-run noise.
  `listCache()` lists the entries as `….hub`.
- **The download is the cache.** In a worker, `CachedHttpSource` writes the download into OPFS and serves
  reads from it as bytes land. Interrupted downloads resume from the last flushed byte recorded in the
  metadata (not the file size, which can include writes lost when a worker is killed), and a cached
  file whose header no longer parses, or whose bytes fail to read back, is downloaded again in place.
  Each file is stored as 1 GiB segments, because Chrome's in-memory storage (embedded browsers,
  incognito) fails writes past 2000 MiB in one file while holding more in total; files cached by
  earlier versions as one piece over 1 GiB are downloaded again. When the store stops taking bytes
  (a full disk), the rest of the file is read over HTTP instead of failing the load, and the next
  load resumes from the cached part. `listCache()` reports what is cached, and the playground shows
  it next to a Clear cache button.

## Develop

Needs Node 22.12 or newer (vite 7, vitest 5).

```bash
npm install
npm test          # unit tests (Node)
npm run typecheck
npm run dev       # playground at http://localhost:5173, kernel check at /kernels.html, profiler at /bench.html
```

### Package and demo builds

```bash
npm run build:package      # packages/kiln-webgpu/dist: one ES module, its worker, bundled .d.ts
npm run build:playground   # examples/playground/dist: static demo, works from any path
```

`kiln-webgpu` is the published form: the workspace packages bundled into one ES module plus the model
worker as its own file, resolved with `new URL(…, import.meta.url)` so it works under Vite, webpack 5
and other bundlers that understand that pattern. Its declarations are the workspace's, with `@kiln/*`
imports rewritten to relative paths, so they type-check without the workspace (checked with
`skipLibCheck: false` in a fresh Vite app installed from the `npm pack` tarball, which then loaded
SmolLM2, chatted and produced schema-constrained JSON). It has not been published; to do so:

```bash
npm publish -w kiln-webgpu
```

The playground build uses relative URLs, so it can be served from a subpath such as a GitHub Pages
project site; served from `/kiln/`, it loads and chats and its kernel-check page passes.
Pages cannot send the COOP/COEP headers the dev server does, so the pages register a service worker
(`public/coi-sw.js`) that adds them to the site's own responses and reload once; served by a plain
static server, the built demo is then cross-origin isolated (its workers too) and still downloads,
caches and chats.

The kernel-check page runs every WGSL kernel against `@kiln/kernels` reference code on the current GPU.
The bench page reports GPU time per step and per kernel (matmuls are labelled with their shape) plus
end-to-end decode speed. The dev server reloads the page whenever a source file changes, which aborts a
load in progress.

## Performance

### Against transformers.js

transformers.js 4.3.2 and Kiln in the same browser on the Intel Mac (Iris Plus 650), the same
inputs, both on WebGPU in fp32 unless noted (transformers.js runs the repos' ONNX exports, Kiln the
original weights). Median of three warm calls after one warm-up; every pair of outputs is identical
unless noted.

| Task | Model | Kiln | transformers.js | |
|---|---|---|---|---|
| text-generation, decode | SmolLM2-135M, q4 | 64 tok/s (74 with draft tokens) | 13.4 tok/s (q4), 14.5 (q4f16) | 4.8× |
| text-generation, decode | Qwen2.5-0.5B, q4 | 31 tok/s | 4.9 tok/s (q4; q4f16 gave "---" repeated) | 6.4× |
| text-generation, decode, CPU | SmolLM2-135M, q8 | 57 tok/s | 24 tok/s (WASM, 2 threads) | 2.4× |
| feature-extraction, 16 sentences | all-MiniLM-L6-v2 | 209 ms | 298 ms | 1.4× |
| sentiment-analysis | DistilBERT SST-2 | 51 ms | 101 ms | 2.0× |
| ner | bert-base-NER | 100 ms | 184 ms | 1.8× |
| question-answering | DistilBERT SQuAD | 63 ms | 99 ms | 1.6× |
| fill-mask | bert-base-uncased | 149 ms | 300 ms | 2.0× |
| zero-shot-classification, 5 labels | DistilBERT MNLI | 260–306 ms | 711 ms | 2.3–2.7× |
| summarization, 64 tokens | DistilBART-CNN-6-6 | 3.4 s | 9.95 s | 2.9× |
| translation | opus-mt-en-de | 1.02 s | 2.85 s | 2.8× |
| text2text-generation | flan-t5-small | 331 ms | 1,198 ms | 3.6× |

LLM outputs differ in q4 (the two 4-bit schemes round differently) and agree in f32. Two of these
rows came out of this comparison: feature extraction was 400 ms (each sentence's forward pass
waited on the previous one's readback; now they are queued back to back and read once), and
fill-mask was 851 ms (the 30k-word output layer ran in TypeScript; now on the GPU). transformers.js
loads faster from its cache (0.9 s against 1.8 s for SmolLM2 when measured; Kiln has since started
caching its converted weights, which about halves its time, and the pair has not been re-measured) and
downloads less (its files are pre-quantized), and its runtime adds 6.7 MB (gzipped) of WASM to
Kiln's 236 KB (the whole package, its worker included).

SmolLM2-135M-Instruct, q4, Intel Gen-9 integrated GPU (Intel Mac), Chrome/Dawn. GPU clocks on this
machine drift 10–15% between runs, so comparisons below are always from one session.

| | Baseline kernels | Current |
|---|---|---|
| Dispatches per decoded token | 515 | 214 |
| Decode step, GPU time (position 256) | 38.5 ms | 15.7 ms |
| Decode step, GPU time (position 1536) | — | 19.1 ms (24.7 ms unfused with an f32 cache) |
| Decode, end to end | 26 tok/s | 59.5 tok/s (4 steps per submit), 71 tok/s (8) |
| 64-token prefill | 1380 ms | 170–180 ms |
| 64-token prefill at position 1024 | 361 ms (per-row attention, scores buffer) | 234 ms |
| 128-token prefill at position 1536 | 808 ms (per-row attention) | 522 ms |
| Intermediates, all plans (T = 1, 16, 64, 128) | 19.3 MiB | 1.8 MiB |
| Time to first token, 18-token chat prompt | 1.4 s | 0.12–0.34 s |
| Time to first token, 82-token prompt sharing 81 tokens with the last call | — | 133 ms (431 ms without reuse) |
| Decode under a JSON schema (greedy, 53 tokens) | — | 33 tok/s, against 39 unconstrained at one step per submit in the same run (a slow-clock run: 46 at 4 per submit) |

How it got there, each step measured in one session with identical greedy output unless noted:

- **Matmul:** walks K in 32-element blocks with interleaved lanes per weight row and stages an x tile in
  workgroup memory shared by every row of the workgroup (conflict-free, coalesced). Prefill decodes each
  weight block once per 8 tokens. 1380 → 170 ms prefill.
- **Attention:** prefill uses 256 threads per (head, token). Decode is split-K ("flash-decoding") over
  64-key splits plus a combine pass, and splits past the current length exit at once. At position 1536:
  33.7 → 25.0 ms per step.
- **GPU-resident decode loop:** 33.5–34.8 tok/s (CPU round trip per token) → 53–54 (4 steps per submit)
  → 58–59 (8).
- **Fusion:** 515 → 214 dispatches per token; 8 steps per submit 58–59 → 71 tok/s. Prefill keeps a
  separate RMSNorm pass, because recomputing 8 row norms per workgroup cost more than the dispatch it
  saved. The fused graph produces the same 128 greedy tokens as the unfused one.
- **f16 KV cache** (pairs packed with `pack2x16float`, no `shader-f16` needed): at position 1536,
  21.8 → 18.8 ms per step, split attention 7.6 → 4.9 ms. Same 128 greedy tokens as the f32 cache.
- **Memory planner:** liveness-based placement in one pool shared by all plans, plus online-softmax
  prefill attention instead of a [T, heads, maxSeq] f32 score buffer (9 MiB at T = 128 for this model,
  and it grows with context and heads). Intermediates 19.3 → 1.8 MiB. The new attention is also faster:
  a 64-token prefill at position 1024 went 361 → 327 ms (attention 216 → 173 ms), at position 0
  180 → 170 ms. Sharing buffers did not change decode time (same session: 16.5 / 17.2 ms per step
  shared, 17.1 / 17.3 ms unshared).
- **Tiled prefill attention:** the per-row kernel ran one workgroup per (head, token), so each query row
  re-read the keys and values it attends to. A first tiled version that only shared the K/V loads was no
  faster at 64 tokens (170 vs 167 ms of attention at position 1024) and slower at 16: the cache had been
  absorbing the re-reads, and the inner loops were bound by workgroup-memory loads. Giving each thread a
  vec4 of output and folding the softmax into the value pass fixed that. Same session, attention / whole
  prefill: 64 tokens at 1024 165 → 72 ms / 329 → 234 ms, 128 at 1536 500 → 209 ms / 808 → 522 ms,
  16 at 1024 45 → 26 ms. Greedy output unchanged. The per-row kernel remains as `setPrefillAttention("rows")`.
- **Prefill GEMM and large-head attention** (measured on Gemma 3 270M, q8: hidden 640, head dim 256,
  262k vocabulary). Prefill matmuls of 64 rows and up use a tiled GEMM: a 64-token × 64-row output tile
  per workgroup and a 4 × 4 block per thread, so each k step is two workgroup-memory vec4 reads for 16
  multiply-adds instead of one per 4 (matmuls in a 128-token chunk 274 → 186 ms). Tiled attention had
  only 4 rows × 8 keys per tile at head dim 256, and each of those 32 dot products ran serially on one
  thread while 224 waited; splitting each across 8 threads took attention in a 128-token chunk at
  position 1024 from 331 to 105 ms. Prefill chunks before the last also skip the output head and
  sampling, whose logits nobody reads. A 1,138-token prompt: 5.1–5.2 → 2.4–2.5 s, with the same
  answer tokens as before and as the NumPy reference. On SmolLM2-135M (head dim 64, untouched
  attention) a 250-token prompt prefills in 516 against 670 ms with the GEMM off, with the same 40
  greedy tokens. The GEMM computes in f32 even with `f16Math`.
- **GEMM with 8 × 4 per thread** (Gemma 3 4B Q4_0, 128-token chunk at position 1024, GPU time per
  layer). The GEMM now runs 128 threads per 64 × 64 tile with an 8 × 4 block each, three vec4 reads
  for 32 multiply-adds per k step: matmuls 350 → 198 ms per layer (gate/up 179 → 106, down 100 → 56,
  qkv 50 → 22), about 120 GFLOP/s. In a sweep of tile and block shapes, 64 × 128 tiles, 8 × 8 or
  16 × 4 blocks, and 256 threads were all slower. The 1,138-token prompt prefills in 65.6 against
  106 s, with the same answer tokens ("BLUE HERON"). Attention is now about 20% of a prefill chunk.
- **GEMM reads issued up front** (Intel Iris Plus 650, the kernels alternated in one page, medians of
  6–8 rounds). Each K block's global reads (x, weight words, scales) are now all issued before the
  workgroup tiles are written, and branch-free: rows past the edge read a clamped row with a zero
  scale instead of sitting behind `if (row < N)`. Gemma 3 4B in a 128-token chunk: gate/up 121 →
  100 ms, down 78 → 65 ms; the vision tower (4096 patches, q8): MLP up 291 → 272 ms, down 286 →
  272 ms, q/k/v/o 96 → 92 ms, and a whole vision layer 1.70 → 1.66 s. Tried and not kept: 128-row
  tiles with 8 × 8 per thread or 256 threads, prefetching the next block into registers, unrolling
  the k loop, f16 tiles or f16 math (f16 math alone was no faster: the loop is limited by workgroup
  memory reads and register pressure, not arithmetic). The 32 accumulators are now separate
  variables with an unrolled epilogue, which on its own changed nothing. Some of the discarded
  variants looked 10–20% faster in short runs; under sustained load, interleaved with the old
  kernel, none were.
- **Attention tiles for head dim 256.** Tiled prefill attention held 4 query rows and 8 keys per
  tile at head dim 256, one output vec4 per thread. Threads can now own several output rows (each
  with its own running max and sum), and Gemma's heads take 8 rows of 4 keys: 29-31 against 35-42 ms
  for a 128-token chunk at position 1024. Larger tiles (16 or 32 rows, 8 or 16 keys) were slower, so
  re-reading K/V per tile was not the cost. Gemma 3 4B, 1,518-token prompt: 86.2 against 91.6 s,
  same tokens. Tiles size themselves to the device's workgroup memory, which is now requested in
  full.
- **Fused q/k/v for qk-norm models, and gated prefill without the gated GEMM** (Gemma 3 4B Q4_0).
  Gemma and Qwen3 normalise each q and k head, so their q/k/v ran as three matmuls followed by an
  input norm, two head norms, two RoPE passes and a cache write. They now stack q/k/v like the
  other models and finish in one kernel (head norms, RoPE, cache write, ring slots included):
  dispatches per decode step 616 → 378. Measuring that change separately (`fuse: { qkv, mlp }`)
  showed the fused gate/up GEMM, whose tiles pair 32 gate rows with 32 up rows, costing 7% of a
  prefill; prefill now runs a plain GEMM over [gate | up] and an elementwise act(gate) · up, and the
  decode GEMV keeps its gated epilogue. 1,138-token prompt 64.3 → 60.2 s, decode step 178 → 174 ms,
  same tokens; Gemma 3 1B still matches the NumPy reference in q8 and q4 on that prompt, its
  640-slot rings wrapping almost twice.
- **Decode GEMV:** the general matmul gives each lane half of a q4 word per step (4 bytes shared by two
  lanes, 4 multiply-adds), and ran at 6.5–8.5 GB/s. One-token matmuls now use a kernel where each lane
  takes whole 32-element blocks: one 16-byte load and 32 multiply-adds for q4, 4 lanes per row,
  neighbouring lanes on neighbouring blocks, and the x chunk staged transposed so those lanes read
  consecutive vec4s of it. Gemma 3 1B q4, decode step at position 256 (GPU time): 99.1 → 55.9 ms
  (gate/up 40.5 → 20.1, the 262144×1152 head 28.0 → 14.8, down 17.1 → 10.1). SmolLM2-135M q4,
  alternating runs: 20.0 → 14.1 ms at position 256 and 25.2 → 19.9 ms at 1536, about 40 → 60 tok/s
  at one step per submit with the page in the background. Greedy output is unchanged on both models,
  and Gemma 1B q4 and q8 still match the NumPy reference token for token. `setGemv(false)` restores
  the general kernel, which forced `lanes`, `f16Math` and subgroups still use.
- **GEMV rows per thread** (Gemma 3 4B Q4_0, decode step at position 256): this GPU reads about 22
  GB/s in a plain streaming kernel, and the GEMV reached 11-15. With one row per thread, each lane
  read 128 bytes of x from workgroup memory per 16-byte weight block. Large matrices now give each
  thread 4 rows (8,192 output rows and up) or 2 rows (K of 4,096 and up) on the x block it read,
  with 8 lanes per row group: step 190.6 → 177-180 ms (gate/up 75.5 → 72, down 43.8 → 36, head 28.0
  → 24.7). Small matrices keep 4 lanes and one row, since several rows per thread made SmolLM2's
  (K = 576) and 4B's 1024-row k/v projections 10-40% slower. Same answer tokens on the 1,138-token
  prompt, SmolLM2's step unchanged.
- **Autotuner:** on this GPU it picks 16 lanes per row for the 576×576 projections and 8 elsewhere; gains
  here are within noise. It exists for GPUs with more cores, where small N leaves most of them idle.
  It tunes the general kernel, so with the GEMV it only matters for prefill and forced variants.
- **Measured and not adopted:** f16 dot products in the matmul (`f16Math: true`) ran at 64–66 tok/s
  against 70–71 with f32 here, with the same greedy tokens. Memory and latency bound the matmuls on this
  GPU, not arithmetic. The option stays for GPUs where it helps.
- **Measured and removed: jump-forward for grammars** (feeding text the grammar forces, such as a key,
  in one 8-row pass instead of sampling it). Optional whitespace leaves little forced: a key, rarely
  more than 2 tokens once the last token is left to sampling. That has to be left, because a forced run
  tokenized on its own splits where the model would not (`city"` + `:` instead of `city` + `":`), and
  the first version, which forced it too, derailed the model into stray newlines. Same run, alternating:
  identical 53-token output, 50 forward passes against 52, and 26.0 against 27.8 tok/s, since the
  8-row pass costs more than the two steps it saves.

### Apple M3 (MacBook Air)

Same model and build, Chromium 152 (Metal 3), one run, every check passing (184 kernel checks, same
greedy ids at every chain length, f16 math and the f32 cache matching 128 of 128 greedy tokens):

| | Apple M3 | Intel Gen-9 (above) |
|---|---|---|
| Decode step, GPU time (position 256 / 1536) | 3.4 / 3.8 ms | 15.7 / 19.1 ms |
| Decode, end to end (1 / 4 / 8 steps per submit) | 302 / 307 / 307 tok/s | 39–53 / 59–69 / 71–74 tok/s |
| Decode under a JSON schema | 241 tok/s | 33 tok/s |
| 64-token prefill at position 1024, tiled / per-row attention | 44 / 74 ms | 234 / 329 ms |
| 128-token prefill at position 1536, tiled / per-row | 98 / 188 ms | 522 / 808 ms |
| Time to first token, 81 of 82 tokens cached / cold | 12 / 69 ms | 110–166 / 373–492 ms |
| Copy bandwidth / fixed cost per dispatch | 82 GB/s / 1.5 µs | 13.8 GB/s / 11 µs |

On the M3, chaining steps barely matters (302 → 307 tok/s): submit and readback latency is small there,
whereas the Intel GPU gains 50% from it. The grammar costs more relative to the step (241 against
302 tok/s), because a 3.4 ms step leaves little to hide the mask computation behind.

Gemma 3 4B Q4_0 on the M3 (commit 04c4c1a, one run, 520 kernel checks passing with the subgroup
variants): loads from the cache in 2.4 s (about 10 s on the Intel GPU), prefills a 2,005-token prompt
in 11.2 s (about 179 tok/s, against about 17 on the Intel GPU), decodes at 24.6 tok/s (about 5.5),
and answers the long-range question correctly. The rows-per-thread GEMV (5afb94c) was A/B tested
there in alternating runs: identical output, SmolLM2 faster in every pair (378 against 336 tok/s,
best of three), Gemma 4B faster in two of three pairs. The machine was swapping heavily, which
halved the 4B numbers partway through, so only adjacent runs compare.

Where the remaining time goes at position 256 on the Intel GPU: the tables above predate the decode
GEMV, which took the step from 20.0 to 14.1 ms; the matmuls now read at 11–15 GB/s, close to the GPU's
copy bandwidth (13.8 GB/s, read plus write), and each of the 214 dispatches has about 11 µs of fixed
cost.

## Status

Verified on this GPU: all kernel checks pass (817 now that Chrome exposes shader-f16 on it, including q4 and q8 weights, symmetric and affine, in every matmul variant, bidirectional-span attention, LayerNorm and pooling, the decode GEMV and the prefill GEMM, both prefill attention kernels, fused variants, f16 cache paths,
sliding-window attention, GELU gating, grammar masks and sampling distributions). SmolLM2-135M-Instruct chats from Hub safetensors (q4, f32) and from a GGUF
Q4_K_M file.

Models run end to end (greedy answers to fixed questions checked by hand; tool calls where noted):

| Architecture | Model | Source | Intel Iris Plus 650 |
|---|---|---|---|
| `llama` | SmolLM2-135M-Instruct | safetensors, GGUF Q4_K_M | |
| `llama` | TinyLlama-1.1B-Chat | GGUF Q4_K_M | 7.6 tok/s |
| `llama` | Llama-3.2-1B-Instruct (Llama 3 rope scaling) | safetensors (2.5 GB), GGUF Q4_K_M | 12–13 tok/s |
| `qwen2` | Qwen2.5-0.5B-Instruct (tool calls, `toolChoice`) | safetensors | |
| `qwen2` | Qwen2-0.5B merge, two shards | safetensors | |
| `qwen2` | DeepSeek-R1-Distill-Qwen-1.5B (reasons, then answers) | GGUF Q4_K_M | 11 tok/s |
| `qwen3` | Qwen3-0.6B (`toolChoice`) | safetensors | 20–32 tok/s |
| `gemma3_text` | Gemma 3 1B-it (sliding window: recalls a fact 1,100 tokens back) | safetensors (unsloth mirror) | 12–20 tok/s |
| `gemma3_text` | Gemma 3 270M-it, with `quant: "q8"` or `"f32"` | safetensors, GGUF Q8_0 | 25–42 tok/s |
| `granite` | granite-3.3-2b-instruct (`<\|tool_call\|>` calls) | safetensors (5 GB), GGUF Q4_K_M | 5.5 tok/s (q8), 6.4 tok/s (Q4_K_M) |
| `granitemoehybrid` | granite-4.0-350m (tool calls), with `quant: "q8"` | safetensors, GGUF Q8_0 | 26 tok/s; 38.7 tok/s (Q8_0) |
| `granitemoehybrid` | granite-4.0-1b (tool calls) | safetensors (3.3 GB) | 11.5 tok/s (q4) |
| `gemma3` | Gemma 3 4B-it (recalls a fact 1,100 and 3,000 tokens back; images) | GGUF Q4_0 (2.4 GB) + mmproj-F16; safetensors (8.6 GB, the vision tower in the same files) | 177–180 ms GPU per step (about 5.5 tok/s); 24.6 tok/s on an M3 |
| `idefics3` | SmolVLM-256M-Instruct, SmolVLM-500M-Instruct (images in chat) | safetensors | a 640 × 480 photo in about 17 s (13 tiles), 2–4 s without splitting |

Encoder models (BERT, DistilBERT, RoBERTa, from safetensors) run the six text tasks transformers.js
runs on them, with its task names and output shapes. Checked against Python transformers (4.1.1,
PyTorch 1.7) on the CPU backend: identical token ids, final hidden states within 8e-6, and task
scores equal to six decimals; the same outputs on WebGPU through `pipeline()`:

| Task | Model | Kiln | Python transformers |
|---|---|---|---|
| `feature-extraction` (mean, normalized) | all-MiniLM-L6-v2 | max difference 1.3e-7 | |
| `text-classification` | distilbert-base-uncased-finetuned-sst-2-english | POSITIVE 0.999807 | POSITIVE 0.999807 |
| `token-classification` | dslim/bert-base-NER | Wolfgang B-PER 0.999131, Berlin B-LOC 0.999534 | same |
| `question-answering` | distilbert-base-cased-distilled-squad | "a nice puppet" | hidden states within 4e-6 |
| `zero-shot-classification` | typeform/distilbert-base-uncased-mnli | phone 0.870278, computer 0.095473 … | same |
| `fill-mask` | bert-base-uncased | life 0.109333, survival 0.039419 … | same |
| `fill-mask` | roberta-base | Paris 0.903619, Lyon 0.080292 … | Paris 0.903627, Lyon 0.080292 … |

The encoder reuses the vision tower's pieces (biased matmuls, LayerNorm, attention over a
bidirectional span) plus an exact erf GELU kernel; heads (pooler, classifier, QA span, masked-LM)
run in TypeScript on the final hidden states. Tokenizers: WordPiece from tokenizer.json or vocab.txt,
and RoBERTa's byte-level BPE, both matching Python tokenizers on 56 test encodings. Not yet: Unigram
tokenizers (XLM-R), ModernBERT, MPNet, DeBERTa, and batching several inputs into one forward pass.

Encoder-decoder models run `summarization`, `translation` and `text2text-generation` with
transformers.js's settings: greedy search (transformers.js ignores `num_beams` too), the model's
`min_length`, `no_repeat_ngram_size`, forced first and last tokens, its config prefix, and 256 new
tokens unless told otherwise. The encoder runs once and writes every decoder layer's cross-attention
K/V; each decoder step reads them through a step block whose span covers every position.

| Family | Checked with | Result |
|---|---|---|
| BART (`facebook/bart-base`) | Python transformers | logits within 3.2e-5 over 12 steps, greedy output identical |
| DistilBART (`sshleifer/distilbart-cnn-6-6`, 2020 pickle checkpoint) | transformers.js (`Xenova/distilbart-cnn-6-6`, fp32) | the same summary; 3.2 s on WebGPU against 22 s for transformers.js on WASM |
| Marian (`Helsinki-NLP/opus-mt-en-de`, source.spm + vocab.json) | transformers.js (`Xenova/opus-mt-en-de`) | the same German; 1.1 s against 4.2 s |
| T5 (`google/flan-t5-small`) | an independent NumPy forward pass over the safetensors; transformers.js | logits within 7.8e-6 over 9 steps, outputs identical to both |
| NLLB (`facebook/nllb-200-distilled-600M`, q8) | by hand | "La maison est merveilleuse. J'aimerais l'acheter, mais elle est trop chère." |
| Whisper (`openai/whisper-tiny.en`, `whisper-tiny`, `whisper-base`) | transformers.js (`Xenova/whisper-*`, fp32) | the JFK clip transcribed to the same text by all three; log-mel features within 1.2e-7 of transformers.js's |
| Whisper word timestamps (`whisper-tiny.en`, `whisper-base`) | transformers.js (`onnx-community/whisper-*_timestamped`, fp32) | the same words and start / end times in all six runs (single window, 30 s chunks, 10 s chunks) |
| wav2vec2 CTC (`facebook/wav2vec2-base-960h`) | an independent NumPy (float64) forward pass over the safetensors; transformers.js | logits within 6.3e-4 on all 549 frames of the JFK clip, the same token in every frame; transformers.js (fp32 WebGPU) printed "FELOW" for "FELLOW" (two L frames three blank frames apart) |
| wav2vec2 large (`facebook/wav2vec2-large-robust-ft-libri-960h`: LayerNorm front end, pre-LayerNorm layers) | the NumPy forward pass | logits within 3.1e-4 over 149 frames, the same token in every frame |
| wav2vec2 classifier (`superb/wav2vec2-base-superb-ks`, weighted layer sum) | transformers.js (`Xenova/wav2vec2-base-superb-ks`, fp32) | the same five labels, scores within 2e-7 |
| Image preprocessing (all six PIL filters; DeiT and CLIP pipelines) | PIL 8 + numpy | resized bytes identical; pixel values identical (float32) |
| CLIP tokenizer (`openai/clip-vit-base-patch32`) | transformers.js | the same ids for 8 texts (punctuation, accents, CJK, emoji, long words) |
| ResNet-50 (`microsoft/resnet-50`), ConvNeXt (`facebook/convnext-tiny-224`) | transformers.js (`Xenova/*`, fp32) on the same pixel values | every uniquely labelled class within 1.7e-8 (ResNet) and 4.7e-7 (ConvNeXt); 337 / 374 ms end to end against 370 / 434 ms |
| VITS text-to-speech (`facebook/mms-tts-eng`) | transformers.js (`Xenova/mms-tts-eng`, fp32), its two noise nodes set to zero, Kiln at zero noise | the same 33,024 samples, within 1.4e-4 (peak 0.87); 617 ms end to end against 1.33 s (WebGPU) |
| Kokoro (`hexgrad/Kokoro-82M`) | transformers.js (`onnx-community/Kokoro-82M-v1.0-ONNX`, fp32, which has no noise), Kiln without noise, the same phonemes and style | the same durations and lengths on 3 sentences; given the export's harmonic spectrum, the vocoder within 9.1e-4 and the audio at 0.997–0.999 correlation; from the text, 0.979–0.988 (float32 phases of the sine source). The export's speech is 1.5× PyTorch's (an iSTFT normalization traced for one frame). 1.5 s against 2.5 s, 3.8 s against 6.3 s |
| SpeechT5 text-to-speech (`microsoft/speecht5_tts` + `microsoft/speecht5_hifigan`) | transformers.js (`Xenova/speecht5_tts`, fp32), its prenet dropout rewritten to keep every unit, Kiln with `prenet_dropout: "keep"` | the same ids; the same 122,880 samples, within 1e-5 over the first 80 frames, then drifting as frames feed back (mean difference 3.3e-4). With dropout: 1.9 s against 6.4 s (WebGPU) for "Hello, my dog is cute.", 3.6 s against 12.7 s for a 44-character sentence |
| ViT + GPT-2 captioning (`nlpconnect/vit-gpt2-image-captioning`) | transformers.js (`Xenova/vit-gpt2-image-captioning`, fp32) on the same pixel values | the same caption on all three photos tried |
| Florence-2 (`microsoft/Florence-2-base-ft`) | transformers.js (`onnx-community/Florence-2-base-ft`, fp32) on the same 768 × 768 pixels | 3 of 20 outputs identical (4 photos × caption, detailed caption, detection, grounding, OCR): its export fixed the channel-attention scale at a 224 × 224 trace, 3.4× too large at 768 (see below). With that scale given to Kiln too, the image tokens agree within 2.6e-4 and all 20 outputs are identical, every location token included. 2.9 s against 5.8 s for a caption, 3.0 s against 6.1 s for detection (WebGPU) |
| Donut DocVQA (`naver-clova-ix/donut-base-finetuned-docvqa`) | transformers.js 4.3.2 (`Xenova/donut-base-finetuned-docvqa`, fp32) on the same pixel values, and its own pipeline | the same answers in all 8 questions on 5 pages (invoice number, total, due date, receipt total and store, balance due, schedule time, book title), from both; the same prompt ids; encoder output within 8e-5 for 99% of values (median 7e-7; at most 0.11, in an outlier channel near −91). 16.9–18.3 s against 33.1–34.3 s for a new page; 1.3 s against 31–34 s for another question on the same page |
| Donut image processing | PIL 8 + numpy (transformers v4's steps) | pixel values identical (float32), including thumbnails that PIL reduces before resampling |
| Gemma 3 4B with images (`unsloth/gemma-3-4b-it`, the transformers checkpoint) | transformers.js (`onnx-community/gemma-3-4b-it-ONNX`: vision encoder fp32, decoder q4) on the same 896 × 896 pixels | image features at a mean cosine similarity of 0.9995 (lowest token 0.988; Kiln's tower in q8); the same sensible descriptions of 3 photos in different words (different quantizations). 36–42 s against 49–61 s for a 40-token answer with the image; its q4f16 decoder printed only `<pad>` on this GPU |
| SmolVLM (`HuggingFaceTB/SmolVLM-256M-Instruct`) | transformers.js (the repo's ONNX export, fp32) on the same tiles and prompt | image features within 2.1e-4 (values up to 96); the same greedy replies in all 17 runs, 9 photo-question pairs with image splitting (13 to 17 tiles) and 8 without. 17.4 s against 25.0 s for a short answer with splitting, 2.3 s against 4.3 s without (WebGPU) |
| DINOv2 (`facebook/dinov2-small-imagenet1k-1-layer`) | transformers.js (`Xenova/dinov2-small-imagenet1k-1-layer`, fp32) on the same pixel values | within 6e-3: its export resizes position embeddings bilinearly, Python bicubically (Kiln); with its resize, within 3.2e-7. 383 ms end to end against 683 ms |
| Swin2SR (`caidas/swin2SR-classical-sr-x2-64`) | transformers.js (`Xenova/swin2SR-classical-sr-x2-64`, fp32) on the same pixel values | 4 of 147,456 output bytes one level apart (128 × 96 → 256 × 192); 3.4 s end to end against 10.4 s |
| SegFormer (`nvidia/segformer-b0-finetuned-ade-512-512`) | transformers.js (`Xenova/segformer-b0-finetuned-ade-512-512`, fp32) on the same pixel values | the same 14 classes on the cats photo, all 307,200 pixels the same class; 0.75 s end to end against 3.9 s |
| RMBG-1.4 (`briaai/RMBG-1.4`) | transformers.js (the repo's ONNX export, fp32) on the same pixel values | 3 of 1,048,576 matte bytes one level apart; 2.9 s end to end against 4.0 s |
| DETR panoptic (`facebook/detr-resnet-50-panoptic`) | transformers.js (`Xenova/detr-resnet-50-panoptic`, fp32) on the same pixel values | the same five segments on the cats photo, scores within 1.4e-6, one pixel of 307,200 assigned differently; 2.4 s end to end against 28.2 s |
| Depth Anything V2 (`depth-anything/Depth-Anything-V2-Small-hf`) | transformers.js (`onnx-community/depth-anything-v2-small`, fp32) on the same pixel values | depth within 1.7e-5 relative on the cats photo; 20 of 307,200 grey bytes one level apart; 3.1 s end to end against 7.4 s |
| DETR (`facebook/detr-resnet-50`) | transformers.js (`Xenova/detr-resnet-50`, fp32) on the same pixel values | the same five detections and labels on the cats photo; scores within 2.8e-7, boxes within 2.3e-6; 1.57 s end to end against 4.02 s |
| YOLOS (`hustvl/yolos-tiny`, `yolos-small`) | transformers.js (`Xenova/yolos-*`, fp32) on the same pixel values | the same detections and labels on the cats photo; scores within 8.2e-6, boxes within 3e-7 |
| OWLv2 (`google/owlv2-base-patch16-ensemble`) | Python transformers' documented output; transformers.js 3.8.1 (`Xenova/owlv2-base-patch16-ensemble`, fp32) on the same pixel values | cats photo: 0.665 and 0.613 against Python's 0.665 and 0.614, boxes the same to the pixel. Against transformers.js on 3 photos: logits within 0.0066 (they span −40 to 1.5; scores within 5.2e-6), boxes within 3.4e-4, the same best patch. 5.4 s against 32.7 s end to end; new labels on the same image 59 ms. transformers.js 4.3.2 cannot load either OWL export (see below) |
| OWL-ViT (`google/owlvit-base-patch32`) | the same | cats photo: 0.716 and 0.706 against Python's 0.717 and 0.707; against transformers.js 3.8.1 on 2 photos, logits within 4.3e-4, boxes within 3.5e-5. 639 ms against 1,674 ms |
| OWLv2 image processing | numpy + scipy 1.7 (transformers v4's steps) | pixel values identical (float32) on 7 images from 23 × 37 to 4000 × 3000 |
| Position-embedding interpolation | torch 1.7.1 `interpolate(mode="bicubic")` | within 1e-5 |
| CLIP zero-shot (`openai/clip-vit-base-patch32`) | transformers.js (`Xenova/clip-vit-base-patch32`, fp32) on the same pixel values | six labels' probabilities within 1.8e-7, the labels run as one batch |
| ViT (`google/vit-base-patch16-224`), DeiT (`facebook/deit-base-distilled-patch16-224`) | transformers.js (`Xenova/*`, fp32) on the same pixel values | all 1,000 class probabilities within 2.2e-6 (ViT) and 7.8e-7 (DeiT) |

Whisper (`automatic-speech-recognition`) takes 16 kHz samples, a URL, a Blob or encoded bytes (WAV
parsed directly; other formats decoded and resampled by the page's Web Audio API). Its log-mel
features follow transformers' WhisperFeatureExtractor (a mixed-radix FFT for the 400-point frames,
slaney mel filters); the two convolutions run as matmuls over im2col rows; decoding starts from
transformers.js's prompt (start, language and task for multilingual models, no-timestamps) with the
model's suppressed tokens. `return_timestamps: true` returns segments (`chunks: [{ timestamp:
[start, end], text }]`) as transformers.js computes them: its timestamp rule (pairs, and a timestamp
whenever the timestamp tokens' total probability beats the best text token, decided from 1,501
timestamp logits and the chosen text token's logit read back with the step's token), its seek loop
(a pass that stops on a segment boundary early is followed by another from there), and its segment
decoding. `chunk_length_s` / `stride_length_s` cut long audio into overlapping windows merged where
their tokens agree; audio past 30 s is chunked in 30 s windows by default (transformers.js keeps only
the first 30 s unless `chunk_length_s` is set). Against transformers.js with whisper-tiny.en and
whisper-base on the JFK clip, once alone and three times back to back (33 s): the same text,
segments and timestamps in all eight runs (timestamps, 30 s chunks, both, 10 s chunks with
timestamps); 1.6–4.6 s a case against 5.4–16.4 s for tiny.en, 2.4–7.4 s against 9.1–27.4 s for base
(one warm call each, backgrounded pane). `return_timestamps: "word"` returns words (`chunks: [{
text, timestamp: [start, end] }]`) timed as transformers.js times them: the decoder records the
cross-attention weights of the model's `alignment_heads` at each step (one small kernel per layer,
left out of the plan when words are not asked for), and after the pass they are normalized over
tokens, median-filtered over time, averaged over heads and aligned to the audio frames by dynamic
time warping; tokens are grouped into words with punctuation attached (per character for Chinese,
Japanese, Thai, Lao and Burmese when `language` names them, as transformers does; transformers.js's
pipeline always splits on spaces). Against transformers.js's
`_timestamped` exports of the same two models: the same words and times in all six runs (the 11 s
clip, the 33 s clip in 30 s chunks, 10 s chunks); 1.4–4.6 s a case against 4.1–14.2 s for tiny.en,
2.2–6.8 s against 6.4–21.2 s for base. Not yet: a stable speed measurement (runs in a backgrounded browser pane varied 5× between
calls; Kiln took 6.6–7.3 s for the 11 s clip against 10.5–18 s for transformers.js there). Encoder-
decoder decoding picks each token on the backend with the LLM engine's sampling kernel (greedy,
under a banned-token mask built from the logits processors), so a step reads back 4 bytes instead
of the whole vocabulary: Whisper's per-token cost in that pane went from about 120 ms to too small to
measure (1, 8, 16 and 24 new tokens took the same time), with identical outputs for BART, T5,
Marian and Whisper.

wav2vec2 and HuBERT run on raw samples (normalized when the preprocessor says so). The conv front
end is im2col rows times the tuned matmul per layer, with GroupNorm (a per-channel kernel) after the
first layer or LayerNorm after each; the grouped positional convolution (128 taps, 16 groups) is one
kernel over weights stored tap-major, with weight norm folded in at load. CTC decoding takes each
frame's best token and decodes as transformers.js's CTC decoder does; classifiers average the
frames, then project and classify in TypeScript (the projector is affine, so averaging first is
exact), over a weighted sum of every layer's output accumulated on the GPU when the model uses one.
On WebGPU in the backgrounded pane (one warm call each, so rough): the 11 s clip in 1.7 s against
6.3 s for transformers.js, and keyword spotting on 1 s in 238 ms against 482 ms. On the CPU backend
the clip takes 33 s, mostly in the front end's convolutions. HuBERT shares wav2vec2's layers and tensor names and loads
through the same code, but has not been checked against a reference yet.

Images (`image-classification`, `image-feature-extraction`) run ViT and DeiT checkpoints. The
preprocessing is transformers' (Python's, not transformers.js's): resizing reimplements PIL's
ImagingResample with its fixed-point weights and byte rounding after each pass, so a resized image
equals PIL's byte for byte for all six filters, and rescale / normalize round to float32 as numpy
does. transformers.js in a browser resizes with a canvas instead, so its pixels differ slightly from
Python's. Images are decoded in the worker (createImageBitmap, no color conversion). The patch
convolution is one matmul over patch rows; the CLS and distillation tokens, position embeddings and
patch bias are one precomputed addend; layers are pre-norm with full span attention, and the
classifier (DeiT with a teacher: the mean of its two heads) runs on the CPU from the CLS rows. On the
tiger photo, fed the same pixel values: every class probability within 2.2e-6 of transformers.js for
ViT and 7.8e-7 for DeiT. One warm call took 364 ms (ViT) and 474 ms (DeiT) in Kiln, decoding and
preprocessing included, against 1.5 s and 0.8 s for transformers.js's model call alone (rough,
backgrounded pane). The pooled output (`pool: true`) has not been checked against a reference.

CLIP (`zero-shot-image-classification`) runs its vision tower on the same code (LayerNorm after the
embeddings, and post-LayerNorm plus the visual projection on the pooled token, on the CPU) and its
text tower as a causal transformer that runs every label in one batch (each padded to the longest
after its end token, which the causal mask keeps from touching it, through a block-causal attention
kernel), queued on the GPU alongside the image; the text embedding is pooled at the end
token (or, for legacy configs, the highest id, as transformers does), projected, normalized and
cached per text, so asking about the same labels again only runs the image tower. Scores are
exp(logit_scale) times the cosine similarity, softmaxed over the labels, with transformers.js's
default template "This is a photo of {}". CLIP's tokenizer is its own small class: NFC, whitespace
runs to one space, lowercase, CLIP's split (whitespace dropped), byte-level mapping, BPE with `</w>`
word ends. On the tiger photo with six labels, fed the same pixel values: every probability within
2.2e-7 of transformers.js (1.8e-7 with batched labels). Both libraries' own pipelines, image URL and
labels in, median of five warm calls on WebGPU: 413 ms in Kiln against 786 ms for transformers.js with
six new labels, 516 ms against 855 ms with twenty (357 ms when the labels are cached). Before
batching, Kiln ran labels one at a time and took 0.4–1.2 s.

Loading CLIP's `pytorch_model.bin` showed three loader problems, now fixed. The cache downloads a
file front to back, but a zip keeps its directory at the end and a small header before every entry,
so the header parse waited for nearly the whole file with no progress shown; reads ahead of the
download now go to the network on their own (small ones anywhere past it, large ones 64 MiB past
it). A tensor saved as a transposed view (CLIP's projections) was refused; it is now read and
transposed when loaded on the CPU (GPU-loaded weights still must be contiguous). And an unsupported
tensor made the loader take the cached file for corrupt and download it again; it now fails at once.

Object detection (`object-detection`) runs YOLOS (`hustvl/yolos-*`) on the same ViT code. The image
keeps its aspect ratio (transformers' YOLOS resize: shortest edge, capped longest edge, both sides
rounded down to multiples of 16), so each patch grid gets its own plan, and the position embeddings
(and yolos-small's and -base's per-layer ones) are fitted to it with torch's bicubic interpolation
(checked against torch). The 100 detection tokens go first in the sequence, since full attention
does not depend on order, so the class and box MLPs run on the GPU over just those rows and only
their outputs are read back. Post-processing is transformers.js's: the best class per token unless
it is "no object", its softmax score against `threshold` (default 0.9), boxes from center and size
to corners in pixels of the original image. On the COCO cats photo, fed the same pixel values as
transformers.js (`Xenova/yolos-*`, fp32): the same five detections and labels, scores within 8.2e-6
(tiny) and 3.2e-7 (small), boxes within 3e-7. Both pipelines end to end with yolos-tiny (median of
five warm calls, WebGPU, in the same backgrounded pane): 0.59 s in Kiln against 24.6 s for
transformers.js, whose model call alone took 26 s at either input size.

DETR (`facebook/detr-resnet-50`, `-101`) is the default detector, as it is in transformers.js. Its
ResNet backbone runs on a 2-D convolution kernel that is the GEMM's tiling with the image patches
gathered as the tiles load (implicit GEMM: no im2col copy), bias, residual add and ReLU in the
epilogue, and each frozen batch norm folded into its convolution when loading. The encoder and
decoder reuse the attention and matmul kernels, with DETR's sine position embeddings computed for
each feature-map size. The image is resized as transformers' DETR processor does it (shortest edge
800, longest at most 1333). On the cats photo, fed the same pixel values as transformers.js
(`Xenova/detr-resnet-50`, fp32): the same five detections and labels, scores within 2.8e-7, boxes
within 2.3e-6 of the image size. Both pipelines end to end (median of five warm calls, WebGPU,
same backgrounded pane): 1.57 s in Kiln against 4.02 s for transformers.js. Of Kiln's 1.18 s of GPU
time, about 1 s is the backbone's convolutions, which run at the matmul kernel's rate.

Zero-shot object detection (`zero-shot-object-detection`) runs OWL-ViT (`google/owlvit-*`, the
default being `owlvit-base-patch32` as in transformers.js) and OWLv2 (`google/owlv2-*`): boxes for
any labels.

```js
const detector = await pipeline("zero-shot-object-detection", "google/owlv2-base-patch16-ensemble");
await detector("cats.jpg", ["a photo of a cat", "a photo of a dog"]);
// [{ score: 0.665, label: "a photo of a cat", box: { xmin: 6, ymin: 51, xmax: 326, ymax: 473 } }, ...]
```

The image tower is a CLIP tower on the ViT code; its heads run in the same GPU plan: every token's
post-LayerNorm, the patches times the class token (a new kernel), LayerNorm, the box MLP with the
box bias, and the class head's three linears as one matmul. The labels go through CLIP's text
tower in one batch, and each patch's class embedding meets them on the CPU, so new labels on an
image seen recently (the last four are kept) skip the image: 59 ms instead of 5.4 s with OWLv2.
OWLv2 preprocesses as transformers v4 and the original implementation do: padded at the bottom and
right to a square with grey, a Gaussian blur when shrinking and a linear resize, both as scipy
computes them (identical float32 pixel values; the blur runs only for the rows and columns the
resize reads, 0.3 s for a 1280 × 853 photo, 1.3 s for 4000 × 3000). Results are Python's: one
detection per patch at its best label, scored by the sigmoid; OWLv2's boxes are scaled to the
padded square, so they land on the image.

Python's documented results come out the same: the two cats at 0.665 and 0.613 with OWLv2 (Python
0.665 and 0.614, boxes within a pixel) and 0.716 and 0.706 with OWL-ViT (0.717 and 0.707). Against
transformers.js on the same pixel values, logits agree within 0.0066 (OWLv2) and 4.3e-4 (OWL-ViT).
End to end (median of five warm calls, WebGPU, backgrounded pane), OWLv2 takes 5.4 s against
32.7 s and OWL-ViT 639 ms against 1,674 ms. The comparison ran on transformers.js 3.8.1: 4.3.2
fails to create a session for every OWL export tried (`Xenova/` and `onnx-community/`, OWL-ViT and
OWLv2, WebGPU and WASM: no kernel for the Cast node `/class_head/Cast`). transformers.js also
differs from Python in its output: it squashes OWLv2's input to 960 × 960 instead of padding it
(the cats score 0.631 and 0.635), and it lists a box once per label above the threshold. Not yet:
image-guided detection (a query image instead of labels).

Depth estimation (`depth-estimation`) runs Depth Anything (`depth-anything/Depth-Anything-V2-Small-hf`,
the default, as in transformers.js; base and large load the same way). Its DINOv2 encoder is the
ViT code's layers with the layer scales folded into the projections before them, the position
embeddings fitted to each patch grid with torch's bicubic interpolation (as transformers does it),
and the CLS token placed last so the patch rows are a prefix. The neck and head run on the
convolution kernel (a ReLU on the input as it loads, for the pre-activation residual blocks), a
transposed convolution kernel and a bilinear resize kernel (torch's align_corners conventions,
checked against torch). The image is resized as DPTImageProcessor does it (aspect ratio kept,
sides rounded to multiples of 14). Output is transformers.js's: `predicted_depth` resized to the
image and `depth`, that map as 0–255 grey bytes. On the cats photo, fed the same pixel values as
transformers.js (`onnx-community/depth-anything-v2-small`, fp32) and post-processed by its own
code: depth within 1.7e-5 of its range, and 20 of 307,200 pixels a grey level apart. Both pipelines
end to end (median of five warm calls, WebGPU, backgrounded pane): 3.1 s in Kiln against 7.4 s for
transformers.js.

Image segmentation (`image-segmentation`) runs DETR's panoptic model
(`facebook/detr-resnet-50-panoptic`, the default, as in transformers.js; its `pytorch_model.bin`
read by Kiln's own zip and pickle reader). The mask head is recorded in the DETR plan, so the
planner keeps the backbone features it reads, but it runs only for the queries that pass the score
threshold, in batches of eight, instead of for all 100: each query's box attention map (softmax
over heads and pixels together), a first convolution split in two so its large shared half runs
once, convolutions with GroupNorm (new kernel) and ReLU, and nearest upsampling added to the
backbone's stride-16, 8 and 4 features (new kernel). Post-processing is transformers.js's panoptic
one: mask probabilities resized to the image and weighted by score, each pixel to the strongest
segment, segments that lose too much of their own mask dropped. Output: `{ score, label, mask }`,
the mask 255 inside, 0 outside. On the cats photo, fed the same pixel values as transformers.js
(`Xenova/detr-resnet-50-panoptic`, fp32) and post-processed by its own code: the same five segments
in the same order, scores within 1.4e-6, and one pixel of the 307,200 assigned differently. Both
pipelines end to end (median of five warm calls, WebGPU, backgrounded pane): 2.4 s in Kiln against
28.2 s for transformers.js, which runs the mask head for every query.

Background removal (`background-removal`) runs BRIA's RMBG-1.4 (`briaai/RMBG-1.4`, IS-Net; its
license allows non-commercial use only). transformers.js's default, MODNet (`Xenova/modnet`), is
published only as ONNX with folded, renamed weights, which Kiln does not load. The network is
written once as a description (briarmbg.py's nested U-blocks) that both lists the convolutions for
loading and builds the plan. Every convolution runs on the conv kernel, now with dilation, its
batch norm folded in. A channel concatenation becomes one convolution per input, summed in the
epilogue, so nothing is copied. Max pooling rounds up as ceil_mode does, and convolutions with at
most four output channels use a direct kernel instead of a mostly idle 64-channel tile. Output is
transformers.js's: the image as RGBA with the matte (sigmoid × 255, truncated) as its alpha, the
matte resized to the image with PIL's bilinear filter; `image-segmentation` on the same model
returns the matte as a mask with null label and score. On a 1024 × 1024 copy of the cats photo
(no resize on either side), fed the same pixel values as transformers.js's ONNX export of the
same model (fp32): 3 of the 1,048,576 matte bytes one level apart. transformers.js's own
background-removal pipeline rejects RMBG-1.4's config; run as its model card shows (AutoModel,
processor, RawImage), it took 4.0 s end to end against 2.9 s in Kiln (median of five warm
calls, WebGPU, backgrounded pane). Kiln's GPU time is the 3×3 convolutions at the GEMM's rate.

`image-segmentation` also runs SegFormer (`nvidia/segformer-*-finetuned-*`), semantic segmentation.
Each stage's overlapping patch convolution and the sequence-reduction convolution (keys and values
from a strided convolution, so the 16,384 first-stage queries attend to 256 keys) run on the conv
kernel; the Mix-FFN's depthwise 3×3 convolution has a kernel of its own. The decode head's fuse
over the concatenated stages is one 1×1 convolution per stage summed, with its batch norm folded
in. transformers.js resizes all 150 class score maps to the image size and then takes each pixel's
best class; Kiln does both in one kernel, so the full-size score maps are never stored. Output is
transformers.js's: one mask per class present, in class order, with null scores. On the cats photo,
fed the same pixel values as transformers.js (`Xenova/segformer-b0-finetuned-ade-512-512`, fp32)
and post-processed by its own code: the same 14 classes, and every one of the 307,200 pixels given
the same class. Both pipelines end to end with b0 (median of five warm calls, WebGPU, backgrounded
pane): 0.75 s in Kiln against 3.9 s for transformers.js.

Image-to-image (`image-to-image`) runs Swin2SR super-resolution (`caidas/swin2SR-classical-sr-x2-64`,
the default, as in transformers.js). Its width of 180 is padded to 192 in every row (zero weight rows
and columns), so the matmul and conv kernels take it unchanged and LayerNorm takes its statistics over
the real width. A new kernel runs Swin V2's shifted-window attention for 30-wide heads. It does the
cyclic shift, the window partition and the shifted-window mask as index arithmetic, so nothing is
rolled or copied. Within each window it computes cosine scores with each head's logit scale and the
position bias, which is precomputed at load from the continuous-position MLP (checked against torch).
A pixel-shuffle kernel and a leaky-ReLU epilogue on the conv kernel handle the upsampler. Pre- and
post-processing follow transformers.js: pixels rescaled to 0–1, padded by reflection only when a side
is not a multiple of 8 (Python pads every image), output clamped and rounded to bytes. On a 128 × 96
copy of the cats photo, fed the same pixel values as transformers.js
(`Xenova/swin2SR-classical-sr-x2-64`, fp32): 4 of the 147,456 output bytes one level apart. Both
pipelines end to end (median of five warm calls, WebGPU, backgrounded pane): 3.4 s in Kiln against
10.4 s for transformers.js. Profiling this model also led to a LayerNorm kernel for many short rows,
8 rows per workgroup, which every vision model now uses.

`image-classification` also runs ResNet (`microsoft/resnet-*`, bottleneck and basic blocks),
ConvNeXt (`facebook/convnext-*`) and DINOv2 (`facebook/dinov2-*-imagenet1k-1-layer`). ResNet's
convolutions run on the conv kernel with batch norms folded in, and the residual and ReLU in the
epilogue. ConvNeXt adds a 7 × 7 depthwise convolution (the depthwise kernel now takes any odd size)
and folds its layer scale into the second MLP matrix. DINOv2 runs on the ViT code: layer scales
folded, position embeddings fitted to the 224 crop by bicubic interpolation as Python transformers
does it, and the head over [CLS, mean of the patches]. Pooling and the heads run on the CPU. On
the tiger photo, fed the same pixel values as transformers.js (`Xenova/*`, fp32), probabilities of
every uniquely labelled class: ResNet-50 within 1.7e-8, ConvNeXt-tiny within 4.7e-7. DINOv2
differs by up to 6e-3 (tiger 0.7865 against 0.7805) because transformers.js's export resizes the
position embeddings bilinearly at a fixed scale, which Python transformers does not. With that
resize copied for the check, Kiln matches it within 3.2e-7, so the difference is in the resize
alone. Both pipelines end to end (median of five warm calls, WebGPU, backgrounded pane): ResNet-50
337 against 370 ms, ConvNeXt-tiny 374 against 434 ms, DINOv2-small 383 against 683 ms. Checking
them found a preprocessing bug, now fixed: an integer `size` on a ConvNeXt-style processor
(`microsoft/resnet-50`'s config) skipped the `crop_pct` resize and crop, and at 384 and above
ConvNeXt now resizes straight to a square as Python does.

Image captioning (`image-to-text`) runs VisionEncoderDecoder models with a ViT encoder and a GPT-2
decoder (`nlpconnect/vit-gpt2-image-captioning`, the default, as in transformers.js). The encoder is
the ViT code. The decoder is the seq2seq code, with the image's last hidden state standing in for an
encoder output: each layer's cross-attention keys and values are computed from it once, then
decoding runs one token per step with the greedy choice made on the GPU. GPT-2's Conv1D matrices
(stored input-major) are transposed when loading and its fused attention projections split.
Generation follows transformers.js: greedy from the decoder start token to end-of-sequence or a
`max_length` of 20, special tokens dropped. On three photos (cats, tiger, a football match), fed the
same pixel values as transformers.js (`Xenova/vit-gpt2-image-captioning`, fp32), the captions were
identical, including the model's own mistake of calling the tiger a zebra.

Florence-2 (`microsoft/Florence-2-base`, `-large` and their `-ft` versions) runs in `image-to-text`
with its task tokens: `task` picks one (default `<CAPTION>`) and `text_input` adds the text of the
tasks that take one. The answer comes back parsed as Florence2Processor's post_process_generation
returns it, under the task token, next to the raw text:

```js
const florence = await pipeline("image-to-text", "microsoft/Florence-2-base-ft");
await florence("car.jpg", { task: "<OD>" });
// [{ generated_text: "</s><s>car<loc_54><loc_334><loc_933><loc_774>wheel…</s>",
//    "<OD>": { labels: ["car", "wheel", "wheel"], bboxes: [[34.9, 160.6, 597.4, 371.8], …] } }]
await florence("car.jpg", { task: "<CAPTION_TO_PHRASE_GROUNDING>", text_input: "A green car" });
```

Tasks: `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`, `<OD>`,
`<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, `<OCR>`, `<OCR_WITH_REGION>` (quad boxes),
`<CAPTION_TO_PHRASE_GROUNDING>`, `<REFERRING_EXPRESSION_SEGMENTATION>` and
`<REGION_TO_SEGMENTATION>` (polygons, which transformers.js does not parse),
`<OPEN_VOCABULARY_DETECTION>`, and the `<REGION_TO_…>` tasks, whose input is a box as location
tokens. Boxes are in the image's pixels. The pieces: the DaViT tower on the GPU, its spatial blocks
on a window-attention kernel and its channel blocks on a channel-attention kernel (a per-group
channel Gram matrix, softmaxed, applied to the values), the 577 image tokens (their mean, then
each) ahead of the prompt's in the BART encoder, and greedy generation with the model's processors
(forced BOS and EOS, no repeated 3-grams; `num_beams` is ignored, as transformers.js does).
Microsoft's tokenizer.json lacks the 1,024 task and location tokens the processor adds at load;
Kiln adds them at the same ids.

Document question answering (`document-question-answering`) runs Donut
(`naver-clova-ix/donut-base-finetuned-docvqa`, the default; transformers.js's default is its ONNX
export): a question about a page image, answered without OCR.

```js
const qa = await pipeline("document-question-answering");
await qa("invoice.png", "What is the invoice number?"); // [{ answer: "us-001" }]
```

Donut is a VisionEncoderDecoder: a Swin encoder over the page at 2560 × 1920, and a 4-layer mBART
decoder. The encoder runs on new kernels for Swin's shifted-window attention (10 × 10 windows,
relative-position bias, the -100 mask between wrapped regions) and patch merging; its 307,200
first-stage tokens make it the heaviest image model Kiln runs (about 2.4 TFLOP, submitted a block
at a time so no command buffer holds seconds of work). The decoder is the seq2seq code given the
encoder's 4,800 tokens. The page goes through DonutImageProcessor's steps with PIL's resampling:
the shortest edge to 1920 (bilinear), a thumbnail that fits 2560 × 1920 (bicubic with PIL's
`reducing_gap`, which first box-reduces images shrunk 4× or more; Kiln has PIL's reduce too), and
black padding around the center. The prompt is transformers' (`<s_docvqa><s_question>…</s_question><s_answer>`),
decoding is greedy, and the answer is the text before `</s_answer>` (`null` when the model never
closes it). Kiln keeps the last two pages' encodings, so another question on a page costs only the
decoder: 1.3 s instead of 17 s.

On 8 questions over 5 pages from transformers.js's examples, Kiln, transformers.js on the same pixel
values and transformers.js's own pipeline gave the same 8 answers. A new page takes 16.9–18.3 s
against 33.1–34.3 s for transformers.js, and another question on it 1.3 s against 31–34 s (WebGPU,
backgrounded pane). The weights come as a 2022 `pytorch_model.bin` only; Kiln reads it with its own
zip and pickle reader.

Donut's other fine-tunes run in `image-to-text`, prompted with their task token as their model cards
do (Kiln finds it among the tokenizer's added tokens; `prompt` overrides it) and parsed with a port
of DonutProcessor's `token2json` into `json`:

```js
const parser = await pipeline("image-to-text", "naver-clova-ix/donut-base-finetuned-cord-v2");
const [{ json }] = await parser("receipt.png");
// { menu: { nm: "CINNAMON SUGAR", unitprice: "17,000", cnt: "1 x", price: "17,000" }, total: { total_price: "17,000", … } }
```

The receipt parser (`donut-base-finetuned-cord-v2`, pages at 1280 × 960) gave the same sequence as
transformers.js (`Xenova/donut-base-finetuned-cord-v2`, fp32, same pixels, `<unk>` banned as the
model card does) on the receipt and, 1,106 characters long, on the invoice, in 5.9 s against 12.8 s
and 13.3 s against 29.2 s. The document classifier (`donut-base-finetuned-rvlcdip`, which has no
transformers.js export) labelled the invoice "invoice", the advertisement "advertisement" and both
papers "scientific_report" (the receipt "budget": its 16 classes have no receipt), about 14 s a page.

SmolVLM (`HuggingFaceTB/SmolVLM-256M-Instruct`, `-500M-Instruct`, and other Idefics3 checkpoints)
answers questions about images in chat. `load(id)`, then messages with image parts, as for Gemma 3; no
extra file is needed. Images go through Idefics3's processor with PIL's Lanczos resampling (Kiln's
resizer is PIL's, byte for byte). The longest edge is set to 2048, both sides are taken up to
multiples of 512, and the image is cut into 512 × 512 tiles, plus the whole image squashed to one
tile as a global view: 13 tiles, 832 image tokens, for a 640 × 480 photo. `imageSplitting: false`
keeps only one 512 × 512 view (64 tokens). Each tile runs the SigLIP tower Kiln has for Gemma 3. The
connector's pixel shuffle (each 4 × 4 group of patches into one token of 16 × 768 values) costs
nothing: the tower treats its patches as a set, so they are fed grouped by output token, and the
post-norm rows already lie in the shuffled order (the position embeddings are permuted to match at
load). The tiles' tokens go into the prompt with their row and column tags, causal like the text.
Each tile keeps its features in a slot (`imageSlots` images' worth, 4 by default), so asking again
about the same image runs no vision and reuses the cached prompt: 1.8 s instead of 17.4.

To check it against transformers.js (the repo's ONNX export, fp32), both were given the same tiles and
prompt. Its own processor resizes bilinearly, in floats, and anchors the last tile to the image's
edge, so its pixels differ from Python's. The image features agreed within 2.1e-4, for values up to
96. The greedy replies were identical in all 17 runs: 9 photo-question pairs with splitting (13 to
17 tiles; descriptions up to 64 tokens, counts, an invoice total, a receipt) and 8 without. End to
end on the same GPU (hidden pane, warm), with splitting: 17.4 s against 25.0 s (invoice total),
16.9 s against 23.9 s (football players), 18.7 s against 34.9 s (a 64-token description). Without
splitting: 2.3 s against 4.3 s, 2.1 s against 3.4 s, and 4.0 s against 11.3 s. A tile is about
1.0 s of GPU work on the Intel Mac (f32 GEMMs at about 230 GFLOP/s, then attention), q8 weights
no faster. The tower now keeps one layer queued ahead instead of waiting on each layer, which
saved 0.3 s per tile.

Images combine with structured output: an image part and `format: { type: "json", schema }` read a
document into the shape you give, which transformers.js has no way to do (it has no constrained
decoding). With SmolVLM-500M, four requests on three documents, each with the same prompt naming the
keys, constrained and not: all 4 constrained replies were valid and matched their schemas, against 0
of 4 unconstrained (one not JSON at all; the others nested objects, arrays of arrays, or the CORD
receipt format the model saw in training). On the bicycle-shop invoice the five header fields
(number, dates, customer, total 154.06) were all right under the schema, while the free reply got the
invoice number wrong; its three line items and the other invoice's number, date and balance due
(2,844.80) were right too. The receipt shows the limit: under the schema the reply was valid but wrong
(a second item made of the quantity line, the change given as the total), so the schema fixes the
shape, not the reading. The first request on an image took 21–25 s (splitting into tiles, then the
vision tower), the same request again 2.8–4.0 s with the same answer.

transformers.js's export differs from the model. DaViT's channel attention scales queries by
N^-½, N the number of tokens; the ONNX graph froze that number when it was traced at 224 × 224
(1/56 at the first stage), while the processor feeds 768 × 768 images (1/192). Each of the 12
channel blocks runs 3.4× too sharp, and the image tokens move by up to 8.4. The captions and boxes
mostly survive it: 3 of the 20 outputs tried came out identical, the boxes of the others within a
few bins, and the captions worded differently ("Two cats laying on a pink blanket with remotes."
against Kiln's "Two cats are sleeping on a pink blanket."). Kiln computes the scale from the image,
as the PyTorch code does. To check everything else, the vision encoder's intermediate tensors were
exposed in a copy of the ONNX graph: the stem and the first spatial block agree within 2.6e-6,
and the first difference is the channel attention. Kiln run with the export's frozen scales gave
image tokens within 2.6e-4 of transformers.js's, and all 20 outputs identical, token for token. End
to end on the same GPU (medians of three warm runs, hidden pane): a caption in 2.9 s against
5.8 s, a detailed caption in 3.4 s against 7.4 s, detection in 3.0 s against 6.1 s.

Text-to-speech (`text-to-speech`, `text-to-audio`) runs VITS, the MMS-TTS voices
(`facebook/mms-tts-eng` by default, and the other languages' `facebook/mms-tts-*`).
transformers.js's own default (Supertonic) exists only as ONNX. The pieces:
- Text encoder, on the GPU: windowed relative-position attention gets a kernel of its own; its
  kernel-3 feed-forward runs on the conv kernel, now taking 1-D shapes.
- Stochastic duration predictor, on the CPU: its rational-quadratic spline flows run per character.
- Flow, on the GPU: the WaveNet coupling layers have their weight norm folded in at load, and the
  channel splits and flips are column copies.
- HiFi-GAN vocoder, on the GPU: a transposed-convolution kernel, and LeakyReLU folded into the
  convolutions' input reads.

VITS samples noise twice, so its output differs on every run, in transformers.js as in Python.
`seed` makes Kiln's repeatable, and `noise_scale: 0, noise_scale_duration: 0` takes the mean
path. For the check, the two random nodes of transformers.js's export
(`Xenova/mms-tts-eng`, fp32) were rewritten to output zeros. On "Hello, my dog is cute." both gave
33,024 samples (the same durations), within 1.4e-4 of each other, peak 0.87. With noise, Whisper
tiny.en heard Kiln's speech as "Hello my dog is cute.", "Hello my dot is cute." and "Hello my Doug is
cute." for three seeds, and the zero-noise speech as "Hello, my dog is cute." exactly; a noisy
transformers.js sample came out as "Tellomide.is Qt.". Both pipelines end to end (median of five
warm calls, backgrounded pane): 617 ms in Kiln on WebGPU against 1.33 s for transformers.js on
WebGPU and 4.4 s on WASM.

SpeechT5 (`microsoft/speecht5_tts`, with `microsoft/speecht5_hifigan` as the vocoder, or another
repo through the `vocoder` load option) takes a speaker's x-vector as `speaker_embeddings`: 512
float32 values, or the URL of a raw file of them, as in transformers.js's example:

```js
const tts = await pipeline("text-to-speech", "microsoft/speecht5_tts");
const speaker = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin";
const { audio, sampling_rate } = await tts("Hello, my dog is cute.", { speaker_embeddings: speaker });
```

The pieces: the character tokenizer (SentencePiece CHAR model behind its NFKC charsmap; the same
ids as transformers.js), the text encoder on the seq2seq code with a clamped relative-position
bias (the relative-attention kernel's second mode), the speech decoder one step per two frames
with its input vector written each step, the speech prenet and the stop and frame heads on the
CPU, then the postnet (batch norms folded into its convolutions, tanh on the conv kernel) and the
HiFi-GAN vocoder, the same builder VITS uses, on the GPU. Generation stops as transformers'
Python does: when a step's two stop probabilities sum to `threshold` (0.5), or at 20 steps per
two characters; transformers.js stops when either reaches it, which can end a step or so earlier.

transformers keeps the prenet's dropout on at inference (p = 0.5), so SpeechT5's output also
differs on every run; `seed` makes Kiln's repeatable. For the check, transformers.js's decoder
(`Xenova/speecht5_tts`, fp32) had its four random nodes rewritten to keep every unit, and Kiln ran
with `prenet_dropout: "keep"`. Without dropout the model never stops, so both ran to the limit
(240 steps for "Hello, my dog is cute."). The first 80 frames match within 1e-5. After that the
difference grows as each frame feeds the next, to a mean difference of 3.3e-4. With dropout,
Whisper tiny.en heard Kiln's speech as "Hello, my dog is cute." for five seeds out of five, and
"For quick brown fox jumps over the lazy dog." for the pangram; transformers.js's as "Hello, my
dog is cute." and "the quick brown fox jumps over the lazy dog." End to end with a hidden pane:
1.9 s in Kiln against 6.4 s for transformers.js (median of 3, its decoder and vocoder on WebGPU)
for "Hello, my dog is cute.", and 3.6 s against 12.7 s for the pangram.

Kokoro (`hexgrad/Kokoro-82M`, StyleTTS 2 with an iSTFTNet vocoder) loads from its original
`kokoro-v1_0.pth` checkpoint, with the voices of its ONNX export by name (`voice`, default
`af_heart`; "a…" American and "b…" British English, the other languages' voices with phonemes
given) or as a style table. Kokoro reads phonemes, not
text. Kiln does what kokoro-js does: it normalizes the text (numbers, money, times, titles),
phonemizes the parts between punctuation and fixes the phonemes up. The phonemizer is espeak-ng,
which is GPL, so Kiln does not bundle it: pass `phonemize` from the `phonemizer` package (espeak-ng
compiled to WebAssembly), or give phonemes with `phonemes: true`.

```js
import { phonemize } from "phonemizer";
const tts = await pipeline("text-to-speech", "hexgrad/Kokoro-82M");
const { audio, sampling_rate } = await tts("Hello, my dog is cute.", { voice: "af_heart", phonemize }); // 24 kHz
```

The pieces: PL-BERT (ALBERT, one layer applied 12 times), the duration encoder's three
bidirectional LSTMs with style-conditioned LayerNorm, the duration LSTM, and the text encoder on
the GPU as one plan; durations rounded on the CPU and the encodings repeated per frame; the shared
LSTM and the F0 and energy stacks (style-conditioned InstanceNorm residual blocks, a depthwise
transposed convolution for the 2× upsampling) as a second plan; then the iSTFTNet decoder and
generator (Snake activations, the harmonic source's spectrum mixed in at both upsampling stages) as
a third, sliced across submits. New kernels: a bidirectional LSTM (one workgroup per direction, the
state in workgroup memory), Snake, the depthwise transposed convolution, LeakyReLU and reflection
padding. The harmonic source (nine sines from the F0 curve, in float32 as torch computes it), its
20-point STFT and the inverse STFT run on the CPU. Python's Kokoro starts each harmonic at a random
phase and adds a little noise; so does Kiln (`seed` repeats it, `noise_scale: 0` leaves both out),
and the voice's style row is Python's (the phoneme count minus one; kokoro-js takes the next row).

The ONNX export transformers.js and kokoro-js run (`onnx-community/Kokoro-82M-v1.0-ONNX`, fp32)
has no noise, so Kiln without noise was compared with it, on the same phonemes, ids and style.
The durations, and so the lengths, were identical on three sentences. The vocoder's last
convolution agreed within 9.1e-4 (values up to 63) when given the export's own harmonic spectrum,
and the audio then correlated at 0.997 to 0.999. From the text, the audio correlates at 0.979 to
0.988: the sines' phase reaches millions of radians at the ninth harmonic, where float32 holds
only a few digits, so the two sources differ by a few 1e-4, and their spectra's near-empty bins
take different phases. The export also outputs speech 1.5× louder than PyTorch: its inverse STFT
divides by a window sum traced for one frame, so after the first 20 samples it skips the
normalization `torch.istft` does. Kiln follows PyTorch (the comparison scales transformers.js's
audio by 1 / 1.5). End to end on WebGPU (median of 3, same pane): 1.5 s against 2.5 s for
"Hello, my dog is cute.", 3.8 s against 6.3 s for an 82-character sentence.

transformers.js's lower-level API is there too, so code written for it moves over by changing the
import: `AutoTokenizer.from_pretrained(id)` (a callable `PreTrainedTokenizer` with `encode`, `decode`,
`batch_decode`, `tokenize`, the special tokens and their ids, padding and truncation, and
`apply_chat_template` with the model's own Jinja template), `RawImage` (`read`, `fromURL`,
`fromBlob`, `rgb`, `grayscale`, `resize` with PIL's filters, `crop`, `toTensor`, `toBlob`; pipelines
take one wherever they take an image), `read_audio` and `env` (`env.remoteHost` points Kiln at a
mirror; the other settings are accepted and do nothing). Tokenizers run on the page's thread, and
each pipeline's `tokenizer` is now the model's own. Against transformers.js 4.3.2's tokenizers in
Node, on 31 repos (BERT, DistilBERT, RoBERTa, MiniLM, BGE, XLM-R, T5, Flan-T5, NLLB, M2M100, Marian,
CLIP, Whisper, Qwen 2.5 and 3, SmolLM 2 and 3, SmolVLM, Llama 2 and 3.2, Gemma 3, Mistral v0.3,
Phi-3.5, Phi-4, Granite 3.3 and 4, LFM2, GPT-2) and 12 strings each (accents, CJK, emoji, code,
doubled and edge spaces, special-token text), the ids of `encode`, with and without special tokens,
were identical, and so were `decode`, batched calls and the rendered and tokenized chat templates
of the 11 repos whose template transformers.js loads. The differences left are transformers.js's: with no pad token in a
repo's tokenizer_config.json it pads with `undefined` and then fails to build the tensor (Kiln pads
with the tokenizer's own pad token, and errors as Python does when there is none); with a ByteLevel
post-processor (Qwen, Granite, GPT-2) it drops the second text of a pair; it cannot load SmolVLM's
template or repos without a tokenizer.json (Kiln builds BERT's from vocab.txt and RoBERTa / BART's
from vocab.json and merges.txt, which then match the ids of the same tokenizers' ONNX copies); and
`tokenize` shows the original text for a piece outside a Unigram vocabulary, where Kiln shows
`<unk>`. The comparison found four bugs in Kiln's tokenizers, now fixed: added tokens ignored
`lstrip` / `rstrip`, so Phi-3.5's `<|end|>` kept the newline after it and chat prompts differed
from transformers' by a token per turn; T5 and XLM-R did not split on whitespace before their
Metaspace step (doubled or trailing spaces made extra "▁" tokens); NLLB's lost a "▁" before an added
token; and Mistral v0.3's Metaspace step added a "▁" to text already starting with one.

One model, many tasks: `tasks(llm)` gives `classify` (your labels), `extract` (your JSON schema),
`ner` (your entity types), `answer` (extractive QA) and `summarize`. Classification reads the
model's probabilities: `llm.scoreChoices(messages, choices)` prefills the chat prompt once (reusing
the cached prefix) and runs each choice's tokens, then the end-of-turn token, as one verify pass
(the speculative-decoding plan, up to 32 rows), summing the log-softmax of each row's logits at the
next token. On the CPU backend its sums matched token-by-token forward passes within 1e-4, after a
reused prefix and after another prompt too. On small hand-labelled sets with Qwen2.5-0.5B-Instruct
(q8): 12 of 12 support tickets classified into three custom labels against 7 of 12 for the NLI
zero-shot pipeline (typeform/distilbert-base-uncased-mnli), 8 of 8 sentiments against 7 of 8 for the
SST-2 model, 6 of 6 extractive answers with null for both unanswerable questions (DistilBERT SQuAD
returns a span for those too), an email read into a five-field schema correctly, and 11 of 13 named
entities (persons, organizations, locations), all typed correctly and none invented, where BERT-NER
found 13. Two prompt shapes mattered for the 0.5B model. A schema with a `found` flag before the
answer made it give up on every question, so answerability is a separate scored yes / no. Asking
an entity's type as an instruction confused people with organizations, while a completion
(`… "Paris" is a`) typed all 11 correctly.

Retrieval in one library: `Retriever.create({ model })` loads a sentence-transformers embedder
(its pooling from 1_Pooling/config.json, max_seq_length from sentence_bert_config.json, query and
passage prompts from config_sentence_transformers.json, or BGE v1.5's and E5's known ones);
`add(docs)` splits each document at sentence boundaries into chunks of at most max_seq_length of
the embedder's own tokens, with whole sentences of overlap, and embeds them; `search(query)` is
exact cosine search over the normalized vectors; `ask(llm, question)` answers with a Kiln LLM over
the passages found, numbered in the system message in document order, so the same passages make
the same prefix; `export()` / `fromSnapshot` store and restore an index without embedding again.
On six short documents of made-up facts and eight questions, the scores agreed with transformers.js
embeddings of the same chunks within 4.5e-7 for all-MiniLM-L6-v2 (mean pooling) and 2.4e-7 for
bge-small-en-v1.5 (CLS pooling and its query prompt), with the same and the right top passage for
all eight questions with each. Qwen2.5-0.5B-Instruct (q8) answered from the passages ("Mount Ardel
is 3,412 metres tall."; "Ines Calloway made the first recorded ascent of Mount Ardel in 1891."), and
the second question, over the same passages, reused 201 of its 215 prompt tokens from the cache.
Asked something the passages do not cover (the capital of Australia, best passage scoring 0.15), the
0.5B model answered from its own knowledge despite the instructions; `minScore` keeps weak passages
out. A follow-up is searched on its own words ("And how high is it in feet?" still found the Ardel
passage). Loading the same embedding model twice in a page fails (its cached file is open in the
first worker), so retrievers share one through `{ embedder }`.

LoRA adapters (PEFT's adapter_config.json and adapter_model.safetensors) run beside the base
weights: `llm.loadAdapter(name, repo)`, then `llm.setAdapter(name)` or a call's `{ adapter: name }`.
Each projection's update B · (A · x), its scale folded into B, is added to the matmul's output
before anything fused after it (q/k/v before RoPE and the cache write, gate/up before the gated
activation, so those norms run on their own); a stacked matmul takes its parts' A rows stacked and
their B blocks on the diagonal. The updates live in slots sized for the largest rank loaded (a
multiple of 32), and each adapter is laid out for them on the GPU when it loads, so switching
between adapters is a GPU copy: 4 to 8 ms between two rank-16 adapters on SmolLM2-135M, against
245 ms for the first switch, which allocates the slots. On the CPU backend, a random two-layer
model with runtime LoRA matched the same weights with the updates merged in within 1e-4 (fused and
unfused, prefill and decode, after swaps), and cleared slots gave the base model back. Against a
NumPy forward pass of SmolLM2-135M-Instruct with each adapter merged in (float64), greedy decoding
on WebGPU (f32) gave the same tokens for two prompts with the base model, a seven-projection
adapter (scientific answers: "Photosynthesis is the process by which plants convert light energy
into chemical energy.") and a q/v-only one (rewrites for LinkedIn), down to top-two logit margins of
0.007. Switching drops cached prefixes, whose K/V came from other weights.

The model side follows: `AutoModel`, `AutoModelForSequenceClassification`,
`AutoModelForTokenClassification`, `AutoModelForQuestionAnswering` and `AutoModelForMaskedLM` for
BERT, DistilBERT and RoBERTa; `AutoModel` and `AutoModelForImageClassification` for ViT, DeiT,
DINOv2, ResNet and ConvNeXt; `CLIPModel`, `CLIPTextModelWithProjection` and
`CLIPVisionModelWithProjection`; and `AutoModelForCausalLM`, whose `generate()` takes the
tokenizer's ids (with `max_new_tokens`, sampling options, a streamer, and the defaults of the
repo's generation_config.json as transformers.js applies them) and returns the sequences as int64
ids. A model is called with the tokenizer's or processor's output and returns transformers.js's
tensors (`last_hidden_state`, `logits`, `start_logits` / `end_logits`, `text_embeds`,
`logits_per_image` …), shaped [batch, tokens, …]; each row runs without its padding, so padded
positions hold zeros where transformers.js holds the masked rows. `AutoProcessor` (and
`AutoImageProcessor`, `AutoFeatureExtractor`) reads preprocessor_config.json: images to
`pixel_values` with Kiln's PIL-exact resize, Whisper audio to `input_features`, wav2vec2 audio to
`input_values`. The models run in Kiln's worker; the processors and tokenizers on the page.

Against transformers.js 4.3.2 (fp32, WASM) on the same inputs: last_hidden_state of
all-MiniLM-L6-v2 within 3.3e-6, DistilBERT SST-2 logits within 7.2e-7, BERT NER logits within
4.8e-6, DistilBERT SQuAD start and end logits within 1.6e-5, BERT's masked-LM logits (30,522 per
token, values up to 24) within 9.1e-5. SmolLM2-135M-Instruct's greedy `generate()` from the chat
template's ids gave the same 45 ids, its stop token included (0.33 s against 0.85 s), and the
streamer received the reply's text. Whisper's `input_features` agreed within 1.2e-7 and wav2vec2's
`input_values` exactly, and so did `read_audio`, which decodes as transformers.js does (the
browser's AudioContext at the target rate, stereo mixed as √2 · (left + right) / 2); Kiln's own
pipelines keep their WAV reader. Images differ before the model: transformers.js resizes in the
browser, Kiln as PIL does, so the cats photo's pixel values differed by up to 0.82 (ViT's top three
classes were still the same). Given the same pixels, ViT's logits agreed within 1.4e-5 (0.34 s
against 1.5 s) and CLIP's `logits_per_image` within 1.3e-5, its text embeddings within 1.6e-7. The
masked-LM head runs every position through one batched matmul (0.15 s against 0.11 s for nine tokens;
0.24 s when it ran one position at a time).

T5's learned relative position bias is added to each score by the per-row attention kernels (GPU
and CPU, buckets computed in the kernel; bucket function equal to transformers' over ±300
positions, kernel checks within 6.1e-7). SentencePiece tokenizers: Unigram (Viterbi) and character
BPE behind the Precompiled normalizer (its double-array trie read as SentencePiece does), from
tokenizer.json or a .model / .spm file (protobuf). Against Python `sentencepiece` on 11 strings
(accents, CJK, emoji, fullwidth and ligature characters, odd spacing): T5 and Marian ids identical;
NLLB's text ids identical, its language-code template built by Kiln. Repos without safetensors load
from `pytorch_model.bin`: zip and legacy checkpoints are read as byte ranges from their pickle (a
small pickle reader for state dicts; nothing is executed), and match the repos' safetensors copies
exactly. Range reads without the OPFS cache (Node, or a full browser store) retry network and
server errors with backoff, which a 2.4 GB NLLB download in Node needed twice.

Gemma 3 was checked token by token against an independent NumPy forward pass written from
transformers' `modeling_gemma3`: greedy output matched on every prompt tried, for 1B in q4 and q8 and 270M
in q8 and f32. On a 1,138-token prompt, where the fact asked about lies outside the sliding layers' 512-token
window, 1B in q4 gave the same answer tokens as the reference (then a newline before ending the turn,
where the reference ended at once), and 270M in q8 matched the reference exactly. The 270M model does not survive q4: its first answer token wins by 1.4 logits, q4 flips it,
and the reply turns to nonsense, so load it with `quant: "q8"` (same tokens as f32 at a quarter of
the memory).

`mistral` loads through the same code as `llama` but has not been run (the smallest Mistral is 7B).
Gemma 3 4B and larger load as text models: the multimodal checkpoint's language model is read
(`text_config`, weights under `language_model.` or `model.language_model.`, the vision tower skipped),
with linear rope scaling on the global layers only, from safetensors or GGUF. 4B from
`gemma-3-4b-it-Q4_0.gguf` answers "The capital of France is Paris." and, on the 1,138-token prompt,
"BLUE HERON", which needs the global layers since the fact lies outside the 1,024-token sliding
window. There is no token-level reference for 4B (the NumPy pass would need the 8.6 GB checkpoint);
its greedy tokens on both prompts are the same as 1B's. The 2.4 GB file is cached whole (three
segments) and the next load reads it from the cache in about 10 s on the test Mac (Iris Plus 650), 6 s
of it reading; converting on the CPU instead takes about 24 s.

Images work for Gemma 3 4B and larger, from the transformers checkpoint, whose vision tower and
projector Kiln reads from the same safetensors (`load("unsloth/gemma-3-4b-it")`; `vision: false`
skips them), or with llama.cpp's mmproj file:
`load("unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_0.gguf", { mmproj: "mmproj-F16.gguf" })`. Then
messages take `{ type: "image", image }` parts (a Blob, ImageBitmap or ImageData, resized to 896 × 896
by the browser). The vision tower is SigLIP: 4096 patches through 27 layers with full attention
(the text model's attention kernels, given a bidirectional span over every patch), then Gemma 3's
projector pools them 4 × 4 into 256 tokens. Its matmul weights load as q8; the MLP's width (4304) is
padded with zeros to 4320 for the kernels. In the prompt each image becomes Gemma 3's `<start_of_image>`,
256 soft tokens and `<end_of_image>`, with token ids past the vocabulary that the embedding reads from
the image features; the 256 tokens attend to each other both ways, as Gemma 3 was trained, so they are
prefilled together in one chunk. Kiln's image features, for a synthetic picture (a red disc and a blue
square on white), matched an independent NumPy forward pass of the f32 weights to a cosine similarity
of 0.9998 (at least 0.999 for every token), and 4B Q4_0 described it as "a red circle and a blue square
arranged vertically on a white background". Through the API, a 448 × 448 PNG of a green triangle on
yellow: "The image shows a green triangle on a yellow background." On the Intel Mac the vision tower
takes about 46 s per image (90 s before span attention, below); the follow-up "And what color is the background?" reused the encoded image
and 290 cached prompt tokens and answered "The background is yellow." in 6 s. An image's features stay
on the GPU (`imageSlots`, 4 by default), keyed by its pixels, so a conversation that keeps the image
does not run the tower again.

From the safetensors, on the same 896 × 896 pixels as transformers.js's export (its vision encoder
in fp32), Kiln's 256 image features agree at a mean cosine similarity of 0.9995 (lowest token
0.988, largest difference 1.75 for values up to 33), the tower in q8 as it loads by default. Asked
about three photos with the text model at Kiln's default quantization, against the export's q4
decoder, both answered sensibly in different words ("Two tabby cats are sprawled out on a bright
pink couch, each with a remote control nearby…" and "Two tabby kittens are sprawled out on a pink
blanket, each with a remote control resting beside them…"), in 36–42 s against 49–61 s for a
40-token answer with the image. The export's q4f16 decoder printed only `<pad>` on this GPU
(float16 overflow, a known Gemma 3 problem). Loading the 8.6 GB checkpoint took 15 minutes here,
almost all of it the download.

Granite 4's dense models (`granitemoehybrid` with every layer "attention" and no experts: 350M and
1B) are Llama with four multipliers, folded in at load: the residual multiplier into o_proj and
down_proj, 1 / logits_scaling into the tied head (and the embedding scale raised to match), plus the
embedding and attention multipliers; the shared MLP's `input_linear` is already the fused gate/up
weight. They were checked against an independent NumPy forward pass written from transformers'
`modeling_granitemoehybrid`, applying the multipliers at run time: 350M in f32 and q8 and 1B in q8
gave the reference's greedy tokens on every prompt tried, including a `<tool_call>` for
`get_weather` with `{"city": "Paris"}`, and conversion on the GPU and the CPU gave the same tokens.
350M does not survive q4 on that prompt (it opens a `<tool_call>` and then writes a refusal), so load
it with `quant: "q8"`; 1B in q4 still makes the call.

Granite 3 (`granite`, separate gate and up projections) uses the same mapping. GGUF `granite` files
load too, Granite 3 and the dense Granite 4 models alike (llama.cpp writes both under that
architecture): the multipliers come from the metadata, a per-layer `head_count_kv` list is accepted
when its entries agree, and Q/K are read in llama.cpp's interleaved order. granite-4.0-350m-Q8_0.gguf
gave the reference's greedy tokens on both prompts above. Its tokenizer (pre-tokenizer `dbrx`, which
needs Llama 3's split pattern) and that of granite-3.3-2b-instruct-Q4_K_M.gguf (`refact`) give the same ids as
`tokenizer.json` on every test string. Granite 3.3 2B Instruct answered "What's the weather like in
Paris right now?" with `<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris", "unit":
"celsius"}}]`, the same tokens from GGUF Q4_K_M and from safetensors in q8, and `parseToolCalls` read the call.
The q8 reply matched the NumPy reference (extended to Granite 3's separate gate and up projections)
token for token, all 28 and the end of turn.

Other architectures are rejected at load: SmolLM3, hybrid (Mamba) and mixture-of-experts Granite, Phi-3, Gemma 1/2, mixture-of-experts
models, Llama 4, and rope scaling other than Llama 3's and linear (Gemma 3).

GGUF `gemma3` files load the way llama.cpp writes them: norms already hold Gemma's 1 + w, and the
sliding layers' rope base (10k) and pattern (every 6th layer global) are not in the metadata. From
`gemma-3-270m-it-Q8_0.gguf` the tokenizer gives the same ids as `tokenizer.json` on every test string,
and greedy output matched the reference on the short prompts and the 1,138-token one.

Supported:

- Models and formats: Llama-family decoders; safetensors F32/F16/BF16; GGUF F32/F16/BF16/Q4_0/Q4_1/
  Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K. Weights are stored on the GPU in blocks of 32 as q4 (4-bit codes,
  about an eighth of f32), q8 (8-bit, about a quarter, close to f32 in accuracy) or f32. Either block
  format can be affine, with a min per block beside the scale, which is how GGUF Q4_1/Q4_K (in q4) and
  Q5_1/Q5_K (in q8) are held: every value exactly as llama.cpp dequantizes it. `quant: "auto"` (the
  default) stores float weights as q4 and keeps each GGUF tensor at its file's precision: Q4_0, Q4_1
  and Q4_K as q4, Q5_0, Q5_1, Q5_K and Q8_0 as q8, all without rounding, and Q6_K as q8 (one rounding,
  to 8 bits). `"q4"` rounds the 5-, 6- and 8-bit GGUF types to 4 bits as before, `"q8"` rounds Q6_K and
  float weights to 8 bits, and `"f32"` dequantizes everything.
- Text: byte-level BPE (GPT-2, Llama 3, Qwen, SmolLM) and SentencePiece-style BPE with byte
  fallback (Llama 2, TinyLlama, Mistral, Phi-3), from `tokenizer.json` or GGUF (`gpt2` and `llama`
  vocabularies; the latter merge by piece score). On TinyLlama, Mistral v0.3 and Phi-3.5 the
  SentencePiece path gives the same ids as the `sentencepiece` library on every test string (accents,
  CJK, emoji through byte tokens, runs of spaces, tabs, code), and TinyLlama-1.1B-Chat chats from a
  Q4_K_M GGUF with its own template (7.6 tok/s on the Intel GPU). Chats are formatted by the model's own Jinja `chat_template`
  (from `chat_template.jinja`, `tokenizer_config.json` or GGUF metadata), run by a Jinja2 subset with
  transformers' settings. It renders the templates of SmolLM2/3, Qwen2.5/3, Phi-3.5, TinyLlama,
  Mistral v0.3, DeepSeek-R1-Distill, Granite 3.3, Llama 3.2 and Gemma 2 byte-for-byte like jinja2
  (the tests compare against Python output). Models without a template fall back to built-in
  chatml / Llama 3 formats, and so do templates using Jinja the interpreter lacks (`{% filter %}`,
  `{% include %}`, …), with a warning.
- Tool calling: `chat(messages, { tools })` passes OpenAI-style function specs to the template the way
  transformers does (`tools` and `documents` are None when absent), and messages can carry
  `tool_calls` and the `tool` role. The test fixture's tool and document cases for SmolLM3,
  Qwen2.5/3, Mistral v0.3 and Granite 3.3 render byte-for-byte like jinja2. `parseToolCalls` reads
  the `<tool_call>{…}</tool_call>` format (Qwen and SmolLM3 templates; SmolLM3 itself does not load
  yet), Granite's `<|tool_call|>[…]` and Mistral's `[TOOL_CALLS]` (a JSON list, or `name[ARGS]{…}` in
  newer tokenizers); the decoder keeps those marker tokens in the text. Mistral calls without an id get
  the 9-character one its templates require; on Qwen2.5-0.5B a call, its result
  and the final answer made a full round trip. Templates that never read `tools` (SmolLM2) and the
  built-in formats refuse tools rather than drop them. `toolChoice: "required"` (or one function by
  name) constrains the reply to exactly one `<tool_call>` whose name is a listed tool and whose
  arguments follow its `parameters` schema: where Qwen2.5-0.5B answered "What's the weather in
  Paris?" with a question, it then called `get_weather` with `{"city": "Paris"}` (24 tokens, greedy
  or sampled).
- Generation: temperature, top-k, top-p and HF-style repetition penalty, all on the GPU; prefix KV
  reuse; f16 or f32 KV cache; speculative decoding by prompt lookup. With drafts, 10 of 10 test
  prompts (greedy and sampled) gave the same text as plain decoding. The verify pass uses the 8-row
  kernels, which round differently from single-token decode, so a near-tied token can still differ.
- Structured output: any JSON, JSON Schema (`type`, `properties`, `required`, `additionalProperties`,
  `items`, `minItems`/`maxItems`, `minLength`/`maxLength`, `pattern`, `format` (date, time,
  date-time, email, uuid, ipv4; others stay annotations), `minimum`/`maximum` and their exclusive
  forms, `multipleOf`, `enum`, `const`, `anyOf`/`oneOf`, local `$ref`, including recursive ones), a regular
  expression (`{ type: "regex", pattern }`), a GBNF grammar in llama.cpp's format (`{ type: "grammar",
  grammar }`), or one of a list of strings. Properties come in declared order. Keywords that would
  change what matches but are not enforced (`not`, `if`, `uniqueItems`, …) are rejected rather than
  ignored. Bounded numbers carry their text so far, and a digit is refused as soon as no value within
  the bounds can still follow it (`{"minimum": 18, "maximum": 120}` refuses "13" but keeps "1", which
  may become 18 or 1xx), and under `multipleOf` as soon as no multiple can follow (`0.35` under 0.1;
  decimal steps are compared with a 1e-9 relative tolerance); they are written without exponents. Regex and GBNF compile to two new
  grammar nodes, a Unicode character class read byte by byte as UTF-8 (refused at the lead byte when
  no code point it starts is in the class) and a bounded repetition, on top of literals, sequences and
  alternatives; a repetition round that matches nothing is cut off where the matcher would otherwise
  loop. On Qwen2.5-0.5B: `\(\d{3}\) \d{3}-\d{4}` gave "(123) 555-5555"; a GBNF list of three
  yes/no answers gave "no, no, no" (the format holds, the 0.5B model's answers do not); a schema
  with a `date` and bounded integers gave `{"name": "Ada", "born": "1815-12-10", "age": 25, "score":
  1}`; `^[A-Z]{3}-[0-9]{4}$` as a `pattern` gave "ABC-1234". Masks took 3 to 25 ms per call. Whitespace between tokens is compact by default (one space, no line
  breaks): on SmolLM2-135M it cut a schema answer from 57 to 47 tokens and 1958 to 1521 ms, and
  stopped the model repeating itself. `whitespace: "pretty"` allows indentation.

Known limits, roughly in roadmap order:

- Subgroup reductions exist for the general matmul only (`subgroups: true` at load: a shuffle
  butterfly across each row's lanes and `subgroupAdd` for the fused norm) and are off by default.
  On an M3 they pass every kernel check and give the same text, but decode with them was slower in
  every run (best 22.1 against 26.5 tok/s on Gemma 3 4B, both orders) and prefill about 5% slower,
  since turning them on replaces the decode GEMV. The Intel development GPU has no `subgroups`.
- Kernels were tuned on one integrated GPU; the autotuner only chooses lanes per row, not tile sizes.
- Matmul needs K to be a multiple of 32; attention needs head dim ≤ 256 and a multiple of 4.
- Prefill attention does not split keys across workgroups, so a short prompt (few tiles) far into a long
  context runs on few workgroups.
- One sequence decodes at a time: overlapping calls wait their turn (each from its own saved prefix)
  rather than being batched into one forward pass, and the cache is contiguous, not paged.
- Constrained decoding waits for the CPU's mask between forward passes (drafts let one pass check
  several tokens). Regexes have no backreferences, lookaround or word boundaries; GBNF rules may not
  be left-recursive; a string with `pattern` cannot also have
  `minLength`/`maxLength`.
  `toolChoice` forces one call; under "auto" calls are not checked against the tool schemas.
- Q6_K (scales every 16 values) is rounded to q8 on load rather than run as stored; the other GGUF
  block types are held exactly.
- `toolChoice` only constrains the `<tool_call>` format.
- Single storage binding per tensor: very large f32 embeddings can exceed `maxStorageBufferBindingSize`
  (use q4 or q8).
- Images: Gemma 3 (from its transformers checkpoint or llama.cpp's mmproj GGUF) and SmolVLM; for
  Gemma 3 one 896 × 896 crop per image (no pan-and-scan). Span attention
  covers the vision tower only; the text model's image chunks (f16 cache, causal text rows, sliding
  layers) use the tiled kernel. Measured with the profiler, that attention is 16.7 ms per layer for
  a 275-token image prompt on the Intel Mac, about 0.6 s of Gemma 3 4B's 14 s to the first token
  (the matmuls are the rest), so a span kernel there would save at most a few percent.
- Without WebGPU, models run on the CPU: matmuls in WASM SIMD at about 8 GMAC/s per thread on the
  test Mac, q4/q8 with x as int16 (several threads only on cross-origin isolated pages), attention
  in WASM SIMD, the rest
  in TypeScript, and every buffer in one
  wasm32 memory (4 GiB at most). Small models only. WebNN and Node/Dawn backends come later.

## License

MIT, see [LICENSE](../LICENSE).
