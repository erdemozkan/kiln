# Kiln

WebGPU-native LLM inference for the browser. Loads Hugging Face safetensors or GGUF directly —
no ONNX export, no ahead-of-time compile step.

```ts
import { load } from "@kiln/api";

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
| `@kiln/api` | Main-thread client, worker RPC, streaming `TextStream` |
| `@kiln/runtime` | `Engine`: chunked prefill, pipelined decode, weight streaming |
| `@kiln/models` | Code-defined architectures (Llama / Mistral / Qwen2 / Qwen3 / SmolLM / Granite / Gemma 3), HF + GGUF config mapping |
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
- **A CPU fallback.** Every kernel the models use also exists in TypeScript, attached to its WGSL spec
  with the same bindings and uniform layout, and a `CpuBackend` runs them on the model's worker
  thread. `load` uses it when the browser has no WebGPU or no adapter (`device: "auto"`, the
  default), or when asked (`device: "cpu"`); `info.backend` says which ran. The whole engine works
  unchanged on it: grammars, saved prefixes, the prompt cache. The kernel check page runs every check
  against it too (`/kernels.html?backend=cpu`, 757 of 757 passing; the f16 matmul variants need
  shader-f16), and Node tests cover it in CI. Matmuls run in WASM SIMD (`packages/kernels/src/simd.wat`,
  1.4 KB compiled): every CPU buffer lives in one WebAssembly memory, so the kernels read weights,
  activations and scratch in place, taking f32x4 dot products straight from the 4- and 8-bit codes.
  That runs a 1536 × 576 q4 or q8 matvec at 3.2–3.6 GMAC/s on the Intel Mac, against 0.24–0.42 in
  TypeScript (which remains the path without WASM SIMD). Prefill builds plans of the exact prompt
  length instead of padded buckets, and draft tokens are off by default (a checked row costs as much as
  a decoded one). SmolLM2-135M gives the GPU's tokens in q4 and q8, decoding at 18 tok/s on the Intel
  Mac (it was 1–2 with TypeScript matmuls), with a 42-token prompt in 0.4 s.
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

The kernel-check page runs every WGSL kernel against `@kiln/kernels` reference code on the current GPU.
The bench page reports GPU time per step and per kernel (matmuls are labelled with their shape) plus
end-to-end decode speed. The dev server reloads the page whenever a source file changes, which aborts a
load in progress.

## Performance

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

Verified on this GPU: all kernel checks pass (802 now that Chrome exposes shader-f16 on it, including q4 and q8 weights, symmetric and affine, in every matmul variant, bidirectional-span attention, LayerNorm and pooling, the decode GEMV and the prefill GEMM, both prefill attention kernels, fused variants, f16 cache paths,
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
| `gemma3` | Gemma 3 4B-it (recalls a fact 1,100 and 3,000 tokens back; images with `mmproj`) | GGUF Q4_0 (2.4 GB) + mmproj-F16 | 177–180 ms GPU per step (about 5.5 tok/s); 24.6 tok/s on an M3 |

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

Images work for Gemma 3 4B and larger with llama.cpp's mmproj file:
`load("unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_0.gguf", { mmproj: "mmproj-F16.gguf" })`, then
messages with `{ type: "image", image }` parts (a Blob, ImageBitmap or ImageData, resized to 896 × 896
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
takes about 78 s per image, nearly all of it attention over the 4096 patches (2.05 s per layer with
32-row tiles, against 2.76 s with the text prefill's 14-row default; the matmuls take 0.85 s); the follow-up "And what color is the background?" reused the encoded image
and 290 cached prompt tokens and answered "The background is yellow." in 6 s. An image's features stay
on the GPU (`imageSlots`, 4 by default), keyed by its pixels, so a conversation that keeps the image
does not run the tower again.

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
- Unigram tokenizers (T5, XLM-R, some SentencePiece models) are not supported. `toolChoice` only
  constrains the `<tool_call>` format.
- Single storage binding per tensor: very large f32 embeddings can exceed `maxStorageBufferBindingSize`
  (use q4 or q8).
- Images: Gemma 3 only, from llama.cpp's mmproj GGUF (the vision weights inside safetensors
  checkpoints are still skipped), one 896 × 896 crop per image (no pan-and-scan). The vision tower
  reuses the text model's tiled attention, which reaches about a fifth of the matmuls' throughput on
  its 4096 × 4096 bidirectional span; a register-tiled kernel for long spans would roughly halve it.
- Without WebGPU, models run on the CPU: single-threaded (no worker pool), matmuls in WASM SIMD at
  about 3.5 GMAC/s on the test Mac, attention and the rest in TypeScript, and every buffer in one
  wasm32 memory (4 GiB at most). Small models only. WebNN and Node/Dawn backends come later.

## License

MIT, see [LICENSE](LICENSE).
