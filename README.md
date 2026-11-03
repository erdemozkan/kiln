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
const json = llm.chat(messages, { format: { type: "json", schema } }); // or { type: "json" }, { type: "choice", choices: ["yes", "no"] }
```

## Design

| Package | Role |
|---|---|
| `@kiln/api` | Main-thread client, worker RPC, streaming `TextStream` |
| `@kiln/runtime` | `Engine`: chunked prefill, pipelined decode, weight streaming |
| `@kiln/models` | Code-defined architectures (Llama / Mistral / Qwen2 / Qwen3 / SmolLM), HF + GGUF config mapping |
| `@kiln/tokenizers` | Byte-level and SentencePiece-style BPE from `tokenizer.json` or GGUF metadata, streaming UTF-8 decode, a Jinja interpreter for chat templates |
| `@kiln/grammar` | JSON / JSON Schema / choice grammars as a byte-level pushdown matcher, token masks over a vocabulary trie |
| `@kiln/kernels` | WGSL kernels, op wrappers, CPU reference implementations |
| `@kiln/core` | Backend HAL interface, tensors, uniform packing, `Ctx` plan recorder |
| `@kiln/backend-webgpu` | WebGPU backend: explicit bind group layouts, pipeline cache, staging pool, error reporting |
| `@kiln/loader` | Byte sources (HTTP range, OPFS cache), safetensors/GGUF parsers, dtype conversion, q4 quantization |

Key ideas:

- **Compile once, replay many.** `LlamaModel.buildPlan(T)` records every dispatch for a T-token forward
  pass. Positions, the sampled row, seed and sampling settings live in a small step buffer, so one plan
  serves every decode step.
- **The decode loop runs on the GPU.** Sampling runs on the GPU, and a one-thread `advance` kernel feeds
  the sampled id back as the next input, appends it to a history buffer, marks it for the repetition
  penalty and moves the position forward. So `stepsPerSubmit` decode steps (default 4) go out in one
  submit and come back in one 16-byte readback, with two batches in flight.
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
  same grammar positions. The sampler skips masked ids before top-k / top-p, so truncation applies among
  allowed tokens only. The forward pass needs only the last sampled id, so it is submitted before the
  mask is computed and the CPU work overlaps it.
- **The prompt prefix is reused.** The engine remembers which tokens' K/V are in the cache and only
  prefills what changed, so a multi-turn chat prefills just the new turn.
- **1× memory while loading.** Weights stream row-chunk by row-chunk from the network into GPU buffers,
  quantized to q4 on the way if requested. Nothing holds the whole model in CPU memory.
- **The download is the cache.** In a worker, `CachedHttpSource` writes the download into OPFS and serves
  reads from it as bytes land. Interrupted downloads resume from the last flushed byte recorded in the
  metadata (not the file size, which can include writes lost when a worker is killed), and a cached
  file whose header no longer parses is evicted and downloaded again.

## Develop

```bash
npm install
npm test          # unit tests (Node)
npm run typecheck
npm run dev       # playground at http://localhost:5173, kernel check at /kernels.html, profiler at /bench.html
```

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
- **Autotuner:** on this GPU it picks 16 lanes per row for the 576×576 projections and 8 elsewhere; gains
  here are within noise. It exists for GPUs with more cores, where small N leaves most of them idle.
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

Where the remaining time goes at position 256 on the Intel GPU: the 49152×576 LM head, gate/up and the projections run
at about half the GPU's copy bandwidth (13.8 GB/s), and each of the 214 dispatches has about 11 µs of
fixed cost.

## Status

Verified on this GPU: all kernel checks pass (184, including both prefill attention kernels, fused variants, f16 cache paths, grammar masks and
sampling distributions). SmolLM2-135M-Instruct chats from Hub safetensors (q4, f32) and from a GGUF
Q4_K_M file.

Supported:

- Models and formats: Llama-family decoders; safetensors F32/F16/BF16; GGUF F32/F16/BF16/Q4_0/Q4_1/
  Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K. K-quants and Q5 types are dequantized on load and stored as q4, or as
  f32 with `quant: "f32"`.
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
- Generation: temperature, top-k, top-p and HF-style repetition penalty, all on the GPU; prefix KV
  reuse; f16 or f32 KV cache.
- Structured output: any JSON, JSON Schema (`type`, `properties`, `required`, `additionalProperties`,
  `items`, `minItems`/`maxItems`, `minLength`/`maxLength`, `enum`, `const`, `anyOf`/`oneOf`, local
  `$ref`, including recursive ones), or one of a list of strings. Properties come in declared order.
  Keywords that would change what matches but are not enforced (`pattern`, `not`, `if`, …) are
  rejected rather than ignored. Whitespace between tokens is compact by default (one space, no line
  breaks): on SmolLM2-135M it cut a schema answer from 57 to 47 tokens and 1958 to 1521 ms, and
  stopped the model repeating itself. `whitespace: "pretty"` allows indentation.

Known limits, roughly in roadmap order:

- Subgroup reductions exist for matmuls only (`subgroups: true` at load: a shuffle butterfly across
  each row's lanes and `subgroupAdd` for the fused norm) and are off by default: the Intel
  development GPU has no `subgroups`, so they are not yet verified or measured. The kernel-check
  page and the bench test them on GPUs that have the feature.
- Kernels were tuned on one integrated GPU; the autotuner only chooses lanes per row, not tile sizes.
- Matmul needs K to be a multiple of 32; attention needs head dim ≤ 256 and a multiple of 4.
- Prefill attention does not split keys across workgroups, so a short prompt (few tiles) far into a long
  context runs on few workgroups.
- The KV cache is contiguous: one sequence, no paging, and the reused prefix lives only in memory.
- Constrained decoding runs one step per submit (each mask depends on the token before it). No regex
  or GBNF grammars, and `minimum`/`maximum`/`format` are not enforced.
- K-quants are re-quantized to q4 on load (a second rounding) instead of being run natively.
- Unigram tokenizers (T5, XLM-R, some SentencePiece models) are not supported. Chat templates get `messages`, `bos_token`,
  `eos_token`, `add_generation_prompt` and `enable_thinking`, but no `tools` or `documents` yet.
- Single storage binding per tensor: very large f32 embeddings can exceed `maxStorageBufferBindingSize`
  (use q4).
- WebGPU only; WASM fallback, WebNN and Node/Dawn backends come later.
