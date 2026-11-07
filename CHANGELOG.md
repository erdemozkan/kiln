# Changelog

## 0.1.0 (unreleased)

First release of `kiln-webgpu`: LLM inference in the browser on WebGPU, loading Hugging Face
safetensors and GGUF files directly.

**Models.** Llama (including Llama 3 rope scaling), Mistral, Qwen2, Qwen3, SmolLM, Granite 3 and 4
(350M to 2B; for Granite 4, the attention-only models) and Gemma 3 (270M to 4B and larger, text
only), from safetensors (F32/F16/BF16) or GGUF
(F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K). Weights are held as q4, q8 or f32; the
default `quant: "auto"` keeps each GGUF tensor at its own precision, every type but Q6_K exactly
(affine q4/q8 planes for Q4_1, Q4_K, Q5_1 and Q5_K).

**API.** `load(id)` runs the model in a module worker; `chat` and `complete` return async iterators
of text, with `stats` (timings, token counts, `stopReason`). The model's own Jinja chat template,
tools and documents, parsed tool calls (`<tool_call>`, Mistral's `[TOOL_CALLS]`, Granite's
`<|tool_call|>`) and `toolChoice`. Structured output: JSON, JSON Schema or a
fixed choice, enforced by a grammar mask at every token. Draft tokens (speculative decoding by
prompt lookup) on by default, also under a grammar.

**Loading.** Weights stream into GPU buffers without holding the model in CPU memory. The download is
the cache: files land in OPFS as 1 GiB segments, interrupted downloads resume, dropped connections
retry, and `listCache` / `clearCache` manage it. `load` asks the browser to keep the cache
(`navigator.storage.persist()`); `info.persistent` gives the answer. Conversion to q4 and q8 runs on the GPU for F32/F16/BF16 safetensors and every GGUF block type
Kiln reads, with the CPU converter's exact bits. The
last prompt's K/V survives page reloads, and earlier prefixes stay on the GPU (`prefixCacheBytes`,
256 MiB by default), so conversations used in turn each prefill only their new turn; overlapping
calls queue.

**CPU fallback.** Without WebGPU (or with `device: "cpu"`), the same engine runs on TypeScript
versions of every kernel; slow, for small models. `info.backend` reports which backend runs.

**Context.** The default context is the model's trained context, up to 8192 tokens and 512 MiB of
KV cache; sliding-window layers keep a ring of window + 128 slots, so Gemma 3 4B gets 8192 tokens in
290 MB. Prompts that do not fit say how to make room.

**Speed** (Gemma 3 4B Q4_0). Intel Iris Plus 650: about 5.5 tok/s decode, a 1,138-token prompt in
66 s, a cached load in about 10 s. Apple M3: 24.6 tok/s decode, about 179 tok/s prefill, a cached
load in 2.4 s. Decode uses a block-per-lane GEMV with several rows per thread on large matrices,
prefill a tiled GEMM and tiled attention, and GPU work is cut into short submits so the page stays
responsive.

**Checks.** 138 unit tests; a kernel check page (785 checks on the Intel GPU) compares every kernel with a CPU reference; Gemma 3 1B matches a NumPy reference token for
token in q8 and q4.
