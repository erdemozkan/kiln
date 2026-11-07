# kiln-webgpu

Run LLMs in the browser on WebGPU, straight from Hugging Face safetensors or GGUF files: no ONNX
export, no compile step, no server.

```bash
npm install kiln-webgpu
```

```ts
import { load } from "kiln-webgpu";

const llm = await load("HuggingFaceTB/SmolLM2-135M-Instruct", { quant: "q4", onProgress: console.log });
for await (const text of llm.chat([{ role: "user", content: "Hi!" }], { temperature: 0.7 })) render(text);
```

The model runs in a module worker; the main thread only sees an async iterator. The context defaults
to the model's own, up to 8192 tokens and 512 MiB of KV cache; `load(id, { maxSeq })` sets it, and
`llm.info.contextLength` says how long the model was trained for. A model id is
`org/repo` for safetensors or `org/repo/file.gguf` for one GGUF file.

Several conversations can share one model: each call reuses the cached K/V of whichever earlier
prompt shares the most tokens with it (kept on the GPU up to `load(id, { prefixCacheBytes })`, 256 MiB
by default), and calls that overlap run one after another.

Structured output is a generate option:

```ts
const schema = { type: "object", properties: { city: { type: "string" }, population: { type: "integer" } }, required: ["city", "population"] };
const json = llm.chat(messages, { format: { type: "json", schema } }); // or { type: "json" }, { type: "choice", choices: ["yes", "no"] }
const phone = llm.chat(messages, { format: { type: "regex", pattern: "\\(\\d{3}\\) \\d{3}-\\d{4}" } });
const list = llm.chat(messages, { format: { type: "grammar", grammar: 'root ::= item (", " item)*\nitem ::= "yes" | "no"' } });
```

Schemas may use `pattern`, `format` (date, time, date-time, email, uuid, ipv4) and
`minimum`/`maximum`; grammars are llama.cpp's GBNF.

Each stream's `stats` resolve when it ends, with token counts, timings and `stopReason`: `"stop"`
(end of turn), `"maxTokens"`, `"context"` (the reply filled the context), `"format"` (the structured
value is complete) or `"abort"`.

Weights are cached in the browser (OPFS) as they download, so later loads skip the network, and an
interrupted download resumes. `listCache()` reports what is cached and `clearCache()` deletes it.
`load` asks the browser to keep the cache (`navigator.storage.persist()`) so it survives low disk
space; `llm.info.persistent` says whether it agreed, and `load(id, { persist: false })` skips asking.

Gemma 3 (4B and larger) also takes images, with llama.cpp's mmproj file from the same repo:

```ts
const llm = await load("unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_0.gguf", { mmproj: "mmproj-F16.gguf" });
const reply = llm.chat([{ role: "user", content: [{ type: "image", image: blob }, { type: "text", text: "What is in this picture?" }] }]);
```

An image may be a Blob, File, ImageBitmap or ImageData; it is resized to 896 × 896. Running the
vision tower takes a while on integrated GPUs (about 90 s on an Intel Iris Plus 650); later turns that
keep the image reuse its features and their cached tokens.

## Models

Llama, Mistral, Qwen2, Qwen3, SmolLM, Granite 3 and 4 (350M to 2B) and Gemma 3 (270M to 4B and larger;
images as above) decoders, from
safetensors (F32/F16/BF16) or GGUF (F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K).
Weights are held on the GPU as q4, q8 or f32. The default, `quant: "auto"`, stores float weights as q4
and keeps a GGUF file's tensors at their own precision (exactly, except Q6_K, which becomes q8);
`"q4"`, `"q8"` or `"f32"` force one format. Gemma 3 270M and
Granite 4 350M need `"q8"` or `"f32"`.

## Requirements

A browser with WebGPU (Chrome or Edge 113+, Safari 26+, Firefox 141+ on Windows) for real speed.
Without it, `load` falls back to running on the CPU (much slower; fine for models around 135M) and
`llm.info.backend` is `"cpu"`; `load(id, { device: "gpu" })` fails instead. Use it from a bundler
that understands `new URL("…", import.meta.url)` workers (Vite, webpack 5, Parcel, esbuild with
a plugin). To host the worker yourself, pass `load(id, { worker })`.

## License

MIT
