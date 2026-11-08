# Kiln.js (`kiln-webgpu`)

Run LLMs in the browser on WebGPU, straight from Hugging Face safetensors or GGUF files: no ONNX
export, no compile step, no server. Without WebGPU it runs on the CPU (WASM SIMD).

```bash
npm install kiln-webgpu
```

Or with no install and no bundler, from a CDN:

```html
<script type="module">
  import { pipeline } from "https://cdn.jsdelivr.net/npm/kiln-webgpu";
</script>
```

## transformers.js-style pipeline

```js
import { pipeline, TextStreamer } from "kiln-webgpu";

const generator = await pipeline("text-generation", "HuggingFaceTB/SmolLM2-135M-Instruct", { dtype: "q4", device: "webgpu" });
const streamer = new TextStreamer(generator.tokenizer, { skip_prompt: true, callback_function: (text) => render(text) });
const output = await generator([{ role: "user", content: "Hi!" }], { max_new_tokens: 128, streamer });
output[0].generated_text.at(-1).content; // the reply
```

Code written for transformers.js's text-generation pipeline should mostly need only the new import.
A string input is completed (`generated_text` is the prompt plus the completion, or the completion
alone with `return_full_text: false`); a list of messages is a chat (`generated_text` is the messages
with the reply appended); an array of either runs them one after another. Options take
transformers.js's names: `max_new_tokens`, `do_sample` (default false: greedy; when true,
`temperature` 1, `top_k` 50 and `top_p` 1 unless given), `repetition_penalty`, `streamer`,
`progress_callback` (`initiate` / `progress` / `done` per file, then `ready`). The differences:

- Text tasks only: `"text-generation"` for the decoder families Kiln runs (Llama, Mistral, Qwen2/3,
  SmolLM, Granite, Gemma 3), and the encoder tasks below for BERT, DistilBERT and RoBERTa models.
- `dtype` picks the format Kiln quantizes to at load: 4-bit names (`q4`, `q4f16`, `bnb4`, `int4`)
  load as q4, 8-bit ones (`q8`, `int8`, `uint8`) as q8, `fp16` / `fp32` as f32.
- `device` defaults to `"auto"` (WebGPU where there is one, else the CPU); `"wasm"` means the CPU.
- The model's `generation_config.json` is not read, and without `max_new_tokens` a reply runs until the
  model ends its turn (or the context fills).
- Streamers receive text, not token ids (`TextStreamer`, or any object with `put(text)` and `end()`).
- Kiln's own options can be passed alongside: `format` (JSON schema, regex, grammar), `tools`,
  `signal`, and on load `maxSeq`, `cpuThreads`, `mmproj`. `generator.model` is the underlying
  `KilnModel`, `generator.info` its details, `generator.dispose()` unloads it.

## Encoder tasks

BERT, DistilBERT and RoBERTa models (and those built on them: MiniLM, BGE, GTE…) load from their
original safetensors repos, with transformers.js's task names and output shapes:

```js
import { pipeline, cos_sim } from "kiln-webgpu";

const extractor = await pipeline("feature-extraction", "sentence-transformers/all-MiniLM-L6-v2");
const emb = await extractor(["A man is eating.", "Someone is having a meal."], { pooling: "mean", normalize: true });
const [a, b] = emb.tolist();
cos_sim(a, b);

const classifier = await pipeline("sentiment-analysis"); // [{ label: "POSITIVE", score: 0.9998 }]
const ner = await pipeline("ner");                       // [{ entity: "B-PER", score, index, word: "Wolfgang" }, …]
const qa = await pipeline("question-answering");          // { answer: "a nice puppet", score }
const unmasker = await pipeline("fill-mask");             // [{ score, token, token_str: "life", sequence }, …]
const zeroShot = await pipeline("zero-shot-classification"); // { sequence, labels, scores }
```

Without a model id, each task loads the repo in `DEFAULT_MODELS`. Options follow transformers.js:
`pooling` ("none", "mean", "cls") and `normalize`; `top_k`; `ignore_labels` and
`aggregation_strategy` ("none", "simple"); `hypothesis_template` and `multi_label`. Encoders load
in f32 by default (`dtype: "q8"` or `"q4"` to quantize their matrices). Inputs longer than the model
takes are truncated. Each input runs on its own (no padding), so feature extraction without pooling
returns zeros past a shorter input's last token.

## Summarization, translation, text2text

```js
const summarizer = await pipeline("summarization"); // sshleifer/distilbart-cnn-6-6
const [{ summary_text }] = await summarizer(article, { max_new_tokens: 100 });
```

BART and DistilBART models run today (`summarization`, `translation`, `text2text-generation`, with
transformers.js's output keys). Generation is greedy, as in transformers.js, with the model's
`min_length`, `no_repeat_ngram_size` and forced tokens; options override them. Repos that only ship
`pytorch_model.bin` load too (the checkpoint's pickle is parsed for tensor offsets, never executed).

## Kiln's own API

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
`minimum`/`maximum`/`multipleOf`; grammars are llama.cpp's GBNF.

Each stream's `stats` resolve when it ends, with token counts, timings and `stopReason`: `"stop"`
(end of turn), `"maxTokens"`, `"context"` (the reply filled the context), `"format"` (the structured
value is complete) or `"abort"`.

To see where time goes, `await llm.startProfile()`, make calls, then `await llm.stopProfile()`: time
per kernel and shape, most expensive first (GPU timestamp queries where the device offers
`timestamp-query`, wall clock on the CPU fallback). Each GPU dispatch then runs in a pass of its own,
which costs a little, so profile only when looking.

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
vision tower takes a while on integrated GPUs (about 46 s on an Intel Iris Plus 650); later turns that
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
Without it, `load` falls back to running on the CPU (matmuls in WASM SIMD; much slower than a GPU, fine for models around 135M;
on a cross-origin isolated page, with `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`, large matmuls also use several threads, `cpuThreads`) and
`llm.info.backend` is `"cpu"` (`llm.info.cpuThreads` says how many threads it got);
`load(id, { device: "gpu" })` fails instead, and `device: "cpu"` uses the CPU even where WebGPU works.

The package is one file: the model worker's code is inside it and starts from a blob URL, so it works
from a CDN, with any bundler and in Vite's dev server, with no worker file to serve. A page with a
Content-Security-Policy needs `worker-src blob:`; to host the worker yourself, pass
`load(id, { worker })`. The in-process engine, for code that already runs in a worker, is
`import { createEngine } from "kiln-webgpu/engine"`.

## License

MIT
