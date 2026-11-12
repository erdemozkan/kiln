# Kiln.js (`kiln-webgpu`)

Run LLMs in the browser on WebGPU, straight from Hugging Face safetensors or GGUF files: no ONNX
export, no compile step, no server. Without WebGPU it runs on the CPU (WASM SIMD).

```bash
npm install kiln-webgpu
```

Or with no install and no bundler, from a CDN:

```html
<script type="module">
  import { pipeline } from "https://cdn.jsdelivr.net/npm/kiln-webgpu@0.1.0";
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

- `"text-generation"` runs the decoder families Kiln implements (Llama, Mistral, Qwen2/3, SmolLM,
  Granite, Gemma 3); the other tasks, below, have their own model lists.
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

const translator = await pipeline("translation", "Helsinki-NLP/opus-mt-en-de");
await translator("The house is wonderful."); // [{ translation_text: "Das Haus ist wunderbar." }]

const nllb = await pipeline("translation", "facebook/nllb-200-distilled-600M", { dtype: "q8" });
await nllb("The house is wonderful.", { src_lang: "eng_Latn", tgt_lang: "fra_Latn" });

const t2t = await pipeline("text2text-generation", "google/flan-t5-small");
```

BART / DistilBART, Marian (OPUS-MT), M2M100 / NLLB and T5 / Flan-T5 / mT5 load from their original
repos, with transformers.js's output keys. Generation is greedy, as in transformers.js, with the
model's `min_length`, `no_repeat_ngram_size` and forced tokens; options override them. Repos that
only ship `pytorch_model.bin` load too (the checkpoint's pickle is parsed for tensor offsets, never
executed).

## Speech recognition

```js
const transcriber = await pipeline("automatic-speech-recognition"); // openai/whisper-tiny.en
const { text } = await transcriber("https://example.com/speech.wav");

const multilingual = await pipeline("automatic-speech-recognition", "openai/whisper-base");
await multilingual(blob, { language: "french", task: "translate" }); // French speech → English text
```

Whisper models load from their original repos. Audio can be 16 kHz `Float32Array` samples, a URL, a
Blob / File, or encoded bytes: WAV is parsed directly, other formats are decoded by the browser's
Web Audio API on the page's main thread (`readAudio` is exported for doing it yourself).
Timestamps and long audio work as in transformers.js:

```js
const out = await transcriber(audio, { return_timestamps: true, chunk_length_s: 30 });
// { text, chunks: [{ timestamp: [0, 8], text: " And so my fellow Americans ..." }, ...] }
```

Audio longer than 30 s is chunked in 30 s windows (5 s overlap each side) even without
`chunk_length_s`. Word-level timestamps work too, computed as transformers.js computes them (from
the cross-attention of the model's alignment heads), with no special export needed:

```js
const out = await transcriber(audio, { return_timestamps: "word" });
// { text, chunks: [{ text: " And", timestamp: [0.76, 1.06] }, { text: " so", timestamp: [1.06, 1.44] }, ...] }
```

wav2vec2 and HuBERT CTC models work in the same pipeline (Kiln reads the model's config to tell them
from Whisper), and their sequence classifiers run `audio-classification`:

```js
const ctc = await pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h");
await ctc(samples); // { text: "AND SO MY FELLOW AMERICANS ..." }

const keywords = await pipeline("audio-classification"); // superb/wav2vec2-base-superb-ks
await keywords(samples, { top_k: 3 }); // [{ label: "yes", score: 0.98 }, ...]
```

### Images

ViT and DeiT checkpoints run `image-classification` and `image-feature-extraction`. Images are
resized, cropped and normalized exactly as Python transformers does it (PIL's resampling, reimplemented
byte for byte), not by a canvas resize. Images can be URLs, Blobs / Files, bytes, `ImageBitmap` /
`ImageData`, `<img>` / `<canvas>` elements, or RGB(A) pixels:

```js
const classifier = await pipeline("image-classification"); // google/vit-base-patch16-224
await classifier("tiger.jpg"); // [{ label: "tiger, Panthera tigris", score: 0.886 }, ...]

const extractor = await pipeline("image-feature-extraction", "google/vit-base-patch16-224-in21k");
const features = await extractor(image); // Tensor [1, 197, 768]

// CLIP: any labels, no training.
const zeroShot = await pipeline("zero-shot-image-classification"); // openai/clip-vit-base-patch32
await zeroShot(image, ["tiger", "cat", "dog"]); // [{ score: 0.996, label: "tiger" }, ...]

// DETR (or YOLOS, "hustvl/yolos-tiny"): boxes in pixels of the original image.
const detector = await pipeline("object-detection"); // facebook/detr-resnet-50
await detector("cats.jpg", { threshold: 0.9 }); // [{ score: 0.998, label: "remote", box: { xmin: 40, ymin: 70, xmax: 175, ymax: 117 } }, ...]

// OWL-ViT / OWLv2: boxes for any labels.
const owl = await pipeline("zero-shot-object-detection"); // google/owlvit-base-patch32
await owl("cats.jpg", ["a photo of a cat", "a photo of a dog"]); // [{ score: 0.716, label: "a photo of a cat", box: { xmin: 1, ymin: 55, xmax: 315, ymax: 472 } }, ...]

// Depth Anything V2: relative depth at the image's size, and as grey bytes.
const depth = await pipeline("depth-estimation"); // depth-anything/Depth-Anything-V2-Small-hf
const { predicted_depth, depth: grey } = await depth("cats.jpg"); // Tensor [480, 640]; { width, height, channels: 1, data }

// DETR panoptic: one mask per segment, 255 inside.
const segmenter = await pipeline("image-segmentation"); // facebook/detr-resnet-50-panoptic
await segmenter("cats.jpg"); // [{ score: 0.994, label: "cat", mask: { width: 640, height: 480, channels: 1, data } }, ...]

// SegFormer: semantic segmentation, one mask per class present.
const semantic = await pipeline("image-segmentation", "nvidia/segformer-b0-finetuned-ade-512-512");
await semantic("cats.jpg"); // [{ score: null, label: "wall", mask }, { score: null, label: "floor", mask }, ...]

// ResNet, ConvNeXt and DINOv2 classifiers as well as ViT / DeiT.
const resnet = await pipeline("image-classification", "microsoft/resnet-50");
await resnet("tiger.jpg"); // [{ label: "tiger, Panthera tigris", score: 0.940 }, ...]

// ViT + GPT-2: a caption.
const captioner = await pipeline("image-to-text"); // nlpconnect/vit-gpt2-image-captioning
await captioner("cats.jpg"); // [{ generated_text: "a cat laying on a blanket next to a cat laying on a bed" }]

// Florence-2: captions, detection, grounding, OCR and segmentation by task token.
const florence = await pipeline("image-to-text", "microsoft/Florence-2-base-ft");
await florence("car.jpg", { task: "<OD>" }); // [{ generated_text, "<OD>": { labels: ["car", "wheel", "wheel"], bboxes: [...] } }]

// Donut: questions about a document image, no OCR.
const qa = await pipeline("document-question-answering"); // naver-clova-ix/donut-base-finetuned-docvqa
await qa("invoice.png", "What is the invoice number?"); // [{ answer: "us-001" }]

// Donut fine-tunes in image-to-text: receipts to JSON, or a document's class.
const parser = await pipeline("image-to-text", "naver-clova-ix/donut-base-finetuned-cord-v2");
const [{ json }] = await parser("receipt.png"); // { menu: { nm: "CINNAMON SUGAR", … }, total: { total_price: "17,000", … } }

// MMS-TTS (VITS): speech from text, 16 kHz mono.
const tts = await pipeline("text-to-speech"); // facebook/mms-tts-eng
const { audio, sampling_rate } = await tts("Hello, my dog is cute.", { seed: 1 });

// SpeechT5: a voice from a speaker x-vector (512 float32 values or their URL).
const t5 = await pipeline("text-to-speech", "microsoft/speecht5_tts");
const speech = await t5("Hello, my dog is cute.", { speaker_embeddings: "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin" });

// Kokoro: English voices at 24 kHz; text through your phonemizer (espeak-ng's `phonemizer` package).
import { phonemize } from "phonemizer";
const kokoro = await pipeline("text-to-speech", "hexgrad/Kokoro-82M");
const voice = await kokoro("Hello, my dog is cute.", { voice: "af_heart", phonemize });

// Swin2SR: 2× super-resolution.
const upscaler = await pipeline("image-to-image"); // caidas/swin2SR-classical-sr-x2-64
const big = await upscaler("small.png"); // { width: 2 × w, height: 2 × h, channels: 3, data }

// RMBG-1.4 (non-commercial license): the image as RGBA, the background transparent.
const remover = await pipeline("background-removal"); // briaai/RMBG-1.4
const cutout = await remover("portrait.jpg"); // { width, height, channels: 4, data }
```

With a CLIP model, `image-feature-extraction` returns the image embedding (the visual projection's
output). All labels go through the text tower in one batch, alongside the image, and their
embeddings are cached, so asking about the same labels again only runs the image tower.

## One model, many tasks

A loaded LLM covers the tasks that otherwise each need their own model, with your own labels and
schemas:

```js
import { load, tasks } from "kiln-webgpu";

const t = tasks(await load("Qwen/Qwen2.5-0.5B-Instruct"));
await t.classify("My card was charged twice", ["billing", "bug report", "feature request"]);
// { label: "billing", labels: [...], scores: [...] }: the model's probabilities over the labels
await t.extract(email, { type: "object", properties: { name: { type: "string" }, day: { enum: ["Mon", "Tue"] } }, required: ["name", "day"] });
await t.ner("Angela Merkel met Emmanuel Macron in Paris.", ["person", "location"]);
// [{ entity_group: "person", word: "Angela Merkel", start: 0, end: 13 }, …]
await t.answer("When did it open?", context);  // { answer: "1907", start, end } or { answer: null }
await t.summarize(text, { sentences: 2 });
```

`classify` scores each label as the model's whole reply (its tokens and the end-of-turn token), so
the scores are probabilities, not a sampled guess; `multi_label: true` asks yes / no per label, and
`examples` adds labelled turns first. `extract` goes through Kiln's grammar, so its JSON always
parses and matches the schema. `ner` lists mentions (all types at once, then each type), keeps
those found verbatim in the text, and types each by scoring the types (or "other", which drops it);
`fast: true` keeps only the first listing. `answer` first scores whether the context answers the
question at all (`score`; below `threshold`, 0.5, the answer is null), then copies a span from it.
Every task takes generation options (`adapter`, `maxTokens`, …) and `instructions` to replace its
prompt. `llm.scoreChoices(messages, choices)` is the scoring underneath.

With Qwen2.5-0.5B-Instruct (q8), on small hand-labelled sets: 12 of 12 support tickets put in the
right one of three custom labels (the NLI zero-shot pipeline: 7 of 12), 8 of 8 sentiments (the
SST-2 model: 7 of 8), 6 of 6 extractive answers including null for both unanswerable questions
(DistilBERT SQuAD always returns a span), and 11 of 13 person / organization / location mentions,
all typed right and none invented (BERT-NER found all 13: for the standard types a dedicated model
is still better; the LLM's are your own types).

## Retrieval (RAG)

Embeddings, an index and generation in one library:

```js
import { Retriever, load } from "kiln-webgpu";

const rag = await Retriever.create();            // sentence-transformers/all-MiniLM-L6-v2, or { model: "BAAI/bge-small-en-v1.5" }
await rag.add(["Some text…", { id: "faq", text: "…", metadata: { url } }]);
const hits = await rag.search("question", { k: 4 }); // [{ id, docId, text, score, metadata }]

const llm = await load("Qwen/Qwen2.5-0.5B-Instruct");
const { stream, sources } = await rag.ask(llm, "question", { k: 3 });
for await (const text of stream) render(text);    // cites passages as [1], [2] … = sources[0], sources[1] …

const snapshot = rag.export();                      // plain data: store it, then Retriever.fromSnapshot(snapshot)
```

Documents are split into chunks at sentence boundaries, up to the model's max_seq_length in its own
tokens, with a sentence or two of overlap (`chunkTokens`, `overlapTokens`). The pooling (mean or
CLS) and query / passage prompts come from the repo's sentence-transformers files (BGE v1.5's and
E5's prompts are known when the repo does not list them). The index is exact cosine search over
normalized vectors on the page. `ask` puts the passages in the system message, in document order,
so a follow-up over the same passages reuses the cached prompt prefix; `rag.prompt(question, hits)`
gives the messages to build your own call. Retrievers can share one embedding model:
`Retriever.create({ embedder: rag.embedder })` (also for `fromSnapshot`).

## LoRA adapters

Load a base model once, then any number of PEFT LoRA adapters trained on it, and switch between
them per call:

```js
const llm = await load("HuggingFaceTB/SmolLM2-135M-Instruct");
await llm.loadAdapter("science", "Miladsaeedi70/smollm2-135m-scientific-sft-lora");
await llm.loadAdapter("linkedin", "ArtemMozharov/smollm2-135m-linkedin-translator");

await llm.setAdapter("science");                       // from now on
llm.chat(messages, { adapter: "linkedin" });           // or for one call
await llm.setAdapter(null);                            // the base model again
```

An adapter repo needs adapter_config.json and adapter_model.safetensors; LoRA on q/k/v/o and
gate/up/down (rank and alpha per module, rsLoRA) is supported, DoRA, embedding adapters and
`modules_to_save` are not. Each adapter is laid out on the GPU when it loads, so switching between
loaded adapters is a GPU copy (a few milliseconds); the first switch, and going back to the base
model, rebuild the forward plans. The update runs beside the base weights (two small f32 matmuls per
projection), whatever the base weights are stored as (checked with f32 base weights). A switch drops the cached prompt prefixes, whose
K/V came from other weights, and the prompt cache is not saved while an adapter is in use.

## transformers.js's tokenizer, RawImage, read_audio and env

transformers.js's lower-level helpers, under the same names:

```js
import { AutoTokenizer, RawImage, read_audio, env } from "kiln-webgpu";

const tokenizer = await AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct");
const { input_ids, attention_mask } = tokenizer(["Hello", "Hi there"], { padding: true }); // int64 Tensors
tokenizer.decode(input_ids.tolist()[0], { skip_special_tokens: true });
const prompt = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });

const image = await RawImage.read("https://example.com/cat.jpg"); // or fromBlob, fromURL
const small = await (await image.resize(224, null)).grayscale(); // PIL's resampling
const audio = await read_audio(url, 16000); // Float32Array

env.remoteHost = "https://my-mirror.example.com/"; // a Hub mirror for everything Kiln loads
```

The tokenizer reads tokenizer.json (or BERT's vocab.txt, RoBERTa / BART's vocab.json and
merges.txt) and runs on the calling thread; its ids match transformers.js's on the 31 repos
compared. Every pipeline's `tokenizer` is the model's own (an empty object for GGUF files), and
pipelines take a `RawImage` wherever they take an image. Of `env`, only `remoteHost` does anything.

## transformers.js's AutoModel and AutoProcessor

```js
import { AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoModelForImageClassification, RawImage } from "kiln-webgpu";

const tokenizer = await AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2");
const model = await AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2");
const { last_hidden_state } = await model(tokenizer(["Hello", "World"], { padding: true })); // [2, tokens, 384]

const lm = await AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct");
const t = await AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct");
const inputs = t.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true });
const ids = await lm.generate({ ...inputs, max_new_tokens: 64 }); // int64 [1, prompt + new]
t.batch_decode(ids, { skip_special_tokens: true });

const processor = await AutoProcessor.from_pretrained("google/vit-base-patch16-224");
const vit = await AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224");
const { logits } = await vit(await processor(await RawImage.read(url)));
```

Encoders (BERT, DistilBERT, RoBERTa): `AutoModel`, `AutoModelForSequenceClassification`,
`AutoModelForTokenClassification`, `AutoModelForQuestionAnswering`, `AutoModelForMaskedLM`. Images:
`AutoModel` and `AutoModelForImageClassification` (ViT, DeiT, DINOv2, ResNet, ConvNeXt), `CLIPModel`,
`CLIPTextModelWithProjection`, `CLIPVisionModelWithProjection`. Language models:
`AutoModelForCausalLM` with `generate()` (calling the model directly for logits is not supported).
`AutoProcessor` makes `pixel_values` (resized as PIL does, so values can differ from
transformers.js's browser resize), Whisper's `input_features` and wav2vec2's `input_values`.
`model.dispose()` unloads a model.

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
interrupted download resumes. Float weights loaded as q4 or q8 are cached as converted as well (about
a third to a half of the download on top), so later loads skip reading and converting the original:
roughly half the load time. `load(id, { weightCache: false })` keeps only the download.
`listCache()` reports what is cached and `clearCache()` deletes it.
`load` asks the browser to keep the cache (`navigator.storage.persist()`) so it survives low disk
space; `llm.info.persistent` says whether it agreed, and `load(id, { persist: false })` skips asking.

Gemma 3 (4B and larger) also takes images: from the transformers checkpoint, whose vision tower
Kiln reads from the same files, or from a GGUF with llama.cpp's mmproj file:

```ts
const llm = await load("unsloth/gemma-3-4b-it"); // or load("unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_0.gguf", { mmproj: "mmproj-F16.gguf" })
const reply = llm.chat([{ role: "user", content: [{ type: "image", image: blob }, { type: "text", text: "What is in this picture?" }] }]);
```

An image may be a Blob, File, ImageBitmap or ImageData; it is resized to 896 × 896. Running the
vision tower takes a while on integrated GPUs (about 46 s on an Intel Iris Plus 650); later turns that
keep the image reuse its features and their cached tokens.

SmolVLM (`HuggingFaceTB/SmolVLM-256M-Instruct`, `-500M-Instruct` and other Idefics3 checkpoints)
takes images the same way, with no extra file:

```ts
const vlm = await load("HuggingFaceTB/SmolVLM-256M-Instruct");
const reply = vlm.chat([{ role: "user", content: [{ type: "image", image: blob }, { type: "text", text: "Can you describe this image?" }] }]);
```

As its processor does, the image is cut into 512 × 512 tiles plus a global view (13 tiles, 832
image tokens, for a 640 × 480 photo); `load(id, { imageSplitting: false })` uses one 512 × 512 view
instead (64 tokens, several times faster, less detail).

Image parts work with structured output, to read a document into a shape:

```ts
const schema = { type: "object", properties: { invoice_number: { type: "string" }, total: { type: "number" } }, required: ["invoice_number", "total"], additionalProperties: false };
const reply = vlm.chat([{ role: "user", content: [{ type: "image", image: blob }, { type: "text", text: "Read this invoice. Reply with JSON: invoice_number and total." }] }], { format: { type: "json", schema } });
```

The reply always parses and fits the schema; whether the values are right is still up to the model.

## Models

Llama, Mistral, Qwen2, Qwen3, SmolLM, Granite 3 and 4 (350M to 2B), Gemma 3 (270M to 4B and larger;
images as above) and SmolVLM (images as above) decoders, from
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
