import { readImage, type ImageInput } from "@kiln/loader";
import { createAudio, createVision, TtsEngine, type VisionEngine, createEncoder, createEngine, createSeq2Seq, type AudioEngine, loadPromptCache, promptCacheName, savePromptCache, type EncoderEngine, type Engine, type Seq2SeqEngine, type Seq2SeqGenerateOptions, type TextDelta } from "@kiln/runtime";
import type { AudioTask, VisionTask, EncoderTask, RawOutputs, Request, Response, Seq2SeqTask } from "./protocol";

/** The slice of DedicatedWorkerGlobalScope we use (lib.webworker conflicts with lib.dom in one program). */
const scope = self as unknown as {
  postMessage(msg: Response): void;
  onmessage: ((e: MessageEvent<Request>) => void) | null;
};
let engine: Engine | undefined;
let encoder: EncoderEngine | undefined;
let seq2seq: Seq2SeqEngine | undefined;
let audio: AudioEngine | TtsEngine | undefined;
let vision: VisionEngine | undefined;

const OUTPUT_KEY: Record<Seq2SeqTask, string> = { summarization: "summary_text", translation: "translation_text", "text2text-generation": "generated_text", "automatic-speech-recognition": "text" };

/**
 * A text2text call as transformers.js makes it: config.prefix and the task's prefix from
 * task_specific_params before each text, 256 new tokens unless told otherwise, one result object
 * per text.
 */
async function runSeq2Seq(e: Seq2SeqEngine, task: Seq2SeqTask, texts: string | string[], options: Seq2SeqGenerateOptions = {}): Promise<Record<string, string>[]> {
  const raw = e.cfg.raw;
  const tsp = (raw.task_specific_params as Record<string, { prefix?: string }> | undefined)?.[task];
  const prefix = String(raw.prefix ?? "") + (tsp?.prefix ?? "");
  const out: Record<string, string>[] = [];
  for (const t of Array.isArray(texts) ? texts : [texts]) {
    out.push({ [OUTPUT_KEY[task]]: await e.run(t, { max_new_tokens: 256, ...options }, prefix) });
  }
  return out;
}

/** Runs an encoder task with transformers.js's arguments. */
function runTask(e: EncoderEngine, task: EncoderTask, args: unknown[]): Promise<unknown> {
  const a = args as never[];
  switch (task) {
    case "feature-extraction":
      return e.featureExtraction(a[0], a[1]);
    case "text-classification":
      return e.textClassification(a[0], a[1]);
    case "token-classification":
      return e.tokenClassification(a[0], a[1]);
    case "question-answering":
      return e.questionAnswering(a[0], a[1], a[2]);
    case "fill-mask":
      return e.fillMask(a[0], a[1]);
    case "zero-shot-classification":
      return e.zeroShotClassification(a[0], a[1], a[2]);
  }
}
const aborts = new Map<number, AbortController>();
/** Prompt cache file of the loaded model (undefined when off), and the tokens last written to it. */
let cacheName: string | undefined;
let savedTokens: readonly number[] = [];
let saving: Promise<void> = Promise.resolve();

/** Writes the engine's cached K/V unless a call has started since or nothing changed. */
async function saveCache(e: Engine, name: string): Promise<void> {
  // The prompt cache restores into the base model at the next load: K/V an adapter made would not fit it.
  if (e.adapterInfo.active !== undefined) return;
  try {
    const snap = await e.exportCache();
    if (snap.tokens.length === savedTokens.length && snap.tokens.every((t, i) => t === savedTokens[i])) return;
    if (await savePromptCache(name, snap)) savedTokens = snap.tokens;
  } catch {
    // Busy with the next call, or storage refused: the cache is only an optimisation.
  }
}

const post = (msg: Response) => scope.postMessage(msg);

/**
 * Generation calls run one at a time, in arrival order: a second conversation's call waits for the
 * first instead of failing, and starts from its own saved prefix (see `Engine.prefixCacheBytes`).
 */
let queue: Promise<void> = Promise.resolve();

/**
 * An image part as the vision tower takes it: RGBA bytes at its size, or at the image's own size
 * (`size` undefined). Pixel data of that size passes through; anything else (Blob, File, ImageBitmap,
 * ImageData of another size) is decoded, and resized by the browser (high-quality resampling),
 * through an OffscreenCanvas.
 */
async function toPixels(image: unknown, size: number | undefined): Promise<{ width: number; height: number; data: Uint8ClampedArray | Uint8Array }> {
  const px = image as { width?: number; height?: number; data?: Uint8ClampedArray | Uint8Array };
  const bytes = px && (px.data instanceof Uint8ClampedArray || px.data instanceof Uint8Array);
  if (bytes && (size === undefined || (px.width === size && px.height === size))) return { width: px.width!, height: px.height!, data: px.data! };
  const source = px && px.data && !(image instanceof ImageData) ? new ImageData(new Uint8ClampedArray(px.data), px.width!, px.height!) : (image as ImageBitmapSource);
  const bitmap = await createImageBitmap(source, size === undefined ? {} : { resizeWidth: size, resizeHeight: size, resizeQuality: "high" });
  const [w, h] = [bitmap.width, bitmap.height];
  const ctx = new OffscreenCanvas(w, h).getContext("2d")!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, w, h);
}

async function generate(msg: Extract<Request, { type: "chat" | "complete" }>, ac: AbortController): Promise<void> {
  try {
    if (!engine) throw new Error("no model loaded");
    const opts = { ...msg.opts, signal: ac.signal };
    let messages = msg.type === "chat" ? msg.messages : [];
    // Gemma 3's tower takes the image squashed to its size; SmolVLM's processor tiles the full image.
    const vision = engine.vision;
    const size = vision?.processing ? undefined : vision?.cfg.imageSize;
    if (vision && messages.some((m) => typeof m.content !== "string")) {
      messages = await Promise.all(
        messages.map(async (m) =>
          typeof m.content === "string" ? m : { ...m, content: await Promise.all(m.content.map(async (p) => (p.type === "image" ? { type: "image" as const, image: await toPixels(p.image, size) } : p))) },
        ),
      );
    }
    const stream: AsyncGenerator<TextDelta> = msg.type === "chat" ? engine.chat(messages, opts) : engine.complete(msg.prompt, opts);
    try {
      for await (const d of stream) post({ id: msg.id, type: "delta", text: d.text });
    } finally {
      aborts.delete(msg.id);
    }
    post({ id: msg.id, type: "done", stats: engine.lastStats });
    // After the reply is out, so saving never delays it.
    const e = engine;
    const name = cacheName;
    if (name) saving = saving.then(() => saveCache(e, name));
  } catch (err) {
    aborts.delete(msg.id);
    post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
  }
}

scope.onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case "load": {
        await queue;
        await saving;
        engine?.destroy();
        engine = await createEngine(msg.model, { ...msg.opts, onProgress: (progress) => post({ id: msg.id, type: "progress", progress }) });
        cacheName = msg.opts.promptCache === false ? undefined : promptCacheName(msg.model, msg.opts);
        savedTokens = [];
        const snap = cacheName ? await loadPromptCache(cacheName) : undefined;
        if (snap && engine.importCache(snap)) savedTokens = snap.tokens;
        const cfg = engine.model.cfg;
        post({
          id: msg.id,
          type: "loaded",
          info: {
            model: msg.model,
            family: cfg.family,
            backend: engine.backend.name,
            cpuThreads: engine.backend.name === "cpu" ? (engine.backend as { threads?: number }).threads : undefined,
            vision: !!engine.vision,
            layers: cfg.numLayers,
            hiddenSize: cfg.hiddenSize,
            vocabSize: cfg.vocabSize,
            maxSeq: engine.model.maxSeq,
            contextLength: cfg.contextLength,
            chatFormat: engine.chatFormat?.name,
            backendFeatures: [...engine.backend.limits.features],
            restoredTokens: savedTokens.length,
          },
        });
        break;
      }
      case "loadEncoder": {
        await queue;
        encoder?.destroy();
        encoder = undefined;
        const enc = await createEncoder(msg.model, { ...msg.opts, onProgress: (progress) => post({ id: msg.id, type: "progress", progress }) });
        encoder = enc;
        post({
          id: msg.id,
          type: "encoderLoaded",
          info: {
            model: msg.model,
            family: enc.cfg.family,
            layers: enc.cfg.layers,
            hiddenSize: enc.cfg.hidden,
            vocabSize: enc.cfg.vocab,
            maxLength: enc.maxLength,
            backend: enc.backend.name,
            cpuThreads: enc.backend.name === "cpu" ? (enc.backend as { threads?: number }).threads : undefined,
            labels: enc.cfg.id2label,
          },
        });
        break;
      }
      case "loadSeq2Seq": {
        await queue;
        seq2seq?.destroy();
        seq2seq = undefined;
        const m = await createSeq2Seq(msg.model, { ...msg.opts, onProgress: (progress) => post({ id: msg.id, type: "progress", progress }) });
        seq2seq = m;
        post({
          id: msg.id,
          type: "seq2seqLoaded",
          info: {
            model: msg.model,
            family: m.cfg.family,
            encoderLayers: m.cfg.encLayers,
            decoderLayers: m.cfg.decLayers,
            dModel: m.cfg.dModel,
            vocabSize: m.cfg.vocab,
            backend: m.backend.name,
            cpuThreads: m.backend.name === "cpu" ? (m.backend as { threads?: number }).threads : undefined,
          },
        });
        break;
      }
      case "loadAudio": {
        await queue;
        audio?.destroy();
        audio = undefined;
        const a = await createAudio(msg.model, { ...msg.opts, onProgress: (progress) => post({ id: msg.id, type: "progress", progress }) });
        audio = a;
        post({
          id: msg.id,
          type: "audioLoaded",
          info:
            a instanceof TtsEngine
              ? { model: msg.model, family: a.info.family, head: "tts", layers: a.info.layers, hiddenSize: a.info.hidden, backend: a.backend.name, cpuThreads: a.backend.name === "cpu" ? (a.backend as { threads?: number }).threads : undefined }
              : {
                  model: msg.model,
                  family: a.cfg.modelType,
                  head: a.classifier ? "classification" : "ctc",
                  layers: a.cfg.layers,
                  hiddenSize: a.cfg.hidden,
                  backend: a.backend.name,
                  cpuThreads: a.backend.name === "cpu" ? (a.backend as { threads?: number }).threads : undefined,
                  labels: a.cfg.id2label,
                },
        });
        break;
      }
      case "loadVision": {
        await queue;
        vision?.destroy();
        vision = undefined;
        const v = await createVision(msg.model, { ...msg.opts, onProgress: (progress) => post({ id: msg.id, type: "progress", progress }) });
        vision = v;
        post({
          id: msg.id,
          type: "visionLoaded",
          info: {
            model: msg.model,
            ...v.summary(),
            backend: v.backend.name,
            cpuThreads: v.backend.name === "cpu" ? (v.backend as { threads?: number }).threads : undefined,
          },
        });
        break;
      }
      case "visionTask": {
        const v = vision;
        if (!v) throw new Error("no image model loaded");
        queue = queue.then(async () => {
          try {
            const task = msg.task as VisionTask;
            const img = await readImage(msg.args[0] as ImageInput);
            const result =
              task === "zero-shot-image-classification"
                ? await v.zeroShot(img, msg.args[1] as string[], msg.args[2] as { hypothesis_template?: string } | undefined)
                : task === "image-classification"
                  ? await v.classify(img, msg.args[1] as Record<string, unknown> | undefined)
                  : task === "object-detection"
                    ? await v.detect(img, msg.args[1] as Record<string, unknown> | undefined)
                    : task === "zero-shot-object-detection"
                      ? await v.detectZeroShot(img, msg.args[1] as string[], msg.args[2] as Record<string, unknown> | undefined)
                    : task === "depth-estimation"
                      ? await v.estimateDepth(img)
                      : task === "image-segmentation"
                        ? await v.segment(img, msg.args[1] as Record<string, unknown> | undefined)
                        : task === "background-removal"
                          ? await v.removeBackground(img)
                          : task === "document-question-answering"
                            ? await v.answerDocument(img, msg.args[1] as string, msg.args[2] as Record<string, unknown> | undefined)
                          : task === "image-to-image"
                            ? await v.upscale(img)
                            : task === "image-to-text"
                              ? await v.caption(img, msg.args[1] as Record<string, unknown> | undefined)
                      : await v.features(img, msg.args[1] as Record<string, unknown> | undefined);
            post({ id: msg.id, type: "result", result });
          } catch (err) {
            post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
          }
        });
        break;
      }
      case "audioTask": {
        const a = audio;
        if (!a) throw new Error("no speech model loaded");
        queue = queue.then(async () => {
          try {
            let result: unknown;
            if (a instanceof TtsEngine) {
              if ((msg.task as AudioTask) !== "text-to-audio") throw new Error("this is a text-to-speech model");
              result = await a.synthesize(msg.args[0] as string, msg.args[1] as Record<string, unknown> | undefined);
            } else {
              if ((msg.task as AudioTask) === "text-to-audio") throw new Error("this model does not speak; load a text-to-speech model");
              const [samples, opts] = msg.args as [Float32Array, Record<string, unknown> | undefined];
              result = (msg.task as AudioTask) === "audio-classification" ? await a.classify(samples, opts) : await a.transcribe(samples);
            }
            post({ id: msg.id, type: "result", result });
          } catch (err) {
            post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
          }
        });
        break;
      }
      case "task": {
        const isS2S = msg.task in OUTPUT_KEY;
        const enc = encoder;
        const s2s = seq2seq;
        if (isS2S ? !s2s : !enc) throw new Error(`no ${isS2S ? "encoder-decoder" : "encoder"} model loaded`);
        // One call at a time, like generation.
        queue = queue.then(async () => {
          try {
            const a = msg.args as never[];
            const result =
              msg.task === "automatic-speech-recognition"
                ? await s2s!.transcribe(a[0] as unknown as Float32Array, a[1])
                : isS2S
                  ? await runSeq2Seq(s2s!, msg.task as Seq2SeqTask, a[0], a[1])
                  : await runTask(enc!, msg.task as EncoderTask, msg.args);
            post({ id: msg.id, type: "result", result });
          } catch (err) {
            post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
          }
        });
        break;
      }
      case "encoderForward": {
        const enc = encoder;
        if (!enc) throw new Error("no encoder model loaded");
        queue = queue.then(async () => {
          try {
            post({ id: msg.id, type: "result", result: await enc.forwardHeads(msg.rows, msg.head) });
          } catch (err) {
            post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
          }
        });
        break;
      }
      case "visionForward": {
        const v = vision;
        if (!v) throw new Error("no image model loaded");
        queue = queue.then(async () => {
          try {
            const result: { images: RawOutputs[]; texts?: Float32Array[]; scale?: number } = { images: [] };
            for (const im of msg.images ?? []) result.images.push(await v.forwardPixels(im.data, im.width, im.height, msg.head));
            if (msg.texts?.length) {
              const t = await v.clipTextEmbeds(msg.texts);
              result.texts = t.embeds;
              result.scale = t.scale;
            }
            post({ id: msg.id, type: "result", result });
          } catch (err) {
            post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
          }
        });
        break;
      }
      case "score": {
        const e = engine;
        if (!e) throw new Error("no model loaded");
        queue = queue.then(async () => {
          try {
            post({ id: msg.id, type: "result", result: await e.scoreChoices(msg.messages, msg.choices, msg.opts) });
          } catch (err) {
            post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
          }
        });
        break;
      }
      case "adapter": {
        const e = engine;
        if (!e) throw new Error("no model loaded");
        // After the calls already queued, which run with the adapter they started with.
        queue = queue.then(async () => {
          try {
            let result: unknown;
            if (msg.op === "load") result = await e.loadAdapter(msg.name!, msg.model!, { revision: msg.revision, endpoint: msg.endpoint });
            else if (msg.op === "set") e.setAdapter(msg.name ?? null);
            else if (msg.op === "unload") e.unloadAdapter(msg.name!);
            post({ id: msg.id, type: "result", result: result ?? e.adapterInfo });
          } catch (err) {
            post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
          }
        });
        break;
      }
      case "generateIds": {
        const e = engine;
        if (!e) throw new Error("no model loaded");
        const ac = new AbortController();
        aborts.set(msg.id, ac);
        queue = queue.then(async () => {
          try {
            const decoder = e.tokenizer.decoder();
            for await (const tokenId of e.generateTokens(msg.ids, { ...msg.opts, signal: ac.signal })) post({ id: msg.id, type: "delta", text: decoder.push(tokenId), tokenId });
            const tail = decoder.flush();
            if (tail) post({ id: msg.id, type: "delta", text: tail });
            post({ id: msg.id, type: "done", stats: e.lastStats, stopId: e.lastStopId });
          } catch (err) {
            post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
          } finally {
            aborts.delete(msg.id);
          }
        });
        break;
      }
      case "chat":
      case "complete": {
        if (!engine) throw new Error("no model loaded");
        // Registered now, so a call still waiting in the queue can be aborted too.
        const ac = new AbortController();
        aborts.set(msg.id, ac);
        queue = queue.then(() => generate(msg, ac));
        break;
      }
      case "abort":
        aborts.get(msg.target)?.abort();
        break;
      case "profile": {
        const be = (engine ?? vision ?? audio ?? encoder ?? seq2seq)?.backend;
        if (!be) throw new Error("no model loaded");
        if (!be.startProfile || !be.stopProfile) throw new Error(`[kiln] the ${be.name} backend cannot profile`);
        if (msg.on) {
          be.startProfile();
          post({ id: msg.id, type: "profiled", profile: undefined });
        } else {
          // After the calls already queued, so their kernels are counted.
          await queue;
          post({ id: msg.id, type: "profiled", profile: await be.stopProfile() });
        }
        break;
      }
      case "unload":
        await queue;
        await saving;
        engine?.destroy();
        engine = undefined;
        encoder?.destroy();
        encoder = undefined;
        seq2seq?.destroy();
        seq2seq = undefined;
        audio?.destroy();
        audio = undefined;
        vision?.destroy();
        vision = undefined;
        post({ id: msg.id, type: "done", stats: undefined });
        break;
    }
  } catch (err) {
    post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
