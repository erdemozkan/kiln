import { createEngine, loadPromptCache, promptCacheName, savePromptCache, type Engine, type TextDelta } from "@kiln/runtime";
import type { Request, Response } from "./protocol";

/** The slice of DedicatedWorkerGlobalScope we use (lib.webworker conflicts with lib.dom in one program). */
const scope = self as unknown as {
  postMessage(msg: Response): void;
  onmessage: ((e: MessageEvent<Request>) => void) | null;
};
let engine: Engine | undefined;
const aborts = new Map<number, AbortController>();
/** Prompt cache file of the loaded model (undefined when off), and the tokens last written to it. */
let cacheName: string | undefined;
let savedTokens: readonly number[] = [];
let saving: Promise<void> = Promise.resolve();

/** Writes the engine's cached K/V unless a call has started since or nothing changed. */
async function saveCache(e: Engine, name: string): Promise<void> {
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
 * An image part as the vision tower takes it: RGBA bytes at its size. Pixel data of that size passes
 * through; anything else (Blob, File, ImageBitmap, ImageData of another size) is resized by the
 * browser (high-quality resampling) through an OffscreenCanvas.
 */
async function toPixels(image: unknown, size: number): Promise<{ width: number; height: number; data: Uint8ClampedArray | Uint8Array }> {
  const px = image as { width?: number; height?: number; data?: Uint8ClampedArray | Uint8Array };
  if (px && px.width === size && px.height === size && (px.data instanceof Uint8ClampedArray || px.data instanceof Uint8Array)) {
    return { width: size, height: size, data: px.data };
  }
  const source = px && px.data && !(image instanceof ImageData) ? new ImageData(new Uint8ClampedArray(px.data), px.width!, px.height!) : (image as ImageBitmapSource);
  const bitmap = await createImageBitmap(source, { resizeWidth: size, resizeHeight: size, resizeQuality: "high" });
  const ctx = new OffscreenCanvas(size, size).getContext("2d")!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, size, size);
}

async function generate(msg: Extract<Request, { type: "chat" | "complete" }>, ac: AbortController): Promise<void> {
  try {
    if (!engine) throw new Error("no model loaded");
    const opts = { ...msg.opts, signal: ac.signal };
    let messages = msg.type === "chat" ? msg.messages : [];
    const size = engine.vision?.cfg.imageSize;
    if (size && messages.some((m) => typeof m.content !== "string")) {
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
      case "unload":
        await queue;
        await saving;
        engine?.destroy();
        engine = undefined;
        post({ id: msg.id, type: "done", stats: undefined });
        break;
    }
  } catch (err) {
    post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
