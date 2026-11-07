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

async function generate(msg: Extract<Request, { type: "chat" | "complete" }>, ac: AbortController): Promise<void> {
  try {
    if (!engine) throw new Error("no model loaded");
    const opts = { ...msg.opts, signal: ac.signal };
    const stream: AsyncGenerator<TextDelta> = msg.type === "chat" ? engine.chat(msg.messages, opts) : engine.complete(msg.prompt, opts);
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
