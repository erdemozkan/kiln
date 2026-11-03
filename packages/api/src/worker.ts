import { createEngine, type Engine, type TextDelta } from "@kiln/runtime";
import type { Request, Response } from "./protocol";

/** The slice of DedicatedWorkerGlobalScope we use (lib.webworker conflicts with lib.dom in one program). */
const scope = self as unknown as {
  postMessage(msg: Response): void;
  onmessage: ((e: MessageEvent<Request>) => void) | null;
};
let engine: Engine | undefined;
const aborts = new Map<number, AbortController>();

const post = (msg: Response) => scope.postMessage(msg);

scope.onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case "load": {
        engine?.destroy();
        engine = await createEngine(msg.model, { ...msg.opts, onProgress: (progress) => post({ id: msg.id, type: "progress", progress }) });
        const cfg = engine.model.cfg;
        post({
          id: msg.id,
          type: "loaded",
          info: {
            model: msg.model,
            family: cfg.family,
            layers: cfg.numLayers,
            hiddenSize: cfg.hiddenSize,
            vocabSize: cfg.vocabSize,
            maxSeq: engine.model.maxSeq,
            chatFormat: engine.chatFormat?.name,
            backendFeatures: [...engine.backend.limits.features],
          },
        });
        break;
      }
      case "chat":
      case "complete": {
        if (!engine) throw new Error("no model loaded");
        const ac = new AbortController();
        aborts.set(msg.id, ac);
        const opts = { ...msg.opts, signal: ac.signal };
        const stream: AsyncGenerator<TextDelta> = msg.type === "chat" ? engine.chat(msg.messages, opts) : engine.complete(msg.prompt, opts);
        try {
          for await (const d of stream) post({ id: msg.id, type: "delta", text: d.text });
        } finally {
          aborts.delete(msg.id);
        }
        post({ id: msg.id, type: "done", stats: engine.lastStats });
        break;
      }
      case "abort":
        aborts.get(msg.target)?.abort();
        break;
      case "unload":
        engine?.destroy();
        engine = undefined;
        post({ id: msg.id, type: "done", stats: undefined });
        break;
    }
  } catch (err) {
    post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
