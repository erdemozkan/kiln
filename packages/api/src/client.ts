import type { GenerateStats, LoadProgress } from "@kiln/runtime";
import type { ChatMessage } from "@kiln/tokenizers";
import type { ModelInfo, Request, Response, WireGenerateOptions, WireLoadOptions } from "./protocol";

type Listener = (msg: Response) => void;
type RequestBody = Request extends infer R ? (R extends Request ? Omit<R, "id"> : never) : never;

export interface LoadOptions extends WireLoadOptions {
  onProgress?: (p: LoadProgress) => void;
  /** Supply your own worker (e.g. a bundler-specific URL). Defaults to Kiln's module worker. */
  worker?: Worker;
}

export interface GenerateOptions extends WireGenerateOptions {
  signal?: AbortSignal;
}

/** A text stream that also exposes timing once it finishes. */
export interface TextStream extends AsyncIterable<string> {
  /** Resolves when generation ends. */
  readonly stats: Promise<GenerateStats | undefined>;
}

/** Main-thread handle to a model running in a worker. */
export class KilnModel {
  private nextId = 1;
  private readonly listeners = new Map<number, Listener>();

  private constructor(
    private readonly worker: Worker,
    public info: ModelInfo,
  ) {
    worker.onmessage = (e: MessageEvent<Response>) => this.listeners.get(e.data.id)?.(e.data);
    // A worker that fails to start or crashes would otherwise leave every pending call hanging.
    worker.onerror = (e: ErrorEvent) => {
      e.preventDefault();
      const message = `Kiln worker failed: ${e.message || "it could not start (see the browser console)"}`;
      for (const [id, listener] of [...this.listeners]) listener({ id, type: "error", message });
    };
  }

  static async load(model: string, opts: LoadOptions = {}): Promise<KilnModel> {
    const worker = opts.worker ?? new Worker(new URL("./worker.ts", import.meta.url), { type: "module", name: "kiln" });
    const m = new KilnModel(worker, undefined as unknown as ModelInfo);
    const { onProgress, worker: _w, ...wire } = opts;
    m.info = await new Promise<ModelInfo>((resolve, reject) => {
      m.send({ type: "load", model, opts: wire }, (msg) => {
        if (msg.type === "progress") onProgress?.(msg.progress);
        else if (msg.type === "loaded") resolve(msg.info);
        else if (msg.type === "error") reject(new Error(msg.message));
        return msg.type === "loaded" || msg.type === "error";
      });
    });
    return m;
  }

  chat(messages: ChatMessage[], opts: GenerateOptions = {}): TextStream {
    const { signal, ...wire } = opts;
    return this.stream({ type: "chat", messages, opts: wire }, signal);
  }

  complete(prompt: string, opts: GenerateOptions = {}): TextStream {
    const { signal, ...wire } = opts;
    return this.stream({ type: "complete", prompt, opts: wire }, signal);
  }

  async unload(): Promise<void> {
    await new Promise<void>((resolve) => this.send({ type: "unload" }, () => (resolve(), true)));
    this.worker.terminate();
  }

  private stream(body: RequestBody, signal?: AbortSignal): TextStream {
    const queue: string[] = [];
    let wake: (() => void) | undefined;
    let finished = false;
    let failure: Error | undefined;
    let resolveStats!: (s: GenerateStats | undefined) => void;
    const stats = new Promise<GenerateStats | undefined>((r) => (resolveStats = r));

    const id = this.send(body, (msg) => {
      if (msg.type === "delta") queue.push(msg.text);
      else if (msg.type === "done") {
        finished = true;
        resolveStats(msg.stats);
      } else if (msg.type === "error") {
        finished = true;
        failure = new Error(msg.message);
        resolveStats(undefined);
      }
      wake?.();
      return finished;
    });
    signal?.addEventListener("abort", () => this.post({ id: this.nextId++, type: "abort", target: id }), { once: true });

    return {
      stats,
      async *[Symbol.asyncIterator]() {
        for (;;) {
          while (queue.length) yield queue.shift()!;
          if (failure) throw failure;
          if (finished) return;
          await new Promise<void>((r) => (wake = r));
        }
      },
    };
  }

  /** Sends a request; `onMsg` returns true when the exchange is complete. */
  private send(body: RequestBody, onMsg: (msg: Response) => boolean): number {
    const id = this.nextId++;
    this.listeners.set(id, (msg) => {
      if (onMsg(msg)) this.listeners.delete(id);
    });
    this.post({ ...body, id } as Request);
    return id;
  }

  private post(msg: Request): void {
    this.worker.postMessage(msg);
  }
}

export const load = KilnModel.load;
