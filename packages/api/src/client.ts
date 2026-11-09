import type { Profile } from "@kiln/core";
import type { GenerateStats, LoadProgress } from "@kiln/runtime";
import type { ChatMessage } from "@kiln/tokenizers";
import { createWorker } from "./default-worker";
import type { VisionInfo, VisionTask, AudioInfo, AudioTask, EncoderInfo, EncoderTask, ModelInfo, Request, Response, Seq2SeqInfo, Seq2SeqTask, WireEncoderOptions, WireGenerateOptions, WireLoadOptions, WireSeq2SeqOptions } from "./protocol";

type Listener = (msg: Response) => void;
type RequestBody = Request extends infer R ? (R extends Request ? Omit<R, "id"> : never) : never;

export interface LoadOptions extends WireLoadOptions {
  onProgress?: (p: LoadProgress) => void;
  /** Supply your own worker (e.g. a bundler-specific URL). Defaults to Kiln's module worker. */
  worker?: Worker;
  /**
   * Ask the browser to keep the weight cache (`navigator.storage.persist()`, default true), so a
   * multi-GB download is not evicted when the disk runs low. Chrome and Safari decide on their own;
   * Firefox may ask the user. `info.persistent` holds the answer once the browser gives it.
   */
  persist?: boolean;
}

/** True if site storage is (now) persistent, undefined where the browser cannot say. */
async function requestPersistence(): Promise<boolean | undefined> {
  const storage = globalThis.navigator?.storage;
  if (!storage?.persist) return undefined;
  try {
    return (await storage.persisted()) || (await storage.persist());
  } catch {
    return undefined;
  }
}

export interface GenerateOptions extends WireGenerateOptions {
  signal?: AbortSignal;
}

/** A text stream that also exposes timing once it finishes. */
export interface TextStream extends AsyncIterable<string> {
  /** Resolves when generation ends. */
  readonly stats: Promise<GenerateStats | undefined>;
}

/** Request/response plumbing to a Kiln worker, shared by language models and encoders. */
abstract class WorkerChannel {
  protected nextId = 1;
  private readonly listeners = new Map<number, Listener>();

  protected constructor(protected readonly worker: Worker) {
    worker.onmessage = (e: MessageEvent<Response>) => this.listeners.get(e.data.id)?.(e.data);
    // A worker that fails to start or crashes would otherwise leave every pending call hanging.
    worker.onerror = (e: ErrorEvent) => {
      e.preventDefault();
      const message = `Kiln worker failed: ${e.message || "it could not start (see the browser console)"}`;
      for (const [id, listener] of [...this.listeners]) listener({ id, type: "error", message });
    };
  }

  /** Sends a request; `onMsg` returns true when the exchange is complete. */
  protected send(body: RequestBody, onMsg: (msg: Response) => boolean): number {
    const id = this.nextId++;
    this.listeners.set(id, (msg) => {
      if (onMsg(msg)) this.listeners.delete(id);
    });
    this.post({ ...body, id } as Request);
    return id;
  }

  protected post(msg: Request): void {
    this.worker.postMessage(msg);
  }

  /** Unloads the model and stops the worker. */
  async unload(): Promise<void> {
    await new Promise<void>((resolve) => this.send({ type: "unload" }, () => (resolve(), true)));
    this.worker.terminate();
  }
}

/** Main-thread handle to a model running in a worker. */
export class KilnModel extends WorkerChannel {
  private constructor(
    worker: Worker,
    public info: ModelInfo,
  ) {
    super(worker);
  }

  static async load(model: string, opts: LoadOptions = {}): Promise<KilnModel> {
    const worker = opts.worker ?? createWorker();
    const m = new KilnModel(worker, undefined as unknown as ModelInfo);
    const { onProgress, worker: _w, persist = true, ...wire } = opts;
    const persistent = persist ? requestPersistence() : Promise.resolve(undefined);
    m.info = await new Promise<ModelInfo>((resolve, reject) => {
      m.send({ type: "load", model, opts: wire }, (msg) => {
        if (msg.type === "progress") onProgress?.(msg.progress);
        else if (msg.type === "loaded") resolve(msg.info);
        else if (msg.type === "error") reject(new Error(msg.message));
        return msg.type === "loaded" || msg.type === "error";
      });
    });
    // Not awaited: a permission prompt the user ignores must not hold up the load.
    void persistent.then((v) => (m.info.persistent = v));
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

  /**
   * Start timing every kernel the model runs: GPU time per dispatch from timestamp queries on WebGPU
   * (where the device offers them), wall clock on the CPU backend. For finding where time goes, not for
   * normal use: each GPU dispatch then runs in a pass of its own, which costs a little extra.
   */
  startProfile(): Promise<void> {
    return this.profile(true).then(() => undefined);
  }

  /** Stop timing; time per kernel and shape since `startProfile` (after calls already made finish), most expensive first. */
  async stopProfile(): Promise<Profile> {
    return (await this.profile(false))!;
  }

  private profile(on: boolean): Promise<Profile | undefined> {
    return new Promise((resolve, reject) =>
      this.send({ type: "profile", on }, (msg) => {
        if (msg.type === "profiled") resolve(msg.profile);
        else if (msg.type === "error") reject(new Error(msg.message));
        return msg.type === "profiled" || msg.type === "error";
      }),
    );
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
}

export const load = KilnModel.load;

export interface EncoderLoadOptions extends WireEncoderOptions {
  onProgress?: (p: LoadProgress) => void;
  /** Supply your own worker. Defaults to Kiln's module worker. */
  worker?: Worker;
}

/** Main-thread handle to an encoder model (BERT, DistilBERT, RoBERTa) running in a worker. */
export class KilnEncoder extends WorkerChannel {
  private constructor(
    worker: Worker,
    public info: EncoderInfo,
  ) {
    super(worker);
  }

  static async load(model: string, opts: EncoderLoadOptions = {}): Promise<KilnEncoder> {
    const worker = opts.worker ?? createWorker();
    const m = new KilnEncoder(worker, undefined as unknown as EncoderInfo);
    const { onProgress, worker: _w, ...wire } = opts;
    try {
      m.info = await new Promise<EncoderInfo>((resolve, reject) => {
        m.send({ type: "loadEncoder", model, opts: wire }, (msg) => {
          if (msg.type === "progress") onProgress?.(msg.progress);
          else if (msg.type === "encoderLoaded") resolve(msg.info);
          else if (msg.type === "error") reject(new Error(msg.message));
          return msg.type === "encoderLoaded" || msg.type === "error";
        });
      });
    } catch (err) {
      if (!opts.worker) worker.terminate();
      throw err;
    }
    return m;
  }

  /** Runs a task with transformers.js's arguments; resolves to its output. */
  run<T>(task: EncoderTask, ...args: unknown[]): Promise<T> {
    return new Promise((resolve, reject) =>
      this.send({ type: "task", task, args }, (msg) => {
        if (msg.type === "result") resolve(msg.result as T);
        else if (msg.type === "error") reject(new Error(msg.message));
        return msg.type === "result" || msg.type === "error";
      }),
    );
  }
}

export const loadEncoder = KilnEncoder.load;

export interface Seq2SeqLoadOptions extends WireSeq2SeqOptions {
  onProgress?: (p: LoadProgress) => void;
  /** Supply your own worker. Defaults to Kiln's module worker. */
  worker?: Worker;
}

/** Main-thread handle to an encoder-decoder model (BART, Marian, M2M100 / NLLB) running in a worker. */
export class KilnSeq2Seq extends WorkerChannel {
  private constructor(
    worker: Worker,
    public info: Seq2SeqInfo,
  ) {
    super(worker);
  }

  static async load(model: string, opts: Seq2SeqLoadOptions = {}): Promise<KilnSeq2Seq> {
    const worker = opts.worker ?? createWorker();
    const m = new KilnSeq2Seq(worker, undefined as unknown as Seq2SeqInfo);
    const { onProgress, worker: _w, ...wire } = opts;
    try {
      m.info = await new Promise<Seq2SeqInfo>((resolve, reject) => {
        m.send({ type: "loadSeq2Seq", model, opts: wire }, (msg) => {
          if (msg.type === "progress") onProgress?.(msg.progress);
          else if (msg.type === "seq2seqLoaded") resolve(msg.info);
          else if (msg.type === "error") reject(new Error(msg.message));
          return msg.type === "seq2seqLoaded" || msg.type === "error";
        });
      });
    } catch (err) {
      if (!opts.worker) worker.terminate();
      throw err;
    }
    return m;
  }

  /** Runs summarization, translation or text2text-generation on one text or several, or speech recognition on samples. */
  run(task: Seq2SeqTask, texts: string | string[] | Float32Array, options: Record<string, unknown> = {}): Promise<Record<string, string>[]> {
    return new Promise((resolve, reject) =>
      this.send({ type: "task", task, args: [texts, options] }, (msg) => {
        if (msg.type === "result") resolve(msg.result as Record<string, string>[]);
        else if (msg.type === "error") reject(new Error(msg.message));
        return msg.type === "result" || msg.type === "error";
      }),
    );
  }
}

export const loadSeq2Seq = KilnSeq2Seq.load;

export interface AudioLoadOptions extends WireEncoderOptions {
  onProgress?: (p: LoadProgress) => void;
  /** Supply your own worker. Defaults to Kiln's module worker. */
  worker?: Worker;
}

/** Main-thread handle to a speech encoder (wav2vec2, HuBERT) running in a worker. */
export class KilnAudio extends WorkerChannel {
  private constructor(
    worker: Worker,
    public info: AudioInfo,
  ) {
    super(worker);
  }

  static async load(model: string, opts: AudioLoadOptions = {}): Promise<KilnAudio> {
    const worker = opts.worker ?? createWorker();
    const m = new KilnAudio(worker, undefined as unknown as AudioInfo);
    const { onProgress, worker: _w, ...wire } = opts;
    try {
      m.info = await new Promise<AudioInfo>((resolve, reject) => {
        m.send({ type: "loadAudio", model, opts: wire }, (msg) => {
          if (msg.type === "progress") onProgress?.(msg.progress);
          else if (msg.type === "audioLoaded") resolve(msg.info);
          else if (msg.type === "error") reject(new Error(msg.message));
          return msg.type === "audioLoaded" || msg.type === "error";
        });
      });
    } catch (err) {
      if (!opts.worker) worker.terminate();
      throw err;
    }
    return m;
  }

  /** CTC speech recognition ({ text }) or audio classification ([{ label, score }]) on 16 kHz mono samples. */
  run<T>(task: AudioTask, samples: Float32Array, options: Record<string, unknown> = {}): Promise<T> {
    return new Promise((resolve, reject) =>
      this.send({ type: "audioTask", task, args: [samples, options] }, (msg) => {
        if (msg.type === "result") resolve(msg.result as T);
        else if (msg.type === "error") reject(new Error(msg.message));
        return msg.type === "result" || msg.type === "error";
      }),
    );
  }
}

export const loadAudio = KilnAudio.load;

/** Main-thread handle to an image model (ViT, DeiT) running in a worker. */
export class KilnVision extends WorkerChannel {
  private constructor(
    worker: Worker,
    public info: VisionInfo,
  ) {
    super(worker);
  }

  static async load(model: string, opts: AudioLoadOptions = {}): Promise<KilnVision> {
    const worker = opts.worker ?? createWorker();
    const m = new KilnVision(worker, undefined as unknown as VisionInfo);
    const { onProgress, worker: _w, ...wire } = opts;
    try {
      m.info = await new Promise<VisionInfo>((resolve, reject) => {
        m.send({ type: "loadVision", model, opts: wire }, (msg) => {
          if (msg.type === "progress") onProgress?.(msg.progress);
          else if (msg.type === "visionLoaded") resolve(msg.info);
          else if (msg.type === "error") reject(new Error(msg.message));
          return msg.type === "visionLoaded" || msg.type === "error";
        });
      });
    } catch (err) {
      if (!opts.worker) worker.terminate();
      throw err;
    }
    return m;
  }

  /** Image classification ([{ label, score }]) or image features ({ data, dims }) of an image the worker can decode. */
  run<T>(task: VisionTask, image: unknown, options: Record<string, unknown> = {}): Promise<T> {
    return new Promise((resolve, reject) =>
      this.send({ type: "visionTask", task, args: [image, options] }, (msg) => {
        if (msg.type === "result") resolve(msg.result as T);
        else if (msg.type === "error") reject(new Error(msg.message));
        return msg.type === "result" || msg.type === "error";
      }),
    );
  }
}

export const loadVision = KilnVision.load;
