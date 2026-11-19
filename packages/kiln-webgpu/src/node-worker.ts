// Web Workers for Node.js, on worker_threads: enough of the Worker interface for Kiln's model worker
// and the CPU backend's helper threads. Node has no global Worker, so without this the package stops
// at its first model load there.

/** The part of a Web Worker Kiln uses, plus Node's ref/unref (see WorkerChannel). */
export interface NodeWebWorker {
  onmessage: ((e: { data: unknown }) => void) | null;
  onerror: ((e: { message: string; preventDefault(): void }) => void) | null;
  postMessage(msg: unknown, transfer?: unknown[]): void;
  addEventListener(type: "message" | "error", f: (e: never) => void): void;
  terminate(): void;
  ref(): void;
  unref(): void;
}

type WorkerCtor = new (url: string | URL, opts?: { name?: string; type?: string }) => NodeWebWorker;

/**
 * Builds the Worker class, and inside a worker thread also gives the thread what a Web Worker's
 * scope has (self, postMessage, onmessage) and a global Worker for the threads it starts. It must not
 * refer to anything outside itself: its own source is the prelude of every thread it starts.
 */
function installNodeWorkers(inThread: boolean): WorkerCtor {
  const g = globalThis as unknown as Record<string, unknown> & { process: { getBuiltinModule(id: string): unknown } };
  const wt = g.process.getBuiltinModule("node:worker_threads") as {
    parentPort: { postMessage(m: unknown, t?: unknown[]): void; on(ev: "message", f: (d: unknown) => void): void } | null;
    Worker: new (url: URL, opts?: { name?: string }) => {
      postMessage(m: unknown, t?: unknown[]): void;
      on(ev: string, f: (x: never) => void): void;
      terminate(): Promise<number>;
      ref(): void;
      unref(): void;
    };
  };
  const { resolveObjectURL } = g.process.getBuiltinModule("node:buffer") as { resolveObjectURL(url: string): Blob | undefined };
  const prelude = `(${installNodeWorkers.toString()})(true);\n`;

  class Worker implements NodeWebWorker {
    onmessage: NodeWebWorker["onmessage"] = null;
    onerror: NodeWebWorker["onerror"] = null;
    private failed = false;
    private thread: InstanceType<typeof wt.Worker> | undefined;
    private readonly pending: [unknown, unknown[] | undefined][] = [];
    private readonly listeners: { message: ((e: never) => void)[]; error: ((e: never) => void)[] } = { message: [], error: [] };
    private referenced = false;
    private ended = false;

    constructor(url: string | URL, opts: { name?: string } = {}) {
      const href = String(url);
      let code: string | Promise<string>;
      if (href.startsWith("blob:")) {
        // Read now: the caller may revoke the URL as soon as this returns.
        const blob = resolveObjectURL(href);
        if (!blob) throw new Error(`Worker: ${href} names no Blob`);
        code = blob.text();
      } else if (href.startsWith("data:")) {
        const comma = href.indexOf(",");
        const body = href.slice(comma + 1);
        code = href.slice(0, comma).endsWith(";base64") ? atob(body) : decodeURIComponent(body);
      } else {
        throw new Error(`Worker: only blob: and data: scripts start in Node.js (got ${href.slice(0, 40)})`);
      }
      void Promise.resolve(code).then(
        (text) => {
          if (this.ended) return;
          const t = new wt.Worker(new URL(`data:text/javascript;charset=utf-8,${encodeURIComponent(prelude + text)}`), { name: opts.name });
          if (!this.referenced) t.unref();
          t.on("message", (data: never) => {
            const e = { data } as never;
            this.onmessage?.(e);
            for (const f of this.listeners.message) f(e);
          });
          t.on("error", (err: never) => this.fail(err));
          // A thread that stops by itself (out of memory, process.exit in it) raises no error: say so,
          // or the calls waiting on it would wait for ever.
          t.on("exit", (code: never) => {
            if (!this.ended) this.fail(new Error(`the worker thread stopped (exit code ${String(code)})`));
          });
          this.thread = t;
          for (const [m, tr] of this.pending.splice(0)) t.postMessage(m, tr);
        },
        (err: unknown) => this.fail(err),
      );
    }

    private fail(err: unknown): void {
      if (this.failed) return;
      this.failed = true;
      const e = { message: err instanceof Error ? err.message : String(err), error: err, preventDefault() {} } as never;
      this.onerror?.(e);
      for (const f of this.listeners.error) f(e);
    }

    postMessage(msg: unknown, transfer?: unknown[]): void {
      if (this.thread) this.thread.postMessage(msg, transfer);
      else this.pending.push([msg, transfer]);
    }

    addEventListener(type: "message" | "error", f: (e: never) => void): void {
      this.listeners[type]?.push(f);
    }

    terminate(): void {
      this.ended = true;
      void this.thread?.terminate();
    }

    /** Keeps the process alive while the thread has work (Node exits otherwise once the caller's own work is done). */
    ref(): void {
      this.referenced = true;
      this.thread?.ref();
    }

    unref(): void {
      this.referenced = false;
      this.thread?.unref();
    }
  }

  if (inThread && wt.parentPort) {
    const port = wt.parentPort;
    g.self = g;
    g.onmessage = null;
    g.postMessage = (m: unknown, t?: unknown[]) => port.postMessage(m, t);
    port.on("message", (data) => (g.onmessage as ((e: { data: unknown }) => void) | null)?.({ data }));
    // Threads share memory and may block on Atomics.wait, which is what the CPU backend's helpers need.
    g.crossOriginIsolated = true;
    if (typeof g.Worker !== "function") g.Worker = Worker;
  }
  return Worker;
}

/** Node.js without a global Worker (Deno and Bun have one), where Kiln's threads run on worker_threads. */
export function needsNodeWorkers(): boolean {
  const g = globalThis as { Worker?: unknown; process?: { getBuiltinModule?: unknown } };
  return typeof g.Worker !== "function" && typeof g.process?.getBuiltinModule === "function";
}

let NodeWorker: WorkerCtor | undefined;

/** Runs `start` with a global Worker that starts worker_threads, and gives the global back after. */
export function withNodeWorker<T>(start: () => T): T {
  NodeWorker ??= installNodeWorkers(false);
  const g = globalThis as { Worker?: unknown };
  const before = g.Worker;
  g.Worker = NodeWorker;
  try {
    return start();
  } finally {
    g.Worker = before;
  }
}
