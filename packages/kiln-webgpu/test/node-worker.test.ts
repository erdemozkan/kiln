import { describe, expect, it } from "vitest";
import { needsNodeWorkers, withNodeWorker, type NodeWebWorker } from "../src/node-worker";

const start = (code: string): NodeWebWorker => withNodeWorker(() => new (globalThis as unknown as { Worker: new (url: string) => NodeWebWorker }).Worker(URL.createObjectURL(new Blob([code]))));
const reply = (w: NodeWebWorker) => new Promise<unknown>((resolve, reject) => ((w.onmessage = (e) => resolve(e.data)), (w.onerror = (e) => reject(new Error(e.message)))));

describe("Web Workers on Node's worker_threads", () => {
  it("applies in Node, and leaves no global Worker behind", () => {
    expect(needsNodeWorkers()).toBe(true);
    start("onmessage = () => {};").terminate();
    expect((globalThis as { Worker?: unknown }).Worker).toBeUndefined();
  });

  it("gives the thread a worker's scope: self, onmessage, postMessage with transfers", async () => {
    const w = start("self.onmessage = (e) => { const b = new Float32Array(e.data.n).fill(2); postMessage({ b, iso: crossOriginIsolated }, [b.buffer]); };");
    w.postMessage({ n: 3 });
    expect(await reply(w)).toEqual({ b: new Float32Array([2, 2, 2]), iso: true });
    w.terminate();
  });

  it("lets a thread start its own threads from blob URLs, sharing memory", async () => {
    const inner = "onmessage = (e) => { const a = new Int32Array(e.data); a[0] = 7; Atomics.notify(a, 0); };";
    // The thread starts once its script is read (a turn later), so the caller waits without blocking.
    const w = start(`onmessage = async () => {
      const a = new Int32Array(new SharedArrayBuffer(4));
      const url = URL.createObjectURL(new Blob([${JSON.stringify(inner)}]));
      const h = new Worker(url);
      URL.revokeObjectURL(url);
      h.postMessage(a.buffer);
      await Atomics.waitAsync(a, 0, 0, 5000).value;
      postMessage(a[0]);
      h.terminate();
    };`);
    w.postMessage(0);
    expect(await reply(w)).toBe(7);
    w.terminate();
  });

  it("reports a thread that throws", async () => {
    const w = start("throw new Error('boom');");
    await expect(reply(w)).rejects.toThrow(/boom/);
    w.terminate();
  });
});
