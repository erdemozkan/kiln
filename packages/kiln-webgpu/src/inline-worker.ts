// The published package's worker: its code is embedded in the bundle and started from a blob URL, so
// it runs wherever the package is loaded from (a CDN, a bundler's output, Vite's pre-bundled deps)
// with no separate file to find. Swapped in for @kiln/api's default-worker by vite.config.ts.
// In Node.js the same code starts on a worker thread (see node-worker.ts).
import KilnWorker from "../../api/src/worker.ts?worker&inline";
import { needsNodeWorkers, withNodeWorker } from "./node-worker";

export function createWorker(): Worker {
  if (needsNodeWorkers()) return withNodeWorker(() => new KilnWorker({ name: "kiln" }));
  return new KilnWorker({ name: "kiln" });
}
