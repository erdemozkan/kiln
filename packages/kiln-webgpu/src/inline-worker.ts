// The published package's worker: its code is embedded in the bundle and started from a blob URL, so
// it runs wherever the package is loaded from (a CDN, a bundler's output, Vite's pre-bundled deps)
// with no separate file to find. Swapped in for @kiln/api's default-worker by vite.config.ts.
import KilnWorker from "../../api/src/worker.ts?worker&inline";

export function createWorker(): Worker {
  return new KilnWorker({ name: "kiln" });
}
