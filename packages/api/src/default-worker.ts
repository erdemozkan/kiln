/**
 * Starts Kiln's model worker from its own module file. The kiln-webgpu build swaps this module for
 * one that embeds the worker's code, so the published package is a single file.
 */
export function createWorker(): Worker {
  return new Worker(new URL("./worker.ts", import.meta.url), { type: "module", name: "kiln" });
}
