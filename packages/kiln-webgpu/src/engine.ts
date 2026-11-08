// kiln-webgpu/engine: the in-process engine, for code that already runs in a worker (or for tests).
// Kept out of the main entry, whose model code lives in the embedded worker.
export { createEngine, Engine } from "@kiln/runtime";
export type { LoadProgress, GenerateStats } from "@kiln/runtime";
