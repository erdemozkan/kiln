import { resolve } from "node:path";
import { defineConfig } from "vite";

// Library build of one entry: the main one (index.js, the model worker embedded, see
// src/inline-worker.ts), or with KILN_ENTRY=engine the in-process engine (engine.js). Two builds, so
// neither file imports a chunk shared with the other.
const entry = process.env.KILN_ENTRY === "engine" ? "engine" : "index";
export default defineConfig({
  base: "./",
  worker: { format: "es" },
  resolve: { alias: [{ find: /^\.\/default-worker$/, replacement: resolve(__dirname, "src/inline-worker.ts") }] },
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: entry === "index",
    lib: { entry: resolve(__dirname, `src/${entry}.ts`), formats: ["es"], fileName: entry },
  },
  esbuild: { target: "es2022" },
});
