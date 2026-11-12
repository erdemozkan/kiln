import { resolve } from "node:path";
import { transform } from "esbuild";
import { defineConfig, type Plugin } from "vite";

// Library builds keep whitespace in ES output, and the embedded worker is a string the browser parses
// on every page load: minify it fully.
const minifyWorker = (): Plugin => ({
  name: "kiln-minify-worker",
  // After Vite's own renderChunk pass, which would print the code with whitespace again.
  async generateBundle(_opts, bundle) {
    for (const chunk of Object.values(bundle)) {
      if (chunk.type === "chunk") chunk.code = (await transform(chunk.code, { minify: true, target: "es2022", format: "esm" })).code;
    }
  },
});

// Library build of one entry: the main one (index.js, the model worker embedded, see
// src/inline-worker.ts), or with KILN_ENTRY=engine the in-process engine (engine.js). Two builds, so
// neither file imports a chunk shared with the other.
const entry = process.env.KILN_ENTRY === "engine" ? "engine" : "index";
export default defineConfig({
  base: "./",
  worker: { format: "es", plugins: () => [minifyWorker()] },
  resolve: { alias: [{ find: /^\.\/default-worker$/, replacement: resolve(__dirname, "src/inline-worker.ts") }] },
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: entry === "index",
    lib: { entry: resolve(__dirname, `src/${entry}.ts`), formats: ["es"], fileName: entry },
  },
  esbuild: { target: "es2022" },
});
