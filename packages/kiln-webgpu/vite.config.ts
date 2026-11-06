import { resolve } from "node:path";
import { defineConfig } from "vite";

// Library build: one ES module plus the model worker as its own chunk.
export default defineConfig({
  // Relative, so the worker URL resolves next to index.js wherever the package is installed.
  base: "./",
  worker: { format: "es" },
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: true,
    lib: { entry: resolve(__dirname, "src/index.ts"), formats: ["es"], fileName: "index" },
  },
  esbuild: { target: "es2022" },
});
