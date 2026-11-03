import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  worker: { format: "es" },
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        chat: resolve(__dirname, "index.html"),
        kernels: resolve(__dirname, "kernels.html"),
        bench: resolve(__dirname, "bench.html"),
      },
    },
  },
  esbuild: { target: "es2022" },
  optimizeDeps: { esbuildOptions: { target: "es2022" } },
});
