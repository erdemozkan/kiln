import { readdirSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";

// Cross-origin isolation (SharedArrayBuffer), so Kiln's CPU fallback can split matmuls across threads.
// Static hosts that cannot send these headers get them from public/coi-sw.js.
const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };

const pages = Object.fromEntries([
  ...["index", "docs", "migrate", "benchmarks", "examples"].map((p) => [p, resolve(__dirname, `${p}.html`)]),
  ...readdirSync(resolve(__dirname, "examples"))
    .filter((f) => f.endsWith(".html"))
    .map((f) => [`examples/${f.slice(0, -5)}`, resolve(__dirname, "examples", f)]),
]);

export default defineConfig({
  // Relative URLs, so the site works from any path (a GitHub Pages project site, say).
  base: "./",
  server: { headers: isolation },
  preview: { headers: isolation },
  build: { target: "es2022", rollupOptions: { input: pages } },
  esbuild: { target: "es2022" },
});
