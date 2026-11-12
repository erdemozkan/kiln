# Releasing

Steps for the first public release. Everything up to step 3 is checked in CI on every push.

1. **Check locally.** `npm ci`, `npm run typecheck`, `npm test`, `npm run build:package`,
   `npm run build:playground`, `npm run build:site`. Open `examples/playground` with `npm run dev` and run
   `/kernels.html`: every row should pass.
2. **Check the tarball.** `npm pack -w kiln-webgpu`, install the `.tgz` in a fresh Vite app, and load
   `HuggingFaceTB/SmolLM2-135M-Instruct` with `load(...)` and `chat(...)`.
3. **Set the version.** Edit `packages/kiln-webgpu/package.json` and the heading in `CHANGELOG.md`
   (drop "unreleased", add the date). Commit.
4. **Publish to npm.** `npm login`, then `npm publish -w kiln-webgpu`. `prepack` builds the package
   first. Then tag the commit: `git tag kiln-webgpu-v0.2.0 && git push --tags`.
5. **Make the repository public** (GitHub: Settings → General → Danger Zone → Change visibility).
6. **Publish the site.** Settings → Pages → Source: GitHub Actions. Then Actions → "Deploy demo" →
   Run workflow. The site (docs, examples, migration guide, benchmarks; `site/`) is served from
   `https://<user>.github.io/kiln/`, the playground from `…/kiln/playground/`; both builds use relative
   paths, so the subpath works as is. Check it locally first with `npm run dev:site`. Pages cannot send COOP/COEP headers, so a small service worker
   (`public/coi-sw.js`, registered by `public/coi.js`) adds them and the page reloads once: the demo is
   then cross-origin isolated and its CPU fallback (no WebGPU) splits matmuls across threads.

The demo deploy only runs when started by hand, so nothing is published by a push.
