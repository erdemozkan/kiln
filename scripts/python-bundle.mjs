// Copies Kiln's built index.js into the Python package, which ships it (run after build:package).
import { copyFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";

const from = fileURLToPath(new URL("../packages/kiln-webgpu/dist/index.js", import.meta.url));
const to = fileURLToPath(new URL("../python/kiln_webgpu/_js/index.js", import.meta.url));
copyFileSync(from, to);
console.log(`python/kiln_webgpu/_js/index.js: ${(statSync(to).size / 1024).toFixed(0)} kB`);
