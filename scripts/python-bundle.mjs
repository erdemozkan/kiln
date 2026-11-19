// Copies Kiln's built index.js and the licence into the Python package, which ships them (run after build:package).
import { copyFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";

const from = fileURLToPath(new URL("../packages/kiln-webgpu/dist/index.js", import.meta.url));
const to = fileURLToPath(new URL("../python/kiln_webgpu/_js/index.js", import.meta.url));
copyFileSync(from, to);
copyFileSync(fileURLToPath(new URL("../LICENSE", import.meta.url)), fileURLToPath(new URL("../python/LICENSE", import.meta.url)));
console.log(`python/kiln_webgpu/_js/index.js: ${(statSync(to).size / 1024).toFixed(0)} kB`);
