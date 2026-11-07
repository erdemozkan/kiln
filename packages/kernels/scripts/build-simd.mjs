// Compiles src/simd.wat into src/simd-bytes.ts (the module's bytes as base64). Needs wabt's
// wat2wasm: on the PATH, or set WAT2WASM to its path.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const src = fileURLToPath(new URL("../src/", import.meta.url));
const out = join(mkdtempSync(join(tmpdir(), "simd-")), "simd.wasm");
execFileSync(process.env.WAT2WASM ?? "wat2wasm", [join(src, "simd.wat"), "-o", out], { stdio: "inherit" });
const b64 = readFileSync(out).toString("base64");
writeFileSync(
  join(src, "simd-bytes.ts"),
  `// Generated from simd.wat by scripts/build-simd.mjs. Do not edit.\nexport const SIMD_WASM = "${b64}";\n`,
);
console.log(`simd-bytes.ts: ${b64.length} base64 chars`);
