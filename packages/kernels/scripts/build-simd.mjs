// Compiles src/simd.wat into src/simd-bytes.ts (the module's bytes as base64), twice: with a plain
// memory import, and with a shared one for multi-threaded use (cross-origin isolated pages). Needs
// wabt's wat2wasm: on the PATH, or set WAT2WASM to its path.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const src = fileURLToPath(new URL("../src/", import.meta.url));
const dir = mkdtempSync(join(tmpdir(), "simd-"));
const wat = readFileSync(join(src, "simd.wat"), "utf8");
const IMPORT = '(import "env" "memory" (memory 1 65536))';
if (!wat.includes(IMPORT)) throw new Error(`simd.wat must import its memory as ${IMPORT}`);

function compile(name, text, flags) {
  const inp = join(dir, `${name}.wat`);
  const out = join(dir, `${name}.wasm`);
  writeFileSync(inp, text);
  execFileSync(process.env.WAT2WASM ?? "wat2wasm", [...flags, inp, "-o", out], { stdio: "inherit" });
  return readFileSync(out).toString("base64");
}

const plain = compile("simd", wat, []);
const shared = compile("simd-shared", wat.replace(IMPORT, '(import "env" "memory" (memory 1 65536 shared))'), ["--enable-threads"]);
writeFileSync(
  join(src, "simd-bytes.ts"),
  `// Generated from simd.wat by scripts/build-simd.mjs. Do not edit.\nexport const SIMD_WASM = "${plain}";\n/** The same module importing a shared memory, for helper threads. */\nexport const SIMD_WASM_SHARED = "${shared}";\n`,
);
console.log(`simd-bytes.ts: ${plain.length} + ${shared.length} base64 chars`);
