// Makes the emitted declarations self-contained: the workspace packages (@kiln/*) are bundled into
// the JavaScript, so their type imports become relative paths into dist/types, and extensionless
// relative imports get the explicit ".js" that Node16-style resolution needs.
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "dist", "types");

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p);
    else if (p.endsWith(".d.ts")) yield p;
  }
}

const rel = (from, to) => {
  const r = relative(dirname(from), to).split("\\").join("/");
  return r.startsWith(".") ? r : `./${r}`;
};

let files = 0;
for (const file of walk(root)) {
  const src = readFileSync(file, "utf8");
  const out = src.replace(/(from\s+|import\(\s*)(["'])([^"']+)\2/g, (m, lead, q, spec) => {
    let target;
    if (spec.startsWith("@kiln/")) {
      const pkg = spec.slice("@kiln/".length);
      if (!existsSync(join(root, pkg, "src", "index.d.ts"))) throw new Error(`${file}: no declarations for ${spec}`);
      target = rel(file, join(root, pkg, "src", "index.js"));
    } else if (spec.startsWith(".") && !/\.(js|json)$/.test(spec)) {
      const base = join(dirname(file), spec);
      if (existsSync(`${base}.d.ts`)) target = `${spec}.js`;
      else if (existsSync(join(base, "index.d.ts"))) target = `${spec}/index.js`;
      else throw new Error(`${file}: cannot resolve ${spec}`);
    } else return m;
    return `${lead}${q}${target}${q}`;
  });
  if (out !== src) writeFileSync(file, out);
  files++;
}
console.log(`fix-dts: ${files} declaration files`);
