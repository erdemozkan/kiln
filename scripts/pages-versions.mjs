// Puts every released version of the library on the Pages site, beside the latest build:
//   <out>/<version>/kiln-webgpu.js   and   <out>/versions.json
// A Pages deploy replaces the whole site, so the old versions cannot simply be left in place: each
// deploy fetches them again from npm, where a published version never changes. Each tarball is
// checked against the registry's sha512 before anything is taken from it.
//
//   node scripts/pages-versions.mjs site/dist
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, lstatSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const PACKAGE = "kiln-webgpu";
const out = process.argv[2];
if (!out) throw new Error("usage: node scripts/pages-versions.mjs <output folder>");

const res = await fetch(`https://registry.npmjs.org/${PACKAGE}`);
if (!res.ok) throw new Error(`npm registry: HTTP ${res.status}`);
const meta = await res.json();
// Releases only: "1.2.3", not placeholders or pre-releases ("0.0.0-stage").
const versions = Object.keys(meta.versions)
  .filter((v) => /^\d+\.\d+\.\d+$/.test(v))
  .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
if (!versions.length) throw new Error(`${PACKAGE} has no released versions on npm`);

const work = mkdtempSync(join(tmpdir(), "kiln-versions-"));
try {
  for (const version of versions) {
    const { tarball, integrity } = meta.versions[version].dist;
    const tgz = new Uint8Array(await (await fetch(tarball)).arrayBuffer());
    const [algorithm, expected] = integrity.split("-");
    if (algorithm !== "sha512" || createHash("sha512").update(tgz).digest("base64") !== expected) {
      throw new Error(`${PACKAGE}@${version}: the tarball does not match the registry's ${algorithm} checksum`);
    }
    const dir = join(work, version);
    mkdirSync(dir);
    writeFileSync(join(dir, "package.tgz"), tgz);
    // One named member: nothing else in the archive is written to disk.
    execFileSync("tar", ["-xzf", join(dir, "package.tgz"), "-C", dir, "package/dist/index.js"]);
    mkdirSync(join(out, version), { recursive: true });
    // A plain file only: a tarball could hold a link there, and copying would follow it.
    const built = join(dir, "package", "dist", "index.js");
    if (!lstatSync(built).isFile() || !lstatSync(join(dir, "package", "dist")).isDirectory() || !lstatSync(join(dir, "package")).isDirectory()) {
      throw new Error(`${PACKAGE}@${version}: dist/index.js is not a plain file`);
    }
    copyFileSync(built, join(out, version, `${PACKAGE}.js`));
    console.log(`${version}: ${out}/${version}/${PACKAGE}.js`);
  }
  writeFileSync(join(out, "versions.json"), `${JSON.stringify({ latest: meta["dist-tags"]?.latest, versions }, null, 2)}\n`);
} finally {
  rmSync(work, { recursive: true, force: true });
}
