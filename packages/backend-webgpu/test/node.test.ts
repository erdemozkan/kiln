import { describe, expect, it } from "vitest";
import { createWebGPUBackend } from "../src/index";

describe("WebGPU in Node.js", () => {
  it("says which package to install when Dawn's bindings are missing", async () => {
    // This repository does not depend on the webgpu package, so Node has no GPU here.
    await expect(createWebGPUBackend()).rejects.toThrow(/npm install webgpu/);
  });
});
