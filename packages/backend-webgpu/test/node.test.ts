import { describe, expect, it } from "vitest";
import { createWebGPUBackend, nodeGPUBlocked } from "../src/index";

describe("WebGPU in Node.js", () => {
  it("fails with a message, never a GPU, without Dawn's bindings or on an old macOS", async () => {
    // This repository does not depend on the webgpu package, so Node has no GPU here.
    await expect(createWebGPUBackend()).rejects.toThrow(/npm install webgpu|macOS 15/);
  });

  it("is off on macOS before 15 only", () => {
    expect(nodeGPUBlocked("darwin", "22.6.0")).toMatch(/macOS 15/);
    expect(nodeGPUBlocked("darwin", "23.4.0")).toMatch(/macOS 15/);
    expect(nodeGPUBlocked("darwin", "24.0.0")).toBeUndefined();
    expect(nodeGPUBlocked("darwin", "25.1.0")).toBeUndefined();
    expect(nodeGPUBlocked("linux", "6.8.0")).toBeUndefined();
    expect(nodeGPUBlocked("win32", "10.0.22631")).toBeUndefined();
  });
});
