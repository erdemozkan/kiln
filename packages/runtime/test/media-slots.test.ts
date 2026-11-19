import { describe, expect, it, vi } from "vitest";
import { CpuBackend } from "@kiln/kernels";
import { LlamaModel, type Gemma4AudioTower, type ImagePixels, type VisionTower } from "@kiln/models";
import type { ChatFormat, ChatMessage } from "@kiln/tokenizers";
import { Engine } from "../src/engine";
import { TINY, tinyTokenizer, tinyWeights } from "./tiny-model";

const BOI = 90;
const image = (v: number): ImagePixels => ({ width: 2, height: 2, data: new Uint8Array(12).fill(v) });
const asks = (images: ImagePixels[]): ChatMessage[] => [{ role: "user", content: [...images.map((img) => ({ type: "image" as const, image: img })), { type: "text" as const, text: "a" }] }];

/** A Gemma 3-style vision tower of 4 tokens an image, whose encode the test controls. */
function tower() {
  const t = { tokens: 4, bidirectional: true, fail: false, gate: undefined as Promise<void> | undefined, encode: vi.fn(), destroy: vi.fn() };
  t.encode.mockImplementation(async () => {
    await t.gate;
    if (t.fail) throw new Error("encode failed");
  });
  return t;
}

function withVision(slots = 4) {
  const be = new CpuBackend({ threads: 1 });
  const v = tower();
  const model = new LlamaModel(be, TINY, tinyWeights(be), { maxSeq: 64, kvDtype: "f32", imageTokens: slots * v.tokens });
  const e = new Engine(be, model, tinyTokenizer({ "<start_of_image>": BOI }), v as unknown as VisionTower);
  // A template that writes the marker for each image part.
  const format: ChatFormat = { name: "test", stop: [], render: (ms) => ms.map((m) => (typeof m.content === "string" ? m.content : m.content.map((p) => (p.type === "image" ? "<start_of_image>" : p.type === "text" ? p.text : "")).join(""))).join("") };
  Object.defineProperty(e, "chatFormat", { value: format });
  return { be, e, v, priv: e as unknown as { busy: boolean; imageSlot(img: ImagePixels): Promise<number>; imageSlots: unknown[] } };
}

async function drain(it: AsyncIterable<unknown>) {
  for await (const _ of it);
}

describe("image feature slots", () => {
  it("refuse more distinct images than fit, before encoding any", async () => {
    const { e, v, priv } = withVision(4);
    await expect(drain(e.chat(asks([1, 2, 3, 4, 5].map(image)), { maxTokens: 1 }))).rejects.toThrow(/5 different images, but 4 fit; load with a larger imageSlots/);
    expect(v.encode).not.toHaveBeenCalled();
    expect(priv.busy).toBe(false);
    // The same picture twice takes one slot.
    v.fail = true;
    await expect(drain(e.chat(asks([1, 2, 3, 4, 1].map(image)), { maxTokens: 1 }))).rejects.toThrow(/encode failed/);
    expect(v.encode).toHaveBeenCalledTimes(1);
  });

  it("encode an image again after an encode that failed", async () => {
    const { v, priv } = withVision(1);
    v.fail = true;
    await expect(priv.imageSlot(image(1))).rejects.toThrow(/encode failed/);
    v.fail = false;
    expect(await priv.imageSlot(image(1))).toBe(0);
    expect(v.encode).toHaveBeenCalledTimes(2);
    expect(await priv.imageSlot(image(2))).toBe(0);
    expect(v.encode).toHaveBeenCalledTimes(3);
  });

  it("make a second call fail while the first is still encoding", async () => {
    const { e, v, priv } = withVision();
    let open!: () => void;
    v.gate = new Promise((r) => (open = r));
    v.fail = true;
    const first = drain(e.chat(asks([image(1)]), { maxTokens: 1 }));
    await vi.waitFor(() => expect(v.encode).toHaveBeenCalled());
    await expect(drain(e.chat(asks([image(2)]), { maxTokens: 1 }))).rejects.toThrow(/already generating/);
    await expect(drain(e.chat([{ role: "user", content: "a" }], { maxTokens: 1 }))).rejects.toThrow(/already generating/);
    open();
    await expect(first).rejects.toThrow(/encode failed/);
    expect(priv.busy).toBe(false);
    expect(v.encode).toHaveBeenCalledTimes(1);
  });

  it("go with the engine, and so does a backend the engine owns", () => {
    const { be, e, v } = withVision();
    const destroy = vi.spyOn(be, "destroy");
    e.destroy();
    expect(v.destroy).toHaveBeenCalled();
    expect(destroy).not.toHaveBeenCalled();
    const o = withVision();
    o.e.ownsBackend = true;
    const owned = vi.spyOn(o.be, "destroy");
    o.e.destroy();
    expect(owned).toHaveBeenCalled();
  });
});

describe("audio feature slots", () => {
  it("stay usable after an encode that failed", async () => {
    const be = new CpuBackend({ threads: 1 });
    let fail = true;
    const t = {
      tokens: 4,
      tokensFor: () => 4,
      encode: vi.fn(async () => {
        if (fail) throw new Error("encode failed");
        return 4;
      }),
      destroy: () => {},
    };
    const model = new LlamaModel(be, TINY, tinyWeights(be), { maxSeq: 64, kvDtype: "f32", imageTokens: t.tokens });
    const e = new Engine(be, model, tinyTokenizer(), undefined, { tower: t as unknown as Gemma4AudioTower, slots: 1 });
    const slot = (e as unknown as { audioSlot(s: Float32Array): Promise<{ base: number; count: number }> }).audioSlot.bind(e);
    await expect(slot(new Float32Array(16000).fill(0.1))).rejects.toThrow(/encode failed/);
    fail = false;
    expect(await slot(new Float32Array(16000).fill(0.2))).toEqual({ base: TINY.vocabSize, count: 4 });
    expect(await slot(new Float32Array(16000).fill(0.1))).toEqual({ base: TINY.vocabSize, count: 4 });
    expect(t.encode).toHaveBeenCalledTimes(3);
  });
});
