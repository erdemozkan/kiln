import { describe, expect, it } from "vitest";
import { StreamingTranscriber, type StreamingRecognizer, type StreamingUpdate } from "../src/streaming-asr";

const RATE = 16000;
/** A sentence spoken at 2.5 words a second from t = 0.4 s; every sample holds its own time, so a fake recogniser knows where its window lies. */
const SENTENCE = "The harbour town of Velden sits where the river Arn meets the sea and its lighthouse stands forty one metres tall above the red and white rocks of the northern shore".split(" ");
const TRUTH = SENTENCE.map((text, i) => ({ text: ` ${text}`, start: 0.4 + i * 0.4, end: 0.4 + i * 0.4 + 0.3 }));
const stream = (seconds: number) => Float32Array.from({ length: Math.round(seconds * RATE) }, (_, i) => (i + 1) / 1e7);
const timeOf = (sample: number) => Math.round(sample * 1e7 - 1) / RATE;

/** Hears the truth inside its window; a word the window's end cuts through comes out wrong, as with a real model. */
function fake(calls: { from: number; to: number }[] = []): StreamingRecognizer {
  return async (audio) => {
    const from = timeOf(audio[0]!);
    const to = from + audio.length / RATE;
    calls.push({ from, to });
    const chunks = TRUTH.filter((w) => w.end > from + 0.05 && w.start < to).map((w) => ({
      text: w.end > to ? ` ${w.text.trim().slice(0, 2)}zz` : w.text,
      timestamp: [w.start - from, Math.min(w.end, to) - from] as [number, number],
    }));
    return { text: chunks.map((c) => c.text).join("").trim(), chunks };
  };
}

/** Feeds `audio` in pieces of `piece` seconds, letting passes run between them. */
async function play(t: StreamingTranscriber, audio: Float32Array, piece = 0.1): Promise<void> {
  const n = Math.round(piece * RATE);
  for (let i = 0; i < audio.length; i += n) {
    t.feed(audio.subarray(i, i + n));
    await new Promise((r) => setTimeout(r, 0));
  }
}

describe("streaming transcription", () => {
  it("commits words two passes agree on and ends with the whole text", async () => {
    const updates: StreamingUpdate[] = [];
    const t = new StreamingTranscriber(fake(), { stepSeconds: 1, onUpdate: (u) => updates.push(u) });
    await play(t, stream(13));
    // Mid-stream: committed text is a prefix of the truth, and never shrinks.
    expect(updates.length).toBeGreaterThan(5);
    let before = "";
    for (const u of updates) {
      expect(SENTENCE.join(" ").startsWith(u.committed)).toBe(true);
      expect(u.committed.startsWith(before)).toBe(true);
      expect(u.text).toBe(`${u.committed} ${u.tentative}`.trim());
      before = u.committed;
    }
    expect(before.split(" ").length).toBeGreaterThan(15);
    // A cut word is never committed.
    expect(updates.some((u) => u.committed.includes("zz"))).toBe(false);
    expect(await t.finish()).toBe(SENTENCE.join(" "));
    expect(t.tentative).toBe("");
  });

  it("drops settled audio once the window is long", async () => {
    const calls: { from: number; to: number }[] = [];
    const t = new StreamingTranscriber(fake(calls), { stepSeconds: 0.5, maxWindowSeconds: 4 });
    await play(t, stream(13));
    expect(await t.finish()).toBe(SENTENCE.join(" "));
    // No pass heard more than the window plus one step, and later passes start past the beginning.
    expect(Math.max(...calls.map((c) => c.to - c.from))).toBeLessThan(4 + 0.5 + 0.2);
    expect(calls[calls.length - 1]!.from).toBeGreaterThan(6);
  });

  it("resamples audio fed at another rate", async () => {
    const heard: number[] = [];
    const t = new StreamingTranscriber(async (audio) => (heard.push(audio.length), { text: "", chunks: [] }), { stepSeconds: 1 });
    const tone = Float32Array.from({ length: 48000 }, (_, i) => Math.sin(i / 20));
    t.feed(tone, 48000);
    await t.finish();
    // One second at 48 kHz is one second at 16 kHz.
    expect(Math.abs(heard[0]! - RATE)).toBeLessThan(8);
  });

  it("does not transcribe silence, and reports a failing recogniser", async () => {
    let calls = 0;
    const quiet = new StreamingTranscriber(async () => (calls++, { text: " Thank you.", chunks: [{ text: " Thank", timestamp: [0, 1] }] }), { stepSeconds: 0.5 });
    await play(quiet, new Float32Array(3 * RATE));
    expect(await quiet.finish()).toBe("");
    expect(calls).toBe(0);

    const errors: unknown[] = [];
    const broken = new StreamingTranscriber(async () => Promise.reject(new Error("device lost")), { stepSeconds: 0.5, onError: (e) => errors.push(e) });
    await play(broken, stream(2));
    expect(errors.length).toBe(1);
    await expect(broken.finish()).rejects.toThrow(/device lost/);
  });
});
