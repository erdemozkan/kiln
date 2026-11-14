import { resample } from "@kiln/loader";
import type { SpeechChunk, SpeechRecognitionOptions } from "./pipeline";

/** What a streaming transcription needs of a recogniser: a Whisper `pipeline("automatic-speech-recognition")` fits. */
export type StreamingRecognizer = (audio: Float32Array, options?: SpeechRecognitionOptions) => Promise<{ text: string; chunks?: SpeechChunk[] } | { text: string; chunks?: SpeechChunk[] }[]>;

export interface StreamingUpdate {
  /** Text that will not change any more. */
  committed: string;
  /** The words after it as the last pass heard them; the next pass may revise them. */
  tentative: string;
  /** Both, as one string. */
  text: string;
}

export interface StreamingOptions {
  /** Seconds of new audio between passes (default 1). Each pass transcribes the whole window again, so shorter costs more. */
  stepSeconds?: number;
  /** Longest stretch of audio transcribed at once (default 20, at most 28): past it, audio already committed is dropped. */
  maxWindowSeconds?: number;
  /** Passed to the recogniser (multilingual Whisper models). */
  language?: string;
  task?: "transcribe" | "translate";
  /** Called after every pass that changed the text. */
  onUpdate?: (update: StreamingUpdate) => void;
  /** Called if a pass fails; `finish()` rejects with the same error. */
  onError?: (error: unknown) => void;
}

interface Word {
  text: string;
  start: number;
  end: number;
}

const RATE = 16000;
/** Letters and digits of a word, lower case: what two passes must agree on. */
const key = (w: string): string => w.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");

/**
 * Live transcription of audio that arrives in pieces (a microphone, a call). Whisper hears fixed
 * windows, so the audio since the last settled word is transcribed again on every pass, with word
 * times, and a word is committed once two passes in a row agree on it (the scheme of
 * "whisper-streaming", LocalAgreement-2): the words still being spoken at the end of a window, which
 * the next pass often revises, stay tentative. When the window grows past `maxWindowSeconds`, the
 * audio before the last committed word is dropped.
 */
export class StreamingTranscriber {
  /** Audio not yet settled, 16 kHz, starting `offset` seconds into the stream. */
  private buffer = new Float32Array(0);
  private offset = 0;
  /** Pieces fed since the last pass, at their own rate. */
  private pending: { samples: Float32Array; rate: number }[] = [];
  private pendingSeconds = 0;
  private readonly words: Word[] = [];
  private previous: Word[] = [];
  private running: Promise<void> | undefined;
  private failed: unknown;
  private readonly step: number;
  private readonly window: number;
  private last = "";

  constructor(
    private readonly recognize: StreamingRecognizer,
    private readonly options: StreamingOptions = {},
  ) {
    this.step = Math.max(0.2, options.stepSeconds ?? 1);
    this.window = Math.min(28, Math.max(2 * this.step, options.maxWindowSeconds ?? 20));
  }

  /** The text committed so far. */
  get committed(): string {
    return this.words.map((w) => w.text).join("").trim();
  }

  /** The words after the committed text, as the last pass heard them. */
  get tentative(): string {
    return this.previous.map((w) => w.text).join("").trim();
  }

  /** Adds audio (mono samples in −1…1 at `sampleRate`). A pass starts once `stepSeconds` of new audio is in and none is running. */
  feed(samples: Float32Array, sampleRate = RATE): void {
    if (!samples.length) return;
    this.pending.push({ samples: samples.slice(), rate: sampleRate });
    this.pendingSeconds += samples.length / sampleRate;
    this.pump();
  }

  /** Transcribes what is left and commits it all; returns the whole text. */
  async finish(): Promise<string> {
    await this.running;
    if (this.failed !== undefined) throw this.failed;
    await this.pass(true);
    return this.committed;
  }

  private pump(): void {
    if (this.running || this.failed !== undefined || this.pendingSeconds < this.step) return;
    this.running = this.pass(false)
      .catch((err: unknown) => {
        this.failed = err;
        this.options.onError?.(err);
      })
      .finally(() => {
        this.running = undefined;
        this.pump();
      });
  }

  /** Moves the fed pieces into the 16 kHz buffer. Pieces at one rate are resampled together, so only pass edges are seams. */
  private take(): void {
    const pieces = this.pending;
    this.pending = [];
    this.pendingSeconds = 0;
    const parts: Float32Array[] = [this.buffer];
    for (let i = 0; i < pieces.length; ) {
      let j = i;
      let n = 0;
      while (j < pieces.length && pieces[j]!.rate === pieces[i]!.rate) n += pieces[j++]!.samples.length;
      const run = new Float32Array(n);
      let at = 0;
      for (let k = i; k < j; k++) {
        run.set(pieces[k]!.samples, at);
        at += pieces[k]!.samples.length;
      }
      parts.push(pieces[i]!.rate === RATE ? run : resample(run, pieces[i]!.rate, RATE));
      i = j;
    }
    const total = parts.reduce((s, p) => s + p.length, 0);
    const out = new Float32Array(total);
    parts.reduce((o, p) => (out.set(p, o), o + p.length), 0);
    this.buffer = out;
  }

  private async pass(final: boolean): Promise<void> {
    this.take();
    const audio = this.buffer;
    const offset = this.offset;
    let heard: Word[] = [];
    // Whisper invents words for silence: a window with no sound in it is not transcribed.
    if (audio.length >= RATE / 10 && audio.some((v) => Math.abs(v) > 1e-3)) {
      const got = await this.recognize(audio, { return_timestamps: "word", ...(this.options.language ? { language: this.options.language } : {}), ...(this.options.task ? { task: this.options.task } : {}) });
      const result = Array.isArray(got) ? got[0] : got;
      const end = offset + audio.length / RATE;
      heard = (result?.chunks ?? []).map((c) => ({ text: c.text, start: offset + (c.timestamp[0] ?? 0), end: offset + (c.timestamp[1] ?? c.timestamp[0] ?? 0) })).filter((w) => key(w.text) && w.start <= end);
    }
    // Only what follows the committed text: by time, then by dropping words that repeat its tail
    // (a word on the boundary is heard on both sides of it).
    const after = this.words.length ? this.words[this.words.length - 1]!.end : 0;
    let fresh = heard.filter((w) => w.end > after + 0.05 && w.start >= after - 0.3);
    for (let n = Math.min(5, fresh.length, this.words.length); n > 0; n--) {
      const tail = this.words.slice(-n).map((w) => key(w.text)).join(" ");
      if (fresh.slice(0, n).map((w) => key(w.text)).join(" ") === tail && fresh[0]!.start < after + 0.3) {
        fresh = fresh.slice(n);
        break;
      }
    }
    let agreed = 0;
    if (final) agreed = fresh.length;
    else while (agreed < fresh.length && agreed < this.previous.length && key(fresh[agreed]!.text) === key(this.previous[agreed]!.text)) agreed++;
    this.words.push(...fresh.slice(0, agreed));
    this.previous = fresh.slice(agreed);

    // Keep the window short: drop the audio the committed words cover, or, with nothing committed in
    // it (noise, music), all but the newest half.
    const seconds = this.buffer.length / RATE;
    if (final) {
      this.buffer = new Float32Array(0);
      this.offset += seconds;
    } else if (seconds > this.window) {
      const settled = this.words.length ? this.words[this.words.length - 1]!.end : 0;
      const cut = settled > this.offset + 0.5 ? settled : this.offset + seconds - this.window / 2;
      const drop = Math.min(this.buffer.length, Math.max(0, Math.round((cut - this.offset) * RATE)));
      this.buffer = this.buffer.slice(drop);
      this.offset += drop / RATE;
      // Words heard before the cut cannot be confirmed again.
      this.previous = this.previous.filter((w) => w.start >= this.offset);
    }
    const committed = this.committed;
    const tentative = this.tentative;
    const text = tentative ? `${committed} ${tentative}`.trim() : committed;
    if (text !== this.last) {
      this.last = text;
      this.options.onUpdate?.({ committed, tentative, text });
    }
  }
}

export interface MicrophoneOptions extends StreamingOptions {
  /** A stream to listen to (a call's remote audio, say); by default the microphone is asked for. */
  stream?: MediaStream;
}

export interface MicrophoneTranscription {
  readonly transcriber: StreamingTranscriber;
  /** Stops listening (and releases the microphone if this call opened it); resolves with the whole text. */
  stop(): Promise<string>;
}

const CAPTURE = `class KilnCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) this.port.postMessage(channel.slice());
    return true;
  }
}
registerProcessor("kiln-capture", KilnCapture);`;

/**
 * Transcribes the microphone (or `options.stream`) live: `onUpdate` gets the text as it settles.
 * The browser asks the user for the microphone; the audio stays on the device.
 *
 *   const asr = await pipeline("automatic-speech-recognition", "openai/whisper-base.en");
 *   const mic = await transcribeMicrophone(asr, { onUpdate: (u) => (caption.textContent = u.text) });
 *   // later
 *   const text = await mic.stop();
 */
export async function transcribeMicrophone(recognize: StreamingRecognizer, options: MicrophoneOptions = {}): Promise<MicrophoneTranscription> {
  if (typeof AudioContext === "undefined" || typeof AudioWorkletNode === "undefined") throw new Error("live transcription needs AudioContext and AudioWorklet (a browser page)");
  const own = !options.stream;
  const stream = options.stream ?? (await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } }));
  const stopTracks = () => {
    if (own) for (const t of stream.getTracks()) t.stop();
  };
  let ctx: AudioContext | undefined;
  try {
    ctx = new AudioContext();
    const url = URL.createObjectURL(new Blob([CAPTURE], { type: "text/javascript" }));
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    const transcriber = new StreamingTranscriber(recognize, options);
    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "kiln-capture", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: "explicit" });
    const rate = ctx.sampleRate;
    node.port.onmessage = (e: MessageEvent<Float32Array>) => transcriber.feed(e.data, rate);
    // The graph only runs nodes that lead to the output: a silent path keeps the capture node pulled.
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(node).connect(mute).connect(ctx.destination);
    if (ctx.state === "suspended") await ctx.resume();
    const context = ctx;
    return {
      transcriber,
      async stop() {
        node.port.onmessage = null;
        source.disconnect();
        node.disconnect();
        stopTracks();
        await context.close();
        return transcriber.finish();
      },
    };
  } catch (err) {
    stopTracks();
    await ctx?.close().catch(() => undefined);
    throw err;
  }
}
