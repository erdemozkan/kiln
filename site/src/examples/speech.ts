import type { SpeechRecognitionPipeline } from "kiln-webgpu";
import { $, backendNote, busy, pipelineProgress, status } from "./common";
// The library loads when a demo first runs, not with the page.
const kiln = () => import("kiln-webgpu");

const SAMPLE = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";
let asr: SpeechRecognitionPipeline | undefined;
let audio: Blob | string = SAMPLE;
backendNote();
$<HTMLAudioElement>("player").src = SAMPLE;
$<HTMLInputElement>("file").addEventListener("change", (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) {
    audio = f;
    const player = $<HTMLAudioElement>("player");
    if (player.src.startsWith("blob:")) URL.revokeObjectURL(player.src);
    player.src = URL.createObjectURL(f);
  }
});

$("run").addEventListener("click", () =>
  busy($<HTMLButtonElement>("run"), async () => {
    status("Loading Whisper…");
    asr ??= await (await kiln()).pipeline("automatic-speech-recognition", "openai/whisper-tiny.en", { progress_callback: pipelineProgress("Loading Whisper") });
    status("Transcribing…");
    const t0 = performance.now();
    const out = (await asr(audio, { return_timestamps: "word" })) as { text: string; chunks?: { text: string; timestamp: [number, number] }[] };
    $("output").textContent = out.text.trim();
    $("words").textContent = (out.chunks ?? []).map((c) => `${c.timestamp[0].toFixed(2).padStart(6)}–${c.timestamp[1].toFixed(2).padEnd(6)} ${c.text.trim()}`).join("\n");
    status(`${((performance.now() - t0) / 1000).toFixed(1)} s, with word timestamps`);
  }),
);
