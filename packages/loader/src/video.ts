// Video frames for a model that takes video (Gemma 4): decoded with the browser's <video> element.
import type { RgbImage } from "./image";

/** Anything a video can come from in a page: a URL, a Blob / File, or a <video> element. */
export type VideoInput = string | URL | Blob | HTMLVideoElement;

/**
 * Samples `frames` frames evenly over a video (default 32, as Gemma 4's video processor does), each
 * at its own size, with its time in seconds: the value of a `{ type: "video", video }` chat part.
 * Needs a page (a <video> element and a 2D canvas); in a worker, decode the frames yourself and pass
 * `{ frames, times }`.
 */
export async function readVideo(input: VideoInput, opts: { frames?: number } = {}): Promise<{ frames: RgbImage[]; times: number[] }> {
  if (typeof document === "undefined") throw new Error("readVideo needs a page (a <video> element); in a worker, pass { frames, times } yourself");
  const count = Math.max(1, Math.floor(opts.frames ?? 32));
  const own = !(typeof HTMLVideoElement !== "undefined" && input instanceof HTMLVideoElement);
  const video = own ? document.createElement("video") : (input as HTMLVideoElement);
  let url: string | undefined;
  if (own) {
    video.muted = true;
    video.preload = "auto";
    video.crossOrigin = "anonymous";
    url = input instanceof Blob ? URL.createObjectURL(input) : undefined;
    video.src = url ?? String(input);
  }
  try {
    if (video.readyState < 1) {
      await new Promise<void>((resolve, reject) => {
        video.addEventListener("loadedmetadata", () => resolve(), { once: true });
        video.addEventListener("error", () => reject(new Error(`readVideo: the video could not be loaded (${video.error?.message ?? "unknown error"})`)), { once: true });
      });
    }
    // A recorded WebM may not state its length: seeking far past the end makes the browser find it.
    if (video.duration === Infinity) await seek(video, 1e9);
    const duration = video.duration;
    if (!Number.isFinite(duration) || duration <= 0) throw new Error("readVideo: the video has no known duration");
    const [w, h] = [video.videoWidth, video.videoHeight];
    const canvas = document.createElement("canvas");
    [canvas.width, canvas.height] = [w, h];
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    const frames: RgbImage[] = [];
    const times: number[] = [];
    for (let i = 0; i < count; i++) {
      // The middle of each of `count` equal stretches, so the first and last frames are inside the video.
      const t = ((i + 0.5) * duration) / count;
      await seek(video, t);
      ctx.drawImage(video, 0, 0, w, h);
      const d = ctx.getImageData(0, 0, w, h);
      frames.push({ width: w, height: h, data: d.data });
      times.push(t);
    }
    return { frames, times };
  } finally {
    if (url) URL.revokeObjectURL(url);
    if (own) video.removeAttribute("src");
  }
}

/** Seeks a video and waits until the frame there can be drawn. */
function seek(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    video.addEventListener("seeked", () => resolve(), { once: true });
    video.addEventListener("error", () => reject(new Error("readVideo: seeking failed")), { once: true });
    video.currentTime = t;
  });
}
