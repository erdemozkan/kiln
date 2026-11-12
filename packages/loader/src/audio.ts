// Audio front end for speech models: Whisper's log-mel features (as transformers'
// WhisperFeatureExtractor computes them), a mixed-radix FFT, and WAV decoding.

/** In-place complex FFT of any length whose factors are 2, 3, 4 and 5 (re, im split arrays). */
export class FFT {
  private readonly factors: number[] = [];
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  private readonly tmpRe: Float64Array;
  private readonly tmpIm: Float64Array;

  constructor(readonly n: number) {
    let m = n;
    for (const f of [4, 2, 3, 5]) {
      while (m % f === 0) {
        this.factors.push(f);
        m /= f;
      }
    }
    if (m !== 1) throw new Error(`FFT: length ${n} has prime factors other than 2, 3 and 5`);
    this.cos = new Float64Array(n);
    this.sin = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      this.cos[k] = Math.cos((-2 * Math.PI * k) / n);
      this.sin[k] = Math.sin((-2 * Math.PI * k) / n);
    }
    this.tmpRe = new Float64Array(n);
    this.tmpIm = new Float64Array(n);
  }

  /** Forward transform of (re, im), in place. */
  transform(re: Float64Array, im: Float64Array): void {
    this.rec(re, im, 0, 1, this.n, 0, this.tmpRe, this.tmpIm);
    re.set(this.tmpRe);
    im.set(this.tmpIm);
  }

  /**
   * Decimation in time: the input of length `len`, read from `off` with `stride`, transforms into
   * out[outOff…outOff + len). Each factor f splits it into f interleaved sub-sequences.
   */
  private rec(re: Float64Array, im: Float64Array, off: number, stride: number, len: number, depth: number, outRe: Float64Array, outIm: Float64Array, outOff = 0): void {
    if (len === 1) {
      outRe[outOff] = re[off]!;
      outIm[outOff] = im[off]!;
      return;
    }
    const f = this.factors[depth]!;
    const m = len / f;
    // Sub-transforms into consecutive blocks of the output.
    for (let j = 0; j < f; j++) this.rec(re, im, off + j * stride, stride * f, m, depth + 1, outRe, outIm, outOff + j * m);
    // Butterflies: X[k + q·m] = Σ_j W_len^{j(k + q·m)} · Y_j[k].
    const step = this.n / len;
    const yr = new Float64Array(f);
    const yi = new Float64Array(f);
    for (let k = 0; k < m; k++) {
      for (let j = 0; j < f; j++) {
        yr[j] = outRe[outOff + j * m + k]!;
        yi[j] = outIm[outOff + j * m + k]!;
      }
      for (let q = 0; q < f; q++) {
        let sr = 0;
        let si = 0;
        for (let j = 0; j < f; j++) {
          const t = ((j * (k + q * m)) % len) * step;
          const c = this.cos[t]!;
          const s = this.sin[t]!;
          sr += yr[j]! * c - yi[j]! * s;
          si += yr[j]! * s + yi[j]! * c;
        }
        outRe[outOff + q * m + k] = sr;
        outIm[outOff + q * m + k] = si;
      }
    }
  }
}

const hzToMel = (f: number) => (f < 1000 ? (3 * f) / 200 : 15 + (27 * Math.log(f / 1000)) / Math.log(6.4));
const melToHz = (m: number) => (m < 15 ? (200 * m) / 3 : 1000 * Math.exp((Math.log(6.4) * (m - 15)) / 27));

/**
 * Slaney-style mel filters [bins, mels] (librosa's default, Whisper's): triangles between mel-spaced
 * points from 0 Hz to the Nyquist frequency, each scaled by 2 / its width in Hz.
 */
export function melFilters(bins: number, mels: number, sampleRate: number, fmin = 0, fmax = sampleRate / 2): Float64Array {
  const pts = Array.from({ length: mels + 2 }, (_, i) => melToHz(hzToMel(fmin) + ((hzToMel(fmax) - hzToMel(fmin)) * i) / (mels + 1)));
  const out = new Float64Array(bins * mels);
  for (let b = 0; b < bins; b++) {
    const f = (b * (sampleRate / 2)) / (bins - 1);
    for (let m = 0; m < mels; m++) {
      const lower = (f - pts[m]!) / (pts[m + 1]! - pts[m]!);
      const upper = (pts[m + 2]! - f) / (pts[m + 2]! - pts[m + 1]!);
      out[b * mels + m] = Math.max(0, Math.min(lower, upper)) * (2 / (pts[m + 2]! - pts[m]!));
    }
  }
  return out;
}

export interface WhisperFeatureConfig {
  /** Mel bands (80; 128 for large-v3). */
  mels: number;
  sampleRate?: number;
  nFft?: number;
  hop?: number;
  /** Seconds per window (30): samples are padded or cut to this length. */
  chunkSeconds?: number;
}

const fftCache = new Map<number, FFT>();
const filterCache = new Map<string, Float64Array>();

/**
 * Whisper's log-mel spectrogram of up to 30 s of 16 kHz audio, time-major [frames, mels] (3000 × 80):
 * zero-padded to the window, reflect-padded by nFft / 2, periodic Hann window, |STFT|², mel filters,
 * log10 (floored at 1e-10), the last frame dropped, clamped to 8 below the maximum, then (x + 4) / 4.
 */
export function whisperFeatures(samples: Float32Array, cfg: WhisperFeatureConfig): Float32Array {
  const sr = cfg.sampleRate ?? 16000;
  const nFft = cfg.nFft ?? 400;
  const hop = cfg.hop ?? 160;
  const total = (cfg.chunkSeconds ?? 30) * sr;
  const mels = cfg.mels;
  const x = new Float64Array(total);
  x.set(samples.subarray(0, total));
  const half = nFft / 2;
  // Reflect padding: x[-i] = x[i], x[n - 1 + i] = x[n - 1 - i].
  const at = (i: number) => (i < 0 ? x[-i]! : i >= total ? x[2 * (total - 1) - i]! : x[i]!);
  const frames = Math.floor(total / hop); // 3001 frames from center padding, minus the last one
  const bins = half + 1;
  let fft = fftCache.get(nFft);
  if (!fft) fftCache.set(nFft, (fft = new FFT(nFft)));
  const key = `${bins}/${mels}/${sr}`;
  let filters = filterCache.get(key);
  if (!filters) filterCache.set(key, (filters = melFilters(bins, mels, sr)));
  const window = Float64Array.from({ length: nFft }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft));
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  const power = new Float64Array(bins);
  const out = new Float32Array(frames * mels);
  let max = -Infinity;
  for (let t = 0; t < frames; t++) {
    const start = t * hop - half;
    for (let i = 0; i < nFft; i++) {
      re[i] = at(start + i) * window[i]!;
      im[i] = 0;
    }
    fft.transform(re, im);
    for (let b = 0; b < bins; b++) power[b] = re[b]! * re[b]! + im[b]! * im[b]!;
    for (let m = 0; m < mels; m++) {
      let s = 0;
      for (let b = 0; b < bins; b++) s += power[b]! * filters[b * mels + m]!;
      const v = Math.log10(Math.max(s, 1e-10));
      out[t * mels + m] = v;
      if (v > max) max = v;
    }
  }
  for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i]!, max - 8) + 4) / 4;
  return out;
}

/** Mono samples of a WAV file (PCM 8/16/24/32-bit or float 32), channels averaged. */
export function readWav(bytes: Uint8Array): { samples: Float32Array; sampleRate: number } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (o: number) => String.fromCharCode(bytes[o]!, bytes[o + 1]!, bytes[o + 2]!, bytes[o + 3]!);
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("not a WAV file");
  let fmt: { format: number; channels: number; rate: number; bits: number } | undefined;
  for (let o = 12; o + 8 <= bytes.length; ) {
    const id = tag(o);
    const size = dv.getUint32(o + 4, true);
    const body = o + 8;
    if (id === "fmt ") {
      let format = dv.getUint16(body, true);
      // WAVE_FORMAT_EXTENSIBLE: the real format is the subformat GUID's first two bytes.
      if (format === 0xfffe) format = dv.getUint16(body + 24, true);
      fmt = { format, channels: dv.getUint16(body + 2, true), rate: dv.getUint32(body + 4, true), bits: dv.getUint16(body + 14, true) };
    } else if (id === "data") {
      if (!fmt) throw new Error("WAV: data before fmt");
      const { format, channels, rate, bits } = fmt;
      // From the file's header: a wrong rate would size the resampled output (1 Hz → 16,000× the samples).
      if (!(rate >= 1000 && rate <= 384000) || !(channels >= 1 && channels <= 64) || ![8, 16, 24, 32].includes(bits)) {
        throw new Error(`WAV: ${channels} channels of ${bits}-bit samples at ${rate} Hz is not supported`);
      }
      const width = bits / 8;
      const n = Math.floor(Math.min(size, bytes.length - body) / (width * channels));
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let c = 0; c < channels; c++) {
          const p = body + (i * channels + c) * width;
          let v: number;
          if (format === 3 && bits === 32) v = dv.getFloat32(p, true);
          else if (format !== 1) throw new Error(`WAV: format ${format} is not supported (PCM or float only)`);
          else if (bits === 16) v = dv.getInt16(p, true) / 32768;
          else if (bits === 8) v = (dv.getUint8(p) - 128) / 128;
          else if (bits === 24) v = ((dv.getUint8(p) | (dv.getUint8(p + 1) << 8) | (dv.getInt8(p + 2) << 16)) / 8388608);
          else if (bits === 32) v = dv.getInt32(p, true) / 2147483648;
          else throw new Error(`WAV: ${bits}-bit samples are not supported`);
          sum += v;
        }
        out[i] = sum / channels;
      }
      return { samples: out, sampleRate: rate };
    }
    o = body + size + (size & 1);
  }
  throw new Error("WAV: no data chunk");
}

/** Linear-interpolation resampling (for audio not already at the model's rate). */
export function resample(samples: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return samples;
  if (!(from > 0 && to > 0)) throw new Error(`resample: invalid rates ${from} → ${to}`);
  const n = Math.floor((samples.length * to) / from);
  // 2^28 samples is over 4 hours at 16 kHz (1 GiB of float32): anything longer is a bad header or input.
  if (n > 2 ** 28) throw new Error(`resample: ${n} output samples is too many`);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = (i * from) / to;
    const j = Math.floor(p);
    const f = p - j;
    out[i] = (samples[j] ?? 0) * (1 - f) + (samples[j + 1] ?? samples[j] ?? 0) * f;
  }
  return out;
}

/** Audio as a pipeline accepts it: samples at the model's rate, a URL, a Blob / File, or encoded bytes. */
export type AudioInput = Float32Array | string | URL | Blob | ArrayBuffer | Uint8Array;

/**
 * Mono samples at `rate` (16 kHz for speech models). Samples pass through as they are; WAV files are
 * parsed here; other formats (MP3, OGG, FLAC, M4A…) are decoded and resampled by the browser's Web
 * Audio API, which exists on the page's main thread only.
 */
export async function readAudio(input: AudioInput, rate = 16000): Promise<Float32Array> {
  if (input instanceof Float32Array) return input;
  let bytes: Uint8Array;
  if (typeof input === "string" || input instanceof URL) {
    const res = await fetch(input);
    if (!res.ok) throw new Error(`${String(input)}: HTTP ${res.status}`);
    bytes = new Uint8Array(await res.arrayBuffer());
  } else if (input instanceof Uint8Array) bytes = input;
  else if (input instanceof ArrayBuffer) bytes = new Uint8Array(input);
  else bytes = new Uint8Array(await input.arrayBuffer());
  const tag = (o: number) => String.fromCharCode(bytes[o]!, bytes[o + 1]!, bytes[o + 2]!, bytes[o + 3]!);
  if (bytes.length > 12 && tag(0) === "RIFF" && tag(8) === "WAVE") {
    const wav = readWav(bytes);
    return resample(wav.samples, wav.sampleRate, rate);
  }
  const Ctx = (globalThis as { OfflineAudioContext?: new (o: { numberOfChannels: number; length: number; sampleRate: number }) => { decodeAudioData(b: ArrayBuffer): Promise<{ numberOfChannels: number; getChannelData(c: number): Float32Array }> } }).OfflineAudioContext;
  if (!Ctx) throw new Error("decoding this audio format needs the Web Audio API (a page's main thread); pass WAV bytes or 16 kHz samples instead");
  const decoded = await new Ctx({ numberOfChannels: 1, length: 1, sampleRate: rate }).decodeAudioData(bytes.slice().buffer);
  const n = decoded.getChannelData(0).length;
  const out = new Float32Array(n);
  for (let c = 0; c < decoded.numberOfChannels; c++) {
    const ch = decoded.getChannelData(c);
    for (let i = 0; i < n; i++) out[i]! += ch[i]! / decoded.numberOfChannels;
  }
  return out;
}

/** Zero mean, unit variance (variance + 1e-7), as Wav2Vec2FeatureExtractor's do_normalize. */
export function normalizeAudio(x: Float32Array): Float32Array {
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i]!;
  mean /= x.length || 1;
  let v = 0;
  for (let i = 0; i < x.length; i++) v += (x[i]! - mean) ** 2;
  const inv = 1 / Math.sqrt(v / (x.length || 1) + 1e-7);
  return x.map((s) => (s - mean) * inv);
}
