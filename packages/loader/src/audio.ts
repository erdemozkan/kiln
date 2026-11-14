// Audio front end for speech models: Whisper's log-mel features (as transformers'
// WhisperFeatureExtractor computes them), a mixed-radix FFT, and WAV decoding.

// cos and sin of 2π/5 and 4π/5, for the radix-5 butterfly.
const C1 = Math.cos((2 * Math.PI) / 5);
const C2 = Math.cos((4 * Math.PI) / 5);
const S1 = Math.sin((2 * Math.PI) / 5);
const S2 = Math.sin((4 * Math.PI) / 5);

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
    if (this.n === 1) return;
    this.rec(re, im, 0, 1, this.n, 0, this.tmpRe, this.tmpIm);
    re.set(this.tmpRe);
    im.set(this.tmpIm);
  }

  /** The sub-transform outputs of one butterfly, for factors without a written-out butterfly. */
  private readonly yr = new Float64Array(5);
  private readonly yi = new Float64Array(5);

  /**
   * Decimation in time: the input of length `len`, read from `off` with `stride`, transforms into
   * out[outOff…outOff + len). Each factor f splits it into f interleaved sub-sequences. Factors 2
   * and 4, most of any length used here, have their butterflies written out (W_4 is -i, so a
   * radix-4 step needs three twiddle multiplications and no others); 3 and 5 take the general sum.
   */
  private rec(re: Float64Array, im: Float64Array, off: number, stride: number, len: number, depth: number, outRe: Float64Array, outIm: Float64Array, outOff = 0): void {
    const f = this.factors[depth]!;
    const m = len / f;
    // Sub-transforms into consecutive blocks of the output (of one sample each at the last factor).
    if (m === 1) {
      for (let j = 0; j < f; j++) {
        outRe[outOff + j] = re[off + j * stride]!;
        outIm[outOff + j] = im[off + j * stride]!;
      }
    } else {
      for (let j = 0; j < f; j++) this.rec(re, im, off + j * stride, stride * f, m, depth + 1, outRe, outIm, outOff + j * m);
    }
    // Butterflies: X[k + q·m] = Σ_j W_len^{j(k + q·m)} · Y_j[k]; W_len^x is table entry x·step.
    const step = this.n / len;
    const { cos, sin, n } = this;
    if (f === 2) {
      for (let k = 0, a = outOff, b = outOff + m; k < m; k++, a++, b++) {
        const c = cos[k * step]!;
        const sn = sin[k * step]!;
        const br = outRe[b]! * c - outIm[b]! * sn;
        const bi = outRe[b]! * sn + outIm[b]! * c;
        const ar = outRe[a]!;
        const ai = outIm[a]!;
        outRe[a] = ar + br;
        outIm[a] = ai + bi;
        outRe[b] = ar - br;
        outIm[b] = ai - bi;
      }
      return;
    }
    if (f === 4) {
      for (let k = 0; k < m; k++) {
        const [i0, i1, i2, i3] = [outOff + k, outOff + m + k, outOff + 2 * m + k, outOff + 3 * m + k];
        const t = k * step;
        const br = outRe[i1]! * cos[t]! - outIm[i1]! * sin[t]!;
        const bi = outRe[i1]! * sin[t]! + outIm[i1]! * cos[t]!;
        const cr = outRe[i2]! * cos[2 * t]! - outIm[i2]! * sin[2 * t]!;
        const ci = outRe[i2]! * sin[2 * t]! + outIm[i2]! * cos[2 * t]!;
        const dr = outRe[i3]! * cos[3 * t]! - outIm[i3]! * sin[3 * t]!;
        const di = outRe[i3]! * sin[3 * t]! + outIm[i3]! * cos[3 * t]!;
        const ar = outRe[i0]!;
        const ai = outIm[i0]!;
        // q = 0…3 weigh b, c, d by (-i)^q, (-1)^q and i^q.
        outRe[i0] = ar + br + cr + dr;
        outIm[i0] = ai + bi + ci + di;
        outRe[i1] = ar + bi - cr - di;
        outIm[i1] = ai - br - ci + dr;
        outRe[i2] = ar - br + cr - dr;
        outIm[i2] = ai - bi + ci - di;
        outRe[i3] = ar - bi - cr + di;
        outIm[i3] = ai + br - ci - dr;
      }
      return;
    }
    if (f === 5) {
      // The inputs are turned by their twiddles, then a 5-point transform whose roots pair up:
      // ω and ω⁴ share a cosine and differ in the sign of the sine, as do ω² and ω³.
      for (let k = 0; k < m; k++) {
        const o = outOff + k;
        const t = k * step;
        const y0r = outRe[o]!, y0i = outIm[o]!;
        const y1r = outRe[o + m]! * cos[t]! - outIm[o + m]! * sin[t]!, y1i = outRe[o + m]! * sin[t]! + outIm[o + m]! * cos[t]!;
        const y2r = outRe[o + 2 * m]! * cos[2 * t]! - outIm[o + 2 * m]! * sin[2 * t]!, y2i = outRe[o + 2 * m]! * sin[2 * t]! + outIm[o + 2 * m]! * cos[2 * t]!;
        const y3r = outRe[o + 3 * m]! * cos[3 * t]! - outIm[o + 3 * m]! * sin[3 * t]!, y3i = outRe[o + 3 * m]! * sin[3 * t]! + outIm[o + 3 * m]! * cos[3 * t]!;
        const y4r = outRe[o + 4 * m]! * cos[4 * t]! - outIm[o + 4 * m]! * sin[4 * t]!, y4i = outRe[o + 4 * m]! * sin[4 * t]! + outIm[o + 4 * m]! * cos[4 * t]!;
        const t1r = y1r + y4r, t1i = y1i + y4i, t2r = y2r + y3r, t2i = y2i + y3i;
        const t3r = y1r - y4r, t3i = y1i - y4i, t4r = y2r - y3r, t4i = y2i - y3i;
        const p1r = y0r + C1 * t1r + C2 * t2r, p1i = y0i + C1 * t1i + C2 * t2i;
        const p2r = y0r + C2 * t1r + C1 * t2r, p2i = y0i + C2 * t1i + C1 * t2i;
        const ar = S1 * t3r + S2 * t4r, ai = S1 * t3i + S2 * t4i;
        const br = S2 * t3r - S1 * t4r, bi = S2 * t3i - S1 * t4i;
        outRe[o] = y0r + t1r + t2r;
        outIm[o] = y0i + t1i + t2i;
        // X1 = p1 - i·a, X4 = p1 + i·a, X2 = p2 - i·b, X3 = p2 + i·b.
        outRe[o + m] = p1r + ai;
        outIm[o + m] = p1i - ar;
        outRe[o + 2 * m] = p2r + bi;
        outIm[o + 2 * m] = p2i - br;
        outRe[o + 3 * m] = p2r - bi;
        outIm[o + 3 * m] = p2i + br;
        outRe[o + 4 * m] = p1r - ai;
        outIm[o + 4 * m] = p1i + ar;
      }
      return;
    }
    const { yr, yi } = this;
    for (let k = 0; k < m; k++) {
      for (let j = 0; j < f; j++) {
        yr[j] = outRe[outOff + j * m + k]!;
        yi[j] = outIm[outOff + j * m + k]!;
      }
      for (let q = 0; q < f; q++) {
        // The table index of W_len^{j(k + q·m)} grows by the same amount with each j, modulo n.
        const base = ((k + q * m) * step) % n;
        let sr = 0;
        let si = 0;
        for (let j = 0, t = 0; j < f; j++) {
          const c = cos[t]!;
          const sn = sin[t]!;
          sr += yr[j]! * c - yi[j]! * sn;
          si += yr[j]! * sn + yi[j]! * c;
          t += base;
          if (t >= n) t -= n;
        }
        outRe[outOff + q * m + k] = sr;
        outIm[outOff + q * m + k] = si;
      }
    }
  }
}

/**
 * |DFT|² of real frames of an even length n, bins 0…n/2, from a complex FFT of half the length:
 * z[j] = x[2j] + i·x[2j + 1] transforms to Z, from which the transforms of the even and the odd
 * samples follow (E = (Z[k] + Z*[h - k]) / 2, O = (Z[k] - Z*[h - k]) / 2i), and X[k] = E + W_n^k · O.
 */
export class RealPowerSpectrum {
  private readonly fft: FFT;
  private readonly re: Float64Array;
  private readonly im: Float64Array;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;

  constructor(readonly n: number) {
    if (n % 2 !== 0) throw new Error(`RealPowerSpectrum: length ${n} is not even`);
    const h = n / 2;
    this.fft = new FFT(h);
    this.re = new Float64Array(h);
    this.im = new Float64Array(h);
    this.cos = Float64Array.from({ length: h + 1 }, (_, k) => Math.cos((-2 * Math.PI * k) / n));
    this.sin = Float64Array.from({ length: h + 1 }, (_, k) => Math.sin((-2 * Math.PI * k) / n));
  }

  /** Writes |X[k]|², k = 0…n/2, of the n real samples in `x` to `power`. */
  run(x: Float64Array, power: Float64Array): void {
    const h = this.n / 2;
    const { re, im, cos, sin } = this;
    for (let j = 0; j < h; j++) {
      re[j] = x[2 * j]!;
      im[j] = x[2 * j + 1]!;
    }
    this.fft.transform(re, im);
    for (let k = 0; k <= h; k++) {
      const a = k === h ? 0 : k;
      const b = k === 0 ? 0 : h - k;
      // Z[a] and the conjugate of Z[b].
      const er = (re[a]! + re[b]!) / 2;
      const ei = (im[a]! - im[b]!) / 2;
      // (Z[a] - Z*[b]) / 2i.
      const or = (im[a]! + im[b]!) / 2;
      const oi = (re[b]! - re[a]!) / 2;
      const xr = er + or * cos[k]! - oi * sin[k]!;
      const xi = ei + or * sin[k]! + oi * cos[k]!;
      power[k] = xr * xr + xi * xi;
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

const fftCache = new Map<number, RealPowerSpectrum>();
const filterCache = new Map<string, Float64Array>();
const spanCache = new Map<string, Int32Array>();

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
  if (!fft) fftCache.set(nFft, (fft = new RealPowerSpectrum(nFft)));
  const key = `${bins}/${mels}/${sr}`;
  let filters = filterCache.get(key);
  if (!filters) filterCache.set(key, (filters = melFilters(bins, mels, sr)));
  let spans = spanCache.get(key);
  if (!spans) spanCache.set(key, (spans = filterSpans(filters, bins, mels)));
  const window = Float64Array.from({ length: nFft }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft));
  const frame = new Float64Array(nFft);
  const power = new Float64Array(bins);
  const out = new Float32Array(frames * mels);
  let max = -Infinity;
  for (let t = 0; t < frames; t++) {
    const start = t * hop - half;
    // Only the first and the last frames reach past the samples and need the reflection.
    if (start >= 0 && start + nFft <= total) for (let i = 0; i < nFft; i++) frame[i] = x[start + i]! * window[i]!;
    else for (let i = 0; i < nFft; i++) frame[i] = at(start + i) * window[i]!;
    fft.run(frame, power);
    for (let m = 0; m < mels; m++) {
      let s = 0;
      // Each triangle covers a few bins; the zeros outside it add nothing (the sum is the same bits).
      for (let b = spans[2 * m]!, end = spans[2 * m + 1]!; b < end; b++) s += power[b]! * filters[b * mels + m]!;
      const v = Math.log10(Math.max(s, 1e-10));
      out[t * mels + m] = v;
      if (v > max) max = v;
    }
  }
  for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i]!, max - 8) + 4) / 4;
  return out;
}

/**
 * HTK-style mel filters [bins, mels] without area normalisation (transformers' `mel_filter_bank`
 * with `mel_scale: "htk"`, `norm: None`): triangles between points evenly spaced in mel
 * (2595 · log10(1 + f / 700)) from fmin to fmax, each peaking at 1.
 */
export function htkMelFilters(bins: number, mels: number, sampleRate: number, fmin = 0, fmax = sampleRate / 2): Float64Array {
  const mel = (f: number) => 2595 * Math.log10(1 + f / 700);
  const hz = (m: number) => 700 * (10 ** (m / 2595) - 1);
  const pts = Array.from({ length: mels + 2 }, (_, i) => hz(mel(fmin) + ((mel(fmax) - mel(fmin)) * i) / (mels + 1)));
  const out = new Float64Array(bins * mels);
  for (let b = 0; b < bins; b++) {
    const f = (b * Math.floor(sampleRate / 2)) / (bins - 1);
    for (let m = 0; m < mels; m++) {
      const down = (f - pts[m]!) / (pts[m + 1]! - pts[m]!);
      const up = (pts[m + 2]! - f) / (pts[m + 2]! - pts[m + 1]!);
      out[b * mels + m] = Math.max(0, Math.min(down, up));
    }
  }
  return out;
}

/** Samples Gemma 4's audio tower takes at most: 30 s at 16 kHz. */
export const GEMMA4_AUDIO_MAX_SAMPLES = 480000;

/**
 * Gemma 4's log-mel features of 16 kHz audio (at most 30 s; the rest is dropped), time-major
 * [frames, 128]: frames of 320 samples every 160, the first centred on the first sample (160 zeros
 * before it), periodic Hann window, |FFT| of 512 points, HTK mel filters, log(x + 0.001). Only frames
 * that lie wholly inside the audio are kept, as the model's mask does.
 */
export function gemma4AudioFeatures(samples: Float32Array): { features: Float32Array; frames: number } {
  const [frameLen, hop, nFft, mels] = [320, 160, 512, 128];
  const n = Math.min(samples.length, GEMMA4_AUDIO_MAX_SAMPLES);
  // A frame is kept when the sample after its window is still audio (the extractor reads one past it).
  const frames = Math.max(0, Math.floor((n + frameLen / 2 - (frameLen + 1)) / hop) + 1);
  const bins = nFft / 2 + 1;
  let fft = fftCache.get(nFft);
  if (!fft) fftCache.set(nFft, (fft = new RealPowerSpectrum(nFft)));
  const key = `htk/${bins}/${mels}`;
  let filters = filterCache.get(key);
  if (!filters) filterCache.set(key, (filters = htkMelFilters(bins, mels, 16000, 0, 8000)));
  let spans = spanCache.get(key);
  if (!spans) spanCache.set(key, (spans = filterSpans(filters, bins, mels)));
  const window = Float32Array.from({ length: frameLen }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / frameLen));
  const frame = new Float64Array(nFft);
  const power = new Float64Array(bins);
  const out = new Float32Array(frames * mels);
  for (let t = 0; t < frames; t++) {
    const start = t * hop - frameLen / 2;
    for (let i = 0; i < frameLen; i++) frame[i] = start + i < 0 ? 0 : samples[start + i]! * window[i]!;
    fft.run(frame, power);
    for (let b = 0; b < bins; b++) power[b] = Math.sqrt(power[b]!);
    for (let m = 0; m < mels; m++) {
      let sum = 0;
      for (let b = spans[2 * m]!, end = spans[2 * m + 1]!; b < end; b++) sum += power[b]! * filters[b * mels + m]!;
      out[t * mels + m] = Math.log(sum + 1e-3);
    }
  }
  return { features: out, frames };
}

/** [first, end) of the nonzero bins of each mel filter, as pairs. */
function filterSpans(filters: Float64Array, bins: number, mels: number): Int32Array {
  const spans = new Int32Array(2 * mels);
  for (let m = 0; m < mels; m++) {
    let lo = bins;
    let hi = 0;
    for (let b = 0; b < bins; b++) {
      if (filters[b * mels + m] !== 0) {
        lo = Math.min(lo, b);
        hi = b + 1;
      }
    }
    spans[2 * m] = Math.min(lo, hi);
    spans[2 * m + 1] = hi;
  }
  return spans;
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
