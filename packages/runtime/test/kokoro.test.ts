import { describe, expect, it } from "vitest";
import { istft, kokoroDurations, repeatRows, stftMagPhase, strideConv } from "../src/kokoro";

describe("Kokoro's CPU steps", () => {
  it("rounds durations as torch.round (halves to even), at least one frame", () => {
    // Logits whose sigmoids sum to 2.5, 3.5, 0.2 (one column each).
    const logit = (p: number) => Math.log(p / (1 - p));
    const rows = [2.5, 3.5, 0.2].map((s) => [logit(s / 4), logit(s / 4), logit(s / 4), logit(s / 4)]);
    expect(kokoroDurations(Float32Array.from(rows.flat()), 3, 4)).toEqual([2, 4, 1]);
    expect(kokoroDurations(Float32Array.from(rows.flat()), 3, 4, 2)).toEqual([1, 2, 1]);
    expect(Array.from(repeatRows(Float32Array.of(1, 2, 3, 4), 2, [2, 1]))).toEqual([1, 2, 1, 2, 3, 4]);
  });

  it("downsamples a curve as Conv1d(1, 1, 3, stride 2, padding 1)", () => {
    expect(Array.from(strideConv(Float32Array.of(1, 2, 3, 4), Float32Array.of(1, 10, 100), 0.5))).toEqual([210.5, 432.5]);
  });

  // numpy's rfft / irfft with torch.stft / torch.istft's framing (n_fft 4, hop 1, periodic Hann, centered, reflect padding).
  it("takes the STFT as torch.stft does", () => {
    const mag = [0.25, 0.5, 0.75, 0.5, 0.353553, 1.0, 1.25, 1.118034, 0.75, 1.0, 1.06066, 0.5, 0.0, 0.559017, 1.0, 0.25, 0.25, 0.75, 0.0, 0.559017, 1.0];
    const ph = [0.0, -3.141593, 0.0, 0.0, 0.785398, 3.141593, 0.0, 2.677945, 0.0, 0.0, -2.356194, 0.0, 0.0, -0.463648, 3.141593, 3.141593, -3.141593, 0.0, 0.0, 0.463648, 3.141593];
    const s = stftMagPhase(Float32Array.of(0.5, -0.25, 1, 0.75, -0.5, 0.25), 4, 1);
    for (let f = 0; f < 7; f++)
      for (let b = 0; b < 3; b++) {
        expect(s[f * 8 + b]).toBeCloseTo(mag[f * 3 + b]!, 5);
        // Phases of empty bins are arbitrary; ±π are the same angle.
        if (mag[f * 3 + b]! > 1e-6) expect(Math.cos(s[f * 8 + 3 + b]! - ph[f * 3 + b]!)).toBeCloseTo(1, 5);
      }
  });

  it("inverts log-magnitudes and sine phases as torch.istft does", () => {
    const y = [0.062865, -0.066052, 0.320211, 0.05245, -0.267835, 0.180798, 0.652, 0.47354, -0.351868, -0.632711, -0.311637, 0.020663, -1.162515, -0.109396, -0.622955, -0.366134, -0.272129, -0.15815, 0.205815, 0.521257, -0.064267, 0.683232, -0.332597, 0.175755, 0.451735, 0.047006, -0.37175, -0.460863, -0.228863, 0.110098];
    const out = istft(Float32Array.from(y), 5, 4, 1);
    [0.308283, -0.172667, -0.061837, -0.171572].forEach((v, i) => expect(out[i]).toBeCloseTo(v, 5));
  });
});
