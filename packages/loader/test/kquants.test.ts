import { describe, expect, it } from "vitest";
import { dequantQ4_1, dequantQ4_K, dequantQ5, dequantQ6_K, rowBytes } from "../src";

const ONE = 0x3c00; // f16 1.0
const HALF = 0x3800; // f16 0.5

function block(size: number, fill: (b: Uint8Array, dv: DataView) => void): Uint8Array {
  const b = new Uint8Array(size);
  fill(b, new DataView(b.buffer));
  return b;
}

describe("GGUF block dequantization", () => {
  it("Q4_1: q * d + m, low nibbles first", () => {
    const b = block(20, (u, dv) => {
      dv.setUint16(0, ONE, true);
      dv.setUint16(2, HALF, true);
      for (let j = 0; j < 16; j++) u[4 + j] = j | ((15 - j) << 4);
    });
    const y = dequantQ4_1(b, 32);
    expect(y[3]).toBe(3.5);
    expect(y[16 + 3]).toBe(12.5);
  });

  it("Q5_0 / Q5_1: fifth bit from qh (bit j for low, bit j + 16 for high)", () => {
    const b = block(22, (_u, dv) => {
      dv.setUint16(0, ONE, true);
      dv.setUint32(2, (1 << 3) | (1 << (5 + 16)), true);
    });
    const y = dequantQ5(b, 32, false);
    expect(y[3]).toBe(0);
    expect(y[4]).toBe(-16);
    expect(y[16 + 5]).toBe(0);
    const b1 = block(24, (u, dv) => {
      dv.setUint16(0, ONE, true);
      dv.setUint16(2, HALF, true);
      dv.setUint32(4, 1 << 0, true);
      u[8] = 0x21;
    });
    const y1 = dequantQ5(b1, 32, true);
    expect(y1[0]).toBe(17.5); // (1 | 16) + 0.5
    expect(y1[16]).toBe(2.5);
  });

  it("Q4_K: per-sub-block 6-bit scales and mins, both halves of the scale table", () => {
    const b = block(144, (u, dv) => {
      dv.setUint16(0, ONE, true);
      dv.setUint16(2, ONE, true);
      const s = 4;
      u[s + 0] = 2; // sub-block 0 scale
      u[s + 4] = 1; // sub-block 0 min
      u[s + 1] = 3; // sub-block 1 scale, min 0
      u[s + 8] = 0x07; // sub-block 4 scale low bits
      u[s + 0] |= 0x40; // sub-block 4 scale high bits (bit 4)
      for (let l = 0; l < 128; l++) u[16 + l] = 0x51;
    });
    const y = dequantQ4_K(b, 256);
    expect(y[0]).toBe(1); // 2·1 − 1
    expect(y[32]).toBe(15); // 3·5
    expect(y[128]).toBe(23); // (7 | 16)·1
    expect(y[160]).toBe(0); // sub-block 5 has scale 0
  });

  it("Q6_K: 4 + 2 bits, signed scales, two 128-element halves", () => {
    const b = block(210, (u, dv) => {
      u[0] = 5; // ql[0] low nibble
      u[128] = 2; // qh[0] bits 0-1 → high bits of element 0
      dv.setInt8(192 + 0, 2);
      dv.setInt8(192 + 4, 1);
      dv.setInt8(192 + 8, -1);
      dv.setUint16(208, ONE, true);
    });
    const y = dequantQ6_K(b, 256);
    expect(y[0]).toBe(10); // ((5 | 32) − 32)·2
    expect(y[64]).toBe(-32); // q3 = 0 − 32, scale 1
    expect(y[128]).toBe(32); // second half, scale −1
  });

  it("k-quant row sizes need multiples of 256", () => {
    expect(rowBytes("Q4_K", 512)).toBe(288);
    expect(rowBytes("Q6_K", 256)).toBe(210);
    expect(() => rowBytes("Q4_K", 576)).toThrow(/multiple of 256/);
  });
});
