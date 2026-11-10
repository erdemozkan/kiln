import { describe, expect, it } from "vitest";
import { kokoroNormalize, kokoroPhonemes } from "../src/kokoro-text";

describe("Kokoro's text front end (kokoro-js's)", () => {
  it("normalizes numbers, money, times and titles", () => {
    expect(kokoroNormalize("Dr. Smith paid $5.50 in 1990.")).toBe("Doctor Smith paid 5 dollars and 50 cents in 19 90.");
    expect(kokoroNormalize("Pi is 3.14, it's 10:05.")).toBe("Pi is 3 point 1 4, it's 10 oh 5.");
    expect(kokoroNormalize("Pages 4-7 cost £1")).toBe("Pages 4 to 7 cost 1 pound");
  });

  it("phonemizes the text between punctuation and fixes the phonemes up", async () => {
    const calls: [string, string][] = [];
    const fake = async (t: string, lang: string) => {
      calls.push([t, lang]);
      return [t === "Hello" ? "həlˈoʊ" : "wˈɜːld r"];
    };
    expect(await kokoroPhonemes("Hello, world!", fake)).toBe("həlˈoʊ, wˈɜːld ɹ!");
    expect(calls).toEqual([["Hello", "en-us"], ["world", "en-us"]]);
    await kokoroPhonemes("Hi", async (_t, lang) => (calls.push(["", lang]), "hˈaɪ"), "bf_emma");
    expect(calls.at(-1)).toEqual(["", "en"]);
  });
});
