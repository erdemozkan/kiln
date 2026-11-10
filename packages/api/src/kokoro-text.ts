// Kokoro's text front end, ported from kokoro-js 1.2.1 (Apache-2.0, by Joshua Lochner): English text
// normalization (numbers, money, abbreviations), splitting around punctuation, and the fixes that
// bring espeak-ng's phonemes closer to Kokoro's (misaki's) set. The phonemizer itself is the caller's:
// a function from text and language ("en-us" or "en") to phoneme strings, such as `phonemize` from
// the `phonemizer` package (espeak-ng compiled to WebAssembly).

/** A phonemizer: text and language ("en-us" or "en") → phonemes, as the `phonemizer` package's `phonemize`. */
export type Phonemizer = (text: string, language: string) => Promise<string[] | string> | string[] | string;

function splitNum(s: string): string {
  if (s.includes(".")) return s;
  if (s.includes(":")) {
    const [h, m] = s.split(":").map(Number) as [number, number];
    return m === 0 ? `${h} o'clock` : m < 10 ? `${h} oh ${m}` : `${h} ${m}`;
  }
  const year = parseInt(s.slice(0, 4), 10);
  if (year < 1100 || year % 1000 < 10) return s;
  const left = s.slice(0, 2);
  const right = parseInt(s.slice(2, 4), 10);
  const suffix = s.endsWith("s") ? "s" : "";
  if (year % 1000 >= 100 && year % 1000 <= 999) {
    if (right === 0) return `${left} hundred${suffix}`;
    if (right < 10) return `${left} oh ${right}${suffix}`;
  }
  return `${left} ${right}${suffix}`;
}

function flipMoney(m: string): string {
  const bill = m[0] === "$" ? "dollar" : "pound";
  if (isNaN(Number(m.slice(1)))) return `${m.slice(1)} ${bill}s`;
  if (!m.includes(".")) return `${m.slice(1)} ${bill}${m.slice(1) === "1" ? "" : "s"}`;
  const [b, c] = m.slice(1).split(".") as [string, string];
  const d = parseInt(c.padEnd(2, "0"), 10);
  const coins = m[0] === "$" ? (d === 1 ? "cent" : "cents") : d === 1 ? "penny" : "pence";
  return `${b} ${bill}${b === "1" ? "" : "s"} and ${d} ${coins}`;
}

function pointNum(n: string): string {
  const [a, b] = n.split(".") as [string, string];
  return `${a} point ${b.split("").join(" ")}`;
}

/** kokoro-js's English text normalization. */
export function kokoroNormalize(text: string): string {
  return text
    .replace(/[‘’]/g, "'")
    .replace(/«/g, "“")
    .replace(/»/g, "”")
    .replace(/[“”]/g, '"')
    .replace(/\(/g, "«")
    .replace(/\)/g, "»")
    .replace(/、/g, ", ")
    .replace(/。/g, ". ")
    .replace(/！/g, "! ")
    .replace(/，/g, ", ")
    .replace(/：/g, ": ")
    .replace(/；/g, "; ")
    .replace(/？/g, "? ")
    .replace(/[^\S \n]/g, " ")
    .replace(/  +/, " ")
    .replace(/(?<=\n) +(?=\n)/g, "")
    .replace(/\bD[Rr]\.(?= [A-Z])/g, "Doctor")
    .replace(/\b(?:Mr\.|MR\.(?= [A-Z]))/g, "Mister")
    .replace(/\b(?:Ms\.|MS\.(?= [A-Z]))/g, "Miss")
    .replace(/\b(?:Mrs\.|MRS\.(?= [A-Z]))/g, "Mrs")
    .replace(/\betc\.(?! [A-Z])/gi, "etc")
    .replace(/\b(y)eah?\b/gi, "$1e'a")
    .replace(/\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)/g, splitNum)
    .replace(/(?<=\d),(?=\d)/g, "")
    .replace(/[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b/gi, flipMoney)
    .replace(/\d*\.\d+/g, pointNum)
    .replace(/(?<=\d)-(?=\d)/g, " to ")
    .replace(/(?<=\d)S/g, " S")
    .replace(/(?<=[BCDFGHJ-NP-TV-Z])'?s\b/g, "'S")
    .replace(/(?<=X')S\b/g, "s")
    .replace(/(?:[A-Za-z]\.){2,} [a-z]/g, (m) => m.replace(/\./g, "-"))
    .replace(/(?<=[A-Z])\.(?=[A-Z])/gi, "-")
    .trim();
}

const PUNCTUATION = ';:,.!?¡¿—…"«»“”(){}[]';
const PUNCTUATION_RUN = new RegExp(`(\\s*[${PUNCTUATION.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}]+\\s*)+`, "g");

/**
 * Text to Kokoro phonemes as kokoro-js does it: normalized (unless `normalize` is false), split
 * around punctuation (kept as is), each piece phonemized, the joined result fixed up. `voice`'s first
 * letter picks the language: "a" American English (en-us), "b" British (en).
 */
export async function kokoroPhonemes(text: string, phonemize: Phonemizer, voice = "af_heart", normalize = true): Promise<string> {
  const lang = voice[0] === "b" ? "b" : "a";
  const t = normalize ? kokoroNormalize(text) : text;
  const pieces: { match: boolean; text: string }[] = [];
  let at = 0;
  for (const m of t.matchAll(PUNCTUATION_RUN)) {
    if (at < m.index!) pieces.push({ match: false, text: t.slice(at, m.index) });
    if (m[0].length) pieces.push({ match: true, text: m[0] });
    at = m.index! + m[0].length;
  }
  if (at < t.length) pieces.push({ match: false, text: t.slice(at) });
  const language = lang === "a" ? "en-us" : "en";
  const parts = await Promise.all(
    pieces.map(async (p) => {
      if (p.match) return p.text;
      const r = await phonemize(p.text, language);
      return Array.isArray(r) ? r.join(" ") : r;
    }),
  );
  let ph = parts
    .join("")
    .replace(/kəkˈoːɹoʊ/g, "kˈoʊkəɹoʊ")
    .replace(/kəkˈɔːɹəʊ/g, "kˈəʊkəɹəʊ")
    .replace(/ʲ/g, "j")
    .replace(/r/g, "ɹ")
    .replace(/x/g, "k")
    .replace(/ɬ/g, "l")
    .replace(/(?<=[a-zɹː])(?=hˈʌndɹɪd)/g, " ")
    .replace(/ z(?=[;:,.!?¡¿—…"«»“” ]|$)/g, "z");
  if (lang === "a") ph = ph.replace(/(?<=nˈaɪn)ti(?!ː)/g, "di");
  return ph.trim();
}
