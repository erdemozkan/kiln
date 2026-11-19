// Skills: tools that come with Kiln, ready for `runTools`. Each does real work: arithmetic (exact for
// integers up to 2^53), the clock, Wikipedia, a web page's text. The last two go out to the network.
import type { RunnableTool, ToolContext } from "./tools";

export interface SkillOptions {
  /** The `fetch` the network skills use (default: the global one). */
  fetch?: typeof fetch;
  /** Most characters of text a network skill hands the model (default 2000): it all goes into the context. */
  maxChars?: number;
  /** The Wikipedia edition `search_wikipedia` reads when the model names none (default "en"); a bad code throws here. */
  language?: string;
  /**
   * Decides whether `read_web_page` may fetch a URL (each redirect too). The default refuses this
   * machine, private networks (localhost, 10.x, 192.168.x …) and the embedding page's own origin: the
   * model picks the address, and a page it read can suggest one. In Node.js it also refuses a name
   * that resolves to such an address; a custom `allowUrl` decides alone.
   */
  allowUrl?(url: URL): boolean;
  /**
   * The application's name, sent to Wikipedia as `Api-User-Agent` (default "Rectus"). Wikimedia asks
   * clients to identify themselves and limits anonymous ones sooner. It is sent to Wikipedia only.
   */
  userAgent?: string;
}

/** Seams for tests: no real DNS, sleeping or 15 s waits. Not part of the public API. */
export interface SkillHooks extends SkillOptions {
  /** Resolves a host name to its addresses (default in Node.js: `dns.promises.lookup(host, { all: true })`). */
  lookup?(host: string): Promise<{ address: string }[]>;
  sleep?(ms: number): Promise<void>;
  /** Milliseconds each request may take (default 15000). */
  timeout?: number;
}

const skill = (name: string, description: string, properties: Record<string, unknown>, required: string[], execute: RunnableTool["execute"]): RunnableTool => ({
  type: "function",
  function: { name, description, parameters: { type: "object", properties, required } },
  execute,
});

// ---- calculate ------------------------------------------------------------------------------

/** Each function with the fewest and most arguments it takes. */
const FUNCTIONS: Record<string, [(...a: number[]) => number, number, number]> = {
  sqrt: [Math.sqrt, 1, 1],
  abs: [Math.abs, 1, 1],
  round: [(x, digits = 0) => Math.round(x * 10 ** digits) / 10 ** digits, 1, 2],
  floor: [Math.floor, 1, 1],
  ceil: [Math.ceil, 1, 1],
  sin: [Math.sin, 1, 1],
  cos: [Math.cos, 1, 1],
  tan: [Math.tan, 1, 1],
  ln: [Math.log, 1, 1],
  log: [Math.log10, 1, 1],
  log10: [Math.log10, 1, 1],
  log2: [Math.log2, 1, 1],
  exp: [Math.exp, 1, 1],
  min: [Math.min, 1, Infinity],
  max: [Math.max, 1, Infinity],
  pow: [Math.pow, 2, 2],
};
const CONSTANTS: Record<string, number> = { pi: Math.PI, e: Math.E };
const MAX_DEPTH = 200;

/**
 * The value of an arithmetic expression: numbers, + - * / ^ (or **), % (remainder), parentheses, the
 * functions above and pi / e. A parser of its own: the text comes from a model, so it is never run as code.
 * Integers come back exact (up to 2^53); other results are rounded to 12 significant digits.
 */
export function evaluate(expression: string): number {
  // "×" is how people (and models) write a product; "1,234" is handled where numbers are read.
  const src = expression.replace(/[×·]/g, "*").replace(/÷/g, "/").replace(/−/g, "-");
  let i = 0;
  let depth = 0;
  let inArgs = 0;
  const fail = (what: string): never => {
    throw new Error(`cannot calculate ${JSON.stringify(expression)}: ${what}`);
  };
  const deeper = () => {
    if (++depth > MAX_DEPTH) fail(`it nests more than ${MAX_DEPTH} levels deep`);
  };
  const skip = () => {
    while (src[i] === " " || src[i] === "\t" || src[i] === "\n" || src[i] === "\r") i++;
  };
  const eat = (s: string): boolean => {
    skip();
    if (!src.startsWith(s, i)) return false;
    i += s.length;
    return true;
  };
  // Thousands separators only outside a function's arguments: there a comma separates them (max(1,500)).
  const GROUPED = /(?:\d+(?:,\d{3}(?!\d))*\.?\d*|\.\d+)(?:e[+-]?\d+)?/iy;
  const PLAIN = /(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/iy;
  const WORD = /[a-z_]+\d*/iy;
  const match = (re: RegExp): string | undefined => {
    re.lastIndex = i;
    const m = re.exec(src)?.[0];
    if (m) i += m.length;
    return m;
  };
  const primary = (): number => {
    skip();
    if (eat("(")) {
      deeper();
      const v = sum();
      if (!eat(")")) fail("a ) is missing");
      depth--;
      return v;
    }
    const num = match(inArgs ? PLAIN : GROUPED);
    if (num) return Number(num.replace(/,/g, ""));
    const word = match(WORD);
    if (word) {
      const name = word.toLowerCase();
      if (eat("(")) {
        // Own properties only: "constructor" is not a function here.
        const [f, least, most] = Object.hasOwn(FUNCTIONS, name) ? FUNCTIONS[name]! : fail(`there is no function ${name}`);
        deeper();
        inArgs++;
        const args: number[] = [];
        if (!eat(")")) {
          do args.push(sum());
          while (eat(","));
          if (!eat(")")) fail("a ) is missing");
        }
        inArgs--;
        depth--;
        if (args.length < least || args.length > most) fail(`${name} takes ${most === Infinity ? "one or more numbers" : least === most ? (least === 1 ? "one number" : `${least} numbers`) : `${least} or ${most} numbers`}, not ${args.length}`);
        return f(...args);
      }
      return Object.hasOwn(CONSTANTS, name) ? CONSTANTS[name]! : fail(`${word} is not a number`);
    }
    return fail(i >= src.length ? "it ends too early" : `unexpected ${JSON.stringify(src[i])}`);
  };
  // Signs in a loop, not by recursion: "------1" must not use up the stack.
  const unary = (): number => {
    let negative = false;
    for (;;) {
      if (eat("-")) negative = !negative;
      else if (!eat("+")) break;
    }
    const v = power();
    return negative ? -v : v;
  };
  // Right-associative, and tighter than a sign before it: -2^2 is -4.
  function power(): number {
    const base = primary();
    if (!(eat("**") || eat("^"))) return base;
    deeper();
    const v = base ** unary();
    depth--;
    return v;
  }
  const product = (): number => {
    let v = unary();
    for (;;) {
      if (eat("*")) v *= unary();
      else if (eat("/")) v /= unary();
      else if (eat("%")) v %= unary();
      else return v;
    }
  };
  function sum(): number {
    let v = product();
    for (;;) {
      if (eat("+")) v += product();
      else if (eat("-")) v -= product();
      else return v;
    }
  }
  const v = sum();
  skip();
  if (i < src.length) fail(`unexpected ${JSON.stringify(src[i])}`);
  if (!Number.isFinite(v)) fail("the result is not a finite number");
  // 0.1 + 0.2 reads as 0.3: float noise past 12 digits would only mislead the model. Integers are
  // left alone: rounding 9449772114007 to 12 digits would make it wrong.
  return Number.isInteger(v) ? v : Number(v.toPrecision(12));
}

/** Arithmetic done for the model: models guess at sums they should compute. */
export function calculatorSkill(): RunnableTool {
  return skill(
    "calculate",
    "Calculate an arithmetic expression exactly. Use it for any arithmetic instead of working it out yourself.",
    // No worked example here: a small model copies it instead of writing the user's sum.
    { expression: { type: "string", description: "The arithmetic to compute, written with numbers, + - * / ^ and parentheses. Functions: sqrt abs round floor ceil sin cos tan ln log log2 exp min max pow. Constants: pi e." } },
    ["expression"],
    ({ expression }) => ({ expression, result: evaluate(String(expression)) }),
  );
}

// ---- get_date_time --------------------------------------------------------------------------

const key = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z]/g, "");
/** Cities people ask about whose zone carries another city's name. */
const ZONE_OF: Record<string, string> = {
  washington: "America/New_York", boston: "America/New_York", miami: "America/New_York", sanfrancisco: "America/Los_Angeles", seattle: "America/Los_Angeles",
  beijing: "Asia/Shanghai", delhi: "Asia/Kolkata", newdelhi: "Asia/Kolkata", mumbai: "Asia/Kolkata", bangalore: "Asia/Kolkata", ankara: "Europe/Istanbul",
  izmir: "Europe/Istanbul", munich: "Europe/Berlin", frankfurt: "Europe/Berlin", hamburg: "Europe/Berlin", barcelona: "Europe/Madrid", milan: "Europe/Rome",
  geneva: "Europe/Zurich", rio: "America/Sao_Paulo", riodejaneiro: "America/Sao_Paulo", canberra: "Australia/Sydney", ottawa: "America/Toronto",
  montreal: "America/Toronto", abudhabi: "Asia/Dubai", osaka: "Asia/Tokyo", kyoto: "Asia/Tokyo", capetown: "Africa/Johannesburg", utc: "UTC", gmt: "UTC",
  // Older ICU data (some browsers) knows only the old name.
  kyiv: "Europe/Kiev",
};

const validZone = (zone: string): boolean => {
  try {
    new Intl.DateTimeFormat("en", { timeZone: zone });
    return true;
  } catch {
    return false;
  }
};

/** The IANA time zone for a zone name ("Europe/Paris"), a city ("Paris", "new york") or an offset ("UTC+3"); undefined when unknown. */
export function timeZoneOf(place: string): string | undefined {
  const p = place.trim();
  const zones = (Intl as { supportedValuesOf?(k: string): string[] }).supportedValuesOf?.("timeZone") ?? [];
  if (p.includes("/")) {
    const name = p.replace(/ /g, "_");
    if (!validZone(name)) return undefined;
    // The name as asked: Intl would answer "Asia/Calcutta" for "Asia/Kolkata".
    return zones.find((z) => z.toLowerCase() === name.toLowerCase()) ?? name;
  }
  const offset = /^(?:utc|gmt)\s*([+−-])\s*(\d{1,2})(?::?(\d{2}))?$/i.exec(p);
  if (offset) {
    const [h, m] = [Number(offset[2]), Number(offset[3] ?? 0)];
    if (h > 14 || m >= 60) return undefined;
    if (!h && !m) return "UTC";
    const sign = offset[1] === "+" ? "+" : "-";
    const zone = `${sign}${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
    if (validZone(zone)) return zone;
    // Etc/GMT zones have the sign the other way round, and whole hours only.
    const etc = `Etc/GMT${sign === "+" ? "-" : "+"}${h}`;
    return !m && validZone(etc) ? etc : undefined;
  }
  // "Paris2" or "+Tokyo" is not a place: the key would drop what made it different.
  if (/[\d+−]/.test(p)) return undefined;
  const k = key(p);
  if (!k) return undefined;
  if (Object.hasOwn(ZONE_OF, k)) return ZONE_OF[k];
  return zones.find((z) => key(z.slice(z.lastIndexOf("/") + 1)) === k);
}

/** The date and time now: models do not know either. */
export function dateTimeSkill(now: () => Date = () => new Date()): RunnableTool {
  return skill(
    "get_date_time",
    "Get the current date and time, here or in a city or time zone. Use it whenever the answer depends on today's date or the time.",
    { place: { type: "string", description: 'A city ("Tokyo") or an IANA time zone ("Asia/Tokyo"). Leave out for the local time.' } },
    [],
    ({ place }) => {
      const asked = typeof place === "string" ? place.trim() : "";
      const zone = asked ? timeZoneOf(asked) : undefined;
      if (asked && !zone) throw new Error(`no time zone known for ${JSON.stringify(asked)}; give an IANA zone such as "Europe/Paris" or a larger city`);
      const d = now();
      const part = (o: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat("en-GB", { timeZone: zone, ...o });
      const p = Object.fromEntries(part({ year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).formatToParts(d).map((x) => [x.type, x.value]));
      const offset = part({ timeZoneName: "longOffset" }).formatToParts(d).find((x) => x.type === "timeZoneName")?.value ?? "";
      return {
        ...(asked ? { place: asked } : {}),
        date: `${p.year}-${p.month}-${p.day}`,
        weekday: part({ weekday: "long" }).format(d),
        time: `${p.hour}:${p.minute}`,
        time_zone: zone ?? part({}).resolvedOptions().timeZone,
        utc_offset: offset === "GMT" ? "GMT+00:00" : offset,
      };
    },
  );
}

// ---- the network skills -----------------------------------------------------------------------

/** `text` cut to `max` characters at a sentence or word end, saying so when it was cut. */
function clip(text: string, max: number): string {
  if (text.length <= max) return text;
  let head = text.slice(0, max);
  const stop = Math.max(head.lastIndexOf(". "), head.lastIndexOf("\n"));
  const space = head.lastIndexOf(" ");
  if (stop > max * 0.6) head = head.slice(0, stop + 1);
  else if (space > max * 0.6) head = head.slice(0, space);
  // Japanese, Chinese, Thai have no spaces: cut where the room ends, but not inside a surrogate pair.
  else if (/[\uD800-\uDBFF]$/.test(head)) head = head.slice(0, -1);
  return `${head.trimEnd()} […]`;
}

const TIMEOUT = 15_000;
/** Most bytes of a page read: the model sees `maxChars` of it, and a huge page only costs time and memory. */
const MAX_BYTES = 2_000_000;

/** A signal that fires at the deadline or when the run is stopped, whichever comes first. */
function deadline(ms: number, signal?: AbortSignal): AbortSignal {
  const timer = AbortSignal.timeout(ms);
  if (!signal) return timer;
  const any = (AbortSignal as { any?(s: AbortSignal[]): AbortSignal }).any;
  if (any) return any([signal, timer]);
  const both = new AbortController();
  for (const s of [signal, timer]) {
    if (s.aborted) both.abort(s.reason);
    else s.addEventListener("abort", () => both.abort(s.reason), { once: true });
  }
  return both.signal;
}

/** What a failed request tells the model. */
function failure(err: unknown, what: string): Error {
  const name = (err as { name?: unknown } | null)?.name;
  if (name === "TimeoutError") return new Error(`${what} did not answer in time`);
  if (name === "AbortError") return new Error(`stopped before ${what} answered`);
  // In a browser this is also what a site that forbids other pages to read it looks like.
  return new Error(`could not reach ${what} (${err instanceof Error ? err.message : String(err)}); the site may be down or may not allow being read from here`);
}

interface Ask {
  what: string;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  redirect?: RequestRedirect;
}

const isRedirect = (res: Response) => res.type === "opaqueredirect" || (res.status >= 300 && res.status < 400 && res.headers.has("location"));

async function get(opts: SkillHooks, url: string, ask: Ask): Promise<Response> {
  const f = opts.fetch ?? fetch;
  const once = async (): Promise<Response> => {
    try {
      // No cookies, no Referer: the request must not carry anything of the page Kiln runs in.
      return await f(url, {
        headers: { Accept: "text/html,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.1", ...ask.headers },
        credentials: "omit",
        referrerPolicy: "no-referrer",
        redirect: ask.redirect ?? "follow",
        signal: deadline(opts.timeout ?? TIMEOUT, ask.signal),
      });
    } catch (err) {
      throw failure(err, ask.what);
    }
  };
  let res = await once();
  if (res.status === 429) {
    // Too many requests: one more try after the pause the site asks for (a few seconds at most; 2 when it names none).
    const header = res.headers.get("retry-after")?.trim();
    const asked = header ? Number(header) : NaN;
    await (opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms))))(Math.min(Number.isFinite(asked) && asked >= 0 ? asked : 2, 5) * 1000);
    res = await once();
  }
  if (res.status === 429) throw new Error(`${ask.what} is refusing requests for now (too many, HTTP 429); try again in a minute`);
  if (!res.ok && !(ask.redirect === "manual" && isRedirect(res))) throw new Error(`${ask.what} answered ${res.status}`);
  return res;
}

/** At most `MAX_BYTES` of a response's text; the rest is not downloaded. */
async function readText(res: Response, what: string): Promise<string> {
  try {
    if (!res.body) return (await res.text()).slice(0, MAX_BYTES);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let text = "";
    let bytes = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      const take = value.subarray(0, MAX_BYTES - bytes);
      bytes += take.length;
      text += decoder.decode(take, { stream: true });
      if (bytes >= MAX_BYTES) {
        void reader.cancel().catch(() => {});
        break;
      }
    }
    return text + decoder.decode();
  } catch (err) {
    throw failure(err, what);
  }
}

const wikiHeaders = (opts: SkillOptions): Record<string, string> => ({ "Api-User-Agent": opts.userAgent || "Rectus" });

/** A Wikipedia language code ("en", "simple", "zh-yue", "be-tarask"), lower-cased; undefined if it could not be a host label. */
const wikiLanguage = (s: unknown): string | undefined => {
  const code = typeof s === "string" ? s.trim().toLowerCase() : "";
  return /^[a-z0-9][a-z0-9-]{0,18}[a-z0-9]$/.test(code) ? code : undefined;
};

/** Wikipedia's article for a search: what a model was never trained on, or half remembers. */
export function wikipediaSkill(opts: SkillOptions = {}): RunnableTool {
  const max = opts.maxChars ?? 2000;
  const fallback = opts.language === undefined ? "en" : wikiLanguage(opts.language);
  if (!fallback) throw new Error(`wikipediaSkill: ${JSON.stringify(opts.language)} is not a Wikipedia language code`);
  return skill(
    "search_wikipedia",
    "Look something up on Wikipedia and get the start of the best matching article. Use it for facts about people, places, events and things.",
    { query: { type: "string", description: "What to look up: a name or a few words" }, language: { type: "string", pattern: "[a-zA-Z0-9-]{2,20}", description: `The Wikipedia edition, e.g. "en", "tr", "de" (default "${fallback}")` } },
    ["query"],
    async ({ query, language }, context?: ToolContext) => {
      const lang = wikiLanguage(language) ?? fallback;
      const q = String(query ?? "").trim();
      if (!q) throw new Error("the query is empty");
      // exlimit: without it only the first of the three hits carries its extract.
      const api = `https://${lang}.wikipedia.org/w/api.php?action=query&format=json&origin=*&generator=search&gsrlimit=3&gsrsearch=${encodeURIComponent(q)}&prop=extracts%7Cinfo&exintro=1&explaintext=1&exlimit=3&inprop=url&redirects=1`;
      const res = await get(opts, api, { what: "Wikipedia", headers: wikiHeaders(opts), signal: context?.signal });
      const json = JSON.parse(await readText(res, "Wikipedia")) as { query?: { pages?: Record<string, { index: number; title: string; extract?: string; fullurl?: string }> } };
      const pages = Object.values(json.query?.pages ?? {}).sort((a, b) => a.index - b.index);
      const page = pages.find((p) => p.extract?.trim());
      if (!page) throw new Error(`Wikipedia (${lang}) has no article for ${JSON.stringify(q)}`);
      return { title: page.title, url: page.fullurl, text: clip(page.extract!.trim(), max), ...(pages.length > 1 ? { other_matches: pages.filter((p) => p !== page).map((p) => p.title) } : {}) };
    },
  );
}

// ---- HTML to text: a linear scanner, never a backtracking regex (a page is whatever a site sends) ----

const ENTITIES: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", ndash: "–", mdash: "—", hellip: "…", rsquo: "’", lsquo: "‘", rdquo: "”", ldquo: "“", copy: "©", deg: "°" };
/** Elements dropped whole, content and all. */
const DROPPED = new Set(["script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside", "iframe", "head"]);
/** Closing tags that end a line (and <br>). */
const BLOCKS = new Set(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "section", "article", "blockquote"]);
/** Tags a README's HTML is laid out with: removed, their text kept. */
const LAYOUT = new Set(["h1", "h2", "h3", "h4", "h5", "h6", "img", "p", "a", "div", "br", "hr", "picture", "source", "span", "b", "i", "em", "strong", "sub", "sup", "center", "table", "thead", "tbody", "tr", "td", "th", "details", "summary", "kbd"]);

const decode = (s: string) =>
  s.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+\d*);/gi, (m, e: string) => {
    if (e[0] !== "#") return ENTITIES[e.toLowerCase()] ?? m;
    const code = e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
    return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : m;
  });

/** Characters of a tag name: the scan stops at anything else, so "<<<<" is never read twice. */
const nameChar = (c: number) => (c >= 97 && c <= 122) || (c >= 48 && c <= 57) || c === 45 || c === 58 || c === 95;
const endsName = (c: string | undefined) => c === undefined || c === ">" || c === "/" || c === " " || c === "\t" || c === "\n" || c === "\r" || c === "\f";

/**
 * Searches in `lower` (the text with ASCII lower-cased, same length) from positions that only grow.
 * Each needle's last answer is kept: asked again from before it, the answer stands, so no stretch is
 * searched twice and a needle once missing is never looked for again.
 */
function finder(lower: string) {
  const last = new Map<string, number>();
  const find = (needle: string, from: number): number => {
    const k = last.get(needle);
    if (k !== undefined && (k < 0 || k >= from)) return k;
    const at = lower.indexOf(needle, from);
    last.set(needle, at);
    return at;
  };
  /** The next "<name" or "</name" (`close`) whose name is exactly `name`. */
  const tag = (name: string, from: number, close = false): number => {
    const needle = (close ? "</" : "<") + name;
    for (let at = find(needle, from); at >= 0; at = find(needle, at + 1)) if (endsName(lower[at + needle.length])) return at;
    return -1;
  };
  return { find, tag };
}

/** The last "</name" with exactly that name, after `after`; -1 when none. */
function lastClose(lower: string, name: string, after: number): number {
  const needle = `</${name}`;
  for (let at = lower.lastIndexOf(needle); at > after; at = lower.lastIndexOf(needle, at - 1)) if (endsName(lower[at + needle.length])) return at;
  return -1;
}

const asciiLower = (s: string) => s.replace(/[A-Z]+/g, (m) => m.toLowerCase());

/** A page's readable text: its title, and its body without scripts, styles, navigation and markup. */
export function htmlToText(html: string): { title: string; text: string } {
  const lower = asciiLower(html);
  let title = "";
  {
    const { find, tag } = finder(lower);
    const open = tag("title", 0);
    const gt = open < 0 ? -1 : find(">", open);
    const close = gt < 0 ? -1 : tag("title", gt, true);
    if (close >= 0) title = decode(html.slice(gt + 1, close)).replace(/\s+/g, " ").trim();
  }
  // The article when the page marks one (its first opening tag to the last closing one); else the body.
  let [start, end] = [0, html.length];
  const region = (name: string): [number, number, number] | undefined => {
    const { find, tag } = finder(lower);
    const open = tag(name, 0);
    const gt = open < 0 ? -1 : find(">", open);
    const close = gt < 0 ? -1 : lastClose(lower, name, gt);
    return close < 0 ? undefined : [open, gt + 1, close];
  };
  const main = [region("main"), region("article")].filter((r) => r).sort((a, b) => a![0] - b![0])[0] ?? region("body");
  if (main) [, start, end] = main;

  const { find, tag } = finder(lower);
  const out: string[] = [];
  let pos = start;
  while (pos < end) {
    const lt = find("<", pos);
    if (lt < 0 || lt >= end) {
      out.push(html.slice(pos, end));
      break;
    }
    out.push(html.slice(pos, lt));
    if (lower.startsWith("<!--", lt)) {
      const c = find("-->", lt + 4);
      pos = c < 0 ? end : c + 3;
      out.push(" ");
      continue;
    }
    const closing = lower[lt + 1] === "/";
    let k = lt + (closing ? 2 : 1);
    const first = lower.charCodeAt(k);
    // "a < b", "<<<": not a tag, the "<" is text. "<!doctype", "<?xml": markup up to ">".
    if (!(first >= 97 && first <= 122) && lower[k] !== "!" && lower[k] !== "?") {
      out.push("<");
      pos = lt + 1;
      continue;
    }
    if (!(first >= 97 && first <= 122)) k++;
    while (k < end && nameChar(lower.charCodeAt(k))) k++;
    // Exact names: "<main-nav>" is not <main>, "<script<" is no script.
    const name = endsName(lower[k]) ? lower.slice(lt + (closing ? 2 : 1), k) : "";
    const gt = find(">", k);
    if (gt < 0 || gt >= end) break;
    pos = gt + 1;
    if (!closing && DROPPED.has(name)) {
      const close = tag(name, pos, true);
      if (close >= 0 && close < end) {
        const cgt = find(">", close);
        pos = cgt < 0 || cgt >= end ? end : cgt + 1;
      }
      out.push(" ");
      continue;
    }
    out.push(name === "br" || (closing && BLOCKS.has(name)) ? "\n" : " ");
  }
  const text = decode(out.join(""))
    .replace(/[ \t\r\f\v]+/g, " ")
    .replace(/ ?\n ?/g, "\n")
    .replace(/\n{2,}/g, "\n")
    .trim();
  return { title, text };
}

/** A text file as the model should read it: a Markdown file's HTML (logos, badges, centred headings) is layout, not text. */
function plainText(body: string): string {
  const lower = asciiLower(body);
  const { find } = finder(lower);
  const out: string[] = [];
  let pos = 0;
  for (;;) {
    const lt = find("<", pos);
    if (lt < 0) break;
    out.push(body.slice(pos, lt));
    pos = lt;
    if (lower.startsWith("<!--", lt)) {
      const c = find("-->", lt + 4);
      // An unclosed comment stays, as text.
      if (c >= 0) pos = c + 3;
      else (out.push("<"), (pos = lt + 1));
      continue;
    }
    let k = lt + (lower[lt + 1] === "/" ? 2 : 1);
    const from = k;
    while (k < lower.length && nameChar(lower.charCodeAt(k))) k++;
    const gt = LAYOUT.has(lower.slice(from, k)) && endsName(lower[k]) ? find(">", k) : -1;
    if (gt >= 0) pos = gt + 1;
    else (out.push("<"), (pos = lt + 1));
  }
  out.push(body.slice(pos));
  // Spaces before a line end go; at most one empty line in a row. Line by line, no backtracking.
  const lines = out.join("").split("\n").map((line) => {
    let e = line.length;
    while (e > 0 && (line[e - 1] === " " || line[e - 1] === "\t")) e--;
    return line.slice(0, e);
  });
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

// ---- which addresses may be read ----------------------------------------------------------------

function parseV4(h: string): number[] | undefined {
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(h);
  const b = m?.slice(1).map(Number);
  return b && b.every((x) => x <= 255) ? b : undefined;
}

/** The eight 16-bit groups of an IPv6 address (a trailing dotted IPv4 allowed), or undefined. */
function parseV6(text: string): number[] | undefined {
  let h = text.replace(/%.*$/, "");
  let tail: number[] = [];
  const dotted = /(\d+\.\d+\.\d+\.\d+)$/.exec(h);
  if (dotted) {
    const b = parseV4(dotted[1]!);
    if (!b) return undefined;
    tail = [(b[0]! << 8) | b[1]!, (b[2]! << 8) | b[3]!];
    h = h.slice(0, dotted.index);
    if (h.endsWith(":") && !h.endsWith("::")) h = h.slice(0, -1);
  }
  const halves = h.split("::");
  if (halves.length > 2) return undefined;
  const groups = (s: string) => (s ? s.split(":").map((g) => (/^[0-9a-f]{1,4}$/i.test(g) ? parseInt(g, 16) : NaN)) : []);
  const head = groups(halves[0]!);
  const rest = halves.length === 2 ? groups(halves[1]!) : [];
  const room = 8 - tail.length - head.length - rest.length;
  if (room < 0 || (halves.length === 1 && room !== 0)) return undefined;
  const words = [...head, ...Array<number>(room).fill(0), ...rest, ...tail];
  return words.some((w) => Number.isNaN(w)) ? undefined : words;
}

function isPrivateV4([a, b]: number[]): boolean {
  return (
    a === 0 || a === 10 || a === 127 || (a === 169 && b === 254) || (a === 172 && b! >= 16 && b! <= 31) || (a === 192 && b === 168) || (a === 100 && b! >= 64 && b! <= 127) ||
    // Benchmarking (198.18/15); multicast, reserved and broadcast (224/3).
    (a === 198 && (b === 18 || b === 19)) || a! >= 224
  );
}

function isPrivateV6(w: number[]): boolean {
  const zeros = (n: number) => w.slice(0, n).every((x) => x === 0);
  if (zeros(7) && w[7]! <= 1) return true; // :: and ::1
  const top = w[0]!;
  if ((top & 0xfe00) === 0xfc00 || (top & 0xffc0) === 0xfe80 || (top & 0xffc0) === 0xfec0 || (top & 0xff00) === 0xff00) return true;
  // An IPv4 address carried inside: mapped (::ffff:a.b.c.d), compatible (::a.b.c.d), NAT64 (64:ff9b::/96), 6to4 (2002::/16).
  const v4 = (hi: number, lo: number) => isPrivateV4([hi >> 8, hi & 255, lo >> 8, lo & 255]);
  if ((zeros(5) && w[5] === 0xffff) || zeros(6)) return v4(w[6]!, w[7]!);
  if (top === 0x64 && w[1] === 0xff9b && w.slice(2, 6).every((x) => x === 0)) return v4(w[6]!, w[7]!);
  if (top === 0x2002) return v4(w[1]!, w[2]!);
  return false;
}

/** True for an IP address (as DNS gives it) on this machine or a private network. */
export function isPrivateAddress(address: string): boolean {
  const h = address.replace(/^\[|\]$/g, "").toLowerCase();
  const v4 = parseV4(h);
  if (v4) return isPrivateV4(v4);
  const v6 = h.includes(":") ? parseV6(h) : undefined;
  return v6 ? isPrivateV6(v6) : false;
}

/** True for addresses on this machine or a private network, as far as the URL's host text tells. */
function isPrivate(url: URL): boolean {
  // "localhost." is localhost: a trailing dot only marks the name as absolute.
  const h = url.hostname.replace(/^\[|\]$/g, "").toLowerCase().replace(/\.+$/, "");
  if (!h) return true;
  if (h.includes(":")) return parseV6(h) ? isPrivateV6(parseV6(h)!) : true;
  if (h === "localhost" || h.endsWith(".localhost") || h.endsWith(".local") || h.endsWith(".internal") || !h.includes(".")) return true;
  const v4 = parseV4(h);
  return v4 ? isPrivateV4(v4) : false;
}

/** Node.js's resolver, when there is one (a bundle for browsers must not import node:dns). */
function nodeLookup(): SkillHooks["lookup"] {
  try {
    const get = (globalThis as { process?: { getBuiltinModule?(id: string): unknown } }).process?.getBuiltinModule;
    const dns = get?.("node:dns") as { promises?: { lookup(host: string, o: { all: true }): Promise<{ address: string }[]> } } | undefined;
    const lookup = dns?.promises?.lookup;
    return lookup ? (host) => lookup(host, { all: true }) : undefined;
  } catch {
    return undefined;
  }
}

/** A web page's text. In a browser it reads only sites that allow other pages to (CORS); in Node.js any. */
export function webPageSkill(opts: SkillOptions = {}): RunnableTool {
  const hooks = opts as SkillHooks;
  const max = opts.maxChars ?? 2000;
  // The page Kiln runs in may trust requests from itself: the model is not to read it.
  const own = (u: URL) => {
    const origin = (globalThis as { location?: { origin?: string } }).location?.origin;
    return !!origin && /^https?:/.test(origin) && u.origin === origin;
  };
  const allow = opts.allowUrl ?? ((u: URL) => !isPrivate(u) && !own(u));
  const lookup = opts.allowUrl ? undefined : (hooks.lookup ?? nodeLookup());
  const allowed = (u: URL): URL => {
    if (u.protocol !== "https:" && u.protocol !== "http:") throw new Error("only http and https pages can be read");
    if (u.username || u.password) throw new Error("addresses with a user name or password are not read");
    if (!allow(u)) throw new Error(`reading ${u.hostname} is not allowed`);
    return u;
  };
  /**
   * The host text can hide a private address ("127.0.0.1.nip.io"); where names can be resolved, the
   * addresses are checked too. The name can still resolve elsewhere by the time of the request
   * (DNS rebinding): only connecting to the checked address would close that.
   */
  const resolves = async (u: URL) => {
    const host = u.hostname.replace(/^\[|\]$/g, "");
    if (!lookup || parseV4(host) || host.includes(":")) return;
    let found: { address: string }[];
    try {
      found = await lookup(host);
    } catch {
      return; // fetch will say the name does not resolve
    }
    if (found.some((a) => isPrivateAddress(a.address))) throw new Error(`reading ${u.hostname} is not allowed (it leads to a private address)`);
  };
  return skill(
    "read_web_page",
    "Fetch a web page and get its text. Use it when the user gives a link, or to read a page found another way.",
    { url: { type: "string", description: "The page's full address, starting with https://" } },
    ["url"],
    async ({ url }, context?: ToolContext) => {
      let u: URL;
      try {
        u = new URL(/^[a-z][a-z0-9+.-]*:/i.test(String(url)) ? String(url) : `https://${String(url)}`);
      } catch {
        throw new Error(`${JSON.stringify(url)} is not a web address`);
      }
      allowed(u);
      const signal = context?.signal;
      // A Wikipedia article: its text from Wikipedia's API, which any page may read (the article's own address is closed to browsers).
      const wiki = /^([a-z0-9-]{2,20})(?:\.m)?\.wikipedia\.org$/.exec(u.hostname);
      if (wiki && u.pathname.startsWith("/wiki/") && u.pathname.length > 6) {
        const api = `https://${wiki[1]}.wikipedia.org/w/api.php?action=query&format=json&origin=*&prop=extracts&explaintext=1&redirects=1&titles=${encodeURIComponent(decodeURIComponent(u.pathname.slice(6)).replace(/_/g, " "))}`;
        const res = await get(hooks, api, { what: "Wikipedia", headers: wikiHeaders(opts), signal });
        const json = JSON.parse(await readText(res, "Wikipedia")) as { query?: { pages?: Record<string, { title: string; extract?: string }> } };
        const article = Object.values(json.query?.pages ?? {}).find((p) => p.extract?.trim());
        if (!article) throw new Error(`Wikipedia (${wiki[1]}) has no article at ${u.href}; try search_wikipedia`);
        return { url: u.href, title: article.title, text: clip(article.extract!.trim(), max) };
      }
      // Redirects one at a time, each target checked before it is asked for.
      let at = u;
      let res: Response;
      for (let hop = 0; ; hop++) {
        await resolves(at);
        res = await get(hooks, at.href, { what: at.hostname, signal, redirect: "manual" });
        if (res.type === "opaqueredirect") {
          // A browser does not say where a manual redirect leads: let it follow, then check where it ended.
          res = await get(hooks, at.href, { what: at.hostname, signal });
          break;
        }
        const location = isRedirect(res) ? res.headers.get("location") : null;
        if (!location) break;
        void res.body?.cancel().catch(() => {});
        if (hop === 5) throw new Error(`${u.href} redirects too many times`);
        at = allowed(new URL(location, at));
      }
      // Where the page ended up must be allowed as well.
      if (res.url && !allow(new URL(res.url))) throw new Error(`reading ${new URL(res.url).hostname} is not allowed`);
      const type = res.headers.get("content-type") ?? "";
      if (type && !/text\/|json|xml/i.test(type)) throw new Error(`${at.href} is not a text page (${type.split(";")[0]})`);
      const body = await readText(res, at.hostname);
      const page = /html/i.test(type) || /^\s*<(!doctype|html)/i.test(body) ? htmlToText(body) : { title: "", text: plainText(body) };
      if (!page.text) throw new Error(`${at.href} has no readable text (it may be drawn by scripts)`);
      return { url: res.url || at.href, ...(page.title ? { title: page.title } : {}), text: clip(page.text, max) };
    },
  );
}

/**
 * A system message for a chat with skills. Small models otherwise ask the user for what a skill
 * would tell them, or answer from memory. *
 * Not for models that bring a tool prompt of their own: Hermes 3 does worse with this in front of it.
 */
export const SKILL_INSTRUCTIONS =
  "You are a helpful assistant with tools. When a tool can find or compute what is asked, call it right away instead of asking the user or answering from memory: calculate for any arithmetic, get_date_time for the date or the time anywhere, search_wikipedia for facts, read_web_page for a link. Then answer briefly from the tool's result. If no tool fits, just answer.";

/**
 * Kiln's built-in skills, ready for `runTools`: `calculate`, `get_date_time`, `search_wikipedia` and
 * `read_web_page`. The first two work offline; the last two send the model's query or address to
 * Wikipedia or the page's site, and nothing else.
 */
export function skills(opts: SkillOptions = {}): RunnableTool[] {
  return [calculatorSkill(), dateTimeSkill(), wikipediaSkill(opts), webPageSkill(opts)];
}
