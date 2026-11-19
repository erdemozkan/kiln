// Skills: tools that come with Kiln, ready for `runTools`. Each does real work: exact arithmetic,
// the clock, Wikipedia, a web page's text. The last two go out to the network.
import type { RunnableTool } from "./tools";

export interface SkillOptions {
  /** The `fetch` the network skills use (default: the global one). */
  fetch?: typeof fetch;
  /** Most characters of text a network skill hands the model (default 2000): it all goes into the context. */
  maxChars?: number;
  /** The Wikipedia edition `search_wikipedia` reads when the model names none (default "en"). */
  language?: string;
  /**
   * Decides whether `read_web_page` may fetch a URL. The default refuses this machine and private
   * networks (localhost, 10.x, 192.168.x …): the model picks the address, and a page it read can
   * suggest one.
   */
  allowUrl?(url: URL): boolean;
}

const skill = (name: string, description: string, properties: Record<string, unknown>, required: string[], execute: RunnableTool["execute"]): RunnableTool => ({
  type: "function",
  function: { name, description, parameters: { type: "object", properties, required } },
  execute,
});

// ---- calculate ------------------------------------------------------------------------------

const FUNCTIONS: Record<string, (...a: number[]) => number> = {
  sqrt: Math.sqrt,
  abs: Math.abs,
  round: Math.round,
  floor: Math.floor,
  ceil: Math.ceil,
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  ln: Math.log,
  log: Math.log10,
  exp: Math.exp,
  min: Math.min,
  max: Math.max,
  pow: Math.pow,
};
const CONSTANTS: Record<string, number> = { pi: Math.PI, e: Math.E };

/**
 * The value of an arithmetic expression: numbers, + - * / ^ (or **), % (remainder), parentheses, the
 * functions above and pi / e. A parser of its own: the text comes from a model, so it is never run as code.
 */
export function evaluate(expression: string): number {
  // "1,234" and "×" are how people (and models) write numbers and products.
  const src = expression.replace(/(\d),(?=\d{3}(\D|$))/g, "$1").replace(/[×·]/g, "*").replace(/÷/g, "/").replace(/−/g, "-");
  let i = 0;
  const fail = (what: string): never => {
    throw new Error(`cannot calculate ${JSON.stringify(expression)}: ${what}`);
  };
  const skip = () => {
    while (src[i] === " ") i++;
  };
  const eat = (s: string): boolean => {
    skip();
    if (!src.startsWith(s, i)) return false;
    i += s.length;
    return true;
  };
  const primary = (): number => {
    skip();
    if (eat("(")) {
      const v = sum();
      if (!eat(")")) fail("a ) is missing");
      return v;
    }
    const num = /^(\d+\.?\d*|\.\d+)(e[+-]?\d+)?/i.exec(src.slice(i));
    if (num) {
      i += num[0].length;
      return Number(num[0]);
    }
    const word = /^[a-z_]+\d*/i.exec(src.slice(i));
    if (word) {
      i += word[0].length;
      const name = word[0].toLowerCase();
      if (eat("(")) {
        const f = FUNCTIONS[name] ?? fail(`there is no function ${name}`);
        const args: number[] = [];
        if (!eat(")")) {
          do args.push(sum());
          while (eat(","));
          if (!eat(")")) fail("a ) is missing");
        }
        return f(...args);
      }
      return CONSTANTS[name] ?? fail(`${word[0]} is not a number`);
    }
    return fail(i >= src.length ? "it ends too early" : `unexpected ${JSON.stringify(src[i])}`);
  };
  const unary = (): number => (eat("-") ? -unary() : eat("+") ? unary() : power());
  // Right-associative, and tighter than a sign before it: -2^2 is -4.
  function power(): number {
    const base = primary();
    return eat("**") || eat("^") ? base ** unary() : base;
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
  // 0.1 + 0.2 reads as 0.3: float noise past 12 digits would only mislead the model.
  return Number(v.toPrecision(12));
}

/** Exact arithmetic: models guess at sums they should compute. */
export function calculatorSkill(): RunnableTool {
  return skill(
    "calculate",
    "Calculate an arithmetic expression exactly. Use it for any arithmetic instead of working it out yourself.",
    // No worked example here: a small model copies it instead of writing the user's sum.
    { expression: { type: "string", description: "The arithmetic to compute, written with numbers, + - * / ^ and parentheses. Functions: sqrt abs round floor ceil sin cos tan ln log exp min max pow. Constants: pi e." } },
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
};

/** The IANA time zone for a zone name ("Europe/Paris") or a city ("Paris", "new york"); undefined when unknown. */
export function timeZoneOf(place: string): string | undefined {
  const valid = (zone: string): string | undefined => {
    try {
      return new Intl.DateTimeFormat("en", { timeZone: zone }).resolvedOptions().timeZone;
    } catch {
      return undefined;
    }
  };
  if (place.includes("/")) return valid(place.trim().replace(/ /g, "_"));
  const k = key(place);
  if (!k) return undefined;
  if (ZONE_OF[k]) return ZONE_OF[k];
  const zones = (Intl as { supportedValuesOf?(k: string): string[] }).supportedValuesOf?.("timeZone") ?? [];
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
  const head = text.slice(0, max);
  const stop = Math.max(head.lastIndexOf(". "), head.lastIndexOf("\n"));
  return `${(stop > max * 0.6 ? head.slice(0, stop + 1) : head.slice(0, Math.max(head.lastIndexOf(" "), 1))).trimEnd()} […]`;
}

async function get(f: typeof fetch, url: string, what: string): Promise<Response> {
  let res: Response;
  try {
    res = await f(url, { headers: { Accept: "text/html,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.1" } });
  } catch (err) {
    // In a browser this is also what a site that forbids other pages to read it looks like.
    throw new Error(`could not reach ${what} (${err instanceof Error ? err.message : String(err)}); the site may be down or may not allow being read from here`);
  }
  if (!res.ok) throw new Error(`${what} answered ${res.status}`);
  return res;
}

/** Wikipedia's article for a search: what a model was never trained on, or half remembers. */
export function wikipediaSkill(opts: SkillOptions = {}): RunnableTool {
  const max = opts.maxChars ?? 2000;
  return skill(
    "search_wikipedia",
    "Look something up on Wikipedia and get the start of the best matching article. Use it for facts about people, places, events and things.",
    { query: { type: "string", description: "What to look up: a name or a few words" }, language: { type: "string", pattern: "[a-z]{2,3}", description: 'The Wikipedia edition, e.g. "en", "tr", "de" (default "en")' } },
    ["query"],
    async ({ query, language }) => {
      const lang = typeof language === "string" && /^[a-z]{2,3}$/.test(language) ? language : (opts.language ?? "en");
      const q = String(query ?? "").trim();
      if (!q) throw new Error("the query is empty");
      const api = `https://${lang}.wikipedia.org/w/api.php?action=query&format=json&origin=*&generator=search&gsrlimit=3&gsrsearch=${encodeURIComponent(q)}&prop=extracts%7Cinfo&exintro=1&explaintext=1&inprop=url&redirects=1`;
      const json = (await (await get(opts.fetch ?? fetch, api, "Wikipedia")).json()) as { query?: { pages?: Record<string, { index: number; title: string; extract?: string; fullurl?: string }> } };
      const pages = Object.values(json.query?.pages ?? {}).sort((a, b) => a.index - b.index);
      const page = pages.find((p) => p.extract?.trim());
      if (!page) throw new Error(`Wikipedia (${lang}) has no article for ${JSON.stringify(q)}`);
      return { title: page.title, url: page.fullurl, text: clip(page.extract!.trim(), max), ...(pages.length > 1 ? { other_matches: pages.filter((p) => p !== page).map((p) => p.title) } : {}) };
    },
  );
}

const ENTITIES: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", ndash: "–", mdash: "—", hellip: "…", rsquo: "’", lsquo: "‘", rdquo: "”", ldquo: "“", copy: "©", deg: "°" };

/** A page's readable text: its title, and its body without scripts, styles, navigation and markup. */
export function htmlToText(html: string): { title: string; text: string } {
  const decode = (s: string) =>
    s.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+\d*);/gi, (m, e: string) => {
      if (e[0] !== "#") return ENTITIES[e.toLowerCase()] ?? m;
      const code = e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
      return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : m;
    });
  const title = decode(/<title[^>]*>([\s\S]*?)<\/title>/i.exec(html)?.[1] ?? "").replace(/\s+/g, " ").trim();
  // The article when the page marks one; else the body.
  const main = /<(main|article)\b[^>]*>([\s\S]*)<\/\1>/i.exec(html)?.[2] ?? /<body\b[^>]*>([\s\S]*)<\/body>/i.exec(html)?.[1] ?? html;
  const text = decode(
    main
      .replace(/<!--[\s\S]*?-->/g, " ")
      .replace(/<(script|style|noscript|template|svg|nav|header|footer|aside|form|iframe|head)\b[\s\S]*?<\/\1>/gi, " ")
      .replace(/<(br|\/p|\/div|\/h[1-6]|\/li|\/tr|\/section|\/article|\/blockquote)\b[^>]*>/gi, "\n")
      .replace(/<[^>]+>/g, " "),
  )
    .replace(/[ \t\r\f\v]+/g, " ")
    .replace(/ ?\n ?/g, "\n")
    .replace(/\n{2,}/g, "\n")
    .trim();
  return { title, text };
}

/** A text file as the model should read it: a Markdown file's HTML (logos, badges, centred headings) is layout, not text. */
function plainText(body: string): string {
  return body
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<\/?(h[1-6]|img|p|a|div|br|hr|picture|source|span|b|i|em|strong|sub|sup|center|table|thead|tbody|tr|td|th|details|summary|kbd)\b[^>]*>/gi, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** True for addresses on this machine or a private network. */
function isPrivate(url: URL): boolean {
  const h = url.hostname.replace(/^\[|\]$/g, "").toLowerCase();
  if (h === "localhost" || h.endsWith(".localhost") || h.endsWith(".local") || h.endsWith(".internal") || !h.includes(".") && !h.includes(":")) return true;
  if (h.includes(":")) return h === "::1" || h === "::" || /^f[cd]/.test(h) || /^fe[89ab]/.test(h) || h.startsWith("::ffff:");
  const ip = /^(\d+)\.(\d+)\.(\d+)\.(\d+)$/.exec(h);
  if (!ip) return false;
  const [a, b] = [Number(ip[1]), Number(ip[2])];
  return a === 0 || a === 10 || a === 127 || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127);
}

/** A web page's text. In a browser it reads only sites that allow other pages to (CORS); in Node.js any. */
export function webPageSkill(opts: SkillOptions = {}): RunnableTool {
  const max = opts.maxChars ?? 2000;
  const allow = opts.allowUrl ?? ((u: URL) => !isPrivate(u));
  return skill(
    "read_web_page",
    "Fetch a web page and get its text. Use it when the user gives a link, or to read a page found another way.",
    { url: { type: "string", description: "The page's full address, starting with https://" } },
    ["url"],
    async ({ url }) => {
      let u: URL;
      try {
        u = new URL(/^[a-z][a-z0-9+.-]*:/i.test(String(url)) ? String(url) : `https://${String(url)}`);
      } catch {
        throw new Error(`${JSON.stringify(url)} is not a web address`);
      }
      if (u.protocol !== "https:" && u.protocol !== "http:") throw new Error("only http and https pages can be read");
      if (u.username || u.password) throw new Error("addresses with a user name or password are not read");
      if (!allow(u)) throw new Error(`reading ${u.hostname} is not allowed`);
      // A Wikipedia article: its text from Wikipedia's API, which any page may read (the article's own address is closed to browsers).
      const wiki = /^([a-z-]+)(?:\.m)?\.wikipedia\.org$/.exec(u.hostname);
      if (wiki && u.pathname.startsWith("/wiki/") && u.pathname.length > 6) {
        const api = `https://${wiki[1]}.wikipedia.org/w/api.php?action=query&format=json&origin=*&prop=extracts&explaintext=1&redirects=1&titles=${encodeURIComponent(decodeURIComponent(u.pathname.slice(6)).replace(/_/g, " "))}`;
        const json = (await (await get(opts.fetch ?? fetch, api, "Wikipedia")).json()) as { query?: { pages?: Record<string, { title: string; extract?: string }> } };
        const article = Object.values(json.query?.pages ?? {}).find((p) => p.extract?.trim());
        if (!article) throw new Error(`Wikipedia (${wiki[1]}) has no article at ${u.href}; try search_wikipedia`);
        return { url: u.href, title: article.title, text: clip(article.extract!.trim(), max) };
      }
      const res = await get(opts.fetch ?? fetch, u.href, u.hostname);
      // Where the page ended up after redirects must be allowed as well.
      if (res.url && !allow(new URL(res.url))) throw new Error(`reading ${new URL(res.url).hostname} is not allowed`);
      const type = res.headers.get("content-type") ?? "";
      if (type && !/text\/|json|xml/i.test(type)) throw new Error(`${u.href} is not a text page (${type.split(";")[0]})`);
      const body = await res.text();
      const page = /html/i.test(type) || /^\s*<(!doctype|html)/i.test(body) ? htmlToText(body) : { title: "", text: plainText(body) };
      if (!page.text) throw new Error(`${u.href} has no readable text (it may be drawn by scripts)`);
      return { url: res.url || u.href, ...(page.title ? { title: page.title } : {}), text: clip(page.text, max) };
    },
  );
}

/**
 * A system message for a chat with skills. Small models otherwise ask the user for what a skill
 * would tell them, or answer from memory.
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
