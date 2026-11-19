import { afterEach, describe, expect, it, vi } from "vitest";
import { compileFormat } from "@kiln/grammar";
import { calculatorSkill, dateTimeSkill, evaluate, htmlToText, isPrivateAddress, skills, timeZoneOf, webPageSkill as webPageSkillWithDns, wikipediaSkill, type SkillHooks } from "../src/skills";

// Never real DNS in tests: every name resolves to a public address unless a test says otherwise.
const webPageSkill = (opts: SkillHooks = {}) => webPageSkillWithDns({ lookup: async () => [{ address: "93.184.215.14" }], ...opts } as SkillHooks);
type Init = RequestInit & { headers: Record<string, string>; signal: AbortSignal };

const respond = (body: string, type = "text/html; charset=utf-8", url = "") => {
  const r = new Response(body, { headers: { "content-type": type } });
  if (url) Object.defineProperty(r, "url", { value: url });
  return r;
};

describe("calculate", () => {
  it("follows precedence, signs, powers and functions", () => {
    expect(evaluate("1234 * 5678")).toBe(7006652);
    expect(evaluate("2 + 3 * 4")).toBe(14);
    expect(evaluate("(2 + 3) * 4")).toBe(20);
    expect(evaluate("-2 ^ 2")).toBe(-4);
    expect(evaluate("2 ^ 3 ^ 2")).toBe(512);
    expect(evaluate("2 ** -1")).toBe(0.5);
    expect(evaluate("0.1 + 0.2")).toBe(0.3);
    expect(evaluate("sqrt(16) + max(1, 7, 3) - abs(-2)")).toBe(9);
    expect(evaluate("round(2 * pi * 10)")).toBe(63);
    expect(evaluate("17 % 5")).toBe(2);
    expect(evaluate("1,234,567 × 2")).toBe(2469134);
    expect(evaluate("1e3 / 4")).toBe(250);
    expect(evaluate("2022 - 1998")).toBe(24);
  });

  it("refuses what is not arithmetic, without running it", () => {
    for (const bad of ["", "2 +", "(1 + 2", "1 / 0", "alert(1)", "process.exit()", "x + 1", "2 2", "constructor", "1; 2"]) {
      expect(() => evaluate(bad), bad).toThrow(/cannot calculate/);
    }
    expect(calculatorSkill().execute({ expression: "6 * 7" })).toEqual({ expression: "6 * 7", result: 42 });
  });
});

describe("get_date_time", () => {
  const at = () => new Date("2026-03-05T22:30:00Z");

  it("finds a zone for a city or a zone name", () => {
    expect(timeZoneOf("Tokyo")).toBe("Asia/Tokyo");
    expect(timeZoneOf("new york")).toBe("America/New_York");
    expect(timeZoneOf("São Paulo")).toBe("America/Sao_Paulo");
    expect(timeZoneOf("Europe/Paris")).toBe("Europe/Paris");
    expect(timeZoneOf("Ankara")).toBe("Europe/Istanbul");
    expect(timeZoneOf("Atlantis")).toBeUndefined();
    expect(timeZoneOf("Mars/Olympus")).toBeUndefined();
  });

  it("gives the date and time there", () => {
    const t = dateTimeSkill(at);
    expect(t.execute({ place: "Tokyo" })).toEqual({ place: "Tokyo", date: "2026-03-06", weekday: "Friday", time: "07:30", time_zone: "Asia/Tokyo", utc_offset: "GMT+09:00" });
    expect(t.execute({ place: "New York" })).toMatchObject({ date: "2026-03-05", weekday: "Thursday", time: "17:30", utc_offset: "GMT-05:00" });
    expect(t.execute({ place: "UTC" })).toMatchObject({ time: "22:30", utc_offset: "GMT+00:00" });
    expect(t.execute({})).toMatchObject({ date: expect.stringMatching(/^2026-03-0[56]$/) });
    expect(() => t.execute({ place: "Atlantis" })).toThrow(/no time zone known/);
  });
});

describe("search_wikipedia", () => {
  it("returns the best match's introduction, cut to size", async () => {
    const asked: string[] = [];
    const pages = {
      "7": { index: 2, title: "Ada (name)", extract: "A name.", fullurl: "https://en.wikipedia.org/wiki/Ada_(name)" },
      "5": { index: 1, title: "Ada Lovelace", extract: `Ada Lovelace was a mathematician. ${"She wrote notes. ".repeat(40)}`, fullurl: "https://en.wikipedia.org/wiki/Ada_Lovelace" },
    };
    const fetch = (async (url: string) => (asked.push(url), respond(JSON.stringify({ query: { pages } }), "application/json"))) as unknown as typeof globalThis.fetch;
    const r = (await wikipediaSkill({ fetch, maxChars: 120 }).execute({ query: "Ada Lovelace & co", language: "tr" })) as { title: string; url: string; text: string; other_matches: string[] };
    expect(asked[0]).toContain("https://tr.wikipedia.org/w/api.php?");
    expect(asked[0]).toContain("gsrsearch=Ada%20Lovelace%20%26%20co");
    expect(asked[0]).toContain("origin=*");
    expect(r.title).toBe("Ada Lovelace");
    expect(r.url).toBe("https://en.wikipedia.org/wiki/Ada_Lovelace");
    expect(r.text.length).toBeLessThanOrEqual(126);
    expect(r.text.endsWith(". […]")).toBe(true);
    expect(r.other_matches).toEqual(["Ada (name)"]);
    // A language that is not a language code never reaches the host name.
    await wikipediaSkill({ fetch }).execute({ query: "x", language: "evil.example/" });
    expect(asked[1]).toContain("https://en.wikipedia.org/");
  });

  it("says so when nothing matches or the site fails", async () => {
    const none = (async () => respond("{}", "application/json")) as unknown as typeof globalThis.fetch;
    await expect(wikipediaSkill({ fetch: none }).execute({ query: "zzzz" })).rejects.toThrow(/no article/);
    const down = (async () => new Response("", { status: 503 })) as unknown as typeof globalThis.fetch;
    await expect(wikipediaSkill({ fetch: down }).execute({ query: "x" })).rejects.toThrow(/answered 503/);
    // Refused for too many requests: one more try, with the application's name when one is given.
    const seen: (string | undefined)[] = [];
    let n = 0;
    const busy = (async (_url: string, init: { headers: Record<string, string> }) => {
      seen.push(init.headers["Api-User-Agent"]);
      return n++ === 0 ? new Response("", { status: 429, headers: { "retry-after": "0" } }) : respond(JSON.stringify({ query: { pages: { "1": { index: 1, title: "X", extract: "An x." } } } }), "application/json");
    }) as unknown as typeof globalThis.fetch;
    expect(await wikipediaSkill({ fetch: busy, userAgent: "my-app/1.0" }).execute({ query: "x" })).toMatchObject({ title: "X" });
    expect(seen).toEqual(["my-app/1.0", "my-app/1.0"]);
    await wikipediaSkill({ fetch: busy }).execute({ query: "x" });
    expect(seen[2]).toBe("Rectus");
    const refusing = (async () => new Response("", { status: 429, headers: { "retry-after": "0" } })) as unknown as typeof globalThis.fetch;
    await expect(wikipediaSkill({ fetch: refusing }).execute({ query: "x" })).rejects.toThrow(/refusing requests for now/);
    const blocked = (async () => Promise.reject(new TypeError("Failed to fetch"))) as unknown as typeof globalThis.fetch;
    await expect(wikipediaSkill({ fetch: blocked }).execute({ query: "x" })).rejects.toThrow(/could not reach Wikipedia/);
  });
});

describe("read_web_page", () => {
  const HTML = `<!doctype html><html><head><title>Tea &amp; Co</title><style>p{color:red}</style><script>var x = "<p>no</p>";</script></head>
    <body><nav><a href="/">Home</a></nav><main><h1>Green tea</h1><p>Brew at 80&nbsp;&deg;C for 2&#8211;3 minutes.</p><!-- hidden --><ul><li>Sencha</li><li>Matcha</li></ul></main><footer>© nobody</footer></body></html>`;

  it("keeps the article's text and drops markup, scripts and navigation", () => {
    expect(htmlToText(HTML)).toEqual({ title: "Tea & Co", text: "Green tea\nBrew at 80 °C for 2–3 minutes.\nSencha\nMatcha" });
  });

  it("fetches a page, adding https, and returns its text", async () => {
    const asked: string[] = [];
    const fetch = (async (url: string) => (asked.push(url), respond(HTML, "text/html", "https://example.com/tea"))) as unknown as typeof globalThis.fetch;
    const r = await webPageSkill({ fetch }).execute({ url: "example.com/tea" });
    expect(asked).toEqual(["https://example.com/tea"]);
    expect(r).toEqual({ url: "https://example.com/tea", title: "Tea & Co", text: "Green tea\nBrew at 80 °C for 2–3 minutes.\nSencha\nMatcha" });
    const plain = (async () => respond("just text", "text/plain")) as unknown as typeof globalThis.fetch;
    expect(await webPageSkill({ fetch: plain }).execute({ url: "https://example.com/a.txt" })).toMatchObject({ text: "just text" });
    const md = (async () => respond('<h1 align="center">\n  <img src="logo.svg" alt="Kiln">\n</h1>\n\n# Kiln\n\nRuns `Array<string>` models.\n', "text/plain")) as unknown as typeof globalThis.fetch;
    expect(await webPageSkill({ fetch: md }).execute({ url: "https://example.com/README.md" })).toMatchObject({ text: "# Kiln\n\nRuns `Array<string>` models." });
  });

  it("reads a Wikipedia article through Wikipedia's API", async () => {
    const asked: string[] = [];
    const fetch = (async (url: string) => (asked.push(url), respond(JSON.stringify({ query: { pages: { "1": { title: "Kapadokya", extract: "Kapadokya tarihî bir bölgedir." } } } }), "application/json"))) as unknown as typeof globalThis.fetch;
    const r = await webPageSkill({ fetch }).execute({ url: "https://tr.m.wikipedia.org/wiki/G%C3%B6reme_Vadisi" });
    expect(asked).toEqual([expect.stringContaining("https://tr.wikipedia.org/w/api.php?")]);
    expect(asked[0]).toContain("titles=G%C3%B6reme%20Vadisi");
    expect(asked[0]).toContain("origin=*");
    expect(r).toEqual({ url: "https://tr.m.wikipedia.org/wiki/G%C3%B6reme_Vadisi", title: "Kapadokya", text: "Kapadokya tarihî bir bölgedir." });
    const none = (async () => respond("{}", "application/json")) as unknown as typeof globalThis.fetch;
    await expect(webPageSkill({ fetch: none }).execute({ url: "https://en.wikipedia.org/wiki/Nope" })).rejects.toThrow(/no article/);
  });

  it("refuses this machine, private networks, other schemes and files that are not text", async () => {
    let fetched = 0;
    const fetch = (async () => (fetched++, respond("x"))) as unknown as typeof globalThis.fetch;
    const page = webPageSkill({ fetch });
    for (const url of ["http://localhost:8080/", "http://127.0.0.1/", "http://192.168.1.10/admin", "http://10.0.0.1/", "http://[::1]/", "http://169.254.169.254/latest/meta-data", "http://intranet/", "http://printer.local/"]) {
      await expect(page.execute({ url }), url).rejects.toThrow(/not allowed/);
    }
    await expect(page.execute({ url: "file:///etc/passwd" })).rejects.toThrow(/only http and https/);
    await expect(page.execute({ url: "https://user:pw@example.com/" })).rejects.toThrow(/user name or password/);
    expect(fetched).toBe(0);
    // A public address that redirects inside is refused too.
    const inward = (async () => respond("secret", "text/html", "http://192.168.1.1/")) as unknown as typeof globalThis.fetch;
    await expect(webPageSkill({ fetch: inward }).execute({ url: "https://example.com/" })).rejects.toThrow(/192\.168\.1\.1 is not allowed/);
    const pdf = (async () => respond("%PDF", "application/pdf")) as unknown as typeof globalThis.fetch;
    await expect(webPageSkill({ fetch: pdf }).execute({ url: "https://example.com/a.pdf" })).rejects.toThrow(/not a text page/);
    expect(await webPageSkill({ fetch, allowUrl: () => true }).execute({ url: "http://localhost/" })).toMatchObject({ text: "x" });
  });
});

describe("skills", () => {
  it("are four tools whose schemas the call grammar compiles", () => {
    const all = skills();
    expect(all.map((t) => t.function.name)).toEqual(["calculate", "get_date_time", "search_wikipedia", "read_web_page"]);
    expect(() => compileFormat({ type: "tool_call", tools: all.map((t) => ({ name: t.function.name, parameters: t.function.parameters as never })) })).not.toThrow();
  });
});

describe("calculate, exactly", () => {
  it("keeps integers exact and rounds the rest", () => {
    expect(evaluate("1234567*7654321")).toBe(9449772114007);
    expect(evaluate("999999999999+2")).toBe(1000000000001);
    expect(evaluate("pow(2,100)")).toBe(2 ** 100);
    expect(evaluate("1 / 3")).toBe(0.333333333333);
  });

  it("reads a comma as a thousands separator only outside a function's arguments", () => {
    expect(evaluate("max(1,500)")).toBe(500);
    expect(evaluate("min(3,000)")).toBe(0);
    expect(evaluate("1,234 + 1")).toBe(1235);
    expect(evaluate("max(1, 2) + 1,000")).toBe(1002);
  });

  it("checks argument counts, own names only, nesting, whitespace and logarithms", () => {
    expect(evaluate("round(3.14159, 2)")).toBe(3.14);
    expect(() => evaluate("pow(2)")).toThrow(/pow takes 2 numbers, not 1/);
    expect(() => evaluate("sqrt(1, 2)")).toThrow(/sqrt takes one number/);
    expect(() => evaluate("round(1, 2, 3)")).toThrow(/round takes 1 or 2 numbers/);
    expect(() => evaluate("max()")).toThrow(/max takes one or more numbers/);
    expect(() => evaluate("constructor(5)")).toThrow(/no function constructor/);
    expect(() => evaluate("toString")).toThrow(/is not a number/);
    expect(evaluate("(".repeat(150) + "1" + ")".repeat(150))).toBe(1);
    expect(() => evaluate("(".repeat(300) + "1" + ")".repeat(300))).toThrow(/nests more than 200/);
    expect(() => evaluate("2^".repeat(300) + "1")).toThrow(/nests more than 200/);
    expect(evaluate("-".repeat(10001) + "1")).toBe(-1);
    expect(evaluate("1\t+\n2\r\n* 3")).toBe(7);
    expect(evaluate("log10(1000) + log(100) + log2(8)")).toBe(8);
  });
});

describe("time zones, more", () => {
  const at = () => new Date("2026-03-05T22:30:00Z");

  it("reads UTC and GMT offsets", () => {
    expect(timeZoneOf("UTC+3")).toMatch(/^(\+03:00|Etc\/GMT-3)$/);
    expect(timeZoneOf("GMT-5")).toMatch(/^(-05:00|Etc\/GMT\+5)$/);
    expect(timeZoneOf("UTC+5:30")).toBe("+05:30");
    expect(timeZoneOf("utc +03:00")).toMatch(/^(\+03:00|Etc\/GMT-3)$/);
    expect(timeZoneOf("UTC+0")).toBe("UTC");
    expect(timeZoneOf("UTC+15")).toBeUndefined();
    expect(dateTimeSkill(at).execute({ place: "UTC+5:30" })).toMatchObject({ time: "04:00", utc_offset: "GMT+05:30" });
  });

  it("matches places by their letters only when nothing else was in the name, knows Kyiv, and keeps the name asked", () => {
    expect(timeZoneOf("Paris2")).toBeUndefined();
    expect(timeZoneOf("+Tokyo")).toBeUndefined();
    expect(timeZoneOf("Port-au-Prince")).toBe("America/Port-au-Prince");
    expect(timeZoneOf("Kyiv")).toBe("Europe/Kiev");
    expect(timeZoneOf("Asia/Kolkata")).toBe("Asia/Kolkata");
    expect(dateTimeSkill(at).execute({ place: "Asia/Kolkata" })).toMatchObject({ time_zone: "Asia/Kolkata", utc_offset: "GMT+05:30" });
  });
});

describe("network skills, the requests", () => {
  afterEach(() => vi.unstubAllGlobals());
  const json = (pages: unknown) => respond(JSON.stringify({ query: { pages } }), "application/json");

  it("send no cookies or referrer, and stop with the run or after a while", async () => {
    const inits: Init[] = [];
    const fetch = (async (_url: string, init: Init) => (inits.push(init), respond("<p>hi</p>"))) as unknown as typeof globalThis.fetch;
    await webPageSkill({ fetch }).execute({ url: "https://example.com/" });
    const wfetch = (async (_url: string, init: Init) => (inits.push(init), json({ "1": { index: 1, title: "X", extract: "An x." } }))) as unknown as typeof globalThis.fetch;
    await wikipediaSkill({ fetch: wfetch }).execute({ query: "x" });
    for (const init of inits) expect(init).toMatchObject({ credentials: "omit", referrerPolicy: "no-referrer" });
    expect(inits[0]!.redirect).toBe("manual");
    expect(inits[1]!.redirect).toBe("follow");

    const hang = (async (_url: string, init: Init) =>
      new Promise((_, reject) => {
        if (init.signal.aborted) reject(init.signal.reason);
        init.signal.addEventListener("abort", () => reject(init.signal.reason));
      })) as unknown as typeof globalThis.fetch;
    await expect(webPageSkill({ fetch: hang, timeout: 20 }).execute({ url: "https://example.com/" })).rejects.toThrow(/example\.com did not answer in time/);
    await expect(wikipediaSkill({ fetch: hang, timeout: 20 } as SkillHooks).execute({ query: "x" })).rejects.toThrow(/Wikipedia did not answer in time/);
    const stop = new AbortController();
    stop.abort();
    await expect(webPageSkill({ fetch: hang }).execute({ url: "https://example.com/" }, { signal: stop.signal })).rejects.toThrow(/stopped before example\.com answered/);
  });

  it("waits 2 s after a 429 that names no pause", async () => {
    const waits: number[] = [];
    let n = 0;
    const busy = (async () => (n++ === 0 ? new Response("", { status: 429 }) : json({ "1": { index: 1, title: "X", extract: "An x." } }))) as unknown as typeof globalThis.fetch;
    const sleep = async (ms: number) => void waits.push(ms);
    expect(await wikipediaSkill({ fetch: busy, sleep } as SkillHooks).execute({ query: "x" })).toMatchObject({ title: "X" });
    expect(waits).toEqual([2000]);
  });

  it("takes Wikipedia language codes of any shape a host label allows, and asks for every hit's extract", async () => {
    const asked: string[] = [];
    const fetch = (async (url: string) => (asked.push(url), json({ "1": { index: 1, title: "X", extract: "An x." } }))) as unknown as typeof globalThis.fetch;
    for (const language of ["simple", "zh-yue", "BE-TARASK"]) await wikipediaSkill({ fetch }).execute({ query: "x", language });
    expect(asked.map((u) => new URL(u).hostname)).toEqual(["simple.wikipedia.org", "zh-yue.wikipedia.org", "be-tarask.wikipedia.org"]);
    expect(asked[0]).toContain("exlimit=3");
    await wikipediaSkill({ fetch, language: "Simple" }).execute({ query: "x" });
    expect(asked[3]).toContain("https://simple.wikipedia.org/");
    for (const language of ["evil.example", "e", "-en", "a".repeat(21), ""]) expect(() => wikipediaSkill({ language }), language).toThrow(/not a Wikipedia language code/);
    await webPageSkill({ fetch }).execute({ url: "https://zh-yue.wikipedia.org/wiki/X" });
    expect(asked[4]).toContain("https://zh-yue.wikipedia.org/w/api.php?");
  });
});

describe("read_web_page, which addresses", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("refuses private addresses however they are written", async () => {
    let fetched = 0;
    const fetch = (async () => (fetched++, respond("x"))) as unknown as typeof globalThis.fetch;
    const page = webPageSkill({ fetch });
    for (const url of [
      "http://localhost./", "http://foo.local./", "http://metadata.google.internal./", "http://198.18.0.1/", "http://198.19.255.1/", "http://224.0.0.1/", "http://240.0.0.1/", "http://255.255.255.255/",
      "http://[::ffff:127.0.0.1]/", "http://[::ffff:7f00:1]/", "http://[::127.0.0.1]/", "http://[::7f00:1]/", "http://[64:ff9b::a00:1]/", "http://[2002:c0a8:101::]/", "http://[fec0::1]/", "http://[fe80::1]/", "http://[fd00::1]/", "http://[::]/",
    ]) {
      await expect(page.execute({ url }), url).rejects.toThrow(/not allowed/);
    }
    expect(fetched).toBe(0);
    for (const url of ["http://[2606:4700::1111]/", "http://8.8.8.8/", "http://[2002:808:808::]/", "http://[::ffff:8.8.8.8]/"]) {
      expect(await page.execute({ url }), url).toMatchObject({ text: "x" });
    }
    expect(isPrivateAddress("::ffff:192.168.0.1")).toBe(true);
    expect(isPrivateAddress("10.1.2.3")).toBe(true);
    expect(isPrivateAddress("2606:4700::1111")).toBe(false);
  });

  it("refuses a name that resolves to a private address, unless a custom check decides", async () => {
    const asked: string[] = [];
    const fetch = (async (url: string) => (asked.push(url), respond("x"))) as unknown as typeof globalThis.fetch;
    const lookup = async (host: string) => (host.startsWith("127.") ? [{ address: "127.0.0.1" }] : host === "two.example" ? [{ address: "8.8.8.8" }, { address: "::1" }] : [{ address: "8.8.8.8" }]);
    const page = webPageSkill({ fetch, lookup });
    await expect(page.execute({ url: "http://127.0.0.1.nip.io/" })).rejects.toThrow(/not allowed \(it leads to a private address\)/);
    await expect(page.execute({ url: "http://two.example/" })).rejects.toThrow(/not allowed/);
    expect(asked).toEqual([]);
    // A name that does not resolve: fetch says so.
    const failing = webPageSkill({ fetch, lookup: () => Promise.reject(new Error("ENOTFOUND")) });
    expect(await failing.execute({ url: "https://nowhere.example/" })).toMatchObject({ text: "x" });
    let looked = 0;
    const custom = webPageSkill({ fetch, lookup: async () => (looked++, [{ address: "127.0.0.1" }]), allowUrl: () => true });
    expect(await custom.execute({ url: "http://127.0.0.1.nip.io/" })).toMatchObject({ text: "x" });
    expect(looked).toBe(0);
  });

  it("refuses the embedding page's own origin by default", async () => {
    vi.stubGlobal("location", { origin: "https://app.example" });
    const fetch = (async () => respond("x")) as unknown as typeof globalThis.fetch;
    await expect(webPageSkill({ fetch }).execute({ url: "https://app.example/api/me" })).rejects.toThrow(/app\.example is not allowed/);
    expect(await webPageSkill({ fetch }).execute({ url: "https://other.example/" })).toMatchObject({ text: "x" });
    expect(await webPageSkill({ fetch, allowUrl: () => true }).execute({ url: "https://app.example/" })).toMatchObject({ text: "x" });
  });

  it("checks every redirect before following it", async () => {
    const hops = (targets: Record<string, string>) => {
      const asked: string[] = [];
      const fetch = (async (url: string) => {
        asked.push(url);
        return targets[url] ? new Response(null, { status: 302, headers: { location: targets[url]! } }) : respond("<p>arrived</p>", "text/html", url);
      }) as unknown as typeof globalThis.fetch;
      return { fetch, asked };
    };
    const ok = hops({ "https://a.example/": "/b", "https://a.example/b": "https://c.example/" });
    expect(await webPageSkill({ fetch: ok.fetch }).execute({ url: "https://a.example/" })).toEqual({ url: "https://c.example/", text: "arrived" });
    expect(ok.asked).toEqual(["https://a.example/", "https://a.example/b", "https://c.example/"]);

    const inward = hops({ "https://a.example/": "http://169.254.169.254/latest/meta-data" });
    await expect(webPageSkill({ fetch: inward.fetch }).execute({ url: "https://a.example/" })).rejects.toThrow(/169\.254\.169\.254 is not allowed/);
    expect(inward.asked).toEqual(["https://a.example/"]);
    const toFile = hops({ "https://a.example/": "file:///etc/passwd" });
    await expect(webPageSkill({ fetch: toFile.fetch }).execute({ url: "https://a.example/" })).rejects.toThrow(/only http and https/);
    const resolvesInward = hops({ "https://a.example/": "https://evil.example/" });
    const lookup = async (host: string) => [{ address: host === "evil.example" ? "10.0.0.1" : "8.8.8.8" }];
    await expect(webPageSkill({ fetch: resolvesInward.fetch, lookup }).execute({ url: "https://a.example/" })).rejects.toThrow(/private address/);
    expect(resolvesInward.asked).toEqual(["https://a.example/"]);

    const loop = hops(Object.fromEntries(Array.from({ length: 10 }, (_, i) => [`https://a.example/${i}`, `/${i + 1}`])));
    await expect(webPageSkill({ fetch: loop.fetch }).execute({ url: "https://a.example/0" })).rejects.toThrow(/redirects too many times/);
    expect(loop.asked.length).toBe(6);
  });

  it("lets a browser follow a redirect it will not show, then checks where it ended", async () => {
    const modes: (RequestRedirect | undefined)[] = [];
    const browser = (end: string) =>
      (async (_url: string, init: Init) => {
        modes.push(init.redirect);
        if (init.redirect === "manual") return { type: "opaqueredirect", status: 0, ok: false, url: "", headers: new Headers(), body: null } as unknown as Response;
        return respond("<p>there</p>", "text/html", end);
      }) as unknown as typeof globalThis.fetch;
    expect(await webPageSkill({ fetch: browser("https://b.example/") }).execute({ url: "https://a.example/" })).toEqual({ url: "https://b.example/", text: "there" });
    expect(modes).toEqual(["manual", "follow"]);
    await expect(webPageSkill({ fetch: browser("http://192.168.0.1/") }).execute({ url: "https://a.example/" })).rejects.toThrow(/192\.168\.0\.1 is not allowed/);
  });

  it("reads at most 2 MB of a page", async () => {
    let pulls = 0;
    let cancelled = false;
    const chunk = new TextEncoder().encode("word ".repeat(13_107)); // 65535 bytes
    const endless = () =>
      new Response(new ReadableStream<Uint8Array>({ pull: (c) => (pulls++, c.enqueue(chunk)), cancel: () => void (cancelled = true) }), { headers: { "content-type": "text/plain" } });
    const r = (await webPageSkill({ fetch: (async () => endless()) as unknown as typeof globalThis.fetch, maxChars: 50 }).execute({ url: "https://example.com/big.txt" })) as { text: string };
    expect(r.text.endsWith("[…]")).toBe(true);
    expect(cancelled).toBe(true);
    expect(pulls).toBeLessThan(40);
  });
});

describe("HTML and text, linear", () => {
  it("matches tag names exactly and keeps forms", () => {
    expect(htmlToText("<body><main-nav>Menu</main-nav><p>Body text</p></body>").text).toBe("Menu Body text");
    expect(htmlToText("<body><form><p>Inside a form</p></form></body>").text).toBe("Inside a form");
    expect(htmlToText("<body><scripts>kept</scripts><script>dropped</script></body>").text).toBe("kept");
    expect(htmlToText("<body><article>One</article><div>between</div><article>Two</article></body>").text).toBe("One\nbetween\nTwo");
    expect(htmlToText("<title>A <b>t</b></title><body>a < b &lt; c<br>d</body>")).toEqual({ title: "A <b>t</b>", text: "a < b < c\nd" });
  });

  it("finishes crafted input in well under a second", async () => {
    const plain = (body: string) => Promise.resolve(webPageSkill({ fetch: (async () => respond(body, "text/plain")) as unknown as typeof globalThis.fetch }).execute({ url: "https://example.com/a.txt" })).catch(() => undefined);
    for (const crafted of ["<".repeat(200000), "<!--".repeat(100000), "<title".repeat(50000), "<main>".repeat(50000), "<script>".repeat(50000), "<a".repeat(100000), "</main".repeat(50000), " ".repeat(200000) + "x", "&a".repeat(100000)]) {
      let t = performance.now();
      htmlToText(crafted);
      expect(performance.now() - t, `htmlToText ${crafted.slice(0, 8)}`).toBeLessThan(250);
      t = performance.now();
      await plain(crafted);
      expect(performance.now() - t, `plainText ${crafted.slice(0, 8)}`).toBeLessThan(250);
    }
  });
});

describe("clip", () => {
  it("cuts text without spaces where the room ends, not after one character", async () => {
    const ja = "日本語の文章には空白がありません".repeat(20);
    const fetch = (async () => respond(JSON.stringify({ query: { pages: { "1": { index: 1, title: "X", extract: ja } } } }), "application/json")) as unknown as typeof globalThis.fetch;
    const r = (await wikipediaSkill({ fetch, maxChars: 100 }).execute({ query: "x" })) as { text: string };
    expect(r.text).toBe(`${ja.slice(0, 100)} […]`);
    const emoji = "😀".repeat(100);
    const e = (async () => respond(JSON.stringify({ query: { pages: { "1": { index: 1, title: "X", extract: emoji } } } }), "application/json")) as unknown as typeof globalThis.fetch;
    const cut = ((await wikipediaSkill({ fetch: e, maxChars: 51 }).execute({ query: "x" })) as { text: string }).text;
    expect(cut).toBe(`${"😀".repeat(25)} […]`);
  });
});
