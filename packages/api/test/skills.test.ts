import { describe, expect, it } from "vitest";
import { compileFormat } from "@kiln/grammar";
import { calculatorSkill, dateTimeSkill, evaluate, htmlToText, skills, timeZoneOf, webPageSkill, wikipediaSkill } from "../src/skills";

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
