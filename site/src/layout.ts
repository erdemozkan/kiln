// Shared header, footer, theme toggle, code copy buttons and the docs table of contents.
import "./site.css";

const REPO = "https://github.com/erdemozkan/kiln";
const LINKS: [string, string][] = [
  ["docs", "Docs"],
  ["examples", "Examples"],
  ["migrate", "From transformers.js"],
  ["benchmarks", "Benchmarks"],
];

/** The path from this page to the site root ("" or "../"). */
const root = document.body.dataset.root ?? "";

// The wordmark of docs/media/logo.svg, its ink following the page's text color.
const LOGO = `<svg height="32" viewBox="150 110 760 380" role="img" aria-label="Kiln"><g fill="currentColor" stroke="currentColor" stroke-width="10" stroke-linejoin="round"><rect x="175" y="135" width="82" height="328" rx="14"/><path d="M250,300 L340,230 L400,230 L405,240 L315,330 L405,455 L400,463 L345,463 L262,350 L250,350 Z"/><rect x="437" y="293" width="74" height="170" rx="12"/><rect x="541" y="135" width="78" height="328" rx="14"/><rect x="649" y="227" width="78" height="236" rx="12"/><path d="M700,300 C704,244 742,220 790,220 C848,220 883,254 883,320 L883,451 Q883,463 871,463 L817,463 Q805,463 805,451 L805,332 C805,306 792,294 770,294 C746,294 727,310 727,340 L700,340 Z"/></g><path d="M486,183 C469,181 464,188 464,200 L464,212 C464,220 458,225 449,226 C458,227 464,232 464,240 L464,254 C464,266 469,273 486,271" fill="none" stroke="#f4571b" stroke-width="13" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

function header(): HTMLElement {
  const h = document.createElement("header");
  h.className = "site-header";
  const page = document.body.dataset.page;
  h.innerHTML = `<div class="bar">
    <a class="brand" href="${root}index.html">${LOGO}</a>
    <nav class="site-nav" aria-label="Site">${LINKS.map(([p, label]) => `<a href="${root}${p}.html"${page === p ? ' aria-current="page"' : ""}>${label}</a>`).join("")}<a href="${REPO}">GitHub</a></nav>
    <button class="theme-toggle" type="button" aria-label="Toggle dark mode" title="Toggle dark mode">◐</button>
  </div>`;
  h.querySelector(".theme-toggle")!.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("kiln-theme", next);
    } catch {
      // Storage may be unavailable (private mode); the toggle still works for this page.
    }
  });
  return h;
}

function footer(): HTMLElement {
  const f = document.createElement("footer");
  f.className = "site-footer";
  f.innerHTML = `<div class="bar"><span>Kiln (kiln-webgpu), MIT licensed. Models run in your browser; nothing is sent to a server.</span><span><a href="${REPO}">Source</a> · <a href="${REPO}/blob/main/CHANGELOG.md">Changelog</a> · <a href="${root}index.html">Home</a></span></div>`;
  return f;
}

function copyButtons(): void {
  for (const pre of document.querySelectorAll("pre:has(> code)")) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "copy";
    b.textContent = "Copy";
    b.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector("code")?.textContent ?? pre.textContent ?? "");
        b.textContent = "Copied";
      } catch {
        b.textContent = "Select and copy";
      }
      setTimeout(() => (b.textContent = "Copy"), 1500);
    });
    pre.append(b);
  }
}

/** Fills `nav.toc` with the page's h2 headings and highlights the one in view. */
function toc(): void {
  const nav = document.querySelector<HTMLElement>("nav.toc");
  if (!nav) return;
  const heads = [...document.querySelectorAll<HTMLHeadingElement>("main h2[id]")];
  for (const h of heads) {
    const a = document.createElement("a");
    a.href = `#${h.id}`;
    a.textContent = h.textContent;
    nav.append(a);
  }
  const links = [...nav.querySelectorAll("a")];
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        for (const l of links) l.classList.toggle("active", l.getAttribute("href") === `#${e.target.id}`);
      }
    },
    { rootMargin: "-70px 0px -70% 0px" },
  );
  heads.forEach((h) => io.observe(h));
}

try {
  const t = localStorage.getItem("kiln-theme");
  if (t === "dark" || t === "light") document.documentElement.dataset.theme = t;
} catch {
  // No stored theme.
}
document.body.prepend(header());
document.body.append(footer());
copyButtons();
toc();
