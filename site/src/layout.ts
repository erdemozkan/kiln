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

const LOGO = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 21V10a8 8 0 0 1 16 0v11"/><path d="M2 21h20"/><path d="M9 21v-5a3 3 0 0 1 6 0v5"/></svg>`;

function header(): HTMLElement {
  const h = document.createElement("header");
  h.className = "site-header";
  const page = document.body.dataset.page;
  h.innerHTML = `<div class="bar">
    <a class="brand" href="${root}index.html">${LOGO}<span>Kiln</span></a>
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
