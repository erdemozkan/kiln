"""Draws docs/media/benchmarks-{light,dark}.svg from the rows in site/src/benchmarks.ts.

Run from the repo root after changing those rows: python3 docs/media/make-benchmarks.py
"""
import re
from html import escape
from pathlib import Path

root = Path(__file__).resolve().parents[2]
src = (root / "site/src/benchmarks.ts").read_text()
rows = []
for m in re.finditer(r'\{ area: "[^"]+", task: "([^"]+)", model: "([^"]+)", kiln: ([\d.]+), tjs: ([\d.]+), unit: "(ms|tok/s)"', src):
    task, model, kiln, tjs, unit = m[1], m[2], float(m[3]), float(m[4]), m[5]
    rows.append((f"{model}: {task}", tjs / kiln if unit == "ms" else kiln / tjs))
assert rows, "no rows found in site/src/benchmarks.ts"
rows.sort(key=lambda r: -r[1])

THEMES = {
    "light": dict(bar="#2a78d6", text="#52514e", ink="#0b0b0b", grid="#dcdad5"),
    "dark": dict(bar="#3987e5", text="#c3c2b7", ink="#f0efec", grid="#34332f"),
}
W, LABEL, BAR, GAP, TOP = 820, 330, 14, 7, 30
H = TOP + len(rows) * (BAR + GAP) + 26
top = max(s for _, s in rows)
hi = int(top) + 1
x = lambda v: LABEL + v / hi * (W - LABEL - 52)

for name, c in THEMES.items():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Speed-up of Kiln over transformers.js on {len(rows)} task and model pairs">',
         '<style>text{font:12px system-ui,-apple-system,"Segoe UI",sans-serif}</style>',
         f'<text x="0" y="14" fill="{c["ink"]}" style="font-weight:600;font-size:13px">Times faster than transformers.js, end to end (same browser, same machine)</text>']
    for v in range(0, hi + 1, 2 if hi > 10 else 1):
        o.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{TOP - 4}" y2="{H - 22}" stroke="{c["grid"]}" stroke-width="0.5"/>')
        o.append(f'<text x="{x(v):.1f}" y="{H - 6}" text-anchor="middle" fill="{c["text"]}">{v}×</text>')
    o.append(f'<line x1="{x(1):.1f}" x2="{x(1):.1f}" y1="{TOP - 4}" y2="{H - 22}" stroke="{c["text"]}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    o.append(f'<text x="{x(1):.1f}" y="{H - 6}" text-anchor="middle" fill="{c["text"]}">1×</text>')
    for i, (label, s) in enumerate(rows):
        y = TOP + i * (BAR + GAP)
        if len(label) > 54:
            label = label[:53] + "…"
        w = max(2.0, x(s) - x(0))
        r = min(4.0, w / 2)
        o.append(f'<text x="{LABEL - 10}" y="{y + BAR - 3}" text-anchor="end" fill="{c["text"]}">{escape(label)}</text>')
        o.append(f'<path d="M{x(0):.1f},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{BAR - 2 * r} a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z" fill="{c["bar"]}"/>')
        o.append(f'<text x="{x(s) + 6:.1f}" y="{y + BAR - 3}" fill="{c["ink"]}">{s:.1f}×</text>')
    o.append("</svg>")
    (root / f"docs/media/benchmarks-{name}.svg").write_text("\n".join(o) + "\n")

speed = sorted(s for _, s in rows)
n = len(speed)
median = speed[n // 2] if n % 2 else (speed[n // 2 - 1] + speed[n // 2]) / 2
print(f"{n} rows, median {median:.1f}x, range {speed[0]:.1f}x to {speed[-1]:.1f}x")
