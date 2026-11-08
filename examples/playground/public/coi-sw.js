// Cross-origin isolation where the host cannot send headers (GitHub Pages): this service worker adds
// COOP / COEP to the site's own responses, which makes SharedArrayBuffer available, so the CPU
// backend can split matmuls across threads. Other origins (the Hugging Face downloads) are not touched:
// they are CORS requests, which require-corp allows.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (new URL(req.url).origin !== self.location.origin) return;
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  e.respondWith(
    fetch(req).then((res) => {
      if (res.status === 0) return res;
      const headers = new Headers(res.headers);
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    }),
  );
});
