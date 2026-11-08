// Registers coi-sw.js where the page is not cross-origin isolated (no COOP / COEP headers), then
// reloads once so the worker's headers apply. Does nothing on the dev server, which sends them.
(() => {
  const KEY = "kiln-coi-reloaded";
  if (window.crossOriginIsolated) return sessionStorage.removeItem(KEY);
  if (!window.isSecureContext || !("serviceWorker" in navigator)) return;
  const url = new URL("coi-sw.js", document.currentScript.src);
  navigator.serviceWorker.register(url).then(
    () => {
      // One try per tab session: if the reload did not isolate the page, carry on single-threaded.
      if (sessionStorage.getItem(KEY)) return;
      const reload = () => {
        sessionStorage.setItem(KEY, "1");
        location.reload();
      };
      if (navigator.serviceWorker.controller) reload();
      else navigator.serviceWorker.addEventListener("controllerchange", reload);
    },
    (err) => console.warn("[kiln] cross-origin isolation service worker not registered:", err),
  );
})();
