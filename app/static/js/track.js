/* Behavioural tracking that stays out of the user's way.
 *
 * Rules this file obeys:
 *   - nothing on the interaction path ever awaits the network
 *   - events are buffered and sent in batches via sendBeacon, which the browser
 *     delivers even as the page is being torn down
 *   - high-frequency signals (scroll, dwell) are throttled at the source, not
 *     on the server
 *   - every event carries a UUID, because sendBeacon fires twice on tab close
 *     (pagehide AND visibilitychange) and the server dedupes on it
 */
(function () {
  "use strict";

  var ENDPOINT = "/api/events";
  var FLUSH_SIZE = 10;
  var FLUSH_MS = 5000;
  var DWELL_MIN_S = 20;      // below this it is not real engagement
  var BOUNCE_MAX_S = 3;      // below this it is a rejection, and counts negative

  var buffer = [];
  var sessionId = getSession();
  var sent = Object.create(null);   // one-shot guards, e.g. scroll depth marks

  function uuid() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
      var r = (Math.random() * 16) | 0;
      return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
    });
  }

  function getSession() {
    try {
      var k = "reckon.sid", v = sessionStorage.getItem(k);
      if (!v) { v = uuid(); sessionStorage.setItem(k, v); }
      return v;
    } catch (e) { return uuid(); }
  }

  function push(type, fields) {
    var e = fields || {};
    e.id = uuid();
    e.type = type;
    e.session_id = sessionId;
    e.ts = new Date().toISOString();
    buffer.push(e);
    if (buffer.length >= FLUSH_SIZE) flush();
  }

  function flush(final) {
    if (!buffer.length) return;
    var payload = JSON.stringify({ events: buffer.splice(0, buffer.length) });

    // sendBeacon is the only transport the browser guarantees during unload.
    if (navigator.sendBeacon) {
      var blob = new Blob([payload], { type: "application/json" });
      if (navigator.sendBeacon(ENDPOINT, blob)) return;
    }
    // keepalive so an in-flight request survives navigation.
    try {
      fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
        keepalive: true,
      }).catch(function () {});
    } catch (e) { /* tracking must never break the page */ }
  }

  // --- page view + dwell ---------------------------------------------------
  var productEl = document.querySelector("[data-product]");
  var productId = productEl ? parseInt(productEl.getAttribute("data-product"), 10) : null;
  var isDetail = !!document.querySelector(".detail[data-product]");
  var enteredAt = performance.now();
  var dwellClosed = false;

  if (isDetail && productId) push("view", { product_id: productId });

  function closeDwell() {
    if (!isDetail || !productId) return;
    // pagehide AND visibilitychange both fire when a tab closes or navigates.
    // Without this guard the second call measures the ~0s since the first one
    // reset the clock and emits a phantom bounce -- which would push the intent
    // vector AWAY from a course the user actually read. The UUID dedup cannot
    // save us here: these are genuinely distinct events.
    if (dwellClosed) return;
    dwellClosed = true;

    var seconds = (performance.now() - enteredAt) / 1000;
    if (seconds < BOUNCE_MAX_S) {
      // A fast exit is information: it pushes the intent vector AWAY.
      push("bounce", { product_id: productId, value: round(seconds) });
    } else if (seconds >= DWELL_MIN_S) {
      push("dwell", { product_id: productId, value: round(seconds) });
    }
  }

  function round(n) { return Math.round(n * 10) / 10; }

  // --- clicks on catalog cards --------------------------------------------
  document.addEventListener("click", function (ev) {
    var card = ev.target.closest ? ev.target.closest(".card[data-product]") : null;
    if (!card) return;
    var pid = parseInt(card.getAttribute("data-product"), 10);
    if (pid) push("click", { product_id: pid });
  }, { passive: true });

  // --- search, debounced ---------------------------------------------------
  var searchInput = document.getElementById("search-input");
  if (searchInput) {
    var t = null;
    searchInput.addEventListener("input", function () {
      clearTimeout(t);
      var q = searchInput.value.trim();
      if (q.length < 3) return;
      t = setTimeout(function () { push("search", { query: q }); }, 500);
    }, { passive: true });
  }
  // A submitted search is a stronger signal than a typed one.
  var q0 = new URLSearchParams(location.search).get("q");
  if (q0 && q0.trim().length >= 2) push("search", { query: q0.trim() });

  // --- scroll depth, rAF-throttled, one event per quartile -----------------
  if (isDetail && productId) {
    var ticking = false;
    window.addEventListener("scroll", function () {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        ticking = false;
        var h = document.documentElement.scrollHeight - window.innerHeight;
        if (h <= 0) return;
        var pct = Math.min(100, Math.round(((window.scrollY || 0) / h) * 100));
        [25, 50, 75, 100].forEach(function (mark) {
          var key = "scroll" + mark;
          if (pct >= mark && !sent[key]) {
            sent[key] = true;
            push("scroll", { product_id: productId, value: mark });
          }
        });
      });
    }, { passive: true });
  }

  // --- flush triggers ------------------------------------------------------
  setInterval(flush, FLUSH_MS);

  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") { closeDwell(); flush(true); }
    else { enteredAt = performance.now(); dwellClosed = false; }  // tab came back
  });
  window.addEventListener("pagehide", function () { closeDwell(); flush(true); });

  window.Reckon = { push: push, flush: flush };
})();
