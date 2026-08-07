/* Arguing with the dossier.
 *
 * Striking a claim removes a retrieval probe, so the picks change on the next
 * query. No LLM call is involved, which is why this is instant -- and the
 * instantness is the point: it makes the mechanism legible without explaining it.
 */
(function () {
  "use strict";

  var panel = document.getElementById("dossier");
  var recoEl = document.getElementById("reco");
  if (!panel) return;

  function post(url) {
    return fetch(url, { method: "POST", credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.json() : null; });
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);   // never innerHTML
    return n;
  }

  function renderPicks(picks) {
    var ol = recoEl && recoEl.querySelector(".picks");
    if (!ol || !picks) return;

    // Built with createElement/textContent rather than innerHTML on purpose.
    // A claim's label is LLM-written from the user's own activity -- a search
    // query flows into the reflect prompt and out again as claim text -- so it
    // is untrusted input and must never be parsed as markup.
    ol.classList.add("settling");
    ol.replaceChildren();

    picks.forEach(function (p, i) {
      var prod = p.product || {};
      var a = el("a", "pick");
      a.href = "/p/" + encodeURIComponent(prod.id);
      a.setAttribute("data-product", prod.id);
      a.appendChild(el("span", "n", i + 1));

      var body = el("span", "body");
      body.appendChild(el("span", "t", prod.title));

      var why = (p.because || [])[0];
      if (why && why.label) {
        var r = el("span", "r because", "surfaced by ");
        r.appendChild(el("b", null, why.label));
        body.appendChild(r);
      }

      var meta = [prod.category, prod.level,
                  prod.price ? "₹" + Math.round(prod.price).toLocaleString("en-IN") : "Free"];
      body.appendChild(el("span", "m", meta.filter(Boolean).join(" · ")));

      a.appendChild(body);
      var li = document.createElement("li");
      li.appendChild(a);
      ol.appendChild(li);
    });

    // The narrative was written for the old picks, so say so rather than
    // leaving stale prose sitting above changed content.
    if (recoEl) recoEl.classList.add("stale");
    setTimeout(function () { ol.classList.remove("settling"); }, 30);
  }

  panel.addEventListener("click", function (ev) {
    var btn = ev.target.closest ? ev.target.closest("[data-toggle]") : null;
    if (!btn) return;
    ev.preventDefault();

    var li = btn.closest(".claim");
    var id = li.getAttribute("data-claim");
    var striking = !li.classList.contains("struck");

    li.classList.toggle("struck", striking);        // optimistic, it never fails loudly
    btn.setAttribute("aria-pressed", String(striking));
    btn.title = striking ? "put this back" : "not true";

    post("/api/dossier/claims/" + encodeURIComponent(id) + "/toggle?enabled=" + !striking)
      .then(function (data) {
        if (!data || data.error) return;
        var pr = panel.querySelector(".dossier-prose");
        if (pr && data.dossier) pr.textContent = data.dossier.prose;
        var ver = panel.querySelector(".ver");
        if (ver && data.dossier) ver.textContent = "v" + data.dossier.version + " · your edit";
        renderPicks(data.picks);
      })
      .catch(function () { li.classList.toggle("struck", !striking); });
  });

  var optIn = document.getElementById("digest-opt-in");
  var optStatus = document.getElementById("digest-status");
  if (optIn) {
    optIn.addEventListener("change", function () {
      post("/api/digest/opt-in?enabled=" + optIn.checked).then(function (d) {
        if (!d) return;
        optStatus.hidden = false;
        optStatus.textContent = !d.opted_in
          ? "digest off"
          : (d.delivery_configured ? "digest on" : "digest on, but email is not configured yet");
      });
    });
  }

  var refresh = document.getElementById("refresh-picks");
  if (refresh) {
    refresh.addEventListener("click", function () {
      refresh.disabled = true;
      refresh.textContent = "Thinking";
      post("/api/recommendations/refresh").then(function () {
        // The agent runs in the background; poll until the id changes.
        var id = parseInt(recoEl.getAttribute("data-reco-id") || "0", 10);
        var tries = 0;
        var t = setInterval(function () {
          fetch("/api/recommendations?since=" + id, { credentials: "same-origin" })
            .then(function (r) { return r.json(); })
            .then(function (d) {
              if (d && d.changed) { clearInterval(t); location.reload(); }
              if (++tries > 20) { clearInterval(t); refresh.disabled = false;
                                  refresh.textContent = "Refresh my picks"; }
            });
        }, 2000);
      });
    });
  }
})();
