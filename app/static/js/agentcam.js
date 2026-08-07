/* Agent Cam: the backstage view.
 *
 * The only dark surface in the app, and the only place the words "drift",
 * "probe" and "token" are allowed to appear. Everything shown here comes from
 * the stored trace of the last real run -- nothing is illustrative.
 *
 * The graph is drawn from LangGraph's own draw_mermaid() output, parsed for its
 * edges, with the path this run actually took highlighted. No charting library:
 * the topology is small and a dependency would be doing less than these 40 lines.
 */
(function () {
  "use strict";

  var panel = document.getElementById("agentcam");
  if (!panel) return;
  var body = panel.querySelector(".cam-body");

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }

  function open() {
    panel.hidden = false;
    requestAnimationFrame(function () { panel.classList.add("in"); });
    load();
    document.addEventListener("keydown", onKey);
  }
  function close() {
    panel.classList.remove("in");
    document.removeEventListener("keydown", onKey);
    setTimeout(function () { panel.hidden = true; }, 220);
  }
  function onKey(e) { if (e.key === "Escape") close(); }

  /* Node declarations, e.g. "read_behavior(read_behavior)".
     LangGraph emits these in the order the graph was BUILT, which reads
     top-to-bottom as the pipeline. Its edge lines are alphabetical, so
     ordering from those puts persist above retrieve. */
  function nodesFrom(mermaid) {
    var out = [];
    (mermaid || "").split("\n").forEach(function (line) {
      var m = line.match(/^\s*([A-Za-z_][\w]*)\s*[([]/);
      if (m && m[1] !== "__start__" && m[1] !== "__end__" && out.indexOf(m[1]) < 0) {
        out.push(m[1]);
      }
    });
    return out;
  }

  function renderGraph(mermaid, visited) {
    var wrap = el("div", "cam-graph");
    var seen = {};
    visited.forEach(function (n) { seen[n] = (seen[n] || 0) + 1; });

    nodesFrom(mermaid).forEach(function (name) {
      var n = seen[name] || 0;
      var row = el("div", "gnode" + (n ? " on" : " off"));
      row.appendChild(el("span", "dot"));
      row.appendChild(el("span", "gname", name));
      if (n > 1) row.appendChild(el("span", "gloop", "x" + n));
      if (!n) row.appendChild(el("span", "gskip", "skipped"));
      wrap.appendChild(row);
    });
    return wrap;
  }

  function kv(label, value) {
    var d = el("div", "kv");
    d.appendChild(el("span", "k", label));
    d.appendChild(el("span", "v", value));
    return d;
  }

  function renderSteps(steps) {
    var wrap = el("div", "cam-steps");
    steps.forEach(function (s) {
      var d = el("details", "step" + (s.llm ? " llm" : ""));
      var sum = el("summary");
      sum.appendChild(el("span", "sname", s.node));
      sum.appendChild(el("span", "sms", s.ms + "ms"));
      if (s.llm) {
        var c = String(s.cache || "MISS").toUpperCase();
        sum.appendChild(el("span", "stag " + (c === "MISS" ? "miss" : "hit"), c));
      }
      d.appendChild(sum);

      var pre = el("pre", "sjson");
      var copy = {};
      Object.keys(s).forEach(function (k) {
        if (k !== "node" && k !== "ms") copy[k] = s[k];
      });
      pre.textContent = JSON.stringify(copy, null, 1);
      d.appendChild(pre);
      wrap.appendChild(d);
    });
    return wrap;
  }

  function load() {
    body.replaceChildren(el("p", "cam-loading", "reading the last run"));
    fetch("/api/trace", { credentials: "same-origin" })
      .then(function (r) { return r.json(); })
      .then(function (t) {
        body.replaceChildren();
        if (!t.has_run) {
          body.appendChild(el("p", "cam-loading",
            "The agent has not run yet. Browse a few courses."));
          if (t.mermaid) body.appendChild(renderGraph(t.mermaid, []));
          return;
        }

        var head = el("div", "cam-facts");
        head.appendChild(kv("woke because", t.trigger));
        if (t.drift != null) {
          head.appendChild(kv("drift", t.drift.toFixed(4) + " / " + t.threshold));
        }
        head.appendChild(kv("dossier", "v" + (t.dossier_version || 0)));
        head.appendChild(kv("elapsed", t.totals.ms + "ms"));
        head.appendChild(kv("llm calls", t.totals.llm_calls));
        head.appendChild(kv("tokens", t.totals.prompt_tokens + " in / " +
                                      t.totals.completion_tokens + " out"));
        head.appendChild(kv("cache hits", t.totals.cache_hits + " of " + t.totals.llm_calls));
        body.appendChild(head);

        body.appendChild(el("p", "cam-h", "path taken"));
        body.appendChild(renderGraph(t.mermaid, t.visited || []));

        body.appendChild(el("p", "cam-h", "every step"));
        body.appendChild(renderSteps(t.steps || []));
      })
      .catch(function () {
        body.replaceChildren(el("p", "cam-loading", "could not read the trace"));
      });
  }

  var camBtn = document.getElementById("open-agent-cam");
  if (camBtn) {
    camBtn.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      open();
    });
  }
  panel.querySelector(".cam-close").addEventListener("click", close);
  panel.querySelector(".cam-scrim").addEventListener("click", close);
  window.ReckonCam = { open: open };
})();
