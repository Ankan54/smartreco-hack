/* The Horizon: a hairline that tilts as the user's interests move.
 *
 * No number, no label, no "AI is thinking". It reads as decoration for the
 * first thirty seconds, then you notice it is responding to you. The tilt is
 * the real drift value as a fraction of the fire threshold -- never faked.
 */
(function () {
  "use strict";

  var el = document.getElementById("horizon");
  if (!el) return;                          // signed out: nothing to track
  var line = el.querySelector("i");
  var POLL_MS = 10000;
  var timer = null;

  function apply(s) {
    line.style.setProperty("--tilt", s.tilt_deg + "deg");
    el.setAttribute("data-state", s.state);
    // Telemetry lives on the element for Agent Cam to read, not on screen.
    el.dataset.drift = s.drift;
    el.dataset.threshold = s.threshold;
    el.dataset.events = s.events_since_reco;
    el.dataset.reason = s.reason;
  }

  function poll() {
    fetch("/api/horizon", { credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (s) { if (s) apply(s); })
      .catch(function () { /* the horizon simply stops moving; nothing breaks */ });
  }

  function start() {
    if (timer) return;
    poll();
    timer = setInterval(poll, POLL_MS);
  }
  function stop() {
    clearInterval(timer);
    timer = null;
  }

  // Don't poll a tab nobody is looking at.
  document.addEventListener("visibilitychange", function () {
    document.visibilityState === "hidden" ? stop() : start();
  });
  start();
})();
