/* Compact account menu. Outside click + Escape close; aria-expanded tracks state. */
(function () {
  "use strict";
  var root = document.getElementById("profile-menu");
  if (!root) return;
  var btn = document.getElementById("profile-btn");
  var drop = document.getElementById("profile-dropdown");

  function open() {
    drop.hidden = false;
    btn.setAttribute("aria-expanded", "true");
    root.classList.add("open");
  }
  function close() {
    drop.hidden = true;
    btn.setAttribute("aria-expanded", "false");
    root.classList.remove("open");
  }
  function toggle() {
    drop.hidden ? open() : close();
  }

  btn.addEventListener("click", function (e) {
    e.stopPropagation();
    toggle();
  });
  document.addEventListener("click", function () { close(); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") close();
  });
  drop.addEventListener("click", function (e) { e.stopPropagation(); });
})();
