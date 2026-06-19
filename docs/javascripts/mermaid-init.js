/* Mermaid diagram renderer initialization for MkDocs Material.
 * Paired with pymdownx.superfences custom_fences (name: mermaid) in mkdocs.yml.
 * Fenced ```mermaid blocks are emitted as <pre class="mermaid"> by superfences;
 * this initializer renders them and re-runs on theme toggle / instant nav.
 */
(function () {
  function initMermaid() {
    if (typeof mermaid === "undefined") {
      return;
    }
    var dark = document.body &&
      document.body.getAttribute("data-md-color-scheme") === "slate";
    mermaid.initialize({
      startOnLoad: false,
      theme: dark ? "dark" : "default",
      securityLevel: "loose",
      flowchart: { useMaxWidth: true, htmlLabels: true, curve: "basis" },
      sequence: { useMaxWidth: true },
      gantt: { useMaxWidth: true }
    });
    mermaid.run({ querySelector: ".mermaid" });
  }

  document.addEventListener("DOMContentLoaded", initMermaid);
  document.addEventListener("click", function (event) {
    if (event.target.closest && event.target.closest(".md-header__button")) {
      setTimeout(initMermaid, 150);
    }
  });
  if (typeof document$ !== "undefined") {
    document$.subscribe(initMermaid);
  }
})();
