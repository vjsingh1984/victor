"""Guard: keep the CLI cold-start import chain free of heavy optional deps.

Background: ``python -m victor.ui.cli --help`` once took ~5s because importing
the CLI eagerly pulled ``scipy`` (~180ms) and ``jsonschema``/``rfc3987``
(~660ms). Those expensive optional deps were deferred to first use:

* ``victor/tools/base.py`` — ``jsonschema`` moved function-local into
  ``BaseTool.validate_parameters_detailed``.
* ``victor/experiments/ab_testing/{statistics,metrics}.py`` — ``scipy`` replaced
  with a ``find_spec`` availability flag + a lazy ``stats`` proxy.

This test pins that: after importing the CLI module in a fresh interpreter,
neither may be resident. A regression that re-introduces a module-level
``import scipy`` / ``import jsonschema`` in victor source will turn this red.

NOTE on ``lancedb``/``pyarrow``: these ARE still loaded at CLI import time, but
by EXTERNAL vertical packages (``victor-rag``/``victor-dataanalysis``) during the
mandatory plugin bootstrap (``cli.py::_register_plugin_commands`` ->
``ensure_bootstrapped``) — not by victor-ai source. Deferring the bootstrap would
break plugin-command resolution, so they are intentionally NOT asserted here.
Reducing that remaining cost is tracked as follow-up work.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Heavy optional deps that must NOT be eagerly imported by victor-ai source at
# CLI load time. (lancedb/pyarrow excluded — see module docstring.)
FORBIDDEN_AT_CLI_IMPORT = {
    "scipy",  # victor/experiments/ab_testing — deferred to first use
    "jsonschema",  # victor/tools/base.py — deferred into validate_parameters_detailed
    "rfc3987",  # transitive of jsonschema (format checker)
    "rfc3987_syntax",
}


class TestCliImportWeightGuard:
    """Prevent heavy optional deps from re-entering the CLI cold-start path."""

    def test_cli_import_does_not_eagerly_load_heavy_deps(self) -> None:
        """A fresh interpreter importing victor.ui.cli must leave heavy deps un-imported."""
        forbidden = tuple(sorted(FORBIDDEN_AT_CLI_IMPORT))
        code = (
            "import sys, json\n"
            "import victor.ui.cli\n"
            f"loaded = [m for m in {forbidden!r} if m in sys.modules]\n"
            "print('LOADED:' + json.dumps(loaded))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert (
            result.returncode == 0
        ), f"`import victor.ui.cli` failed in a fresh subprocess:\n{result.stderr}"
        loaded_line = next(
            (ln for ln in result.stdout.splitlines() if ln.startswith("LOADED:")),
            "",
        )
        loaded = json.loads(loaded_line[len("LOADED:") :] or "[]")
        assert loaded == [], (
            "CLI cold-start eagerly imported heavy optional deps — defer them to "
            f"first use instead of importing at module scope: {loaded}. "
            "See this module's docstring for the why and the fix pattern."
        )
