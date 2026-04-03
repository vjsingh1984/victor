"""Tests for the analyze CLI command (WS-6)."""

import re

import pytest
from typer.testing import CliRunner

from victor.verticals.contrib.coding.commands.analyze import app

runner = CliRunner()

# Rich / Typer may inject ANSI escape sequences into the help output,
# which can split flag names (e.g. "--td\x1b[0md-priority") and cause
# plain substring checks to fail.  Strip them before asserting.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestAnalyzeCommand:
    """Tests for analyze command."""

    def test_help_text(self):
        result = runner.invoke(app, ["--help"])
        output = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert "Analyze" in output or "analyze" in output

    def test_format_json_flag_exists(self):
        """Verify --format flag is accepted."""
        result = runner.invoke(app, ["--help"])
        assert "--format" in _strip_ansi(result.output)

    def test_tdd_priority_flag_exists(self):
        result = runner.invoke(app, ["--help"])
        assert "--tdd-priority" in _strip_ansi(result.output)

    def test_hotspots_flag_exists(self):
        result = runner.invoke(app, ["--help"])
        assert "--hotspots" in _strip_ansi(result.output)

    def test_refresh_flag_exists(self):
        result = runner.invoke(app, ["--help"])
        assert "--refresh" in _strip_ansi(result.output)


def test_analyze_app_importable_from_contrib():
    """Analyze app is always importable from contrib — CLI fallback path relies on this."""
    import typer
    from victor.verticals.contrib.coding.commands.analyze import app as _app

    assert _app is not None
    assert isinstance(_app, typer.Typer)
