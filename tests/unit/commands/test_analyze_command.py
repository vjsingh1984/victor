"""Tests for the analyze CLI command (WS-6)."""

import pytest
from typer.testing import CliRunner

from victor.ui.commands.analyze import app

runner = CliRunner()


class TestAnalyzeCommand:
    """Tests for analyze command."""

    def test_help_text(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Analyze" in result.output or "analyze" in result.output

    def test_format_json_flag_exists(self):
        """Verify --format flag is accepted."""
        result = runner.invoke(app, ["--help"])
        assert "--format" in result.output

    def test_tdd_priority_flag_exists(self):
        result = runner.invoke(app, ["--help"])
        assert "--tdd-priority" in result.output

    def test_hotspots_flag_exists(self):
        result = runner.invoke(app, ["--help"])
        assert "--hotspots" in result.output

    def test_refresh_flag_exists(self):
        result = runner.invoke(app, ["--help"])
        assert "--refresh" in result.output
