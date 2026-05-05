"""Tests for shared Bayesian monitoring reporting and CLI compatibility."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from victor.framework.rl.monitoring.reporting import parse_agent_ids

runner = CliRunner()


def test_parse_agent_ids_trims_and_filters_empty_values() -> None:
    assert parse_agent_ids(" agent_a, ,agent_b ,, agent_c ") == [
        "agent_a",
        "agent_b",
        "agent_c",
    ]
    assert parse_agent_ids("") is None
    assert parse_agent_ids(None) is None


def test_legacy_monitoring_summary_cli_uses_shared_service() -> None:
    from victor.framework.rl.monitoring.cli import app

    service = MagicMock()
    service.render_summary.return_value = "Legacy summary output"

    with patch(
        "victor.framework.rl.monitoring.cli.get_bayesian_monitoring_service",
        return_value=service,
    ):
        result = runner.invoke(app, ["summary", "--days", "21"])

    assert result.exit_code == 0
    assert "Legacy summary output" in result.output
    service.render_summary.assert_called_once_with(21)


def test_legacy_monitoring_belief_cli_supports_export() -> None:
    from victor.framework.rl.monitoring.cli import app

    service = MagicMock()
    service.render_belief.return_value = "Belief output"

    with patch(
        "victor.framework.rl.monitoring.cli.get_bayesian_monitoring_service",
        return_value=service,
    ):
        result = runner.invoke(
            app,
            ["belief", "belief-123", "--export", "/tmp/belief.csv"],
        )

    assert result.exit_code == 0
    assert "Belief output" in result.output
    assert "Belief evolution exported to /tmp/belief.csv" in result.output
    service.render_belief.assert_called_once_with("belief-123")
    service.export_belief_csv.assert_called_once_with("belief-123", "/tmp/belief.csv")
