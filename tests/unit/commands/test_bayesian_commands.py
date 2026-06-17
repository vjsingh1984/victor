"""Tests for Bayesian monitoring CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

runner = CliRunner()


def test_bayesian_summary_uses_shared_monitoring_service() -> None:
    from victor.ui.commands.bayesian import bayesian_app

    service = MagicMock()
    service.render_summary.return_value = "Bayesian summary body"

    with patch(
        "victor.ui.commands.bayesian.get_bayesian_monitoring_service",
        return_value=service,
    ):
        result = runner.invoke(bayesian_app, ["summary", "--days", "14"])

    assert result.exit_code == 0
    assert "Bayesian summary body" in result.output
    service.render_summary.assert_called_once_with(14)


def test_bayesian_reliability_supports_export() -> None:
    from victor.ui.commands.bayesian import bayesian_app

    service = MagicMock()
    service.render_reliability.return_value = "Reliability body"

    with patch(
        "victor.ui.commands.bayesian.get_bayesian_monitoring_service",
        return_value=service,
    ):
        result = runner.invoke(
            bayesian_app,
            [
                "reliability",
                "--agents",
                "agent_a,agent_b",
                "--days",
                "30",
                "--export",
                "/tmp/reliability.csv",
            ],
        )

    assert result.exit_code == 0
    assert "Reliability body" in result.output
    assert "Reliability trends exported to /tmp/reliability.csv" in result.output
    service.render_reliability.assert_called_once_with(["agent_a", "agent_b"], 30)
    service.export_reliability_csv.assert_called_once_with(
        "/tmp/reliability.csv",
        ["agent_a", "agent_b"],
        30,
    )


def test_top_level_cli_registers_bayesian_group() -> None:
    from victor.ui.cli import app

    result = runner.invoke(app, ["bayesian", "--help"])

    assert result.exit_code == 0
    assert "Inspect Bayesian orchestration metrics" in result.output
    assert "summary" in result.output
    assert "belief" in result.output
