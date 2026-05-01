"""Tests for orchestrator coordination suggestion helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator


def test_get_team_suggestions_uses_shared_framework_runtime_helper() -> None:
    fake_orchestrator = SimpleNamespace()
    expected = SimpleNamespace(kind="coordination")

    with patch(
        "victor.framework.coordination_runtime.build_runtime_coordination_suggestion",
        return_value=expected,
    ) as build_suggestion:
        result = AgentOrchestrator.get_team_suggestions(fake_orchestrator, "feature", "high")

    assert result is expected
    build_suggestion.assert_called_once_with(
        runtime_subject=fake_orchestrator,
        task_type="feature",
        complexity="high",
    )


def test_get_team_suggestions_prefers_coordination_runtime_surface() -> None:
    coordination_runtime = SimpleNamespace(suggest_for_task=MagicMock(return_value="suggestion"))
    fake_orchestrator = SimpleNamespace(_coordination_advisor_runtime=coordination_runtime)

    result = AgentOrchestrator.get_team_suggestions(fake_orchestrator, "feature", "high")

    assert result == "suggestion"
    coordination_runtime.suggest_for_task.assert_called_once_with(
        runtime_subject=fake_orchestrator,
        task_type="feature",
        complexity="high",
    )
