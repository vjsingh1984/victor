"""Tests for orchestrator coordination suggestion helpers."""

from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator


def test_get_coordination_suggestion_auto_provisions_coordination_runtime_from_factory() -> None:
    expected = SimpleNamespace(kind="coordination")
    coordination_runtime = SimpleNamespace(suggest_for_task=MagicMock(return_value=expected))
    factory = SimpleNamespace(
        create_coordination_advisor_runtime=MagicMock(return_value=coordination_runtime)
    )
    fake_orchestrator = SimpleNamespace(
        _factory=factory,
        mode_controller=SimpleNamespace(current_mode=SimpleNamespace(value="build")),
    )

    result = AgentOrchestrator.get_coordination_suggestion(
        fake_orchestrator,
        "feature",
        "high",
    )

    assert result is expected
    assert fake_orchestrator._coordination_advisor_runtime is coordination_runtime
    factory.create_coordination_advisor_runtime.assert_called_once_with()
    coordination_runtime.suggest_for_task.assert_called_once_with(
        runtime_subject=fake_orchestrator,
        task_type="feature",
        complexity="high",
        mode="build",
    )


def test_get_coordination_suggestion_prefers_coordination_runtime_surface() -> None:
    coordination_runtime = SimpleNamespace(suggest_for_task=MagicMock(return_value="suggestion"))
    fake_orchestrator = SimpleNamespace(
        _coordination_advisor_runtime=coordination_runtime,
        mode_controller=SimpleNamespace(current_mode=SimpleNamespace(value="plan")),
    )

    result = AgentOrchestrator.get_coordination_suggestion(
        fake_orchestrator,
        "feature",
        "high",
    )

    assert result == "suggestion"
    coordination_runtime.suggest_for_task.assert_called_once_with(
        runtime_subject=fake_orchestrator,
        task_type="feature",
        complexity="high",
        mode="plan",
    )


def test_get_coordination_suggestion_honors_explicit_mode_override() -> None:
    coordination_runtime = SimpleNamespace(suggest_for_task=MagicMock(return_value="suggestion"))
    fake_orchestrator = SimpleNamespace(
        _coordination_advisor_runtime=coordination_runtime,
        mode_controller=SimpleNamespace(current_mode=SimpleNamespace(value="plan")),
    )

    result = AgentOrchestrator.get_coordination_suggestion(
        fake_orchestrator,
        "feature",
        "high",
        mode="build",
    )

    assert result == "suggestion"
    coordination_runtime.suggest_for_task.assert_called_once_with(
        runtime_subject=fake_orchestrator,
        task_type="feature",
        complexity="high",
        mode="build",
    )


def test_get_coordination_suggestion_falls_back_to_local_service_runtime_when_factory_absent() -> (
    None
):
    expected = SimpleNamespace(kind="coordination")
    fake_orchestrator = SimpleNamespace(
        mode_controller=SimpleNamespace(current_mode=SimpleNamespace(value="explore"))
    )

    with patch(
        "victor.agent.services.coordination_advisor_runtime.CoordinationAdvisorRuntime"
    ) as runtime_cls:
        runtime_cls.return_value.suggest_for_task.return_value = expected
        result = AgentOrchestrator.get_coordination_suggestion(
            fake_orchestrator,
            "feature",
            "high",
        )

    assert result is expected
    assert fake_orchestrator._coordination_advisor_runtime is runtime_cls.return_value
    runtime_cls.return_value.suggest_for_task.assert_called_once_with(
        runtime_subject=fake_orchestrator,
        task_type="feature",
        complexity="high",
        mode="explore",
    )


def test_get_team_suggestions_warns_and_delegates_to_coordination_suggestion() -> None:
    fake_orchestrator = SimpleNamespace(
        get_coordination_suggestion=MagicMock(return_value="suggestion")
    )

    with pytest.warns(
        DeprecationWarning,
        match="AgentOrchestrator.get_team_suggestions\\(\\.\\.\\.\\) is deprecated",
    ):
        result = AgentOrchestrator.get_team_suggestions(fake_orchestrator, "feature", "high")

    assert result == "suggestion"
    fake_orchestrator.get_coordination_suggestion.assert_called_once_with("feature", "high")


def test_get_runtime_coordination_suggestion_prefers_public_runtime_api() -> None:
    runtime_subject = SimpleNamespace(
        get_coordination_suggestion=MagicMock(return_value="suggestion")
    )

    from victor.framework.coordination_runtime import get_runtime_coordination_suggestion

    result = get_runtime_coordination_suggestion(
        runtime_subject=runtime_subject,
        task_type="feature",
        complexity="high",
    )

    assert result == "suggestion"
    runtime_subject.get_coordination_suggestion.assert_called_once_with(
        "feature",
        "high",
        mode=None,
    )
