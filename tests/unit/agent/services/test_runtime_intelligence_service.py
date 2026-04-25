from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.runtime_intelligence import (
    PromptOptimizationBundle,
    RuntimeIntelligenceService,
)


@pytest.mark.asyncio
async def test_analyze_turn_returns_perception_backed_snapshot():
    task_analysis = MagicMock(task_type="code_generation")
    perception = SimpleNamespace(task_analysis=task_analysis, confidence=0.8)
    perception_integration = SimpleNamespace(perceive=AsyncMock(return_value=perception))
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=perception_integration,
        optimization_injector=None,
        decision_service=MagicMock(),
    )

    snapshot = await service.analyze_turn(
        "Fix the bug",
        context={"project": "myapp"},
        conversation_history=[{"role": "user", "content": "previous"}],
    )

    assert snapshot.query == "Fix the bug"
    assert snapshot.perception is perception
    assert snapshot.task_analysis is task_analysis
    assert snapshot.decision_service_available is True
    perception_integration.perceive.assert_awaited_once_with(
        "Fix the bug",
        {"project": "myapp"},
        [{"role": "user", "content": "previous"}],
    )


def test_get_prompt_optimization_bundle_collects_optimizer_outputs():
    optimizer = MagicMock()
    optimizer.get_evolved_sections.return_value = ["Prefer read over cat."]
    optimizer.get_few_shots.return_value = "Example few shot"
    optimizer.get_failure_hint.return_value = "Check the file path before editing."
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=optimizer,
        decision_service=None,
    )
    turn_context = SimpleNamespace(
        provider_name="test",
        model="test-model",
        task_type="edit",
        last_turn_failed=True,
        last_failure_category="file_not_found",
        last_failure_error="no such file",
    )

    bundle = service.get_prompt_optimization_bundle("Fix the bug", turn_context)

    assert bundle == PromptOptimizationBundle(
        evolved_sections=["Prefer read over cat."],
        few_shots="Example few shot",
        failure_hint="Check the file path before editing.",
    )


def test_reset_decision_budget_delegates_to_service():
    decision_service = MagicMock()
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=decision_service,
    )

    service.reset_decision_budget()

    decision_service.reset_budget.assert_called_once_with()


def test_decide_sync_delegates_to_decision_service():
    decision_service = MagicMock()
    expected = MagicMock()
    decision_service.decide_sync.return_value = expected
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=decision_service,
    )

    result = service.decide_sync(
        DecisionType.TASK_COMPLETION,
        {"response_tail": "done"},
        heuristic_confidence=0.4,
    )

    assert result is expected
    decision_service.decide_sync.assert_called_once_with(
        DecisionType.TASK_COMPLETION,
        {"response_tail": "done"},
        heuristic_confidence=0.4,
    )
