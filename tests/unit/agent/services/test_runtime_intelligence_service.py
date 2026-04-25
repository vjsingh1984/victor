from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.runtime_intelligence import (
    ClarificationDecision,
    PromptOptimizationBundle,
    RuntimeIntelligenceService,
)
from victor.evaluation.runtime_feedback import save_runtime_evaluation_feedback
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.framework.perception_integration import PerceptionIntegration
from victor.framework.runtime_evaluation_policy import (
    RuntimeEvaluationFeedback,
    RuntimeEvaluationPolicy,
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


def test_get_clarification_decision_uses_default_prompt_when_missing():
    perception = SimpleNamespace(
        needs_clarification=True,
        clarification_reason="target artifact or scope is underspecified",
        clarification_prompt=None,
        confidence=0.31,
    )

    decision = RuntimeIntelligenceService.get_clarification_decision(perception)

    assert decision == ClarificationDecision(
        requires_clarification=True,
        reason="target artifact or scope is underspecified",
        prompt="Please clarify the target file, component, or bug before I continue.",
        confidence=0.31,
    )


def test_get_clarification_decision_merges_override_prompt_into_policy():
    perception = SimpleNamespace(
        needs_clarification=True,
        clarification_reason="target artifact or scope is underspecified",
        clarification_prompt=None,
        confidence=0.31,
    )
    policy = RuntimeEvaluationPolicy(
        default_clarification_prompt="Use the policy prompt unless explicitly overridden."
    )

    decision = RuntimeIntelligenceService.get_clarification_decision(
        perception,
        default_prompt="Use the override prompt.",
        policy=policy,
    )

    assert decision.prompt == "Use the override prompt."


def test_get_confidence_evaluation_emits_retry_without_budget_metadata():
    result = RuntimeIntelligenceService.get_confidence_evaluation(0.3)

    assert result.decision == EvaluationDecision.RETRY
    assert result.reason == "Low confidence - retry"
    assert result.metadata == {}


def test_get_confidence_evaluation_merges_threshold_override_into_policy():
    policy = RuntimeEvaluationPolicy(
        medium_confidence_threshold=0.5,
        low_confidence_reason="Retry with stronger evidence",
    )

    result = RuntimeIntelligenceService.get_confidence_evaluation(
        0.65,
        medium_confidence_threshold=0.7,
        policy=policy,
    )

    assert result.decision == EvaluationDecision.RETRY
    assert result.reason == "Retry with stronger evidence"


def test_apply_low_confidence_retry_budget_increments_retry_count():
    evaluation = EvaluationResult(
        decision=EvaluationDecision.RETRY,
        score=0.2,
        reason="Low confidence - retry",
    )
    state = {}

    result = RuntimeIntelligenceService.apply_low_confidence_retry_budget(
        evaluation,
        state,
        retry_limit=2,
    )

    assert result.decision == EvaluationDecision.RETRY
    assert state["low_confidence_retries"] == 1
    assert result.metadata["low_confidence_retries"] == 1
    assert result.metadata["low_confidence_retry_limit"] == 2


def test_apply_low_confidence_retry_budget_exhausts_after_limit():
    evaluation = EvaluationResult(
        decision=EvaluationDecision.RETRY,
        score=0.2,
        reason="Low confidence - retry",
        metadata={"source": "enhanced"},
    )
    state = {"low_confidence_retries": 2}

    result = RuntimeIntelligenceService.apply_low_confidence_retry_budget(
        evaluation,
        state,
        retry_limit=2,
    )

    assert result.decision == EvaluationDecision.FAIL
    assert result.metadata["low_confidence_retry_exhausted"] is True
    assert result.metadata["low_confidence_retries"] == 2
    assert result.metadata["source"] == "enhanced"


def test_evaluate_confidence_progress_resets_retry_budget_on_progress():
    state = {"low_confidence_retries": 1}

    result = RuntimeIntelligenceService.evaluate_confidence_progress(
        0.6,
        state,
        retry_limit=2,
    )

    assert result.decision == EvaluationDecision.CONTINUE
    assert state["low_confidence_retries"] == 0


def test_evaluate_confidence_progress_merges_threshold_override_into_policy():
    state = {}
    policy = RuntimeEvaluationPolicy(
        medium_confidence_threshold=0.5,
        low_confidence_reason="Retry with stronger evidence",
    )

    result = RuntimeIntelligenceService.evaluate_confidence_progress(
        0.6,
        state,
        medium_confidence_threshold=0.7,
        policy=policy,
    )

    assert result.decision == EvaluationDecision.RETRY
    assert result.reason == "Retry with stronger evidence"
    assert state["low_confidence_retries"] == 1


def test_from_container_applies_decision_service_runtime_feedback():
    from victor.agent.services.protocols.decision_service import LLMDecisionServiceProtocol

    container = MagicMock()
    decision_service = MagicMock()
    decision_service.get_runtime_evaluation_feedback.return_value = RuntimeEvaluationFeedback(
        completion_threshold=0.77,
        enhanced_progress_threshold=0.62,
        minimum_supported_evidence_score=0.84,
    )
    container.get_optional.side_effect = lambda protocol: (
        decision_service if protocol is LLMDecisionServiceProtocol else None
    )

    service = RuntimeIntelligenceService.from_container(container)

    assert service.evaluation_policy.completion_threshold == pytest.approx(0.77)
    assert service.evaluation_policy.enhanced_progress_threshold == pytest.approx(0.62)
    assert service.evaluation_policy.minimum_supported_evidence_score == pytest.approx(0.84)
    assert service.perception_integration.evaluation_policy.completion_threshold == pytest.approx(
        0.77
    )


def test_runtime_intelligence_loads_persisted_evaluation_feedback(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            completion_threshold=0.74,
            enhanced_progress_threshold=0.58,
            minimum_supported_evidence_score=0.86,
            metadata={"source": "benchmark_truth_feedback"},
        ),
        path=tmp_path / "runtime_evaluation_feedback.json",
    )

    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=feedback_path,
    )

    assert service.evaluation_policy.completion_threshold == pytest.approx(0.74)
    assert service.evaluation_policy.enhanced_progress_threshold == pytest.approx(0.58)
    assert service.evaluation_policy.minimum_supported_evidence_score == pytest.approx(0.86)
    assert (
        service.perception_integration.evaluation_policy.minimum_supported_evidence_score
        == pytest.approx(0.86)
    )


def test_from_container_merges_persisted_feedback_before_decision_service_feedback(tmp_path):
    from victor.agent.services.protocols.decision_service import LLMDecisionServiceProtocol

    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            completion_threshold=0.71,
            enhanced_progress_threshold=0.56,
            minimum_supported_evidence_score=0.89,
            metadata={"source": "benchmark_truth_feedback"},
        ),
        path=tmp_path / "runtime_evaluation_feedback.json",
    )
    container = MagicMock()
    decision_service = MagicMock()
    decision_service.get_runtime_evaluation_feedback.return_value = RuntimeEvaluationFeedback(
        completion_threshold=0.79,
        enhanced_progress_threshold=None,
        minimum_supported_evidence_score=None,
        metadata={"source": "decision_service"},
    )
    container.get_optional.side_effect = lambda protocol: (
        decision_service if protocol is LLMDecisionServiceProtocol else None
    )

    service = RuntimeIntelligenceService.from_container(
        container,
        evaluation_feedback_path=feedback_path,
    )

    assert service.evaluation_policy.completion_threshold == pytest.approx(0.79)
    assert service.evaluation_policy.enhanced_progress_threshold == pytest.approx(0.56)
    assert service.evaluation_policy.minimum_supported_evidence_score == pytest.approx(0.89)


def test_runtime_intelligence_keeps_explicit_config_thresholds_over_persisted_feedback(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            completion_threshold=0.74,
            enhanced_progress_threshold=0.58,
            minimum_supported_evidence_score=0.86,
            metadata={"source": "benchmark_truth_feedback"},
        ),
        path=tmp_path / "runtime_evaluation_feedback.json",
    )
    perception_integration = PerceptionIntegration(config={"completion_threshold": 0.93})

    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=perception_integration,
        optimization_injector=None,
        decision_service=None,
        evaluation_policy=RuntimeEvaluationPolicy.from_config({"completion_threshold": 0.93}),
        evaluation_feedback_path=feedback_path,
    )

    assert service.evaluation_policy.completion_threshold == pytest.approx(0.93)
    assert service.evaluation_policy.enhanced_progress_threshold == pytest.approx(0.58)
    assert service.evaluation_policy.minimum_supported_evidence_score == pytest.approx(0.86)
    assert service.perception_integration.evaluation_policy.completion_threshold == pytest.approx(
        0.93
    )
