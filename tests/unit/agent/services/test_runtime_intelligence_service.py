import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.runtime_intelligence import (
    ClarificationDecision,
    PromptOptimizationIdentity,
    PromptOptimizationBundle,
    RuntimeIntelligenceService,
)
from victor.evaluation.experiment_memory import (
    ExperimentInsight,
    ExperimentMemoryRecord,
    ExperimentMemoryStore,
    ExperimentScope,
)
from victor.evaluation.runtime_feedback import (
    RuntimeEvaluationFeedbackScope,
    build_validated_session_feedback_payload,
    save_runtime_evaluation_feedback,
)
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


def test_get_prompt_optimization_bundle_tracks_canonical_prompt_identity():
    optimizer = MagicMock()
    optimizer.get_evolved_section_payloads.return_value = [
        {
            "text": "Prefer read over cat.",
            "provider": "anthropic",
            "prompt_candidate_hash": "cand-123",
            "section_name": "GROUNDING_RULES",
            "prompt_section_name": "GROUNDING_RULES",
            "strategy_name": "gepa",
            "source": "candidate",
        }
    ]
    optimizer.get_few_shot_payload.return_value = {
        "text": "Example few shot",
        "provider": "anthropic",
        "prompt_candidate_hash": None,
        "section_name": "FEW_SHOT_EXAMPLES",
        "prompt_section_name": "FEW_SHOT_EXAMPLES",
        "strategy_name": "miprov2",
        "source": "query_aware_strategy",
    }
    optimizer.get_failure_hint.return_value = None
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=optimizer,
        decision_service=None,
    )
    turn_context = SimpleNamespace(
        provider_name="anthropic",
        model="claude-sonnet",
        task_type="edit",
        last_turn_failed=False,
    )

    bundle = service.get_prompt_optimization_bundle("Fix the bug", turn_context)

    assert bundle == PromptOptimizationBundle(
        evolved_sections=["Prefer read over cat."],
        few_shots="Example few shot",
        failure_hint=None,
        identities=[
            PromptOptimizationIdentity(
                provider="anthropic",
                prompt_candidate_hash="cand-123",
                section_name="GROUNDING_RULES",
                prompt_section_name="GROUNDING_RULES",
                strategy_name="gepa",
                source="candidate",
            ),
            PromptOptimizationIdentity(
                provider="anthropic",
                prompt_candidate_hash=None,
                section_name="FEW_SHOT_EXAMPLES",
                prompt_section_name="FEW_SHOT_EXAMPLES",
                strategy_name="miprov2",
                source="query_aware_strategy",
            ),
        ],
    )
    assert bundle.to_session_metadata() == {
        "entries": [
            {
                "provider": "anthropic",
                "prompt_candidate_hash": "cand-123",
                "section_name": "GROUNDING_RULES",
                "prompt_section_name": "GROUNDING_RULES",
                "strategy_name": "gepa",
                "source": "candidate",
            },
            {
                "provider": "anthropic",
                "prompt_candidate_hash": None,
                "section_name": "FEW_SHOT_EXAMPLES",
                "prompt_section_name": "FEW_SHOT_EXAMPLES",
                "strategy_name": "miprov2",
                "source": "query_aware_strategy",
            },
        ],
        "by_section": {
            "GROUNDING_RULES": {
                "provider": "anthropic",
                "prompt_candidate_hash": "cand-123",
                "section_name": "GROUNDING_RULES",
                "prompt_section_name": "GROUNDING_RULES",
                "strategy_name": "gepa",
                "source": "candidate",
            },
            "FEW_SHOT_EXAMPLES": {
                "provider": "anthropic",
                "prompt_candidate_hash": None,
                "section_name": "FEW_SHOT_EXAMPLES",
                "prompt_section_name": "FEW_SHOT_EXAMPLES",
                "strategy_name": "miprov2",
                "source": "query_aware_strategy",
            },
        },
    }


def test_get_prompt_optimization_bundle_falls_back_when_payload_hooks_are_empty():
    optimizer = MagicMock()
    optimizer.get_evolved_section_payloads.return_value = []
    optimizer.get_few_shot_payload.return_value = {}
    optimizer.get_evolved_sections.return_value = ["Prefer read over cat."]
    optimizer.get_few_shots.return_value = "Example few shot"
    optimizer.get_failure_hint.return_value = None
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=optimizer,
        decision_service=None,
    )
    turn_context = SimpleNamespace(
        provider_name="anthropic",
        model="claude-sonnet",
        task_type="edit",
        last_turn_failed=False,
    )

    bundle = service.get_prompt_optimization_bundle("Fix the bug", turn_context)

    assert bundle == PromptOptimizationBundle(
        evolved_sections=["Prefer read over cat."],
        few_shots="Example few shot",
        failure_hint=None,
        identities=[],
    )


def test_get_prompt_optimization_bundle_includes_experiment_memory_guidance_without_optimizer(
    tmp_path,
):
    store = ExperimentMemoryStore(persist_path=tmp_path / "experiment_memory.json")
    store.record(
        ExperimentMemoryRecord(
            record_id="exp-1",
            created_at=10.0,
            scope=ExperimentScope(
                benchmark="guide",
                provider="openai",
                model="gpt-5",
            ),
            summary_metrics={},
            insights=[
                ExperimentInsight(
                    kind="environment_constraint",
                    summary="Similar runs failed because tests_pass was skipped.",
                    confidence=0.8,
                    evidence={"gate_failure": "tests_pass"},
                ),
                ExperimentInsight(
                    kind="next_candidate",
                    summary="Use read_file on the failing module, then execute_bash for pytest.",
                    confidence=0.9,
                ),
            ],
            keywords=["pytest", "read_file", "execute_bash"],
        )
    )
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=tmp_path / "runtime_evaluation_feedback.json",
        experiment_memory_path=tmp_path / "experiment_memory.json",
    )
    turn_context = SimpleNamespace(
        provider_name="openai",
        model="gpt-5",
        task_type="edit",
        last_turn_failed=False,
    )

    bundle = service.get_prompt_optimization_bundle(
        "Fix the failing pytest in the module",
        turn_context,
    )

    assert bundle.evolved_sections == []
    assert bundle.few_shots is None
    assert bundle.failure_hint is None
    assert bundle.identities == []
    assert bundle.experiment_guidance == [
        "Experiment constraint from similar runs: satisfy tests_pass before broadening the plan.",
        "Experiment-guided next candidate: Use read_file on the failing module, then execute_bash for pytest.",
    ]
    assert bundle.experiment_memory_hints == {
        "experiment_memory_match_count": 1,
        "experiment_memory_support": 0.3333,
        "experiment_memory_selection_policy_bias": 0.0,
        "experiment_memory_preferred_selection_policy": None,
        "experiment_memory_planning_policy_bias": 0.0,
        "experiment_memory_preferred_planning_policy": None,
        "experiment_memory_constraint_tags": ["tests_pass"],
        "experiment_memory_next_candidate_hints": [
            "Use read_file on the failing module, then execute_bash for pytest."
        ],
        "experiment_memory_record_ids": ["exp-1"],
    }
    assert bundle.to_session_metadata() == {
        "entries": [],
        "by_section": {},
        "experiment_memory": {
            "experiment_memory_match_count": 1,
            "experiment_memory_support": 0.3333,
            "experiment_memory_selection_policy_bias": 0.0,
            "experiment_memory_preferred_selection_policy": None,
            "experiment_memory_planning_policy_bias": 0.0,
            "experiment_memory_preferred_planning_policy": None,
            "experiment_memory_constraint_tags": ["tests_pass"],
            "experiment_memory_next_candidate_hints": [
                "Use read_file on the failing module, then execute_bash for pytest."
            ],
            "experiment_memory_record_ids": ["exp-1"],
            "prompt_guidance": [
                "Experiment constraint from similar runs: satisfy tests_pass before broadening the plan.",
                "Experiment-guided next candidate: Use read_file on the failing module, then execute_bash for pytest.",
            ],
        },
    }


def test_get_prompt_optimization_bundle_merges_optimizer_and_experiment_memory_guidance(
    tmp_path,
):
    optimizer = MagicMock()
    optimizer.get_evolved_sections.return_value = ["Prefer read over cat."]
    optimizer.get_few_shots.return_value = "Example few shot"
    optimizer.get_failure_hint.return_value = None
    store = ExperimentMemoryStore(persist_path=tmp_path / "experiment_memory.json")
    store.record(
        ExperimentMemoryRecord(
            record_id="exp-2",
            created_at=12.0,
            scope=ExperimentScope(
                benchmark="guide",
                provider="openai",
                model="gpt-5",
            ),
            summary_metrics={},
            insights=[
                ExperimentInsight(
                    kind="next_candidate",
                    summary="Verify tests_pass before widening topology.",
                    confidence=0.7,
                )
            ],
            keywords=["tests_pass", "topology"],
        )
    )
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=optimizer,
        decision_service=None,
        evaluation_feedback_path=tmp_path / "runtime_evaluation_feedback.json",
        experiment_memory_path=tmp_path / "experiment_memory.json",
    )
    turn_context = SimpleNamespace(
        provider_name="openai",
        model="gpt-5",
        task_type="edit",
        last_turn_failed=False,
    )

    bundle = service.get_prompt_optimization_bundle("Fix the bug", turn_context)

    assert bundle.evolved_sections == ["Prefer read over cat."]
    assert bundle.few_shots == "Example few shot"
    assert bundle.experiment_guidance == [
        "Experiment-guided next candidate: Verify tests_pass before widening topology."
    ]
    assert bundle.experiment_memory_hints["experiment_memory_record_ids"] == ["exp-2"]


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


def test_runtime_intelligence_exposes_topology_routing_context_from_feedback(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            completion_threshold=0.74,
            enhanced_progress_threshold=0.58,
            minimum_supported_evidence_score=0.86,
            metadata={
                "source": "benchmark_truth_feedback",
                "topology_feedback_coverage": 0.64,
                "avg_topology_reward": 0.71,
                "avg_topology_confidence": 0.79,
                "topology_final_actions": {"team_plan": 5, "single_agent": 2},
                "topology_final_kinds": {"team": 5, "single_agent": 2},
                "topology_execution_modes": {"team_execution": 5},
                "topology_providers": {"anthropic": 4, "openai": 1},
                "topology_formations": {"hierarchical": 3, "parallel": 2},
                "topology_selection_policies": {"heuristic": 2, "learned_close_override": 3},
                "topology_selection_policy_reward_totals": {
                    "heuristic": 1.1,
                    "learned_close_override": 2.4,
                },
                "topology_selection_policy_optimization_counts": {
                    "heuristic": 2,
                    "learned_close_override": 3,
                },
                "topology_selection_policy_optimization_reward_totals": {
                    "heuristic": 1.0,
                    "learned_close_override": 2.7,
                },
                "topology_selection_policy_feasible_counts": {
                    "heuristic": 1,
                    "learned_close_override": 3,
                },
                "topology_selection_policy_scope_metrics": {
                    "task_type": {
                        "edit": {
                            "policy_counts": {"heuristic": 2, "learned_close_override": 2},
                            "policy_reward_totals": {
                                "heuristic": 0.9,
                                "learned_close_override": 1.4,
                            },
                            "policy_optimization_counts": {
                                "heuristic": 2,
                                "learned_close_override": 2,
                            },
                            "policy_optimization_reward_totals": {
                                "heuristic": 0.7,
                                "learned_close_override": 1.6,
                            },
                            "policy_feasible_counts": {
                                "heuristic": 1,
                                "learned_close_override": 2,
                            },
                            "avg_reward_by_policy": {
                                "heuristic": 0.45,
                                "learned_close_override": 0.7,
                            },
                            "learned_override_reward_delta": 0.25,
                            "avg_optimization_reward_by_policy": {
                                "heuristic": 0.35,
                                "learned_close_override": 0.8,
                            },
                            "feasibility_rate_by_policy": {
                                "heuristic": 0.5,
                                "learned_close_override": 1.0,
                            },
                            "learned_override_optimization_reward_delta": 0.45,
                            "learned_override_feasibility_delta": 0.5,
                        }
                    },
                    "provider": {
                        "openai": {
                            "policy_counts": {"heuristic": 3, "learned_close_override": 3},
                            "policy_reward_totals": {
                                "heuristic": 1.2,
                                "learned_close_override": 2.4,
                            },
                            "policy_optimization_counts": {
                                "heuristic": 3,
                                "learned_close_override": 3,
                            },
                            "policy_optimization_reward_totals": {
                                "heuristic": 1.2,
                                "learned_close_override": 2.55,
                            },
                            "policy_feasible_counts": {
                                "heuristic": 1,
                                "learned_close_override": 3,
                            },
                            "avg_reward_by_policy": {
                                "heuristic": 0.4,
                                "learned_close_override": 0.8,
                            },
                            "learned_override_reward_delta": 0.4,
                            "avg_optimization_reward_by_policy": {
                                "heuristic": 0.4,
                                "learned_close_override": 0.85,
                            },
                            "feasibility_rate_by_policy": {
                                "heuristic": 0.3333,
                                "learned_close_override": 1.0,
                            },
                            "learned_override_optimization_reward_delta": 0.45,
                            "learned_override_feasibility_delta": 0.6667,
                        }
                    },
                    "model_family": {
                        "gpt": {
                            "policy_counts": {"heuristic": 4, "learned_close_override": 5},
                            "policy_reward_totals": {
                                "heuristic": 1.6,
                                "learned_close_override": 4.25,
                            },
                            "policy_optimization_counts": {
                                "heuristic": 4,
                                "learned_close_override": 5,
                            },
                            "policy_optimization_reward_totals": {
                                "heuristic": 1.8,
                                "learned_close_override": 4.6,
                            },
                            "policy_feasible_counts": {
                                "heuristic": 2,
                                "learned_close_override": 5,
                            },
                            "avg_reward_by_policy": {
                                "heuristic": 0.4,
                                "learned_close_override": 0.85,
                            },
                            "learned_override_reward_delta": 0.45,
                            "avg_optimization_reward_by_policy": {
                                "heuristic": 0.45,
                                "learned_close_override": 0.92,
                            },
                            "feasibility_rate_by_policy": {
                                "heuristic": 0.5,
                                "learned_close_override": 1.0,
                            },
                            "learned_override_optimization_reward_delta": 0.47,
                            "learned_override_feasibility_delta": 0.5,
                        }
                    },
                },
            },
        ),
        path=tmp_path / "runtime_evaluation_feedback.json",
    )

    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=feedback_path,
        evaluation_feedback_scope=RuntimeEvaluationFeedbackScope(
            provider="openai",
            model="gpt-5",
            task_type="edit",
        ),
    )

    feedback = service.get_topology_routing_feedback()
    hints = service.get_topology_routing_context(
        scope_context={
            "task_type": "edit",
            "provider_hint": "openai",
            "model": "gpt-5",
        }
    )

    assert feedback is not None
    assert feedback.preferred_action == "team_plan"
    assert feedback.preferred_topology == "team"
    assert feedback.preferred_provider == "anthropic"
    assert feedback.preferred_formation == "hierarchical"
    assert feedback.support > 0.0
    assert hints["learned_topology_action"] == "team_plan"
    assert hints["learned_topology_kind"] == "team"
    assert hints["learned_provider_hint"] == "anthropic"
    assert hints["learned_formation_hint"] == "hierarchical"
    assert hints["learned_override_policy_scope_dimension"] == "model_family"
    assert hints["learned_override_policy_scope_label"] == "gpt"
    assert hints["learned_override_policy_count"] == 5
    assert hints["heuristic_policy_count"] == 4
    assert hints["learned_override_policy_reward"] == pytest.approx(0.85)
    assert hints["heuristic_policy_reward"] == pytest.approx(0.4)
    assert hints["learned_override_policy_reward_delta"] == pytest.approx(0.45)
    assert hints["learned_override_policy_optimization_reward"] == pytest.approx(0.92)
    assert hints["heuristic_policy_optimization_reward"] == pytest.approx(0.45)
    assert hints["learned_override_policy_optimization_reward_delta"] == pytest.approx(0.47)
    assert hints["learned_override_policy_feasibility_rate"] == pytest.approx(1.0)
    assert hints["heuristic_policy_feasibility_rate"] == pytest.approx(0.5)
    assert hints["learned_override_policy_feasibility_delta"] == pytest.approx(0.5)


def test_runtime_intelligence_exposes_team_routing_hints_from_feedback(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "source": "benchmark_truth_feedback",
                "team_feedback_coverage": 0.8,
                "tasks_with_team_feedback": 4,
                "team_formations": {"parallel": 3, "hierarchical": 1},
                "team_merge_risk_levels": {"low": 3, "medium": 1},
                "team_worktree_plan_count": 4,
                "team_worktree_materialized_count": 3,
                "team_low_risk_task_count": 3,
                "team_medium_risk_task_count": 1,
                "team_high_risk_task_count": 0,
                "team_merge_conflict_task_count": 0,
                "team_cleanup_error_task_count": 0,
                "avg_team_assignments": 3.0,
                "avg_team_scoped_members": 3.0,
                "avg_team_members_with_changes": 2.5,
                "avg_team_changed_file_count": 4.5,
            }
        ),
        path=tmp_path / "runtime_evaluation_feedback.json",
    )

    perception = SimpleNamespace(task_analysis=MagicMock(task_type="design"), confidence=0.81)
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=SimpleNamespace(
            perceive=AsyncMock(return_value=perception),
            evaluation_policy=RuntimeEvaluationPolicy(),
            config={},
        ),
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=feedback_path,
    )

    feedback = service.get_team_routing_feedback()
    hints = service.get_topology_routing_context(
        scope_context={"task_type": "design", "provider": "openai", "model": "gpt-5"}
    )

    assert feedback is not None
    assert feedback.preferred_formation == "parallel"
    assert feedback.recommended_max_workers == 3
    assert feedback.recommends_worktree_isolation is True
    assert feedback.recommends_materialized_worktrees is True
    assert hints["learned_team_support"] > 0.0
    assert hints["learned_formation_hint"] == "parallel"
    assert hints["learned_team_max_workers_hint"] == 3
    assert hints["learned_worktree_isolation_hint"] is True
    assert hints["learned_materialize_worktrees_hint"] is True


def test_runtime_intelligence_builds_structured_routing_policy(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "source": "benchmark_truth_feedback",
                "topology_feedback_coverage": 0.7,
                "avg_topology_reward": 0.74,
                "avg_topology_confidence": 0.79,
                "topology_final_actions": {"team_plan": 4},
                "topology_final_kinds": {"team": 4},
                "topology_execution_modes": {"team_execution": 4},
                "topology_providers": {"anthropic": 4},
                "topology_formations": {"parallel": 4},
                "team_feedback_coverage": 0.8,
                "tasks_with_team_feedback": 4,
                "team_formations": {"parallel": 4},
                "team_worktree_plan_count": 4,
                "team_worktree_materialized_count": 3,
                "team_low_risk_task_count": 3,
                "avg_team_assignments": 3.0,
                "avg_team_scoped_members": 3.0,
                "degradation_feedback_coverage": 0.7,
                "degradation_event_count": 4,
                "degraded_task_count": 3,
                "avg_degradation_drift_score": 0.63,
                "degradation_drift_rate": 0.66,
                "degradation_intervention_rate": 0.33,
                "degradation_stability_score": 0.4,
                "degradation_sources": {"provider_performance": 4},
                "degradation_kinds": {"persistent_provider_degradation": 2},
            }
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

    policy = service.get_structured_routing_policy(
        query="inspect the degraded provider path",
        scope_context={"task_type": "analysis", "provider": "openai", "model": "gpt-5"},
    )

    assert policy.scope_context["task_type"] == "analysis"
    assert policy.topology_hints["learned_topology_action"] == "team_plan"
    assert policy.team_hints["learned_worktree_isolation_hint"] is True
    assert policy.degradation_hints["learned_degradation_conservative_routing_hint"] is True
    assert policy.selector_context()["learned_provider_hint"] == "anthropic"
    assert "planning_force_llm" not in policy.selector_context()


def test_runtime_intelligence_team_feedback_prefers_safer_parallelism_when_risk_is_high(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "source": "benchmark_truth_feedback",
                "team_feedback_coverage": 0.75,
                "tasks_with_team_feedback": 4,
                "team_formations": {"parallel": 4},
                "team_merge_risk_levels": {"high": 3, "medium": 1},
                "team_worktree_plan_count": 4,
                "team_worktree_materialized_count": 4,
                "team_low_risk_task_count": 0,
                "team_medium_risk_task_count": 1,
                "team_high_risk_task_count": 3,
                "team_merge_conflict_task_count": 2,
                "team_cleanup_error_task_count": 1,
                "avg_team_assignments": 4.0,
                "avg_team_scoped_members": 4.0,
            }
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

    feedback = service.get_team_routing_feedback()
    hints = service.get_topology_routing_context(scope_context={"task_type": "design"})

    assert feedback is not None
    assert feedback.preferred_formation == "hierarchical"
    assert feedback.recommended_max_workers == 2
    assert feedback.recommends_worktree_isolation is True
    assert feedback.recommends_materialized_worktrees is True
    assert hints["learned_formation_hint"] == "hierarchical"
    assert hints["learned_team_max_workers_hint"] == 2
    assert hints["learned_team_risk_score"] >= 0.5
    assert hints["learned_worktree_isolation_hint"] is True
    assert hints["learned_materialize_worktrees_hint"] is True


def test_runtime_intelligence_prefers_scoped_dry_run_worktree_policy(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "source": "benchmark_truth_feedback",
                "team_feedback_coverage": 0.8,
                "tasks_with_team_feedback": 4,
                "team_formations": {"parallel": 4},
                "team_merge_risk_levels": {"low": 3, "medium": 1},
                "team_worktree_plan_count": 4,
                "team_worktree_materialized_count": 4,
                "team_low_risk_task_count": 3,
                "team_medium_risk_task_count": 1,
                "team_high_risk_task_count": 0,
                "team_merge_conflict_task_count": 0,
                "team_cleanup_task_count": 4,
                "team_cleanup_error_task_count": 0,
                "avg_team_assignments": 3.0,
                "avg_team_scoped_members": 3.0,
                "team_worktree_scope_metrics": {
                    "task_type": {
                        "analysis": {
                            "tasks_with_team_feedback": 4.0,
                            "team_feedback_coverage": 0.8,
                            "team_worktree_plan_count": 4.0,
                            "team_worktree_materialized_count": 1.0,
                            "team_worktree_dry_run_count": 3.0,
                            "team_cleanup_task_count": 3.0,
                            "team_cleanup_error_task_count": 2.0,
                            "team_merge_conflict_task_count": 1.0,
                            "team_low_risk_task_count": 1.0,
                            "team_medium_risk_task_count": 1.0,
                            "team_high_risk_task_count": 2.0,
                            "avg_team_assignments": 3.0,
                            "avg_team_scoped_members": 3.0,
                            "avg_team_members_with_changes": 2.0,
                            "avg_team_changed_file_count": 4.0,
                            "team_formations": {"parallel": 3.0},
                            "team_merge_risk_levels": {
                                "high": 2.0,
                                "medium": 1.0,
                                "low": 1.0,
                            },
                        }
                    }
                },
            }
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

    feedback = service.resolve_team_routing_feedback(scope_context={"task_type": "analysis"})
    hints = service.get_topology_routing_context(scope_context={"task_type": "analysis"})

    assert feedback is not None
    assert feedback.scope_dimension == "task_type"
    assert feedback.scope_label == "analysis"
    assert feedback.recommends_worktree_isolation is True
    assert feedback.recommends_dry_run_worktrees is True
    assert feedback.recommended_cleanup_worktrees is False
    assert hints["learned_team_policy_scope_dimension"] == "task_type"
    assert hints["learned_team_policy_scope_label"] == "analysis"
    assert hints["learned_worktree_isolation_hint"] is True
    assert hints["learned_dry_run_worktrees_hint"] is True
    assert hints["learned_materialize_worktrees_hint"] is False
    assert hints["learned_cleanup_worktrees_hint"] is False


def test_runtime_intelligence_exposes_degradation_routing_hints_from_feedback(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "source": "benchmark_truth_feedback",
                "degradation_feedback_coverage": 0.85,
                "degradation_event_count": 6,
                "degraded_task_count": 4,
                "recovered_task_count": 2,
                "degradation_recovery_rate": 0.5,
                "avg_degradation_adaptation_cost": 2.2,
                "avg_degradation_time_to_recover_seconds": 4.5,
                "avg_degradation_cost_variance": 1.1,
                "avg_degradation_recovery_time_variance": 0.9,
                "avg_degradation_intervention_count": 1.8,
                "avg_degradation_confidence": 0.35,
                "avg_degradation_drift_score": 0.72,
                "content_degradation_task_count": 1,
                "confidence_degradation_task_count": 2,
                "provider_degradation_task_count": 3,
                "persistent_degradation_task_count": 1,
                "drift_task_count": 4,
                "degradation_drift_rate": 0.8,
                "degradation_intervention_task_count": 3,
                "degradation_intervention_rate": 0.6,
                "high_adaptation_cost_task_count": 2,
                "degradation_high_cost_rate": 0.4,
                "degradation_confidence_rate": 0.5,
                "degradation_stability_score": 0.28,
                "degradation_sources": {
                    "provider_performance": 4,
                    "streaming_confidence": 2,
                },
                "degradation_kinds": {
                    "persistent_provider_degradation": 2,
                    "confidence_early_stop": 2,
                    "recovery_action": 2,
                },
                "degradation_failure_types": {
                    "PROVIDER_ERROR": 4,
                    "CONFIDENCE_LOW": 2,
                },
                "degradation_providers": {"ollama": 4},
                "degradation_reasons": {
                    "failure_streak": 3,
                    "confidence_threshold_reached": 2,
                },
            }
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

    feedback = service.get_degradation_routing_feedback()
    hints = service.get_topology_routing_context(scope_context={"task_type": "analysis"})

    assert feedback is not None
    assert feedback.dominant_source == "provider_performance"
    assert feedback.dominant_kind == "persistent_provider_degradation"
    assert feedback.dominant_provider == "ollama"
    assert feedback.recommends_conservative_routing is True
    assert feedback.recommends_recovery_budget_buffer is True
    assert hints["learned_degradation_support"] > 0.0
    assert hints["learned_degradation_severity_score"] > 0.4
    assert hints["learned_degradation_drift_rate"] == pytest.approx(0.8)
    assert hints["learned_degradation_intervention_rate"] == pytest.approx(0.6)
    assert hints["learned_degradation_high_cost_rate"] == pytest.approx(0.4)
    assert hints["learned_degradation_dominant_source"] == "provider_performance"
    assert hints["learned_degradation_dominant_kind"] == "persistent_provider_degradation"
    assert hints["learned_degradation_dominant_provider"] == "ollama"
    assert hints["learned_degradation_conservative_routing_hint"] is True
    assert hints["learned_degradation_recovery_buffer_hint"] is True


def test_runtime_intelligence_falls_back_to_task_type_scoped_policy_metrics(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            completion_threshold=0.74,
            enhanced_progress_threshold=0.58,
            minimum_supported_evidence_score=0.86,
            metadata={
                "source": "benchmark_truth_feedback",
                "topology_feedback_coverage": 0.64,
                "avg_topology_reward": 0.71,
                "avg_topology_confidence": 0.79,
                "topology_final_actions": {"team_plan": 5, "single_agent": 2},
                "topology_final_kinds": {"team": 5, "single_agent": 2},
                "topology_execution_modes": {"team_execution": 5},
                "topology_selection_policies": {"heuristic": 2, "learned_close_override": 3},
                "topology_selection_policy_reward_totals": {
                    "heuristic": 1.1,
                    "learned_close_override": 2.4,
                },
                "topology_selection_policy_optimization_counts": {
                    "heuristic": 2,
                    "learned_close_override": 2,
                },
                "topology_selection_policy_optimization_reward_totals": {
                    "heuristic": 0.8,
                    "learned_close_override": 1.6,
                },
                "topology_selection_policy_feasible_counts": {
                    "heuristic": 1,
                    "learned_close_override": 2,
                },
                "topology_selection_policy_scope_metrics": {
                    "task_type": {
                        "analysis": {
                            "policy_counts": {"heuristic": 2, "learned_close_override": 2},
                            "policy_reward_totals": {
                                "heuristic": 0.8,
                                "learned_close_override": 1.6,
                            },
                            "policy_optimization_counts": {
                                "heuristic": 2,
                                "learned_close_override": 2,
                            },
                            "policy_optimization_reward_totals": {
                                "heuristic": 0.8,
                                "learned_close_override": 1.7,
                            },
                            "policy_feasible_counts": {
                                "heuristic": 1,
                                "learned_close_override": 2,
                            },
                            "avg_reward_by_policy": {
                                "heuristic": 0.4,
                                "learned_close_override": 0.8,
                            },
                            "learned_override_reward_delta": 0.4,
                            "avg_optimization_reward_by_policy": {
                                "heuristic": 0.4,
                                "learned_close_override": 0.85,
                            },
                            "feasibility_rate_by_policy": {
                                "heuristic": 0.5,
                                "learned_close_override": 1.0,
                            },
                            "learned_override_optimization_reward_delta": 0.45,
                            "learned_override_feasibility_delta": 0.5,
                        }
                    }
                },
            },
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

    hints = service.get_topology_routing_context(scope_context={"task_type": "analysis"})

    assert hints["learned_override_policy_scope_dimension"] == "task_type"
    assert hints["learned_override_policy_scope_label"] == "analysis"
    assert hints["learned_override_policy_reward_delta"] == pytest.approx(0.4)
    assert hints["learned_override_policy_optimization_reward_delta"] == pytest.approx(0.45)
    assert hints["learned_override_policy_feasibility_delta"] == pytest.approx(0.5)


def test_runtime_intelligence_exposes_experiment_memory_routing_hints(tmp_path):
    store = ExperimentMemoryStore(persist_path=tmp_path / "experiment_memory.json")
    store.record(
        ExperimentMemoryRecord(
            record_id="guide-memory-1",
            created_at=10.0,
            scope=ExperimentScope(
                benchmark="guide",
                provider="openai",
                model="gpt-5",
                prompt_candidate_hash="cand-123",
                section_name="GROUNDING_RULES",
            ),
            summary_metrics={
                "topology_learned_override_optimization_reward_delta": -0.3,
            },
            task_summaries=[],
            insights=[
                ExperimentInsight(
                    kind="failed_hypothesis",
                    summary="Learned close override underperformed heuristic routing for this scope.",
                    confidence=0.9,
                    evidence={"selection_policy": "learned_close_override"},
                ),
                ExperimentInsight(
                    kind="environment_constraint",
                    summary="Repeated hard constraint failure: tests_pass.",
                    confidence=0.8,
                    evidence={"gate_failure": "tests_pass", "count": 2},
                ),
                ExperimentInsight(
                    kind="next_candidate",
                    summary="Tighten learned override thresholds for this scope.",
                    confidence=0.76,
                ),
            ],
            keywords=["learned_close_override", "heuristic", "tests_pass", "guide"],
        )
    )
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=tmp_path / "runtime_evaluation_feedback.json",
        evaluation_feedback_scope=RuntimeEvaluationFeedbackScope(
            provider="openai",
            model="gpt-5",
            task_type="edit",
        ),
    )

    hints = service.get_topology_routing_context(
        query="Fix the guide flow and verify tests pass",
        scope_context={
            "provider": "openai",
            "model": "gpt-5",
            "prompt_candidate_hash": "cand-123",
            "section_name": "GROUNDING_RULES",
        },
    )

    assert hints["experiment_memory_match_count"] == 1
    assert hints["experiment_memory_support"] > 0.0
    assert hints["experiment_memory_preferred_selection_policy"] == "heuristic"
    assert hints["experiment_memory_selection_policy_bias"] < 0.0
    assert hints["experiment_memory_constraint_tags"] == ["tests_pass"]
    assert (
        "Tighten learned override thresholds" in hints["experiment_memory_next_candidate_hints"][0]
    )


def test_runtime_intelligence_exposes_planning_force_hints_from_experiment_constraints(
    tmp_path,
):
    store = ExperimentMemoryStore(persist_path=tmp_path / "experiment_memory.json")
    store.record(
        ExperimentMemoryRecord(
            record_id="guide-memory-2",
            created_at=20.0,
            scope=ExperimentScope(
                benchmark="guide",
                provider="openai",
                model="gpt-5",
            ),
            summary_metrics={},
            task_summaries=[],
            insights=[
                ExperimentInsight(
                    kind="environment_constraint",
                    summary="Repeated hard constraint failure: tests_pass.",
                    confidence=0.82,
                    evidence={"gate_failure": "tests_pass", "count": 3},
                ),
                ExperimentInsight(
                    kind="next_candidate",
                    summary="Verify tests_pass before widening topology.",
                    confidence=0.74,
                ),
            ],
            keywords=["tests_pass", "guide"],
        )
    )
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=tmp_path / "runtime_evaluation_feedback.json",
        experiment_memory_path=tmp_path / "experiment_memory.json",
    )

    hints = service.get_planning_routing_context(
        query="run the tests",
        scope_context={"provider": "openai", "model": "gpt-5", "task_type": "action"},
    )

    assert hints["planning_force_llm"] is True
    assert hints["planning_force_reason"] == "experiment_constraints: tests_pass"
    assert hints["planning_constraint_tags"] == ["tests_pass"]
    assert hints["planning_experiment_support"] > 0.0
    assert hints["planning_match_count"] == 1
    assert (
        "Verify tests_pass before widening topology." in hints["planning_next_candidate_hints"][0]
    )


def test_runtime_intelligence_exposes_planning_policy_bias_from_experiment_memory(tmp_path):
    store = ExperimentMemoryStore(persist_path=tmp_path / "experiment_memory.json")
    store.record(
        ExperimentMemoryRecord(
            record_id="guide-memory-3",
            created_at=30.0,
            scope=ExperimentScope(
                benchmark="guide",
                provider="openai",
                model="gpt-5",
            ),
            summary_metrics={},
            task_summaries=[],
            insights=[
                ExperimentInsight(
                    kind="successful_transformation",
                    summary="Forced LLM planning outperformed heuristic fast-path for this scope.",
                    confidence=0.86,
                    evidence={"completion_delta": 0.42},
                )
            ],
            keywords=["planning", "fast_path", "llm_planning"],
        )
    )
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=tmp_path / "runtime_evaluation_feedback.json",
        experiment_memory_path=tmp_path / "experiment_memory.json",
    )

    experiment_hints = service.get_experiment_routing_context(
        query="design the fix",
        scope_context={"provider": "openai", "model": "gpt-5"},
    )
    planning_hints = service.get_planning_routing_context(
        query="design the fix",
        scope_context={"provider": "openai", "model": "gpt-5", "task_type": "analysis"},
    )

    assert experiment_hints["experiment_memory_planning_policy_bias"] > 0.0
    assert experiment_hints["experiment_memory_preferred_planning_policy"] == (
        "experiment_forced_slow_path"
    )
    assert planning_hints["planning_preferred_policy"] == "experiment_forced_slow_path"
    assert planning_hints["planning_force_llm"] is True
    assert planning_hints["planning_force_reason"] == (
        "experiment_policy_bias: experiment_forced_slow_path"
    )


def test_runtime_intelligence_exposes_fast_path_preference_from_negative_planning_bias(tmp_path):
    store = ExperimentMemoryStore(persist_path=tmp_path / "experiment_memory.json")
    store.record(
        ExperimentMemoryRecord(
            record_id="guide-memory-4",
            created_at=40.0,
            scope=ExperimentScope(
                benchmark="guide",
                provider="openai",
                model="gpt-5",
            ),
            summary_metrics={},
            task_summaries=[],
            insights=[
                ExperimentInsight(
                    kind="failed_hypothesis",
                    summary="Forced LLM planning underperformed heuristic fast-path for this scope.",
                    confidence=0.9,
                    evidence={"completion_delta": -0.31},
                )
            ],
            keywords=["planning", "fast_path", "llm_planning"],
        )
    )
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=tmp_path / "runtime_evaluation_feedback.json",
        experiment_memory_path=tmp_path / "experiment_memory.json",
    )

    planning_hints = service.get_planning_routing_context(
        query="run the tests",
        scope_context={"provider": "openai", "model": "gpt-5", "task_type": "action"},
    )

    assert planning_hints["planning_preferred_policy"] == "heuristic_fast_path"
    assert planning_hints["planning_policy_bias"] < 0.0
    assert planning_hints["planning_prefer_fast_path"] is True
    assert planning_hints["planning_prefer_reason"] == (
        "experiment_policy_bias: heuristic_fast_path"
    )
    assert planning_hints["planning_fast_path_tool_budget_limit"] >= 4
    assert planning_hints["planning_fast_path_query_length_limit"] > 50
    assert planning_hints["planning_fast_path_complexity_threshold"] > 0.3


def test_runtime_intelligence_records_live_topology_outcomes_before_steering(tmp_path):
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=tmp_path / "runtime_evaluation_feedback.json",
    )
    payload = {
        "status": "completed",
        "completion_score": 0.83,
        "tool_calls": 3,
        "turns": 2,
        "topology_events": [
            {
                "action": "team_plan",
                "topology": "team",
                "execution_mode": "team_execution",
                "provider": "anthropic",
                "formation": "hierarchical",
                "confidence": 0.78,
                "outcome": {"runtime": "batch"},
            }
        ],
    }

    first_feedback = service.record_topology_outcome(payload)

    assert first_feedback is not None
    assert first_feedback.observation_count == 1
    assert service.get_topology_routing_context() == {}

    second_feedback = service.record_topology_outcome(payload)
    hints = service.get_topology_routing_context()

    assert second_feedback is not None
    assert second_feedback.observation_count == 2
    assert second_feedback.preferred_action == "team_plan"
    assert second_feedback.preferred_topology == "team"
    assert hints["learned_topology_action"] == "team_plan"
    assert hints["learned_provider_hint"] == "anthropic"
    assert hints["learned_topology_observation_count"] == 2


def test_runtime_intelligence_reloads_scoped_live_topology_feedback_across_sessions(tmp_path):
    scope = RuntimeEvaluationFeedbackScope(
        project="codingagent",
        provider="openai",
        model="gpt-5",
        task_type="edit",
    )
    feedback_path = tmp_path / "runtime_evaluation_feedback.json"
    payload = {
        "status": "completed",
        "completion_score": 0.84,
        "tool_calls": 4,
        "turns": 2,
        "topology_events": [
            {
                "action": "team_plan",
                "topology": "team",
                "execution_mode": "team_execution",
                "provider": "anthropic",
                "formation": "parallel",
                "confidence": 0.8,
                "outcome": {"runtime": "batch"},
            }
        ],
    }
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=feedback_path,
        evaluation_feedback_scope=scope,
    )

    service.record_topology_outcome(payload)
    service.record_topology_outcome(payload)

    persisted_files = list(tmp_path.glob("runtime_topology_feedback.*.json"))
    assert len(persisted_files) == 1

    reloaded = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=feedback_path,
        evaluation_feedback_scope=scope,
    )
    hints = reloaded.get_topology_routing_context()

    assert hints["learned_topology_action"] == "team_plan"
    assert hints["learned_topology_kind"] == "team"
    assert hints["learned_provider_hint"] == "anthropic"
    assert hints["learned_formation_hint"] == "parallel"
    assert hints["learned_topology_observation_count"] >= 2


def test_runtime_intelligence_with_conflicted_feedback_withholds_soft_hints(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            completion_threshold=0.74,
            enhanced_progress_threshold=0.58,
            minimum_supported_evidence_score=0.86,
            metadata={
                "source": "benchmark_truth_feedback",
                "topology_feedback_coverage": 0.78,
                "avg_topology_reward": 0.74,
                "avg_topology_confidence": 0.79,
                "topology_final_actions": {"team_plan": 5, "single_agent": 4},
                "topology_final_kinds": {"team": 5, "single_agent": 4},
                "topology_execution_modes": {"team_execution": 5, "single_agent": 4},
                "topology_providers": {"anthropic": 5, "openai": 4},
                "topology_formations": {"parallel": 5, "hierarchical": 4},
                "task_count": 9,
            },
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

    feedback = service.get_topology_routing_feedback()
    hints = service.get_topology_routing_context()

    assert feedback is not None
    assert feedback.action_agreement == pytest.approx(5 / 9, rel=1e-3)
    assert feedback.conflict_score > 0.4
    assert hints["learned_topology_support"] > 0.0
    assert hints["learned_topology_conflict_score"] > 0.4
    assert "learned_topology_action" not in hints
    assert "learned_provider_hint" not in hints
    assert "learned_formation_hint" not in hints


@pytest.mark.asyncio
async def test_analyze_turn_includes_topology_feedback_metadata(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            completion_threshold=0.74,
            enhanced_progress_threshold=0.58,
            minimum_supported_evidence_score=0.86,
            metadata={
                "source": "benchmark_truth_feedback",
                "topology_feedback_coverage": 0.61,
                "avg_topology_reward": 0.68,
                "topology_final_actions": {"single_agent": 3},
                "topology_final_kinds": {"single_agent": 3},
            },
        ),
        path=tmp_path / "runtime_evaluation_feedback.json",
    )
    perception = SimpleNamespace(task_analysis=MagicMock(task_type="edit"), confidence=0.81)
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=SimpleNamespace(
            perceive=AsyncMock(return_value=perception),
            evaluation_policy=RuntimeEvaluationPolicy(),
            config={},
        ),
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=feedback_path,
    )

    snapshot = await service.analyze_turn("Fix the bug")

    assert snapshot.metadata["topology_feedback"]["preferred_action"] == "single_agent"
    assert snapshot.metadata["topology_routing_hints"]["learned_topology_action"] == (
        "single_agent"
    )
    assert (
        snapshot.metadata["structured_routing_policy"]["selector_context"][
            "learned_topology_action"
        ]
        == "single_agent"
    )


@pytest.mark.asyncio
async def test_analyze_turn_includes_degradation_feedback_metadata(tmp_path):
    feedback_path = save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            completion_threshold=0.74,
            enhanced_progress_threshold=0.58,
            minimum_supported_evidence_score=0.86,
            metadata={
                "source": "benchmark_truth_feedback",
                "degradation_feedback_coverage": 0.9,
                "degradation_event_count": 4,
                "degraded_task_count": 3,
                "recovered_task_count": 1,
                "degradation_recovery_rate": 0.3333,
                "avg_degradation_adaptation_cost": 2.1,
                "avg_degradation_time_to_recover_seconds": 4.2,
                "avg_degradation_cost_variance": 1.0,
                "avg_degradation_intervention_count": 1.3,
                "avg_degradation_drift_score": 0.69,
                "degradation_drift_rate": 0.75,
                "degradation_intervention_rate": 0.5,
                "degradation_high_cost_rate": 0.33,
                "degradation_stability_score": 0.31,
                "persistent_degradation_task_count": 1,
                "degradation_sources": {"provider_performance": 3, "agentic_loop": 1},
                "degradation_kinds": {
                    "persistent_provider_degradation": 2,
                    "content_repetition": 1,
                    "recovery_action": 1,
                },
                "degradation_providers": {"ollama": 3},
            },
        ),
        path=tmp_path / "runtime_evaluation_feedback.json",
    )
    perception = SimpleNamespace(task_analysis=MagicMock(task_type="edit"), confidence=0.81)
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=SimpleNamespace(
            perceive=AsyncMock(return_value=perception),
            evaluation_policy=RuntimeEvaluationPolicy(),
            config={},
        ),
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=feedback_path,
    )

    snapshot = await service.analyze_turn("Fix the flaky provider path")

    assert snapshot.metadata["degradation_feedback"]["dominant_source"] == "provider_performance"
    assert snapshot.metadata["degradation_feedback"]["severity_score"] > 0.4
    assert (
        snapshot.metadata["topology_routing_hints"]["learned_degradation_conservative_routing_hint"]
        is True
    )


@pytest.mark.asyncio
async def test_analyze_turn_includes_experiment_memory_hints(tmp_path):
    store = ExperimentMemoryStore(persist_path=tmp_path / "experiment_memory.json")
    store.record(
        ExperimentMemoryRecord(
            record_id="guide-memory-2",
            created_at=20.0,
            scope=ExperimentScope(
                benchmark="guide",
                provider="openai",
                model="gpt-5",
                prompt_candidate_hash="cand-789",
                section_name="GROUNDING_RULES",
            ),
            summary_metrics={
                "topology_learned_override_optimization_reward_delta": -0.22,
            },
            task_summaries=[],
            insights=[
                ExperimentInsight(
                    kind="failed_hypothesis",
                    summary="Learned close override underperformed heuristic routing for this scope.",
                    confidence=0.88,
                )
            ],
            keywords=["learned_close_override", "heuristic", "guide"],
        )
    )
    perception = SimpleNamespace(task_analysis=MagicMock(task_type="edit"), confidence=0.81)
    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=SimpleNamespace(
            perceive=AsyncMock(return_value=perception),
            evaluation_policy=RuntimeEvaluationPolicy(),
            config={},
        ),
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=tmp_path / "runtime_evaluation_feedback.json",
        evaluation_feedback_scope=RuntimeEvaluationFeedbackScope(
            provider="openai",
            model="gpt-5",
            task_type="edit",
        ),
    )

    snapshot = await service.analyze_turn(
        "Fix the guide issue",
        context={
            "provider": "openai",
            "model": "gpt-5",
            "prompt_candidate_hash": "cand-789",
            "section_name": "GROUNDING_RULES",
        },
    )

    assert snapshot.metadata["experiment_memory_hints"]["experiment_memory_match_count"] == 1
    assert snapshot.metadata["experiment_memory_hints"][
        "experiment_memory_preferred_selection_policy"
    ] == ("heuristic")
    assert (
        snapshot.metadata["topology_routing_hints"]["experiment_memory_selection_policy_bias"] < 0.0
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


def test_runtime_intelligence_loads_aggregated_validated_feedback_from_results_dir(tmp_path):
    (tmp_path / "eval_guide_20260401_010101.json").write_text(
        json.dumps(
            {
                "runtime_evaluation_feedback": {
                    "completion_threshold": 0.88,
                    "enhanced_progress_threshold": 0.71,
                    "minimum_supported_evidence_score": 0.9,
                    "metadata": {
                        "source": "benchmark_truth_feedback",
                        "validated_evaluation_truth": True,
                        "truth_alignment_rate": 0.94,
                        "task_count": 20,
                        "saved_at": "2026-04-01T00:00:00+00:00",
                    },
                }
            }
        )
    )
    (tmp_path / "eval_session_20260425_010101.json").write_text(
        json.dumps(
            {
                "runtime_evaluation_feedback": {
                    "completion_threshold": 0.72,
                    "enhanced_progress_threshold": 0.55,
                    "minimum_supported_evidence_score": 0.81,
                    "metadata": {
                        "source": "validated_session_truth_feedback",
                        "validated_evaluation_truth": True,
                        "truth_alignment_rate": 0.9,
                        "task_count": 14,
                        "saved_at": "2026-04-25T00:00:00+00:00",
                    },
                }
            }
        )
    )

    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=tmp_path / "runtime_evaluation_feedback.json",
    )

    assert service.evaluation_policy.completion_threshold is not None
    assert 0.72 < service.evaluation_policy.completion_threshold < 0.84
    assert service.evaluation_policy.enhanced_progress_threshold is not None
    assert 0.55 <= service.evaluation_policy.enhanced_progress_threshold < 0.66


def test_runtime_intelligence_prefers_scope_adjacent_validated_feedback(tmp_path):
    (tmp_path / "eval_runtime_20260425_010101.json").write_text(
        json.dumps(
            build_validated_session_feedback_payload(
                RuntimeEvaluationFeedback(
                    completion_threshold=0.61,
                    enhanced_progress_threshold=0.48,
                    minimum_supported_evidence_score=0.71,
                ),
                scope=RuntimeEvaluationFeedbackScope(
                    project="other-project",
                    provider="anthropic",
                    model="claude-sonnet",
                    task_type="analyze",
                ),
                metadata={
                    "truth_alignment_rate": 0.95,
                    "task_count": 18,
                    "saved_at": "2026-04-25T00:00:00+00:00",
                },
            )
        )
    )
    (tmp_path / "eval_runtime_20260420_010101.json").write_text(
        json.dumps(
            build_validated_session_feedback_payload(
                RuntimeEvaluationFeedback(
                    completion_threshold=0.82,
                    enhanced_progress_threshold=0.67,
                    minimum_supported_evidence_score=0.87,
                ),
                scope=RuntimeEvaluationFeedbackScope(
                    project="codingagent",
                    provider="openai",
                    model="gpt-5",
                    task_type="edit",
                ),
                metadata={
                    "truth_alignment_rate": 0.88,
                    "task_count": 9,
                    "saved_at": "2026-04-20T00:00:00+00:00",
                },
            )
        )
    )

    service = RuntimeIntelligenceService(
        task_analyzer=MagicMock(),
        perception_integration=None,
        optimization_injector=None,
        decision_service=None,
        evaluation_feedback_path=tmp_path / "runtime_evaluation_feedback.json",
        evaluation_feedback_scope=RuntimeEvaluationFeedbackScope(
            project="codingagent",
            provider="openai",
            model="gpt-5",
            task_type="edit",
        ),
    )

    assert service.evaluation_policy.completion_threshold is not None
    assert service.evaluation_policy.completion_threshold > 0.73
    assert service.evaluation_policy.enhanced_progress_threshold is not None
    assert service.evaluation_policy.enhanced_progress_threshold > 0.58
