from __future__ import annotations

import json
from pathlib import Path

import pytest

from victor.evaluation.runtime_feedback import (
    AGGREGATED_SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE,
    SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE,
    RuntimeEvaluationFeedbackScope,
    build_browser_validated_session_feedback_payload,
    build_deep_research_validated_session_feedback_payload,
    build_swe_bench_validated_session_feedback_payload,
    build_validated_session_feedback_payload,
    derive_runtime_evaluation_feedback,
    load_runtime_evaluation_feedback,
    save_session_topology_runtime_feedback,
    save_runtime_evaluation_feedback,
)
from victor.evaluation.baseline_validator import (
    BaselineStatus,
    BaselineValidationResult,
    TestBaseline,
)
from victor.evaluation.result_correlation import SWEBenchScore
from victor.evaluation.test_runners import TestRunResults
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationFeedback


def test_derive_runtime_feedback_raises_threshold_for_overconfident_failures():
    payload = {
        "config": {
            "benchmark": "dr3_eval",
            "model": "test-model",
            "provider": "anthropic",
            "prompt_candidate_hash": "cand-123",
            "section_name": "GROUNDING_RULES",
        },
        "summary": {
            "truth_alignment_rate": 0.8,
            "overconfidence_rate": 0.4,
            "underconfidence_rate": 0.0,
            "topology_feedback_coverage": 0.5,
            "avg_topology_reward": 0.64,
            "avg_topology_confidence": 0.73,
            "degradation_feedback_coverage": 0.5,
            "degradation_event_count": 1,
            "degraded_task_count": 1,
            "recovered_task_count": 0,
            "degradation_recovery_rate": 0.0,
            "degradation_sources": {"provider_performance": 1},
            "degradation_kinds": {"persistent_provider_degradation": 1},
            "topology_actions": {"team_plan": 1},
            "topology_execution_modes": {"team_execution": 1},
        },
        "tasks": [
            {
                "status": "passed",
                "confidence_assessment": {
                    "confidence_score": 0.95,
                    "evidence_score": 0.9,
                    "truth_aligned": True,
                    "bucket": "high",
                },
            },
            {
                "status": "failed",
                "confidence_assessment": {
                    "confidence_score": 0.7,
                    "evidence_score": 0.8,
                    "truth_aligned": False,
                    "bucket": "high",
                },
            },
        ],
    }

    feedback = derive_runtime_evaluation_feedback(payload)

    assert feedback.completion_threshold is not None
    assert feedback.completion_threshold > 0.8
    assert feedback.minimum_supported_evidence_score is not None
    assert feedback.minimum_supported_evidence_score > feedback.completion_threshold
    assert feedback.metadata["source"] == "benchmark_truth_feedback"
    assert feedback.metadata["benchmark"] == "dr3_eval"
    assert feedback.metadata["model"] == "test-model"
    assert feedback.metadata["prompt_candidate_hash"] == "cand-123"
    assert feedback.metadata["section_name"] == "GROUNDING_RULES"
    assert feedback.metadata["truth_alignment_rate"] == pytest.approx(0.8)
    assert feedback.metadata["overconfidence_rate"] == pytest.approx(0.4)
    assert feedback.metadata["underconfidence_rate"] == pytest.approx(0.0)
    assert feedback.metadata["task_count"] == 2
    assert feedback.metadata["scope"]["provider"] == "anthropic"
    assert feedback.metadata["topology_feedback_coverage"] == pytest.approx(0.5)
    assert feedback.metadata["avg_topology_reward"] == pytest.approx(0.64)
    assert feedback.metadata["avg_topology_confidence"] == pytest.approx(0.73)
    assert feedback.metadata["degradation_feedback_coverage"] == pytest.approx(0.5)
    assert feedback.metadata["degradation_event_count"] == 1
    assert feedback.metadata["degraded_task_count"] == 1
    assert feedback.metadata["degradation_sources"] == {"provider_performance": 1}
    assert feedback.metadata["topology_actions"] == {"team_plan": 1}
    assert feedback.metadata["topology_execution_modes"] == {"team_execution": 1}
    assert feedback.metadata["topology_selection_policies"] == {}
    assert feedback.metadata["avg_topology_reward_by_selection_policy"] == {}
    assert feedback.metadata["topology_learned_override_reward_delta"] is None


def test_derive_runtime_feedback_reads_agentic_harness_sections_and_optimization_metrics():
    payload = {
        "summary": {
            "total_tasks": 2,
            "optimization_feasible_tasks": 1,
            "optimization_infeasible_tasks": 1,
            "optimization_feasibility_rate": 0.5,
            "topology_feedback_coverage": 1.0,
            "degradation_feedback_coverage": 1.0,
            "degradation_event_count": 2,
            "degraded_task_count": 2,
            "recovered_task_count": 1,
            "degradation_recovery_rate": 0.5,
            "avg_degradation_adaptation_cost": 2.5,
            "avg_degradation_time_to_recover_seconds": 4.0,
            "avg_degradation_cost_variance": 0.75,
            "avg_degradation_recovery_time_variance": 1.25,
            "avg_degradation_intervention_count": 1.5,
            "avg_degradation_confidence": 0.42,
            "avg_degradation_drift_score": 0.67,
            "content_degradation_task_count": 1,
            "confidence_degradation_task_count": 1,
            "provider_degradation_task_count": 1,
            "persistent_degradation_task_count": 1,
            "drift_task_count": 2,
            "degradation_drift_rate": 1.0,
            "degradation_intervention_task_count": 2,
            "degradation_intervention_rate": 1.0,
            "high_adaptation_cost_task_count": 1,
            "degradation_high_cost_rate": 0.5,
            "degradation_confidence_rate": 0.5,
            "degradation_stability_score": 0.38,
            "tasks_with_team_feedback": 2,
            "team_feedback_coverage": 1.0,
            "team_formations": {"parallel": 2},
            "team_merge_risk_levels": {"low": 1, "high": 1},
            "team_worktree_plan_count": 2,
            "team_worktree_materialized_count": 1,
            "team_high_risk_task_count": 1,
            "team_merge_conflict_task_count": 1,
            "team_cleanup_error_task_count": 1,
            "avg_team_assignments": 2.5,
            "avg_team_scoped_members": 2.0,
        },
        "quality": {
            "avg_topology_reward": 0.73,
            "avg_topology_confidence": 0.81,
            "avg_optimization_reward": 0.64,
            "avg_feasible_optimization_reward": 0.82,
            "avg_infeasible_optimization_reward": 0.46,
        },
        "optimization": {
            "gate_failures": {"tests_pass": 1},
            "feasible_tasks": 1,
            "infeasible_tasks": 1,
            "avg_reward": 0.64,
            "avg_feasible_reward": 0.82,
            "avg_infeasible_reward": 0.46,
        },
        "topology": {
            "topology_actions": {"single_agent": 1, "team_plan": 1},
            "topology_execution_modes": {"single_agent": 1, "team_execution": 1},
            "topology_selection_policies": {"heuristic": 1, "learned_close_override": 1},
            "topology_selection_policy_reward_totals": {
                "heuristic": 0.61,
                "learned_close_override": 0.85,
            },
            "topology_selection_policy_optimization_counts": {
                "heuristic": 1,
                "learned_close_override": 1,
            },
            "topology_selection_policy_optimization_reward_totals": {
                "heuristic": 0.46,
                "learned_close_override": 0.82,
            },
            "topology_selection_policy_feasible_counts": {
                "heuristic": 0,
                "learned_close_override": 1,
            },
            "topology_selection_policy_feasibility_rates": {
                "heuristic": 0.0,
                "learned_close_override": 1.0,
            },
            "topology_learned_override_reward_delta": 0.24,
            "topology_learned_override_optimization_reward_delta": 0.36,
            "topology_learned_override_feasibility_delta": 1.0,
        },
        "degradation": {
            "degradation_sources": {"provider_performance": 1, "agentic_loop": 1},
            "degradation_kinds": {
                "provider_recovered": 1,
                "content_repetition": 1,
            },
            "degradation_failure_types": {"PROVIDER_ERROR": 1, "STUCK_LOOP": 1},
            "degradation_providers": {"ollama": 1},
            "degradation_reasons": {"failure_streak": 1, "content_repetition": 1},
        },
        "tasks": [
            {"status": "passed"},
            {"status": "failed"},
        ],
    }

    feedback = derive_runtime_evaluation_feedback(payload)

    assert feedback.metadata["optimization_feasible_tasks"] == 1
    assert feedback.metadata["degradation_feedback_coverage"] == pytest.approx(1.0)
    assert feedback.metadata["degradation_event_count"] == 2
    assert feedback.metadata["recovered_task_count"] == 1
    assert feedback.metadata["avg_degradation_cost_variance"] == pytest.approx(0.75)
    assert feedback.metadata["avg_degradation_recovery_time_variance"] == pytest.approx(1.25)
    assert feedback.metadata["avg_degradation_intervention_count"] == pytest.approx(1.5)
    assert feedback.metadata["avg_degradation_confidence"] == pytest.approx(0.42)
    assert feedback.metadata["avg_degradation_drift_score"] == pytest.approx(0.67)
    assert feedback.metadata["confidence_degradation_task_count"] == 1
    assert feedback.metadata["persistent_degradation_task_count"] == 1
    assert feedback.metadata["drift_task_count"] == 2
    assert feedback.metadata["degradation_drift_rate"] == pytest.approx(1.0)
    assert feedback.metadata["degradation_intervention_task_count"] == 2
    assert feedback.metadata["degradation_intervention_rate"] == pytest.approx(1.0)
    assert feedback.metadata["high_adaptation_cost_task_count"] == 1
    assert feedback.metadata["degradation_high_cost_rate"] == pytest.approx(0.5)
    assert feedback.metadata["degradation_confidence_rate"] == pytest.approx(0.5)
    assert feedback.metadata["degradation_stability_score"] == pytest.approx(0.38)
    assert feedback.metadata["degradation_reasons"] == {
        "failure_streak": 1,
        "content_repetition": 1,
    }
    assert feedback.metadata["tasks_with_team_feedback"] == 2
    assert feedback.metadata["team_feedback_coverage"] == pytest.approx(1.0)
    assert feedback.metadata["team_formations"] == {"parallel": 2}
    assert feedback.metadata["team_merge_risk_levels"] == {"low": 1, "high": 1}
    assert feedback.metadata["team_worktree_plan_count"] == 2
    assert feedback.metadata["team_worktree_materialized_count"] == 1
    assert feedback.metadata["team_high_risk_task_count"] == 1
    assert feedback.metadata["team_merge_conflict_task_count"] == 1
    assert feedback.metadata["team_cleanup_error_task_count"] == 1
    assert feedback.metadata["avg_team_assignments"] == pytest.approx(2.5)
    assert feedback.metadata["avg_team_scoped_members"] == pytest.approx(2.0)
    assert feedback.metadata["optimization_infeasible_tasks"] == 1
    assert feedback.metadata["optimization_feasibility_rate"] == pytest.approx(0.5)
    assert feedback.metadata["avg_optimization_reward"] == pytest.approx(0.64)
    assert feedback.metadata["avg_feasible_optimization_reward"] == pytest.approx(0.82)
    assert feedback.metadata["avg_infeasible_optimization_reward"] == pytest.approx(0.46)
    assert feedback.metadata["optimization_gate_failures"] == {"tests_pass": 1}
    assert feedback.metadata["topology_selection_policy_optimization_counts"] == {
        "heuristic": 1,
        "learned_close_override": 1,
    }
    assert feedback.metadata["avg_topology_optimization_reward_by_selection_policy"] == {
        "heuristic": 0.46,
        "learned_close_override": 0.82,
    }
    assert feedback.metadata[
        "topology_learned_override_optimization_reward_delta"
    ] == pytest.approx(0.36)
    assert feedback.metadata["topology_learned_override_feasibility_delta"] == pytest.approx(1.0)


def test_save_and_load_runtime_feedback_round_trip(tmp_path):
    feedback = RuntimeEvaluationFeedback(
        completion_threshold=0.77,
        enhanced_progress_threshold=0.62,
        minimum_supported_evidence_score=0.84,
        metadata={"benchmark": "guide"},
    )

    path = save_runtime_evaluation_feedback(feedback, path=tmp_path / "runtime_feedback.json")
    loaded = load_runtime_evaluation_feedback(path)

    assert path.exists()
    assert loaded == RuntimeEvaluationFeedback(
        completion_threshold=0.77,
        enhanced_progress_threshold=0.62,
        minimum_supported_evidence_score=0.84,
        metadata=loaded.metadata,
    )
    raw = json.loads(path.read_text())
    assert raw["metadata"]["benchmark"] == "guide"
    assert raw["metadata"]["saved_at"]


def test_save_runtime_feedback_records_source_result_path(tmp_path):
    feedback = RuntimeEvaluationFeedback(
        completion_threshold=0.8,
        metadata={"benchmark": "guide"},
    )
    source_result_path = tmp_path / "eval_guide_20260425.json"

    path = save_runtime_evaluation_feedback(
        feedback,
        path=tmp_path / "runtime_feedback.json",
        source_result_path=source_result_path,
    )

    raw = json.loads(path.read_text())

    assert raw["metadata"]["source_result_path"] == str(source_result_path)


def test_build_validated_session_feedback_payload_uses_explicit_scope_schema():
    feedback = RuntimeEvaluationFeedback(
        completion_threshold=0.76,
        enhanced_progress_threshold=0.61,
        minimum_supported_evidence_score=0.83,
    )

    payload = build_validated_session_feedback_payload(
        feedback,
        scope=RuntimeEvaluationFeedbackScope(
            project="codingagent",
            provider="openai",
            model="gpt-5",
            task_type="edit",
            vertical="coding",
            workflow="agentic_loop",
            tags=("repair", "session"),
        ),
        validation_label="human_verified_session",
        metadata={"truth_alignment_rate": 0.91, "task_count": 7},
    )

    assert payload["metadata"]["source"] == "validated_session_truth_feedback"
    assert payload["metadata"]["validated_evaluation_truth"] is True
    assert payload["metadata"]["truth_validation_mode"] == "human_verified_session"
    assert payload["metadata"]["scope"] == {
        "project": "codingagent",
        "provider": "openai",
        "model": "gpt-5",
        "task_type": "edit",
        "benchmark": None,
        "vertical": "coding",
        "workflow": "agentic_loop",
        "tags": ["repair", "session"],
    }


def test_load_runtime_feedback_aggregates_recent_validated_truth_artifacts(tmp_path):
    old_result_path = tmp_path / "eval_guide_20260401_010101.json"
    fresh_result_path = tmp_path / "eval_guide_20260425_010101.json"
    ignored_result_path = tmp_path / "eval_runtime_20260425_020202.json"

    old_result_path.write_text(
        json.dumps(
            {
                "runtime_evaluation_feedback": {
                    "completion_threshold": 0.9,
                    "enhanced_progress_threshold": 0.7,
                    "minimum_supported_evidence_score": 0.92,
                    "metadata": {
                        "source": "benchmark_truth_feedback",
                        "validated_evaluation_truth": True,
                        "truth_alignment_rate": 0.95,
                        "task_count": 24,
                        "topology_feedback_coverage": 0.25,
                        "avg_topology_reward": 0.52,
                        "topology_actions": {"single_agent": 2},
                        "tasks_with_team_feedback": 2,
                        "team_feedback_coverage": 0.2,
                        "team_formations": {"parallel": 1},
                        "team_merge_risk_levels": {"low": 1},
                        "saved_at": "2026-04-01T00:00:00+00:00",
                    },
                }
            }
        )
    )
    fresh_result_path.write_text(
        json.dumps(
            {
                "runtime_evaluation_feedback": {
                    "completion_threshold": 0.7,
                    "enhanced_progress_threshold": 0.52,
                    "minimum_supported_evidence_score": 0.8,
                    "metadata": {
                        "source": "validated_session_truth_feedback",
                        "validated_evaluation_truth": True,
                        "truth_alignment_rate": 0.88,
                        "task_count": 12,
                        "topology_feedback_coverage": 0.9,
                        "avg_topology_reward": 0.79,
                        "topology_actions": {"team_plan": 3},
                        "tasks_with_team_feedback": 3,
                        "team_feedback_coverage": 1.0,
                        "team_formations": {"hierarchical": 2, "parallel": 1},
                        "team_merge_risk_levels": {"high": 1, "low": 2},
                        "team_worktree_materialized_count": 2,
                        "saved_at": "2026-04-25T00:00:00+00:00",
                    },
                }
            }
        )
    )
    ignored_result_path.write_text(
        json.dumps(
            {
                "runtime_evaluation_feedback": {
                    "completion_threshold": 0.55,
                    "enhanced_progress_threshold": 0.4,
                    "minimum_supported_evidence_score": 0.6,
                    "metadata": {
                        "source": "heuristic_runtime_feedback",
                        "truth_alignment_rate": 0.99,
                        "task_count": 100,
                        "saved_at": "2026-04-25T00:00:00+00:00",
                    },
                }
            }
        )
    )

    loaded = load_runtime_evaluation_feedback(tmp_path / "runtime_evaluation_feedback.json")

    assert loaded is not None
    assert loaded.completion_threshold is not None
    assert 0.7 < loaded.completion_threshold < 0.85
    assert loaded.enhanced_progress_threshold is not None
    assert 0.52 <= loaded.enhanced_progress_threshold < 0.65
    assert loaded.metadata["source"] == "validated_evaluation_truth_aggregate"
    assert loaded.metadata["aggregated_artifact_count"] == 2
    assert loaded.metadata["excluded_artifact_count"] == 1
    assert loaded.metadata["freshest_saved_at"] == "2026-04-25T00:00:00+00:00"
    assert loaded.metadata["topology_feedback_coverage"] is not None
    assert loaded.metadata["avg_topology_reward"] is not None
    assert loaded.metadata["topology_actions"]
    assert loaded.metadata["team_feedback_coverage"] is not None
    assert loaded.metadata["team_formations"]
    assert loaded.metadata["team_merge_risk_levels"]


def test_load_runtime_feedback_builds_scoped_team_worktree_metrics(tmp_path):
    save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "source": "validated_session_truth_feedback",
                "validated_evaluation_truth": True,
                "saved_at": "2026-04-25T00:00:00+00:00",
                "scope": RuntimeEvaluationFeedbackScope(
                    provider="openai",
                    model="gpt-5",
                    task_type="edit",
                ).to_dict(),
                "tasks_with_team_feedback": 2,
                "team_feedback_coverage": 1.0,
                "team_formations": {"parallel": 2},
                "team_merge_risk_levels": {"low": 2},
                "team_worktree_plan_count": 2,
                "team_worktree_materialized_count": 2,
                "team_cleanup_task_count": 2,
                "team_cleanup_error_task_count": 0,
                "avg_team_assignments": 2.0,
                "avg_team_scoped_members": 2.0,
            }
        ),
        path=tmp_path / "eval_edit_feedback.json",
    )
    save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "source": "validated_session_truth_feedback",
                "validated_evaluation_truth": True,
                "saved_at": "2026-04-26T00:00:00+00:00",
                "scope": RuntimeEvaluationFeedbackScope(
                    provider="anthropic",
                    model="claude-4",
                    task_type="analysis",
                ).to_dict(),
                "tasks_with_team_feedback": 3,
                "team_feedback_coverage": 1.0,
                "team_formations": {"parallel": 3},
                "team_merge_risk_levels": {"high": 2, "medium": 1},
                "team_worktree_plan_count": 3,
                "team_worktree_materialized_count": 1,
                "team_worktree_dry_run_count": 2,
                "team_cleanup_task_count": 2,
                "team_cleanup_error_task_count": 1,
                "team_merge_conflict_task_count": 1,
                "avg_team_assignments": 3.0,
                "avg_team_scoped_members": 3.0,
                "avg_team_changed_file_count": 4.0,
            }
        ),
        path=tmp_path / "eval_analysis_feedback.json",
    )

    loaded = load_runtime_evaluation_feedback(tmp_path / "runtime_evaluation_feedback.json")

    assert loaded is not None
    scope_metrics = loaded.metadata["team_worktree_scope_metrics"]
    assert scope_metrics["task_type"]["edit"]["team_worktree_materialized_count"] == pytest.approx(
        2.0
    )
    assert scope_metrics["task_type"]["analysis"]["team_worktree_dry_run_count"] == pytest.approx(
        2.0
    )
    assert scope_metrics["provider"]["anthropic"]["team_cleanup_error_task_count"] == pytest.approx(
        1.0
    )
    assert scope_metrics["model_family"]["gpt"]["team_formations"]["parallel"] == pytest.approx(2.0)


def test_load_runtime_feedback_aggregates_long_horizon_degradation_metrics(tmp_path):
    save_runtime_evaluation_feedback(
        RuntimeEvaluationFeedback(
            completion_threshold=0.78,
            enhanced_progress_threshold=0.61,
            minimum_supported_evidence_score=0.85,
            metadata={
                "source": "validated_session_truth_feedback",
                "validated_evaluation_truth": True,
                "saved_at": "2026-04-26T00:00:00+00:00",
                "scope": RuntimeEvaluationFeedbackScope(
                    provider="openai",
                    model="gpt-5",
                    task_type="analysis",
                ).to_dict(),
                "degradation_feedback_coverage": 1.0,
                "degradation_event_count": 3,
                "degraded_task_count": 2,
                "recovered_task_count": 1,
                "degradation_recovery_rate": 0.5,
                "avg_degradation_adaptation_cost": 2.25,
                "avg_degradation_time_to_recover_seconds": 4.5,
                "avg_degradation_cost_variance": 1.1,
                "avg_degradation_recovery_time_variance": 0.9,
                "avg_degradation_intervention_count": 1.5,
                "avg_degradation_confidence": 0.41,
                "avg_degradation_drift_score": 0.72,
                "content_degradation_task_count": 1,
                "confidence_degradation_task_count": 1,
                "provider_degradation_task_count": 1,
                "persistent_degradation_task_count": 1,
                "drift_task_count": 2,
                "degradation_drift_rate": 1.0,
                "degradation_intervention_task_count": 2,
                "degradation_intervention_rate": 1.0,
                "high_adaptation_cost_task_count": 1,
                "degradation_high_cost_rate": 0.5,
                "degradation_confidence_rate": 0.5,
                "degradation_stability_score": 0.33,
                "degradation_sources": {"provider_performance": 2, "streaming_confidence": 1},
                "degradation_kinds": {
                    "persistent_provider_degradation": 1,
                    "confidence_early_stop": 1,
                    "recovery_action": 1,
                },
            },
        ),
        path=tmp_path / "eval_degradation_feedback.json",
    )

    loaded = load_runtime_evaluation_feedback(tmp_path / "runtime_evaluation_feedback.json")

    assert loaded is not None
    assert loaded.metadata["avg_degradation_cost_variance"] == pytest.approx(1.1)
    assert loaded.metadata["avg_degradation_recovery_time_variance"] == pytest.approx(0.9)
    assert loaded.metadata["avg_degradation_intervention_count"] == pytest.approx(1.5)
    assert loaded.metadata["avg_degradation_confidence"] == pytest.approx(0.41)
    assert loaded.metadata["avg_degradation_drift_score"] == pytest.approx(0.72)
    assert loaded.metadata["confidence_degradation_task_count"] == pytest.approx(1.0)
    assert loaded.metadata["persistent_degradation_task_count"] == pytest.approx(1.0)
    assert loaded.metadata["drift_task_count"] == pytest.approx(2.0)
    assert loaded.metadata["degradation_drift_rate"] == pytest.approx(1.0)
    assert loaded.metadata["degradation_intervention_task_count"] == pytest.approx(2.0)
    assert loaded.metadata["degradation_intervention_rate"] == pytest.approx(1.0)
    assert loaded.metadata["high_adaptation_cost_task_count"] == pytest.approx(1.0)
    assert loaded.metadata["degradation_high_cost_rate"] == pytest.approx(0.5)
    assert loaded.metadata["degradation_confidence_rate"] == pytest.approx(0.5)
    assert loaded.metadata["degradation_stability_score"] == pytest.approx(0.33)
    assert loaded.metadata["degradation_sources"]["provider_performance"] > 0.0


def test_load_runtime_feedback_prefers_directory_aggregate_over_stale_canonical_file(tmp_path):
    canonical_path = tmp_path / "runtime_evaluation_feedback.json"
    canonical_path.write_text(
        json.dumps(
            {
                "completion_threshold": 0.91,
                "enhanced_progress_threshold": 0.76,
                "minimum_supported_evidence_score": 0.93,
                "metadata": {
                    "source": "benchmark_truth_feedback",
                    "validated_evaluation_truth": True,
                    "saved_at": "2026-03-01T00:00:00+00:00",
                    "task_count": 8,
                    "truth_alignment_rate": 0.8,
                },
            }
        )
    )
    (tmp_path / "eval_guide_20260425_030303.json").write_text(
        json.dumps(
            {
                "runtime_evaluation_feedback": {
                    "completion_threshold": 0.73,
                    "enhanced_progress_threshold": 0.57,
                    "minimum_supported_evidence_score": 0.82,
                    "metadata": {
                        "source": "benchmark_truth_feedback",
                        "validated_evaluation_truth": True,
                        "saved_at": "2026-04-25T03:03:03+00:00",
                        "task_count": 16,
                        "truth_alignment_rate": 0.9,
                    },
                }
            }
        )
    )

    loaded = load_runtime_evaluation_feedback(canonical_path)

    assert loaded is not None
    assert loaded.completion_threshold == pytest.approx(0.73)
    assert loaded.metadata["aggregated_artifact_count"] == 1


def test_load_runtime_feedback_prefers_project_model_task_adjacent_artifacts(tmp_path):
    unrelated_path = tmp_path / "eval_runtime_20260425_010101.json"
    adjacent_path = tmp_path / "eval_runtime_20260420_010101.json"

    unrelated_path.write_text(
        json.dumps(
            build_validated_session_feedback_payload(
                RuntimeEvaluationFeedback(
                    completion_threshold=0.62,
                    enhanced_progress_threshold=0.5,
                    minimum_supported_evidence_score=0.72,
                ),
                scope=RuntimeEvaluationFeedbackScope(
                    project="other-project",
                    provider="anthropic",
                    model="claude-sonnet",
                    task_type="analyze",
                ),
                metadata={
                    "truth_alignment_rate": 0.95,
                    "task_count": 20,
                    "saved_at": "2026-04-25T00:00:00+00:00",
                },
            )
        )
    )
    adjacent_path.write_text(
        json.dumps(
            build_validated_session_feedback_payload(
                RuntimeEvaluationFeedback(
                    completion_threshold=0.84,
                    enhanced_progress_threshold=0.68,
                    minimum_supported_evidence_score=0.88,
                ),
                scope=RuntimeEvaluationFeedbackScope(
                    project="codingagent",
                    provider="openai",
                    model="gpt-5",
                    task_type="edit",
                ),
                metadata={
                    "truth_alignment_rate": 0.88,
                    "task_count": 10,
                    "saved_at": "2026-04-20T00:00:00+00:00",
                },
            )
        )
    )

    loaded = load_runtime_evaluation_feedback(
        tmp_path / "runtime_evaluation_feedback.json",
        scope=RuntimeEvaluationFeedbackScope(
            project="codingagent",
            provider="openai",
            model="gpt-5",
            task_type="edit",
        ),
    )

    assert loaded is not None
    assert loaded.completion_threshold is not None
    assert loaded.completion_threshold > 0.74
    assert (
        loaded.metadata["scope_selection_strategy"]
        == "scoped_relevance_recency_reliability_weighted"
    )
    assert loaded.metadata["scope_target"] == {
        "project": "codingagent",
        "provider": "openai",
        "model": "gpt-5",
        "task_type": "edit",
        "benchmark": None,
        "vertical": None,
        "workflow": None,
        "tags": [],
    }


def test_load_runtime_feedback_overlays_scoped_live_topology_feedback(tmp_path):
    scope = RuntimeEvaluationFeedbackScope(
        project="codingagent",
        provider="openai",
        model="gpt-5",
        task_type="edit",
    )
    canonical_path = tmp_path / "runtime_evaluation_feedback.json"
    canonical_path.write_text(
        json.dumps(
            build_validated_session_feedback_payload(
                RuntimeEvaluationFeedback(
                    completion_threshold=0.82,
                    enhanced_progress_threshold=0.67,
                    minimum_supported_evidence_score=0.88,
                ),
                scope=scope,
                metadata={
                    "truth_alignment_rate": 0.9,
                    "task_count": 8,
                    "topology_feedback_coverage": 0.4,
                    "avg_topology_reward": 0.56,
                    "avg_topology_confidence": 0.61,
                    "topology_final_actions": {"single_agent": 2},
                    "topology_final_kinds": {"single_agent": 2},
                    "topology_execution_modes": {"single_agent": 2},
                    "saved_at": "2026-04-20T00:00:00+00:00",
                },
            )
        )
    )
    save_session_topology_runtime_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "topology_feedback_coverage": 1.0,
                "avg_topology_reward": 0.88,
                "avg_topology_confidence": 0.81,
                "topology_actions": {"team_plan": 4},
                "topology_final_actions": {"team_plan": 4},
                "topology_kinds": {"team": 4},
                "topology_final_kinds": {"team": 4},
                "topology_execution_modes": {"team_execution": 4},
                "topology_providers": {"anthropic": 4},
                "topology_formations": {"parallel": 4},
                "task_count": 4,
            }
        ),
        base_path=canonical_path,
        scope=scope,
    )

    loaded = load_runtime_evaluation_feedback(canonical_path, scope=scope)

    assert loaded is not None
    assert loaded.completion_threshold == pytest.approx(0.82)
    assert loaded.metadata["topology_final_actions"]["team_plan"] > 0.0
    assert loaded.metadata["topology_providers"]["anthropic"] > 0.0
    assert SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE in loaded.metadata["topology_feedback_sources"]
    assert loaded.metadata["topology_feedback_live_artifact_count"] == 1


def test_load_runtime_feedback_scopes_live_topology_overlay_to_matching_scope(tmp_path):
    target_scope = RuntimeEvaluationFeedbackScope(
        project="codingagent",
        provider="openai",
        model="gpt-5",
        task_type="edit",
    )
    other_scope = RuntimeEvaluationFeedbackScope(
        project="other-project",
        provider="anthropic",
        model="claude-sonnet",
        task_type="analysis",
    )
    feedback_path = tmp_path / "runtime_evaluation_feedback.json"

    save_session_topology_runtime_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "topology_feedback_coverage": 1.0,
                "avg_topology_reward": 0.86,
                "avg_topology_confidence": 0.79,
                "topology_actions": {"team_plan": 3},
                "topology_final_actions": {"team_plan": 3},
                "topology_kinds": {"team": 3},
                "topology_final_kinds": {"team": 3},
                "topology_execution_modes": {"team_execution": 3},
                "task_count": 3,
            }
        ),
        base_path=feedback_path,
        scope=target_scope,
    )
    save_session_topology_runtime_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "topology_feedback_coverage": 1.0,
                "avg_topology_reward": 0.92,
                "avg_topology_confidence": 0.84,
                "topology_actions": {"single_agent": 5},
                "topology_final_actions": {"single_agent": 5},
                "topology_kinds": {"single_agent": 5},
                "topology_final_kinds": {"single_agent": 5},
                "topology_execution_modes": {"single_agent": 5},
                "task_count": 5,
            }
        ),
        base_path=feedback_path,
        scope=other_scope,
    )

    loaded = load_runtime_evaluation_feedback(feedback_path, scope=target_scope)

    assert loaded is not None
    assert loaded.completion_threshold is None
    assert loaded.metadata["source"] == AGGREGATED_SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE
    assert loaded.metadata["topology_final_actions"]["team_plan"] > 0.0
    assert loaded.metadata["topology_final_actions"].get("single_agent", 0.0) < (
        loaded.metadata["topology_final_actions"]["team_plan"]
    )
    assert loaded.metadata["topology_scope_target"]["project"] == "codingagent"


def test_load_runtime_feedback_decays_stale_live_topology_feedback(tmp_path):
    scope = RuntimeEvaluationFeedbackScope(
        project="codingagent",
        provider="openai",
        model="gpt-5",
        task_type="edit",
    )
    canonical_path = tmp_path / "runtime_evaluation_feedback.json"
    canonical_path.write_text(
        json.dumps(
            build_validated_session_feedback_payload(
                RuntimeEvaluationFeedback(
                    completion_threshold=0.8,
                    enhanced_progress_threshold=0.64,
                    minimum_supported_evidence_score=0.87,
                ),
                scope=scope,
                metadata={
                    "truth_alignment_rate": 0.91,
                    "task_count": 3,
                    "topology_feedback_coverage": 0.72,
                    "avg_topology_reward": 0.77,
                    "avg_topology_confidence": 0.74,
                    "topology_actions": {"single_agent": 3},
                    "topology_final_actions": {"single_agent": 3},
                    "topology_kinds": {"single_agent": 3},
                    "topology_final_kinds": {"single_agent": 3},
                    "saved_at": "2026-04-26T00:00:00+00:00",
                },
            )
        )
    )
    live_path = save_session_topology_runtime_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "topology_feedback_coverage": 1.0,
                "avg_topology_reward": 0.92,
                "avg_topology_confidence": 0.86,
                "topology_actions": {"team_plan": 100},
                "topology_final_actions": {"team_plan": 100},
                "topology_kinds": {"team": 100},
                "topology_final_kinds": {"team": 100},
                "topology_execution_modes": {"team_execution": 100},
                "task_count": 100,
            }
        ),
        base_path=canonical_path,
        scope=scope,
    )
    live_payload = json.loads(live_path.read_text())
    live_payload["metadata"]["saved_at"] = "2026-03-01T00:00:00+00:00"
    live_path.write_text(json.dumps(live_payload))

    loaded = load_runtime_evaluation_feedback(canonical_path, scope=scope)

    assert loaded is not None
    assert loaded.metadata["topology_final_actions"]["single_agent"] > (
        loaded.metadata["topology_final_actions"].get("team_plan", 0.0)
    )


def test_load_runtime_feedback_reports_topology_conflict_metrics_for_split_feedback(tmp_path):
    scope = RuntimeEvaluationFeedbackScope(
        project="codingagent",
        provider="openai",
        model="gpt-5",
        task_type="edit",
    )
    feedback_path = tmp_path / "runtime_evaluation_feedback.json"

    save_session_topology_runtime_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "topology_feedback_coverage": 1.0,
                "avg_topology_reward": 0.84,
                "avg_topology_confidence": 0.81,
                "topology_actions": {"team_plan": 5, "single_agent": 4},
                "topology_final_actions": {"team_plan": 5, "single_agent": 4},
                "topology_kinds": {"team": 5, "single_agent": 4},
                "topology_final_kinds": {"team": 5, "single_agent": 4},
                "topology_execution_modes": {"team_execution": 5, "single_agent": 4},
                "topology_providers": {"anthropic": 5, "openai": 4},
                "topology_formations": {"parallel": 5, "hierarchical": 4},
                "task_count": 9,
            }
        ),
        base_path=feedback_path,
        scope=scope,
    )

    loaded = load_runtime_evaluation_feedback(feedback_path, scope=scope)

    assert loaded is not None
    assert loaded.metadata["source"] == AGGREGATED_SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE
    assert loaded.metadata["topology_action_agreement"] == pytest.approx(5 / 9, rel=1e-3)
    assert loaded.metadata["topology_kind_agreement"] == pytest.approx(5 / 9, rel=1e-3)
    assert loaded.metadata["topology_provider_agreement"] == pytest.approx(5 / 9, rel=1e-3)
    assert loaded.metadata["topology_formation_agreement"] == pytest.approx(5 / 9, rel=1e-3)
    assert loaded.metadata["topology_conflict_score"] > 0.4


def test_load_runtime_feedback_aggregates_selection_policy_reward_metrics(tmp_path):
    scope = RuntimeEvaluationFeedbackScope(
        project="codingagent",
        provider="openai",
        model="gpt-5",
        task_type="edit",
    )
    feedback_path = tmp_path / "runtime_evaluation_feedback.json"

    save_session_topology_runtime_feedback(
        RuntimeEvaluationFeedback(
            metadata={
                "topology_feedback_coverage": 1.0,
                "avg_topology_reward": 0.82,
                "avg_topology_confidence": 0.79,
                "topology_actions": {"team_plan": 3},
                "topology_final_actions": {"team_plan": 3},
                "topology_kinds": {"team": 3},
                "topology_final_kinds": {"team": 3},
                "topology_execution_modes": {"team_execution": 3},
                "topology_selection_policies": {
                    "heuristic": 2,
                    "learned_close_override": 3,
                },
                "topology_selection_policy_reward_totals": {
                    "heuristic": 1.1,
                    "learned_close_override": 2.4,
                },
                "topology_selection_policy_optimization_counts": {
                    "heuristic": 2,
                    "learned_close_override": 3,
                },
                "topology_selection_policy_optimization_reward_totals": {
                    "heuristic": 1.04,
                    "learned_close_override": 2.43,
                },
                "topology_selection_policy_feasible_counts": {
                    "heuristic": 1,
                    "learned_close_override": 3,
                },
                "task_count": 5,
            }
        ),
        base_path=feedback_path,
        scope=scope,
    )

    loaded = load_runtime_evaluation_feedback(feedback_path, scope=scope)

    assert loaded is not None
    assert loaded.metadata["avg_topology_reward_by_selection_policy"] == {
        "heuristic": 0.55,
        "learned_close_override": 0.8,
    }
    assert loaded.metadata["topology_learned_override_reward_delta"] == pytest.approx(0.25)
    assert loaded.metadata["topology_selection_policy_scope_metrics"]["task_type"]["edit"][
        "learned_override_reward_delta"
    ] == pytest.approx(0.25)
    assert loaded.metadata["topology_selection_policy_scope_metrics"]["provider"]["openai"][
        "avg_reward_by_policy"
    ] == {
        "heuristic": 0.55,
        "learned_close_override": 0.8,
    }
    assert loaded.metadata["topology_selection_policy_scope_metrics"]["provider"]["openai"][
        "avg_optimization_reward_by_policy"
    ] == {
        "heuristic": 0.52,
        "learned_close_override": 0.81,
    }
    assert loaded.metadata["topology_selection_policy_scope_metrics"]["provider"]["openai"][
        "feasibility_rate_by_policy"
    ] == {
        "heuristic": 0.5,
        "learned_close_override": 1.0,
    }
    assert loaded.metadata["topology_selection_policy_scope_metrics"]["model_family"]["gpt"][
        "policy_counts"
    ] == {"heuristic": 2.0, "learned_close_override": 3.0}
    assert loaded.metadata["topology_selection_policy_scope_metrics"]["model_family"]["gpt"][
        "learned_override_optimization_reward_delta"
    ] == pytest.approx(0.29)
    assert loaded.metadata["topology_selection_policy_scope_metrics"]["model_family"]["gpt"][
        "learned_override_feasibility_delta"
    ] == pytest.approx(0.5)


def test_build_swe_bench_validated_session_feedback_payload_uses_real_validator_outputs():
    baseline = TestBaseline(
        instance_id="django__123",
        repo="django/django",
        base_commit="abc123",
        fail_to_pass=["test_fix_a", "test_fix_b"],
        pass_to_pass=["test_keep_green"],
        status=BaselineStatus.VALID,
    )
    validation_result = BaselineValidationResult(
        instance_id="django__123",
        baseline=baseline,
        post_change_results=TestRunResults(total=3, passed=2, failed=1, duration_seconds=12.0),
        fail_to_pass_fixed=["test_fix_a"],
        pass_to_pass_broken=[],
        success=False,
        partial_success=True,
        score=0.5,
    )
    score = SWEBenchScore(
        instance_id="django__123",
        resolved=False,
        partial=True,
        fail_to_pass_score=0.5,
        pass_to_pass_score=1.0,
        overall_score=0.7,
        tests_fixed=1,
        tests_broken=0,
        total_fail_to_pass=2,
        total_pass_to_pass=1,
    )

    payload = build_swe_bench_validated_session_feedback_payload(
        validation_result,
        score=score,
        source_result_path=Path("/tmp/swe_bench_results/django__123.json"),
    )

    assert payload["metadata"]["source"] == "validated_session_truth_feedback"
    assert payload["metadata"]["truth_validation_mode"] == "swe_bench_posthoc_validation"
    assert payload["metadata"]["scope"] == {
        "project": "django/django",
        "provider": None,
        "model": None,
        "task_type": "edit",
        "benchmark": "swe_bench",
        "vertical": "coding",
        "workflow": "evaluation_orchestrator",
        "tags": ["agentic", "coding", "validated-session"],
    }
    assert payload["metadata"]["validation_summary"] == {
        "success": False,
        "partial_success": True,
        "fail_to_pass_total": 2,
        "fail_to_pass_fixed": 1,
        "pass_to_pass_total": 1,
        "pass_to_pass_broken": 0,
        "post_change_total": 3,
        "post_change_passed": 2,
    }
    assert payload["completion_threshold"] is not None
    assert payload["minimum_supported_evidence_score"] is not None


# ---------------------------------------------------------------------------
# Phase 5.9: browser and deep-research post-hoc validator session-truth tests
# ---------------------------------------------------------------------------


def test_build_browser_validated_session_feedback_payload_full_pass():
    """Browser task with full action and answer coverage emits strong session-truth."""
    evaluation_result = {
        "task_id": "clawbench_001",
        "status": "PASSED",
        "completion_score": 1.0,
        "failure_details": {
            "action_coverage": 1.0,
            "answer_coverage": 1.0,
            "matched_actions": ["click", "fill"],
            "missing_actions": [],
            "forbidden_action_hits": [],
            "matched_answer_phrases": ["success"],
            "missing_answer_phrases": [],
        },
        "benchmark": "clawbench",
        "total_tasks": 10,
        "passed_tasks": 9,
        "failed_tasks": 1,
    }
    payload = build_browser_validated_session_feedback_payload(evaluation_result)

    assert payload is not None
    assert payload["metadata"]["source"] == "validated_session_truth_feedback"
    assert payload["metadata"]["truth_validation_mode"] == "browser_posthoc_validation"
    assert payload["metadata"]["scope"]["vertical"] == "browser"
    assert payload["metadata"]["scope"]["benchmark"] == "clawbench"
    assert payload["metadata"]["scope"]["task_type"] == "interaction"
    assert 0.5 <= payload["completion_threshold"] <= 1.0
    assert 0.5 <= payload["minimum_supported_evidence_score"] <= 1.0


def test_build_browser_validated_session_feedback_payload_partial():
    """Browser task with partial coverage produces calibrated thresholds."""
    evaluation_result = {
        "task_id": "vlaa_002",
        "status": "FAILED",
        "completion_score": 0.55,
        "failure_details": {
            "action_coverage": 0.5,
            "answer_coverage": 0.6,
            "matched_actions": ["click"],
            "missing_actions": ["fill", "submit"],
            "forbidden_action_hits": [],
            "matched_answer_phrases": [],
            "missing_answer_phrases": ["confirm"],
        },
        "benchmark": "vlaa",
        "total_tasks": 5,
        "passed_tasks": 2,
        "failed_tasks": 3,
    }
    payload = build_browser_validated_session_feedback_payload(evaluation_result)

    assert payload is not None
    assert payload["metadata"]["validation_summary"]["action_coverage"] == 0.5
    assert payload["metadata"]["validation_summary"]["answer_coverage"] == 0.6
    assert payload["completion_threshold"] > 0.6


def test_build_browser_validated_session_feedback_payload_returns_none_for_empty():
    """Returns None when evaluation_result has no usable coverage data."""
    payload = build_browser_validated_session_feedback_payload({})
    assert payload is None


def test_build_browser_validated_session_feedback_payload_forbidden_action():
    """Forbidden action hit raises the evidence threshold."""
    evaluation_result = {
        "completion_score": 0.3,
        "failure_details": {
            "action_coverage": 0.3,
            "answer_coverage": 0.0,
            "forbidden_action_hits": ["delete_all"],
        },
        "benchmark": "guide",
    }
    payload = build_browser_validated_session_feedback_payload(evaluation_result)
    assert payload is not None
    assert payload["minimum_supported_evidence_score"] >= 0.75


def test_build_deep_research_validated_session_feedback_payload_full_pass():
    """Deep-research with full claim and citation coverage emits strong session-truth."""
    evaluation_result = {
        "task_id": "dr3_001",
        "status": "PASSED",
        "completion_score": 1.0,
        "failure_details": {
            "claim_coverage": 1.0,
            "citation_coverage": 1.0,
            "matched_claims": ["market growing"],
            "missing_claims": [],
            "matched_citations": ["Smith 2024"],
            "missing_citations": [],
            "forbidden_claim_hits": [],
            "report_length_chars": 4500,
        },
        "benchmark": "dr3",
        "total_tasks": 8,
        "passed_tasks": 7,
        "failed_tasks": 1,
    }
    payload = build_deep_research_validated_session_feedback_payload(evaluation_result)

    assert payload is not None
    assert payload["metadata"]["source"] == "validated_session_truth_feedback"
    assert payload["metadata"]["truth_validation_mode"] == "deep_research_posthoc_validation"
    assert payload["metadata"]["scope"]["vertical"] == "research"
    assert payload["metadata"]["scope"]["benchmark"] == "dr3"
    assert payload["metadata"]["scope"]["task_type"] == "analysis"
    assert 0.5 <= payload["completion_threshold"] <= 1.0


def test_build_deep_research_validated_session_feedback_payload_partial():
    """Deep-research with partial coverage produces calibrated thresholds."""
    evaluation_result = {
        "status": "FAILED",
        "completion_score": 0.45,
        "failure_details": {
            "claim_coverage": 0.4,
            "citation_coverage": 0.5,
            "matched_claims": ["growing trend"],
            "missing_claims": ["market size", "competition"],
            "matched_citations": [],
            "missing_citations": ["Jones 2023"],
            "forbidden_claim_hits": [],
            "report_length_chars": 800,
        },
        "benchmark": "dr3",
    }
    payload = build_deep_research_validated_session_feedback_payload(evaluation_result)

    assert payload is not None
    assert payload["metadata"]["validation_summary"]["claim_coverage"] == 0.4
    assert payload["metadata"]["validation_summary"]["citation_coverage"] == 0.5
    assert payload["completion_threshold"] > 0.6


def test_build_deep_research_validated_session_feedback_payload_returns_none_for_empty():
    """Returns None when evaluation_result has no usable coverage data."""
    payload = build_deep_research_validated_session_feedback_payload({})
    assert payload is None


def test_build_deep_research_validated_session_feedback_payload_forbidden_claim():
    """Forbidden claim hit raises the evidence threshold."""
    evaluation_result = {
        "completion_score": 0.2,
        "failure_details": {
            "claim_coverage": 0.2,
            "citation_coverage": 0.0,
            "forbidden_claim_hits": ["unverified stat"],
        },
        "benchmark": "dr3",
    }
    payload = build_deep_research_validated_session_feedback_payload(evaluation_result)
    assert payload is not None
    assert payload["minimum_supported_evidence_score"] >= 0.75
