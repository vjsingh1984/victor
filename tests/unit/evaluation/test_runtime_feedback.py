from __future__ import annotations

import json
from pathlib import Path

import pytest

from victor.evaluation.runtime_feedback import (
    RuntimeEvaluationFeedbackScope,
    build_browser_validated_session_feedback_payload,
    build_deep_research_validated_session_feedback_payload,
    build_swe_bench_validated_session_feedback_payload,
    build_validated_session_feedback_payload,
    derive_runtime_evaluation_feedback,
    load_runtime_evaluation_feedback,
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
