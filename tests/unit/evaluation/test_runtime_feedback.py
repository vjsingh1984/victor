from __future__ import annotations

import json

import pytest

from victor.evaluation.runtime_feedback import (
    derive_runtime_evaluation_feedback,
    load_runtime_evaluation_feedback,
    save_runtime_evaluation_feedback,
)
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationFeedback


def test_derive_runtime_feedback_raises_threshold_for_overconfident_failures():
    payload = {
        "config": {"benchmark": "dr3_eval", "model": "test-model"},
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
    assert feedback.metadata["truth_alignment_rate"] == pytest.approx(0.8)
    assert feedback.metadata["overconfidence_rate"] == pytest.approx(0.4)
    assert feedback.metadata["underconfidence_rate"] == pytest.approx(0.0)
    assert feedback.metadata["task_count"] == 2


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
