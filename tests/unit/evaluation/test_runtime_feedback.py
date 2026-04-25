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
