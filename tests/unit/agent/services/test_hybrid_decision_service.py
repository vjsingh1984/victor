"""Focused runtime-feedback tests for HybridDecisionService."""

from __future__ import annotations

import pytest

from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.hybrid_decision_service import (
    HybridDecisionService,
    HybridDecisionServiceConfig,
)


def test_runtime_feedback_uses_calibrator_thresholds():
    service = HybridDecisionService(
        provider=None,
        model="test",
        config=HybridDecisionServiceConfig(enable_calibration=True, enable_llm_fallback=False),
    )
    assert service._calibrator is not None
    service._calibrator._thresholds[DecisionType.TASK_COMPLETION] = 0.81

    for i in range(12):
        service._calibrator.record_outcome(
            DecisionType.TASK_COMPLETION,
            heuristic_confidence=0.8,
            used_llm=False,
            was_correct=i < 10,
        )

    feedback = service.get_runtime_evaluation_feedback()

    assert feedback.completion_threshold == pytest.approx(0.81)
    assert feedback.enhanced_progress_threshold == pytest.approx(0.66)
    assert feedback.minimum_supported_evidence_score > feedback.completion_threshold


def test_runtime_feedback_falls_back_to_base_threshold_without_calibrator():
    service = HybridDecisionService(
        provider=None,
        model="test",
        config=HybridDecisionServiceConfig(
            enable_calibration=False,
            enable_llm_fallback=False,
            base_threshold=0.73,
        ),
    )

    feedback = service.get_runtime_evaluation_feedback()

    assert feedback.completion_threshold == pytest.approx(0.73)
    assert feedback.enhanced_progress_threshold == pytest.approx(0.58)
    assert feedback.minimum_supported_evidence_score == pytest.approx(0.78)
