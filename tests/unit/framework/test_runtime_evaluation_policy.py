import pytest

from victor.framework.evaluation_nodes import EvaluationDecision
from victor.framework.runtime_evaluation_policy import (
    RuntimeEvaluationFeedback,
    RuntimeEvaluationPolicy,
)


def test_from_config_applies_shared_thresholds_and_messages():
    policy = RuntimeEvaluationPolicy.from_config(
        {
            "clarification_confidence_threshold": 0.6,
            "low_confidence_retry_limit": 4,
            "underspecified_target_prompt": "Name the exact file to change.",
            "high_confidence_reason": "Strong confidence",
        }
    )

    assert policy.clarification_confidence_threshold == 0.6
    assert policy.low_confidence_retry_limit == 4
    assert policy.underspecified_target_prompt == "Name the exact file to change."
    assert policy.high_confidence_reason == "Strong confidence"


def test_get_confidence_evaluation_uses_policy_reason():
    policy = RuntimeEvaluationPolicy(low_confidence_reason="Confidence too low - retry")

    result = policy.get_confidence_evaluation(0.3)

    assert result.decision == EvaluationDecision.RETRY
    assert result.reason == "Confidence too low - retry"


def test_calibrate_completion_uses_policy_weights_and_penalties():
    policy = RuntimeEvaluationPolicy(
        calibrated_completion_raw_weight=0.6,
        calibrated_completion_evidence_weight=0.4,
        continuation_request_penalty=0.15,
        unsupported_requirement_penalty=0.2,
        minimum_supported_evidence_score=0.8,
    )

    calibration = policy.calibrate_completion(
        raw_score=0.95,
        evidence_score=0.4,
        threshold=0.9,
        continuation_requested=True,
        requirements_satisfied=False,
    )

    assert calibration.support_penalty == pytest.approx(0.35)
    assert calibration.calibrated_score == pytest.approx(0.38)
    assert calibration.requires_additional_support is True
    assert "continuation_requested" in calibration.reasons
    assert "requirements_not_fully_satisfied" in calibration.reasons


def test_build_completion_evaluation_uses_policy_templates_and_thresholds():
    policy = RuntimeEvaluationPolicy(
        enhanced_progress_threshold=0.65,
        completion_requires_support_reason="Need stronger support",
        completion_success_reason_template="Done: {score:.2f}/{threshold:.2f}",
        completion_progress_reason_template="Keep going: {score:.2f}/{threshold:.2f}",
        completion_retry_reason_template="Retry: {score:.2f}",
    )

    support_result = policy.build_completion_evaluation(
        score=0.7,
        threshold=0.85,
        requires_additional_support=True,
    )
    progress_result = policy.build_completion_evaluation(score=0.7, threshold=0.85)
    retry_result = policy.build_completion_evaluation(score=0.6, threshold=0.85)

    assert support_result.decision == EvaluationDecision.CONTINUE
    assert support_result.reason == "Need stronger support"
    assert progress_result.decision == EvaluationDecision.CONTINUE
    assert progress_result.reason == "Keep going: 0.70/0.85"
    assert retry_result.decision == EvaluationDecision.RETRY
    assert retry_result.reason == "Retry: 0.60"


def test_with_feedback_applies_calibrated_runtime_thresholds():
    policy = RuntimeEvaluationPolicy()
    feedback = RuntimeEvaluationFeedback(
        completion_threshold=0.74,
        enhanced_progress_threshold=0.59,
        minimum_supported_evidence_score=0.83,
        metadata={"source": "decision_service"},
    )

    calibrated = policy.with_feedback(feedback)

    assert calibrated.completion_threshold == 0.74
    assert calibrated.enhanced_progress_threshold == 0.59
    assert calibrated.minimum_supported_evidence_score == 0.83
