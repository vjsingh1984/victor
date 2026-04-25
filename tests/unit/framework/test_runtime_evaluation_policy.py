from victor.framework.evaluation_nodes import EvaluationDecision
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy


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
