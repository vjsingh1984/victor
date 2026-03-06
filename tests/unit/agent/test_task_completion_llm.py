"""Tests for TaskCompletionDetector with LLM decision service augmentation."""

from unittest.mock import MagicMock

import pytest

from victor.agent.decisions.schemas import DecisionType, TaskCompletionDecision
from victor.agent.services.protocols.decision_service import DecisionResult
from victor.agent.task_completion import (
    CompletionConfidence,
    TaskCompletionDetector,
)


def _make_decision_service(
    is_complete: bool = True,
    confidence: float = 0.9,
    phase: str = "done",
    source: str = "llm",
):
    """Create a mock decision service that returns a predetermined result."""
    service = MagicMock()
    result = DecisionResult(
        decision_type=DecisionType.TASK_COMPLETION,
        result=TaskCompletionDecision(
            is_complete=is_complete, confidence=confidence, phase=phase
        ),
        source=source,
        confidence=confidence,
    )
    service.decide_sync.return_value = result
    return service


class TestTaskCompletionWithoutService:
    """Verify existing behavior is unchanged when no service is provided."""

    def test_no_service_default(self):
        detector = TaskCompletionDetector()
        assert detector._decision_service is None

    def test_active_signal_still_high(self):
        detector = TaskCompletionDetector()
        detector.analyze_response("**DONE**: Task complete")
        assert detector.get_completion_confidence() == CompletionConfidence.HIGH

    def test_passive_signal_still_low(self):
        detector = TaskCompletionDetector()
        detector.analyze_response("The file has been created successfully")
        assert detector.get_completion_confidence() == CompletionConfidence.LOW

    def test_no_signal_still_none(self):
        detector = TaskCompletionDetector()
        detector.analyze_response("Let me think about this")
        assert detector.get_completion_confidence() == CompletionConfidence.NONE


class TestTaskCompletionWithLLMService:
    """Test LLM augmentation in task completion detection."""

    def test_llm_upgrades_none_to_medium(self):
        """When heuristic is NONE but LLM says complete, upgrades to MEDIUM."""
        service = _make_decision_service(is_complete=True, confidence=0.9, phase="done")
        detector = TaskCompletionDetector(decision_service=service)

        # No heuristic signals at all
        assert detector.get_completion_confidence() == CompletionConfidence.MEDIUM
        service.decide_sync.assert_called_once()

    def test_llm_low_confidence_no_upgrade(self):
        """When LLM confidence is below 0.7, no upgrade happens."""
        service = _make_decision_service(is_complete=True, confidence=0.5, phase="done")
        detector = TaskCompletionDetector(decision_service=service)

        assert detector.get_completion_confidence() == CompletionConfidence.NONE

    def test_llm_says_not_complete(self):
        """When LLM says not complete, stays NONE."""
        service = _make_decision_service(is_complete=False, confidence=0.9, phase="working")
        detector = TaskCompletionDetector(decision_service=service)

        assert detector.get_completion_confidence() == CompletionConfidence.NONE

    def test_llm_not_called_when_heuristic_high(self):
        """LLM is never consulted when heuristic confidence is HIGH."""
        service = _make_decision_service()
        detector = TaskCompletionDetector(decision_service=service)
        detector.analyze_response("**DONE**: Task is complete")

        result = detector.get_completion_confidence()
        assert result == CompletionConfidence.HIGH
        # decide_sync may have been called in analyze_response but should not
        # override the HIGH result from get_completion_confidence

    def test_llm_analyze_response_detects_completion(self):
        """LLM detects completion in analyze_response when heuristics miss it."""
        service = _make_decision_service(is_complete=True, confidence=0.9, phase="done")
        detector = TaskCompletionDetector(decision_service=service)

        # Ambiguous response with no clear signals
        detector.analyze_response("I have completed all the requested changes to the codebase.")

        # LLM should have added a completion signal
        assert "llm:task_complete" in detector._state.completion_signals

    def test_llm_analyze_response_detects_stuck(self):
        """LLM detects stuck phase and increments continuation requests."""
        service = _make_decision_service(is_complete=False, confidence=0.8, phase="stuck")
        detector = TaskCompletionDetector(decision_service=service)

        detector.analyze_response("I'm not sure what to do next. Let me think about this.")
        assert detector._state.continuation_requests >= 1

    def test_llm_failure_graceful(self):
        """Service failure doesn't break detection."""
        service = MagicMock()
        service.decide_sync.side_effect = RuntimeError("LLM unavailable")
        detector = TaskCompletionDetector(decision_service=service)

        # Should not raise
        detector.analyze_response("Some response text")
        assert detector.get_completion_confidence() == CompletionConfidence.NONE

    def test_llm_not_called_when_active_signal_exists(self):
        """LLM is not consulted when active signal already detected."""
        service = _make_decision_service()
        detector = TaskCompletionDetector(decision_service=service)

        detector.analyze_response("**DONE**: Implementation complete")

        # Active signal was detected, LLM should NOT be called
        # (the analyze_response returns early on active signal)
        service.decide_sync.assert_not_called()

    def test_budget_exhausted_source(self):
        """Budget exhausted returns heuristic fallback."""
        service = MagicMock()
        service.decide_sync.return_value = DecisionResult(
            decision_type=DecisionType.TASK_COMPLETION,
            result=None,
            source="budget_exhausted",
            confidence=0.0,
        )
        detector = TaskCompletionDetector(decision_service=service)
        detector.analyze_response("Some text without signals")

        # budget_exhausted means no LLM result, should be NONE
        assert detector.get_completion_confidence() == CompletionConfidence.NONE
