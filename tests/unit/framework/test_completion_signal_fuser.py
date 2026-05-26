"""TDD tests for CompletionSignalFuser — Wave 3.

Verifies: 4-signal fusion, velocity computation, backslide prevention,
and agentic loop DECIDE integration.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest


class TestFuserResult:
    def test_fuser_result_importable(self):
        from victor.framework.completion_signal_fuser import FuserResult

        result = FuserResult(score=0.9, decision="complete", reason="done")
        assert result.score == 0.9

    def test_fuser_result_velocity_defaults_to_zero(self):
        from victor.framework.completion_signal_fuser import FuserResult

        result = FuserResult(score=0.5, decision="continue", reason="")
        assert result.velocity == 0.0

    def test_fuser_result_signals_used_defaults_empty(self):
        from victor.framework.completion_signal_fuser import FuserResult

        result = FuserResult(score=0.5, decision="continue", reason="")
        assert result.signals_used == {}


class TestCompletionSignalFuser:
    def test_fuser_importable(self):
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        fuser = CompletionSignalFuser()
        assert fuser is not None

    def test_fuser_aggregates_all_four_signals(self):
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        fuser = CompletionSignalFuser()
        result = fuser.fuse(
            fulfillment=0.9,
            requirement=0.8,
            keyword=0.7,
            confidence=0.85,
            score_history=[],
        )
        assert "fulfillment" in result.signals_used
        assert "requirement" in result.signals_used
        assert "keyword" in result.signals_used
        assert "confidence" in result.signals_used

    def test_fuser_score_is_weighted_average_of_signals(self):
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        fuser = CompletionSignalFuser()
        result = fuser.fuse(
            fulfillment=1.0,
            requirement=1.0,
            keyword=1.0,
            confidence=1.0,
            score_history=[],
        )
        assert result.score == pytest.approx(1.0, abs=0.01)

    def test_fuser_score_reflects_low_signals(self):
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        fuser = CompletionSignalFuser()
        result = fuser.fuse(
            fulfillment=0.0,
            requirement=0.0,
            keyword=0.0,
            confidence=0.0,
            score_history=[],
        )
        assert result.score < 0.3

    def test_fuser_velocity_zero_on_first_turn(self):
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        fuser = CompletionSignalFuser()
        result = fuser.fuse(
            fulfillment=0.8,
            requirement=0.7,
            keyword=0.8,
            confidence=0.9,
            score_history=[],  # no history → first turn
        )
        assert result.velocity == 0.0

    def test_fuser_velocity_computed_from_score_history(self):
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        fuser = CompletionSignalFuser()
        result = fuser.fuse(
            fulfillment=0.8,
            requirement=0.8,
            keyword=0.8,
            confidence=0.8,
            score_history=[0.5],  # previous score was 0.5; new score should be ~0.8
        )
        assert result.velocity > 0  # improved from 0.5

    def test_fuser_velocity_negative_when_score_drops(self):
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        fuser = CompletionSignalFuser()
        result = fuser.fuse(
            fulfillment=0.1,
            requirement=0.1,
            keyword=0.1,
            confidence=0.1,
            score_history=[0.9],  # was 0.9, now dropping
        )
        assert result.velocity < 0

    def test_fuser_decision_complete_when_score_above_threshold(self):
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        fuser = CompletionSignalFuser()
        result = fuser.fuse(
            fulfillment=0.95,
            requirement=0.95,
            keyword=0.95,
            confidence=0.95,
            score_history=[0.9],  # positive velocity
        )
        assert result.decision in ("complete", "COMPLETE")

    def test_fuser_decision_not_complete_when_negative_velocity_despite_high_score(self):
        """High score with negative velocity (backslide) should not yield COMPLETE."""
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        fuser = CompletionSignalFuser(backslide_threshold=-0.1)
        result = fuser.fuse(
            fulfillment=0.85,
            requirement=0.85,
            keyword=0.85,
            confidence=0.85,
            score_history=[0.99],  # was 0.99; now 0.85 → velocity < -0.1
        )
        # Despite 0.85 score (above default 0.8 threshold), backslide prevents COMPLETE
        assert result.decision not in ("complete", "COMPLETE")

    def test_fuser_decision_continue_when_score_below_threshold(self):
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        fuser = CompletionSignalFuser()
        result = fuser.fuse(
            fulfillment=0.2,
            requirement=0.2,
            keyword=0.2,
            confidence=0.2,
            score_history=[],
        )
        assert result.decision in ("continue", "CONTINUE")

    def test_fuser_configurable_completion_threshold(self):
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        fuser = CompletionSignalFuser(completion_threshold=0.5)
        result = fuser.fuse(
            fulfillment=0.6,
            requirement=0.6,
            keyword=0.6,
            confidence=0.6,
            score_history=[0.5],  # positive velocity
        )
        assert result.decision in ("complete", "COMPLETE")

    def test_fuser_configurable_backslide_threshold(self):
        from victor.framework.completion_signal_fuser import CompletionSignalFuser

        # With a very negative backslide threshold, even a small dip should pass
        fuser = CompletionSignalFuser(backslide_threshold=-0.5)
        result = fuser.fuse(
            fulfillment=0.9,
            requirement=0.9,
            keyword=0.9,
            confidence=0.9,
            score_history=[0.92],  # small dip: velocity ~ -0.02; above -0.5
        )
        assert result.decision in ("complete", "COMPLETE")


class TestAgenticLoopVelocityIntegration:
    """Verify agentic loop DECIDE tracks score_history and applies backslide guard."""

    def test_agentic_loop_exposes_score_history_tracking(self):
        """AgenticLoop should accept or track loop state with score_history key."""
        from victor.framework.agentic_loop import AgenticLoop

        # AgenticLoop should have a _score_history attribute or track it via loop state
        loop = AgenticLoop.__new__(AgenticLoop)
        # Verify the backslide guard method or config exists
        has_backslide = hasattr(AgenticLoop, "_apply_backslide_guard") or hasattr(
            AgenticLoop, "_check_backslide"
        )
        has_score_history = hasattr(AgenticLoop, "_score_history")
        assert (
            has_backslide or has_score_history
        ), "AgenticLoop should expose backslide guard or score_history tracking"

    def test_backslide_guard_prevents_premature_complete(self):
        """_apply_backslide_guard() should downgrade COMPLETE to CONTINUE on backslide."""
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        if not hasattr(AgenticLoop, "_apply_backslide_guard"):
            pytest.skip("_apply_backslide_guard not yet implemented")

        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = [0.99]

        evaluation = EvaluationResult(
            decision=EvaluationDecision.COMPLETE,
            score=0.85,  # dropped from 0.99
            reason="test",
        )
        result = loop._apply_backslide_guard(evaluation)
        assert result.decision != EvaluationDecision.COMPLETE

    def test_backslide_guard_allows_complete_on_positive_velocity(self):
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        if not hasattr(AgenticLoop, "_apply_backslide_guard"):
            pytest.skip("_apply_backslide_guard not yet implemented")

        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = [0.7]

        evaluation = EvaluationResult(
            decision=EvaluationDecision.COMPLETE,
            score=0.9,  # improved from 0.7
            reason="test",
        )
        result = loop._apply_backslide_guard(evaluation)
        assert result.decision == EvaluationDecision.COMPLETE
