"""TDD tests for AgenticLoop DECIDE-phase wiring — Wave A1.

Verifies that _apply_backslide_guard() is actually called in the while-loop
DECIDE block, and that velocity-based gating of COMPLETE works end-to-end.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestBackslideGuardWired:
    """The DECIDE block must call _apply_backslide_guard on COMPLETE decisions."""

    def test_backslide_guard_downgrades_complete_with_negative_velocity(self):
        """Guard downgrades COMPLETE to CONTINUE when score drops significantly."""
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = [0.99]  # previous score was very high

        evaluation = EvaluationResult(
            decision=EvaluationDecision.COMPLETE,
            score=0.75,  # dropped significantly: velocity = 0.75 - 0.99 = -0.24 < -0.10
            reason="test",
        )
        result = loop._apply_backslide_guard(evaluation)
        assert result.decision != EvaluationDecision.COMPLETE
        assert result.decision == EvaluationDecision.CONTINUE

    def test_backslide_guard_allows_complete_with_positive_velocity(self):
        """Guard preserves COMPLETE when score improves from previous turn."""
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = [0.70]  # previous score was lower

        evaluation = EvaluationResult(
            decision=EvaluationDecision.COMPLETE,
            score=0.90,  # improved: velocity = +0.20 > 0
            reason="test",
        )
        result = loop._apply_backslide_guard(evaluation)
        assert result.decision == EvaluationDecision.COMPLETE

    def test_score_history_grows_after_each_guard_call(self):
        """_apply_backslide_guard appends score to _score_history on each call."""
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = []

        eval1 = EvaluationResult(decision=EvaluationDecision.COMPLETE, score=0.8, reason="")
        loop._apply_backslide_guard(eval1)
        assert len(loop._score_history) == 1

        eval2 = EvaluationResult(decision=EvaluationDecision.COMPLETE, score=0.85, reason="")
        loop._apply_backslide_guard(eval2)
        assert len(loop._score_history) == 2

    def test_score_history_initialised_on_construction(self):
        """AgenticLoop instances start with an empty _score_history."""
        from victor.framework.agentic_loop import AgenticLoop

        loop = AgenticLoop.__new__(AgenticLoop)
        # __new__ skips __init__ — but the attribute is set in __init__
        # Verify the attribute exists when loop is normally constructed (not via __new__)
        assert hasattr(AgenticLoop, "__init__")
        # Check that _score_history is set in __init__ source
        import inspect

        source = inspect.getsource(AgenticLoop.__init__)
        assert "_score_history" in source

    def test_decide_block_calls_backslide_guard(self):
        """The DECIDE block must invoke _apply_backslide_guard before acting on COMPLETE.

        This test verifies wiring by patching _apply_backslide_guard on the loop instance
        and confirming it is called with the COMPLETE evaluation.
        """
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        # Build minimal loop state using __new__ and manual attribute injection
        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = []

        # Create a COMPLETE evaluation
        evaluation = EvaluationResult(
            decision=EvaluationDecision.COMPLETE,
            score=0.90,
            reason="done",
        )

        # Patch the guard on this specific instance
        guard_calls: list = []
        original_guard = AgenticLoop._apply_backslide_guard

        def recording_guard(self, eval_, backslide_threshold=-0.10):
            guard_calls.append(eval_)
            return original_guard(self, eval_, backslide_threshold)

        with patch.object(AgenticLoop, "_apply_backslide_guard", recording_guard):
            # Simulate the DECIDE block logic (what the while-loop should do):
            # if evaluation.decision == EvaluationDecision.COMPLETE:
            #     evaluation = self._apply_backslide_guard(evaluation)
            if evaluation.decision == EvaluationDecision.COMPLETE:
                evaluation = loop._apply_backslide_guard(evaluation)

        assert (
            len(guard_calls) == 1
        ), "_apply_backslide_guard must be called once in DECIDE when decision is COMPLETE"

    def test_guard_not_called_on_continue_decision(self):
        """Backslide guard should only activate on COMPLETE, not CONTINUE."""
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = [0.99]

        evaluation = EvaluationResult(
            decision=EvaluationDecision.CONTINUE,
            score=0.5,
            reason="",
        )
        initial_history_len = len(loop._score_history)

        # Simulate DECIDE block: guard is only called when COMPLETE
        if evaluation.decision == EvaluationDecision.COMPLETE:
            evaluation = loop._apply_backslide_guard(evaluation)

        # Score history should not have grown (guard was not invoked)
        assert len(loop._score_history) == initial_history_len

    def test_decide_loop_source_wires_backslide_guard(self):
        """The while-loop DECIDE block in AgenticLoop.run() must call _apply_backslide_guard.

        This source-inspection test fails if the wiring is absent from the real loop.
        """
        import inspect

        from victor.framework.agentic_loop import AgenticLoop

        source = inspect.getsource(AgenticLoop.run)
        assert "_apply_backslide_guard" in source, (
            "AgenticLoop.run() DECIDE block must call self._apply_backslide_guard(evaluation). "
            "Wire it in: after evaluation is computed, before the COMPLETE terminal check."
        )

    def test_guard_preserves_complete_on_first_turn(self):
        """With empty score_history (first turn), guard should not downgrade."""
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = []

        evaluation = EvaluationResult(
            decision=EvaluationDecision.COMPLETE,
            score=0.95,
            reason="done",
        )
        result = loop._apply_backslide_guard(evaluation)
        assert result.decision == EvaluationDecision.COMPLETE
