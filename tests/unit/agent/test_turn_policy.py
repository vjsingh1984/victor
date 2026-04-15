"""Tests for victor.agent.turn_policy module.

Shared turn policy used by both batch (AgenticLoop) and streaming
(StreamingChatPipeline) execution paths.
"""

from __future__ import annotations

import pytest

from victor.agent.turn_policy import (
    FulfillmentCriteria,
    FulfillmentCriteriaBuilder,
    NudgeDecision,
    NudgePolicy,
    NudgeType,
    SpinDetector,
    SpinState,
    MAX_ALL_BLOCKED,
    MAX_NO_TOOL_TURNS,
    NUDGE_THRESHOLD,
    READ_ONLY_ESCALATION_THRESHOLD,
)

# ============================================================================
# SpinDetector tests
# ============================================================================


class TestSpinDetector:
    """Tests for SpinDetector."""

    def test_initial_state_is_normal(self):
        detector = SpinDetector()
        assert detector.state == SpinState.NORMAL

    def test_tool_calls_keep_normal(self):
        detector = SpinDetector()
        state = detector.record_turn(has_tool_calls=True, tool_count=2)
        assert state == SpinState.NORMAL
        assert detector.total_tool_calls == 2

    def test_no_tools_reaches_warning(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=False)
        state = detector.record_turn(has_tool_calls=False)
        assert state == SpinState.WARNING
        assert detector.consecutive_no_tool_turns == NUDGE_THRESHOLD

    def test_no_tools_reaches_terminated(self):
        detector = SpinDetector()
        for _ in range(MAX_NO_TOOL_TURNS):
            detector.record_turn(has_tool_calls=False)
        assert detector.state == SpinState.TERMINATED

    def test_all_blocked_reaches_blocked(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, all_blocked=True)
        state = detector.record_turn(has_tool_calls=True, all_blocked=True)
        assert state == SpinState.BLOCKED

    def test_all_blocked_reaches_terminated(self):
        detector = SpinDetector()
        for _ in range(MAX_ALL_BLOCKED):
            detector.record_turn(has_tool_calls=True, all_blocked=True)
        assert detector.state == SpinState.TERMINATED

    def test_successful_tool_resets_no_tool_counter(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=False)
        detector.record_turn(has_tool_calls=False)
        detector.record_turn(has_tool_calls=True, tool_count=1)
        assert detector.consecutive_no_tool_turns == 0
        assert detector.state == SpinState.NORMAL

    def test_successful_tool_resets_blocked_counter(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, all_blocked=True)
        detector.record_turn(has_tool_calls=True, all_blocked=False, tool_count=1)
        assert detector.consecutive_all_blocked == 0

    def test_read_only_tracking(self):
        detector = SpinDetector()
        for _ in range(READ_ONLY_ESCALATION_THRESHOLD):
            detector.record_turn(
                has_tool_calls=True,
                tool_names={"read", "grep"},
                tool_count=2,
            )
        assert detector.consecutive_read_only_turns == READ_ONLY_ESCALATION_THRESHOLD
        assert detector.has_used_code_search is False

    def test_code_search_breaks_read_only(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, tool_names={"read"}, tool_count=1)
        detector.record_turn(has_tool_calls=True, tool_names={"code_search"}, tool_count=1)
        assert detector.has_used_code_search is True

    def test_non_read_only_resets_counter(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, tool_names={"read"}, tool_count=1)
        detector.record_turn(has_tool_calls=True, tool_names={"edit"}, tool_count=1)
        assert detector.consecutive_read_only_turns == 0

    def test_reset(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=False)
        detector.record_turn(has_tool_calls=True, all_blocked=True, tool_count=3)
        detector.reset()
        assert detector.state == SpinState.NORMAL
        assert detector.total_tool_calls == 0
        assert detector.consecutive_no_tool_turns == 0
        assert detector.consecutive_all_blocked == 0


# ============================================================================
# NudgePolicy tests
# ============================================================================


class TestNudgePolicy:
    """Tests for NudgePolicy."""

    def test_no_nudge_on_normal(self):
        policy = NudgePolicy()
        detector = SpinDetector()
        decision = policy.evaluate(detector)
        assert decision.nudge_type == NudgeType.NONE
        assert decision.should_inject is False

    def test_use_tools_nudge_on_warning(self):
        policy = NudgePolicy()
        detector = SpinDetector()
        for _ in range(NUDGE_THRESHOLD):
            detector.record_turn(has_tool_calls=False)
        decision = policy.evaluate(detector)
        assert decision.nudge_type == NudgeType.USE_TOOLS
        assert decision.should_inject is True
        assert "MUST use a tool" in decision.message
        assert decision.role == "user"

    def test_different_tools_nudge_on_blocked(self):
        policy = NudgePolicy()
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, all_blocked=True)
        detector.record_turn(has_tool_calls=True, all_blocked=True)
        decision = policy.evaluate(detector)
        assert decision.nudge_type == NudgeType.DIFFERENT_TOOLS
        assert decision.should_inject is True
        assert "blocked" in decision.message.lower()
        assert decision.role == "system"

    def test_code_search_nudge_on_read_only(self):
        policy = NudgePolicy()
        detector = SpinDetector()
        for _ in range(READ_ONLY_ESCALATION_THRESHOLD):
            detector.record_turn(has_tool_calls=True, tool_names={"read"}, tool_count=1)
        decision = policy.evaluate(detector)
        assert decision.nudge_type == NudgeType.CODE_SEARCH
        assert "code_search" in decision.message

    def test_no_code_search_nudge_after_use(self):
        policy = NudgePolicy()
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, tool_names={"code_search"}, tool_count=1)
        for _ in range(READ_ONLY_ESCALATION_THRESHOLD):
            detector.record_turn(has_tool_calls=True, tool_names={"read"}, tool_count=1)
        decision = policy.evaluate(detector)
        # Should NOT nudge for code_search since it was already used
        assert decision.nudge_type != NudgeType.CODE_SEARCH

    def test_budget_warning_past_halfway(self):
        policy = NudgePolicy()
        decision = policy.budget_warning(iteration=7, max_iterations=10)
        assert decision.nudge_type == NudgeType.BUDGET_WARNING
        assert decision.should_inject is True
        assert "remaining" in decision.message
        assert decision.role == "system"

    def test_no_budget_warning_before_halfway(self):
        policy = NudgePolicy()
        decision = policy.budget_warning(iteration=3, max_iterations=10)
        assert decision.should_inject is False


# ============================================================================
# FulfillmentCriteriaBuilder tests
# ============================================================================


class TestFulfillmentCriteriaBuilder:
    """Tests for FulfillmentCriteriaBuilder."""

    def test_records_written_files(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "write",
                "args": {"file_path": "/app/main.py"},
                "success": True,
            }
        )
        criteria = builder.build()
        assert "/app/main.py" in criteria.file_paths

    def test_records_edited_files(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "edit",
                "args": {"file_path": "/app/utils.py"},
                "success": True,
            }
        )
        criteria = builder.build()
        assert "/app/utils.py" in criteria.file_paths

    def test_records_test_files_from_shell(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "shell",
                "args": {"command": "pytest tests/test_main.py -v"},
                "success": True,
            }
        )
        criteria = builder.build()
        assert "tests/test_main.py" in criteria.test_files

    def test_records_doc_files(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "write",
                "args": {"file_path": "/app/README.md"},
                "success": True,
            }
        )
        criteria = builder.build()
        assert "/app/README.md" in criteria.doc_files

    def test_records_errors(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "edit",
                "args": {"file_path": "/app/main.py"},
                "success": False,
                "error": "File not found",
            }
        )
        criteria = builder.build()
        assert criteria.original_error == "File not found"
        assert len(criteria.file_paths) == 0  # Failed edits not tracked

    def test_to_dict(self):
        criteria = FulfillmentCriteria(
            file_paths=["/app/main.py"],
            test_files=["tests/test_main.py"],
            original_error="KeyError",
        )
        d = criteria.to_dict()
        assert d["file_path"] == "/app/main.py"
        assert d["test_files"] == ["tests/test_main.py"]
        assert d["original_error"] == "KeyError"

    def test_reset(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "write",
                "args": {"file_path": "/x.py"},
                "success": True,
            }
        )
        builder.reset()
        criteria = builder.build()
        assert len(criteria.file_paths) == 0

    def test_ignores_failed_results(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "write",
                "args": {"file_path": "/fail.py"},
                "success": False,
                "error": "Permission denied",
            }
        )
        criteria = builder.build()
        assert len(criteria.file_paths) == 0


# ============================================================================
# Consistency tests (batch vs streaming should use same logic)
# ============================================================================


class TestBatchStreamingConsistency:
    """Verify batch and streaming paths produce consistent decisions."""

    def test_same_spin_detection_constants(self):
        """Both paths use same thresholds from turn_policy."""
        assert MAX_NO_TOOL_TURNS == 3
        assert MAX_ALL_BLOCKED == 3
        assert NUDGE_THRESHOLD == 2

    def test_spin_detector_deterministic(self):
        """Same input sequence produces same state."""
        d1 = SpinDetector()
        d2 = SpinDetector()

        # Simulate identical turn sequence
        sequence = [
            {"has_tool_calls": True, "all_blocked": False, "tool_count": 2},
            {"has_tool_calls": True, "all_blocked": True, "tool_count": 1},
            {"has_tool_calls": False},
        ]

        for turn in sequence:
            d1.record_turn(**turn)
            d2.record_turn(**turn)

        assert d1.state == d2.state
        assert d1.consecutive_no_tool_turns == d2.consecutive_no_tool_turns
        assert d1.consecutive_all_blocked == d2.consecutive_all_blocked
        assert d1.total_tool_calls == d2.total_tool_calls

    def test_nudge_policy_deterministic(self):
        """Same detector state produces same nudge decision."""
        policy = NudgePolicy()

        d1 = SpinDetector()
        d2 = SpinDetector()

        for _ in range(NUDGE_THRESHOLD):
            d1.record_turn(has_tool_calls=False)
            d2.record_turn(has_tool_calls=False)

        n1 = policy.evaluate(d1)
        n2 = policy.evaluate(d2)

        assert n1.nudge_type == n2.nudge_type
        assert n1.message == n2.message
        assert n1.role == n2.role
