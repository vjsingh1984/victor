"""Tests for SafetyStatePassedCoordinator (SPA-4)."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from victor.agent.coordinators.safety_coordinator import (
    SafetyAction,
    SafetyCategory,
    SafetyRule,
)
from victor.agent.coordinators.safety_state_passed import SafetyStatePassedCoordinator
from victor.agent.coordinators.state_context import (
    CoordinatorResult,
    ContextSnapshot,
    TransitionType,
)


def _make_snapshot(**overrides: Any) -> ContextSnapshot:
    defaults = {
        "messages": (),
        "session_id": "test",
        "conversation_stage": "initial",
        "settings": MagicMock(),
        "model": "test",
        "provider": "test",
        "max_tokens": 4096,
        "temperature": 0.7,
        "conversation_state": {},
        "session_state": {},
        "observed_files": (),
        "capabilities": {},
    }
    defaults.update(overrides)
    return ContextSnapshot(**defaults)


FORCE_PUSH_RULE = SafetyRule(
    rule_id="no_force_push",
    category=SafetyCategory.GIT,
    pattern=r"push.*--force",
    description="Force push blocked",
    action=SafetyAction.BLOCK,
    severity=8,
    tool_names=["git"],
)

CURL_WARN_RULE = SafetyRule(
    rule_id="curl_warn",
    category=SafetyCategory.NETWORK,
    pattern=r"curl.*--insecure",
    description="Insecure curl warning",
    action=SafetyAction.WARN,
    severity=4,
    tool_names=["shell"],
)


class TestSafetyStatePassedInit:
    def test_creates_with_rules(self):
        coord = SafetyStatePassedCoordinator(rules=[FORCE_PUSH_RULE])
        assert coord._inner is not None

    def test_no_orchestrator_reference(self):
        coord = SafetyStatePassedCoordinator()
        assert not hasattr(coord, "_orchestrator")


class TestSafetyCheck:
    @pytest.mark.asyncio
    async def test_safe_operation_continues(self):
        coord = SafetyStatePassedCoordinator(rules=[FORCE_PUSH_RULE])
        snapshot = _make_snapshot()
        result = await coord.check(snapshot, "git", ["status"])
        assert result.should_continue is True
        assert isinstance(result, CoordinatorResult)

    @pytest.mark.asyncio
    async def test_blocked_operation_stops(self):
        coord = SafetyStatePassedCoordinator(rules=[FORCE_PUSH_RULE])
        snapshot = _make_snapshot()
        result = await coord.check(snapshot, "git", ["push", "--force", "origin", "main"])
        assert result.should_continue is False
        assert "BLOCKED" in result.reasoning

    @pytest.mark.asyncio
    async def test_warning_allows_continuation(self):
        coord = SafetyStatePassedCoordinator(rules=[CURL_WARN_RULE])
        snapshot = _make_snapshot()
        result = await coord.check(snapshot, "shell", ["curl", "--insecure", "https://example.com"])
        assert result.should_continue is True
        assert "WARNING" in result.reasoning

    @pytest.mark.asyncio
    async def test_records_safety_check_in_transitions(self):
        coord = SafetyStatePassedCoordinator(rules=[FORCE_PUSH_RULE])
        snapshot = _make_snapshot()
        result = await coord.check(snapshot, "git", ["status"])

        state_transitions = [
            t
            for t in result.transitions.transitions
            if t.transition_type == TransitionType.UPDATE_STATE
            and t.data.get("key") == "last_safety_check"
        ]
        assert len(state_transitions) == 1
        check_data = state_transitions[0].data["value"]
        assert check_data["tool"] == "git"
        assert check_data["is_safe"] is True

    @pytest.mark.asyncio
    async def test_blocked_has_metadata(self):
        coord = SafetyStatePassedCoordinator(rules=[FORCE_PUSH_RULE])
        snapshot = _make_snapshot()
        result = await coord.check(snapshot, "git", ["push", "--force"])
        assert result.metadata["action"] == "block"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_warning_stores_warnings_in_transitions(self):
        coord = SafetyStatePassedCoordinator(rules=[CURL_WARN_RULE])
        snapshot = _make_snapshot()
        result = await coord.check(snapshot, "shell", ["curl", "--insecure", "https://example.com"])
        warning_transitions = [
            t
            for t in result.transitions.transitions
            if t.transition_type == TransitionType.UPDATE_STATE
            and t.data.get("key") == "safety_warnings"
        ]
        assert len(warning_transitions) == 1
