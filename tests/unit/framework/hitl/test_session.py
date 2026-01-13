# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for HITL session management."""

import asyncio
import pytest
import time

from victor.framework.hitl.session import (
    GateExecutionResult,
    HITLSession,
    HITLSessionConfig,
    HITLSessionManager,
    SessionEvent,
    SessionState,
    get_global_session_manager,
)
from victor.framework.hitl.gates import ApprovalGate, TextInputGate


# =============================================================================
# SessionState Tests
# =============================================================================


class TestSessionState:
    """Tests for SessionState enum."""

    def test_state_values(self):
        """SessionState should have correct string values."""
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.PAUSED.value == "paused"
        assert SessionState.COMPLETED.value == "completed"
        assert SessionState.FAILED.value == "failed"
        assert SessionState.TIMEOUT.value == "timeout"


# =============================================================================
# SessionEvent Tests
# =============================================================================


class TestSessionEvent:
    """Tests for SessionEvent."""

    def test_event_initialization(self):
        """SessionEvent should initialize correctly."""
        event = SessionEvent(
            event_type="gate_start",
            gate_id="test_gate",
            data={"key": "value"},
        )

        assert event.event_type == "gate_start"
        assert event.gate_id == "test_gate"
        assert event.data == {"key": "value"}
        assert event.timestamp > 0


# =============================================================================
# HITLSessionConfig Tests
# =============================================================================


class TestHITLSessionConfig:
    """Tests for HITLSessionConfig."""

    def test_default_config(self):
        """HITLSessionConfig should have sensible defaults."""
        config = HITLSessionConfig()

        assert config.default_timeout == 300.0
        assert config.default_fallback_behavior == "abort"
        assert config.auto_resume is True
        assert config.max_concurrent_gates == 10
        assert config.persist_history is True

    def test_custom_config(self):
        """HITLSessionConfig should accept custom values."""
        config = HITLSessionConfig(
            default_timeout=60.0,
            auto_resume=False,
        )

        assert config.default_timeout == 60.0
        assert config.auto_resume is False


# =============================================================================
# HITLSession Tests
# =============================================================================


class TestHITLSession:
    """Tests for HITLSession."""

    def test_session_initialization(self):
        """HITLSession should initialize correctly."""
        session = HITLSession(workflow_id="test_workflow")

        assert session.workflow_id == "test_workflow"
        assert session.state == SessionState.ACTIVE
        assert session.context == {}
        assert session.created_at > 0

    def test_session_with_custom_id(self):
        """Should accept custom session ID."""
        session = HITLSession(
            workflow_id="test",
            session_id="custom_id",
        )

        assert session.session_id == "custom_id"

    def test_session_with_initial_context(self):
        """Should accept initial context."""
        session = HITLSession(
            workflow_id="test",
            initial_context={"key": "value"},
        )

        assert session.context == {"key": "value"}

    def test_session_generates_unique_ids(self):
        """Session IDs should be unique."""
        session1 = HITLSession(workflow_id="test")
        session2 = HITLSession(workflow_id="test")

        assert session1.session_id != session2.session_id

    def test_get_context(self):
        """get_context should retrieve values."""
        session = HITLSession(
            workflow_id="test",
            initial_context={"a": 1, "b": 2},
        )

        assert session.get_context("a") == 1
        assert session.get_context("c") is None
        assert session.get_context("c", "default") == "default"

    def test_set_context(self):
        """set_context should set values."""
        session = HITLSession(workflow_id="test")

        session.set_context("key", "value")

        assert session.get_context("key") == "value"

    def test_update_context(self):
        """update_context should merge values."""
        session = HITLSession(
            workflow_id="test",
            initial_context={"a": 1},
        )

        session.update_context(b=2, c=3)

        assert session.get_context("a") == 1
        assert session.get_context("b") == 2
        assert session.get_context("c") == 3

    def test_pause(self):
        """pause should change state to PAUSED."""
        session = HITLSession(workflow_id="test")

        session.pause()

        assert session.state == SessionState.PAUSED

    def test_pause_when_paused_no_change(self):
        """pause when paused should keep state."""
        session = HITLSession(workflow_id="test")
        session.pause()

        session.pause()  # Already paused

        assert session.state == SessionState.PAUSED

    def test_resume(self):
        """resume should change state to ACTIVE."""
        session = HITLSession(workflow_id="test")
        session.pause()

        session.resume()

        assert session.state == SessionState.ACTIVE

    def test_complete(self):
        """complete should change state to COMPLETED."""
        session = HITLSession(workflow_id="test")

        session.complete()

        assert session.state == SessionState.COMPLETED

    def test_fail(self):
        """fail should change state to FAILED."""
        session = HITLSession(workflow_id="test")

        session.fail()

        assert session.state == SessionState.FAILED

    def test_fail_with_reason(self):
        """fail should record reason in history."""
        session = HITLSession(workflow_id="test")

        session.fail(reason="Test failure")

        assert session.state == SessionState.FAILED

        history = session.get_history()
        failure_events = [e for e in history if e.event_type == "session_failed"]
        assert len(failure_events) == 1
        assert failure_events[0].data["reason"] == "Test failure"

    def test_get_history_empty(self):
        """get_history should return empty list for new session."""
        session = HITLSession(workflow_id="test")

        history = session.get_history()

        assert history == []

    def test_get_summary(self):
        """get_summary should return session summary."""
        session = HITLSession(
            workflow_id="test_workflow",
            initial_context={"key": "value"},
        )

        summary = session.get_summary()

        assert summary["session_id"] == session.session_id
        assert summary["workflow_id"] == "test_workflow"
        assert summary["state"] == "active"
        assert summary["gates_executed"] == 0
        assert "key" in summary["context_keys"]

    @pytest.mark.asyncio
    async def test_execute_gate_success(self):
        """execute_gate should execute gate and record result."""
        session = HITLSession(workflow_id="test")

        gate = ApprovalGate(
            title="Test",
            description="Test approval",
        )

        result = await session.execute_gate(gate)

        assert result.approved is True
        assert session.state == SessionState.ACTIVE

        results = session.get_results()
        assert len(results) == 1
        assert results[0].gate_id == gate.gate_id

    @pytest.mark.asyncio
    async def test_execute_gate_with_context(self):
        """execute_gate should merge context."""
        session = HITLSession(
            workflow_id="test",
            initial_context={"a": 1},
        )

        gate = ApprovalGate(
            title="Test",
            description="Test $b",
        )

        # Gate should have access to session context
        result = await session.execute_gate(gate, context={"b": 2})

        assert result.approved is True

    @pytest.mark.asyncio
    async def test_execute_gate_with_rejection(self):
        """execute_gate should remain PAUSED when rejected."""
        session = HITLSession(workflow_id="test")

        async def rejecting_handler(**kwargs):
            return type("Response", (), {"approved": False, "reason": "No"})()

        gate = ApprovalGate(title="Test", description="Test")

        result = await session.execute_gate(gate, handler=rejecting_handler)

        assert result.approved is False
        assert session.state == SessionState.PAUSED

    @pytest.mark.asyncio
    async def test_execute_gate_in_failed_state_raises(self):
        """execute_gate in FAILED state should raise error."""
        session = HITLSession(workflow_id="test")
        session.fail()

        gate = ApprovalGate(title="Test", description="Test")

        with pytest.raises(RuntimeError, match="Cannot execute gate"):
            await session.execute_gate(gate)

    @pytest.mark.asyncio
    async def test_on_state_change_callback(self):
        """on_state_change should trigger on state change."""
        session = HITLSession(workflow_id="test")
        changes = []

        session.on_state_change(lambda old, new: changes.append((old, new)))

        session.pause()
        session.resume()

        assert len(changes) == 2
        assert changes[0] == (SessionState.ACTIVE, SessionState.PAUSED)
        assert changes[1] == (SessionState.PAUSED, SessionState.ACTIVE)

    @pytest.mark.asyncio
    async def test_on_gate_complete_callback(self):
        """on_gate_complete should trigger on gate completion."""
        session = HITLSession(workflow_id="test")
        results = []

        session.on_gate_complete(lambda r: results.append(r))

        gate = ApprovalGate(title="Test", description="Test")
        await session.execute_gate(gate)

        assert len(results) == 1
        assert results[0].approved is True

    def test_persist_history_false(self):
        """Session should not persist history when disabled."""
        config = HITLSessionConfig(persist_history=False)
        session = HITLSession(workflow_id="test", config=config)

        session.pause()

        # No history events should be recorded
        history = session.get_history()
        assert len(history) == 0


# =============================================================================
# HITLSessionManager Tests
# =============================================================================


class TestHITLSessionManager:
    """Tests for HITLSessionManager."""

    def test_manager_initialization(self):
        """HITLSessionManager should initialize correctly."""
        manager = HITLSessionManager()

        assert manager.list_sessions() == []

    def test_create_session(self):
        """create_session should create and store session."""
        manager = HITLSessionManager()

        session = manager.create_session(workflow_id="test_workflow")

        assert session is not None
        assert session.workflow_id == "test_workflow"
        assert session.session_id in [s.session_id for s in manager.list_sessions()]

    def test_create_session_with_config(self):
        """create_session should apply config."""
        config = HITLSessionConfig(default_timeout=60.0)
        manager = HITLSessionManager(default_config=config)

        session = manager.create_session(workflow_id="test")

        assert session.config.default_timeout == 60.0

    def test_get_session(self):
        """get_session should retrieve session by ID."""
        manager = HITLSessionManager()
        session = manager.create_session(workflow_id="test")

        retrieved = manager.get_session(session.session_id)

        assert retrieved is session

    def test_get_unknown_session_returns_none(self):
        """get_session should return None for unknown ID."""
        manager = HITLSessionManager()

        assert manager.get_session("unknown") is None

    def test_list_sessions_all(self):
        """list_sessions should return all sessions."""
        manager = HITLSessionManager()

        manager.create_session(workflow_id="workflow1")
        manager.create_session(workflow_id="workflow2")

        sessions = manager.list_sessions()

        assert len(sessions) == 2

    def test_list_sessions_by_workflow(self):
        """list_sessions should filter by workflow ID."""
        manager = HITLSessionManager()

        manager.create_session(workflow_id="workflow_a")
        manager.create_session(workflow_id="workflow_a")
        manager.create_session(workflow_id="workflow_b")

        sessions = manager.list_sessions(workflow_id="workflow_a")

        assert len(sessions) == 2
        assert all(s.workflow_id == "workflow_a" for s in sessions)

    def test_list_sessions_by_state(self):
        """list_sessions should filter by state."""
        manager = HITLSessionManager()

        session1 = manager.create_session(workflow_id="test")
        session2 = manager.create_session(workflow_id="test")
        session2.complete()

        active_sessions = manager.list_sessions(state=SessionState.ACTIVE)

        assert len(active_sessions) == 1
        assert active_sessions[0].session_id == session1.session_id

    def test_cleanup_old_sessions(self):
        """cleanup_old_sessions should remove old sessions."""
        manager = HITLSessionManager()

        session1 = manager.create_session(workflow_id="test")
        session1.complete()
        # Manually set old timestamp
        session1._updated_at = time.time() - 4000  # Over an hour ago

        session2 = manager.create_session(workflow_id="test")
        session2.complete()

        removed = manager.cleanup_old_sessions(max_age_seconds=3600)

        assert removed == 1
        assert manager.get_session(session1.session_id) is None
        assert manager.get_session(session2.session_id) is not None

    def test_get_all_summaries(self):
        """get_all_summaries should return all session summaries."""
        manager = HITLSessionManager()

        session = manager.create_session(
            workflow_id="test",
            initial_context={"key": "value"},
        )

        summaries = manager.get_all_summaries()

        assert len(summaries) == 1
        assert summaries[0]["workflow_id"] == "test"


# =============================================================================
# Global Session Manager Tests
# =============================================================================


class TestGlobalSessionManager:
    """Tests for global session manager."""

    def test_get_global_session_manager(self):
        """get_global_session_manager should return singleton."""
        manager1 = get_global_session_manager()
        manager2 = get_global_session_manager()

        assert manager1 is manager2
        assert isinstance(manager1, HITLSessionManager)


# =============================================================================
# GateExecutionResult Tests
# =============================================================================


class TestGateExecutionResult:
    """Tests for GateExecutionResult."""

    def test_result_initialization(self):
        """GateExecutionResult should initialize correctly."""
        result = GateExecutionResult(
            gate_id="test_gate",
            gate_type="approval",
            approved=True,
            value="test_value",
            reason="Test reason",
        )

        assert result.gate_id == "test_gate"
        assert result.gate_type == "approval"
        assert result.approved is True
        assert result.value == "test_value"
        assert result.reason == "Test reason"
        assert result.executed_at > 0

    def test_result_with_defaults(self):
        """GateExecutionResult should have sensible defaults."""
        result = GateExecutionResult(
            gate_id="test",
            gate_type="test",
            approved=False,
        )

        assert result.value is None
        assert result.reason is None
