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

from __future__ import annotations

from typing import Any

"""Integration tests for team coordinator recursion tracking.

Tests the integration between UnifiedTeamCoordinator and RecursionContext
to ensure proper tracking of nested team and workflow execution.
"""

import pytest

from victor.core.errors import RecursionDepthError
from victor.teams import UnifiedTeamCoordinator
from victor.teams.types import AgentMessage
from victor.workflows.recursion import RecursionContext, RecursionGuard


class MockTeamMember:
    """Mock team member for testing."""

    def __init__(self, member_id: str, response: str = "Task complete") -> Any:
        self.id = member_id
        self.role = "test_role"
        self._response = response
        self._messages: list[AgentMessage] = []

    async def execute_task(self, task: str, context: dict) -> str:
        """Execute task and return mock response."""
        return self._response

    async def receive_message(self, message: AgentMessage) -> AgentMessage | None:
        """Receive message and return mock response."""
        self._messages.append(message)
        return AgentMessage(
            sender_id=self.id,
            recipient_id=message.sender_id,
            content=f"Received: {message.content}",
            message_type="response",
        )


class TestTeamCoordinatorRecursionIntegration:
    """Test recursion tracking integration in UnifiedTeamCoordinator."""

    def test_init_with_default_recursion_context(self) -> None:
        """Test that coordinator creates default RecursionContext if not provided."""
        coordinator = UnifiedTeamCoordinator(
            orchestrator=None,
            lightweight_mode=True,
        )

        # Should have a recursion context
        assert hasattr(coordinator, "_recursion_ctx")
        assert isinstance(coordinator._recursion_ctx, RecursionContext)
        assert coordinator._recursion_ctx.max_depth == 3  # Default
        assert coordinator._recursion_ctx.current_depth == 0

    def test_init_with_custom_recursion_context(self) -> None:
        """Test that coordinator accepts custom RecursionContext."""
        custom_ctx = RecursionContext(max_depth=5)
        coordinator = UnifiedTeamCoordinator(
            orchestrator=None,
            recursion_context=custom_ctx,
            lightweight_mode=True,
        )

        # Should use the provided context
        assert coordinator._recursion_ctx is custom_ctx
        assert coordinator._recursion_ctx.max_depth == 5

    def test_get_recursion_depth(self) -> None:
        """Test get_recursion_depth returns current depth."""
        coordinator = UnifiedTeamCoordinator(
            orchestrator=None,
            lightweight_mode=True,
        )

        # Initially at depth 0
        assert coordinator.get_recursion_depth() == 0

        # Simulate entering recursion
        coordinator._recursion_ctx.enter("workflow", "test")
        assert coordinator.get_recursion_depth() == 1

        # Clean up
        coordinator._recursion_ctx.exit()

    def test_can_spawn_nested(self) -> None:
        """Test can_spawn_nested checks recursion limits."""
        coordinator = UnifiedTeamCoordinator(
            orchestrator=None,
            lightweight_mode=True,
        )

        # At depth 0, can spawn 1 level
        assert coordinator.can_spawn_nested() is True

        # Enter to depth 1
        coordinator._recursion_ctx.enter("workflow", "test1")
        assert coordinator.can_spawn_nested() is True

        # Enter to depth 2
        coordinator._recursion_ctx.enter("workflow", "test2")
        assert coordinator.can_spawn_nested() is True

        # Enter to depth 3 (max for default context)
        coordinator._recursion_ctx.enter("team", "test3")
        assert coordinator.can_spawn_nested() is False

        # Clean up
        coordinator._recursion_ctx.exit()
        coordinator._recursion_ctx.exit()
        coordinator._recursion_ctx.exit()

    @pytest.mark.asyncio
    async def test_execute_task_tracks_recursion_depth(self) -> None:
        """Test that execute_task properly tracks recursion depth."""
        coordinator = UnifiedTeamCoordinator(
            orchestrator=None,
            lightweight_mode=True,
        )

        # Add a mock member
        member = MockTeamMember("member1", "Done")
        coordinator.add_member(member)

        # Execute task
        result = await coordinator.execute_task("Test task", {"team_name": "TestTeam"})

        # Verify execution succeeded
        assert result["success"] is True

        # Verify recursion was tracked (depth should be back to 0 after execution)
        assert coordinator.get_recursion_depth() == 0

    @pytest.mark.asyncio
    async def test_execute_task_with_nested_recursion(self) -> None:
        """Test execute_task with pre-existing recursion depth."""
        custom_ctx = RecursionContext(max_depth=3)
        coordinator = UnifiedTeamCoordinator(
            orchestrator=None,
            recursion_context=custom_ctx,
            lightweight_mode=True,
        )

        # Add a mock member
        member = MockTeamMember("member1", "Done")
        coordinator.add_member(member)

        # Simulate being at depth 1 (e.g., workflow spawned team)
        custom_ctx.enter("workflow", "outer_workflow")
        assert custom_ctx.current_depth == 1

        # Execute team (should go to depth 2)
        result = await coordinator.execute_task("Test task", {"team_name": "TestTeam"})

        # Verify execution succeeded
        assert result["success"] is True

        # Depth should be back to 1 after team execution
        assert coordinator.get_recursion_depth() == 1

        # Clean up
        custom_ctx.exit()

    @pytest.mark.asyncio
    async def test_execute_task_respects_recursion_limit(self) -> None:
        """Test that execute_task respects recursion depth limits."""
        custom_ctx = RecursionContext(max_depth=2)
        coordinator = UnifiedTeamCoordinator(
            orchestrator=None,
            recursion_context=custom_ctx,
            lightweight_mode=True,
        )

        # Add a mock member
        member = MockTeamMember("member1", "Done")
        coordinator.add_member(member)

        # Simulate being at max depth
        custom_ctx.enter("workflow", "outer1")
        custom_ctx.enter("team", "inner_team")
        assert custom_ctx.current_depth == 2

        # Try to execute team - should fail with RecursionDepthError
        with pytest.raises(RecursionDepthError) as exc_info:
            await coordinator.execute_task("Test task", {"team_name": "TestTeam"})

        # Verify error details
        error = exc_info.value
        assert error.current_depth == 2
        assert error.max_depth == 2
        assert "team" in error.message.lower() or "recursion" in error.message.lower()

        # Clean up
        custom_ctx.exit()
        custom_ctx.exit()

    @pytest.mark.asyncio
    async def test_execute_task_includes_recursion_in_events(self) -> None:
        """Test that execute_task includes recursion depth in events."""
        # Create coordinator with observability
        coordinator = UnifiedTeamCoordinator(
            orchestrator=None,
            enable_observability=True,
            lightweight_mode=False,
        )

        # Add a mock member
        member = MockTeamMember("member1", "Done")
        coordinator.add_member(member)

        # Track emitted events
        emitted_events = []
        original_emit = coordinator._emit_team_event

        def capture_event(event_type: Any, data: Any) -> Any:
            emitted_events.append((event_type, data))
            return original_emit(event_type, data)

        coordinator._emit_team_event = capture_event

        # Execute task
        result = await coordinator.execute_task("Test task", {"team_name": "TestTeam"})

        # Verify events were emitted
        assert len(emitted_events) >= 1

        # Check that "started" event includes recursion depth
        started_events = [e for e in emitted_events if e[0] == "started"]
        if started_events:
            _, data = started_events[0]
            assert "recursion_depth" in data
            assert isinstance(data["recursion_depth"], int)

    @pytest.mark.asyncio
    async def test_recursion_guard_cleanup_on_error(self) -> None:
        """Test that RecursionGuard properly cleans up even on errors."""
        coordinator = UnifiedTeamCoordinator(
            orchestrator=None,
            lightweight_mode=True,
        )

        # Add a member that raises an exception
        class FailingMember:
            def __init__(self) -> Any:
                self.id = "failing_member"
                self.role = "test"

            async def execute_task(self, task: str, context: dict) -> str:
                raise RuntimeError("Intentional failure")

        coordinator.add_member(FailingMember())

        custom_ctx = RecursionContext(max_depth=3)
        coordinator._recursion_ctx = custom_ctx

        # Execute task (should fail)
        result = await coordinator.execute_task("Test task", {"team_name": "TestTeam"})

        # Verify execution failed but didn't crash
        assert result["success"] is False

        # Verify recursion depth was properly cleaned up (back to 0)
        assert custom_ctx.current_depth == 0
        assert len(custom_ctx.execution_stack) == 0

    @pytest.mark.asyncio
    async def test_shared_recursion_context_across_coordinators(self) -> None:
        """Test that multiple coordinators can share a recursion context."""
        shared_ctx = RecursionContext(max_depth=3)

        # Create two coordinators with shared context
        coordinator1 = UnifiedTeamCoordinator(
            orchestrator=None,
            recursion_context=shared_ctx,
            lightweight_mode=True,
        )

        coordinator2 = UnifiedTeamCoordinator(
            orchestrator=None,
            recursion_context=shared_ctx,
            lightweight_mode=True,
        )

        # Add members to both
        coordinator1.add_member(MockTeamMember("member1", "Done"))
        coordinator2.add_member(MockTeamMember("member2", "Done"))

        # Execute first coordinator
        result1 = await coordinator1.execute_task("Task 1", {"team_name": "Team1"})
        assert result1["success"] is True

        # Depth should be 0 after execution
        assert shared_ctx.current_depth == 0

        # Manually enter depth to simulate nesting
        shared_ctx.enter("workflow", "outer")

        # Execute second coordinator (should be at depth 2 during execution)
        result2 = await coordinator2.execute_task("Task 2", {"team_name": "Team2"})
        assert result2["success"] is True

        # Depth should be back to 1
        assert shared_ctx.current_depth == 1

        # Clean up
        shared_ctx.exit()


class TestRecursionContextMethods:
    """Test RecursionContext helper methods."""

    def test_get_depth_info(self) -> None:
        """Test get_depth_info returns complete information."""
        ctx = RecursionContext(max_depth=5)

        # Enter a few levels
        ctx.enter("workflow", "outer")
        ctx.enter("team", "inner")

        info = ctx.get_depth_info()

        assert info["current_depth"] == 2
        assert info["max_depth"] == 5
        assert info["remaining_depth"] == 3
        assert len(info["execution_stack"]) == 2
        assert "workflow:outer" in info["execution_stack"]
        assert "team:inner" in info["execution_stack"]

        # Clean up
        ctx.exit()
        ctx.exit()

    def test_recursion_context_repr(self) -> None:
        """Test RecursionContext string representation."""
        ctx = RecursionContext(max_depth=3)
        ctx.enter("workflow", "test")

        repr_str = repr(ctx)
        assert "RecursionContext" in repr_str
        assert "depth=1/3" in repr_str
        assert "workflow:test" in repr_str

        ctx.exit()

    def test_recursion_context_context_manager(self) -> None:
        """Test RecursionContext as context manager."""
        ctx = RecursionContext(max_depth=3)

        with ctx:
            ctx.enter("workflow", "test")
            assert ctx.current_depth == 1
            ctx.enter("team", "inner")
            assert ctx.current_depth == 2

        # After context manager exits, should be reset
        assert ctx.current_depth == 0
        assert len(ctx.execution_stack) == 0

    def test_recursion_guard_context_manager(self) -> None:
        """Test RecursionGuard as context manager."""
        ctx = RecursionContext(max_depth=3)

        with RecursionGuard(ctx, "workflow", "test_workflow"):
            assert ctx.current_depth == 1
            assert "workflow:test_workflow" in ctx.execution_stack

        # After guard exits, should be decremented
        assert ctx.current_depth == 0
        assert len(ctx.execution_stack) == 0

    def test_recursion_guard_with_exception(self) -> None:
        """Test RecursionGuard properly exits even with exceptions."""
        ctx = RecursionContext(max_depth=3)

        try:
            with RecursionGuard(ctx, "workflow", "test"):
                # The guard already entered "workflow:test", so we're at depth 1
                assert ctx.current_depth == 1
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass

        # Should still be properly cleaned up
        assert ctx.current_depth == 0
        assert len(ctx.execution_stack) == 0
