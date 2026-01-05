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

"""Tests for UnifiedTeamCoordinator."""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.teams import (
    AgentMessage,
    ITeamCoordinator,
    MessageType,
    TeamFormation,
    UnifiedTeamCoordinator,
    create_coordinator,
)
from victor.teams.protocols import ITeamMember


class MockTeamMember:
    """Mock team member for testing."""

    def __init__(self, member_id: str, output: str = "Done"):
        self._id = member_id
        self._output = output
        self._role = MagicMock()
        self._messages: list = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def role(self) -> Any:
        return self._role

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        return self._output

    async def receive_message(
        self, message: AgentMessage
    ) -> Optional[AgentMessage]:
        self._messages.append(message)
        return None


class FailingMember(MockTeamMember):
    """Mock member that always fails."""

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        raise RuntimeError("Task failed")


class TestProtocolCompliance:
    """Tests for ITeamCoordinator protocol compliance."""

    def test_implements_iteamcoordinator(self):
        """UnifiedTeamCoordinator must implement ITeamCoordinator."""
        coordinator = UnifiedTeamCoordinator()
        # Check protocol methods exist
        assert hasattr(coordinator, "add_member")
        assert hasattr(coordinator, "set_formation")
        assert hasattr(coordinator, "execute_task")
        assert hasattr(coordinator, "broadcast")

    def test_fluent_interface(self):
        """Methods should return self for chaining."""
        coordinator = UnifiedTeamCoordinator()
        member = MockTeamMember("m1")

        result = coordinator.add_member(member).set_formation(
            TeamFormation.PARALLEL
        )
        assert result is coordinator

    def test_factory_returns_protocol(self):
        """Factory should return ITeamCoordinator implementation."""
        coordinator = create_coordinator()
        assert hasattr(coordinator, "add_member")
        assert hasattr(coordinator, "execute_task")


class TestFormations:
    """Tests for different formation patterns."""

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Sequential formation should execute members in order."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Result1"))
        coordinator.add_member(MockTeamMember("m2", "Result2"))
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task("Test task", {})

        assert result["success"] is True
        assert "m1" in result["member_results"]
        assert "m2" in result["member_results"]
        assert result["formation"] == "sequential"

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Parallel formation should execute all members concurrently."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Result1"))
        coordinator.add_member(MockTeamMember("m2", "Result2"))
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task("Test task", {})

        assert result["success"] is True
        assert len(result["member_results"]) == 2
        assert result["formation"] == "parallel"

    @pytest.mark.asyncio
    async def test_hierarchical_execution(self):
        """Hierarchical formation should have manager plan and synthesize."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        manager = MockTeamMember("manager", "Plan: do X")
        worker = MockTeamMember("worker", "Did X")
        coordinator.add_member(manager)
        coordinator.add_member(worker)
        coordinator.set_manager(manager)
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        result = await coordinator.execute_task("Test task", {})

        assert result["success"] is True
        assert "manager" in result["member_results"]
        assert "worker" in result["member_results"]
        assert result["formation"] == "hierarchical"

    @pytest.mark.asyncio
    async def test_pipeline_execution(self):
        """Pipeline formation should chain outputs."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("stage1", "Stage1Output"))
        coordinator.add_member(MockTeamMember("stage2", "Stage2Output"))
        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task("Initial input", {})

        assert result["success"] is True
        assert result["final_output"] == "Stage2Output"  # Last stage output
        assert result["formation"] == "pipeline"

    @pytest.mark.asyncio
    async def test_consensus_execution(self):
        """Consensus formation should require agreement."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "Agreed"))
        coordinator.add_member(MockTeamMember("m2", "Agreed"))
        coordinator.set_formation(TeamFormation.CONSENSUS)

        result = await coordinator.execute_task(
            "Test task", {"max_consensus_rounds": 1}
        )

        assert result["success"] is True
        assert result.get("consensus_achieved") is True
        assert result["formation"] == "consensus"


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_no_members_error(self):
        """Should fail gracefully with no members."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        result = await coordinator.execute_task("Test", {})

        assert result["success"] is False
        assert "No team members" in result["error"]

    @pytest.mark.asyncio
    async def test_member_failure_sequential(self):
        """Sequential should handle member failures."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "OK"))
        coordinator.add_member(FailingMember("m2"))
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task("Test", {})

        assert result["success"] is False
        assert result["member_results"]["m2"].success is False
        assert result["member_results"]["m2"].error is not None

    @pytest.mark.asyncio
    async def test_member_failure_parallel(self):
        """Parallel should handle member failures."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        coordinator.add_member(MockTeamMember("m1", "OK"))
        coordinator.add_member(FailingMember("m2"))
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task("Test", {})

        assert result["success"] is False
        assert result["member_results"]["m1"].success is True
        assert result["member_results"]["m2"].success is False


class TestBroadcast:
    """Tests for message broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self):
        """Broadcast should send to all members."""
        coordinator = UnifiedTeamCoordinator(enable_observability=False)
        m1 = MockTeamMember("m1")
        m2 = MockTeamMember("m2")
        coordinator.add_member(m1)
        coordinator.add_member(m2)

        message = AgentMessage(
            sender_id="external",
            recipient_id=None,
            content="Hello all",
            message_type=MessageType.STATUS,
        )

        responses = await coordinator.broadcast(message)

        assert len(responses) == 2
        assert len(m1._messages) == 1
        assert len(m2._messages) == 1


class TestObservability:
    """Tests for observability features."""

    def test_set_execution_context(self):
        """Should set execution context."""
        coordinator = UnifiedTeamCoordinator()
        coordinator.set_execution_context(
            task_type="feature",
            complexity="high",
            vertical="coding",
            trigger="manual",
        )

        ctx = coordinator._get_observability_context()
        assert ctx["task_type"] == "feature"
        assert ctx["complexity"] == "high"
        assert ctx["vertical"] == "coding"
        assert ctx["trigger"] == "manual"

    def test_progress_callback(self):
        """Should support progress callbacks."""
        coordinator = UnifiedTeamCoordinator()
        progress_calls = []

        def callback(member_id, status, progress):
            progress_calls.append((member_id, status, progress))

        coordinator.set_progress_callback(callback)
        coordinator._report_progress("m1", "running", 0.5)

        assert len(progress_calls) == 1
        assert progress_calls[0] == ("m1", "running", 0.5)


class TestFactoryFunction:
    """Tests for create_coordinator factory."""

    def test_default_creates_unified(self):
        """Default should create UnifiedTeamCoordinator."""
        coordinator = create_coordinator()
        assert isinstance(coordinator, UnifiedTeamCoordinator)

    def test_lightweight_creates_framework(self):
        """Lightweight should create FrameworkTeamCoordinator."""
        coordinator = create_coordinator(lightweight=True)
        # Should be the framework coordinator
        assert hasattr(coordinator, "add_member")
        assert not isinstance(coordinator, UnifiedTeamCoordinator)

    def test_disable_observability(self):
        """Should respect observability flag."""
        coordinator = create_coordinator(with_observability=False)
        assert isinstance(coordinator, UnifiedTeamCoordinator)
        assert coordinator._observability_enabled is False

    def test_disable_rl(self):
        """Should respect RL flag."""
        coordinator = create_coordinator(with_rl=False)
        assert isinstance(coordinator, UnifiedTeamCoordinator)
        assert coordinator._rl_enabled is False


class TestClearAndReset:
    """Tests for clear/reset functionality."""

    def test_clear_members(self):
        """Clear should remove all members."""
        coordinator = UnifiedTeamCoordinator()
        coordinator.add_member(MockTeamMember("m1"))
        coordinator.add_member(MockTeamMember("m2"))

        assert len(coordinator.members) == 2
        coordinator.clear()
        assert len(coordinator.members) == 0

    def test_clear_returns_self(self):
        """Clear should return self for chaining."""
        coordinator = UnifiedTeamCoordinator()
        result = coordinator.clear()
        assert result is coordinator


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
