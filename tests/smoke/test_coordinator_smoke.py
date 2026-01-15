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

"""Smoke tests for coordinator-based orchestrator.

These tests verify basic functionality works correctly and MUST pass quickly.
Target: < 5 minutes for all tests.

Run with:
    pytest tests/smoke/test_coordinator_smoke.py -v
    pytest tests/smoke/test_coordinator_smoke.py -v -m smoke
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from victor.config.settings import Settings
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.coordinators.checkpoint_coordinator import CheckpointCoordinator
from victor.agent.coordinators.evaluation_coordinator import EvaluationCoordinator
from victor.agent.coordinators.metrics_coordinator import MetricsCoordinator
from victor.agent.coordinators.workflow_coordinator import WorkflowCoordinator
from victor.agent.streaming.coordinator import IterationCoordinator
from victor.teams import (
    create_coordinator,
    TeamFormation,
    TeamMessageBus,
    TeamSharedMemory,
)


@pytest.mark.smoke
@pytest.mark.unit
class TestCoordinatorSmokeTests:
    """Smoke tests for all coordinators.

    These tests verify basic instantiation and core functionality.
    They should run quickly and catch major issues.
    """

    def test_checkpoint_coordinator_creation(self):
        """Test CheckpointCoordinator can be created."""
        coordinator = CheckpointCoordinator()
        assert coordinator is not None
        assert hasattr(coordinator, "checkpoint")
        assert hasattr(coordinator, "restore")

    @pytest.mark.asyncio
    async def test_checkpoint_coordinator_basic_operations(self):
        """Test CheckpointCoordinator basic checkpoint/restore."""
        coordinator = CheckpointCoordinator()

        # Test checkpoint
        await coordinator.checkpoint("test_session", {"key": "value"})

        # Test restore
        restored = await coordinator.restore("test_session")
        assert restored is not None
        assert restored["key"] == "value"

    def test_evaluation_coordinator_creation(self):
        """Test EvaluationCoordinator can be created."""
        coordinator = EvaluationCoordinator()
        assert coordinator is not None
        assert hasattr(coordinator, "record_evaluation")
        assert hasattr(coordinator, "get_evaluations")

    @pytest.mark.asyncio
    async def test_evaluation_coordinator_basic_operations(self):
        """Test EvaluationCoordinator basic evaluation recording."""
        coordinator = EvaluationCoordinator()

        # Test recording
        await coordinator.record_evaluation(
            task_id="test_task",
            score=0.9,
            metrics={"accuracy": 0.95},
        )

        # Test retrieval
        evaluations = await coordinator.get_evaluations("test_task")
        assert len(evaluations) == 1
        assert evaluations[0]["score"] == 0.9

    def test_metrics_coordinator_creation(self):
        """Test MetricsCoordinator can be created."""
        coordinator = MetricsCoordinator()
        assert coordinator is not None
        assert hasattr(coordinator, "record_metric")
        assert hasattr(coordinator, "get_metrics")

    @pytest.mark.asyncio
    async def test_metrics_coordinator_basic_operations(self):
        """Test MetricsCoordinator basic metric recording."""
        coordinator = MetricsCoordinator()

        # Test recording
        await coordinator.record_metric("test_metric", 1.5, tags={"env": "test"})

        # Test retrieval
        metrics = await coordinator.get_metrics("test_metric")
        assert len(metrics) == 1
        assert metrics[0]["value"] == 1.5

    def test_workflow_coordinator_creation(self):
        """Test WorkflowCoordinator can be created."""
        coordinator = WorkflowCoordinator()
        assert coordinator is not None
        assert hasattr(coordinator, "compile_workflow")

    @pytest.mark.asyncio
    async def test_workflow_coordinator_basic_compilation(self):
        """Test WorkflowCoordinator basic workflow compilation."""
        coordinator = WorkflowCoordinator()

        workflow_def = {
            "nodes": [
                {"id": "start", "type": "agent", "role": "test"},
            ],
            "edges": [],
        }

        compiled = await coordinator.compile_workflow(workflow_def)
        assert compiled is not None

    def test_streaming_iteration_coordinator_creation(self):
        """Test IterationCoordinator can be created."""
        handler = MagicMock()
        loop_detector = MagicMock()
        settings = Settings()

        coordinator = IterationCoordinator(
            handler=handler,
            loop_detector=loop_detector,
            settings=settings,
        )

        assert coordinator is not None
        assert hasattr(coordinator, "should_continue")


@pytest.mark.smoke
@pytest.mark.unit
class TestTeamCoordinatorSmokeTests:
    """Smoke tests for team coordinators."""

    def test_create_coordinator_lightweight(self):
        """Test lightweight coordinator creation."""
        coordinator = create_coordinator(lightweight=True)
        assert coordinator is not None

    def test_create_coordinator_with_observability(self):
        """Test coordinator creation with observability."""
        coordinator = create_coordinator(
            lightweight=False,
            with_observability=True,
            with_rl=False,
        )
        assert coordinator is not None
        assert hasattr(coordinator, "get_metrics")

    def test_create_coordinator_with_rl(self):
        """Test coordinator creation with RL."""
        coordinator = create_coordinator(
            lightweight=False,
            with_observability=False,
            with_rl=True,
        )
        assert coordinator is not None
        assert hasattr(coordinator, "record_outcome")

    def test_team_formations(self):
        """Test all team formations can be set."""
        coordinator = create_coordinator(lightweight=True)

        formations = [
            TeamFormation.SEQUENTIAL,
            TeamFormation.PARALLEL,
            TeamFormation.HIERARCHICAL,
            TeamFormation.PIPELINE,
            TeamFormation.CONSENSUS,
        ]

        for formation in formations:
            coordinator.set_formation(formation)

    def test_team_message_bus_creation(self):
        """Test TeamMessageBus can be created."""
        bus = TeamMessageBus()
        assert bus is not None

    def test_team_shared_memory_creation(self):
        """Test TeamSharedMemory can be created."""
        memory = TeamSharedMemory()
        assert memory is not None


@pytest.mark.smoke
@pytest.mark.unit
class TestOrchestratorIntegrationSmokeTests:
    """Smoke tests for orchestrator integration."""

    def test_coordinators_importable(self):
        """Test all coordinators can be imported."""
        from victor.agent.coordinators import (
            CheckpointCoordinator,
            EvaluationCoordinator,
            MetricsCoordinator,
            WorkflowCoordinator,
        )

        assert CheckpointCoordinator is not None
        assert EvaluationCoordinator is not None
        assert MetricsCoordinator is not None
        assert WorkflowCoordinator is not None

    def test_team_coordinator_importable(self):
        """Test team coordinator can be imported."""
        from victor.teams import (
            create_coordinator,
            UnifiedTeamCoordinator,
            TeamFormation,
        )

        assert create_coordinator is not None
        assert UnifiedTeamCoordinator is not None
        assert TeamFormation is not None

    @patch("victor.agent.orchestrator.ProviderManager")
    @patch("victor.agent.orchestrator.ToolRegistrar")
    def test_orchestrator_creation_with_coordinators(self, mock_registrar, mock_provider_mgr):
        """Test orchestrator can be created with coordinators."""
        mock_provider = MagicMock()
        mock_provider.name = "test_provider"

        orchestrator = AgentOrchestrator(
            provider=mock_provider,
            settings=Settings(),
        )

        assert orchestrator is not None


@pytest.mark.smoke
@pytest.mark.unit
class TestBasicChatFunctionality:
    """Smoke tests for basic chat functionality."""

    @pytest.mark.asyncio
    async def test_basic_chat_message(self):
        """Test basic chat message processing."""
        from victor.protocols import Message

        # Create mock orchestrator
        with patch("victor.agent.orchestrator.ProviderManager"):
            with patch("victor.agent.orchestrator.ToolRegistrar"):
                orchestrator = AgentOrchestrator(
                    provider=MagicMock(),
                    settings=Settings(),
                )

        # Mock the provider response
        orchestrator.provider.chat = AsyncMock(
            return_value=MagicMock(
                content="Test response",
                usage=MagicMock(
                    total_tokens=10,
                    prompt_tokens=5,
                    completion_tokens=5,
                ),
            )
        )

        # Send message
        message = Message(role="user", content="Hello")
        response = await orchestrator.chat(message)

        assert response is not None
        assert response.content == "Test response"


@pytest.mark.smoke
@pytest.mark.unit
class TestStreamingFunctionality:
    """Smoke tests for streaming functionality."""

    def test_streaming_context_creation(self):
        """Test streaming context can be created."""
        from victor.agent.streaming.context import StreamingChatContext
        from victor.protocols import Message

        messages = [Message(role="user", content="Test")]
        context = StreamingChatContext(messages=messages)

        assert context is not None
        assert len(context.messages) == 1

    @pytest.mark.asyncio
    async def test_streaming_iteration_coordinator(self):
        """Test iteration coordinator for streaming."""
        from victor.agent.streaming.handler import StreamingChatHandler
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        handler = MagicMock(spec=StreamingChatHandler)
        loop_detector = MagicMock(spec=UnifiedTaskTracker)

        coordinator = IterationCoordinator(
            handler=handler,
            loop_detector=loop_detector,
            settings=Settings(),
        )

        # Test should_continue
        from victor.agent.streaming.context import StreamingChatContext
        from victor.protocols import Message

        ctx = StreamingChatContext(messages=[Message(role="user", content="Test")])
        result = MagicMock()

        should_continue = coordinator.should_continue(ctx, result)
        assert should_continue is not None


@pytest.mark.smoke
@pytest.mark.unit
class TestToolCallsSmokeTests:
    """Smoke tests for tool calling functionality."""

    def test_tool_call_creation(self):
        """Test ToolCall can be created."""
        from victor.agent.tool_calling.base import ToolCall

        tool_call = ToolCall(
            id="test_id",
            name="test_tool",
            arguments={"param": "value"},
        )

        assert tool_call is not None
        assert tool_call.name == "test_tool"

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test basic tool execution."""
        from victor.tools.base import BaseTool

        # Create a simple test tool
        class TestTool(BaseTool):
            name = "test_tool"
            description = "Test tool"

            def _execute(self, **kwargs):
                return {"result": "success"}

        tool = TestTool()
        result = tool.execute(param="value")

        assert result is not None
        assert result["result"] == "success"


@pytest.mark.smoke
@pytest.mark.unit
class TestContextCompactionSmokeTests:
    """Smoke tests for context compaction."""

    def test_context_compactor_creation(self):
        """Test context compactor can be created."""
        from victor.agent.context_compactor import ContextCompactor

        compactor = ContextCompactor(threshold_tokens=1000)

        assert compactor is not None

    @pytest.mark.asyncio
    async def test_context_compaction(self):
        """Test basic context compaction."""
        from victor.agent.context_compactor import ContextCompactor
        from victor.protocols import Message

        compactor = ContextCompactor(threshold_tokens=10)

        # Create many messages to trigger compaction
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(100)
        ]

        compacted = await compactor.compact_if_needed(messages)

        assert compacted is not None
        assert len(compacted) < len(messages)


@pytest.mark.smoke
@pytest.mark.unit
class TestAnalyticsTrackingSmokeTests:
    """Smoke tests for analytics tracking."""

    def test_usage_analytics_singleton(self):
        """Test UsageAnalytics singleton can be accessed."""
        from victor.agent.usage_analytics import UsageAnalytics

        analytics = UsageAnalytics.get_instance()

        assert analytics is not None

    @pytest.mark.asyncio
    async def test_analytics_recording(self):
        """Test basic analytics recording."""
        from victor.agent.usage_analytics import UsageAnalytics

        analytics = UsageAnalytics.get_instance()

        # Record some metrics
        await analytics.record_tool_usage(
            tool_name="test_tool",
            duration_ms=100,
            success=True,
        )

        # Record provider usage
        await analytics.record_provider_usage(
            provider_name="test_provider",
            model_name="test_model",
            total_tokens=100,
        )

        # Should not raise any errors
        assert True


@pytest.mark.smoke
@pytest.mark.unit
class TestBackwardCompatibilitySmokeTests:
    """Smoke tests for backward compatibility."""

    def test_legacy_imports(self):
        """Test legacy imports still work."""
        from victor.framework.coordinators import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        assert coordinator is not None

    def test_team_coordinator_factory(self):
        """Test team coordinator factory function."""
        from victor.teams import create_coordinator

        coordinator = create_coordinator(lightweight=True)
        assert coordinator is not None


# Performance marker - ensures tests run quickly
@pytest.mark.smoke
@pytest.mark.unit
class TestPerformanceSmokeTests:
    """Smoke tests for performance validation."""

    def test_coordinator_instantiation_performance(self):
        """Test coordinator instantiation is fast (< 100ms)."""
        import time

        start = time.time()

        coordinators = []
        for _ in range(10):
            coordinator = create_coordinator(lightweight=True)
            coordinators.append(coordinator)

        duration = (time.time() - start) * 1000  # ms
        avg_time = duration / len(coordinators)

        # Each coordinator should instantiate in < 10ms
        assert avg_time < 10.0, f"Coordinator instantiation too slow: {avg_time:.2f}ms"

    @pytest.mark.asyncio
    async def test_checkpoint_restore_performance(self):
        """Test checkpoint/restore is fast (< 50ms)."""
        import time

        coordinator = CheckpointCoordinator()

        # Test checkpoint performance
        start = time.time()
        await coordinator.checkpoint("perf_test", {"data": "value"})
        checkpoint_duration = (time.time() - start) * 1000

        assert checkpoint_duration < 50.0, f"Checkpoint too slow: {checkpoint_duration:.2f}ms"

        # Test restore performance
        start = time.time()
        await coordinator.restore("perf_test")
        restore_duration = (time.time() - start) * 1000

        assert restore_duration < 50.0, f"Restore too slow: {restore_duration:.2f}ms"
