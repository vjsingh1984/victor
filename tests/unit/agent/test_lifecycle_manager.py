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

"""Tests for LifecycleManager."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, call
from typing import Any, Dict, Optional

from victor.agent.lifecycle_manager import LifecycleManager
from victor.protocols.lifecycle import SessionConfig, SessionMetadata, CleanupResult


class TestLifecycleManager:
    """Tests for LifecycleManager class."""

    @pytest.fixture
    def conversation_controller(self):
        """Create a mock conversation controller."""
        controller = MagicMock()
        controller.clear = MagicMock()
        controller.reset = MagicMock()
        return controller

    @pytest.fixture
    def metrics_collector(self):
        """Create a mock metrics collector."""
        collector = MagicMock()
        collector.reset_stats = MagicMock()
        return collector

    @pytest.fixture
    def context_compactor(self):
        """Create a mock context compactor."""
        compactor = MagicMock()
        compactor.reset_statistics = MagicMock()
        return compactor

    @pytest.fixture
    def sequence_tracker(self):
        """Create a mock sequence tracker."""
        tracker = MagicMock()
        tracker.clear_history = MagicMock()
        return tracker

    @pytest.fixture
    def usage_analytics(self):
        """Create a mock usage analytics."""
        analytics = MagicMock()
        analytics._current_session = MagicMock()
        analytics.end_session = MagicMock()
        analytics.start_session = MagicMock()
        return analytics

    @pytest.fixture
    def reminder_manager(self):
        """Create a mock reminder manager."""
        manager = MagicMock()
        manager.reset = MagicMock()
        return manager

    @pytest.fixture
    def provider(self):
        """Create a mock provider."""
        provider = AsyncMock()
        provider.close = AsyncMock()
        return provider

    @pytest.fixture
    def code_manager(self):
        """Create a mock code execution manager."""
        manager = MagicMock()
        manager.stop = MagicMock()
        return manager

    @pytest.fixture
    def semantic_selector(self):
        """Create a mock semantic selector."""
        selector = AsyncMock()
        selector.close = AsyncMock()
        return selector

    @pytest.fixture
    def background_tasks(self):
        """Create mock background tasks."""
        # Create simple mock tasks
        task1 = MagicMock()
        task1.done = MagicMock(return_value=False)
        task1.cancel = MagicMock()
        task1.result = MagicMock(return_value=None)

        task2 = MagicMock()
        task2.done = MagicMock(return_value=True)
        task2.cancel = MagicMock()

        return [task1, task2]

    @pytest.fixture
    def lifecycle_manager(
        self,
        conversation_controller,
        metrics_collector,
        context_compactor,
        sequence_tracker,
        usage_analytics,
        reminder_manager,
    ):
        """Create LifecycleManager with mocks."""
        from victor.agent.lifecycle_manager import LifecycleManager

        return LifecycleManager(
            conversation_controller=conversation_controller,
            metrics_collector=metrics_collector,
            context_compactor=context_compactor,
            sequence_tracker=sequence_tracker,
            usage_analytics=usage_analytics,
            reminder_manager=reminder_manager,
        )

    def test_init(self, lifecycle_manager, conversation_controller):
        """Test lifecycle manager initialization."""
        assert lifecycle_manager._conversation_controller is conversation_controller

    def test_reset_conversation(self, lifecycle_manager, conversation_controller):
        """Test conversation reset."""
        # Reset conversation
        lifecycle_manager.reset_conversation()

        # Verify controller was reset
        conversation_controller.reset.assert_called_once()

    def test_reset_conversation_with_all_components(
        self,
        lifecycle_manager,
        conversation_controller,
        metrics_collector,
        context_compactor,
        sequence_tracker,
        usage_analytics,
        reminder_manager,
    ):
        """Test conversation reset with all components."""
        # Reset
        lifecycle_manager.reset_conversation()

        # Verify all components were reset
        conversation_controller.reset.assert_called_once()
        reminder_manager.reset.assert_called_once()
        metrics_collector.reset_stats.assert_called_once()
        context_compactor.reset_statistics.assert_called_once()
        sequence_tracker.clear_history.assert_called_once()

        # Verify usage analytics session was restarted
        usage_analytics.end_session.assert_called_once()
        usage_analytics.start_session.assert_called_once()

    def test_reset_conversation_without_usage_analytics(
        self, lifecycle_manager, conversation_controller, usage_analytics
    ):
        """Test reset when usage analytics has no active session."""
        # Setup analytics with no active session
        usage_analytics._current_session = None

        # Reset
        lifecycle_manager.reset_conversation()

        # Verify no error, graceful handling
        conversation_controller.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_success(self, lifecycle_manager, usage_analytics):
        """Test successful graceful shutdown."""
        # Mock flush_analytics as async function
        lifecycle_manager._flush_analytics = AsyncMock(return_value={"analytics": True})

        # Mock stop_health_monitoring
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown
        result = await lifecycle_manager.graceful_shutdown()

        # Verify result
        assert result["analytics_flushed"] is True
        assert result["health_monitoring_stopped"] is True
        assert result["session_ended"] is True

        # Verify usage analytics session ended
        usage_analytics.end_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_analytics_failure(
        self, lifecycle_manager, usage_analytics
    ):
        """Test graceful shutdown when analytics flush fails."""
        # Mock flush_analytics that fails
        lifecycle_manager._flush_analytics = AsyncMock(return_value={"analytics": False})

        # Mock stop_health_monitoring
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown
        result = await lifecycle_manager.graceful_shutdown()

        # Verify partial success
        assert result["analytics_flushed"] is False
        assert result["health_monitoring_stopped"] is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown_without_usage_analytics(self, lifecycle_manager):
        """Test graceful shutdown when usage analytics not configured."""
        # Remove usage analytics
        lifecycle_manager._usage_analytics = None

        # Mock dependencies
        lifecycle_manager._flush_analytics = AsyncMock(return_value={})
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown - should not fail
        result = await lifecycle_manager.graceful_shutdown()

        # Verify success despite no analytics
        assert result["session_ended"] is True

    @pytest.mark.asyncio
    async def test_shutdown(
        self,
        lifecycle_manager,
        provider,
        code_manager,
        semantic_selector,
        background_tasks,
    ):
        """Test full shutdown."""
        # Setup dependencies
        lifecycle_manager._provider = provider
        lifecycle_manager._code_manager = code_manager
        lifecycle_manager._semantic_selector = semantic_selector
        lifecycle_manager._background_tasks = background_tasks
        lifecycle_manager._usage_logger = MagicMock()

        # Shutdown
        await lifecycle_manager.shutdown()

        # Verify background tasks cancelled (if any)
        if background_tasks:
            background_tasks[0].cancel.assert_called_once()

        # Verify provider closed
        provider.close.assert_called_once()

        # Verify code manager stopped
        code_manager.stop.assert_called_once()

        # Verify semantic selector closed
        semantic_selector.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_provider_error(self, lifecycle_manager, provider):
        """Test shutdown when provider close fails."""
        # Setup provider that raises error
        provider.close = AsyncMock(side_effect=RuntimeError("Provider error"))

        lifecycle_manager._provider = provider
        lifecycle_manager._code_manager = None
        lifecycle_manager._semantic_selector = None
        lifecycle_manager._background_tasks = []
        lifecycle_manager._usage_logger = MagicMock()

        # Shutdown - should not raise
        await lifecycle_manager.shutdown()

        # Verify close was attempted
        provider.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_without_background_tasks(self, lifecycle_manager, provider):
        """Test shutdown when no background tasks."""
        # Setup with no background tasks
        lifecycle_manager._background_tasks = []
        lifecycle_manager._provider = provider
        lifecycle_manager._code_manager = None
        lifecycle_manager._semantic_selector = None
        lifecycle_manager._usage_logger = MagicMock()

        # Shutdown - should not fail
        await lifecycle_manager.shutdown()

        # Verify provider closed
        provider.close.assert_called_once()

    def test_get_session_stats(self, lifecycle_manager):
        """Test getting session statistics."""
        # Setup conversation controller
        lifecycle_manager._conversation_controller.message_count = MagicMock(return_value=42)

        # Get stats
        stats = lifecycle_manager.get_session_stats()

        # Verify stats
        assert "message_count" in stats
        assert stats["message_count"] == 42

    def test_recover_session_success(self, lifecycle_manager):
        """Test successful session recovery."""
        # Mock memory manager
        memory_manager = MagicMock()
        session = MagicMock()
        session.session_id = "test-session-123"
        session.messages = [MagicMock(role="user", content="Hello", to_provider_format=MagicMock())]

        memory_manager.get_session = MagicMock(return_value=session)

        # Recover
        result = lifecycle_manager.recover_session(
            session_id="test-session-123",
            memory_manager=memory_manager,
        )

        # Verify recovery
        assert result is True
        memory_manager.get_session.assert_called_once_with("test-session-123")

    def test_recover_session_not_found(self, lifecycle_manager):
        """Test session recovery when session not found."""
        # Mock memory manager that returns None
        memory_manager = MagicMock()
        memory_manager.get_session = MagicMock(return_value=None)

        # Recover
        result = lifecycle_manager.recover_session(
            session_id="nonexistent-session",
            memory_manager=memory_manager,
        )

        # Verify failure
        assert result is False

    def test_recover_session_without_memory_manager(self, lifecycle_manager):
        """Test session recovery without memory manager."""
        # Recover without memory manager
        result = lifecycle_manager.recover_session(
            session_id="test-session",
            memory_manager=None,
        )

        # Verify failure
        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_session_basic(self, lifecycle_manager):
        """Test basic session initialization."""
        # Create session config
        config = SessionConfig(
            provider="anthropic",
            model="claude-sonnet-4-5",
            temperature=0.7,
        )

        # Initialize session
        metadata = await lifecycle_manager.initialize_session(
            session_id="test-session-123",
            config=config,
        )

        # Verify metadata structure
        assert isinstance(metadata, SessionMetadata)
        assert metadata.session_id == "test-session-123"
        assert metadata.config is config
        assert isinstance(metadata.created_at, str)
        assert len(metadata.created_at) > 0

        # Verify resources tracked
        assert "conversation_controller" in metadata.resources
        assert "metrics_collector" in metadata.resources
        assert "usage_analytics" in metadata.resources
        assert "context_compactor" in metadata.resources

        # Verify all resources marked as available
        assert metadata.resources["conversation_controller"] is True
        assert metadata.resources["metrics_collector"] is True
        assert metadata.resources["usage_analytics"] is True
        assert metadata.resources["context_compactor"] is True

        # Verify conversation was reset
        lifecycle_manager._conversation_controller.reset.assert_called_once()

        # Verify metrics were reset
        lifecycle_manager._metrics_collector.reset_stats.assert_called_once()

        # Verify analytics session was restarted
        lifecycle_manager._usage_analytics.end_session.assert_called_once()
        lifecycle_manager._usage_analytics.start_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_session_without_analytics(
        self, lifecycle_manager, conversation_controller
    ):
        """Test session initialization when usage analytics is None."""
        # Remove analytics
        lifecycle_manager._usage_analytics = None

        # Create session config
        config = SessionConfig(provider="openai", model="gpt-4")

        # Initialize session - should not fail
        metadata = await lifecycle_manager.initialize_session(
            session_id="test-session-no-analytics",
            config=config,
        )

        # Verify success
        assert metadata.session_id == "test-session-no-analytics"
        assert metadata.resources["usage_analytics"] is False
        conversation_controller.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_session_without_metrics(
        self, lifecycle_manager, conversation_controller
    ):
        """Test session initialization when metrics collector is None."""
        # Remove metrics collector
        lifecycle_manager._metrics_collector = None

        # Create session config
        config = SessionConfig(provider="google", model="gemini-pro")

        # Initialize session - should not fail
        metadata = await lifecycle_manager.initialize_session(
            session_id="test-session-no-metrics",
            config=config,
        )

        # Verify success
        assert metadata.session_id == "test-session-no-metrics"
        assert metadata.resources["metrics_collector"] is False
        conversation_controller.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_session_with_various_configs(self, lifecycle_manager):
        """Test session initialization with various configurations."""
        configs = [
            SessionConfig(provider="anthropic", model="claude-sonnet-4-5", temperature=0.5),
            SessionConfig(provider="openai", model="gpt-4", temperature=1.0, max_tokens=8192),
            SessionConfig(
                provider="google",
                model="gemini-pro",
                temperature=0.0,
                vertical="coding",
            ),
        ]

        for i, config in enumerate(configs):
            metadata = await lifecycle_manager.initialize_session(
                session_id=f"session-{i}",
                config=config,
            )

            assert metadata.config is config
            assert metadata.session_id == f"session-{i}"

    @pytest.mark.asyncio
    async def test_cleanup_session_basic(
        self, lifecycle_manager, conversation_controller, metrics_collector
    ):
        """Test basic session cleanup."""
        # Cleanup session
        result = await lifecycle_manager.cleanup_session("test-session-123")

        # Verify result structure
        assert isinstance(result, CleanupResult)
        assert result.success is True
        assert result.session_id == "test-session-123"
        assert result.error_message is None

        # Verify resources were freed
        assert "conversation" in result.resources_freed
        assert "analytics" in result.resources_freed
        assert "metrics" in result.resources_freed
        assert "context_compactor" in result.resources_freed

        # Verify conversation was reset
        conversation_controller.reset.assert_called()

        # Verify metrics were reset
        metrics_collector.reset_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_session_without_analytics(
        self, lifecycle_manager, conversation_controller
    ):
        """Test session cleanup when usage analytics is None."""
        # Remove analytics
        lifecycle_manager._usage_analytics = None

        # Cleanup session - should not fail
        result = await lifecycle_manager.cleanup_session("test-session-no-analytics")

        # Verify success
        assert result.success is True
        assert "analytics" not in result.resources_freed
        assert "conversation" in result.resources_freed
        conversation_controller.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_session_without_metrics(
        self, lifecycle_manager, conversation_controller, metrics_collector
    ):
        """Test session cleanup when metrics collector is None."""
        # Remove metrics collector
        lifecycle_manager._metrics_collector = None

        # Cleanup session - should not fail
        result = await lifecycle_manager.cleanup_session("test-session-no-metrics")

        # Verify success
        assert result.success is True
        assert "metrics" not in result.resources_freed
        assert "conversation" in result.resources_freed
        conversation_controller.reset.assert_called_once()
        metrics_collector.reset_stats.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_session_without_compactor(self, lifecycle_manager, context_compactor):
        """Test session cleanup when context compactor is None."""
        # Remove context compactor
        lifecycle_manager._context_compactor = None

        # Cleanup session - should not fail
        result = await lifecycle_manager.cleanup_session("test-session-no-compactor")

        # Verify success
        assert result.success is True
        assert "context_compactor" not in result.resources_freed
        context_compactor.reset_statistics.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_session_metadata(self, lifecycle_manager):
        """Test that cleanup result contains correct metadata."""
        # Cleanup session
        result = await lifecycle_manager.cleanup_session("test-session-metadata")

        # Verify metadata
        assert result.metadata is not None
        assert "cleanup_type" in result.metadata
        assert result.metadata["cleanup_type"] == "full"

    @pytest.mark.asyncio
    async def test_initialize_then_cleanup_session_cycle(
        self, lifecycle_manager, conversation_controller
    ):
        """Test complete initialize -> cleanup cycle."""
        config = SessionConfig(provider="anthropic", model="claude-sonnet-4-5")

        # Initialize session
        metadata = await lifecycle_manager.initialize_session(
            session_id="cycle-session",
            config=config,
        )
        assert metadata.session_id == "cycle-session"

        # Verify session was initialized
        assert lifecycle_manager._session_config is config

        # Cleanup session
        result = await lifecycle_manager.cleanup_session("cycle-session")
        assert result.success is True
        assert "conversation" in result.resources_freed

        # Verify conversation was reset during both operations
        assert conversation_controller.reset.call_count >= 2

    @pytest.mark.asyncio
    async def test_get_session_status(self, lifecycle_manager):
        """Test getting session status."""
        # Setup message count
        lifecycle_manager._conversation_controller.message_count = MagicMock(return_value=42)

        # Get status
        status = await lifecycle_manager.get_session_status("test-session-status")

        # Verify status structure
        assert isinstance(status, dict)
        assert "healthy" in status
        assert "session_id" in status
        assert "message_count" in status
        assert "resources" in status

        # Verify values
        assert status["healthy"] is True
        assert status["session_id"] == "test-session-status"
        assert status["message_count"] == 42
        assert isinstance(status["resources"], dict)

        # Verify resources are tracked
        assert "conversation_controller" in status["resources"]
        assert "metrics_collector" in status["resources"]
        assert "usage_analytics" in status["resources"]
        assert "context_compactor" in status["resources"]

    @pytest.mark.asyncio
    async def test_get_session_status_without_message_count(self, lifecycle_manager):
        """Test getting session status when message_count not available."""
        # Setup controller without message_count
        lifecycle_manager._conversation_controller.message_count = None

        # Get status - should not fail
        status = await lifecycle_manager.get_session_status("test-session-no-count")

        # Verify defaults
        assert status["message_count"] == 0
        assert status["healthy"] is True


class TestLifecycleManagerIntegration:
    """Integration tests for LifecycleManager."""

    @pytest.fixture
    def lifecycle_manager(self):
        """Create lifecycle manager with real dependencies."""
        from victor.agent.lifecycle_manager import LifecycleManager

        # Use mocks for dependencies
        return LifecycleManager(
            conversation_controller=MagicMock(),
            metrics_collector=MagicMock(),
            context_compactor=MagicMock(),
            sequence_tracker=MagicMock(),
            usage_analytics=MagicMock(),
            reminder_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, lifecycle_manager):
        """Test complete lifecycle: reset -> shutdown."""
        # Reset
        lifecycle_manager.reset_conversation()

        # Setup for shutdown
        lifecycle_manager._provider = AsyncMock()
        lifecycle_manager._provider.close = AsyncMock()
        lifecycle_manager._code_manager = None
        lifecycle_manager._semantic_selector = None
        lifecycle_manager._background_tasks = []
        lifecycle_manager._usage_logger = MagicMock()
        lifecycle_manager._flush_analytics = AsyncMock(return_value={})
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown
        await lifecycle_manager.graceful_shutdown()
        await lifecycle_manager.shutdown()

        # Verify all components reset
        lifecycle_manager._conversation_controller.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_recover_shutdown_cycle(self, lifecycle_manager):
        """Test cycle: reset -> recover -> shutdown."""
        # Reset
        lifecycle_manager.reset_conversation()

        # Mock recovery
        memory_manager = MagicMock()
        session = MagicMock()
        session.session_id = "test-123"
        session.messages = []
        memory_manager.get_session = MagicMock(return_value=session)

        # Recover
        result = lifecycle_manager.recover_session(
            session_id="test-123",
            memory_manager=memory_manager,
        )
        assert result is True

        # Setup for shutdown
        lifecycle_manager._provider = AsyncMock()
        lifecycle_manager._provider.close = AsyncMock()
        lifecycle_manager._code_manager = None
        lifecycle_manager._semantic_selector = None
        lifecycle_manager._background_tasks = []
        lifecycle_manager._usage_logger = MagicMock()
        lifecycle_manager._flush_analytics = AsyncMock(return_value={})
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown
        await lifecycle_manager.shutdown()


class TestLifecycleManagerErrorHandling:
    """Error handling tests for LifecycleManager."""

    @pytest.fixture
    def lifecycle_manager(self):
        """Create lifecycle manager."""
        from victor.agent.lifecycle_manager import LifecycleManager

        return LifecycleManager(
            conversation_controller=MagicMock(),
            metrics_collector=MagicMock(),
            context_compactor=MagicMock(),
            sequence_tracker=MagicMock(),
            usage_analytics=MagicMock(),
            reminder_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_graceful_shutdown_handles_exception(self, lifecycle_manager):
        """Test that graceful shutdown exceptions are handled gracefully."""
        # Mock flush that raises exception
        lifecycle_manager._flush_analytics = AsyncMock(side_effect=RuntimeError("Flush failed"))
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown should handle exception
        result = await lifecycle_manager.graceful_shutdown()

        # Verify partial success despite error
        assert "analytics_flushed" in result
        assert result["analytics_flushed"] is False

    @pytest.mark.asyncio
    async def test_shutdown_handles_all_component_errors(self, lifecycle_manager):
        """Test shutdown when multiple components fail."""
        # Setup components that fail
        provider = AsyncMock()
        provider.close = AsyncMock(side_effect=RuntimeError("Provider error"))

        code_manager = MagicMock()
        code_manager.stop = MagicMock(side_effect=RuntimeError("Code manager error"))

        semantic_selector = AsyncMock()
        semantic_selector.close = AsyncMock(side_effect=RuntimeError("Selector error"))

        lifecycle_manager._provider = provider
        lifecycle_manager._code_manager = code_manager
        lifecycle_manager._semantic_selector = semantic_selector
        lifecycle_manager._background_tasks = []
        lifecycle_manager._usage_logger = MagicMock()

        # Shutdown should handle all errors
        await lifecycle_manager.shutdown()

        # Verify all cleanup was attempted
        provider.close.assert_called_once()
        code_manager.stop.assert_called_once()
        semantic_selector.close.assert_called_once()
