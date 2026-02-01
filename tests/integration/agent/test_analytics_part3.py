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

"""Integration tests for session stats and flush analytics (Part 3 of 3).

This module tests:
- Session stats integration with MemoryManager (ConversationStore)
- Flush analytics integration with EvaluationCoordinator
- Error handling and graceful degradation
- Integration between orchestrator and coordinators

Test Coverage:
    - TestSessionStatsIntegration (3 tests)
    - TestFlushAnalyticsIntegration (4 tests)

Total tests: 7

This is Part 3 of 3, focusing on session stats and flush analytics.
Follows TDD Red phase - tests will fail initially.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from victor.agent.coordinators.evaluation_coordinator import EvaluationCoordinator
from victor.agent.conversation_memory import ConversationStore, MessageRole


# =============================================================================
# Test Class 3: TestSessionStatsIntegration
# =============================================================================


@pytest.mark.integration
class TestSessionStatsIntegration:
    """Test session stats integration with MemoryManager (ConversationStore)."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> Path:
        """Create a temporary database path for testing."""
        return tmp_path / "test_session_stats.db"

    @pytest.fixture
    def memory_store(self, temp_db_path: Path) -> ConversationStore:
        """Create a ConversationStore instance for testing."""
        store = ConversationStore(db_path=temp_db_path)
        yield store
        # Cleanup
        if temp_db_path.exists():
            temp_db_path.unlink()

    @pytest.mark.asyncio
    async def test_get_session_stats_with_memory_manager_enabled(
        self, memory_store: ConversationStore
    ):
        """Test get_session_stats when MemoryManager is enabled.

        Scenario:
        1. Create orchestrator with memory_manager
        2. Add messages to session
        3. Get session stats
        4. Verify accurate counts and metadata

        Expected:
        - Stats include message_count, total_tokens, available_tokens
        - Stats include role_distribution
        - Stats include tool_usage_count
        - Stats include timestamps (created_at, last_activity, duration_seconds)

        TODO: Implement this test after creating orchestrator fixture with MemoryManager enabled.
        Expected assertions:
        - assert stats["enabled"] is True
        - assert stats["message_count"] == 5
        - assert stats["total_tokens"] > 0
        - assert stats["role_distribution"]["user"] == 3
        - assert stats["role_distribution"]["assistant"] == 2
        - assert "created_at" in stats
        - assert "last_activity" in stats
        - assert "duration_seconds" in stats
        """
        # Create a session
        session = memory_store.create_session(project_path="/tmp/test_project")
        session_id = session.session_id

        # Add messages
        memory_store.add_message(session_id, MessageRole.USER, "Hello")
        memory_store.add_message(session_id, MessageRole.ASSISTANT, "Hi there!")
        memory_store.add_message(session_id, MessageRole.USER, "How are you?")
        memory_store.add_message(session_id, MessageRole.ASSISTANT, "I'm doing well!")
        memory_store.add_message(session_id, MessageRole.USER, "Great!")

        # Get session stats
        stats = memory_store.get_session_stats(session_id)

        # Verify stats (this is what the orchestrator should return)
        assert stats is not None, "Session stats should not be None"
        assert stats["session_id"] == session_id
        assert stats["message_count"] == 5
        assert stats["total_tokens"] >= 0
        assert stats["available_tokens"] >= 0
        assert stats["role_distribution"]["user"] == 3
        assert stats["role_distribution"]["assistant"] == 2
        assert "created_at" in stats
        assert "last_activity" in stats
        assert "duration_seconds" in stats

        # TODO: Create orchestrator with memory_manager and verify orchestrator.get_session_stats()
        # orchestrator = create_orchestrator_with_memory_manager(memory_store)
        # orchestrator_stats = orchestrator.get_session_stats()
        # assert orchestrator_stats["enabled"] is True
        # assert orchestrator_stats["message_count"] == stats["message_count"]

    @pytest.mark.asyncio
    async def test_get_session_stats_without_memory_manager(self):
        """Test get_session_stats when MemoryManager is disabled.

        Scenario:
        1. Create orchestrator without memory_manager
        2. Get session stats
        3. Verify fallback to message count

        Expected:
        - Stats show enabled: False
        - Stats include session_id: None
        - Stats include message_count from orchestrator.messages

        TODO: Implement this test after creating orchestrator fixture without MemoryManager.
        Expected assertions:
        - assert stats["enabled"] is False
        - assert stats["session_id"] is None
        - assert stats["message_count"] == len(orchestrator.messages)
        """
        # Test with ConversationStore directly
        # When session doesn't exist, get_session_stats returns empty dict
        from victor.agent.conversation_memory import ConversationStore
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            store = ConversationStore(db_path=db_path)

            # Non-existent session should return empty dict
            stats = store.get_session_stats("nonexistent_session")
            assert stats == {}

        # TODO: Create orchestrator without memory_manager
        # orchestrator = create_orchestrator_without_memory_manager()
        # orchestrator.add_user_message("Test message")
        # orchestrator.add_assistant_message("Test response")
        # stats = orchestrator.get_session_stats()
        # assert stats["enabled"] is False
        # assert stats["session_id"] is None
        # assert stats["message_count"] == 2

    @pytest.mark.asyncio
    async def test_get_session_stats_with_invalid_session_id(self, memory_store: ConversationStore):
        """Test get_session_stats with non-existent session.

        Scenario:
        1. Create orchestrator with memory_manager
        2. Use invalid session_id
        3. Verify graceful error handling

        Expected:
        - Returns empty dict from ConversationStore.get_session_stats()
        - Orchestrator handles error gracefully
        - No exceptions raised

        TODO: Implement this test after creating orchestrator with MemoryManager.
        Expected assertions:
        - assert stats["enabled"] is True
        - assert stats["error"] == "Session not found"
        - assert "session_id" in stats
        """
        # Test that ConversationStore returns empty dict for invalid session
        stats = memory_store.get_session_stats("invalid_session_id")
        assert stats == {}, "Should return empty dict for non-existent session"

        # TODO: Test orchestrator's error handling when memory_manager returns empty dict
        # orchestrator = create_orchestrator_with_memory_manager(memory_store)
        # orchestrator._memory_session_id = "invalid_session_id"
        # stats = orchestrator.get_session_stats()
        # assert stats["enabled"] is True
        # assert stats["error"] == "Session not found"


# =============================================================================
# Test Class 4: TestFlushAnalyticsIntegration
# =============================================================================


@pytest.mark.integration
class TestFlushAnalyticsIntegration:
    """Test flush_analytics integration."""

    @pytest.fixture
    def mock_usage_analytics(self):
        """Create a mock UsageAnalytics instance."""
        analytics = MagicMock()
        analytics.flush = MagicMock()
        return analytics

    @pytest.fixture
    def mock_sequence_tracker(self):
        """Create a mock ToolSequenceTracker instance."""
        tracker = MagicMock()
        tracker.get_statistics = MagicMock(
            return_value={"unique_transitions": 42, "total_sequences": 10}
        )
        return tracker

    @pytest.fixture
    def evaluation_coordinator(
        self, mock_usage_analytics: MagicMock, mock_sequence_tracker: MagicMock
    ) -> EvaluationCoordinator:
        """Create an EvaluationCoordinator instance with mocked dependencies."""
        coordinator = EvaluationCoordinator(
            usage_analytics=mock_usage_analytics,
            sequence_tracker=mock_sequence_tracker,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "claude-3-5-sonnet",
            get_tool_calls_used_fn=lambda: 5,
            get_intelligent_integration_fn=lambda: None,
        )
        return coordinator

    @pytest.mark.asyncio
    async def test_flush_analytics_calls_evaluation_coordinator(
        self,
        evaluation_coordinator: EvaluationCoordinator,
        mock_usage_analytics: MagicMock,
        mock_sequence_tracker: MagicMock,
    ):
        """Test that flush_analytics calls EvaluationCoordinator.flush_analytics().

        Scenario:
        1. Create orchestrator with evaluation_coordinator
        2. Flush analytics
        3. Verify EvaluationCoordinator.flush_analytics() called
        4. Verify tool_cache flushed if present

        Expected:
        - evaluation_coordinator.flush_analytics() called
        - usage_analytics.flush() called
        - sequence_tracker.get_statistics() called
        - Returns dict with all components

        TODO: Implement this test after creating orchestrator fixture.
        Expected assertions:
        - mock_usage_analytics.flush.assert_called_once()
        - mock_sequence_tracker.get_statistics.assert_called_once()
        - results["usage_analytics"] is True
        - results["sequence_tracker"] is True
        """
        # Test coordinator-level flushing
        results = await evaluation_coordinator.flush_analytics()

        # Verify coordinator called its components
        mock_usage_analytics.flush.assert_called_once()
        mock_sequence_tracker.get_statistics.assert_called_once()

        # Verify results structure
        assert isinstance(results, dict)
        assert "usage_analytics" in results
        assert "sequence_tracker" in results
        assert "tool_cache" in results
        assert results["usage_analytics"] is True
        assert results["sequence_tracker"] is True

        # TODO: Test orchestrator-level flushing
        # orchestrator = create_orchestrator_with_coordinator(evaluation_coordinator)
        # orchestrator.tool_cache = MagicMock()
        # orchestrator.tool_cache.flush = MagicMock()
        # results = orchestrator.flush_analytics()
        # assert results["tool_cache"] is True

    @pytest.mark.asyncio
    async def test_flush_analytics_returns_success_dict(
        self, evaluation_coordinator: EvaluationCoordinator
    ):
        """Test that flush_analytics returns proper success dictionary.

        Scenario:
        1. Flush analytics
        2. Verify return type: Dict[str, bool]
        3. Verify keys: usage_analytics, sequence_tracker, tool_cache

        Expected:
        - Returns dict with 3 keys
        - All values are booleans
        - usage_analytics: True if UsageAnalytics.flush() succeeds
        - sequence_tracker: True if ToolSequenceTracker.get_statistics() succeeds
        - tool_cache: False (not managed by coordinator)

        TODO: Extend this test for orchestrator-level flushing.
        Expected assertions:
        - assert isinstance(results, dict)
        - assert len(results) == 3
        - assert all(isinstance(v, bool) for v in results.values())
        """
        results = await evaluation_coordinator.flush_analytics()

        # Verify return type
        assert isinstance(results, dict), "Results should be a dictionary"
        assert len(results) == 3, "Results should have 3 keys"

        # Verify expected keys
        expected_keys = {"usage_analytics", "sequence_tracker", "tool_cache"}
        assert set(results.keys()) == expected_keys, f"Expected keys {expected_keys}"

        # Verify all values are booleans
        assert all(isinstance(v, bool) for v in results.values()), "All values should be boolean"

        # TODO: Test orchestrator-level return value
        # orchestrator = create_orchestrator_with_coordinator(evaluation_coordinator)
        # results = orchestrator.flush_analytics()
        # assert "tool_cache" in results

    @pytest.mark.asyncio
    async def test_flush_analytics_handles_exporter_errors(self):
        """Test that flush_analytics handles exporter errors gracefully.

        Scenario:
        1. Create coordinator with failing exporter
        2. Flush analytics
        3. Verify errors aggregated, not raised

        Expected:
        - usage_analytics.flush() raises exception
        - Error caught and logged
        - results["usage_analytics"] is False
        - Other components still flushed
        - No exception propagated to caller

        TODO: Implement this test after creating failing UsageAnalytics mock.
        Expected assertions:
        - results["usage_analytics"] is False
        - results["sequence_tracker"] is True
        - assert not raises Exception
        """
        # Create failing analytics
        failing_analytics = MagicMock()
        failing_analytics.flush = MagicMock(side_effect=Exception("Export failed"))

        # Create working tracker
        working_tracker = MagicMock()
        working_tracker.get_statistics = MagicMock(return_value={"unique_transitions": 10})

        # Create coordinator with failing exporter
        coordinator = EvaluationCoordinator(
            usage_analytics=failing_analytics,
            sequence_tracker=working_tracker,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "test-model",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
        )

        # Flush analytics - should not raise
        results = await coordinator.flush_analytics()

        # Verify error handling
        assert results["usage_analytics"] is False, "Failed analytics should return False"
        assert results["sequence_tracker"] is True, "Working tracker should still succeed"

        # TODO: Test orchestrator-level error handling
        # orchestrator = create_orchestrator_with_coordinator(coordinator)
        # results = orchestrator.flush_analytics()
        # assert results["usage_analytics"] is False

    @pytest.mark.asyncio
    async def test_flush_analytics_with_no_exporters(self):
        """Test flush_analytics when no exporters configured.

        Scenario:
        1. Create coordinator with no exporters
        2. Flush analytics
        3. Verify appropriate error/skip handling

        Expected:
        - usage_analytics is None
        - sequence_tracker is None
        - results["usage_analytics"] is False
        - results["sequence_tracker"] is False
        - results["tool_cache"] is False
        - No exceptions raised

        TODO: Implement this test and verify graceful handling.
        Expected assertions:
        - results["usage_analytics"] is False
        - results["sequence_tracker"] is False
        - results["tool_cache"] is False
        """
        # Create coordinator with no exporters
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "test-model",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
        )

        # Flush analytics
        results = await coordinator.flush_analytics()

        # Verify all components return False
        assert results["usage_analytics"] is False, "No analytics should return False"
        assert results["sequence_tracker"] is False, "No tracker should return False"
        assert results["tool_cache"] is False, "No tool cache should return False"

        # TODO: Test orchestrator-level handling with no components
        # orchestrator = create_orchestrator_with_no_analytics()
        # results = orchestrator.flush_analytics()
        # assert all(v is False for v in results.values())
