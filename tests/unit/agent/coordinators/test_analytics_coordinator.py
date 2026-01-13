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

"""Tests for AnalyticsCoordinator.

Tests the analytics collection and export coordination functionality.
"""

import pytest

from victor.agent.coordinators.analytics_coordinator import (
    AnalyticsCoordinator,
    SessionAnalytics,
    BaseAnalyticsExporter,
    ConsoleAnalyticsExporter,
)
from victor.protocols import IAnalyticsExporter, ExportResult, AnalyticsQuery


class MockAnalyticsExporter(BaseAnalyticsExporter):
    """Mock analytics exporter for testing."""

    def __init__(self, exporter_type="mock", should_fail=False):
        super().__init__(exporter_type)
        self._should_fail = should_fail
        self._export_calls = []

    async def export(self, data):
        # Record export call
        self._export_calls.append(data)

        if self._should_fail:
            raise ValueError("Intentional export error")

        return ExportResult(
            success=True,
            exporter_type=self._exporter_type,
            records_exported=len(data.get("events", [])),
        )


class FailingAnalyticsExporter(IAnalyticsExporter):
    """Analytics exporter that always fails."""

    async def export(self, data):
        raise RuntimeError("Intentional exporter error")

    def exporter_type(self):
        return "failing"


class TestBaseAnalyticsExporter:
    """Tests for BaseAnalyticsExporter."""

    def test_exporter_type(self):
        """Test exporter type getter."""
        exporter = BaseAnalyticsExporter(exporter_type="test")

        assert exporter.exporter_type() == "test"

    def test_export_raises_not_implemented(self):
        """Test export raises NotImplementedError."""
        exporter = BaseAnalyticsExporter(exporter_type="test")

        import asyncio
        with pytest.raises(NotImplementedError):
            asyncio.run(exporter.export({}))


class TestConsoleAnalyticsExporter:
    """Tests for ConsoleAnalyticsExporter."""

    def test_init_default(self):
        """Test initialization with defaults."""
        exporter = ConsoleAnalyticsExporter()

        assert exporter.exporter_type() == "console"
        assert exporter._verbose is False

    def test_init_verbose(self):
        """Test initialization with verbose=True."""
        exporter = ConsoleAnalyticsExporter(verbose=True)

        assert exporter._verbose is True

    @pytest.mark.asyncio
    async def test_export(self, capsys):
        """Test exporting to console."""
        exporter = ConsoleAnalyticsExporter()
        data = {
            "session_id": "test123",
            "events": [
                {"type": "tool_call", "timestamp": "2025-01-01T00:00:00"},
                {"type": "llm_request", "timestamp": "2025-01-01T00:01:00"},
            ],
        }

        result = await exporter.export(data)

        assert result.success is True
        assert result.records_exported == 2
        assert result.metadata["output"] == "console"

        # Check console output
        captured = capsys.readouterr()
        assert "Analytics Export: Session test123" in captured.out
        assert "Total events: 2" in captured.out

    @pytest.mark.asyncio
    async def test_export_verbose(self, capsys):
        """Test exporting with verbose=True."""
        exporter = ConsoleAnalyticsExporter(verbose=True)
        data = {
            "session_id": "test123",
            "events": [
                {
                    "type": "tool_call",
                    "timestamp": "2025-01-01T00:00:00",
                    "data": {"tool": "read"},
                }
            ],
        }

        await exporter.export(data)

        # Check console output includes event details
        captured = capsys.readouterr()
        assert "tool_call" in captured.out


class TestAnalyticsCoordinator:
    """Tests for AnalyticsCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create empty coordinator."""
        return AnalyticsCoordinator(exporters=[])

    @pytest.fixture
    def coordinator_with_exporters(self):
        """Create coordinator with mock exporters."""
        exporter1 = MockAnalyticsExporter(exporter_type="exporter1")
        exporter2 = MockAnalyticsExporter(exporter_type="exporter2")
        return AnalyticsCoordinator(exporters=[exporter1, exporter2])

    def test_init_empty(self):
        """Test initialization with no exporters."""
        coordinator = AnalyticsCoordinator(exporters=[])

        assert coordinator._exporters == []
        assert coordinator._enable_memory_storage is True
        assert coordinator._session_analytics == {}

    def test_init_with_exporters(self):
        """Test initialization with exporters."""
        exporter = MockAnalyticsExporter()
        coordinator = AnalyticsCoordinator(
            exporters=[exporter], enable_memory_storage=False
        )

        assert len(coordinator._exporters) == 1
        assert coordinator._enable_memory_storage is False

    @pytest.mark.asyncio
    async def test_track_event(self, coordinator):
        """Test tracking an event."""
        event = {
            "type": "tool_call",
            "timestamp": "2025-01-01T00:00:00",
            "data": {"tool": "read"},
        }

        await coordinator.track_event("session123", event)

        # Session analytics should be created
        assert "session123" in coordinator._session_analytics
        assert len(coordinator._session_analytics["session123"].events) == 1

    @pytest.mark.asyncio
    async def test_track_multiple_events(self, coordinator):
        """Test tracking multiple events."""
        events = [
            {"type": "tool_call", "timestamp": "2025-01-01T00:00:00"},
            {"type": "llm_request", "timestamp": "2025-01-01T00:01:00"},
        ]

        for event in events:
            await coordinator.track_event("session123", event)

        assert len(coordinator._session_analytics["session123"].events) == 2

    @pytest.mark.asyncio
    async def test_export_analytics_no_session(self, coordinator):
        """Test exporting analytics when session doesn't exist."""
        result = await coordinator.export_analytics("nonexistent")

        assert result.success is False
        assert result.records_exported == 0
        assert "No analytics found" in result.error_message

    @pytest.mark.asyncio
    async def test_export_analytics_no_exporters(self, coordinator):
        """Test exporting analytics when no exporters configured."""
        # Track an event first
        await coordinator.track_event("session123", {"type": "test"})

        result = await coordinator.export_analytics("session123")

        assert result.success is False
        assert "No exporters configured" in result.error_message

    @pytest.mark.asyncio
    async def test_export_analytics_success(self, coordinator_with_exporters):
        """Test successful analytics export."""
        # Track events
        await coordinator_with_exporters.track_event(
            "session123", {"type": "tool_call", "timestamp": "2025-01-01T00:00:00"}
        )
        await coordinator_with_exporters.track_event(
            "session123", {"type": "llm_request", "timestamp": "2025-01-01T00:01:00"}
        )

        result = await coordinator_with_exporters.export_analytics("session123")

        assert result.success is True
        assert result.records_exported == 2
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_export_analytics_specific_exporters(self, coordinator_with_exporters):
        """Test exporting to specific exporters only."""
        await coordinator_with_exporters.track_event(
            "session123", {"type": "test"}
        )

        # Export to first exporter only
        result = await coordinator_with_exporters.export_analytics(
            "session123", exporters=[coordinator_with_exporters._exporters[0]]
        )

        assert result.success is True
        assert len(result.metadata["exporters_used"]) == 1

    @pytest.mark.asyncio
    async def test_export_analytics_handles_errors(self):
        """Test that exporter errors are handled gracefully."""
        failing_exporter = FailingAnalyticsExporter()
        working_exporter = MockAnalyticsExporter(exporter_type="working")
        coordinator = AnalyticsCoordinator(exporters=[failing_exporter, working_exporter])

        await coordinator.track_event("session123", {"type": "test"})

        result = await coordinator.export_analytics("session123")

        # Should fail due to one exporter error
        assert result.success is False
        assert "failing" in result.error_message

    @pytest.mark.asyncio
    async def test_query_analytics_all(self, coordinator):
        """Test querying all analytics."""
        await coordinator.track_event("session1", {"type": "tool_call", "timestamp": "2025-01-01T00:00:00"})
        await coordinator.track_event("session2", {"type": "llm_request", "timestamp": "2025-01-01T00:01:00"})

        result = await coordinator.query_analytics({})

        assert len(result.events) == 2
        assert result.total_count == 2

    @pytest.mark.asyncio
    async def test_query_analytics_by_session(self, coordinator):
        """Test querying by session ID."""
        await coordinator.track_event("session1", {"type": "tool_call", "timestamp": "2025-01-01T00:00:00"})
        await coordinator.track_event("session2", {"type": "llm_request", "timestamp": "2025-01-01T00:01:00"})

        result = await coordinator.query_analytics({"session_id": "session1"})

        assert len(result.events) == 1
        assert result.events[0]["type"] == "tool_call"

    @pytest.mark.asyncio
    async def test_query_analytics_by_event_type(self, coordinator):
        """Test querying by event type."""
        await coordinator.track_event("session1", {"type": "tool_call", "timestamp": "2025-01-01T00:00:00"})
        await coordinator.track_event("session1", {"type": "llm_request", "timestamp": "2025-01-01T00:01:00"})

        result = await coordinator.query_analytics({"event_type": "tool_call"})

        assert len(result.events) == 1
        assert result.events[0]["type"] == "tool_call"

    @pytest.mark.asyncio
    async def test_query_analytics_with_limit(self, coordinator):
        """Test querying with limit."""
        for i in range(10):
            await coordinator.track_event("session1", {"type": f"event_{i}", "timestamp": "2025-01-01T00:00:00"})

        result = await coordinator.query_analytics({"limit": 5})

        assert len(result.events) == 5

    @pytest.mark.asyncio
    async def test_get_session_stats_not_found(self, coordinator):
        """Test getting stats for non-existent session."""
        stats = await coordinator.get_session_stats("nonexistent")

        assert stats["found"] is False
        assert stats["session_id"] == "nonexistent"

    @pytest.mark.asyncio
    async def test_get_session_stats(self, coordinator):
        """Test getting session statistics."""
        await coordinator.track_event("session123", {"type": "tool_call", "timestamp": "2025-01-01T00:00:00"})
        await coordinator.track_event("session123", {"type": "tool_call", "timestamp": "2025-01-01T00:01:00"})
        await coordinator.track_event("session123", {"type": "llm_request", "timestamp": "2025-01-01T00:02:00"})

        stats = await coordinator.get_session_stats("session123")

        assert stats["found"] is True
        assert stats["total_events"] == 3
        assert stats["event_counts"]["tool_call"] == 2
        assert stats["event_counts"]["llm_request"] == 1

    def test_add_exporter(self, coordinator):
        """Test adding an exporter."""
        exporter = MockAnalyticsExporter()
        coordinator.add_exporter(exporter)

        assert len(coordinator._exporters) == 1
        assert coordinator._exporters[0] == exporter

    def test_add_duplicate_exporter(self, coordinator):
        """Test that duplicate exporters are not added."""
        exporter = MockAnalyticsExporter()
        coordinator.add_exporter(exporter)
        coordinator.add_exporter(exporter)  # Add again

        assert len(coordinator._exporters) == 1

    def test_remove_exporter(self, coordinator):
        """Test removing an exporter."""
        exporter = MockAnalyticsExporter()
        coordinator.add_exporter(exporter)

        coordinator.remove_exporter(exporter)

        assert len(coordinator._exporters) == 0

    def test_remove_nonexistent_exporter(self, coordinator):
        """Test removing an exporter that doesn't exist."""
        exporter = MockAnalyticsExporter()

        # Should not raise
        coordinator.remove_exporter(exporter)

        assert len(coordinator._exporters) == 0

    def test_clear_session(self, coordinator_with_exporters):
        """Test clearing a session."""
        import asyncio
        asyncio.run(coordinator_with_exporters.track_event("session123", {"type": "test"}))

        coordinator_with_exporters.clear_session("session123")

        assert "session123" not in coordinator_with_exporters._session_analytics

    def test_clear_all_sessions(self, coordinator_with_exporters):
        """Test clearing all sessions."""
        import asyncio
        asyncio.run(coordinator_with_exporters.track_event("session1", {"type": "test"}))
        asyncio.run(coordinator_with_exporters.track_event("session2", {"type": "test"}))

        coordinator_with_exporters.clear_all_sessions()

        assert len(coordinator_with_exporters._session_analytics) == 0

    @pytest.mark.asyncio
    async def test_track_event_updates_timestamp(self, coordinator):
        """Test that tracking an event updates the session timestamp."""
        import time

        await coordinator.track_event("session123", {"type": "test"})
        first_timestamp = coordinator._session_analytics["session123"].updated_at

        # Wait a bit and track another event
        time.sleep(0.01)
        await coordinator.track_event("session123", {"type": "test2"})

        second_timestamp = coordinator._session_analytics["session123"].updated_at

        # Timestamps should be different
        assert first_timestamp != second_timestamp

    @pytest.mark.asyncio
    async def test_export_to_multiple_exporters(self):
        """Test that export reaches all exporters."""
        exporter1 = MockAnalyticsExporter(exporter_type="exporter1")
        exporter2 = MockAnalyticsExporter(exporter_type="exporter2")
        coordinator = AnalyticsCoordinator(exporters=[exporter1, exporter2])

        await coordinator.track_event("session123", {"type": "test", "timestamp": "2025-01-01T00:00:00"})

        result = await coordinator.export_analytics("session123")

        assert result.success is True
        assert len(exporter1._export_calls) == 1
        assert len(exporter2._export_calls) == 1

    @pytest.mark.asyncio
    async def test_query_analytics_filters_by_time(self, coordinator):
        """Test querying with time range filters."""
        await coordinator.track_event("session1", {"type": "test", "timestamp": "2025-01-01T00:00:00"})
        await coordinator.track_event("session1", {"type": "test", "timestamp": "2025-01-02T00:00:00"})
        await coordinator.track_event("session1", {"type": "test", "timestamp": "2025-01-03T00:00:00"})

        result = await coordinator.query_analytics({
            "start_time": "2025-01-02T00:00:00",
            "end_time": "2025-01-02T23:59:59",
        })

        assert len(result.events) == 1
        assert result.events[0]["timestamp"] == "2025-01-02T00:00:00"
