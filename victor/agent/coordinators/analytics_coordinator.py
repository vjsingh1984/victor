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

"""Analytics coordinator for collecting and exporting analytics.

This module implements the AnalyticsCoordinator which manages analytics
collection from multiple sources and export to multiple destinations.

Design Patterns:
    - Observer Pattern: Analytics exporters are notified of events
    - Strategy Pattern: Multiple exporters via IAnalyticsExporter
    - Repository Pattern: In-memory storage of analytics events
    - SRP: Focused only on analytics collection and export

Usage:
    from victor.agent.coordinators.analytics_coordinator import AnalyticsCoordinator
    from victor.protocols import IAnalyticsExporter, AnalyticsEvent

    # Create coordinator with exporters
    coordinator = AnalyticsCoordinator(exporters=[console_exporter, file_exporter])

    # Track event
    await coordinator.track_event(
        session_id="abc123",
        event=AnalyticsEvent(type="tool_call", data={"tool": "read"})
    )

    # Export analytics
    result = await coordinator.export_analytics(session_id="abc123")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.protocols import (
    IAnalyticsExporter,
    ExportResult,
    AnalyticsEvent,
    AnalyticsQuery,
    AnalyticsResult,
)

if TYPE_CHECKING:
    from victor.agent.coordinators.evaluation_coordinator import EvaluationCoordinator
    from victor.agent.coordinators.metrics_coordinator import MetricsCoordinator
    from victor.agent.memory.memory_manager import MemoryManager
    from victor.agent.stream_handler import StreamMetrics

logger = logging.getLogger(__name__)


@dataclass
class SessionAnalytics:
    """Analytics data for a single session.

    Attributes:
        session_id: Session identifier
        events: List of analytics events
        metadata: Additional session metadata
        created_at: When session analytics was created
        updated_at: When session analytics was last updated
    """

    session_id: str
    events: List[AnalyticsEvent] = field(default_factory=list)
    metadata: Dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AnalyticsCoordinator:
    """Analytics collection and export coordination.

    This coordinator manages analytics collection from various sources
    and exports to multiple destinations via IAnalyticsExporter implementations.

    Responsibilities:
    - Track analytics events per session
    - Query analytics data
    - Export analytics to multiple destinations
    - Aggregate analytics across sessions
    - Handle export errors gracefully

    Exporters are called in parallel, with results aggregated.
    """

    def __init__(
        self,
        exporters: Optional[List[IAnalyticsExporter]] = None,
        enable_memory_storage: bool = True,
        metrics_coordinator: Optional["MetricsCoordinator"] = None,
    ) -> None:
        """Initialize the analytics coordinator.

        Args:
            exporters: List of analytics exporters
            enable_memory_storage: Enable in-memory event storage
            metrics_coordinator: Optional MetricsCoordinator for bridging
        """
        self._exporters = exporters or []
        self._enable_memory_storage = enable_memory_storage
        self._metrics_coordinator = metrics_coordinator
        self._session_analytics: Dict[str, SessionAnalytics] = {}

    async def track_event(
        self,
        session_id: str,
        event: AnalyticsEvent,
    ) -> None:
        """Track an analytics event.

        Stores the event in session analytics and notifies exporters.

        Args:
            session_id: Session identifier
            event: Analytics event to track

        Example:
            await coordinator.track_event(
                session_id="abc123",
                event=AnalyticsEvent(
                    type="tool_call",
                    data={"tool": "read", "file_path": "/src/main.py"}
                )
            )
        """
        # Get or create session analytics
        if session_id not in self._session_analytics:
            self._session_analytics[session_id] = SessionAnalytics(session_id=session_id)

        # Add event to session
        self._session_analytics[session_id].events.append(event)
        self._session_analytics[session_id].updated_at = datetime.utcnow().isoformat()

        logger.debug(
            f"Tracked analytics event {event.get('type', 'unknown')} " f"for session {session_id}"
        )

    async def export_analytics(
        self,
        session_id: str,
        exporters: Optional[List[IAnalyticsExporter]] = None,
    ) -> ExportResult:
        """Export analytics for a session.

        Exports analytics data to all exporters (or specific exporters).

        Args:
            session_id: Session identifier
            exporters: Specific exporters to use (None = all)

        Returns:
            ExportResult with export status

        Example:
            result = await coordinator.export_analytics(session_id="abc123")
            if result.success:
                print(f"Exported {result.events_exported} events")
        """
        # Get session analytics
        if session_id not in self._session_analytics:
            return ExportResult(
                success=False,
                exporter_type="analytics_coordinator",
                records_exported=0,
                error_message=f"No analytics found for session {session_id}",
            )

        session_analytics = self._session_analytics[session_id]

        # Use specific exporters or all
        target_exporters = exporters or self._exporters

        if not target_exporters:
            return ExportResult(
                success=False,
                exporter_type="analytics_coordinator",
                records_exported=0,
                error_message="No exporters configured",
            )

        # Prepare export data
        export_data = {
            "session_id": session_id,
            "events": [
                {"type": e.get("type"), "data": e.get("data"), "timestamp": e.get("timestamp")}
                for e in session_analytics.events
            ],
            "metadata": session_analytics.metadata,
            "created_at": session_analytics.created_at,
            "updated_at": session_analytics.updated_at,
        }

        # Export to all exporters in parallel
        results = await asyncio.gather(
            *[self._export_to_destination(exporter, export_data) for exporter in target_exporters],
            return_exceptions=True,
        )

        # Check results
        errors = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"{target_exporters[i].exporter_type()}: {str(result)}")
            elif isinstance(result, ExportResult) and not result.success:
                errors.append(f"{target_exporters[i].exporter_type()}: {result.error_message}")

        return ExportResult(
            success=len(errors) == 0,
            exporter_type="analytics_coordinator",
            records_exported=len(session_analytics.events),
            error_message="; ".join(errors) if errors else None,
            metadata={
                "exporters_used": [e.exporter_type() for e in target_exporters],
                "failed_exporters": len(errors),
            },
        )

    async def query_analytics(
        self,
        query: AnalyticsQuery,
    ) -> AnalyticsResult:
        """Query analytics data.

        Queries analytics data based on query criteria.

        Args:
            query: Analytics query with filters

        Returns:
            AnalyticsResult with matching events

        Example:
            result = await coordinator.query_analytics(
                AnalyticsQuery({
                    "session_id": "abc123",
                    "event_type": "tool_call",
                    "limit": 100
                })
            )
        """
        # Filter events based on query
        matching_events = []

        for session_id, session_analytics in self._session_analytics.items():
            # Filter by session_id if specified
            if query.get("session_id") and session_id != query.get("session_id"):
                continue

            # Filter events
            for event in session_analytics.events:
                # Filter by event_type if specified
                if query.get("event_type") and event.get("type") != query.get("event_type"):
                    continue

                # Filter by date range if specified
                event_timestamp = event.get("timestamp")
                if query.get("start_time") and event_timestamp < query.get("start_time"):
                    continue
                if query.get("end_time") and event_timestamp > query.get("end_time"):
                    continue

                matching_events.append(event)

                # Apply limit if specified
                if query.get("limit") and len(matching_events) >= query.get("limit"):
                    break

            # Apply limit if specified
            if query.get("limit") and len(matching_events) >= query.get("limit"):
                break

        return AnalyticsResult(
            events=matching_events,
            total_count=len(matching_events),
            metadata={"query": query},
        )

    async def get_session_stats(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """Get statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session statistics

        Example:
            stats = await coordinator.get_session_stats(session_id="abc123")
            print(f"Total events: {stats['total_events']}")
        """
        if session_id not in self._session_analytics:
            return {
                "session_id": session_id,
                "found": False,
            }

        session_analytics = self._session_analytics[session_id]

        # Count events by type
        event_counts: Dict[str, int] = {}
        for event in session_analytics.events:
            event_type = event.get("type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "session_id": session_id,
            "found": True,
            "total_events": len(session_analytics.events),
            "event_counts": event_counts,
            "created_at": session_analytics.created_at,
            "updated_at": session_analytics.updated_at,
        }

    # === BRIDGE METHODS: Wrap MetricsCoordinator ===

    def finalize_stream_metrics(
        self, usage_data: Optional[Dict[str, int]] = None
    ) -> Optional["StreamMetrics"]:
        """Bridge to MetricsCoordinator.finalize_stream_metrics().

        Delegates to MetricsCoordinator if available, otherwise returns None.

        Args:
            usage_data: Optional usage data for finalization

        Returns:
            StreamMetrics if coordinator available, None otherwise
        """
        if not self._metrics_coordinator:
            return None
        return self._metrics_coordinator.finalize_stream_metrics(usage_data)

    def get_last_stream_metrics(self) -> Optional["StreamMetrics"]:
        """Bridge to MetricsCoordinator.get_last_stream_metrics().

        Returns:
            Last StreamMetrics if coordinator available, None otherwise
        """
        if not self._metrics_coordinator:
            return None
        return self._metrics_coordinator.get_last_stream_metrics()

    def get_streaming_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Bridge to MetricsCoordinator.get_streaming_metrics_summary().

        Returns:
            Metrics summary dict if coordinator available, None otherwise
        """
        if not self._metrics_coordinator:
            return None
        return self._metrics_coordinator.get_streaming_metrics_summary()

    def get_streaming_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Bridge to MetricsCoordinator.get_streaming_metrics_history().

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of metrics dicts if coordinator available, empty list otherwise
        """
        if not self._metrics_coordinator:
            return []
        return self._metrics_coordinator.get_streaming_metrics_history(limit)

    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Bridge to MetricsCoordinator.get_session_cost_summary().

        Returns:
            Session cost summary if coordinator available, empty dict otherwise
        """
        if not self._metrics_coordinator:
            return {}
        return self._metrics_coordinator.get_session_cost_summary()

    def get_session_cost(self) -> float:
        """Bridge to MetricsCoordinator.get_session_cost().

        Returns:
            Total session cost if coordinator available, 0.0 otherwise
        """
        if not self._metrics_coordinator:
            return 0.0
        return self._metrics_coordinator.get_session_cost()

    def get_token_usage(self) -> Dict[str, int]:
        """Bridge to MetricsCoordinator.get_token_usage().

        Returns:
            Token usage dict if coordinator available, empty dict otherwise
        """
        if not self._metrics_coordinator:
            return {}
        return self._metrics_coordinator.get_token_usage()

    def get_model_usage(self) -> Dict[str, int]:
        """Bridge to MetricsCoordinator.get_model_usage().

        Returns:
            Model usage dict if coordinator available, empty dict otherwise
        """
        if not self._metrics_coordinator:
            return {}
        return self._metrics_coordinator.get_model_usage()

    def get_provider_usage(self) -> Dict[str, int]:
        """Bridge to MetricsCoordinator.get_provider_usage().

        Returns:
            Provider usage dict if coordinator available, empty dict otherwise
        """
        if not self._metrics_coordinator:
            return {}
        return self._metrics_coordinator.get_provider_usage()

    def get_session_cost_formatted(self) -> str:
        """Bridge to MetricsCoordinator.get_session_cost_formatted().

        Returns:
            Formatted cost string if coordinator available, 'cost n/a' otherwise
        """
        if not self._metrics_coordinator:
            return "cost n/a"
        return self._metrics_coordinator.get_session_cost_formatted()

    def export_session_costs(
        self,
        path: str,
        format: str = "json",
    ) -> None:
        """Bridge to MetricsCoordinator.export_session_costs().

        Args:
            path: File path to export costs to
            format: Export format ('json' or 'csv')

        Raises:
            RuntimeError: If MetricsCoordinator not available
        """
        if not self._metrics_coordinator:
            raise RuntimeError("MetricsCoordinator not available")
        return self._metrics_coordinator.export_session_costs(path, format)

    def get_optimization_status(
        self,
        context_compactor: Optional[Any] = None,
        usage_analytics: Optional[Any] = None,
        sequence_tracker: Optional[Any] = None,
        code_correction_middleware: Optional[Any] = None,
        safety_checker: Optional[Any] = None,
        auto_committer: Optional[Any] = None,
        search_router: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Get comprehensive status of all integrated optimization components.

        Provides visibility into the health and statistics of all optimization
        components for debugging, monitoring, and observability.

        Args:
            context_compactor: Optional ContextCompactor instance
            usage_analytics: Optional UsageAnalytics instance
            sequence_tracker: Optional ToolSequenceTracker instance
            code_correction_middleware: Optional code correction middleware
            safety_checker: Optional SafetyChecker instance
            auto_committer: Optional AutoCommitter instance
            search_router: Optional SearchRouter instance

        Returns:
            Dictionary with component status and statistics:
            - context_compactor: Compaction stats, utilization, threshold
            - usage_analytics: Tool/provider metrics, session info
            - sequence_tracker: Pattern learning stats, suggestions
            - code_correction: Enabled status, correction stats
            - safety_checker: Enabled status, pattern counts
            - auto_committer: Enabled status, commit history
            - search_router: Routing stats, pattern matches
        """
        import time

        status: Dict[str, Any] = {
            "timestamp": time.time(),
            "components": {},
        }

        # Context Compactor
        if context_compactor:
            status["components"]["context_compactor"] = context_compactor.get_statistics()

        # Usage Analytics
        if usage_analytics:
            try:
                status["components"]["usage_analytics"] = {
                    "session_active": usage_analytics._current_session is not None,
                    "tool_records_count": len(usage_analytics._tool_records),
                    "provider_records_count": len(usage_analytics._provider_records),
                }
            except Exception:
                status["components"]["usage_analytics"] = {"status": "error"}

        # Sequence Tracker
        if sequence_tracker:
            try:
                status["components"]["sequence_tracker"] = sequence_tracker.get_statistics()
            except Exception:
                status["components"]["sequence_tracker"] = {"status": "error"}

        # Code Correction Middleware
        status["components"]["code_correction"] = {
            "enabled": code_correction_middleware is not None,
        }
        if code_correction_middleware:
            # Support both old-style (with config) and new vertical middleware (without config)
            if hasattr(code_correction_middleware, "config"):
                status["components"]["code_correction"]["config"] = {
                    "auto_fix": code_correction_middleware.config.auto_fix,
                    "max_iterations": code_correction_middleware.config.max_iterations,
                }
            else:
                # New vertical middleware - use get_config() if available or default values
                status["components"]["code_correction"]["config"] = {
                    "auto_fix": getattr(code_correction_middleware, "auto_fix", True),
                    "max_iterations": getattr(
                        code_correction_middleware, "max_iterations", 1
                    ),
                }

        # Safety Checker
        status["components"]["safety_checker"] = {
            "enabled": safety_checker is not None,
            "has_confirmation_callback": (
                safety_checker.confirmation_callback is not None
                if safety_checker
                else False
            ),
        }

        # Auto Committer
        status["components"]["auto_committer"] = {
            "enabled": auto_committer is not None,
        }
        if auto_committer:
            status["components"]["auto_committer"]["auto_commit"] = auto_committer.auto_commit

        # Search Router
        status["components"]["search_router"] = {
            "enabled": search_router is not None,
        }

        # Overall health
        enabled_count = sum(
            1
            for c in status["components"].values()
            if c.get("enabled", True) and c.get("status") != "error"
        )
        status["health"] = {
            "enabled_components": enabled_count,
            "total_components": len(status["components"]),
            "status": "healthy" if enabled_count >= 4 else "degraded",
        }

        return status

    def get_tool_usage_stats(
        self,
        conversation_state_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Bridge to MetricsCollector.get_tool_usage_stats().

        Access MetricsCollector through MetricsCoordinator.

        Args:
            conversation_state_summary: Optional conversation state summary

        Returns:
            Tool usage stats if coordinator available, empty dict otherwise
        """
        if not self._metrics_coordinator:
            return {}
        # Access MetricsCollector through MetricsCoordinator
        return self._metrics_coordinator.get_tool_usage_stats(
            conversation_state_summary=conversation_state_summary
        )

    def get_session_stats_ext(
        self,
        memory_manager: Optional["MemoryManager"] = None,
        session_id: Optional[str] = None,
        fallback_message_count: int = 0,
    ) -> Dict[str, Any]:
        """Get session stats, bridging to MemoryManager.

        This method provides a unified interface for session statistics,
        bridging to MemoryManager when available, with fallback handling.

        Args:
            memory_manager: Optional MemoryManager instance
            session_id: Optional session ID for MemoryManager lookup
            fallback_message_count: Fallback message count if no MemoryManager

        Returns:
            Session stats dict with keys:
            - enabled: bool (whether MemoryManager is active)
            - message_count: int (from MemoryManager or fallback)
            - session_id: str (if available)
            Plus any additional stats from MemoryManager
        """
        if not memory_manager or not session_id:
            return {
                "enabled": False,
                "message_count": fallback_message_count,
            }

        try:
            stats = memory_manager.get_session_stats(session_id)
            return stats
        except Exception as e:
            import logging

            logger_ext = logging.getLogger(__name__)
            logger_ext.warning(f"Failed to get session stats: {e}")
            return {
                "enabled": True,
                "message_count": fallback_message_count,
                "error": str(e),
            }

    def flush_analytics(
        self,
        evaluation_coordinator: Optional["EvaluationCoordinator"] = None,
        tool_cache: Optional[Any] = None,
    ) -> Dict[str, bool]:
        """Flush all analytics data to persistent storage.

        This method provides a unified flush interface that:
        1. Flushes EvaluationCoordinator analytics (UsageAnalytics, ToolSequenceTracker)
        2. Flushes tool_cache if available

        Args:
            evaluation_coordinator: Optional EvaluationCoordinator to flush
            tool_cache: Optional tool cache to flush

        Returns:
            Dictionary with success status for each component:
            - usage_analytics: bool (UsageAnalytics flushed successfully)
            - sequence_tracker: bool (ToolSequenceTracker flushed successfully)
            - tool_cache: bool (tool_cache flushed successfully, or True if no cache)
        """
        results = {
            "usage_analytics": False,
            "sequence_tracker": False,
            "tool_cache": True,  # Default to True if no cache
        }

        if evaluation_coordinator:
            try:
                flush_results = evaluation_coordinator.flush_analytics()
                results.update(flush_results)
            except Exception as e:
                import logging

                logger_flush = logging.getLogger(__name__)
                logger_flush.error(f"Failed to flush evaluation coordinator: {e}")

        if tool_cache:
            try:
                tool_cache.flush()
                results["tool_cache"] = True
            except Exception as e:
                import logging

                logger_flush = logging.getLogger(__name__)
                logger_flush.error(f"Failed to flush tool cache: {e}")
                results["tool_cache"] = False

        return results

    def add_exporter(
        self,
        exporter: IAnalyticsExporter,
    ) -> None:
        """Add an analytics exporter.

        Args:
            exporter: Analytics exporter to add

        Example:
            exporter = FileAnalyticsExporter("/path/to/analytics.json")
            coordinator.add_exporter(exporter)
        """
        if exporter not in self._exporters:
            self._exporters.append(exporter)

    def remove_exporter(
        self,
        exporter: IAnalyticsExporter,
    ) -> None:
        """Remove an analytics exporter.

        Args:
            exporter: Analytics exporter to remove
        """
        if exporter in self._exporters:
            self._exporters.remove(exporter)

    def clear_session(self, session_id: str) -> None:
        """Clear analytics for a session.

        Args:
            session_id: Session to clear

        Example:
            coordinator.clear_session(session_id="abc123")
        """
        self._session_analytics.pop(session_id, None)

    def clear_all_sessions(self) -> None:
        """Clear analytics for all sessions."""
        self._session_analytics.clear()

    async def _export_to_destination(
        self,
        exporter: IAnalyticsExporter,
        data: Dict[str, Any],
    ) -> ExportResult:
        """Export to a specific destination.

        Args:
            exporter: Analytics exporter
            data: Data to export

        Returns:
            Export result
        """
        try:
            return await exporter.export(data)
        except Exception as e:
            logger.error(f"Export failed for {exporter.exporter_type()}: {e}")
            return ExportResult(
                success=False,
                exporter_type="analytics_coordinator",
                records_exported=0,
                error_message=str(e),
            )


# Built-in exporters


class BaseAnalyticsExporter(IAnalyticsExporter):
    """Base class for analytics exporters.

    Provides default implementation for IAnalyticsExporter protocol
    that subclasses can override.

    Attributes:
        _exporter_type: Exporter type identifier
    """

    def __init__(self, exporter_type: str):
        """Initialize the analytics exporter.

        Args:
            exporter_type: Exporter type identifier
        """
        self._exporter_type = exporter_type

    async def export(self, data: Dict[str, Any]) -> ExportResult:
        """Export analytics data.

        Subclasses must implement this method.

        Args:
            data: Analytics data to export

        Returns:
            ExportResult with export status
        """
        raise NotImplementedError("Subclasses must implement export()")

    def exporter_type(self) -> str:
        """Get exporter type."""
        return self._exporter_type


class ConsoleAnalyticsExporter(BaseAnalyticsExporter):
    """Analytics exporter that prints to console.

    This exporter prints analytics data to the console for debugging
    and development purposes.

    Attributes:
        _verbose: Whether to print full event details
    """

    def __init__(self, verbose: bool = False):
        """Initialize the console exporter.

        Args:
            verbose: Print full event details
        """
        super().__init__("console")
        self._verbose = verbose

    async def export(self, data: Dict[str, Any]) -> ExportResult:
        """Export analytics to console.

        Args:
            data: Analytics data to export

        Returns:
            ExportResult
        """
        session_id = data.get("session_id", "unknown")
        events = data.get("events", [])
        event_count = len(events)

        print(f"\n=== Analytics Export: Session {session_id} ===")
        print(f"Total events: {event_count}")

        if self._verbose:
            for event in events:
                print(f"  - {event.get('type', 'unknown')}: {event.get('timestamp', 'N/A')}")
                if event.get("data"):
                    print(f"    Data: {event['data']}")

        print("=== End Export ===\n")

        return ExportResult(
            success=True,
            exporter_type=self._exporter_type,
            records_exported=event_count,
            metadata={"output": "console"},
        )


__all__ = [
    "AnalyticsCoordinator",
    "SessionAnalytics",
    "BaseAnalyticsExporter",
    "ConsoleAnalyticsExporter",
]
