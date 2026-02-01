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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

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
    events: list[AnalyticsEvent] = field(default_factory=list)
    metadata: dict[str, Any] | None = None
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
        exporters: Optional[list[IAnalyticsExporter]] = None,
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
        self._session_analytics: dict[str, SessionAnalytics] = {}

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

        logger.debug(f"Tracked analytics event {event.event_type} for session {session_id}")

    async def export_analytics(
        self,
        session_id: str,
        exporters: Optional[list[IAnalyticsExporter]] = None,
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
                {"type": e.event_type, "data": e.data, "timestamp": e.timestamp}
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
            if query.session_id and session_id != query.session_id:
                continue

            # Filter events
            for event in session_analytics.events:
                # Filter by event_type if specified
                if query.event_types and event.event_type not in query.event_types:
                    continue

                # Filter by date range if specified
                event_timestamp = event.timestamp
                if query.start_time and event_timestamp < query.start_time:
                    continue
                if query.end_time and event_timestamp > query.end_time:
                    continue

                matching_events.append(event)

                # Apply limit if specified
                if query.limit and len(matching_events) >= query.limit:
                    break

            # Apply limit if specified
            if query.limit and len(matching_events) >= query.limit:
                break

        return AnalyticsResult(
            events=matching_events,
            total_count=len(matching_events),
            metadata={"query": query},
        )

    async def get_session_stats(
        self,
        session_id: str,
    ) -> dict[str, Any]:
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
        event_counts: dict[str, int] = {}
        for event in session_analytics.events:
            event_type = event.event_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "session_id": session_id,
            "found": True,
            "total_events": len(session_analytics.events),
            "event_counts": event_counts,
            "created_at": session_analytics.created_at,
            "updated_at": session_analytics.updated_at,
        }

    def get_optimization_status(
        self,
        context_compactor: Optional[Any] = None,
        usage_analytics: Optional[Any] = None,
        sequence_tracker: Optional[Any] = None,
        code_correction_middleware: Optional[Any] = None,
        safety_checker: Optional[Any] = None,
        auto_committer: Optional[Any] = None,
        search_router: Optional[Any] = None,
    ) -> dict[str, Any]:
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

        status: dict[str, Any] = {
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
                    "max_iterations": getattr(code_correction_middleware, "max_iterations", 1),
                }

        # Safety Checker
        status["components"]["safety_checker"] = {
            "enabled": safety_checker is not None,
            "has_confirmation_callback": (
                safety_checker.confirmation_callback is not None if safety_checker else False
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

    async def flush_analytics(
        self,
        evaluation_coordinator: Optional["EvaluationCoordinator"] = None,
        tool_cache: Optional[Any] = None,
    ) -> dict[str, bool]:
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
                flush_results = await evaluation_coordinator.flush_analytics()
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
        data: dict[str, Any],
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

    async def export(self, data: dict[str, Any]) -> ExportResult:
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

    async def export(self, data: dict[str, Any]) -> ExportResult:
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


class FileAnalyticsExporter(BaseAnalyticsExporter):
    """Analytics exporter that writes to files.

    This exporter writes analytics data to JSON or CSV files for
    persistent storage and offline analysis.

    Attributes:
        _file_path: Path to output file
        _format: Output format ('json' or 'csv')
        _append: Whether to append to existing file
    """

    def __init__(
        self,
        file_path: str,
        format: str = "json",
        append: bool = True,
    ):
        """Initialize the file exporter.

        Args:
            file_path: Path to output file
            format: Output format ('json' or 'csv')
            append: Whether to append to existing file (default: True)

        Raises:
            ValueError: If format is not 'json' or 'csv'
        """
        super().__init__("file")
        if format not in ("json", "csv"):
            raise ValueError(f"Invalid format: {format}. Must be 'json' or 'csv'")
        self._file_path = file_path
        self._format = format
        self._append = append

    async def export(self, data: dict[str, Any]) -> ExportResult:
        """Export analytics to file.

        Args:
            data: Analytics data to export

        Returns:
            ExportResult with export status
        """
        from pathlib import Path

        events = data.get("events", [])
        event_count = len(events)

        try:
            # Ensure directory exists
            file_path = Path(self._file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if self._format == "json":
                self._export_json(data, file_path)
            else:  # csv
                self._export_csv(data, file_path)

            return ExportResult(
                success=True,
                exporter_type=self._exporter_type,
                records_exported=event_count,
                metadata={
                    "file_path": str(file_path),
                    "format": self._format,
                },
            )

        except Exception as e:
            logger.error(f"File export failed: {e}")
            return ExportResult(
                success=False,
                exporter_type=self._exporter_type,
                records_exported=0,
                error_message=str(e),
            )

    def _export_json(self, data: dict[str, Any], file_path: Path) -> None:
        """Export data to JSON format.

        Args:
            data: Analytics data to export
            file_path: Path to output file
        """
        import json

        if self._append and file_path.exists():
            # Read existing data
            with open(file_path, "r") as f:
                try:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        existing_data.append(data)
                    elif isinstance(existing_data, dict):
                        existing_data = [existing_data, data]
                    else:
                        existing_data = [data]
                except json.JSONDecodeError:
                    existing_data = [data]
        else:
            existing_data = [data]

        # Write combined data
        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=2)

    def _export_csv(self, data: dict[str, Any], file_path: Path) -> None:
        """Export data to CSV format.

        Args:
            data: Analytics data to export
            file_path: Path to output file
        """
        import csv
        from io import StringIO

        events = data.get("events", [])

        # Prepare CSV data
        output = StringIO()
        if events:
            # Flatten event data for CSV
            fieldnames = ["session_id", "event_type", "timestamp"]
            if events:
                # Add all unique data keys
                data_keys = set()
                for event in events:
                    if event.get("data"):
                        data_keys.update(event["data"].keys())
                fieldnames.extend(sorted(data_keys))

            writer = csv.DictWriter(output, fieldnames=fieldnames)
            if not self._append or not file_path.exists():
                writer.writeheader()

            for event in events:
                row = {
                    "session_id": data.get("session_id", ""),
                    "event_type": event.get("type", ""),
                    "timestamp": event.get("timestamp", ""),
                }
                if event.get("data"):
                    row.update(event["data"])
                writer.writerow(row)

        # Write to file
        mode = "a" if self._append and file_path.exists() else "w"
        with open(file_path, mode, newline="") as f:
            f.write(output.getvalue())


__all__ = [
    "AnalyticsCoordinator",
    "SessionAnalytics",
    "BaseAnalyticsExporter",
    "ConsoleAnalyticsExporter",
    "FileAnalyticsExporter",
]
