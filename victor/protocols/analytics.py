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

"""Analytics exporter protocol for dependency inversion.

This module defines the IAnalyticsExporter protocol that enables
dependency injection for analytics export, following the
Dependency Inversion Principle (DIP).

Design Principles:
    - DIP: AnalyticsCoordinator depends on this protocol, not concrete exporters
    - OCP: New analytics exporters can be added without modifying existing code
    - SRP: Each exporter handles one analytics destination

Usage:
    class PrometheusAnalyticsExporter(IAnalyticsExporter):
        async def export(self, data: Dict[str, Any]) -> ExportResult:
            # Export to Prometheus
            ...

        def exporter_type(self) -> str:
            return "prometheus"

    class DatadogAnalyticsExporter(IAnalyticsExporter):
        async def export(self, data: Dict[str, Any]) -> ExportResult:
            # Export to Datadog
            ...

        def exporter_type(self) -> str:
            return "datadog"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass
class ExportResult:
    """Result from analytics export operation.

    Attributes:
        success: Whether the export succeeded
        exporter_type: Type of exporter that performed the export
        records_exported: Number of records/data points exported
        error_message: Error message if export failed
        metadata: Additional metadata about the export
    """

    success: bool
    exporter_type: str
    records_exported: int
    error_message: str | None = None
    metadata: Dict[str, Any] | None = None


@runtime_checkable
class IAnalyticsExporter(Protocol):
    """Protocol for analytics data exporters.

    Implementations export analytics data to various destinations:
    - Prometheus: Metrics exposition
    - Datadog: Cloud monitoring
    - Custom APIs: Internal analytics systems
    - Files: CSV, JSON log files
    - Databases: SQL, NoSQL storage

    The AnalyticsCoordinator coordinates multiple exporters,
    sending data to all registered exporters for comprehensive
    analytics coverage.
    """

    async def export(self, data: Dict[str, Any]) -> ExportResult:
        """Export analytics data to external system.

        Args:
            data: Analytics data to export

        Returns:
            ExportResult with success status and metadata

        Example data structure:
            {
                "session_id": "abc123",
                "vertical": "coding",
                "metrics": {
                    "chat_requests": 10,
                    "tool_executions": 25,
                    "total_tokens": 5000,
                },
                "events": [
                    {"type": "tool_call", "tool": "read", "timestamp": "..."},
                    {"type": "llm_request", "model": "claude-sonnet-4", "timestamp": "..."},
                ],
            }

        Example:
            async def export(self, data: Dict[str, Any]) -> ExportResult:
                try:
                    # Export to Prometheus
                    for metric_name, value in data.get("metrics", {}).items():
                        self._prometheus_metric.labels(
                            vertical=data["vertical"]
                        ).set(value)
                    return ExportResult(
                        success=True,
                        exporter_type=self.exporter_type(),
                        records_exported=len(data.get("metrics", {})),
                    )
                except Exception as e:
                    return ExportResult(
                        success=False,
                        exporter_type=self.exporter_type(),
                        records_exported=0,
                        error_message=str(e),
                    )
        """
        ...

    def exporter_type(self) -> str:
        """Type identifier for this exporter.

        Used for logging and analytics to identify which
        exporter handled the data.

        Returns:
            Exporter type string (e.g., 'prometheus', 'datadog', 'file')

        Example:
            def exporter_type(self) -> str:
                return "prometheus"
        """
        ...


@dataclass
class AnalyticsEvent:
    """Analytics event data structure.

    Attributes:
        event_type: Type of event (e.g., 'tool_call', 'llm_request')
        timestamp: When the event occurred
        session_id: Session identifier
        data: Event-specific data
    """

    event_type: str
    timestamp: str
    session_id: str
    data: Dict[str, Any]


@dataclass
class AnalyticsQuery:
    """Query for analytics data retrieval.

    Attributes:
        session_id: Session to query (None = all sessions)
        event_types: Event types to include (None = all)
        start_time: Start of time range (None = unbounded)
        end_time: End of time range (None = now)
        limit: Maximum results to return
    """

    session_id: str | None = None
    event_types: List[str] | None = None
    start_time: str | None = None
    end_time: str | None = None
    limit: int = 1000


@dataclass
class AnalyticsResult:
    """Result from analytics query.

    Attributes:
        events: List of matching events
        total_count: Total number of matching events
        metadata: Query metadata
    """

    events: List[AnalyticsEvent]
    total_count: int
    metadata: Dict[str, Any] | None = None


__all__ = [
    "IAnalyticsExporter",
    "ExportResult",
    "AnalyticsEvent",
    "AnalyticsQuery",
    "AnalyticsResult",
]
