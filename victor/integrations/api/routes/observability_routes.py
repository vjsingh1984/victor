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
# specific language governing permissions and
# limitations under the License.

"""Observability API routes for web UI.

Provides REST endpoints for querying events, sessions, and metrics.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

from victor.observability.query_service import (
    QueryService,
    EventFilters,
    Event,
    SessionInfo,
    MetricsSnapshot,
)


# Request/Response Models
class EventsResponse(BaseModel):
    """Response model for events endpoint."""

    events: List[dict] = Field(description="List of events")
    total: int = Field(description="Total count (for pagination)")
    limit: int = Field(description="Page size")
    offset: int = Field(description="Current offset")


class SessionsResponse(BaseModel):
    """Response model for sessions endpoint."""

    sessions: List[dict] = Field(description="List of sessions")
    total: int = Field(description="Total count")
    limit: int = Field(description="Page size")
    offset: int = Field(description="Current offset")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""

    metrics: dict = Field(description="Current metrics snapshot")
    timestamp: str = Field(description="Metrics timestamp")


# Create router
router = APIRouter(
    prefix="/obs",
    tags=["observability"],
)

# Initialize query service (will be injected by FastAPI dependency injection in production)
_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """Get or create query service instance.

    Returns:
        QueryService instance
    """
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service


# =============================================================================
# Event Endpoints
# =============================================================================


@router.get("/events/recent", response_model=EventsResponse)
async def get_recent_events(
    limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
    offset: int = Query(0, ge=0, description="Number of events to skip"),
    event_types: Optional[str] = Query(
        None, description="Comma-separated event types to filter"
    ),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    tool_name: Optional[str] = Query(None, description="Filter by tool name"),
    severity: Optional[str] = Query(
        None, description="Filter by severity (error, warning, info)"
    ),
) -> EventsResponse:
    """Get recent events with pagination.

    Returns the most recent events from the observability system,
    with optional filtering by event type, session, tool, or severity.
    """
    service = get_query_service()

    # Build filters
    filters = None
    if event_types or session_id or tool_name or severity:
        filters = EventFilters(
            event_types=event_types.split(",") if event_types else None,
            session_ids=[session_id] if session_id else None,
            tool_names=[tool_name] if tool_name else None,
            severity=severity,
        )

    # Query events
    events = await service.get_recent_events(
        limit=limit,
        offset=offset,
        filters=filters,
    )

    return EventsResponse(
        events=[e.to_dict() for e in events],
        total=len(events),  # Will be enhanced with true count in AggregationService
        limit=limit,
        offset=offset,
    )


# =============================================================================
# Session Endpoints
# =============================================================================


@router.get("/sessions", response_model=SessionsResponse)
async def get_sessions(
    limit: int = Query(20, ge=1, le=100, description="Maximum sessions to return"),
    offset: int = Query(0, ge=0, description="Number of sessions to skip"),
) -> SessionsResponse:
    """Get list of sessions.

    Returns all sessions with basic metadata, sorted by
    most recently updated.
    """
    service = get_query_service()

    sessions = await service.get_sessions(limit=limit, offset=offset)

    return SessionsResponse(
        sessions=[s.to_dict() for s in sessions],
        total=len(sessions),  # Will be enhanced with true count
        limit=limit,
        offset=offset,
    )


@router.get("/sessions/{session_id}")
async def get_session_details(
    session_id: str,
) -> dict:
    """Get detailed session information.

    Args:
        session_id: Session ID

    Returns:
        Session details with messages, metrics, etc.
    """
    # TODO: Implement in Phase 3
    raise HTTPException(status_code=501, detail="Not implemented yet - Phase 3")


# =============================================================================
# Metrics Endpoints
# =============================================================================


@router.get("/metrics/summary", response_model=MetricsResponse)
async def get_metrics_summary() -> MetricsResponse:
    """Get current metrics snapshot.

    Returns real-time metrics including tool calls, errors,
    token usage, and active sessions.
    """
    service = get_query_service()

    metrics = await service.get_metrics_summary()

    return MetricsResponse(
        metrics=metrics.to_dict(),
        timestamp=datetime.now().isoformat(),
    )


@router.get("/metrics/history")
async def get_metrics_history(
    time_window: str = Query("1h", description="Time window (e.g., 1h, 24h, 7d)"),
) -> dict:
    """Get historical metrics data.

    Args:
        time_window: Time window for metrics

    Returns:
        Time-series metrics data
    """
    # TODO: Implement in Phase 2 with AggregationService
    raise HTTPException(status_code=501, detail="Not implemented yet - Phase 2")


@router.get("/metrics/tools")
async def get_tool_metrics() -> dict:
    """Get tool usage statistics.

    Returns tool execution counts, success rates, and
    performance metrics.
    """
    # TODO: Implement in Phase 2 with AggregationService
    raise HTTPException(status_code=501, detail="Not implemented yet - Phase 2")


@router.get("/metrics/performance")
async def get_performance_metrics() -> dict:
    """Get performance metrics.

    Returns latency percentiles (p50, p95, p99), throughput,
    and resource utilization.
    """
    # TODO: Implement in Phase 2 with AggregationService
    raise HTTPException(status_code=501, detail="Not implemented yet - Phase 2")


# =============================================================================
# Trace Endpoints
# =============================================================================


@router.get("/traces")
async def list_traces() -> dict:
    """List execution traces.

    Returns available traces with metadata.
    """
    # TODO: Implement in Phase 4
    raise HTTPException(status_code=501, detail="Not implemented yet - Phase 4")


@router.get("/traces/{trace_id}")
async def get_trace_details(trace_id: str) -> dict:
    """Get detailed execution trace.

    Args:
        trace_id: Trace ID

    Returns:
        Execution trace with spans
    """
    # TODO: Implement in Phase 4
    raise HTTPException(status_code=501, detail="Not implemented yet - Phase 4")


# =============================================================================
# Export Endpoints
# =============================================================================


@router.get("/export/events")
async def export_events(
    format: str = Query("json", description="Export format (json, csv)"),
) -> dict:
    """Export events data.

    Args:
        format: Export format

    Returns:
        Exported data file
    """
    # TODO: Implement in Phase 5 with ExportService
    raise HTTPException(status_code=501, detail="Not implemented yet - Phase 5")
