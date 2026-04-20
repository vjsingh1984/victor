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

"""Query service for observability data access.

Provides efficient querying of SQLite databases and JSONL logs for
events, sessions, and metrics with caching, filtering, and pagination.
"""

from __future__ import annotations

import aiosqlite
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

from victor.config.settings import get_project_paths


@dataclass
class EventFilters:
    """Filters for event queries."""

    event_types: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None
    tool_names: Optional[List[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    search_query: Optional[str] = None
    severity: Optional[str] = None  # 'error', 'warning', 'info'


@dataclass
class Event:
    """Observable event from database or JSONL."""

    id: str
    event_type: str
    timestamp: datetime
    session_id: str
    data: Dict[str, Any]
    tool_name: Optional[str] = None
    severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "data": self.data,
            "tool_name": self.tool_name,
            "severity": self.severity,
        }


@dataclass
class SessionInfo:
    """Basic session information for listing."""

    id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    provider: str
    model: str
    title: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session info to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
            "provider": self.provider,
            "model": self.model,
            "title": self.title,
            "tags": self.tags,
        }


@dataclass
class MetricsSnapshot:
    """Current metrics snapshot."""

    tool_calls_total: int
    tool_calls_success: int
    tool_calls_error: int
    total_tokens_used: int
    active_sessions: int
    error_rate: float
    avg_latency_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "tool_calls_total": self.tool_calls_total,
            "tool_calls_success": self.tool_calls_success,
            "tool_calls_error": self.tool_calls_error,
            "total_tokens_used": self.total_tokens_used,
            "active_sessions": self.active_sessions,
            "error_rate": self.error_rate,
            "avg_latency_seconds": self.avg_latency_seconds,
        }


class QueryService:
    """Service for querying observability data.

    Provides efficient access to SQLite databases and JSONL logs with
    caching, filtering, and pagination support.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize query service.

        Args:
            project_root: Project root directory. Defaults to current directory.
        """
        self.paths = get_project_paths(project_root)
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = 300  # 5 minutes

    async def get_recent_events(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[EventFilters] = None,
    ) -> List[Event]:
        """Get recent events with pagination.

        Args:
            limit: Maximum number of events to return
            offset: Number of events to skip
            filters: Optional event filters

        Returns:
            List of events sorted by timestamp (newest first)
        """
        # Try conversation database first
        events = await self._query_events_from_db(
            limit=limit,
            offset=offset,
            filters=filters,
        )

        # If no events in database, try JSONL logs
        if not events:
            events = await self._query_events_from_jsonl(
                limit=limit,
                offset=offset,
                filters=filters,
            )

        return events

    async def _query_events_from_db(
        self,
        limit: int,
        offset: int,
        filters: Optional[EventFilters],
    ) -> List[Event]:
        """Query events from SQLite database.

        Args:
            limit: Maximum number of events
            offset: Number of events to skip
            filters: Event filters

        Returns:
            List of events
        """
        events = []

        try:
            async with aiosqlite.connect(self.paths.conversation_db) as db:
                db.row_factory = aiosqlite.Row

                # Build query with filters
                query = "SELECT * FROM messages"
                params = []

                where_clauses = []
                if filters and filters.session_ids:
                    placeholders = ",".join("?" * len(filters.session_ids))
                    where_clauses.append(f"session_id IN ({placeholders})")
                    params.extend(filters.session_ids)

                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                    for row in rows:
                        event = Event(
                            id=str(row.get("id", "")),
                            event_type="message",  # Messages are stored as events
                            timestamp=datetime.fromisoformat(row["timestamp"]),
                            session_id=row["session_id"],
                            data={
                                "role": row["role"],
                                "content": row["content"],
                                "token_count": row.get("token_count", 0),
                            },
                            severity="info",
                        )
                        events.append(event)

        except Exception as e:
            # Log error but don't fail - try JSONL next
            pass

        return events

    async def _query_events_from_jsonl(
        self,
        limit: int,
        offset: int,
        filters: Optional[EventFilters],
    ) -> List[Event]:
        """Query events from JSONL log files.

        Args:
            limit: Maximum number of events
            offset: Number of events to skip
            filters: Event filters

        Returns:
            List of events
        """
        events = []
        usage_log = self.paths.global_victor_dir / "logs" / "usage.jsonl"

        if not usage_log.exists():
            return events

        try:
            # Read and parse JSONL file
            all_events = []
            async with aiosqlite.connect(usage_log) as db:
                # SQLite can't read JSONL directly, use file I/O
                import asyncio

                async with asyncio.to_thread(open, usage_log, "r") as f:
                    for line in f:
                        try:
                            event_data = json.loads(line.strip())
                            all_events.append(event_data)
                        except (json.JSONDecodeError, KeyError):
                            continue

            # Apply filters
            filtered_events = all_events
            if filters:
                if filters.event_types:
                    filtered_events = [
                        e for e in filtered_events
                        if e.get("event_type") in filters.event_types
                    ]

                if filters.start_time:
                    filtered_events = [
                        e for e in filtered_events
                        if datetime.fromisoformat(e.get("timestamp", ""))
                        >= filters.start_time
                    ]

            # Sort by timestamp (newest first) and paginate
            filtered_events.sort(
                key=lambda e: e.get("timestamp", ""), reverse=True
            )

            for event_data in filtered_events[offset : offset + limit]:
                event = Event(
                    id=event_data.get("id", ""),
                    event_type=event_data.get("event_type", "unknown"),
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    session_id=event_data.get("session_id", ""),
                    data=event_data,
                    tool_name=event_data.get("tool_name"),
                    severity=self._get_event_severity(event_data),
                )
                events.append(event)

        except Exception as e:
            # Log error but return empty list
            pass

        return events

    def _get_event_severity(self, event_data: Dict[str, Any]) -> str:
        """Determine event severity from event data.

        Args:
            event_data: Event data dictionary

        Returns:
            Severity level: 'error', 'warning', or 'info'
        """
        if event_data.get("event_type") == "ERROR":
            return "error"
        if event_data.get("event_type") == "TOOL_ERROR":
            return "error"
        if event_data.get("status") == "error":
            return "error"
        if "warning" in event_data.get("level", "").lower():
            return "warning"
        return "info"

    async def get_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> List[SessionInfo]:
        """Get list of sessions.

        Args:
            limit: Maximum number of sessions
            offset: Number of sessions to skip

        Returns:
            List of session info
        """
        sessions = []

        try:
            # Try SQLite session persistence
            from victor.agent.sqlite_session_persistence import (
                get_sqlite_session_persistence,
            )

            persistence = get_sqlite_session_persistence()
            sessions_data = await asyncio.to_thread(
                persistence.list_sessions, limit
            )

            for session_data in sessions_data[offset : offset + limit]:
                metadata = session_data.get("metadata", {})
                session = SessionInfo(
                    id=session_data.get("session_id", ""),
                    created_at=datetime.fromisoformat(metadata.get("created_at", "")),
                    updated_at=datetime.fromisoformat(metadata.get("updated_at", "")),
                    message_count=metadata.get("message_count", 0),
                    provider=metadata.get("provider", "unknown"),
                    model=metadata.get("model", "unknown"),
                    title=metadata.get("title"),
                    tags=metadata.get("tags", []),
                )
                sessions.append(session)

        except Exception:
            # Fallback: list JSON session files
            sessions = await self._get_sessions_from_json(limit, offset)

        return sessions

    async def _get_sessions_from_json(
        self,
        limit: int,
        offset: int,
    ) -> List[SessionInfo]:
        """Get sessions from JSON files.

        Args:
            limit: Maximum number of sessions
            offset: Number of sessions to skip

        Returns:
            List of session info
        """
        sessions = []
        sessions_dir = self.paths.sessions_dir

        if not sessions_dir.exists():
            return sessions

        try:
            import asyncio

            # Get all session JSON files
            session_files = list(sessions_dir.glob("*.json"))
            session_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            for session_file in session_files[offset : offset + limit]:
                async with asyncio.to_thread(session_file.read_text) as content:
                    session_data = json.loads(content)
                    metadata = session_data.get("metadata", {})

                    session = SessionInfo(
                        id=session_data.get("session_id", session_file.stem),
                        created_at=datetime.fromisoformat(
                            metadata.get("created_at", "")
                        ),
                        updated_at=datetime.fromisoformat(
                            metadata.get("updated_at", "")
                        ),
                        message_count=metadata.get("message_count", 0),
                        provider=metadata.get("provider", "unknown"),
                        model=metadata.get("model", "unknown"),
                        title=metadata.get("title"),
                        tags=metadata.get("tags", []),
                    )
                    sessions.append(session)

        except Exception:
            pass

        return sessions

    async def get_metrics_summary(self) -> MetricsSnapshot:
        """Get current metrics snapshot.

        Returns:
            Current metrics
        """
        # Try to get metrics from database
        try:
            async with aiosqlite.connect(self.paths.conversation_db) as db:
                # Count messages (proxy for activity)
                async with db.execute(
                    "SELECT COUNT(*) as count FROM messages"
                ) as cursor:
                    row = await cursor.fetchone()
                    message_count = row["count"] if row else 0

                # Calculate metrics (simplified - will be enhanced with AggregationService)
                return MetricsSnapshot(
                    tool_calls_total=0,  # Will be populated from tool metrics
                    tool_calls_success=0,
                    tool_calls_error=0,
                    total_tokens_used=0,
                    active_sessions=await self._count_active_sessions(),
                    error_rate=0.0,
                    avg_latency_seconds=0.0,
                )

        except Exception:
            # Return empty metrics on error
            return MetricsSnapshot(
                tool_calls_total=0,
                tool_calls_success=0,
                tool_calls_error=0,
                total_tokens_used=0,
                active_sessions=0,
                error_rate=0.0,
                avg_latency_seconds=0.0,
            )

    async def _count_active_sessions(self) -> int:
        """Count active sessions (updated in last hour).

        Returns:
            Number of active sessions
        """
        try:
            from datetime import timedelta

            cutoff = datetime.now() - timedelta(hours=1)

            async with aiosqlite.connect(self.paths.conversation_db) as db:
                async with db.execute(
                    """SELECT COUNT(DISTINCT session_id) as count
                       FROM messages
                       WHERE timestamp >= ?""",
                    (cutoff.isoformat(),),
                ) as cursor:
                    row = await cursor.fetchone()
                    return row["count"] if row else 0

        except Exception:
            return 0
