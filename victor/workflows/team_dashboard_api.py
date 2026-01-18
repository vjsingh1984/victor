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

"""REST API for team collaboration dashboard.

This module provides REST endpoints for querying team execution state,
communication history, shared context, and negotiation status. It complements
the WebSocket server by providing request/response access to dashboard data.

Key Features:
- List active and completed team executions
- Get execution details by ID
- Query member communication history
- Fetch shared context snapshots
- Get negotiation status
- Integration with TeamMetricsCollector

Architecture:
- FastAPI REST endpoints
- JSON response format
- Query parameters for filtering
- Pagination support for history
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    Query = None  # type: ignore

from victor.workflows.team_dashboard_server import (
    TeamDashboardServer,
    TeamExecutionState,
    get_dashboard_server,
)
from victor.workflows.team_metrics import TeamMetricsCollector
from victor.workflows.team_collaboration import (
    TeamCommunicationProtocol,
    SharedTeamContext,
    NegotiationFramework,
    CommunicationLog,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class TeamExecutionSummary(BaseModel):
    """Summary of a team execution."""

    execution_id: str
    team_id: str
    formation: str
    status: str = Field(..., description="Status: running, completed, failed")
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    member_count: int = 0
    recursion_depth: int = 0
    success: Optional[bool] = None
    consensus_achieved: Optional[bool] = None


class MemberStatusResponse(BaseModel):
    """Status of a team member."""

    member_id: str
    role: str
    status: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_seconds: float
    tool_calls_used: int
    tools_used: List[str]
    error_message: Optional[str]
    last_activity: float


class CommunicationLogResponse(BaseModel):
    """Communication log entry."""

    timestamp: float
    message_type: str
    sender_id: str
    recipient_id: Optional[str]
    content: str
    communication_type: str
    duration_ms: Optional[float]


class NegotiationStatusResponse(BaseModel):
    """Negotiation status."""

    success: bool
    rounds: int
    consensus_achieved: bool
    agreed_proposal: Optional[Dict[str, Any]]
    votes: Dict[str, Any]


class ExecutionDetailsResponse(BaseModel):
    """Detailed execution information."""

    execution_id: str
    team_id: str
    formation: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_seconds: float
    success: Optional[bool]
    recursion_depth: int
    consensus_achieved: Optional[bool]
    member_states: Dict[str, MemberStatusResponse]
    shared_context: Dict[str, Any]
    communication_logs: List[CommunicationLogResponse]
    negotiation_status: Optional[NegotiationStatusResponse]


class MetricsSummaryResponse(BaseModel):
    """Metrics summary."""

    total_teams_executed: int
    successful_teams: int
    failed_teams: int
    active_teams: int
    success_rate: float
    average_duration_seconds: float
    average_member_count: float
    total_tool_calls: int
    formation_distribution: Dict[str, int]


# =============================================================================
# API Router
# =============================================================================


class TeamDashboardAPI:
    """REST API for team collaboration dashboard.

    This class provides REST endpoints for querying team execution data.
    It integrates with the dashboard server and metrics collector.

    Attributes:
        _app: FastAPI application
        _dashboard_server: Dashboard server instance
        _metrics_collector: Metrics collector instance
    """

    def __init__(
        self,
        dashboard_server: Optional[TeamDashboardServer] = None,
        metrics_collector: Optional[TeamMetricsCollector] = None,
    ) -> None:
        """Initialize dashboard API.

        Args:
            dashboard_server: Dashboard server instance
            metrics_collector: Metrics collector instance
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for the dashboard API. "
                "Install it with: pip install victor-ai[api]"
            )

        self._dashboard_server = dashboard_server or get_dashboard_server()
        self._metrics_collector = metrics_collector or TeamMetricsCollector.get_instance()

        # Setup routes
        self._setup_routes()

        logger.info("Team Dashboard API initialized")

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self._dashboard_server.app.get(
            "/api/v1/executions",
            response_model=List[TeamExecutionSummary],
            tags=["executions"],
        )
        async def list_executions(
            status: Optional[str] = Query(
                None,
                description="Filter by status: running, completed, failed",
            ),
            formation: Optional[str] = Query(
                None,
                description="Filter by formation type",
            ),
            limit: int = Query(
                50,
                ge=1,
                le=500,
                description="Maximum number of results",
            ),
        ) -> List[TeamExecutionSummary]:
            """List all team executions.

            Returns a list of team executions with optional filtering by status
            and formation type.
            """
            states = self._dashboard_server.get_all_execution_states()

            summaries = []
            for state in states.values():
                # Determine status
                if state.end_time is None:
                    exec_status = "running"
                elif state.success is True:
                    exec_status = "completed"
                else:
                    exec_status = "failed"

                # Apply filters
                if status and exec_status != status:
                    continue
                if formation and state.formation != formation:
                    continue

                summary = TeamExecutionSummary(
                    execution_id=state.execution_id,
                    team_id=state.team_id,
                    formation=state.formation,
                    status=exec_status,
                    start_time=state.start_time.isoformat() if state.start_time else "",
                    end_time=state.end_time.isoformat() if state.end_time else None,
                    duration_seconds=state.duration_seconds,
                    member_count=len(state.member_states),
                    recursion_depth=state.recursion_depth,
                    success=state.success,
                    consensus_achieved=state.consensus_achieved,
                )
                summaries.append(summary)

            # Sort by start time (newest first) and limit
            summaries.sort(key=lambda s: s.start_time, reverse=True)
            return summaries[:limit]

        @self._dashboard_server.app.get(
            "/api/v1/executions/{execution_id}",
            response_model=ExecutionDetailsResponse,
            tags=["executions"],
        )
        async def get_execution_details(execution_id: str) -> ExecutionDetailsResponse:
            """Get detailed information about a team execution.

            Returns comprehensive execution details including member states,
            shared context, communication logs, and negotiation status.
            """
            state = self._dashboard_server.get_execution_state(execution_id)

            if not state:
                raise HTTPException(status_code=404, detail="Execution not found")

            # Convert member states
            member_states = {
                k: MemberStatusResponse(**v.to_dict()) for k, v in state.member_states.items()
            }

            # Convert communication logs
            communication_logs = [
                CommunicationLogResponse(**log.to_dict()) for log in state.communication_logs
            ]

            # Convert negotiation status
            negotiation_status = None
            if state.negotiation_status:
                result = state.negotiation_status
                negotiation_status = NegotiationStatusResponse(
                    success=result.success,
                    rounds=result.rounds,
                    consensus_achieved=result.consensus_achieved,
                    agreed_proposal=(
                        result.agreed_proposal.__dict__ if result.agreed_proposal else None
                    ),
                    votes=result.votes,
                )

            return ExecutionDetailsResponse(
                execution_id=state.execution_id,
                team_id=state.team_id,
                formation=state.formation,
                start_time=state.start_time.isoformat() if state.start_time else None,
                end_time=state.end_time.isoformat() if state.end_time else None,
                duration_seconds=state.duration_seconds,
                success=state.success,
                recursion_depth=state.recursion_depth,
                consensus_achieved=state.consensus_achieved,
                member_states=member_states,
                shared_context=state.shared_context,
                communication_logs=communication_logs,
                negotiation_status=negotiation_status,
            )

        @self._dashboard_server.app.get(
            "/api/v1/executions/{execution_id}/members",
            response_model=Dict[str, MemberStatusResponse],
            tags=["members"],
        )
        async def get_member_statuses(execution_id: str) -> Dict[str, MemberStatusResponse]:
            """Get status of all team members for an execution.

            Returns the current status of each team member including
            execution state, tool usage, and errors.
            """
            state = self._dashboard_server.get_execution_state(execution_id)

            if not state:
                raise HTTPException(status_code=404, detail="Execution not found")

            return {k: MemberStatusResponse(**v.to_dict()) for k, v in state.member_states.items()}

        @self._dashboard_server.app.get(
            "/api/v1/executions/{execution_id}/members/{member_id}",
            response_model=MemberStatusResponse,
            tags=["members"],
        )
        async def get_member_status(
            execution_id: str,
            member_id: str,
        ) -> MemberStatusResponse:
            """Get status of a specific team member.

            Returns detailed status for a single team member.
            """
            state = self._dashboard_server.get_execution_state(execution_id)

            if not state:
                raise HTTPException(status_code=404, detail="Execution not found")

            member_state = state.member_states.get(member_id)

            if not member_state:
                raise HTTPException(status_code=404, detail="Member not found")

            return MemberStatusResponse(**member_state.to_dict())

        @self._dashboard_server.app.get(
            "/api/v1/executions/{execution_id}/communications",
            response_model=List[CommunicationLogResponse],
            tags=["communications"],
        )
        async def get_communication_history(
            execution_id: str,
            limit: int = Query(
                100,
                ge=1,
                le=1000,
                description="Maximum number of logs to return",
            ),
            sender_id: Optional[str] = Query(
                None,
                description="Filter by sender ID",
            ),
            recipient_id: Optional[str] = Query(
                None,
                description="Filter by recipient ID",
            ),
        ) -> List[CommunicationLogResponse]:
            """Get communication history for a team execution.

            Returns the history of messages exchanged between team members.
            Can be filtered by sender and recipient IDs.
            """
            state = self._dashboard_server.get_execution_state(execution_id)

            if not state:
                raise HTTPException(status_code=404, detail="Execution not found")

            logs = state.communication_logs

            # Apply filters
            if sender_id:
                logs = [log for log in logs if log.sender_id == sender_id]
            if recipient_id:
                logs = [log for log in logs if log.recipient_id == recipient_id]

            # Sort by timestamp (most recent first) and limit
            logs = sorted(logs, key=lambda log: log.timestamp, reverse=True)
            logs = logs[:limit]

            return [CommunicationLogResponse(**log.to_dict()) for log in logs]

        @self._dashboard_server.app.get(
            "/api/v1/executions/{execution_id}/context",
            response_model=Dict[str, Any],
            tags=["context"],
        )
        async def get_shared_context(execution_id: str) -> Dict[str, Any]:
            """Get shared context snapshot for a team execution.

            Returns the current state of the shared context key-value store.
            """
            state = self._dashboard_server.get_execution_state(execution_id)

            if not state:
                raise HTTPException(status_code=404, detail="Execution not found")

            return state.shared_context

        @self._dashboard_server.app.get(
            "/api/v1/executions/{execution_id}/context/{key}",
            tags=["context"],
        )
        async def get_context_value(
            execution_id: str,
            key: str,
        ) -> Dict[str, Any]:
            """Get a specific value from the shared context.

            Returns the value for a specific key in the shared context.
            """
            state = self._dashboard_server.get_execution_state(execution_id)

            if not state:
                raise HTTPException(status_code=404, detail="Execution not found")

            if key not in state.shared_context:
                raise HTTPException(status_code=404, detail="Key not found")

            return {"key": key, "value": state.shared_context[key]}

        @self._dashboard_server.app.get(
            "/api/v1/executions/{execution_id}/negotiation",
            response_model=Optional[NegotiationStatusResponse],
            tags=["negotiation"],
        )
        async def get_negotiation_status(
            execution_id: str,
        ) -> Optional[NegotiationStatusResponse]:
            """Get negotiation status for a team execution.

            Returns information about ongoing or completed negotiations,
            including voting progress and consensus status.
            """
            state = self._dashboard_server.get_execution_state(execution_id)

            if not state:
                raise HTTPException(status_code=404, detail="Execution not found")

            if not state.negotiation_status:
                return None

            result = state.negotiation_status
            return NegotiationStatusResponse(
                success=result.success,
                rounds=result.rounds,
                consensus_achieved=result.consensus_achieved,
                agreed_proposal=result.agreed_proposal.__dict__ if result.agreed_proposal else None,
                votes=result.votes,
            )

        @self._dashboard_server.app.get(
            "/api/v1/metrics/summary",
            response_model=MetricsSummaryResponse,
            tags=["metrics"],
        )
        async def get_metrics_summary() -> MetricsSummaryResponse:
            """Get metrics summary for all team executions.

            Returns aggregated metrics including success rate, average duration,
            and formation distribution.
            """
            summary = self._metrics_collector.get_summary()

            return MetricsSummaryResponse(**summary)

        @self._dashboard_server.app.get(
            "/api/v1/metrics/formation/{formation}",
            tags=["metrics"],
        )
        async def get_formation_stats(formation: str) -> Dict[str, Any]:
            """Get statistics for a specific formation type.

            Returns execution statistics for teams with the specified formation.
            """
            stats = self._metrics_collector.get_formation_stats(formation)

            return stats

        @self._dashboard_server.app.get(
            "/api/v1/metrics/recursion",
            tags=["metrics"],
        )
        async def get_recursion_stats() -> Dict[str, Any]:
            """Get recursion depth statistics.

            Returns statistics about recursion depth across all executions.
            """
            stats = self._metrics_collector.get_recursion_depth_stats()

            return stats

        @self._dashboard_server.app.get(
            "/api/v1/health",
            tags=["health"],
        )
        async def health_check() -> Dict[str, Any]:
            """Health check endpoint.

            Returns the health status of the dashboard API.
            """
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tracked_executions": len(self._dashboard_server.get_all_execution_states()),
                "active_executions": len(self._dashboard_server.get_active_executions()),
            }


def create_dashboard_app(
    metrics_collector: Optional[TeamMetricsCollector] = None,
    cors_origins: Optional[List[str]] = None,
) -> FastAPI:
    """Create and configure the dashboard FastAPI application.

    This is the main entry point for creating the dashboard app.
    It initializes both the WebSocket server and REST API.

    Args:
        metrics_collector: Team metrics collector
        cors_origins: CORS allowed origins

    Returns:
        Configured FastAPI application

    Example:
        from victor.workflows.team_dashboard_api import create_dashboard_app

        app = create_dashboard_app()

        # Run with uvicorn
        # uvicorn victor.workflows.team_dashboard_api:app --host 0.0.0.0 --port 8000
    """
    # Create dashboard server
    dashboard_server = get_dashboard_server(
        metrics_collector=metrics_collector,
        cors_origins=cors_origins,
    )

    # Setup API routes
    TeamDashboardAPI(
        dashboard_server=dashboard_server,
        metrics_collector=metrics_collector,
    )

    return dashboard_server.app


# Create app instance for easy import
app = create_dashboard_app()


__all__ = [
    # Request/Response models
    "TeamExecutionSummary",
    "MemberStatusResponse",
    "CommunicationLogResponse",
    "NegotiationStatusResponse",
    "ExecutionDetailsResponse",
    "MetricsSummaryResponse",
    # API
    "TeamDashboardAPI",
    "create_dashboard_app",
    "app",
]
