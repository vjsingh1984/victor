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

"""Team execution metrics collection and aggregation.

This module provides comprehensive metrics collection for team node execution
within workflows. It integrates with the existing metrics system while adding
team-specific observability for formation types, member performance, and
recursion depth tracking.

Key Features:
- Thread-safe metrics collection for concurrent team execution
- Team-level metrics: formation type, member count, execution time
- Member-level metrics: individual agent performance tracking
- Tool usage metrics: tools used per team member
- Recursion depth tracking: nested team/workflow execution monitoring
- Integration with MetricsRegistry for unified observability

Design Patterns:
- Registry Pattern: Metrics registration and lookup
- Observer Pattern: Subscribe to team events for metric updates
- Singleton Pattern: Shared metrics registry across team executions

Example:
    from victor.workflows.team_metrics import TeamMetrics, get_team_metrics_collector

    # Get singleton collector
    collector = get_team_metrics_collector()

    # Record team execution
    collector.record_team_start(
        team_id="review_team",
        formation="parallel",
        member_count=3,
        recursion_depth=1
    )

    # Record member completion
    collector.record_member_complete(
        team_id="review_team",
        member_id="security_reviewer",
        success=True,
        duration_seconds=5.2,
        tool_calls_used=8
    )

    # Record team completion
    collector.record_team_complete(
        team_id="review_team",
        success=True,
        duration_seconds=15.3
    )

    # Get metrics summary
    summary = collector.get_summary()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class MemberExecutionMetrics:
    """Metrics for a single team member execution.

    Attributes:
        member_id: Unique identifier for the team member
        role: Role of the member (e.g., "researcher", "executor")
        success: Whether execution succeeded
        duration_seconds: Execution duration in seconds
        tool_calls_used: Number of tool calls made
        tools_used: Set of tool names used
        error_message: Error message if execution failed
        start_time: Execution start timestamp
        end_time: Execution end timestamp
    """

    member_id: str
    role: str = "assistant"
    success: bool = True
    duration_seconds: float = 0.0
    tool_calls_used: int = 0
    tools_used: Set[str] = field(default_factory=set)
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "member_id": self.member_id,
            "role": self.role,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "tool_calls_used": self.tool_calls_used,
            "tools_used": list(self.tools_used),
            "error_message": self.error_message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass
class TeamExecutionMetrics:
    """Metrics for a complete team execution.

    Attributes:
        team_id: Unique identifier for the team
        formation: Team formation type (e.g., "parallel", "sequential")
        member_count: Number of team members
        recursion_depth: Recursion depth at execution time
        success: Whether team execution succeeded
        duration_seconds: Total execution duration in seconds
        member_metrics: Metrics for each team member
        total_tool_calls: Total tool calls across all members
        unique_tools_used: Set of unique tool names used
        start_time: Execution start timestamp
        end_time: Execution end timestamp
        consensus_achieved: Whether consensus was achieved (for consensus formation)
        consensus_rounds: Number of consensus rounds (if applicable)
    """

    team_id: str
    formation: str = "sequential"
    member_count: int = 0
    recursion_depth: int = 0
    success: bool = True
    duration_seconds: float = 0.0
    member_metrics: Dict[str, MemberExecutionMetrics] = field(default_factory=dict)
    total_tool_calls: int = 0
    unique_tools_used: Set[str] = field(default_factory=set)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    consensus_achieved: Optional[bool] = None
    consensus_rounds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "team_id": self.team_id,
            "formation": self.formation,
            "member_count": self.member_count,
            "recursion_depth": self.recursion_depth,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "member_metrics": {
                k: v.to_dict() for k, v in self.member_metrics.items()
            },
            "total_tool_calls": self.total_tool_calls,
            "unique_tools_used": list(self.unique_tools_used),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "consensus_achieved": self.consensus_achieved,
            "consensus_rounds": self.consensus_rounds,
        }


class MetricPriority(str, Enum):
    """Priority levels for metrics collection.

    Attributes:
        CRITICAL: Metrics that must always be collected (e.g., errors)
        HIGH: Important metrics for monitoring (e.g., execution time)
        MEDIUM: Useful metrics for optimization (e.g., tool usage)
        LOW: Optional metrics for debugging (e.g., detailed traces)
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Team Metrics Collector
# =============================================================================


class TeamMetricsCollector:
    """Collects and aggregates team execution metrics.

    This class provides thread-safe metrics collection for team node execution
    within workflows. It integrates with the MetricsRegistry for unified
    observability across the Victor framework.

    Attributes:
        _enabled: Whether metrics collection is enabled
        _priority_threshold: Minimum priority level to collect
        _team_metrics: Dictionary of team execution metrics
        _active_teams: Set of currently executing teams
        _lock: Thread lock for concurrent access
        _registry: MetricsRegistry instance

    Example:
        collector = TeamMetricsCollector()
        collector.record_team_start("my_team", "parallel", 3, 1)
        collector.record_member_complete("my_team", "member1", True, 5.0, 10)
        collector.record_team_complete("my_team", True, 15.0)

        summary = collector.get_summary()
        print(summary["total_teams_executed"])
    """

    _instance: Optional["TeamMetricsCollector"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        enabled: bool = True,
        priority_threshold: MetricPriority = MetricPriority.MEDIUM,
    ):
        """Initialize team metrics collector.

        Args:
            enabled: Whether metrics collection is enabled
            priority_threshold: Minimum priority level to collect
        """
        self._enabled = enabled
        self._priority_threshold = priority_threshold
        self._team_metrics: Dict[str, TeamExecutionMetrics] = {}
        self._active_teams: Set[str] = set()
        self._lock = threading.RLock()

        # Integration with MetricsRegistry
        try:
            from victor.observability.metrics import MetricsRegistry

            self._registry = MetricsRegistry.get_instance()
            self._setup_registry_metrics()
        except ImportError:
            logger.debug("MetricsRegistry not available, using standalone collection")
            self._registry = None

    def _setup_registry_metrics(self) -> None:
        """Setup metrics in the global registry."""
        if self._registry is None:
            return

        # Team execution counters
        self._registry.counter(
            "victor_teams_executed_total",
            "Total number of team executions",
        )

        self._registry.counter(
            "victor_teams_failed_total",
            "Total number of failed team executions",
        )

        # Team execution duration
        self._registry.histogram(
            "victor_teams_duration_seconds",
            "Team execution duration in seconds",
            buckets=(1, 5, 10, 30, 60, 120, 300, 600),
        )

        # Member metrics
        self._registry.histogram(
            "victor_teams_member_count",
            "Number of members per team",
            buckets=(1, 2, 3, 5, 10, 20),
        )

        # Recursion depth
        self._registry.gauge(
            "victor_teams_recursion_depth",
            "Current recursion depth for team execution",
        )

        # Tool usage
        self._registry.counter(
            "victor_teams_tool_calls_total",
            "Total tool calls made by teams",
        )

        # Formation type distribution
        for formation in ["sequential", "parallel", "pipeline", "hierarchical", "consensus"]:
            self._registry.counter(
                f"victor_teams_formation_{formation}_total",
                f"Total {formation} team executions",
                labels={"formation": formation},
            )

    @classmethod
    def get_instance(cls) -> "TeamMetricsCollector":
        """Get singleton instance of TeamMetricsCollector.

        Returns:
            Singleton TeamMetricsCollector instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_team_start(
        self,
        team_id: str,
        formation: str,
        member_count: int,
        recursion_depth: int,
    ) -> None:
        """Record the start of a team execution.

        Args:
            team_id: Unique identifier for the team
            formation: Team formation type
            member_count: Number of team members
            recursion_depth: Current recursion depth
        """
        if not self._enabled:
            return

        with self._lock:
            self._active_teams.add(team_id)
            self._team_metrics[team_id] = TeamExecutionMetrics(
                team_id=team_id,
                formation=formation,
                member_count=member_count,
                recursion_depth=recursion_depth,
                start_time=datetime.now(timezone.utc),
            )

            # Update registry metrics
            if self._registry:
                self._registry.counter("victor_teams_executed_total").increment()

                formation_counter = self._registry.counter(
                    f"victor_teams_formation_{formation}_total",
                    labels={"formation": formation},
                )
                formation_counter.increment()

                self._registry.gauge("victor_teams_recursion_depth").set(recursion_depth)

        logger.debug(
            f"Team '{team_id}' started: formation={formation}, "
            f"members={member_count}, depth={recursion_depth}"
        )

    def record_member_complete(
        self,
        team_id: str,
        member_id: str,
        success: bool,
        duration_seconds: float,
        tool_calls_used: int,
        tools_used: Optional[Set[str]] = None,
        error_message: Optional[str] = None,
        role: str = "assistant",
    ) -> None:
        """Record the completion of a team member execution.

        Args:
            team_id: Unique identifier for the team
            member_id: Unique identifier for the member
            success: Whether member execution succeeded
            duration_seconds: Execution duration in seconds
            tool_calls_used: Number of tool calls made
            tools_used: Set of tool names used
            error_message: Error message if execution failed
            role: Role of the member
        """
        if not self._enabled:
            return

        with self._lock:
            if team_id not in self._team_metrics:
                logger.warning(f"Team '{team_id}' not found in metrics")
                return

            team_metrics = self._team_metrics[team_id]

            member_metrics = MemberExecutionMetrics(
                member_id=member_id,
                role=role,
                success=success,
                duration_seconds=duration_seconds,
                tool_calls_used=tool_calls_used,
                tools_used=tools_used or set(),
                error_message=error_message,
                end_time=datetime.now(timezone.utc),
            )

            # Calculate start time from duration
            if member_metrics.end_time:
                member_metrics.start_time = datetime.fromtimestamp(
                    member_metrics.end_time.timestamp() - duration_seconds,
                    tz=timezone.utc,
                )

            team_metrics.member_metrics[member_id] = member_metrics
            team_metrics.total_tool_calls += tool_calls_used
            team_metrics.unique_tools_used.update(tools_used or set())

            # Update registry metrics
            if self._registry:
                self._registry.counter("victor_teams_tool_calls_total").increment(tool_calls_used)

        logger.debug(
            f"Member '{member_id}' completed: team={team_id}, "
            f"success={success}, duration={duration_seconds:.2f}s"
        )

    def record_team_complete(
        self,
        team_id: str,
        success: bool,
        duration_seconds: float,
        consensus_achieved: Optional[bool] = None,
        consensus_rounds: Optional[int] = None,
    ) -> None:
        """Record the completion of a team execution.

        Args:
            team_id: Unique identifier for the team
            success: Whether team execution succeeded
            duration_seconds: Total execution duration in seconds
            consensus_achieved: Whether consensus was achieved (consensus formation)
            consensus_rounds: Number of consensus rounds (if applicable)
        """
        if not self._enabled:
            return

        with self._lock:
            if team_id not in self._team_metrics:
                logger.warning(f"Team '{team_id}' not found in metrics")
                return

            team_metrics = self._team_metrics[team_id]
            team_metrics.success = success
            team_metrics.duration_seconds = duration_seconds
            team_metrics.end_time = datetime.now(timezone.utc)

            # Calculate start time from duration
            if team_metrics.end_time:
                team_metrics.start_time = datetime.fromtimestamp(
                    team_metrics.end_time.timestamp() - duration_seconds,
                    tz=timezone.utc,
                )

            team_metrics.consensus_achieved = consensus_achieved
            team_metrics.consensus_rounds = consensus_rounds

            self._active_teams.discard(team_id)

            # Update registry metrics
            if self._registry:
                self._registry.histogram("victor_teams_duration_seconds").observe(duration_seconds)

                if not success:
                    self._registry.counter("victor_teams_failed_total").increment()

        logger.debug(
            f"Team '{team_id}' completed: success={success}, "
            f"duration={duration_seconds:.2f}s"
        )

    def get_team_metrics(self, team_id: str) -> Optional[TeamExecutionMetrics]:
        """Get metrics for a specific team execution.

        Args:
            team_id: Unique identifier for the team

        Returns:
            TeamExecutionMetrics if found, None otherwise
        """
        with self._lock:
            return self._team_metrics.get(team_id)

    def get_active_teams(self) -> Set[str]:
        """Get set of currently executing team IDs.

        Returns:
            Set of active team IDs
        """
        with self._lock:
            return self._active_teams.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all team executions.

        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            total_teams = len(self._team_metrics)
            successful_teams = sum(1 for m in self._team_metrics.values() if m.success)
            failed_teams = total_teams - successful_teams
            active_teams = len(self._active_teams)

            if total_teams == 0:
                return {
                    "total_teams_executed": 0,
                    "successful_teams": 0,
                    "failed_teams": 0,
                    "active_teams": 0,
                    "success_rate": 0.0,
                    "average_duration_seconds": 0.0,
                    "average_member_count": 0.0,
                    "total_tool_calls": 0,
                    "formation_distribution": {},
                }

            durations = [m.duration_seconds for m in self._team_metrics.values()]
            member_counts = [m.member_count for m in self._team_metrics.values()]
            total_tool_calls = sum(m.total_tool_calls for m in self._team_metrics.values())

            formation_distribution: Dict[str, int] = defaultdict(int)
            for metrics in self._team_metrics.values():
                formation_distribution[metrics.formation] += 1

            return {
                "total_teams_executed": total_teams,
                "successful_teams": successful_teams,
                "failed_teams": failed_teams,
                "active_teams": active_teams,
                "success_rate": successful_teams / total_teams if total_teams > 0 else 0.0,
                "average_duration_seconds": sum(durations) / len(durations),
                "average_member_count": sum(member_counts) / len(member_counts),
                "total_tool_calls": total_tool_calls,
                "formation_distribution": dict(formation_distribution),
            }

    def get_formation_stats(self, formation: str) -> Dict[str, Any]:
        """Get statistics for a specific formation type.

        Args:
            formation: Formation type (e.g., "parallel", "sequential")

        Returns:
            Dictionary with formation-specific statistics
        """
        with self._lock:
            formation_teams = [
                m for m in self._team_metrics.values() if m.formation == formation
            ]

            if not formation_teams:
                return {
                    "formation": formation,
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "average_duration_seconds": 0.0,
                    "average_member_count": 0.0,
                }

            successful = sum(1 for m in formation_teams if m.success)
            durations = [m.duration_seconds for m in formation_teams]
            member_counts = [m.member_count for m in formation_teams]

            return {
                "formation": formation,
                "total_executions": len(formation_teams),
                "successful_executions": successful,
                "failed_executions": len(formation_teams) - successful,
                "average_duration_seconds": sum(durations) / len(durations),
                "average_member_count": sum(member_counts) / len(member_counts),
            }

    def get_recursion_depth_stats(self) -> Dict[str, Any]:
        """Get statistics about recursion depth.

        Returns:
            Dictionary with recursion depth statistics
        """
        with self._lock:
            if not self._team_metrics:
                return {
                    "max_depth_observed": 0,
                    "average_depth": 0.0,
                    "depth_distribution": {},
                }

            depths = [m.recursion_depth for m in self._team_metrics.values()]
            max_depth = max(depths)
            avg_depth = sum(depths) / len(depths)

            depth_distribution: Dict[int, int] = defaultdict(int)
            for depth in depths:
                depth_distribution[depth] += 1

            return {
                "max_depth_observed": max_depth,
                "average_depth": avg_depth,
                "depth_distribution": dict(depth_distribution),
            }

    def clear_metrics(self) -> None:
        """Clear all collected metrics.

        Useful for testing or resetting metrics between runs.
        """
        with self._lock:
            self._team_metrics.clear()
            self._active_teams.clear()

            if self._registry:
                self._registry.reset_all()

        logger.debug("Team metrics cleared")

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable metrics collection.

        Args:
            enabled: Whether to enable metrics collection
        """
        self._enabled = enabled
        logger.info(f"Team metrics collection {'enabled' if enabled else 'disabled'}")

    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled


# =============================================================================
# Convenience Functions
# =============================================================================


def get_team_metrics_collector() -> TeamMetricsCollector:
    """Get the singleton TeamMetricsCollector instance.

    This is the preferred way to access the team metrics collector.

    Returns:
        Singleton TeamMetricsCollector instance

    Example:
        collector = get_team_metrics_collector()
        summary = collector.get_summary()
    """
    return TeamMetricsCollector.get_instance()


def record_team_execution(
    team_id: str,
    formation: str,
    member_count: int,
    recursion_depth: int,
    duration_seconds: float,
    success: bool,
    member_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Record a complete team execution in one call.

    This is a convenience function that records all metrics for a team
    execution at once. Useful for simple use cases.

    Args:
        team_id: Unique identifier for the team
        formation: Team formation type
        member_count: Number of team members
        recursion_depth: Current recursion depth
        duration_seconds: Total execution duration in seconds
        success: Whether team execution succeeded
        member_results: Optional dictionary of member results

    Example:
        record_team_execution(
            team_id="review_team",
            formation="parallel",
            member_count=3,
            recursion_depth=1,
            duration_seconds=15.3,
            success=True,
            member_results={
                "member1": {"success": True, "duration": 5.0, "tool_calls": 10},
                "member2": {"success": True, "duration": 6.0, "tool_calls": 12},
                "member3": {"success": True, "duration": 7.0, "tool_calls": 8},
            }
        )
    """
    collector = get_team_metrics_collector()

    collector.record_team_start(team_id, formation, member_count, recursion_depth)

    if member_results:
        for member_id, result in member_results.items():
            collector.record_member_complete(
                team_id=team_id,
                member_id=member_id,
                success=result.get("success", True),
                duration_seconds=result.get("duration", 0.0),
                tool_calls_used=result.get("tool_calls", 0),
                tools_used=result.get("tools_used"),
                error_message=result.get("error"),
                role=result.get("role", "assistant"),
            )

    collector.record_team_complete(team_id, success, duration_seconds)


__all__ = [
    # Data classes
    "MemberExecutionMetrics",
    "TeamExecutionMetrics",
    "MetricPriority",
    # Collector
    "TeamMetricsCollector",
    # Convenience functions
    "get_team_metrics_collector",
    "record_team_execution",
]
