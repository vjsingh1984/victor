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

"""Team execution analytics system.

This module provides comprehensive analytics for team execution including
performance tracking, bottleneck detection, comparison, and visualization.

Example:
    from victor.teams.team_analytics import TeamAnalytics

    analytics = TeamAnalytics()

    # Track execution
    analytics.track_execution(
        team_config=team_config,
        task="Implement authentication",
        result=execution_result
    )

    # Get insights
    insights = analytics.get_insights(team_id="auth_team")
    bottlenecks = analytics.detect_bottlenecks(team_id="auth_team")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from victor.teams.types import TeamConfig, TeamFormation

logger = logging.getLogger(__name__)


# =============================================================================
# Analytics Types
# =============================================================================


class MetricType(str, Enum):
    """Types of metrics tracked."""

    EXECUTION_TIME = "execution_time"
    SUCCESS_RATE = "success_rate"
    TOOL_CALLS = "tool_calls"
    QUALITY_SCORE = "quality_score"
    ITERATION_COUNT = "iteration_count"
    MEMBER_UTILIZATION = "member_utilization"
    FORMATION_EFFECTIVENESS = "formation_effectiveness"


@dataclass
class ExecutionEvent:
    """Event during team execution.

    Attributes:
        timestamp: When event occurred
        event_type: Type of event
        member_id: Member involved (if any)
        data: Event data
    """

    timestamp: datetime
    event_type: str
    member_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "member_id": self.member_id,
            "data": self.data,
        }


@dataclass
class ExecutionRecord:
    """Complete record of a team execution.

    Attributes:
        execution_id: Unique execution identifier
        team_id: Team identifier
        task: Task description
        team_config: Team configuration used
        start_time: Execution start time
        end_time: Execution end time
        success: Whether execution succeeded
        result: Execution result data
        events: List of events during execution
        member_results: Results from each member
        metrics: Computed metrics
    """

    execution_id: str
    team_id: str
    task: str
    team_config: "TeamConfig"
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    result: Dict[str, Any] = field(default_factory=dict)
    events: List[ExecutionEvent] = field(default_factory=list)
    member_results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "team_id": self.team_id,
            "task": self.task,
            "team_config": self.team_config.to_dict(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success": self.success,
            "result": self.result,
            "events": [e.to_dict() for e in self.events],
            "member_results": self.member_results,
            "metrics": self.metrics,
            "duration": self.duration,
        }


@dataclass
class BottleneckInfo:
    """Information about a performance bottleneck.

    Attributes:
        bottleneck_type: Type of bottleneck
        severity: Severity level (0.0-1.0)
        affected_members: Members affected
        description: Description of bottleneck
        suggested_fixes: Suggested fixes
    """

    bottleneck_type: str
    severity: float
    affected_members: List[str]
    description: str
    suggested_fixes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bottleneck_type": self.bottleneck_type,
            "severity": self.severity,
            "affected_members": self.affected_members,
            "description": self.description,
            "suggested_fixes": self.suggested_fixes,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two team configurations.

    Attributes:
        team1_id: First team ID
        team2_id: Second team ID
        metric_comparisons: Comparison of individual metrics
        overall_winner: Which team performed better overall
        confidence: Confidence in comparison (0.0-1.0)
        insights: Key insights from comparison
    """

    team1_id: str
    team2_id: str
    metric_comparisons: Dict[str, Dict[str, float]]
    overall_winner: Optional[str]
    confidence: float
    insights: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "team1_id": self.team1_id,
            "team2_id": self.team2_id,
            "metric_comparisons": self.metric_comparisons,
            "overall_winner": self.overall_winner,
            "confidence": self.confidence,
            "insights": self.insights,
        }


# =============================================================================
# Team Analytics
# =============================================================================


class TeamAnalytics:
    """Analytics system for team execution.

    Tracks team executions, computes metrics, detects bottlenecks,
    and provides insights for team optimization.

    Example:
        analytics = TeamAnalytics()

        # Track execution
        analytics.track_execution(
            team_config=team_config,
            task="Implement feature",
            result=execution_result,
            team_id="my_team"
        )

        # Get analytics
        stats = analytics.get_team_stats("my_team")
        bottlenecks = analytics.detect_bottlenecks("my_team")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize team analytics.

        Args:
            storage_path: Path to store analytics data
        """
        self.storage_path = storage_path

        # Data storage
        self._executions: Dict[str, ExecutionRecord] = {}
        self._team_executions: Dict[str, List[str]] = defaultdict(list)
        self._member_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(list))
        self._formation_stats: Dict[str, List[float]] = defaultdict(list)

        # Load existing data
        if storage_path and storage_path.exists():
            self.load_data(storage_path)

    def track_execution(
        self,
        team_config: "TeamConfig",
        task: str,
        result: Dict[str, Any],
        team_id: str = "default",
        events: Optional[List[ExecutionEvent]] = None,
    ) -> str:
        """Track a team execution.

        Args:
            team_config: Team configuration used
            task: Task description
            result: Execution result
            team_id: Team identifier
            events: List of events during execution

        Returns:
            Execution ID
        """
        import uuid

        execution_id = f"{team_id}_{uuid.uuid4().hex[:8]}"

        # Create execution record
        record = ExecutionRecord(
            execution_id=execution_id,
            team_id=team_id,
            task=task,
            team_config=team_config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=result.get("success", False),
            result=result,
            events=events or [],
            member_results=result.get("member_results", {}),
        )

        # Compute metrics
        record.metrics = self._compute_metrics(record)

        # Store
        self._executions[execution_id] = record
        self._team_executions[team_id].append(execution_id)

        # Update member stats
        for member_id, member_result in result.get("member_results", {}).items():
            self._member_stats[member_id]["execution_times"].append(
                member_result.get("duration_seconds", 0)
            )
            self._member_stats[member_id]["success"].append(member_result.get("success", False))
            self._member_stats[member_id]["tool_calls"].append(
                member_result.get("tool_calls_used", 0)
            )

        # Update formation stats
        formation = team_config.formation.value
        execution_time = result.get("total_duration", 0)
        self._formation_stats[formation].append(execution_time)

        return execution_id

    def _compute_metrics(self, record: ExecutionRecord) -> Dict[str, float]:
        """Compute metrics from execution record.

        Args:
            record: Execution record

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics["execution_time"] = record.duration
        metrics["success"] = float(record.success)
        metrics["total_tool_calls"] = record.result.get("total_tool_calls", 0)

        # Member-level metrics
        member_results = record.result.get("member_results", {})
        if member_results:
            member_times = [m.get("duration_seconds", 0) for m in member_results.values()]
            metrics["avg_member_time"] = np.mean(member_times) if member_times else 0
            metrics["max_member_time"] = max(member_times) if member_times else 0
            metrics["min_member_time"] = min(member_times) if member_times else 0

            member_success = [m.get("success", False) for m in member_results.values()]
            metrics["member_success_rate"] = np.mean(member_success) if member_success else 0

            member_tool_calls = [m.get("tool_calls_used", 0) for m in member_results.values()]
            metrics["avg_tool_calls"] = np.mean(member_tool_calls) if member_tool_calls else 0
            metrics["total_tool_calls"] = sum(member_tool_calls) if member_tool_calls else 0

        return metrics

    def get_team_stats(self, team_id: str) -> Dict[str, Any]:
        """Get statistics for a team.

        Args:
            team_id: Team identifier

        Returns:
            Dictionary with team statistics
        """
        execution_ids = self._team_executions.get(team_id, [])

        if not execution_ids:
            return {
                "team_id": team_id,
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
            }

        records = [self._executions[eid] for eid in execution_ids if eid in self._executions]

        # Compute stats
        success_count = sum(1 for r in records if r.success)
        success_rate = success_count / len(records) if records else 0

        execution_times = [r.duration for r in records]
        avg_execution_time = np.mean(execution_times) if execution_times else 0

        # Member stats
        member_performance = {}
        for record in records:
            for member_id, member_result in record.member_results.items():
                if member_id not in member_performance:
                    member_performance[member_id] = {
                        "executions": 0,
                        "successes": 0,
                        "total_time": 0,
                        "total_tool_calls": 0,
                    }

                member_performance[member_id]["executions"] += 1
                member_performance[member_id]["successes"] += member_result.get("success", 0)
                member_performance[member_id]["total_time"] += member_result.get(
                    "duration_seconds", 0
                )
                member_performance[member_id]["total_tool_calls"] += member_result.get(
                    "tool_calls_used", 0
                )

        # Compute member averages
        for member_id, stats in member_performance.items():
            stats["success_rate"] = int(
                stats["successes"] / stats["executions"] if stats["executions"] > 0 else 0
            )
            stats["avg_time"] = int(
                stats["total_time"] / stats["executions"] if stats["executions"] > 0 else 0
            )
            stats["avg_tool_calls"] = int(
                stats["total_tool_calls"] / stats["executions"] if stats["executions"] > 0 else 0
            )

        return {
            "team_id": team_id,
            "total_executions": len(records),
            "success_rate": round(success_rate, 3),
            "avg_execution_time": round(avg_execution_time, 2),
            "member_performance": member_performance,
        }

    def detect_bottlenecks(self, team_id: str) -> List[BottleneckInfo]:
        """Detect performance bottlenecks for a team.

        Args:
            team_id: Team identifier

        Returns:
            List of bottlenecks
        """
        execution_ids = self._team_executions.get(team_id, [])
        records = [self._executions[eid] for eid in execution_ids if eid in self._executions]

        if not records:
            return []

        bottlenecks = []

        # Check for slow members
        member_times = defaultdict(list)
        for record in records:
            for member_id, member_result in record.member_results.items():
                member_times[member_id].append(member_result.get("duration_seconds", 0))

        for member_id, times in member_times.items():
            avg_time = np.mean(times)
            overall_avg = np.mean([t for times_list in member_times.values() for t in times_list])

            if avg_time > overall_avg * 1.5:
                bottlenecks.append(
                    BottleneckInfo(
                        bottleneck_type="slow_member",
                        severity=min(1.0, (avg_time / overall_avg - 1.0)),
                        affected_members=[member_id],
                        description=f"Member {member_id} takes {avg_time:.2f}s on average, "
                        f"{(avg_time/overall_avg - 1.0)*100:.1f}% slower than team average",
                        suggested_fixes=[
                            "Increase member tool budget",
                            "Optimize member's assigned tasks",
                            "Consider parallelizing member's work",
                        ],
                    )
                )

        # Check for high failure rates
        member_failures = defaultdict(list)
        for record in records:
            for member_id, member_result in record.member_results.items():
                member_failures[member_id].append(not member_result.get("success", True))

        for member_id, failures in member_failures.items():
            failure_rate = np.mean(failures)
            if failure_rate > 0.3:
                bottlenecks.append(
                    BottleneckInfo(
                        bottleneck_type="high_failure_rate",
                        severity=float(failure_rate),
                        affected_members=[member_id],
                        description=f"Member {member_id} fails {failure_rate*100:.1f}% of the time",
                        suggested_fixes=[
                            "Review member's expertise alignment",
                            "Provide member with better context",
                            "Consider replacing member",
                        ],
                    )
                )

        # Check for tool budget issues
        for record in records:
            if (
                record.metrics.get("total_tool_calls", 0)
                >= record.team_config.total_tool_budget * 0.95
            ):
                bottlenecks.append(
                    BottleneckInfo(
                        bottleneck_type="tool_budget_exhaustion",
                        severity=0.8,
                        affected_members=[m.id for m in record.team_config.members],
                        description="Team frequently exhausts tool budget",
                        suggested_fixes=[
                            "Increase total tool budget",
                            "Optimize tool usage",
                            "Use more efficient tools",
                        ],
                    )
                )
                break

        return sorted(bottlenecks, key=lambda b: b.severity, reverse=True)

    def compare_teams(self, team1_id: str, team2_id: str) -> Optional[ComparisonResult]:
        """Compare two teams.

        Args:
            team1_id: First team ID
            team2_id: Second team ID

        Returns:
            Comparison result or None if insufficient data
        """
        stats1 = self.get_team_stats(team1_id)
        stats2 = self.get_team_stats(team2_id)

        if stats1["total_executions"] < 3 or stats2["total_executions"] < 3:
            return None

        # Compare metrics
        metric_comparisons = {}

        # Success rate
        metric_comparisons["success_rate"] = {
            "team1": stats1["success_rate"],
            "team2": stats2["success_rate"],
            "difference": stats1["success_rate"] - stats2["success_rate"],
            "winner": team1_id if stats1["success_rate"] > stats2["success_rate"] else team2_id,
        }

        # Execution time
        metric_comparisons["execution_time"] = {
            "team1": stats1["avg_execution_time"],
            "team2": stats2["avg_execution_time"],
            "difference": stats1["avg_execution_time"] - stats2["avg_execution_time"],
            "winner": (
                team1_id
                if stats1["avg_execution_time"] < stats2["avg_execution_time"]
                else team2_id
            ),
        }

        # Determine overall winner
        team1_wins = sum(1 for m in metric_comparisons.values() if m["winner"] == team1_id)
        team2_wins = sum(1 for m in metric_comparisons.values() if m["winner"] == team2_id)

        overall_winner = team1_id if team1_wins > team2_wins else team2_id
        confidence = min(1.0, (team1_wins + team2_wins) / (2 * len(metric_comparisons)))

        # Generate insights
        insights = []
        for metric_name, comparison in metric_comparisons.items():
            diff_pct = (
                abs(comparison["difference"]) / max(comparison["team1"], comparison["team2"]) * 100
                if max(comparison["team1"], comparison["team2"]) > 0
                else 0
            )
            if diff_pct > 20:
                insights.append(f"{metric_name}: {comparison['winner']} is {diff_pct:.1f}% better")

        return ComparisonResult(
            team1_id=team1_id,
            team2_id=team2_id,
            metric_comparisons=metric_comparisons,
            overall_winner=overall_winner,
            confidence=confidence,
            insights=insights,
        )

    def get_formation_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Get effectiveness metrics for each formation.

        Returns:
            Dictionary mapping formation to metrics
        """
        formation_metrics = {}

        for formation, times in self._formation_stats.items():
            if times:
                formation_metrics[formation] = {
                    "avg_time": round(np.mean(times), 2),
                    "min_time": round(np.min(times), 2),
                    "max_time": round(np.max(times), 2),
                    "std_time": round(np.std(times), 2),
                    "count": len(times),
                }

        return formation_metrics

    def get_member_ranking(self, team_id: Optional[str] = None) -> List[Tuple[str, float]]:
        """Rank members by performance.

        Args:
            team_id: Optional team ID to filter by

        Returns:
            List of (member_id, score) tuples sorted by score
        """
        member_scores: Dict[str, Dict[str, list]] = {}

        if team_id:
            execution_ids = self._team_executions.get(team_id, [])
            records = [self._executions[eid] for eid in execution_ids if eid in self._executions]
        else:
            records = list(self._executions.values())

        for record in records:
            for member_id, member_result in record.member_results.items():
                if member_id not in member_scores:
                    member_scores[member_id] = {"success": [], "time": [], "tool_calls": []}

                member_scores[member_id]["success"].append(member_result.get("success", False))
                member_scores[member_id]["time"].append(member_result.get("duration_seconds", 0))
                member_scores[member_id]["tool_calls"].append(
                    member_result.get("tool_calls_used", 0)
                )

        # Compute composite score
        rankings = []
        for member_id, stats in member_scores.items():
            if not stats["success"]:
                continue

            success_rate = float(np.mean(stats["success"]))
            avg_time = float(np.mean(stats["time"]))
            avg_tool_calls = float(np.mean(stats["tool_calls"]))

            # Composite score: prioritize success, then speed, then efficiency
            score = (
                success_rate * 0.6
                + (1 / max(1, avg_time)) * 0.25
                + (1 / max(1, avg_tool_calls)) * 0.15
            )

            rankings.append((member_id, score))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def save_data(self, path: Path) -> None:
        """Save analytics data to file.

        Args:
            path: Path to save data
        """
        try:
            data = {
                "executions": {eid: record.to_dict() for eid, record in self._executions.items()},
                "team_executions": dict(self._team_executions),
                "member_stats": dict(self._member_stats),
                "formation_stats": dict(self._formation_stats),
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved analytics data to {path}")
        except Exception as e:
            logger.error(f"Failed to save analytics data: {e}")

    def load_data(self, path: Path) -> None:
        """Load analytics data from file.

        Args:
            path: Path to load data from
        """
        try:
            with open(path, "r") as f:
                json.load(f)

            # Note: This is simplified - would need proper deserialization
            # for ExecutionRecord objects in production
            logger.info(f"Loaded analytics data from {path}")
        except Exception as e:
            logger.error(f"Failed to load analytics data: {e}")

    def export_report(self, team_id: str, output_path: Path) -> None:
        """Export analytics report for a team.

        Args:
            team_id: Team identifier
            output_path: Path to save report
        """
        stats = self.get_team_stats(team_id)
        bottlenecks = self.detect_bottlenecks(team_id)

        report = {
            "team_id": team_id,
            "generated_at": datetime.now().isoformat(),
            "statistics": stats,
            "bottlenecks": [b.to_dict() for b in bottlenecks],
            "recommendations": self._generate_recommendations(bottlenecks),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported report for {team_id} to {output_path}")

    def _generate_recommendations(self, bottlenecks: List[BottleneckInfo]) -> List[str]:
        """Generate recommendations from bottlenecks.

        Args:
            bottlenecks: List of bottlenecks

        Returns:
            List of recommendations
        """
        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck.severity > 0.7:
                recommendations.extend(bottleneck.suggested_fixes)

        return list(set(recommendations))  # Remove duplicates


__all__ = [
    "MetricType",
    "ExecutionEvent",
    "ExecutionRecord",
    "BottleneckInfo",
    "ComparisonResult",
    "TeamAnalytics",
]
