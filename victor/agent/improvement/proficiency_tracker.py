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

"""Proficiency tracker for agent self-improvement.

This module tracks tool and task performance metrics over time, enabling
the agent to learn from experience and improve its decision-making.

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                    ProficiencyTracker                            │
│  ├─ Outcome recording                                            │
│  ├─ Metrics calculation (success rate, time, cost)              │
│  ├─ Trend analysis                                               │
│  └─ Improvement suggestions                                      │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Storage Layer                                 │
│  ├─ SQLite persistence (~/.victor/victor.db)                    │
│  └─ In-memory caching                                            │
└──────────────────────────────────────────────────────────────────┘

Key Components:
- ProficiencyTracker: Main tracking class
- ProficiencyScore: Score with metrics and trend
- TaskOutcome: Recorded outcome data
- Suggestion: Improvement recommendation
- ProficiencyMetrics: Aggregate metrics export

Usage:
    tracker = ProficiencyTracker()

    # Record outcomes
    tracker.record_outcome(
        task="code_review",
        tool="ast_analyzer",
        outcome=TaskOutcome(
            success=True,
            duration=1.5,
            cost=0.001,
            quality_score=0.9
        )
    )

    # Get proficiency
    score = tracker.get_proficiency("ast_analyzer")
    print(f"Success rate: {score.success_rate:.2%}")
    print(f"Trend: {score.trend}")

    # Get suggestions
    suggestions = tracker.get_improvement_suggestions(agent_id="agent-1")
"""

from __future__ import annotations

import logging
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


logger = logging.getLogger(__name__)


class TrendDirection(str, Enum):
    """Trend direction for metrics."""

    IMPROVING = "improving"
    """Metric is improving over time."""

    STABLE = "stable"
    """Metric is stable."""

    DECLINING = "declining"
    """Metric is declining over time."""

    UNKNOWN = "unknown"
    """Insufficient data to determine trend."""


@dataclass
class TaskOutcome:
    """Recorded outcome for a task-tool execution.

    Attributes:
        success: Whether the execution was successful
        duration: Execution duration in seconds
        cost: Estimated cost in USD
        errors: List of error messages (if any)
        quality_score: Quality score 0.0-1.0
        timestamp: ISO timestamp of outcome
        metadata: Additional metadata
    """

    success: bool
    duration: float
    cost: float
    errors: list[str] = field(default_factory=list)
    quality_score: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "success": 1 if self.success else 0,
            "duration": self.duration,
            "cost": self.cost,
            "errors": ",".join(self.errors) if self.errors else "",
            "quality_score": self.quality_score,
            "timestamp": self.timestamp,
            "metadata": str(self.metadata) if self.metadata else "",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskOutcome":
        """Create from dictionary."""
        errors_str = data.get("errors", "")
        errors = errors_str.split(",") if errors_str else []

        return cls(
            success=bool(data.get("success", 0)),
            duration=data.get("duration", 0.0),
            cost=data.get("cost", 0.0),
            errors=errors,
            quality_score=data.get("quality_score", 1.0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata={},
        )


@dataclass
class ProficiencyScore:
    """Proficiency score for a tool or task.

    Attributes:
        success_rate: Success rate 0.0-1.0
        avg_execution_time: Average execution time in seconds
        avg_cost: Average cost in USD
        total_executions: Total number of executions
        trend: Trend direction
        last_updated: ISO timestamp of last update
        quality_score: Average quality score 0.0-1.0
    """

    success_rate: float
    avg_execution_time: float
    avg_cost: float
    total_executions: int
    trend: TrendDirection
    last_updated: str
    quality_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success_rate": self.success_rate,
            "avg_execution_time": self.avg_execution_time,
            "avg_cost": self.avg_cost,
            "total_executions": self.total_executions,
            "trend": self.trend.value,
            "last_updated": self.last_updated,
            "quality_score": self.quality_score,
        }


@dataclass
class Suggestion:
    """Improvement suggestion.

    Attributes:
        tool: Tool name
        reason: Reason for suggestion
        expected_improvement: Expected improvement factor
        confidence: Confidence in suggestion 0.0-1.0
        priority: Priority level (high, medium, low)
    """

    tool: str
    reason: str
    expected_improvement: float
    confidence: float
    priority: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool": self.tool,
            "reason": self.reason,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "priority": self.priority,
        }


@dataclass
class ProficiencyMetrics:
    """Aggregate proficiency metrics.

    Attributes:
        total_tools: Total number of tools tracked
        total_tasks: Total number of task types tracked
        total_outcomes: Total outcomes recorded
        tool_scores: Tool proficiency scores
        task_success_rates: Task success rates
        top_performing_tools: Top N tools by success rate
        improvement_opportunities: Tools with low success rates
        timestamp: ISO timestamp of export
    """

    total_tools: int
    total_tasks: int
    total_outcomes: int
    tool_scores: dict[str, ProficiencyScore]
    task_success_rates: dict[str, float]
    top_performing_tools: list[tuple[str, float]]
    improvement_opportunities: list[tuple[str, float]]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tools": self.total_tools,
            "total_tasks": self.total_tasks,
            "total_outcomes": self.total_outcomes,
            "tool_scores": {k: v.to_dict() for k, v in self.tool_scores.items()},
            "task_success_rates": self.task_success_rates,
            "top_performing_tools": self.top_performing_tools,
            "improvement_opportunities": self.improvement_opportunities,
            "timestamp": self.timestamp,
        }


@dataclass
class ImprovementTrajectory:
    """Historical trajectory of proficiency improvement.

    Attributes:
        task_type: Task type name
        timestamp: ISO timestamp of snapshot
        success_rate: Success rate at this point
        avg_time: Average execution time
        avg_quality: Average quality score
        sample_count: Number of samples
        moving_avg_success: Moving average of success rate
        moving_avg_time: Moving average of execution time
        moving_avg_quality: Moving average of quality score
        trend: Current trend direction
    """

    task_type: str
    timestamp: str
    success_rate: float
    avg_time: float
    avg_quality: float
    sample_count: int
    moving_avg_success: float
    moving_avg_time: float
    moving_avg_quality: float
    trend: TrendDirection

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "timestamp": self.timestamp,
            "success_rate": self.success_rate,
            "avg_time": self.avg_time,
            "avg_quality": self.avg_quality,
            "sample_count": self.sample_count,
            "moving_avg_success": self.moving_avg_success,
            "moving_avg_time": self.moving_avg_time,
            "moving_avg_quality": self.moving_avg_quality,
            "trend": self.trend.value,
        }


@dataclass
class MovingAverageMetrics:
    """Moving average metrics for performance tracking.

    Attributes:
        window_size: Size of the moving average window
        success_rate_ma: Moving average of success rate
        execution_time_ma: Moving average of execution time
        quality_score_ma: Moving average of quality score
        cost_ma: Moving average of cost
        variance: Variance in the window
        std_dev: Standard deviation in the window
        min_value: Minimum value in the window
        max_value: Maximum value in the window
    """

    window_size: int
    success_rate_ma: float
    execution_time_ma: float
    quality_score_ma: float
    cost_ma: float
    variance: float
    std_dev: float
    min_value: float
    max_value: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "success_rate_ma": self.success_rate_ma,
            "execution_time_ma": self.execution_time_ma,
            "quality_score_ma": self.quality_score_ma,
            "cost_ma": self.cost_ma,
            "variance": self.variance,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


class ProficiencyTracker:
    """Track proficiency for tools and tasks.

    ProficiencyTracker records outcomes, calculates metrics, analyzes trends,
    and generates improvement suggestions for self-improvement.

    Attributes:
        db: SQLite database connection
        cache: In-memory cache of recent scores

    Example:
        tracker = ProficiencyTracker()

        # Record outcome
        tracker.record_outcome(
            task="code_review",
            tool="ast_analyzer",
            outcome=TaskOutcome(
                success=True,
                duration=1.5,
                cost=0.001,
                quality_score=0.9
            )
        )

        # Get proficiency
        score = tracker.get_proficiency("ast_analyzer")
        print(f"Success rate: {score.success_rate:.2%}")

        # Get suggestions
        suggestions = tracker.get_improvement_suggestions(agent_id="agent-1")
    """

    def __init__(self, db: Optional[sqlite3.Connection] = None, moving_avg_window: int = 20):
        """Initialize ProficiencyTracker.

        Args:
            db: Optional database connection. If not provided, uses default.
            moving_avg_window: Window size for moving averages (default: 20)
        """
        # Support both raw sqlite3.Connection and DatabaseManager
        if db is None:
            from victor.core.database import get_database

            db_manager = get_database()
            self.db = db_manager.get_connection()
        elif hasattr(db, "get_connection"):
            # DatabaseManager passed
            self.db = db.get_connection()
        else:
            # Raw connection passed
            self.db = db

        self._cache: dict[str, ProficiencyScore] = {}
        self._moving_avg_window = moving_avg_window
        self._moving_avg_cache: dict[str, deque[Any]] = {}
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        cursor = self.db.cursor()

        # Tool proficiency table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_proficiency (
                tool TEXT PRIMARY KEY,
                success_count INTEGER DEFAULT 0,
                total_count INTEGER DEFAULT 0,
                total_duration REAL DEFAULT 0.0,
                total_cost REAL DEFAULT 0.0,
                total_quality REAL DEFAULT 0.0,
                last_updated TEXT,
                trend TEXT DEFAULT 'unknown'
            )
        """
        )

        # Task outcomes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS task_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                tool TEXT NOT NULL,
                success INTEGER,
                duration REAL,
                cost REAL,
                errors TEXT,
                quality_score REAL,
                timestamp TEXT,
                metadata TEXT
            )
        """
        )

        # Task success rates table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS task_success_rates (
                task TEXT PRIMARY KEY,
                success_count INTEGER DEFAULT 0,
                total_count INTEGER DEFAULT 0,
                last_updated TEXT
            )
            """
        )

        # Improvement trajectory table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS improvement_trajectory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                success_rate REAL,
                avg_time REAL,
                avg_quality REAL,
                sample_count INTEGER,
                moving_avg_success REAL,
                moving_avg_time REAL,
                moving_avg_quality REAL,
                trend TEXT
            )
        """
        )

        # Create indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_outcomes_task
            ON task_outcomes(task)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_outcomes_tool
            ON task_outcomes(tool)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_outcomes_timestamp
            ON task_outcomes(timestamp)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_improvement_trajectory_task_type
            ON improvement_trajectory(task_type)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_improvement_trajectory_timestamp
            ON improvement_trajectory(timestamp)
        """
        )

        self.db.commit()

    def record_outcome(self, task: str, tool: str, outcome: TaskOutcome) -> None:
        """Record a task-tool execution outcome.

        Args:
            task: Task type (e.g., "code_review", "test_generation")
            tool: Tool name (e.g., "ast_analyzer", "test_generator")
            outcome: Outcome data

        Raises:
            sqlite3.Error: If database operation fails
        """
        cursor = self.db.cursor()

        try:
            # Record outcome
            outcome_dict = outcome.to_dict()
            cursor.execute(
                """
                INSERT INTO task_outcomes
                (task, tool, success, duration, cost, errors, quality_score, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    task,
                    tool,
                    outcome_dict["success"],
                    outcome_dict["duration"],
                    outcome_dict["cost"],
                    outcome_dict["errors"],
                    outcome_dict["quality_score"],
                    outcome_dict["timestamp"],
                    outcome_dict["metadata"],
                ),
            )

            # Update tool proficiency
            self._update_tool_proficiency(tool, outcome)

            # Update task success rate
            self._update_task_success_rate(task, outcome.success)

            self.db.commit()

            # Invalidate cache
            if tool in self._cache:
                del self._cache[tool]

            logger.debug(
                f"ProficiencyTracker: Recorded outcome for task={task}, tool={tool}, "
                f"success={outcome.success}"
            )

        except sqlite3.Error as e:
            logger.error(f"ProficiencyTracker: Failed to record outcome: {e}")
            raise

    def _update_tool_proficiency(self, tool: str, outcome: TaskOutcome) -> None:
        """Update tool proficiency metrics.

        Args:
            tool: Tool name
            outcome: Outcome data
        """
        cursor = self.db.cursor()

        # Check if tool exists
        cursor.execute("SELECT total_count FROM tool_proficiency WHERE tool = ?", (tool,))
        row = cursor.fetchone()

        if row:
            # Update existing
            cursor.execute(
                """
                UPDATE tool_proficiency
                SET success_count = success_count + ?,
                    total_count = total_count + 1,
                    total_duration = total_duration + ?,
                    total_cost = total_cost + ?,
                    total_quality = total_quality + ?,
                    last_updated = ?
                WHERE tool = ?
            """,
                (
                    1 if outcome.success else 0,
                    outcome.duration,
                    outcome.cost,
                    outcome.quality_score,
                    outcome.timestamp,
                    tool,
                ),
            )
        else:
            # Insert new
            cursor.execute(
                """
                INSERT INTO tool_proficiency
                (tool, success_count, total_count, total_duration, total_cost, total_quality, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    tool,
                    1 if outcome.success else 0,
                    1,
                    outcome.duration,
                    outcome.cost,
                    outcome.quality_score,
                    outcome.timestamp,
                ),
            )

        # Update trend
        self._update_trend(tool)

    def _update_task_success_rate(self, task: str, success: bool) -> None:
        """Update task success rate.

        Args:
            task: Task type
            success: Whether successful
        """
        cursor = self.db.cursor()

        cursor.execute("SELECT total_count FROM task_success_rates WHERE task = ?", (task,))
        row = cursor.fetchone()

        if row:
            cursor.execute(
                """
                UPDATE task_success_rates
                SET success_count = success_count + ?,
                    total_count = total_count + 1,
                    last_updated = ?
                WHERE task = ?
            """,
                (1 if success else 0, datetime.now().isoformat(), task),
            )
        else:
            cursor.execute(
                """
                INSERT INTO task_success_rates
                (task, success_count, total_count, last_updated)
                VALUES (?, ?, ?, ?)
            """,
                (task, 1 if success else 0, 1, datetime.now().isoformat()),
            )

    def _update_trend(self, tool: str) -> None:
        """Update trend direction for a tool.

        Args:
            tool: Tool name
        """
        cursor = self.db.cursor()

        # Get recent outcomes (last 20)
        cursor.execute(
            """
            SELECT success, quality_score
            FROM task_outcomes
            WHERE tool = ?
            ORDER BY timestamp DESC
            LIMIT 20
        """,
            (tool,),
        )
        rows = cursor.fetchall()

        if len(rows) < 5:
            trend = TrendDirection.UNKNOWN
        else:
            # Calculate weighted score (recent has more weight)
            scores = []
            for i, (success, quality) in enumerate(reversed(rows)):
                weight = (i + 1) / len(rows)  # Linear increasing weight
                score = (1.0 if success else 0.0) * 0.5 + quality * 0.5
                scores.append(score * weight)

            # Compare recent vs older
            mid = len(scores) // 2
            recent_avg = sum(scores[mid:]) / (len(scores) - mid)
            older_avg = sum(scores[:mid]) / mid

            if recent_avg > older_avg * 1.1:
                trend = TrendDirection.IMPROVING
            elif recent_avg < older_avg * 0.9:
                trend = TrendDirection.DECLINING
            else:
                trend = TrendDirection.STABLE

        cursor.execute("UPDATE tool_proficiency SET trend = ? WHERE tool = ?", (trend.value, tool))

    def get_proficiency(self, tool: str) -> Optional[ProficiencyScore]:
        """Get proficiency score for a tool.

        Args:
            tool: Tool name

        Returns:
            ProficiencyScore or None if no data
        """
        # Check cache
        if tool in self._cache:
            return self._cache[tool]

        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT success_count, total_count, total_duration, total_cost,
                   total_quality, last_updated, trend
            FROM tool_proficiency
            WHERE tool = ?
        """,
            (tool,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        (
            success_count,
            total_count,
            total_duration,
            total_cost,
            total_quality,
            last_updated,
            trend_str,
        ) = row

        score = ProficiencyScore(
            success_rate=success_count / total_count if total_count > 0 else 0.0,
            avg_execution_time=total_duration / total_count if total_count > 0 else 0.0,
            avg_cost=total_cost / total_count if total_count > 0 else 0.0,
            total_executions=total_count,
            trend=TrendDirection(trend_str) if trend_str else TrendDirection.UNKNOWN,
            last_updated=last_updated,
            quality_score=total_quality / total_count if total_count > 0 else 0.0,
        )

        # Cache it
        self._cache[tool] = score

        return score

    def get_task_success_rate(self, task: str) -> float:
        """Get success rate for a task type.

        Args:
            task: Task type

        Returns:
            Success rate 0.0-1.0, or 0.0 if no data
        """
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT success_count, total_count FROM task_success_rates WHERE task = ?",
            (task,),
        )
        row = cursor.fetchone()

        if not row or row[1] == 0:
            return 0.0

        success_rate = row[0] / row[1]
        return float(success_rate)

    def suggest_tool_for_task(self, task: str) -> Optional[str]:
        """Suggest optimal tool for a task.

        Args:
            task: Task type

        Returns:
            Tool name or None if no data
        """
        cursor = self.db.cursor()

        # Get all tools used for this task
        cursor.execute(
            """
            SELECT tool, COUNT(*) as count,
                   SUM(success) as success_count,
                   AVG(quality_score) as avg_quality
            FROM task_outcomes
            WHERE task = ?
            GROUP BY tool
            HAVING count >= 3
            ORDER BY success_count DESC, avg_quality DESC
            LIMIT 1
        """,
            (task,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        return str(row[0]) if row[0] is not None else None

    def get_improvement_suggestions(
        self, agent_id: str, min_executions: int = 10
    ) -> list[Suggestion]:
        """Generate improvement suggestions for an agent.

        Args:
            agent_id: Agent identifier
            min_executions: Minimum executions before suggesting

        Returns:
            List of suggestions
        """
        cursor = self.db.cursor()
        suggestions = []

        # Find underperforming tools
        cursor.execute(
            """
            SELECT tool, success_count, total_count, trend
            FROM tool_proficiency
            WHERE total_count >= ?
            ORDER BY (success_count * 1.0 / total_count) ASC
            LIMIT 5
        """,
            (min_executions,),
        )
        rows = cursor.fetchall()

        for tool, success_count, total_count, trend_str in rows:
            success_rate = success_count / total_count

            if success_rate < 0.5:
                priority = "high"
            elif success_rate < 0.7:
                priority = "medium"
            else:
                priority = "low"

            suggestions.append(
                Suggestion(
                    tool=tool,
                    reason=f"Low success rate ({success_rate:.1%}) with {total_count} executions",
                    expected_improvement=1.0 - success_rate,
                    confidence=min(total_count / 50.0, 1.0),
                    priority=priority,
                )
            )

        # Find declining tools
        cursor.execute(
            """
            SELECT tool, success_count, total_count
            FROM tool_proficiency
            WHERE trend = 'declining' AND total_count >= ?
            LIMIT 3
        """,
            (min_executions,),
        )
        rows = cursor.fetchall()

        for tool, success_count, total_count in rows:
            if not any(s.tool == tool for s in suggestions):
                suggestions.append(
                    Suggestion(
                        tool=tool,
                        reason="Performance declining over recent executions",
                        expected_improvement=0.2,
                        confidence=0.7,
                        priority="medium",
                    )
                )

        return suggestions

    def export_metrics(self, top_n: int = 10) -> ProficiencyMetrics:
        """Export aggregate proficiency metrics.

        Args:
            top_n: Number of top tools to include

        Returns:
            ProficiencyMetrics with all data
        """
        cursor = self.db.cursor()

        # Get totals
        cursor.execute("SELECT COUNT(DISTINCT tool) FROM task_outcomes")
        total_tools = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(DISTINCT task) FROM task_outcomes")
        total_tasks = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM task_outcomes")
        total_outcomes = cursor.fetchone()[0] or 0

        # Get tool scores
        cursor.execute("SELECT tool FROM tool_proficiency")
        tool_names = [row[0] for row in cursor.fetchall()]
        tool_scores = {}
        for tool in tool_names:
            score = self.get_proficiency(tool)
            if score:
                tool_scores[tool] = score

        # Get task success rates
        cursor.execute("SELECT task FROM task_success_rates")
        task_names = [row[0] for row in cursor.fetchall()]
        task_success_rates = {}
        for task in task_names:
            task_success_rates[task] = self.get_task_success_rate(task)

        # Get top performing tools
        cursor.execute(
            """
            SELECT tool, (success_count * 1.0 / total_count) as success_rate
            FROM tool_proficiency
            WHERE total_count >= 5
            ORDER BY success_rate DESC
            LIMIT ?
        """,
            (top_n,),
        )
        top_performing_tools = [(row[0], row[1]) for row in cursor.fetchall()]

        # Get improvement opportunities
        improvement_opportunities = [
            (s.tool, s.expected_improvement)
            for s in self.get_improvement_suggestions(agent_id="export", min_executions=5)
        ]

        return ProficiencyMetrics(
            total_tools=total_tools,
            total_tasks=total_tasks,
            total_outcomes=total_outcomes,
            tool_scores=tool_scores,
            task_success_rates=task_success_rates,
            top_performing_tools=top_performing_tools,
            improvement_opportunities=improvement_opportunities,
            timestamp=datetime.now().isoformat(),
        )

    def get_all_tools(self) -> list[str]:
        """Get all tracked tools.

        Returns:
            List of tool names
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT tool FROM tool_proficiency")
        return [row[0] for row in cursor.fetchall()]

    def get_all_tasks(self) -> list[str]:
        """Get all tracked task types.

        Returns:
            List of task names
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT task FROM task_success_rates")
        return [row[0] for row in cursor.fetchall()]

    def reset_tool(self, tool: str) -> None:
        """Reset metrics for a tool.

        Args:
            tool: Tool name
        """
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM tool_proficiency WHERE tool = ?", (tool,))
        self.db.commit()

        if tool in self._cache:
            del self._cache[tool]

    def reset_all(self) -> None:
        """Reset all metrics."""
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM tool_proficiency")
        cursor.execute("DELETE FROM task_outcomes")
        cursor.execute("DELETE FROM task_success_rates")
        self.db.commit()

        self._cache.clear()

    def get_moving_average_metrics(
        self, task_type: str, window_size: Optional[int] = None
    ) -> Optional[MovingAverageMetrics]:
        """Calculate moving average metrics for a task type.

        Args:
            task_type: Task type name
            window_size: Optional window size (uses default if not provided)

        Returns:
            MovingAverageMetrics or None if insufficient data
        """
        window = window_size or self._moving_avg_window
        cursor = self.db.cursor()

        # Get recent outcomes
        cursor.execute(
            """
            SELECT success, duration, cost, quality_score
            FROM task_outcomes
            WHERE task = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (task_type, window),
        )
        rows = cursor.fetchall()

        if len(rows) < 3:
            return None

        # Extract metrics
        successes = [1 if row[0] else 0 for row in rows]
        durations = [row[1] for row in rows]
        costs = [row[2] for row in rows]
        qualities = [row[3] for row in rows]

        # Calculate moving averages
        success_ma = sum(successes) / len(successes)
        time_ma = sum(durations) / len(durations)
        quality_ma = sum(qualities) / len(qualities)
        cost_ma = sum(costs) / len(costs)

        # Calculate variance and std dev (using success rate)
        mean = success_ma
        variance = sum((x - mean) ** 2 for x in successes) / len(successes)
        std_dev = variance**0.5

        return MovingAverageMetrics(
            window_size=window,
            success_rate_ma=success_ma,
            execution_time_ma=time_ma,
            quality_score_ma=quality_ma,
            cost_ma=cost_ma,
            variance=variance,
            std_dev=std_dev,
            min_value=min(successes),
            max_value=max(successes),
        )

    def compute_moving_average(self, values: list[float], window: int) -> list[float]:
        """Compute simple moving average.

        Args:
            values: List of values
            window: Window size

        Returns:
            List of moving averages
        """
        if len(values) < window:
            return []

        moving_avgs = []
        for i in range(len(values) - window + 1):
            window_values = values[i : i + window]
            moving_avgs.append(sum(window_values) / window)

        return moving_avgs

    def detect_trend_direction(self, values: list[float], threshold: float = 0.1) -> TrendDirection:
        """Detect trend direction from time series data.

        Args:
            values: List of values in chronological order
            threshold: Threshold for trend detection (default: 0.1 = 10%)

        Returns:
            TrendDirection
        """
        if len(values) < 5:
            return TrendDirection.UNKNOWN

        # Split into first half and second half
        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        # Calculate percentage change
        if avg_first == 0:
            return TrendDirection.UNKNOWN

        change = (avg_second - avg_first) / avg_first

        if change > threshold:
            return TrendDirection.IMPROVING
        elif change < -threshold:
            return TrendDirection.DECLINING
        else:
            return TrendDirection.STABLE

    def get_top_proficiencies(self, n: int = 10) -> list[tuple[str, ProficiencyScore]]:
        """Get top N proficiencies by success rate.

        Args:
            n: Number of top proficiencies to return

        Returns:
            List of (tool_name, ProficiencyScore) tuples
        """
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT tool
            FROM tool_proficiency
            WHERE total_count >= 5
            ORDER BY (success_count * 1.0 / total_count) DESC
            LIMIT ?
        """,
            (n,),
        )
        tools = [row[0] for row in cursor.fetchall()]

        results = []
        for tool in tools:
            score = self.get_proficiency(tool)
            if score:
                results.append((tool, score))

        return results

    def get_weaknesses(self, threshold: float = 0.7, min_executions: int = 5) -> list[str]:
        """Get tools with success rate below threshold.

        Args:
            threshold: Success rate threshold (default: 0.7)
            min_executions: Minimum executions required

        Returns:
            List of tool names
        """
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT tool, success_count, total_count
            FROM tool_proficiency
            WHERE total_count >= ?
            ORDER BY (success_count * 1.0 / total_count) ASC
        """,
            (min_executions,),
        )

        weaknesses = []
        for tool, success_count, total_count in cursor.fetchall():
            success_rate = success_count / total_count
            if success_rate < threshold:
                weaknesses.append(tool)

        return weaknesses

    def update_proficiency(self, task_type: str, delta: float) -> None:
        """Manually update proficiency score for a task type.

        This is useful for external adjustments or RL feedback.

        Args:
            task_type: Task type name
            delta: Delta to apply to success rate (clamped to 0.0-1.0)
        """
        cursor = self.db.cursor()

        # Get current metrics
        cursor.execute(
            "SELECT success_count, total_count FROM task_success_rates WHERE task = ?",
            (task_type,),
        )
        row = cursor.fetchone()

        if not row:
            return

        success_count, total_count = row
        current_rate = success_count / total_count if total_count > 0 else 0.0

        # Apply delta and clamp
        new_rate = max(0.0, min(1.0, current_rate + delta))

        # Update success count to match new rate
        new_success_count = int(new_rate * total_count)
        cursor.execute(
            """
            UPDATE task_success_rates
            SET success_count = ?, last_updated = ?
            WHERE task = ?
        """,
            (new_success_count, datetime.now().isoformat(), task_type),
        )
        self.db.commit()

        logger.info(f"Updated proficiency for {task_type}: {current_rate:.2%} -> {new_rate:.2%}")

    def get_improvement_trajectory(
        self, task_type: str, limit: int = 100
    ) -> list[ImprovementTrajectory]:
        """Get historical improvement trajectory for a task type.

        Args:
            task_type: Task type name
            limit: Maximum number of trajectory points

        Returns:
            List of ImprovementTrajectory objects
        """
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT task_type, timestamp, success_rate, avg_time, avg_quality,
                   sample_count, moving_avg_success, moving_avg_time,
                   moving_avg_quality, trend
            FROM improvement_trajectory
            WHERE task_type = ?
            ORDER BY timestamp ASC
            LIMIT ?
        """,
            (task_type, limit),
        )

        trajectory = []
        for row in cursor.fetchall():
            trajectory.append(
                ImprovementTrajectory(
                    task_type=row[0],
                    timestamp=row[1],
                    success_rate=row[2],
                    avg_time=row[3],
                    avg_quality=row[4],
                    sample_count=row[5],
                    moving_avg_success=row[6],
                    moving_avg_time=row[7],
                    moving_avg_quality=row[8],
                    trend=TrendDirection(row[9]) if row[9] else TrendDirection.UNKNOWN,
                )
            )

        return trajectory

    def record_trajectory_snapshot(self, task_type: str) -> None:
        """Record a trajectory snapshot for a task type.

        This should be called periodically to track improvement over time.

        Args:
            task_type: Task type name
        """
        # Get current metrics
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT success_count, total_count FROM task_success_rates WHERE task = ?",
            (task_type,),
        )
        row = cursor.fetchone()

        if not row or row[1] == 0:
            return

        success_rate = row[0] / row[1]

        # Get average time and quality
        cursor.execute(
            """
            SELECT AVG(duration), AVG(quality_score)
            FROM task_outcomes
            WHERE task = ?
        """,
            (task_type,),
        )
        time_row = cursor.fetchone()
        avg_time = time_row[0] or 0.0
        avg_quality = time_row[1] or 0.0

        # Get moving averages
        ma_metrics = self.get_moving_average_metrics(task_type)
        if not ma_metrics:
            return

        # Detect trend
        trajectory = self.get_improvement_trajectory(task_type, limit=20)
        trend = TrendDirection.UNKNOWN
        if len(trajectory) >= 5:
            recent_rates = [t.success_rate for t in trajectory[-10:]]
            trend = self.detect_trend_direction(recent_rates)

        # Record snapshot
        cursor.execute(
            """
            INSERT INTO improvement_trajectory
            (task_type, timestamp, success_rate, avg_time, avg_quality,
             sample_count, moving_avg_success, moving_avg_time,
             moving_avg_quality, trend)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task_type,
                datetime.now().isoformat(),
                success_rate,
                avg_time,
                avg_quality,
                row[1],
                ma_metrics.success_rate_ma,
                ma_metrics.execution_time_ma,
                ma_metrics.quality_score_ma,
                trend.value,
            ),
        )
        self.db.commit()

        logger.debug(f"Recorded trajectory snapshot for {task_type}")

    def export_training_data(self) -> "pd.DataFrame":
        """Export training data for RL models.

        Returns:
            pandas DataFrame with all historical outcomes suitable for RL training.
            Columns: task, tool, success, duration, cost, quality_score, timestamp
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for export_training_data(). " "Install with: pip install pandas"
            )

        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT task, tool, success, duration, cost, quality_score, timestamp
            FROM task_outcomes
            ORDER BY timestamp ASC
        """
        )

        rows = cursor.fetchall()
        df = pd.DataFrame(
            rows,
            columns=["task", "tool", "success", "duration", "cost", "quality_score", "timestamp"],
        )

        logger.info(f"Exported {len(df)} training samples")
        return df

    def export_proficiency_history(self, task_type: str) -> "pd.DataFrame":
        """Export proficiency history for a specific task type.

        Args:
            task_type: Task type name

        Returns:
            pandas DataFrame with trajectory data
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for export_proficiency_history(). "
                "Install with: pip install pandas"
            )

        trajectory = self.get_improvement_trajectory(task_type)
        data = [t.to_dict() for t in trajectory]

        df = pd.DataFrame(data)
        logger.info(f"Exported {len(df)} trajectory points for {task_type}")
        return df

    def get_statistics_summary(self) -> dict[str, Any]:
        """Get statistical summary of all proficiency data.

        Returns:
            Dictionary with summary statistics
        """
        cursor = self.db.cursor()

        # Basic counts
        cursor.execute("SELECT COUNT(DISTINCT tool) FROM task_outcomes")
        total_tools = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(DISTINCT task) FROM task_outcomes")
        total_tasks = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM task_outcomes")
        total_outcomes = cursor.fetchone()[0] or 0

        # Success rate statistics
        cursor.execute(
            """
            SELECT AVG(success * 1.0), MIN(success * 1.0), MAX(success * 1.0)
            FROM task_outcomes
        """
        )
        avg_success, min_success, max_success = cursor.fetchone()

        # Duration statistics
        cursor.execute("SELECT AVG(duration), MIN(duration), MAX(duration) FROM task_outcomes")
        avg_duration, min_duration, max_duration = cursor.fetchone()

        # Quality score statistics
        cursor.execute(
            "SELECT AVG(quality_score), MIN(quality_score), MAX(quality_score) FROM task_outcomes"
        )
        avg_quality, min_quality, max_quality = cursor.fetchone()

        return {
            "total_tools": total_tools,
            "total_tasks": total_tasks,
            "total_outcomes": total_outcomes,
            "success_rate": {
                "average": avg_success or 0.0,
                "min": min_success or 0.0,
                "max": max_success or 0.0,
            },
            "duration": {
                "average": avg_duration or 0.0,
                "min": min_duration or 0.0,
                "max": max_duration or 0.0,
            },
            "quality_score": {
                "average": avg_quality or 0.0,
                "min": min_quality or 0.0,
                "max": max_quality or 0.0,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def analyze_performance_patterns(self) -> dict[str, Any]:
        """Analyze performance patterns across all tools and tasks.

        Returns:
            Dictionary with pattern analysis results
        """
        cursor = self.db.cursor()

        # Find tools with improving trends
        cursor.execute(
            """
            SELECT tool, trend, success_count, total_count
            FROM tool_proficiency
            WHERE total_count >= 10 AND trend = 'improving'
        """
        )
        improving_tools = [
            {"tool": row[0], "success_rate": row[2] / row[3]} for row in cursor.fetchall()
        ]

        # Find tools with declining trends
        cursor.execute(
            """
            SELECT tool, trend, success_count, total_count
            FROM tool_proficiency
            WHERE total_count >= 10 AND trend = 'declining'
        """
        )
        declining_tools = [
            {"tool": row[0], "success_rate": row[2] / row[3]} for row in cursor.fetchall()
        ]

        # Find fastest tools (by execution time)
        cursor.execute(
            """
            SELECT tool, (total_duration * 1.0 / total_count) as avg_time
            FROM tool_proficiency
            WHERE total_count >= 10
            ORDER BY avg_time ASC
            LIMIT 5
        """
        )
        fastest_tools = [{"tool": row[0], "avg_time": row[1]} for row in cursor.fetchall()]

        # Find most reliable tools (highest success rate)
        cursor.execute(
            """
            SELECT tool, (success_count * 1.0 / total_count) as success_rate
            FROM tool_proficiency
            WHERE total_count >= 10
            ORDER BY success_rate DESC
            LIMIT 5
        """
        )
        most_reliable = [{"tool": row[0], "success_rate": row[1]} for row in cursor.fetchall()]

        return {
            "improving_tools": improving_tools,
            "declining_tools": declining_tools,
            "fastest_tools": fastest_tools,
            "most_reliable_tools": most_reliable,
            "analysis_timestamp": datetime.now().isoformat(),
        }
