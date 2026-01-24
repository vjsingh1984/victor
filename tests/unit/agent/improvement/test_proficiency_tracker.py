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

"""Comprehensive unit tests for ProficiencyTracker."""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from victor.agent.improvement.proficiency_tracker import (
    ImprovementTrajectory,
    MovingAverageMetrics,
    ProficiencyMetrics,
    ProficiencyScore,
    ProficiencyTracker,
    Suggestion,
    TaskOutcome,
    TrendDirection,
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Create database connection
    conn = sqlite3.connect(db_path)

    yield conn

    # Cleanup
    conn.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def tracker(temp_db):
    """Create ProficiencyTracker with temporary database."""
    return ProficiencyTracker(db=temp_db)


@pytest.fixture
def sample_outcomes():
    """Sample outcome data for testing."""
    return [
        TaskOutcome(success=True, duration=1.0, cost=0.001, quality_score=0.9),
        TaskOutcome(success=True, duration=1.2, cost=0.001, quality_score=0.85),
        TaskOutcome(success=False, duration=1.5, cost=0.002, quality_score=0.4, errors=["Error 1"]),
        TaskOutcome(success=True, duration=0.8, cost=0.001, quality_score=0.95),
        TaskOutcome(success=True, duration=1.1, cost=0.001, quality_score=0.88),
    ]


class TestTaskOutcome:
    """Test TaskOutcome dataclass."""

    def test_create_outcome_success(self):
        """Test creating a successful outcome."""
        outcome = TaskOutcome(
            success=True,
            duration=1.5,
            cost=0.001,
            quality_score=0.9,
        )

        assert outcome.success is True
        assert outcome.duration == 1.5
        assert outcome.cost == 0.001
        assert outcome.quality_score == 0.9
        assert outcome.errors == []
        assert outcome.metadata == {}

    def test_create_outcome_failure(self):
        """Test creating a failed outcome."""
        outcome = TaskOutcome(
            success=False,
            duration=2.0,
            cost=0.002,
            errors=["Error 1", "Error 2"],
            quality_score=0.3,
        )

        assert outcome.success is False
        assert outcome.duration == 2.0
        assert outcome.cost == 0.002
        assert outcome.errors == ["Error 1", "Error 2"]
        assert outcome.quality_score == 0.3

    def test_outcome_to_dict(self):
        """Test converting outcome to dictionary."""
        outcome = TaskOutcome(
            success=True,
            duration=1.5,
            cost=0.001,
            errors=["Error 1"],
            metadata={"key": "value"},
        )

        data = outcome.to_dict()

        assert data["success"] == 1
        assert data["duration"] == 1.5
        assert data["cost"] == 0.001
        assert data["errors"] == "Error 1"
        assert "metadata" in data

    def test_outcome_from_dict(self):
        """Test creating outcome from dictionary."""
        data = {
            "success": 1,
            "duration": 1.5,
            "cost": 0.001,
            "errors": "Error 1,Error 2",
            "quality_score": 0.8,
            "timestamp": "2025-01-20T00:00:00",
        }

        outcome = TaskOutcome.from_dict(data)

        assert outcome.success is True
        assert outcome.duration == 1.5
        assert outcome.cost == 0.001
        assert outcome.errors == ["Error 1", "Error 2"]
        assert outcome.quality_score == 0.8


class TestProficiencyTracker:
    """Test ProficiencyTracker class."""

    def test_initialization(self, temp_db):
        """Test tracker initialization."""
        tracker = ProficiencyTracker(db=temp_db)

        assert tracker.db is not None
        assert tracker._cache == {}
        assert tracker._moving_avg_window == 20

    def test_initialization_custom_window(self, temp_db):
        """Test tracker initialization with custom window size."""
        tracker = ProficiencyTracker(db=temp_db, moving_avg_window=50)

        assert tracker._moving_avg_window == 50

    def test_ensure_tables(self, tracker):
        """Test table creation."""
        cursor = tracker.db.cursor()

        # Check tool_proficiency table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tool_proficiency'"
        )
        assert cursor.fetchone() is not None

        # Check task_outcomes table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_outcomes'")
        assert cursor.fetchone() is not None

        # Check task_success_rates table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='task_success_rates'"
        )
        assert cursor.fetchone() is not None

        # Check improvement_trajectory table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='improvement_trajectory'"
        )
        assert cursor.fetchone() is not None


class TestRecordOutcome:
    """Test outcome recording (10 tests)."""

    def test_record_outcome_success(self, tracker):
        """Test recording a successful outcome."""
        outcome = TaskOutcome(
            success=True,
            duration=1.5,
            cost=0.001,
            quality_score=0.9,
        )

        tracker.record_outcome(
            task="code_review",
            tool="ast_analyzer",
            outcome=outcome,
        )

        # Verify outcome was recorded
        cursor = tracker.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM task_outcomes")
        count = cursor.fetchone()[0]
        assert count == 1

    def test_record_outcome_failure(self, tracker):
        """Test recording a failed outcome."""
        outcome = TaskOutcome(
            success=False,
            duration=2.0,
            cost=0.002,
            errors=["Error 1"],
        )

        tracker.record_outcome(
            task="code_review",
            tool="ast_analyzer",
            outcome=outcome,
        )

        cursor = tracker.db.cursor()
        cursor.execute("SELECT success FROM task_outcomes WHERE tool = ?", ("ast_analyzer",))
        success = cursor.fetchone()[0]
        assert success == 0

    def test_record_outcome_with_quality_score(self, tracker):
        """Test recording outcome with quality score."""
        outcome = TaskOutcome(
            success=True,
            duration=1.5,
            cost=0.001,
            quality_score=0.92,
        )

        tracker.record_outcome(
            task="code_review",
            tool="ast_analyzer",
            outcome=outcome,
        )

        score = tracker.get_proficiency("ast_analyzer")
        assert score is not None
        assert score.quality_score == 0.92

    def test_record_outcome_with_duration(self, tracker):
        """Test recording outcome with duration tracking."""
        outcome = TaskOutcome(
            success=True,
            duration=2.5,
            cost=0.001,
        )

        tracker.record_outcome(
            task="test_generation",
            tool="test_gen",
            outcome=outcome,
        )

        score = tracker.get_proficiency("test_gen")
        assert score is not None
        assert score.avg_execution_time == 2.5

    def test_record_outcome_with_cost(self, tracker):
        """Test recording outcome with cost tracking."""
        outcome = TaskOutcome(
            success=True,
            duration=1.0,
            cost=0.005,
        )

        tracker.record_outcome(
            task="code_review",
            tool="expensive_tool",
            outcome=outcome,
        )

        score = tracker.get_proficiency("expensive_tool")
        assert score is not None
        assert score.avg_cost == 0.005

    def test_record_multiple_outcomes_batch(self, tracker, sample_outcomes):
        """Test recording multiple outcomes in batch."""
        for outcome in sample_outcomes:
            tracker.record_outcome(
                task="code_review",
                tool="ast_analyzer",
                outcome=outcome,
            )

        cursor = tracker.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM task_outcomes WHERE tool = ?", ("ast_analyzer",))
        count = cursor.fetchone()[0]
        assert count == 5

    def test_record_outcome_updates_proficiency(self, tracker):
        """Test that recording outcome updates proficiency."""
        outcome = TaskOutcome(success=True, duration=1.5, cost=0.001, quality_score=0.9)

        tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome)

        # Check proficiency was updated
        cursor = tracker.db.cursor()
        cursor.execute(
            "SELECT success_count, total_count FROM tool_proficiency WHERE tool = ?",
            ("ast_analyzer",),
        )
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == 1  # success_count
        assert row[1] == 1  # total_count

    def test_record_outcome_updates_task_success_rate(self, tracker):
        """Test that recording outcome updates task success rate."""
        outcome = TaskOutcome(success=True, duration=1.5, cost=0.001)

        tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome)

        cursor = tracker.db.cursor()
        cursor.execute(
            "SELECT success_count, total_count FROM task_success_rates WHERE task = ?",
            ("code_review",),
        )
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == 1
        assert row[1] == 1

    def test_record_outcome_invalidates_cache(self, tracker):
        """Test that recording outcome invalidates cache."""
        outcome1 = TaskOutcome(success=True, duration=1.0, cost=0.001)
        tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome1)

        # Load into cache
        score1 = tracker.get_proficiency("ast_analyzer")
        assert "ast_analyzer" in tracker._cache

        # Record another outcome
        outcome2 = TaskOutcome(success=False, duration=2.0, cost=0.002)
        tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome2)

        # Verify cache was invalidated
        assert "ast_analyzer" not in tracker._cache

    def test_record_outcome_with_errors(self, tracker):
        """Test recording outcome with error messages."""
        outcome = TaskOutcome(
            success=False,
            duration=1.0,
            cost=0.001,
            errors=["Syntax error", "Type error"],
        )

        tracker.record_outcome(
            task="code_review",
            tool="error_tool",
            outcome=outcome,
        )

        cursor = tracker.db.cursor()
        cursor.execute("SELECT errors FROM task_outcomes WHERE tool = ?", ("error_tool",))
        errors = cursor.fetchone()[0]
        assert "Syntax error" in errors
        assert "Type error" in errors


class TestGetProficiency:
    """Test proficiency queries (10 tests)."""

    def test_get_proficiency_new_tool(self, tracker):
        """Test getting proficiency for new tool."""
        score = tracker.get_proficiency("new_tool")

        assert score is None

    def test_get_proficiency_existing_tool(self, tracker):
        """Test getting proficiency for existing tool."""
        # Record some outcomes
        for i in range(5):
            outcome = TaskOutcome(success=True if i < 4 else False, duration=1.0, cost=0.001)
            tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome)

        score = tracker.get_proficiency("ast_analyzer")

        assert score is not None
        assert score.success_rate == 0.8
        assert score.avg_execution_time == 1.0
        assert score.avg_cost == 0.001
        assert score.total_executions == 5

    def test_get_proficiency_caching(self, tracker):
        """Test proficiency caching."""
        outcome = TaskOutcome(success=True, duration=1.5, cost=0.001)
        tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome)

        # First call - loads from DB
        score1 = tracker.get_proficiency("ast_analyzer")
        # Second call - from cache
        score2 = tracker.get_proficiency("ast_analyzer")

        assert score1 is not None
        assert score1 is score2

    def test_get_proficiency_invalidation(self, tracker):
        """Test cache invalidation after recording."""
        outcome1 = TaskOutcome(success=True, duration=1.0, cost=0.001)
        tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome1)

        # Load into cache
        score1 = tracker.get_proficiency("ast_analyzer")

        # Record another outcome
        outcome2 = TaskOutcome(success=False, duration=2.0, cost=0.002)
        tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome2)

        # Get updated score
        score2 = tracker.get_proficiency("ast_analyzer")

        assert score2 is not None
        assert score2.total_executions == 2
        assert score2.success_rate == 0.5

    def test_get_top_proficiencies(self, tracker):
        """Test getting top N proficiencies."""
        # Record outcomes for different tools
        for i in range(10):
            outcome1 = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(task="task", tool="excellent_tool", outcome=outcome1)

            outcome2 = TaskOutcome(success=(i % 2 == 0), duration=1.0, cost=0.001)
            tracker.record_outcome(task="task", tool="average_tool", outcome=outcome2)

            outcome3 = TaskOutcome(success=(i % 3 == 0), duration=1.0, cost=0.001)
            tracker.record_outcome(task="task", tool="poor_tool", outcome=outcome3)

        top_proficiencies = tracker.get_top_proficiencies(n=5)

        assert len(top_proficiencies) == 3
        # Should be ordered by success rate
        assert top_proficiencies[0][0] == "excellent_tool"
        assert top_proficiencies[0][1].success_rate == 1.0

    def test_get_weaknesses_below_threshold(self, tracker):
        """Test getting tools below threshold."""
        # Record outcomes with varying success rates
        for i in range(10):
            outcome = TaskOutcome(success=(i % 2 == 0), duration=1.0, cost=0.001)
            tracker.record_outcome(task="task", tool="weak_tool", outcome=outcome)

        weaknesses = tracker.get_weaknesses(threshold=0.7, min_executions=5)

        assert "weak_tool" in weaknesses
        assert len(weaknesses) > 0

    def test_get_weaknesses_empty(self, tracker):
        """Test getting weaknesses with no data."""
        weaknesses = tracker.get_weaknesses(threshold=0.7)

        assert weaknesses == []

    def test_get_weaknesses_high_threshold(self, tracker):
        """Test getting weaknesses with high threshold."""
        # Record successful outcomes
        for i in range(10):
            outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(task="task", tool="strong_tool", outcome=outcome)

        weaknesses = tracker.get_weaknesses(threshold=0.9, min_executions=5)

        # Should not include strong_tool (100% success rate)
        assert "strong_tool" not in weaknesses

    def test_get_task_success_rate_new_task(self, tracker):
        """Test success rate for new task."""
        rate = tracker.get_task_success_rate("new_task")

        assert rate == 0.0

    def test_get_task_success_rate_existing_task(self, tracker):
        """Test success rate for existing task."""
        # Record outcomes
        for i in range(10):
            outcome = TaskOutcome(success=True if i < 7 else False, duration=1.0, cost=0.001)
            tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome)

        rate = tracker.get_task_success_rate("code_review")

        assert rate == 0.7


class TestImprovementTracking:
    """Test improvement tracking (10 tests)."""

    def test_update_proficiency_incremental(self, tracker):
        """Test incremental proficiency update."""
        # Record initial outcomes
        for i in range(10):
            outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome)

        initial_rate = tracker.get_task_success_rate("code_review")
        assert initial_rate == 1.0

        # Apply negative delta
        tracker.update_proficiency("code_review", delta=-0.2)

        updated_rate = tracker.get_task_success_rate("code_review")
        assert updated_rate == pytest.approx(0.8, rel=1e-2)

    def test_update_proficiency_clamps_to_zero(self, tracker):
        """Test that proficiency update clamps to zero."""
        # Record initial outcomes
        for i in range(10):
            outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(task="task", tool="tool", outcome=outcome)

        # Apply large negative delta
        tracker.update_proficiency("task", delta=-2.0)

        rate = tracker.get_task_success_rate("task")
        assert rate == 0.0

    def test_update_proficiency_clamps_to_one(self, tracker):
        """Test that proficiency update clamps to one."""
        # Record initial outcomes
        for i in range(10):
            outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(task="task", tool="tool", outcome=outcome)

        # Apply large positive delta
        tracker.update_proficiency("task", delta=2.0)

        rate = tracker.get_task_success_rate("task")
        assert rate == 1.0

    def test_detect_trend_improving(self, tracker):
        """Test detecting improving trend."""
        values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        trend = tracker.detect_trend_direction(values, threshold=0.1)

        assert trend == TrendDirection.IMPROVING

    def test_detect_trend_declining(self, tracker):
        """Test detecting declining trend."""
        values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        trend = tracker.detect_trend_direction(values, threshold=0.1)

        assert trend == TrendDirection.DECLINING

    def test_detect_trend_stable(self, tracker):
        """Test detecting stable trend."""
        values = [0.7, 0.72, 0.68, 0.71, 0.69, 0.7]
        trend = tracker.detect_trend_direction(values, threshold=0.1)

        assert trend == TrendDirection.STABLE

    def test_detect_trend_insufficient_data(self, tracker):
        """Test trend detection with insufficient data."""
        values = [0.5, 0.6, 0.7]
        trend = tracker.detect_trend_direction(values)

        assert trend == TrendDirection.UNKNOWN

    def test_get_improvement_trajectory(self, tracker):
        """Test getting improvement trajectory."""
        # Record some outcomes first
        for i in range(5):
            outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome)

        # Record a trajectory snapshot
        tracker.record_trajectory_snapshot("code_review")

        trajectory = tracker.get_improvement_trajectory("code_review")

        assert len(trajectory) >= 1
        assert trajectory[0].task_type == "code_review"

    def test_export_training_data(self, tracker):
        """Test exporting training data."""
        # Record some outcomes
        for i in range(5):
            outcome = TaskOutcome(
                success=True,
                duration=1.0,
                cost=0.001,
                quality_score=0.9,
            )
            tracker.record_outcome(
                task="code_review",
                tool="ast_analyzer",
                outcome=outcome,
            )

        # Skip if pandas not available
        try:
            import pandas as pd

            df = tracker.export_training_data()

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 5
            assert "task" in df.columns
            assert "tool" in df.columns
            assert "success" in df.columns
            assert "duration" in df.columns
            assert "cost" in df.columns
        except ImportError:
            pytest.skip("pandas not available")

    def test_export_training_data_empty(self, tracker):
        """Test exporting training data with no data."""
        try:
            import pandas as pd

            df = tracker.export_training_data()

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
        except ImportError:
            pytest.skip("pandas not available")


class TestMovingAverageMetrics:
    """Test moving average metrics."""

    def test_get_moving_average_metrics(self, tracker):
        """Test getting moving average metrics."""
        # Record sufficient outcomes
        for i in range(10):
            outcome = TaskOutcome(
                success=(i % 2 == 0),
                duration=1.0 + i * 0.1,
                cost=0.001,
                quality_score=0.8,
            )
            tracker.record_outcome(
                task="code_review",
                tool="ast_analyzer",
                outcome=outcome,
            )

        metrics = tracker.get_moving_average_metrics("code_review")

        assert metrics is not None
        assert metrics.window_size == 20
        assert metrics.success_rate_ma >= 0.0
        assert metrics.execution_time_ma > 0
        assert metrics.variance >= 0.0

    def test_get_moving_average_metrics_insufficient_data(self, tracker):
        """Test moving average with insufficient data."""
        # Record only 2 outcomes
        for i in range(2):
            outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(
                task="task",
                tool="tool",
                outcome=outcome,
            )

        metrics = tracker.get_moving_average_metrics("task")

        assert metrics is None

    def test_compute_moving_average(self, tracker):
        """Test computing moving average."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        window = 3

        moving_avgs = tracker.compute_moving_average(values, window)

        assert len(moving_avgs) == 5
        assert moving_avgs[0] == pytest.approx(2.0)
        assert moving_avgs[1] == pytest.approx(3.0)
        assert moving_avgs[-1] == pytest.approx(6.0)

    def test_compute_moving_average_insufficient_data(self, tracker):
        """Test moving average with insufficient data."""
        values = [1.0, 2.0]
        window = 5

        moving_avgs = tracker.compute_moving_average(values, window)

        assert len(moving_avgs) == 0


class TestStatisticsAndExport:
    """Test statistical computations and data export."""

    def test_get_statistics_summary(self, tracker):
        """Test getting statistics summary."""
        # Record outcomes
        for i in range(10):
            outcome = TaskOutcome(
                success=(i % 2 == 0),
                duration=1.0 + i * 0.1,
                cost=0.001 + i * 0.0001,
                quality_score=0.7 + i * 0.02,
            )
            tracker.record_outcome(
                task="code_review",
                tool="ast_analyzer",
                outcome=outcome,
            )

        summary = tracker.get_statistics_summary()

        assert summary["total_tools"] == 1
        assert summary["total_tasks"] == 1
        assert summary["total_outcomes"] == 10
        assert "success_rate" in summary
        assert "duration" in summary
        assert "quality_score" in summary

    def test_analyze_performance_patterns(self, tracker):
        """Test analyzing performance patterns."""
        # Record outcomes for improving tool
        for i in range(15):
            success = i >= 7  # Success rate improves
            outcome = TaskOutcome(
                success=success,
                duration=1.0,
                cost=0.001,
            )
            tracker.record_outcome(
                task="task",
                tool="improving_tool",
                outcome=outcome,
            )

        patterns = tracker.analyze_performance_patterns()

        assert "improving_tools" in patterns
        assert "declining_tools" in patterns
        assert "fastest_tools" in patterns
        assert "most_reliable_tools" in patterns


class TestTrendDirection:
    """Test TrendDirection enum."""

    def test_trend_direction_values(self):
        """Test TrendDirection enum values."""
        assert TrendDirection.IMPROVING == "improving"
        assert TrendDirection.STABLE == "stable"
        assert TrendDirection.DECLINING == "declining"
        assert TrendDirection.UNKNOWN == "unknown"


class TestProficiencyScore:
    """Test ProficiencyScore dataclass."""

    def test_proficiency_score_to_dict(self):
        """Test converting ProficiencyScore to dictionary."""
        score = ProficiencyScore(
            success_rate=0.85,
            avg_execution_time=1.5,
            avg_cost=0.001,
            total_executions=10,
            trend=TrendDirection.IMPROVING,
            last_updated="2025-01-20T00:00:00",
            quality_score=0.9,
        )

        data = score.to_dict()

        assert data["success_rate"] == 0.85
        assert data["avg_execution_time"] == 1.5
        assert data["avg_cost"] == 0.001
        assert data["total_executions"] == 10
        assert data["trend"] == "improving"
        assert data["quality_score"] == 0.9


class TestSuggestion:
    """Test Suggestion dataclass."""

    def test_suggestion_to_dict(self):
        """Test converting Suggestion to dictionary."""
        suggestion = Suggestion(
            tool="ast_analyzer",
            reason="Low success rate",
            expected_improvement=0.2,
            confidence=0.8,
            priority="high",
        )

        data = suggestion.to_dict()

        assert data["tool"] == "ast_analyzer"
        assert data["reason"] == "Low success rate"
        assert data["expected_improvement"] == 0.2
        assert data["confidence"] == 0.8
        assert data["priority"] == "high"


class TestProficiencyMetrics:
    """Test ProficiencyMetrics dataclass."""

    def test_proficiency_metrics_to_dict(self):
        """Test converting ProficiencyMetrics to dictionary."""
        metrics = ProficiencyMetrics(
            total_tools=5,
            total_tasks=3,
            total_outcomes=100,
            tool_scores={},
            task_success_rates={},
            top_performing_tools=[("tool1", 0.9)],
            improvement_opportunities=[("tool2", 0.3)],
            timestamp="2025-01-20T00:00:00",
        )

        data = metrics.to_dict()

        assert data["total_tools"] == 5
        assert data["total_tasks"] == 3
        assert data["total_outcomes"] == 100
        assert len(data["top_performing_tools"]) == 1
        assert len(data["improvement_opportunities"]) == 1


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_all_tools(self, tracker):
        """Test getting all tools."""
        # Record outcomes
        for tool in ["tool1", "tool2", "tool3"]:
            outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(task="task", tool=tool, outcome=outcome)

        tools = tracker.get_all_tools()

        assert len(tools) == 3
        assert "tool1" in tools
        assert "tool2" in tools
        assert "tool3" in tools

    def test_get_all_tasks(self, tracker):
        """Test getting all tasks."""
        # Record outcomes
        for task in ["task1", "task2", "task3"]:
            outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(task=task, tool="tool", outcome=outcome)

        tasks = tracker.get_all_tasks()

        assert len(tasks) == 3
        assert "task1" in tasks
        assert "task2" in tasks
        assert "task3" in tasks

    def test_reset_tool(self, tracker):
        """Test resetting a tool."""
        # Record outcomes
        outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
        tracker.record_outcome(task="task", tool="tool1", outcome=outcome)

        # Verify data exists
        score = tracker.get_proficiency("tool1")
        assert score is not None

        # Reset
        tracker.reset_tool("tool1")

        # Verify data cleared
        score = tracker.get_proficiency("tool1")
        assert score is None

    def test_reset_all(self, tracker):
        """Test resetting all data."""
        # Record outcomes
        outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
        tracker.record_outcome(task="task", tool="tool", outcome=outcome)

        # Reset all
        tracker.reset_all()

        # Verify all data cleared
        assert tracker.get_all_tools() == []
        assert tracker.get_all_tasks() == []
        assert tracker._cache == {}


class TestExportMetrics:
    """Test metrics export."""

    def test_export_metrics_empty(self, tracker):
        """Test exporting metrics with no data."""
        metrics = tracker.export_metrics()

        assert isinstance(metrics, ProficiencyMetrics)
        assert metrics.total_tools == 0
        assert metrics.total_tasks == 0
        assert metrics.total_outcomes == 0

    def test_export_metrics_with_data(self, tracker):
        """Test exporting metrics with data."""
        # Record outcomes
        for i in range(5):
            outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome)

        metrics = tracker.export_metrics()

        assert metrics.total_tools == 1
        assert metrics.total_tasks == 1
        assert metrics.total_outcomes == 5
        assert "ast_analyzer" in metrics.tool_scores
        assert "code_review" in metrics.task_success_rates

    def test_export_metrics_top_performing_tools(self, tracker):
        """Test top performing tools in metrics."""
        # Record outcomes for different tools
        for i in range(10):
            outcome1 = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(task="code_review", tool="good_tool", outcome=outcome1)

            outcome2 = TaskOutcome(success=(i % 2 == 0), duration=1.0, cost=0.001)
            tracker.record_outcome(task="code_review", tool="ok_tool", outcome=outcome2)

        metrics = tracker.export_metrics(top_n=5)

        assert len(metrics.top_performing_tools) > 0
        # good_tool should be first
        assert metrics.top_performing_tools[0][0] == "good_tool"

    def test_export_metrics_improvement_opportunities(self, tracker):
        """Test improvement opportunities in metrics."""
        # Record low success rate outcomes
        for i in range(15):
            outcome = TaskOutcome(success=(i % 3 == 0), duration=1.0, cost=0.001)
            tracker.record_outcome(task="code_review", tool="needs_improvement", outcome=outcome)

        metrics = tracker.export_metrics()

        assert len(metrics.improvement_opportunities) > 0


class TestSuggestToolForTask:
    """Test tool suggestion."""

    def test_suggest_tool_for_task_no_data(self, tracker):
        """Test suggestion with no data."""
        suggestion = tracker.suggest_tool_for_task("code_review")

        assert suggestion is None

    def test_suggest_tool_for_task_with_data(self, tracker):
        """Test suggestion with data."""
        # Record outcomes for different tools
        for i in range(10):
            outcome1 = TaskOutcome(success=True, duration=1.0, cost=0.001, quality_score=0.9)
            tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome1)

            outcome2 = TaskOutcome(
                success=(i % 2 == 0), duration=1.5, cost=0.002, quality_score=0.7
            )
            tracker.record_outcome(task="code_review", tool="semantic_search", outcome=outcome2)

        suggestion = tracker.suggest_tool_for_task("code_review")

        # Should suggest ast_analyzer due to higher success rate
        assert suggestion == "ast_analyzer"

    def test_suggest_tool_for_task_minimum_executions(self, tracker):
        """Test suggestion respects minimum executions."""
        # Record only 2 outcomes
        for i in range(2):
            outcome = TaskOutcome(success=True, duration=1.0, cost=0.001)
            tracker.record_outcome(task="code_review", tool="ast_analyzer", outcome=outcome)

        suggestion = tracker.suggest_tool_for_task("code_review")

        # Should return None (less than 3 executions)
        assert suggestion is None


class TestGetImprovementSuggestions:
    """Test improvement suggestion generation."""

    def test_get_improvement_suggestions_no_data(self, tracker):
        """Test suggestions with no data."""
        suggestions = tracker.get_improvement_suggestions(agent_id="agent-1")

        assert suggestions == []

    def test_get_improvement_suggestions_low_success_rate(self, tracker):
        """Test suggestions for low success rate tools."""
        # Record outcomes with low success rate
        for i in range(15):
            outcome = TaskOutcome(success=(i % 3 == 0), duration=1.0, cost=0.001)
            tracker.record_outcome(task="code_review", tool="poor_tool", outcome=outcome)

        suggestions = tracker.get_improvement_suggestions(agent_id="agent-1", min_executions=10)

        assert len(suggestions) > 0
        assert any(s.tool == "poor_tool" for s in suggestions)
        assert any(s.priority == "high" for s in suggestions)

    def test_get_improvement_suggestions_declining_trend(self, tracker):
        """Test suggestions for declining tools."""
        # Record outcomes with declining performance
        for i in range(15):
            success = i < 10  # Success rate declines
            outcome = TaskOutcome(success=success, duration=1.0, cost=0.001)
            tracker.record_outcome(task="code_review", tool="declining_tool", outcome=outcome)

        suggestions = tracker.get_improvement_suggestions(agent_id="agent-1", min_executions=10)

        # Should have declining tool suggestion
        assert len(suggestions) > 0
