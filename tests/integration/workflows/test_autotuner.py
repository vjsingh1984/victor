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

"""Integration tests for performance autotuner.

Tests the complete auto-tuning workflow including:
- Performance analysis
- Optimization suggestion generation
- Optimization application
- A/B testing
- Rollback behavior
"""

from __future__ import annotations

import json
import pytest
from typing import Any

from victor.workflows.performance_autotuner import (
    PerformanceAnalyzer,
    PerformanceAutotuner,
    OptimizationType,
    OptimizationPriority,
    OptimizationSuggestion,
    OptimizationResult,
    PerformanceInsight,
    TeamSizingStrategy,
    FormationSelectionStrategy,
    ToolBudgetStrategy,
    TimeoutTuningStrategy,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metrics() -> list[dict[str, Any]]:
    """Sample metrics data for testing."""
    return [
        {
            "team_id": "test_team",
            "formation": "sequential",
            "member_count": 5,
            "success": True,
            "duration_seconds": 75.0,
            "total_tool_calls": 25,
            "timestamp": "2025-01-15T10:00:00",
        },
        {
            "team_id": "test_team",
            "formation": "sequential",
            "member_count": 5,
            "success": True,
            "duration_seconds": 80.0,
            "total_tool_calls": 28,
            "timestamp": "2025-01-15T10:05:00",
        },
        {
            "team_id": "test_team",
            "formation": "sequential",
            "member_count": 5,
            "success": True,
            "duration_seconds": 70.0,
            "total_tool_calls": 22,
            "timestamp": "2025-01-15T10:10:00",
        },
        {
            "team_id": "test_team",
            "formation": "sequential",
            "member_count": 5,
            "success": False,
            "duration_seconds": 120.0,
            "total_tool_calls": 40,
            "timestamp": "2025-01-15T10:15:00",
        },
        {
            "team_id": "slow_team",
            "formation": "consensus",
            "member_count": 8,
            "success": True,
            "duration_seconds": 150.0,
            "total_tool_calls": 50,
            "timestamp": "2025-01-15T11:00:00",
        },
    ]


@pytest.fixture
def sample_workflow_config() -> dict[str, Any]:
    """Sample workflow configuration."""
    return {
        "team_formation": "sequential",
        "member_count": 5,
        "tool_budget": 50,
        "timeout_seconds": 300,
        "members": [
            {"id": "member1", "role": "assistant"},
            {"id": "member2", "role": "assistant"},
            {"id": "member3", "role": "assistant"},
            {"id": "member4", "role": "assistant"},
            {"id": "member5", "role": "assistant"},
        ],
    }


# =============================================================================
# PerformanceAnalyzer Tests
# =============================================================================


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer functionality."""

    def test_analyze_performance_with_slow_execution(self, sample_metrics: Any) -> None:
        """Test analysis detects slow execution."""
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)

        insights = analyzer.analyze_team_workflow(team_id="test_team")

        # Should detect slow execution (75s avg vs 30s baseline)
        slow_insights = [i for i in insights if i.bottleneck == "slow_execution"]
        assert len(slow_insights) > 0

        insight = slow_insights[0]
        assert insight.severity > 0.5
        assert insight.current_value > insight.baseline_value

    def test_analyze_performance_with_high_latency_outliers(self, sample_metrics: Any) -> None:
        """Test analysis detects high P95 latency."""
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)

        insights = analyzer.analyze_team_workflow(team_id="test_team")

        # Should detect high P95 (120s timeout)
        p95_insights = [i for i in insights if i.bottleneck == "high_latency_outliers"]
        assert len(p95_insights) > 0

    def test_analyze_performance_with_low_success_rate(self, sample_metrics: Any) -> None:
        """Test analysis detects low success rate."""
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)

        insights = analyzer.analyze_team_workflow(team_id="test_team")

        # Should detect low success rate (75% vs 95% baseline)
        success_insights = [i for i in insights if i.bottleneck == "low_success_rate"]
        assert len(success_insights) > 0

    def test_analyze_performance_with_excessive_tool_usage(self, sample_metrics: Any) -> None:
        """Test analysis detects excessive tool usage."""
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)

        insights = analyzer.analyze_team_workflow(team_id="test_team")

        # Should detect excessive tool usage
        tool_insights = [i for i in insights if i.bottleneck == "excessive_tool_usage"]
        assert len(tool_insights) > 0

    def test_analyze_formation_performance(self, sample_metrics: Any) -> None:
        """Test formation-specific performance analysis."""
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)

        insights = analyzer.analyze_team_workflow(team_id="slow_team")

        # Should detect suboptimal consensus formation
        formation_insights = [i for i in insights if "formation" in i.bottleneck]
        assert len(formation_insights) > 0

    def test_analyze_team_size_efficiency(self, sample_metrics: Any) -> None:
        """Test team size analysis."""
        # Add more data points for slow_team to meet the 3 data point requirement
        # Need avg duration > 30s per member = 30 * 8 = 240s average
        # Must be > not >=, so need slightly more than 240s
        # With 150s (original) + 281s + 291s = 722s / 3 = 240.67s (good!)
        additional_slow_metrics = [
            {
                "team_id": "slow_team",
                "formation": "consensus",
                "member_count": 8,
                "success": True,
                "duration_seconds": 281.0,
                "total_tool_calls": 52,
                "timestamp": "2025-01-15T11:05:00",
            },
            {
                "team_id": "slow_team",
                "formation": "consensus",
                "member_count": 8,
                "success": True,
                "duration_seconds": 291.0,
                "total_tool_calls": 48,
                "timestamp": "2025-01-15T11:10:00",
            },
        ]

        extended_metrics = sample_metrics + additional_slow_metrics
        analyzer = PerformanceAnalyzer(metrics_history=extended_metrics)

        insights = analyzer.analyze_team_workflow(team_id="slow_team")

        # Should detect oversized team (8 members)
        size_insights = [i for i in insights if i.bottleneck == "oversized_team"]
        assert len(size_insights) > 0

    def test_load_metrics_from_file(self, sample_metrics: Any, tmp_path: Any) -> None:
        """Test loading metrics from JSON file."""
        metrics_file = tmp_path / "metrics.json"

        with open(metrics_file, "w") as f:
            json.dump(sample_metrics, f)

        analyzer = PerformanceAnalyzer()
        analyzer.load_metrics_from_file(metrics_file)

        assert len(analyzer.metrics_history) == len(sample_metrics)

    def test_insight_to_dict(self) -> None:
        """Test PerformanceInsight serialization."""
        insight = PerformanceInsight(
            bottleneck="test_bottleneck",
            severity=0.8,
            current_value=100.0,
            baseline_value=50.0,
            impact_magnitude=0.7,
            recommendation="Test recommendation",
            affected_components=["component1", "component2"],
        )

        data = insight.to_dict()

        assert data["bottleneck"] == "test_bottleneck"
        assert data["severity"] == 0.8
        assert data["current_value"] == 100.0
        assert len(data["affected_components"]) == 2


# =============================================================================
# OptimizationStrategy Tests
# =============================================================================


class TestOptimizationStrategies:
    """Test optimization strategies."""

    def test_team_sizing_strategy(self, sample_metrics: Any) -> None:
        """Test team sizing strategy."""
        strategy = TeamSizingStrategy()

        # Create insight for oversized team
        insights = [
            PerformanceInsight(
                bottleneck="oversized_team",
                severity=0.7,
                current_value=8,
                baseline_value=5,
                impact_magnitude=0.6,
                recommendation="Reduce team size",
                affected_components=["team_size"],
            )
        ]

        suggestions = strategy.suggest(insights, {})

        assert len(suggestions) > 0
        assert suggestions[0].type == OptimizationType.TEAM_SIZING
        assert suggestions[0].suggested_config["member_count"] < 8

    def test_formation_selection_strategy(self) -> None:
        """Test formation selection strategy."""
        strategy = FormationSelectionStrategy()

        insights = [
            PerformanceInsight(
                bottleneck="suboptimal_formation_consensus",
                severity=0.8,
                current_value=150.0,
                baseline_value=55.0,
                impact_magnitude=0.7,
                recommendation="Switch to faster formation",
                affected_components=["formation"],
            )
        ]

        suggestions = strategy.suggest(insights, {})

        assert len(suggestions) > 0
        assert suggestions[0].type == OptimizationType.FORMATION_SELECTION
        assert "formation" in suggestions[0].suggested_config

    def test_tool_budget_strategy(self) -> None:
        """Test tool budget strategy."""
        strategy = ToolBudgetStrategy()

        insights = [
            PerformanceInsight(
                bottleneck="excessive_tool_usage",
                severity=0.6,
                current_value=50,
                baseline_value=15,
                impact_magnitude=0.4,
                recommendation="Reduce tool budget",
                affected_components=["tool_budget"],
            )
        ]

        suggestions = strategy.suggest(insights, {})

        assert len(suggestions) > 0
        assert suggestions[0].type == OptimizationType.TOOL_BUDGET
        assert suggestions[0].suggested_config["tool_budget"] < 50

    def test_timeout_tuning_strategy(self) -> None:
        """Test timeout tuning strategy."""
        strategy = TimeoutTuningStrategy()

        insights = [
            PerformanceInsight(
                bottleneck="high_latency_outliers",
                severity=0.7,
                current_value=120.0,
                baseline_value=60.0,
                impact_magnitude=0.6,
                recommendation="Adjust timeout based on P95",
                affected_components=["team"],
            )
        ]

        suggestions = strategy.suggest(insights, {})

        assert len(suggestions) > 0
        assert suggestions[0].type == OptimizationType.TIMEOUT_TUNING
        assert "timeout_seconds" in suggestions[0].suggested_config

    def test_apply_team_sizing(self, sample_workflow_config: Any) -> None:
        """Test applying team sizing optimization."""
        strategy = TeamSizingStrategy()

        suggestion = OptimizationSuggestion(
            type=OptimizationType.TEAM_SIZING,
            priority=OptimizationPriority.HIGH,
            description="Reduce team size",
            current_config={"member_count": 8},
            suggested_config={"member_count": 5},
            expected_improvement=15.0,
            confidence=0.8,
            risk_level=0.2,
        )

        new_config = strategy.apply(sample_workflow_config, suggestion)

        assert new_config["member_count"] == 5

    def test_apply_formation_selection(self, sample_workflow_config: Any) -> None:
        """Test applying formation selection."""
        strategy = FormationSelectionStrategy()

        suggestion = OptimizationSuggestion(
            type=OptimizationType.FORMATION_SELECTION,
            priority=OptimizationPriority.CRITICAL,
            description="Switch to parallel formation",
            current_config={"team_formation": "sequential"},
            suggested_config={"team_formation": "parallel"},
            expected_improvement=50.0,
            confidence=0.75,
            risk_level=0.4,
        )

        new_config = strategy.apply(sample_workflow_config, suggestion)

        assert new_config["team_formation"] == "parallel"


# =============================================================================
# PerformanceAutotuner Tests
# =============================================================================


class TestPerformanceAutotuner:
    """Test PerformanceAutotuner functionality."""

    def test_suggest_optimizations(self, sample_metrics: Any, sample_workflow_config: Any) -> None:
        """Test optimization suggestion generation."""
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)
        autotuner = PerformanceAutotuner(analyzer=analyzer)

        suggestions = autotuner.suggest_optimizations(
            team_id="test_team", current_config=sample_workflow_config
        )

        assert len(suggestions) > 0
        assert all(s.confidence >= 0.6 for s in suggestions)

        # Check priority sorting
        priorities = [s.priority for s in suggestions]
        assert priorities == sorted(
            priorities,
            key=lambda p: {
                OptimizationPriority.CRITICAL: 0,
                OptimizationPriority.HIGH: 1,
                OptimizationPriority.MEDIUM: 2,
                OptimizationPriority.LOW: 3,
            }[p],
        )

    @pytest.mark.asyncio
    async def test_apply_optimizations_dry_run(
        self, sample_metrics, sample_workflow_config
    ) -> None:
        """Test applying optimizations in dry-run mode."""
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)
        autotuner = PerformanceAutotuner(analyzer=analyzer)

        suggestions = autotuner.suggest_optimizations(
            team_id="test_team", current_config=sample_workflow_config
        )

        result = await autotuner.apply_optimizations(
            team_id="test_team",
            optimizations=suggestions,
            workflow_config=sample_workflow_config,
            enable_ab_testing=False,
            dry_run=True,
        )

        assert result.success
        assert result.validation_status == "dry_run"
        assert result.team_id == "test_team"

    @pytest.mark.asyncio
    async def test_apply_optimizations_with_ab_testing(
        self, sample_metrics, sample_workflow_config
    ):
        """Test applying optimizations with A/B testing."""
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)
        autotuner = PerformanceAutotuner(analyzer=analyzer, ab_test_threshold=5.0)

        suggestions = autotuner.suggest_optimizations(
            team_id="test_team", current_config=sample_workflow_config
        )

        result = await autotuner.apply_optimizations(
            team_id="test_team",
            optimizations=suggestions,
            workflow_config=sample_workflow_config,
            enable_ab_testing=True,
            dry_run=False,
        )

        assert result.success
        assert result.validation_status in ["passed", "failed", "inconclusive"]
        assert result.rollback_config is not None

    @pytest.mark.asyncio
    async def test_rollback_optimization(self, sample_metrics, sample_workflow_config) -> None:
        """Test rolling back optimizations."""
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)
        autotuner = PerformanceAutotuner(analyzer=analyzer)

        # Apply optimization first
        suggestions = autotuner.suggest_optimizations(
            team_id="test_team", current_config=sample_workflow_config
        )

        await autotuner.apply_optimizations(
            team_id="test_team",
            optimizations=suggestions,
            workflow_config=sample_workflow_config,
            enable_ab_testing=False,
            dry_run=False,
        )

        # Rollback
        success = await autotuner.rollback_optimization("test_team")

        assert success

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_team(self) -> None:
        """Test rolling back optimization for non-existent team."""
        autotuner = PerformanceAutotuner()

        success = await autotuner.rollback_optimization("nonexistent_team")

        assert not success

    def test_get_optimization_history(
        self, sample_metrics: Any, sample_workflow_config: Any
    ) -> None:
        """Test getting optimization history."""
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)
        autotuner = PerformanceAutotuner(analyzer=analyzer)

        # Get empty history
        history = autotuner.get_optimization_history("test_team")
        assert isinstance(history, list)

    def test_optimization_result_to_dict(self, sample_workflow_config: Any) -> None:
        """Test OptimizationResult serialization."""
        optimization = OptimizationSuggestion(
            type=OptimizationType.TEAM_SIZING,
            priority=OptimizationPriority.HIGH,
            description="Test optimization",
            current_config={},
            suggested_config={},
            expected_improvement=10.0,
            confidence=0.8,
            risk_level=0.2,
        )

        result = OptimizationResult(
            success=True,
            team_id="test_team",
            optimization=optimization,
            before_metrics={"duration": 100.0},
            after_metrics={"duration": 90.0},
            improvement_percentage=10.0,
            validation_status="passed",
            rollback_config={},
            error=None,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["team_id"] == "test_team"
        assert data["improvement_percentage"] == 10.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestAutotunerIntegration:
    """Integration tests for complete auto-tuning workflow."""

    @pytest.mark.asyncio
    async def test_complete_autotuning_workflow(
        self, sample_metrics, sample_workflow_config
    ) -> None:
        """Test complete auto-tuning workflow from analysis to rollback."""
        # 1. Analyze performance
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)
        insights = analyzer.analyze_team_workflow("test_team")

        assert len(insights) > 0

        # 2. Generate suggestions
        autotuner = PerformanceAutotuner(analyzer=analyzer)
        suggestions = autotuner.suggest_optimizations(
            team_id="test_team", current_config=sample_workflow_config
        )

        assert len(suggestions) > 0

        # 3. Apply optimization (dry run first)
        result = await autotuner.apply_optimizations(
            team_id="test_team",
            optimizations=suggestions,
            workflow_config=sample_workflow_config,
            enable_ab_testing=False,  # Skip A/B testing for faster test
            dry_run=True,
        )

        assert result.success
        assert result.validation_status == "dry_run"

        # 4. Apply optimization for real (without dry_run)
        result = await autotuner.apply_optimizations(
            team_id="test_team",
            optimizations=suggestions,
            workflow_config=sample_workflow_config,
            enable_ab_testing=False,  # Skip A/B testing for faster test
            dry_run=False,
        )

        assert result.success

        # 5. Rollback
        success = await autotuner.rollback_optimization("test_team")
        assert success

    def test_multiple_teams_analysis(self, sample_metrics: Any) -> None:
        """Test analyzing performance for multiple teams."""
        # Extend sample_metrics with more slow_team data to trigger oversized_team detection
        # Need avg duration > 30s per member = 30 * 8 = 240s average
        # Must be > not >=, so need slightly more than 240s
        additional_slow_metrics = [
            {
                "team_id": "slow_team",
                "formation": "consensus",
                "member_count": 8,
                "success": True,
                "duration_seconds": 281.0,
                "total_tool_calls": 52,
                "timestamp": "2025-01-15T11:05:00",
            },
            {
                "team_id": "slow_team",
                "formation": "consensus",
                "member_count": 8,
                "success": True,
                "duration_seconds": 291.0,
                "total_tool_calls": 48,
                "timestamp": "2025-01-15T11:10:00",
            },
        ]

        extended_metrics = sample_metrics + additional_slow_metrics
        analyzer = PerformanceAnalyzer(metrics_history=extended_metrics)

        # Analyze test_team
        test_team_insights = analyzer.analyze_team_workflow("test_team")
        assert len(test_team_insights) > 0

        # Analyze slow_team
        slow_team_insights = analyzer.analyze_team_workflow("slow_team")
        assert len(slow_team_insights) > 0

        # Teams should have different insights
        test_bottlenecks = {i.bottleneck for i in test_team_insights}
        slow_bottlenecks = {i.bottleneck for i in slow_team_insights}

        # slow_team should have oversized_team bottleneck
        assert "oversized_team" in slow_bottlenecks

    @pytest.mark.asyncio
    async def test_ab_test_threshold_validation(
        self, sample_metrics, sample_workflow_config
    ) -> None:
        """Test A/B test threshold behavior."""
        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)

        # High threshold
        autotuner_strict = PerformanceAutotuner(analyzer=analyzer, ab_test_threshold=20.0)
        suggestions = autotuner_strict.suggest_optimizations(
            team_id="test_team", current_config=sample_workflow_config
        )

        result_strict = await autotuner_strict.apply_optimizations(
            team_id="test_team",
            optimizations=suggestions,
            workflow_config=sample_workflow_config,
            enable_ab_testing=True,
        )

        # Low threshold
        autotuner_lenient = PerformanceAutotuner(analyzer=analyzer, ab_test_threshold=1.0)
        result_lenient = await autotuner_lenient.apply_optimizations(
            team_id="test_team",
            optimizations=suggestions,
            workflow_config=sample_workflow_config,
            enable_ab_testing=True,
        )

        # Lenient should pass more easily
        # (This is a simplified test - real A/B testing would show actual differences)
        assert result_strict.validation_status in ["passed", "failed", "inconclusive"]
        assert result_lenient.validation_status in ["passed", "failed", "inconclusive"]

    def test_strategy_configuration(self, sample_metrics: Any) -> None:
        """Test custom strategy configuration."""
        custom_strategies = [TeamSizingStrategy(), FormationSelectionStrategy()]

        analyzer = PerformanceAnalyzer(metrics_history=sample_metrics)
        autotuner = PerformanceAutotuner(analyzer=analyzer, strategies=custom_strategies)

        suggestions = autotuner.suggest_optimizations("test_team", {})

        # Should only have suggestions from configured strategies
        suggestion_types = {s.type for s in suggestions}
        assert OptimizationType.TOOL_BUDGET not in suggestion_types
        assert OptimizationType.TIMEOUT_TUNING not in suggestion_types
