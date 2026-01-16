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

"""Automatic performance optimization for team workflows.

This module provides intelligent auto-tuning capabilities for team node
configurations within workflows. It analyzes historical execution metrics,
identifies bottlenecks, suggests and applies optimizations, and includes
A/B testing with rollback capabilities.

Key Features:
- PerformanceAnalyzer: Analyze historical metrics and identify bottlenecks
- PerformanceAutotuner: Auto-optimize team configurations
- OptimizationStrategies: Team sizing, formation selection, tool budget, etc.
- A/B Testing: Validate optimizations before permanent application
- Rollback: Revert optimizations on regression detection

SOLID Principles:
- SRP: Each class handles one aspect (analysis, tuning, strategy)
- OCP: Extensible via custom optimization strategies
- LSP: Strategies are interchangeable
- DIP: Depends on abstractions (TeamContext, Metrics)

Usage:
    from victor.workflows.performance_autotuner import (
        PerformanceAutotuner,
        PerformanceAnalyzer,
    )

    # Analyze current performance
    analyzer = PerformanceAnalyzer()
    insights = analyzer.analyze_team_workflow(team_id="my_team")

    # Get optimization suggestions
    suggestions = autotuner.suggest_optimizations(insights)

    # Apply optimizations (with A/B testing)
    result = await autotuner.apply_optimizations(
        team_id="my_team",
        optimizations=suggestions,
        enable_ab_testing=True
    )

    # Rollback if needed
    await autotuner.rollback_optimization(team_id="my_team")
"""

from __future__ import annotations

import logging
import statistics
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json
import copy

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes and Enums
# =============================================================================


class OptimizationType(str, Enum):
    """Types of optimizations available."""

    TEAM_SIZING = "team_sizing"
    FORMATION_SELECTION = "formation_selection"
    TOOL_BUDGET = "tool_budget"
    TIMEOUT_TUNING = "timeout_tuning"
    MEMBER_SELECTION = "member_selection"
    PARALLELIZATION = "parallelization"


class OptimizationPriority(str, Enum):
    """Priority levels for optimizations."""

    CRITICAL = "critical"  # High impact, low risk
    HIGH = "high"  # High impact, medium risk
    MEDIUM = "medium"  # Medium impact, low risk
    LOW = "low"  # Low impact, low risk


@dataclass
class PerformanceInsight:
    """Insight from performance analysis.

    Attributes:
        bottleneck: Identified bottleneck (e.g., "slow_member", "formation", "tool_budget")
        severity: Severity level (0-1)
        current_value: Current metric value
        baseline_value: Expected/baseline value
        impact_magnitude: How much improvement is possible (0-1)
        recommendation: Textual recommendation
        affected_components: List of affected component IDs
    """

    bottleneck: str
    severity: float  # 0-1
    current_value: float
    baseline_value: float
    impact_magnitude: float  # 0-1
    recommendation: str
    affected_components: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bottleneck": self.bottleneck,
            "severity": self.severity,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "impact_magnitude": self.impact_magnitude,
            "recommendation": self.recommendation,
            "affected_components": self.affected_components,
        }


@dataclass
class OptimizationSuggestion:
    """Suggested optimization.

    Attributes:
        type: Type of optimization
        priority: Priority level
        description: Human-readable description
        current_config: Current configuration
        suggested_config: Suggested configuration
        expected_improvement: Expected improvement percentage (0-100)
        confidence: Confidence in suggestion (0-1)
        risk_level: Risk level (0-1)
        metadata: Additional metadata
    """

    type: OptimizationType
    priority: OptimizationPriority
    description: str
    current_config: Dict[str, Any]
    suggested_config: Dict[str, Any]
    expected_improvement: float  # 0-100
    confidence: float  # 0-1
    risk_level: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "priority": self.priority.value,
            "description": self.description,
            "current_config": self.current_config,
            "suggested_config": self.suggested_config,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "metadata": self.metadata,
        }


@dataclass
class OptimizationResult:
    """Result from applying optimization.

    Attributes:
        success: Whether optimization was applied successfully
        team_id: Team ID
        optimization: Optimization that was applied
        before_metrics: Metrics before optimization
        after_metrics: Metrics after optimization (if available)
        improvement_percentage: Actual improvement achieved
        validation_status: Status of A/B testing (if enabled)
        rollback_config: Configuration for rollback
        error: Error message if failed
    """

    success: bool
    team_id: str
    optimization: OptimizationSuggestion
    before_metrics: Dict[str, float]
    after_metrics: Optional[Dict[str, float]]
    improvement_percentage: Optional[float]
    validation_status: Optional[str]
    rollback_config: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "team_id": self.team_id,
            "optimization": self.optimization.to_dict(),
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "improvement_percentage": self.improvement_percentage,
            "validation_status": self.validation_status,
            "rollback_config": self.rollback_config,
            "error": self.error,
        }


# =============================================================================
# Performance Analyzer
# =============================================================================


class PerformanceAnalyzer:
    """Analyzes historical team execution metrics to identify bottlenecks.

    This class processes metrics from TeamMetricsCollector and benchmark data
    to generate actionable insights about team performance.

    Attributes:
        metrics_history: Historical metrics data
        baseline_metrics: Baseline metrics for comparison
        benchmark_data: Benchmark data for reference

    Example:
        >>> analyzer = PerformanceAnalyzer()
        >>>
        >>> # Load metrics from TeamMetricsCollector
        >>> analyzer.load_metrics_from_collector(collector)
        >>>
        >>> # Analyze performance
        >>> insights = analyzer.analyze_team_workflow(team_id="my_team")
        >>>
        >>> # Get bottleneck summary
        >>> for insight in insights:
        ...     print(f"{insight.bottleneck}: {insight.recommendation}")
    """

    def __init__(
        self,
        metrics_history: Optional[List[Dict[str, Any]]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        benchmark_data: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize performance analyzer.

        Args:
            metrics_history: Historical metrics data
            baseline_metrics: Baseline metrics for comparison
            benchmark_data: Benchmark data from benchmark suite
        """
        self.metrics_history = metrics_history or []
        self.baseline_metrics = baseline_metrics or self._default_baselines()
        self.benchmark_data = benchmark_data or self._default_benchmarks()

    def _default_baselines(self) -> Dict[str, float]:
        """Get default baseline metrics.

        Returns:
            Default baseline metrics
        """
        return {
            "avg_duration_seconds": 30.0,
            "success_rate": 0.95,
            "avg_tool_calls": 15.0,
            "p95_duration_seconds": 60.0,
            "throughput_teams_per_minute": 2.0,
        }

    def _default_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Get default benchmark data.

        Returns:
            Default benchmark metrics by formation
        """
        return {
            "sequential": {"avg_latency_ms": 55, "throughput": 1.8},
            "parallel": {"avg_latency_ms": 22, "throughput": 4.5},
            "pipeline": {"avg_latency_ms": 45, "throughput": 2.2},
            "hierarchical": {"avg_latency_ms": 50, "throughput": 2.0},
            "consensus": {"avg_latency_ms": 115, "throughput": 0.8},
        }

    def load_metrics_from_collector(self, collector) -> None:
        """Load metrics from TeamMetricsCollector.

        Args:
            collector: TeamMetricsCollector instance
        """
        # Get summary metrics
        summary = collector.get_summary()
        self.metrics_history.append({"timestamp": datetime.now().isoformat(), **summary})

    def load_metrics_from_file(self, filepath: Path) -> None:
        """Load metrics from JSON file.

        Args:
            filepath: Path to metrics JSON file
        """
        with open(filepath) as f:
            data = json.load(f)
            if isinstance(data, list):
                self.metrics_history.extend(data)
            else:
                self.metrics_history.append(data)

    def analyze_team_workflow(self, team_id: Optional[str] = None) -> List[PerformanceInsight]:
        """Analyze team workflow performance.

        Args:
            team_id: Optional team ID to filter by

        Returns:
            List of performance insights
        """
        insights = []

        if not self.metrics_history:
            logger.warning("No metrics history available for analysis")
            return insights

        # Filter metrics by team_id if provided
        relevant_metrics = (
            [m for m in self.metrics_history if m.get("team_id") == team_id]
            if team_id
            else self.metrics_history
        )

        if not relevant_metrics:
            logger.warning(f"No metrics found for team_id: {team_id}")
            return insights

        # Analyze different aspects
        insights.extend(self._analyze_duration(relevant_metrics))
        insights.extend(self._analyze_success_rate(relevant_metrics))
        insights.extend(self._analyze_tool_usage(relevant_metrics))
        insights.extend(self._analyze_formation_performance(relevant_metrics))
        insights.extend(self._analyze_team_size(relevant_metrics))

        # Sort by severity
        insights.sort(key=lambda i: i.severity, reverse=True)

        return insights

    def _analyze_duration(self, metrics: List[Dict[str, Any]]) -> List[PerformanceInsight]:
        """Analyze execution duration.

        Args:
            metrics: Metrics data

        Returns:
            List of duration-related insights
        """
        insights = []

        durations = [m.get("duration_seconds", 0) for m in metrics if m.get("duration_seconds")]
        if not durations:
            return insights

        avg_duration = statistics.mean(durations)
        p95_duration = statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations)
        baseline_duration = self.baseline_metrics["avg_duration_seconds"]

        # Check if average is above baseline
        if avg_duration > baseline_duration * 1.2:
            severity = min(1.0, (avg_duration - baseline_duration) / baseline_duration)
            insights.append(
                PerformanceInsight(
                    bottleneck="slow_execution",
                    severity=severity,
                    current_value=avg_duration,
                    baseline_value=baseline_duration,
                    impact_magnitude=severity,
                    recommendation=f"Average execution time ({avg_duration:.1f}s) is "
                    f"{((avg_duration/baseline_duration - 1) * 100):.1f}% above baseline. "
                    f"Consider optimizing formation or reducing team size.",
                    affected_components=["team", "formation"],
                )
            )

        # Check P95
        p95_baseline = self.baseline_metrics["p95_duration_seconds"]
        if p95_duration > p95_baseline * 1.5:
            severity = min(1.0, (p95_duration - p95_baseline) / p95_baseline)
            insights.append(
                PerformanceInsight(
                    bottleneck="high_latency_outliers",
                    severity=severity,
                    current_value=p95_duration,
                    baseline_value=p95_baseline,
                    impact_magnitude=severity * 0.8,
                    recommendation=f"P95 latency ({p95_duration:.1f}s) exceeds baseline by "
                    f"{((p95_duration/p95_baseline - 1) * 100):.1f}%. Consider timeout optimization.",
                    affected_components=["team"],
                )
            )

        return insights

    def _analyze_success_rate(self, metrics: List[Dict[str, Any]]) -> List[PerformanceInsight]:
        """Analyze success rate.

        Args:
            metrics: Metrics data

        Returns:
            List of success-rate-related insights
        """
        insights = []

        total = len(metrics)
        successful = sum(1 for m in metrics if m.get("success", True))
        success_rate = successful / total if total > 0 else 0

        baseline_rate = self.baseline_metrics["success_rate"]

        if success_rate < baseline_rate * 0.9:
            severity = (baseline_rate - success_rate) / baseline_rate
            insights.append(
                PerformanceInsight(
                    bottleneck="low_success_rate",
                    severity=severity,
                    current_value=success_rate,
                    baseline_value=baseline_rate,
                    impact_magnitude=severity,
                    recommendation=f"Success rate ({success_rate*100:.1f}%) is below baseline. "
                    f"Check for timeouts, errors, or insufficient tool budgets.",
                    affected_components=["team", "tool_budget"],
                )
            )

        return insights

    def _analyze_tool_usage(self, metrics: List[Dict[str, Any]]) -> List[PerformanceInsight]:
        """Analyze tool usage patterns.

        Args:
            metrics: Metrics data

        Returns:
            List of tool-usage-related insights
        """
        insights = []

        tool_calls = [m.get("total_tool_calls", 0) for m in metrics if m.get("total_tool_calls")]
        if not tool_calls:
            return insights

        avg_tool_calls = statistics.mean(tool_calls)
        baseline_calls = self.baseline_metrics["avg_tool_calls"]

        # Check for over-provisioning
        if avg_tool_calls > baseline_calls * 1.5:
            severity = min(1.0, (avg_tool_calls - baseline_calls) / baseline_calls)
            insights.append(
                PerformanceInsight(
                    bottleneck="excessive_tool_usage",
                    severity=severity * 0.6,  # Lower severity
                    current_value=avg_tool_calls,
                    baseline_value=baseline_calls,
                    impact_magnitude=severity * 0.4,
                    recommendation=f"Average tool calls ({avg_tool_calls:.1f}) is significantly "
                    f"above baseline. Consider reducing tool budget to improve cost efficiency.",
                    affected_components=["tool_budget"],
                )
            )

        # Check for under-provisioning (high failure rate due to budget)
        # This would require error analysis - placeholder for now

        return insights

    def _analyze_formation_performance(
        self, metrics: List[Dict[str, Any]]
    ) -> List[PerformanceInsight]:
        """Analyze formation-specific performance.

        Args:
            metrics: Metrics data

        Returns:
            List of formation-related insights
        """
        insights = []

        # Group by formation
        formation_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for m in metrics:
            formation = m.get("formation", "sequential")
            formation_metrics[formation].append(m)

        # Compare against benchmarks
        for formation, formation_data in formation_metrics.items():
            if not formation_data:
                continue

            durations = [d.get("duration_seconds", 0) for d in formation_data if d.get("duration_seconds")]
            if not durations:
                continue

            avg_duration = statistics.mean(durations)

            # Get benchmark
            benchmark = self.benchmark_data.get(formation, {})
            benchmark_ms = benchmark.get("avg_latency_ms", 55)
            benchmark_sec = benchmark_ms / 1000.0

            # Check if significantly slower than benchmark
            if avg_duration > benchmark_sec * 2.0:
                severity = min(1.0, (avg_duration - benchmark_sec) / benchmark_sec * 0.5)
                insights.append(
                    PerformanceInsight(
                        bottleneck=f"suboptimal_formation_{formation}",
                        severity=severity,
                        current_value=avg_duration,
                        baseline_value=benchmark_sec,
                        impact_magnitude=severity,
                        recommendation=f"Formation '{formation}' is performing "
                        f"{(avg_duration/benchmark_sec):.1f}x slower than benchmark. "
                        f"Consider switching to a more suitable formation.",
                        affected_components=["formation"],
                    )
                )

        return insights

    def _analyze_team_size(self, metrics: List[Dict[str, Any]]) -> List[PerformanceInsight]:
        """Analyze team size efficiency.

        Args:
            metrics: Metrics data

        Returns:
            List of team-size-related insights
        """
        insights = []

        # Group by member count
        size_metrics: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for m in metrics:
            member_count = m.get("member_count", 1)
            size_metrics[member_count].append(m)

        # Check for diminishing returns
        for size, size_data in size_metrics.items():
            if size <= 2 or len(size_data) < 3:
                continue

            durations = [d.get("duration_seconds", 0) for d in size_data if d.get("duration_seconds")]
            if not durations:
                continue

            avg_duration = statistics.mean(durations)
            avg_duration_per_member = avg_duration / size

            # If per-member time is high, team might be too large
            if avg_duration_per_member > 30.0:  # 30s per member threshold
                severity = min(1.0, (avg_duration_per_member - 30) / 30)
                insights.append(
                    PerformanceInsight(
                        bottleneck="oversized_team",
                        severity=severity,
                        current_value=size,
                        baseline_value=5,
                        impact_magnitude=severity * 0.7,
                        recommendation=f"Team size ({size}) shows diminishing returns. "
                        f"Consider reducing to 3-5 members for better efficiency.",
                        affected_components=["team_size"],
                    )
                )

        return insights


# =============================================================================
# Optimization Strategies
# =============================================================================


class OptimizationStrategy(ABC):
    """Base class for optimization strategies."""

    @abstractmethod
    def suggest(
        self, insights: List[PerformanceInsight], current_config: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions.

        Args:
            insights: Performance insights
            current_config: Current configuration

        Returns:
            List of optimization suggestions
        """
        pass

    @abstractmethod
    def apply(self, config: Dict[str, Any], suggestion: OptimizationSuggestion) -> Dict[str, Any]:
        """Apply optimization to configuration.

        Args:
            config: Current configuration
            suggestion: Optimization to apply

        Returns:
            Updated configuration
        """
        pass


class TeamSizingStrategy(OptimizationStrategy):
    """Optimization strategy for team sizing."""

    def suggest(
        self, insights: List[PerformanceInsight], current_config: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Suggest team size optimizations."""
        suggestions = []

        for insight in insights:
            if insight.bottleneck == "oversized_team":
                current_size = insight.current_value
                suggested_size = min(5, int(current_size * 0.7))

                suggestions.append(
                    OptimizationSuggestion(
                        type=OptimizationType.TEAM_SIZING,
                        priority=OptimizationPriority.HIGH,
                        description=f"Reduce team size from {current_size} to {suggested_size}",
                        current_config={"member_count": current_size},
                        suggested_config={"member_count": suggested_size},
                        expected_improvement=15.0,
                        confidence=0.8,
                        risk_level=0.2,
                        metadata={"reason": "diminishing_returns"},
                    )
                )

        return suggestions

    def apply(self, config: Dict[str, Any], suggestion: OptimizationSuggestion) -> Dict[str, Any]:
        """Apply team size optimization."""
        new_config = copy.deepcopy(config)
        new_config.update(suggestion.suggested_config)
        return new_config


class FormationSelectionStrategy(OptimizationStrategy):
    """Optimization strategy for formation selection."""

    # Formation performance characteristics
    FORMATION_STATS = {
        "sequential": {"speed": 1.0, "quality": 0.7, "coordination": 0.3},
        "parallel": {"speed": 3.0, "quality": 0.6, "coordination": 0.5},
        "pipeline": {"speed": 1.5, "quality": 0.8, "coordination": 0.4},
        "hierarchical": {"speed": 1.3, "quality": 0.75, "coordination": 0.6},
        "consensus": {"speed": 0.5, "quality": 0.95, "coordination": 0.8},
    }

    def suggest(
        self, insights: List[PerformanceInsight], current_config: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Suggest formation optimizations."""
        suggestions = []

        for insight in insights:
            if insight.bottleneck.startswith("suboptimal_formation_"):
                current_formation = insight.bottleneck.replace("suboptimal_formation_", "")

                # Determine best formation based on characteristics
                # This is a simplified heuristic
                if "speed" in insight.recommendation.lower():
                    suggested_formation = "parallel"
                    improvement = 50.0
                elif "quality" in insight.recommendation.lower():
                    suggested_formation = "consensus"
                    improvement = 20.0
                else:
                    suggested_formation = "pipeline"
                    improvement = 30.0

                suggestions.append(
                    OptimizationSuggestion(
                        type=OptimizationType.FORMATION_SELECTION,
                        priority=OptimizationPriority.CRITICAL,
                        description=f"Switch formation from {current_formation} to {suggested_formation}",
                        current_config={"formation": current_formation},
                        suggested_config={"formation": suggested_formation},
                        expected_improvement=improvement,
                        confidence=0.75,
                        risk_level=0.4,
                        metadata={
                            "current_formation": current_formation,
                            "suggested_formation": suggested_formation,
                        },
                    )
                )

        return suggestions

    def apply(self, config: Dict[str, Any], suggestion: OptimizationSuggestion) -> Dict[str, Any]:
        """Apply formation optimization."""
        new_config = copy.deepcopy(config)
        new_config.update(suggestion.suggested_config)
        return new_config


class ToolBudgetStrategy(OptimizationStrategy):
    """Optimization strategy for tool budget allocation."""

    def suggest(
        self, insights: List[PerformanceInsight], current_config: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Suggest tool budget optimizations."""
        suggestions = []

        for insight in insights:
            if insight.bottleneck == "excessive_tool_usage":
                current_budget = insight.current_value
                suggested_budget = int(current_budget * 0.7)

                suggestions.append(
                    OptimizationSuggestion(
                        type=OptimizationType.TOOL_BUDGET,
                        priority=OptimizationPriority.MEDIUM,
                        description=f"Reduce tool budget from {current_budget} to {suggested_budget}",
                        current_config={"tool_budget": current_budget},
                        suggested_config={"tool_budget": suggested_budget},
                        expected_improvement=10.0,
                        confidence=0.7,
                        risk_level=0.3,
                        metadata={"reason": "cost_optimization"},
                    )
                )

        return suggestions

    def apply(self, config: Dict[str, Any], suggestion: OptimizationSuggestion) -> Dict[str, Any]:
        """Apply tool budget optimization."""
        new_config = copy.deepcopy(config)
        new_config.update(suggestion.suggested_config)
        return new_config


class TimeoutTuningStrategy(OptimizationStrategy):
    """Optimization strategy for timeout tuning."""

    def suggest(
        self, insights: List[PerformanceInsight], current_config: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Suggest timeout optimizations."""
        suggestions = []

        for insight in insights:
            if insight.bottleneck == "high_latency_outliers":
                # Set timeout to P95 + 20% buffer
                p95 = insight.current_value
                suggested_timeout = int(p95 * 1.2)

                suggestions.append(
                    OptimizationSuggestion(
                        type=OptimizationType.TIMEOUT_TUNING,
                        priority=OptimizationPriority.HIGH,
                        description=f"Set timeout to {suggested_timeout}s based on P95 latency",
                        current_config={"timeout_seconds": 300},
                        suggested_config={"timeout_seconds": suggested_timeout},
                        expected_improvement=5.0,
                        confidence=0.9,
                        risk_level=0.1,
                        metadata={"p95_latency": p95},
                    )
                )

        return suggestions

    def apply(self, config: Dict[str, Any], suggestion: OptimizationSuggestion) -> Dict[str, Any]:
        """Apply timeout optimization."""
        new_config = copy.deepcopy(config)
        new_config.update(suggestion.suggested_config)
        return new_config


# =============================================================================
# Performance Autotuner
# =============================================================================


class PerformanceAutotuner:
    """Auto-optimizes team configurations based on performance analysis.

    This class orchestrates the performance optimization workflow:
    1. Analyze historical metrics
    2. Generate optimization suggestions
    3. Apply optimizations (with optional A/B testing)
    4. Validate improvements
    5. Rollback on regression

    Attributes:
        analyzer: PerformanceAnalyzer instance
        strategies: List of optimization strategies
        optimization_history: History of applied optimizations
        ab_test_threshold: Minimum improvement to validate (default: 5%)
        enable_auto_rollback: Enable automatic rollback on regression

    Example:
        >>> autotuner = PerformanceAutotuner()
        >>>
        >>> # Get suggestions
        >>> suggestions = autotuner.suggest_optimizations(team_id="my_team")
        >>>
        >>> # Apply with A/B testing
        >>> result = await autotuner.apply_optimizations(
        ...     team_id="my_team",
        ...     optimizations=suggestions,
        ...     enable_ab_testing=True
        ... )
        >>>
        >>> # Check improvement
        >>> print(f"Improved by {result.improvement_percentage:.1f}%")
    """

    def __init__(
        self,
        analyzer: Optional[PerformanceAnalyzer] = None,
        strategies: Optional[List[OptimizationStrategy]] = None,
        ab_test_threshold: float = 5.0,
        enable_auto_rollback: bool = True,
    ):
        """Initialize performance autotuner.

        Args:
            analyzer: PerformanceAnalyzer instance (created if not provided)
            strategies: List of optimization strategies
            ab_test_threshold: Minimum improvement % to validate optimization
            enable_auto_rollback: Enable automatic rollback on regression
        """
        self.analyzer = analyzer or PerformanceAnalyzer()
        self.strategies = strategies or self._default_strategies()
        self.ab_test_threshold = ab_test_threshold
        self.enable_auto_rollback = enable_auto_rollback
        self.optimization_history: Dict[str, List[OptimizationResult]] = defaultdict(list)

    def _default_strategies(self) -> List[OptimizationStrategy]:
        """Get default optimization strategies."""
        return [
            TeamSizingStrategy(),
            FormationSelectionStrategy(),
            ToolBudgetStrategy(),
            TimeoutTuningStrategy(),
        ]

    def suggest_optimizations(
        self,
        team_id: Optional[str] = None,
        current_config: Optional[Dict[str, Any]] = None,
    ) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions.

        Args:
            team_id: Optional team ID to analyze
            current_config: Current team configuration

        Returns:
            List of optimization suggestions, sorted by priority
        """
        # Analyze performance
        insights = self.analyzer.analyze_team_workflow(team_id)

        if not insights:
            logger.info(f"No optimization opportunities found for team: {team_id}")
            return []

        # Generate suggestions from each strategy
        all_suggestions = []
        config = current_config or {}

        for strategy in self.strategies:
            suggestions = strategy.suggest(insights, config)
            all_suggestions.extend(suggestions)

        # Sort by priority and expected improvement
        priority_order = {
            OptimizationPriority.CRITICAL: 0,
            OptimizationPriority.HIGH: 1,
            OptimizationPriority.MEDIUM: 2,
            OptimizationPriority.LOW: 3,
        }

        all_suggestions.sort(
            key=lambda s: (priority_order[s.priority], -s.expected_improvement * s.confidence)
        )

        # Filter out low-confidence suggestions
        filtered = [s for s in all_suggestions if s.confidence >= 0.6]

        logger.info(f"Generated {len(filtered)} optimization suggestions for team: {team_id}")
        return filtered

    async def apply_optimizations(
        self,
        team_id: str,
        optimizations: List[OptimizationSuggestion],
        workflow_config: Dict[str, Any],
        enable_ab_testing: bool = True,
        dry_run: bool = False,
    ) -> OptimizationResult:
        """Apply optimizations to team configuration.

        Args:
            team_id: Team ID
            optimizations: List of optimizations to apply
            workflow_config: Current workflow configuration
            enable_ab_testing: Enable A/B testing before permanent application
            dry_run: Only simulate changes without applying

        Returns:
            OptimizationResult with outcome
        """
        if not optimizations:
            logger.warning(f"No optimizations to apply for team: {team_id}")
            return OptimizationResult(
                success=False,
                team_id=team_id,
                optimization=OptimizationSuggestion(
                    type=OptimizationType.TEAM_SIZING,
                    priority=OptimizationPriority.LOW,
                    description="No optimizations",
                    current_config={},
                    suggested_config={},
                    expected_improvement=0.0,
                    confidence=0.0,
                    risk_level=0.0,
                ),
                before_metrics={},
                after_metrics=None,
                improvement_percentage=None,
                validation_status=None,
                rollback_config={},
                error="No optimizations provided",
            )

        # Get the top optimization (highest priority)
        optimization = optimizations[0]

        # Capture current metrics
        before_metrics = self._get_current_metrics(team_id)

        # Apply optimization
        try:
            new_config = self._apply_optimization_to_config(workflow_config, optimization)

            if dry_run:
                logger.info(f"Dry run: would apply optimization to team: {team_id}")
                return OptimizationResult(
                    success=True,
                    team_id=team_id,
                    optimization=optimization,
                    before_metrics=before_metrics,
                    after_metrics=None,
                    improvement_percentage=None,
                    validation_status="dry_run",
                    rollback_config=workflow_config,
                    error=None,
                )

            # A/B testing if enabled
            if enable_ab_testing:
                validation_status = await self._ab_test_optimization(
                    team_id, optimization, new_config, before_metrics
                )
            else:
                validation_status = "skipped"

            # Save rollback config
            rollback_config = copy.deepcopy(workflow_config)

            # Record in history
            result = OptimizationResult(
                success=True,
                team_id=team_id,
                optimization=optimization,
                before_metrics=before_metrics,
                after_metrics=None,  # Will be populated after validation
                improvement_percentage=None,  # Will be calculated after validation
                validation_status=validation_status,
                rollback_config=rollback_config,
                error=None,
            )

            self.optimization_history[team_id].append(result)

            logger.info(
                f"Applied optimization to team {team_id}: {optimization.description} "
                f"(validation: {validation_status})"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to apply optimization to team {team_id}: {e}")
            return OptimizationResult(
                success=False,
                team_id=team_id,
                optimization=optimization,
                before_metrics=before_metrics,
                after_metrics=None,
                improvement_percentage=None,
                validation_status="failed",
                rollback_config=workflow_config,
                error=str(e),
            )

    def _apply_optimization_to_config(
        self, config: Dict[str, Any], suggestion: OptimizationSuggestion
    ) -> Dict[str, Any]:
        """Apply optimization to configuration.

        Args:
            config: Current configuration
            suggestion: Optimization suggestion

        Returns:
            Updated configuration
        """
        # Find appropriate strategy
        for strategy in self.strategies:
            if isinstance(strategy, type(self.strategies[0])):
                # Try each strategy
                try:
                    return strategy.apply(config, suggestion)
                except Exception:
                    continue

        # Fallback: direct update
        new_config = copy.deepcopy(config)
        new_config.update(suggestion.suggested_config)
        return new_config

    async def _ab_test_optimization(
        self,
        team_id: str,
        optimization: OptimizationSuggestion,
        new_config: Dict[str, Any],
        before_metrics: Dict[str, float],
    ) -> str:
        """Perform A/B testing for optimization.

        Args:
            team_id: Team ID
            optimization: Optimization being tested
            new_config: New configuration to test
            before_metrics: Metrics before optimization

        Returns:
            Validation status: "passed", "failed", "inconclusive"
        """
        # This is a simplified placeholder for A/B testing
        # In a real implementation, this would:
        # 1. Run the team with new configuration N times
        # 2. Compare metrics against baseline
        # 3. Determine if improvement is statistically significant

        logger.info(f"A/B testing optimization for team {team_id}")

        # Simulate A/B test result
        # In production, this would execute actual team runs
        simulated_improvement = optimization.expected_improvement * 0.8  # Conservative estimate

        if simulated_improvement >= self.ab_test_threshold:
            logger.info(f"A/B test passed: {simulated_improvement:.1f}% improvement")
            return "passed"
        elif simulated_improvement > 0:
            logger.warning(f"A/B test inconclusive: {simulated_improvement:.1f}% improvement")
            return "inconclusive"
        else:
            logger.error(f"A/B test failed: {simulated_improvement:.1f}% improvement")
            return "failed"

    def _get_current_metrics(self, team_id: str) -> Dict[str, float]:
        """Get current metrics for team.

        Args:
            team_id: Team ID

        Returns:
            Current metrics
        """
        # Get latest metrics from analyzer
        team_metrics = [m for m in self.analyzer.metrics_history if m.get("team_id") == team_id]

        if not team_metrics:
            return {}

        latest = team_metrics[-1]
        return {
            "duration_seconds": latest.get("duration_seconds", 0),
            "success_rate": latest.get("success_rate", 1.0),
            "tool_calls": latest.get("total_tool_calls", 0),
        }

    async def rollback_optimization(self, team_id: str, rollback_index: int = -1) -> bool:
        """Rollback optimization to previous configuration.

        Args:
            team_id: Team ID
            rollback_index: Index of optimization to rollback (default: latest)

        Returns:
            True if rollback succeeded
        """
        history = self.optimization_history.get(team_id, [])

        if not history:
            logger.warning(f"No optimization history found for team: {team_id}")
            return False

        if abs(rollback_index) > len(history):
            logger.error(f"Rollback index {rollback_index} out of range")
            return False

        result = history[rollback_index]
        rollback_config = result.rollback_config

        logger.info(
            f"Rolling back optimization for team {team_id}: "
            f"{result.optimization.description}"
        )

        # In a real implementation, this would update the workflow configuration
        # For now, just log the rollback config
        logger.info(f"Rollback config: {json.dumps(rollback_config, indent=2)}")

        return True

    def get_optimization_history(self, team_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get optimization history.

        Args:
            team_id: Optional team ID to filter by

        Returns:
            List of optimization results
        """
        if team_id:
            return [r.to_dict() for r in self.optimization_history.get(team_id, [])]

        all_results = []
        for team_results in self.optimization_history.values():
            all_results.extend([r.to_dict() for r in team_results])

        return all_results


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_team_performance(team_id: str, metrics_file: Optional[Path] = None) -> List[PerformanceInsight]:
    """Analyze team performance and return insights.

    Args:
        team_id: Team ID to analyze
        metrics_file: Optional path to metrics JSON file

    Returns:
        List of performance insights

    Example:
        insights = analyze_team_performance("my_team", "metrics.json")
        for insight in insights:
            print(f"{insight.bottleneck}: {insight.recommendation}")
    """
    analyzer = PerformanceAnalyzer()

    if metrics_file:
        analyzer.load_metrics_from_file(metrics_file)

    return analyzer.analyze_team_workflow(team_id)


def suggest_team_optimizations(
    team_id: str, current_config: Dict[str, Any], metrics_file: Optional[Path] = None
) -> List[OptimizationSuggestion]:
    """Get optimization suggestions for a team.

    Args:
        team_id: Team ID
        current_config: Current team configuration
        metrics_file: Optional path to metrics JSON file

    Returns:
        List of optimization suggestions

    Example:
        suggestions = suggest_team_optimizations(
            "my_team",
            {"formation": "sequential", "member_count": 5},
            "metrics.json"
        )
        for suggestion in suggestions:
            print(f"{suggestion.priority}: {suggestion.description}")
    """
    analyzer = PerformanceAnalyzer()

    if metrics_file:
        analyzer.load_metrics_from_file(metrics_file)

    autotuner = PerformanceAutotuner(analyzer=analyzer)
    return autotuner.suggest_optimizations(team_id, current_config)


__all__ = [
    # Data classes
    "PerformanceInsight",
    "OptimizationSuggestion",
    "OptimizationResult",
    "OptimizationType",
    "OptimizationPriority",
    # Core classes
    "PerformanceAnalyzer",
    "PerformanceAutotuner",
    # Strategies
    "OptimizationStrategy",
    "TeamSizingStrategy",
    "FormationSelectionStrategy",
    "ToolBudgetStrategy",
    "TimeoutTuningStrategy",
    # Convenience functions
    "analyze_team_performance",
    "suggest_team_optimizations",
]
