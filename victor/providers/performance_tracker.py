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

"""
Provider Performance Tracking for Adaptive Routing.

This module provides performance tracking and learning for smart routing:
- RequestMetric: Individual request metrics
- ProviderPerformanceTracker: Tracks success rate, latency, trends

Usage:
    from victor.providers.performance_tracker import ProviderPerformanceTracker, RequestMetric

    tracker = ProviderPerformanceTracker(window_size=100)

    # Record successful request
    tracker.record_request(
        RequestMetric(
            provider="ollama",
            model="qwen3-coder:30b",
            success=True,
            latency_ms=1250.0,
            timestamp=datetime.now(),
        )
    )

    # Get provider score for routing decisions
    score = tracker.get_provider_score("ollama")  # 0.0 to 1.0
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    """Individual request metric.

    Attributes:
        provider: Provider name (e.g., "ollama", "anthropic")
        model: Model identifier (e.g., "qwen3-coder:30b", "claude-3-5-sonnet-20241022")
        success: Whether the request succeeded
        latency_ms: Request latency in milliseconds
        timestamp: When the request occurred
        error_type: Type of error if request failed
    """

    provider: str
    model: str
    success: bool
    latency_ms: float
    timestamp: datetime
    error_type: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "model": self.model,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
        }


class ProviderPerformanceTracker:
    """Tracks provider performance for adaptive routing.

    Maintains a sliding window of recent requests for each provider and
    computes metrics for routing decisions:
    - Success rate: Percentage of successful requests
    - Average latency: Mean request latency
    - Latency trend: Improving, stable, or degrading

    Composite score = 40% success_rate + 30% (1/normalized_latency) + 30% trend_score
    """

    def __init__(self, window_size: int = 100):
        """Initialize performance tracker.

        Args:
            window_size: Number of recent requests to track per provider
        """
        self.window_size = window_size
        self.metrics: Dict[str, Deque[RequestMetric]] = {}
        logger.debug(f"ProviderPerformanceTracker initialized with window_size={window_size}")

    def record_request(self, metric: RequestMetric) -> None:
        """Record a request metric.

        Args:
            metric: Request metric to record
        """
        provider = metric.provider.lower()

        # Initialize deque if needed
        if provider not in self.metrics:
            self.metrics[provider] = deque(maxlen=self.window_size)

        # Add metric
        self.metrics[provider].append(metric)

        logger.debug(
            f"Recorded metric for {provider}: success={metric.success}, "
            f"latency={metric.latency_ms:.0f}ms"
        )

    def get_metrics(self, provider: str) -> List[RequestMetric]:
        """Get all metrics for a provider.

        Args:
            provider: Provider name

        Returns:
            List of request metrics
        """
        provider = provider.lower()
        return list(self.metrics.get(provider, []))

    def get_success_rate(self, provider: str) -> float:
        """Calculate success rate for a provider.

        Args:
            provider: Provider name

        Returns:
            Success rate from 0.0 to 1.0 (0% to 100%)
        """
        metrics = self.get_metrics(provider)

        if not metrics:
            return 0.5  # Unknown provider, neutral score

        successful = sum(1 for m in metrics if m.success)
        return successful / len(metrics)

    def get_average_latency(self, provider: str) -> float:
        """Calculate average latency for a provider.

        Args:
            provider: Provider name

        Returns:
            Average latency in milliseconds
        """
        metrics = self.get_metrics(provider)

        if not metrics:
            return 1000.0  # Unknown provider, default to 1s

        # Only consider successful requests for latency
        latencies = [m.latency_ms for m in metrics if m.success]

        if not latencies:
            return 1000.0

        return sum(latencies) / len(latencies)

    def get_latency_trend(self, provider: str) -> str:
        """Analyze latency trend for a provider.

        Compares recent latency to older latency to detect trends.

        Args:
            provider: Provider name

        Returns:
            Trend: "improving", "stable", or "degrading"
        """
        metrics = self.get_metrics(provider)

        if len(metrics) < 10:
            return "stable"  # Not enough data

        # Split into recent and older halves
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        mid = len(sorted_metrics) // 2

        older = sorted_metrics[:mid]
        recent = sorted_metrics[mid:]

        # Calculate average latencies
        older_latency = sum(m.latency_ms for m in older if m.success) / max(
            1, sum(1 for m in older if m.success)
        )
        recent_latency = sum(m.latency_ms for m in recent if m.success) / max(
            1, sum(1 for m in recent if m.success)
        )

        # Determine trend (10% threshold)
        if recent_latency < older_latency * 0.9:
            return "improving"
        elif recent_latency > older_latency * 1.1:
            return "degrading"
        else:
            return "stable"

    def get_provider_score(self, provider: str) -> float:
        """Calculate composite provider score for routing decisions.

        Composite score = 40% success_rate + 30% latency_score + 30% trend_score

        Args:
            provider: Provider name

        Returns:
            Score from 0.0 to 1.0 (higher is better)
        """
        # Get metrics
        success_rate = self.get_success_rate(provider)
        avg_latency = self.get_average_latency(provider)
        trend = self.get_latency_trend(provider)

        # Normalize latency (lower is better, cap at 5000ms)
        # Use inverse: 1.0 for 0ms, 0.0 for 5000ms+
        latency_score = max(0.0, 1.0 - (avg_latency / 5000.0))

        # Trend score
        trend_scores = {"improving": 1.0, "stable": 0.7, "degrading": 0.3}
        trend_score = trend_scores.get(trend, 0.7)

        # Composite score
        composite_score = 0.4 * success_rate + 0.3 * latency_score + 0.3 * trend_score

        logger.debug(
            f"Provider {provider} score: {composite_score:.2f} "
            f"(success={success_rate:.2f}, latency={latency_score:.2f}, trend={trend_score:.2f})"
        )

        return composite_score

    def get_all_scores(self) -> Dict[str, float]:
        """Get scores for all tracked providers.

        Returns:
            Dict mapping provider name to score
        """
        return {provider: self.get_provider_score(provider) for provider in self.metrics.keys()}

    def get_best_provider(self, candidates: List[str]) -> Optional[str]:
        """Get the best provider from a list of candidates.

        Args:
            candidates: List of provider names to consider

        Returns:
            Best provider name, or None if no candidates have data
        """
        if not candidates:
            return None

        scores = {provider: self.get_provider_score(provider) for provider in candidates}

        # Return provider with highest score
        return max(scores.items(), key=lambda x: x[1])[0] if scores else None

    def reset_provider(self, provider: str) -> None:
        """Reset all metrics for a provider.

        Args:
            provider: Provider name
        """
        provider = provider.lower()
        if provider in self.metrics:
            del self.metrics[provider]
            logger.debug(f"Reset metrics for provider {provider}")

    def reset_all(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        logger.debug("Reset all provider metrics")

    def get_stats(self) -> Dict[str, any]:
        """Get tracker statistics.

        Returns:
            Dict with tracker stats
        """
        provider_stats = {}
        for provider, metrics in self.metrics.items():
            provider_stats[provider] = {
                "total_requests": len(metrics),
                "success_rate": self.get_success_rate(provider),
                "average_latency_ms": self.get_average_latency(provider),
                "latency_trend": self.get_latency_trend(provider),
                "score": self.get_provider_score(provider),
            }

        return {
            "window_size": self.window_size,
            "providers_tracked": len(self.metrics),
            "provider_stats": provider_stats,
        }
