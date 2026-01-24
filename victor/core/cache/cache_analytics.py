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

"""Cache analytics and monitoring system.

This module provides comprehensive monitoring and analytics for cache systems:
- Hit rate, miss rate, eviction rate tracking
- Hot key identification
- Cache size optimization recommendations
- Performance metrics (latency by cache level)
- Export to Prometheus metrics
- Dashboard visualizations
- Alerts for cache performance degradation

Benefits:
- Actionable insights for cache optimization
- Real-time performance monitoring
- Proactive alerting on issues
- Data-driven tuning decisions
- Comprehensive observability

Usage:
    analytics = CacheAnalytics(multi_level_cache)

    # Get comprehensive statistics
    stats = analytics.get_comprehensive_stats()

    # Identify hot keys
    hot_keys = analytics.get_hot_keys(top_n=100)

    # Get optimization recommendations
    recommendations = analytics.get_recommendations()

    # Export metrics
    await analytics.export_to_prometheus()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics Data Classes
# =============================================================================


@dataclass
class CacheMetrics:
    """Comprehensive cache metrics.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of cache evictions
        size: Current cache size
        max_size: Maximum cache size
        hit_rate: Cache hit rate (0-1)
        miss_rate: Cache miss rate (0-1)
        avg_latency_ms: Average access latency in milliseconds
        p95_latency_ms: 95th percentile latency
        p99_latency_ms: 99th percentile latency
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


@dataclass
class HotKey:
    """A hot cache key with access statistics.

    Attributes:
        key: Cache key
        namespace: Namespace
        access_count: Number of accesses
        hit_count: Number of hits
        miss_count: Number of misses
        last_access: Last access timestamp
        avg_latency_ms: Average access latency
    """

    key: str
    namespace: str
    access_count: int
    hit_count: int
    miss_count: int
    last_access: float
    avg_latency_ms: float


@dataclass
class Recommendation:
    """A cache optimization recommendation.

    Attributes:
        type: Type of recommendation (size, ttl, warming, etc.)
        priority: Priority level (low, medium, high, critical)
        title: Short title
        description: Detailed description
        expected_impact: Expected impact if implemented
        action: Suggested action
    """

    type: str
    priority: str
    title: str
    description: str
    expected_impact: str
    action: str


# =============================================================================
# Cache Analytics
# =============================================================================


class CacheAnalytics:
    """Comprehensive cache analytics and monitoring.

    Tracks:
    - Hit/miss/eviction rates
    - Access latency distribution
    - Hot key identification
    - Time-based patterns
    - Size utilization trends
    - Performance degradation alerts

    Features:
    - Real-time metrics collection
    - Hot key tracking
    - Optimization recommendations
    - Prometheus export
    - Alert generation
    - Historical data analysis

    Example:
        ```python
        analytics = CacheAnalytics(
            cache=multi_level_cache,
            track_hot_keys=True,
            hot_key_window=1000,
        )

        # Record access
        analytics.record_access("key1", "tool", hit=True, latency_ms=0.5)

        # Get statistics
        stats = analytics.get_comprehensive_stats()

        # Get hot keys
        hot_keys = analytics.get_hot_keys(top_n=100)

        # Get recommendations
        recommendations = analytics.get_recommendations()
        ```
    """

    def __init__(
        self,
        cache: Any,  # MultiLevelCache
        track_hot_keys: bool = True,
        hot_key_window: int = 1000,
        metrics_retention_hours: int = 24,
    ):
        """Initialize cache analytics.

        Args:
            cache: MultiLevelCache instance to monitor
            track_hot_keys: Enable hot key tracking
            hot_key_window: Number of recent accesses to track for hot keys
            metrics_retention_hours: How long to keep historical metrics
        """
        self.cache = cache
        self.track_hot_keys = track_hot_keys
        self.hot_key_window = hot_key_window
        self.metrics_retention_hours = metrics_retention_hours

        # Access tracking
        self._access_log: deque[Tuple[str, str, bool, float]] = deque(maxlen=hot_key_window)
        self._latency_samples: deque[float] = deque(maxlen=10000)

        # Hot key tracking
        self._key_stats: Dict[tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "access_count": 0,
                "hit_count": 0,
                "miss_count": 0,
                "last_access": 0.0,
                "latencies": [],
            }
        )

        # Time-based metrics
        self._hourly_metrics: Dict[int, CacheMetrics] = {}

        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        # Thread safety
        self._lock = threading.RLock()

        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        logger.info(
            f"Initialized cache analytics: hot_keys={track_hot_keys}, " f"window={hot_key_window}"
        )

    def record_access(
        self,
        key: str,
        namespace: str,
        hit: bool,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a cache access for analytics.

        Args:
            key: Cache key that was accessed
            namespace: Namespace
            hit: Whether this was a cache hit
            latency_ms: Access latency in milliseconds
        """
        with self._lock:
            # Log access
            self._access_log.append((key, namespace, hit, latency_ms))
            self._latency_samples.append(latency_ms)

            # Update key stats
            if self.track_hot_keys:
                key_tuple = (key, namespace)
                stats = self._key_stats[key_tuple]
                stats["access_count"] += 1
                stats["last_access"] = time.time()
                if hit:
                    stats["hit_count"] += 1
                else:
                    stats["miss_count"] += 1

                # Keep limited latency history per key
                stats["latencies"].append(latency_ms)
                if len(stats["latencies"]) > 100:
                    stats["latencies"].pop(0)

    def get_hit_rate(self) -> float:
        """Calculate current cache hit rate.

        Returns:
            Hit rate as a float between 0 and 1
        """
        with self._lock:
            if not self._access_log:
                return 0.0

            total = len(self._access_log)
            hits = sum(1 for _, _, hit, _ in self._access_log if hit)
            return hits / total

    def get_miss_rate(self) -> float:
        """Calculate current cache miss rate.

        Returns:
            Miss rate as a float between 0 and 1
        """
        return 1.0 - self.get_hit_rate()

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics.

        Returns:
            Dictionary with latency metrics (avg, p50, p95, p99)
        """
        with self._lock:
            if not self._latency_samples:
                return {
                    "avg_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                }

            samples = list(self._latency_samples)
            samples.sort()

            return {
                "avg_ms": sum(samples) / len(samples),
                "p50_ms": samples[len(samples) // 2],
                "p95_ms": samples[int(len(samples) * 0.95)],
                "p99_ms": samples[int(len(samples) * 0.99)],
            }

    def get_hot_keys(self, top_n: int = 100) -> List[HotKey]:
        """Get most frequently accessed keys.

        Args:
            top_n: Number of top keys to return

        Returns:
            List of HotKey objects sorted by access count
        """
        if not self.track_hot_keys:
            return []

        with self._lock:
            hot_keys = []

            for (key, namespace), stats in self._key_stats.items():
                # Calculate average latency
                latencies = stats["latencies"]
                avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

                hot_keys.append(
                    HotKey(
                        key=key,
                        namespace=namespace,
                        access_count=stats["access_count"],
                        hit_count=stats["hit_count"],
                        miss_count=stats["miss_count"],
                        last_access=stats["last_access"],
                        avg_latency_ms=avg_latency,
                    )
                )

            # Sort by access count
            hot_keys.sort(key=lambda k: k.access_count, reverse=True)
            return hot_keys[:top_n]

    def get_eviction_rate(self) -> float:
        """Calculate cache eviction rate.

        Returns:
            Eviction rate as a float between 0 and 1
        """
        stats = self.cache.get_stats()
        l1_evictions = stats.get("l1", {}).get("evictions", 0)
        l2_evictions = stats.get("l2", {}).get("evictions", 0)
        total_requests = stats.get("l1", {}).get("hits", 0) + stats.get("l1", {}).get("misses", 0)

        if total_requests == 0:
            return 0.0

        return (l1_evictions + l2_evictions) / total_requests

    def get_size_utilization(self) -> Dict[str, float]:
        """Get cache size utilization.

        Returns:
            Dictionary with utilization ratios per cache level
        """
        stats = self.cache.get_stats()

        return {
            "l1_utilization": stats.get("l1", {}).get("size", 0)
            / max(stats.get("l1", {}).get("max_size", 1), 1),
            "l2_utilization": stats.get("l2", {}).get("size", 0)
            / max(stats.get("l2", {}).get("max_size", 1), 1),
        }

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with all cache analytics
        """
        cache_stats = self.cache.get_stats()

        return {
            "hit_rate": self.get_hit_rate(),
            "miss_rate": self.get_miss_rate(),
            "eviction_rate": self.get_eviction_rate(),
            "latency": self.get_latency_stats(),
            "size_utilization": self.get_size_utilization(),
            "hot_key_count": len(self._key_stats),
            "total_accesses": len(self._access_log),
            "cache_stats": cache_stats,
        }

    def get_recommendations(self) -> List[Recommendation]:
        """Get cache optimization recommendations.

        Analyzes metrics and generates actionable recommendations.

        Returns:
            List of Recommendation objects
        """
        recommendations = []

        stats = self.get_comprehensive_stats()
        hit_rate = stats["hit_rate"]
        eviction_rate = self.get_eviction_rate()
        size_util = stats["size_utilization"]
        latency = stats["latency"]

        # Low hit rate recommendations
        if hit_rate < 0.3:
            recommendations.append(
                Recommendation(
                    type="hit_rate",
                    priority="high",
                    title="Low cache hit rate detected",
                    description=f"Current hit rate is {hit_rate:.1%}, which is below optimal (target: >50%). This indicates cache entries are not being reused effectively.",
                    expected_impact="10-30% reduction in API calls and latency",
                    action="Consider: 1) Increasing cache TTL, 2) Enabling cache warming, 3) Adjusting cache keys for better reuse",
                )
            )

        # High eviction rate recommendations
        if eviction_rate > 0.1:
            recommendations.append(
                Recommendation(
                    type="eviction",
                    priority="medium",
                    title="High cache eviction rate",
                    description=f"Current eviction rate is {eviction_rate:.1%}, indicating cache size may be insufficient or entries are too short-lived.",
                    expected_impact="5-15% improvement in hit rate",
                    action="Consider: 1) Increasing cache size, 2) Increasing TTL, 3) Using multi-level cache with larger L2",
                )
            )

        # High size utilization recommendations
        l1_util = size_util.get("l1_utilization", 0)
        if l1_util > 0.9:
            recommendations.append(
                Recommendation(
                    type="size",
                    priority="high",
                    title="L1 cache near capacity",
                    description=f"L1 cache is {l1_util:.1%} full, causing frequent evictions and reduced performance.",
                    expected_impact="20-40% reduction in L1 latency",
                    action="Consider: 1) Increasing L1 cache size, 2) Enabling write-back policy, 3) Reducing entry sizes",
                )
            )

        # High latency recommendations
        avg_latency = latency.get("avg_ms", 0)
        if avg_latency > 5.0:
            recommendations.append(
                Recommendation(
                    type="latency",
                    priority="medium",
                    title="High cache access latency",
                    description=f"Average cache access latency is {avg_latency:.2f}ms, which is higher than optimal (<1ms for L1, <5ms for L2).",
                    expected_impact="50-70% reduction in cache latency",
                    action="Consider: 1) Using in-memory L1 cache, 2) Optimizing serialization, 3) Reducing cache entry sizes",
                )
            )

        # Cache warming recommendations
        if hit_rate < 0.5 and len(self._key_stats) > 100:
            recommendations.append(
                Recommendation(
                    type="warming",
                    priority="low",
                    title="Consider cache warming",
                    description="Cache has many keys but low hit rate, suggesting cold starts are impacting performance.",
                    expected_impact="10-20% improvement in hit rate",
                    action="Enable cache warming to preload frequently accessed items on startup",
                )
            )

        return recommendations

    def register_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register an alert callback for performance degradation.

        Args:
            callback: Function to call when alert is triggered
                     Receives (alert_type, metrics) arguments
        """
        self._alert_callbacks.append(callback)

    async def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start background monitoring task.

        Periodically checks metrics and triggers alerts if needed.

        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self._monitoring_task is not None:
            logger.warning("Monitoring already running")
            return

        async def monitoring_loop():
            """Background monitoring loop."""
            while not self._stop_event.is_set():
                try:
                    # Get current metrics
                    stats = self.get_comprehensive_stats()

                    # Check for performance issues
                    alerts = self._check_performance_issues(stats)

                    # Trigger alerts
                    for alert_type, alert_data in alerts:
                        for callback in self._alert_callbacks:
                            try:
                                callback(alert_type, alert_data)
                            except Exception as e:
                                logger.error(f"Error in alert callback: {e}")

                    # Wait for next interval
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=interval_seconds,
                    )
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    pass
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")

        self._monitoring_task = asyncio.create_task(monitoring_loop())
        logger.info(f"Started cache monitoring (interval: {interval_seconds}s)")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring task."""
        if self._monitoring_task is None:
            return

        self._stop_event.set()
        self._monitoring_task.cancel()
        try:
            await self._monitoring_task
        except asyncio.CancelledError:
            pass

        self._monitoring_task = None
        self._stop_event.clear()
        logger.info("Stopped cache monitoring")

    def _check_performance_issues(self, stats: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Check for performance issues and generate alerts.

        Args:
            stats: Cache statistics

        Returns:
            List of (alert_type, alert_data) tuples
        """
        alerts = []

        # Low hit rate alert
        if stats["hit_rate"] < 0.3:
            alerts.append(
                (
                    "low_hit_rate",
                    {
                        "hit_rate": stats["hit_rate"],
                        "severity": "warning" if stats["hit_rate"] > 0.2 else "critical",
                        "message": f"Cache hit rate is {stats['hit_rate']:.1%} (threshold: 30%)",
                    },
                )
            )

        # High latency alert
        avg_latency = stats["latency"].get("avg_ms", 0)
        if avg_latency > 10.0:
            alerts.append(
                (
                    "high_latency",
                    {
                        "avg_latency_ms": avg_latency,
                        "severity": "warning" if avg_latency < 20 else "critical",
                        "message": f"Cache latency is {avg_latency:.2f}ms (threshold: 10ms)",
                    },
                )
            )

        # High eviction rate alert
        if stats["eviction_rate"] > 0.2:
            alerts.append(
                (
                    "high_eviction_rate",
                    {
                        "eviction_rate": stats["eviction_rate"],
                        "severity": "warning",
                        "message": f"Eviction rate is {stats['eviction_rate']:.1%} (threshold: 20%)",
                    },
                )
            )

        return alerts

    async def export_to_prometheus(self) -> Dict[str, float]:
        """Export metrics in Prometheus format.

        Returns:
            Dictionary of metric names to values
        """
        stats = self.get_comprehensive_stats()

        metrics = {
            "cache_hit_rate": stats["hit_rate"],
            "cache_miss_rate": stats["miss_rate"],
            "cache_eviction_rate": stats["eviction_rate"],
            "cache_latency_avg_ms": stats["latency"]["avg_ms"],
            "cache_latency_p95_ms": stats["latency"]["p95_ms"],
            "cache_latency_p99_ms": stats["latency"]["p99_ms"],
            "cache_l1_utilization": stats["size_utilization"]["l1_utilization"],
            "cache_l2_utilization": stats["size_utilization"]["l2_utilization"],
            "cache_total_accesses": stats["total_accesses"],
            "cache_hot_keys": stats["hot_key_count"],
            "cache_l1_size": stats["cache_stats"]["l1"]["size"],
            "cache_l2_size": stats["cache_stats"]["l2"]["size"],
        }

        logger.debug(f"Exported {len(metrics)} Prometheus metrics")
        return metrics

    def reset_metrics(self) -> None:
        """Reset all analytics metrics."""
        with self._lock:
            self._access_log.clear()
            self._latency_samples.clear()
            self._key_stats.clear()
            self._hourly_metrics.clear()

        logger.info("Reset cache analytics metrics")


__all__ = [
    "CacheAnalytics",
    "CacheMetrics",
    "HotKey",
    "Recommendation",
]
