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

Persistence:
    Metrics are written through to the global database (~/.victor/victor.db)
    in the rl_provider_stat table so that routing decisions benefit from
    cross-session history. On initialization the tracker hydrates its
    in-memory window from the last `window_size` rows per provider.

Usage:
    from victor.providers.performance_tracker import ProviderPerformanceTracker, RequestMetric

    tracker = ProviderPerformanceTracker(window_size=100)

    # Record successful request (persisted to global DB)
    tracker.record_request(
        RequestMetric(
            provider="ollama",
            model="qwen3-coder:30b",
            success=True,
            latency_ms=1250.0,
            timestamp=datetime.now(),
        )
    )

    # Get provider score for routing decisions (uses cross-session history)
    score = tracker.get_provider_score("ollama")  # 0.0 to 1.0
"""

from __future__ import annotations

import logging
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.core.database import DatabaseManager

logger = logging.getLogger(__name__)
_AUTO_DB = object()


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
        task_type: Task type hint for per-task routing analytics
    """

    provider: str
    model: str
    success: bool
    latency_ms: float
    timestamp: datetime
    error_type: Optional[str] = None
    task_type: str = "default"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "model": self.model,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "task_type": self.task_type,
        }


class ProviderPerformanceTracker:
    """Tracks provider performance for adaptive routing.

    Maintains a sliding window of recent requests for each provider and
    computes metrics for routing decisions:
    - Success rate: Percentage of successful requests
    - Average latency: Mean request latency
    - Latency trend: Improving, stable, or degrading

    Composite score = 40% success_rate + 30% (1/normalized_latency) + 30% trend_score

    Persistence:
        Writes through to rl_provider_stat in the global DB on every
        record_request() call.  On first access to a provider's metrics,
        hydrates the in-memory deque from the DB if it is empty (cold start).
        This means routing benefits from history across restarts.
    """

    def __init__(
        self,
        window_size: int = 100,
        db: Optional["DatabaseManager"] | object = _AUTO_DB,
    ):
        """Initialize performance tracker.

        Args:
            window_size: Number of recent requests to track per provider.
            db: DatabaseManager to persist metrics. When omitted, the tracker
                uses the global singleton (``DatabaseManager()``). Pass
                ``None`` to disable persistence and keep all metrics in-memory,
                which is useful for deterministic unit tests and session-local
                routing modes.
        """
        self.window_size = window_size
        self.metrics: Dict[str, Deque[RequestMetric]] = {}
        # Track which providers have been hydrated from DB this session
        self._hydrated: set[str] = set()

        if db is _AUTO_DB:
            try:
                from victor.core.database import DatabaseManager

                self._db: Optional["DatabaseManager"] = DatabaseManager()
                self._ensure_table()
            except Exception:
                logger.debug("Global DB unavailable; performance tracker will be in-memory only")
                self._db = None
        elif db is None:
            self._db = None
        else:
            self._db = db
            if self._db is not None:
                self._ensure_table()

        logger.debug(
            "ProviderPerformanceTracker initialized (window_size=%d, db=%s)",
            window_size,
            "enabled" if self._db else "disabled",
        )

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create rl_provider_stat table if it doesn't exist."""
        if self._db is None:
            return
        try:
            from victor.core.schema import Schema

            conn = self._db.get_connection()
            conn.executescript(Schema.RL_PROVIDER_STAT)
            for stmt in Schema.RL_PROVIDER_STAT_INDEXES.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.commit()
        except Exception as exc:
            logger.warning("Failed to ensure rl_provider_stat table: %s", exc)
            self._db = None  # Degrade gracefully to in-memory

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist(self, metric: RequestMetric) -> None:
        """Write a metric to the global DB (fire-and-forget, non-fatal)."""
        if self._db is None:
            return
        try:
            self._db.execute(
                """
                INSERT INTO rl_provider_stat
                    (provider, model, task_type, success, latency_ms, error_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.provider.lower(),
                    metric.model,
                    metric.task_type,
                    1 if metric.success else 0,
                    metric.latency_ms,
                    metric.error_type,
                    metric.timestamp.isoformat(),
                ),
            )
            # Trim old rows for this provider beyond 10× window_size to bound table growth.
            # We keep 10× so that multiple concurrent sessions don't see an artificially small
            # pool when they each hold a separate in-memory window.
            self._db.execute(
                """
                DELETE FROM rl_provider_stat
                WHERE provider = ?
                  AND id NOT IN (
                      SELECT id FROM rl_provider_stat
                      WHERE provider = ?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (metric.provider.lower(), metric.provider.lower(), self.window_size * 10),
            )
        except Exception as exc:
            logger.debug("rl_provider_stat write failed (non-fatal): %s", exc)

    def _hydrate(self, provider: str) -> None:
        """Populate the in-memory deque for a provider from the DB (once per session)."""
        if self._db is None or provider in self._hydrated:
            return
        self._hydrated.add(provider)
        try:
            rows = self._db.query(
                """
                SELECT provider, model, task_type, success, latency_ms, error_type, created_at
                FROM rl_provider_stat
                WHERE provider = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (provider, self.window_size),
            )
            if not rows:
                return
            if provider not in self.metrics:
                self.metrics[provider] = deque(maxlen=self.window_size)
            deq = self.metrics[provider]
            # Rows come back newest-first; insert oldest-first so the deque is in
            # chronological order for trend analysis.
            for row in reversed(rows):
                try:
                    ts = datetime.fromisoformat(row["created_at"])
                except (ValueError, TypeError):
                    ts = datetime.now()
                deq.append(
                    RequestMetric(
                        provider=row["provider"],
                        model=row["model"],
                        task_type=row["task_type"] or "default",
                        success=bool(row["success"]),
                        latency_ms=float(row["latency_ms"]),
                        error_type=row["error_type"],
                        timestamp=ts,
                    )
                )
            logger.debug(
                "Hydrated %d historical metrics for provider '%s' from DB",
                len(rows),
                provider,
            )
        except Exception as exc:
            logger.debug("rl_provider_stat hydration failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_request(self, metric: RequestMetric) -> None:
        """Record a request metric.

        Args:
            metric: Request metric to record
        """
        provider = metric.provider.lower()

        if provider not in self.metrics:
            self.metrics[provider] = deque(maxlen=self.window_size)

        self.metrics[provider].append(metric)
        self._persist(metric)

        logger.debug(
            "Recorded metric for %s: success=%s, latency=%.0fms",
            provider,
            metric.success,
            metric.latency_ms,
        )

    def get_metrics(self, provider: str) -> List[RequestMetric]:
        """Get all metrics for a provider.

        Hydrates from the global DB on first access if the in-memory window
        is empty (cold start after a restart).

        Args:
            provider: Provider name

        Returns:
            List of request metrics
        """
        provider = provider.lower()
        self._hydrate(provider)
        return list(self.metrics.get(provider, []))

    def get_success_rate(self, provider: str) -> float:
        """Calculate success rate for a provider.

        Args:
            provider: Provider name

        Returns:
            Success rate from 0.0 to 1.0 (0.5 = unknown)
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

        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        mid = len(sorted_metrics) // 2

        older = sorted_metrics[:mid]
        recent = sorted_metrics[mid:]

        older_latency = sum(m.latency_ms for m in older if m.success) / max(
            1, sum(1 for m in older if m.success)
        )
        recent_latency = sum(m.latency_ms for m in recent if m.success) / max(
            1, sum(1 for m in recent if m.success)
        )

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
        success_rate = self.get_success_rate(provider)
        avg_latency = self.get_average_latency(provider)
        trend = self.get_latency_trend(provider)

        # Normalize latency: 1.0 for 0ms, 0.0 for 5000ms+
        latency_score = max(0.0, 1.0 - (avg_latency / 5000.0))

        trend_scores = {"improving": 1.0, "stable": 0.7, "degrading": 0.3}
        trend_score = trend_scores.get(trend, 0.7)

        composite_score = 0.4 * success_rate + 0.3 * latency_score + 0.3 * trend_score

        logger.debug(
            "Provider %s score: %.2f (success=%.2f, latency=%.2f, trend=%.2f)",
            provider,
            composite_score,
            success_rate,
            latency_score,
            trend_score,
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
        return max(scores.items(), key=lambda x: x[1])[0] if scores else None

    def reset_provider(self, provider: str) -> None:
        """Reset in-memory metrics for a provider (does not touch DB).

        Args:
            provider: Provider name
        """
        provider = provider.lower()
        if provider in self.metrics:
            del self.metrics[provider]
        self._hydrated.discard(provider)
        logger.debug("Reset in-memory metrics for provider %s", provider)

    def reset_all(self) -> None:
        """Reset all in-memory metrics (does not touch DB)."""
        self.metrics.clear()
        self._hydrated.clear()
        logger.debug("Reset all in-memory provider metrics")

    def get_stats(self) -> Dict[str, object]:
        """Get tracker statistics.

        Returns:
            Dict with tracker stats
        """
        provider_stats = {}
        for provider in list(self.metrics.keys()):
            provider_stats[provider] = {
                "total_requests": len(self.metrics[provider]),
                "success_rate": self.get_success_rate(provider),
                "average_latency_ms": self.get_average_latency(provider),
                "latency_trend": self.get_latency_trend(provider),
                "score": self.get_provider_score(provider),
            }

        return {
            "window_size": self.window_size,
            "providers_tracked": len(self.metrics),
            "db_enabled": self._db is not None,
            "provider_stats": provider_stats,
        }
