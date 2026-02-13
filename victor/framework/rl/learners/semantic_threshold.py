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

"""RL learner for optimal semantic similarity thresholds per embedding model:task:tool.

This learner determines the optimal similarity threshold for semantic code search
to balance precision and recall based on observed search quality metrics.

This is used by semantic search tools to dynamically adjust thresholds:
- code_search (hybrid keyword + semantic)
- semantic_code_search (pure semantic)

Strategy:
- If zero-result rate > 30% → Lower threshold (increase recall)
- If low-quality rate > 30% → Raise threshold (increase precision)
- Track per (embedding_model, task_type, tool_name) context
- Converge to optimal [0.1, 0.9] range

Migrated from: victor/codebase/semantic_threshold_learner.py
"""

import logging
import sqlite3
from typing import Optional

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


class SemanticThresholdLearner(BaseLearner):
    """Learn optimal semantic similarity thresholds per embedding_model:task:tool.

    Uses multi-armed bandit approach with weighted moving average to find
    optimal similarity thresholds for each context combination.

    Attributes:
        name: Always "semantic_threshold"
        db: SQLite database connection
        learning_rate: Adjustment rate (0.0-1.0)
    """

    def _ensure_tables(self) -> None:
        """Create tables for semantic threshold stats."""
        cursor = self.db.cursor()

        # Stats table: one row per embedding_model:task_type:tool_name
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {Tables.RL_SEMANTIC_STAT} (
                context_key TEXT PRIMARY KEY,
                embedding_model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                total_searches INTEGER DEFAULT 0,
                zero_result_count INTEGER DEFAULT 0,
                low_quality_count INTEGER DEFAULT 0,
                avg_results_count REAL DEFAULT 0.0,
                avg_threshold REAL DEFAULT 0.5,
                recommended_threshold REAL,
                results_sum REAL DEFAULT 0.0,
                threshold_sum REAL DEFAULT 0.0,
                last_updated TEXT
            )
            """)

        # Index for fast lookups
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_rl_semantic_stat_model
            ON {Tables.RL_SEMANTIC_STAT}(embedding_model, task_type, tool_name)
            """)

        self.db.commit()
        logger.debug("RL: semantic_threshold tables ensured")

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record semantic search outcome.

        Expected metadata:
        - embedding_model: Model used (e.g., "bge-small", "all-MiniLM-L12-v2")
        - tool_name: Tool name ("code_search", "semantic_code_search")
        - query: Search query
        - results_count: Number of results returned
        - threshold_used: Similarity threshold that was used
        - false_negatives: True if returned 0 results when shouldn't have
        - false_positives: True if returned irrelevant results

        Args:
            outcome: Outcome with semantic search data
        """
        # Extract metadata
        embedding_model = outcome.metadata.get("embedding_model", "unknown")
        tool_name = outcome.metadata.get("tool_name", "code_search")

        context_key = self._get_context_key(embedding_model, outcome.task_type, tool_name)

        cursor = self.db.cursor()

        # Get or create stats
        cursor.execute(
            f"SELECT * FROM {Tables.RL_SEMANTIC_STAT} WHERE context_key = ?",
            (context_key,),
        )
        row = cursor.fetchone()

        if row:
            # Update existing - convert row to dict using column names from cursor description
            column_names = [description[0] for description in cursor.description]
            stats = dict(zip(column_names, row))
        else:
            # Initialize new entry
            stats = {
                "context_key": context_key,
                "embedding_model": embedding_model,
                "task_type": outcome.task_type,
                "tool_name": tool_name,
                "total_searches": 0,
                "zero_result_count": 0,
                "low_quality_count": 0,
                "avg_results_count": 0.0,
                "avg_threshold": 0.5,
                "recommended_threshold": None,
                "results_sum": 0.0,
                "threshold_sum": 0.0,
                "last_updated": outcome.timestamp,
            }

        # Update counts
        stats["total_searches"] += 1

        results_count = outcome.metadata.get("results_count", 0)
        if results_count == 0:
            stats["zero_result_count"] += 1

        if outcome.metadata.get("false_positives", False):
            stats["low_quality_count"] += 1

        # Update weighted moving average (decay=0.9)
        decay = 0.9
        threshold_used = outcome.metadata.get("threshold_used", 0.5)

        stats["results_sum"] = decay * stats["results_sum"] + results_count
        stats["threshold_sum"] = decay * stats["threshold_sum"] + threshold_used

        stats["avg_results_count"] = stats["results_sum"] / (
            1 + (stats["total_searches"] - 1) * decay
        )
        stats["avg_threshold"] = stats["threshold_sum"] / (
            1 + (stats["total_searches"] - 1) * decay
        )

        stats["last_updated"] = outcome.timestamp

        # Update recommendation if enough data
        if stats["total_searches"] >= 5:
            self._update_threshold(stats)

        # Upsert to database
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_SEMANTIC_STAT}
            (context_key, embedding_model, task_type, tool_name, total_searches,
             zero_result_count, low_quality_count, avg_results_count, avg_threshold,
             recommended_threshold, results_sum, threshold_sum, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stats["context_key"],
                stats["embedding_model"],
                stats["task_type"],
                stats["tool_name"],
                stats["total_searches"],
                stats["zero_result_count"],
                stats["low_quality_count"],
                stats["avg_results_count"],
                stats["avg_threshold"],
                stats["recommended_threshold"],
                stats["results_sum"],
                stats["threshold_sum"],
                stats["last_updated"],
            ),
        )
        self.db.commit()

        logger.debug(
            f"RL: Recorded semantic_threshold outcome for {context_key} "
            f"(searches={stats['total_searches']}, "
            f"zero_rate={stats['zero_result_count']/stats['total_searches']:.1%}, "
            f"threshold={stats['avg_threshold']:.2f})"
        )

    def _update_threshold(self, stats: dict) -> None:
        """Update threshold recommendation based on stats.

        Strategy:
        1. High zero-result rate (>30%) → Lower threshold (increase recall)
        2. High low-quality rate (>30%) → Raise threshold (increase precision)
        3. Good balance → Minor adjustment toward optimal result count
        4. Bounds: [0.1, 0.9]

        Args:
            stats: Statistics dictionary
        """
        total = stats["total_searches"]
        zero_rate = stats["zero_result_count"] / total if total > 0 else 0.0
        low_quality_rate = stats["low_quality_count"] / total if total > 0 else 0.0
        avg_results = stats["avg_results_count"]
        current = stats["avg_threshold"]

        adjustment = 0.0

        # High false negative rate → Lower threshold (increase recall)
        if zero_rate > 0.3:
            adjustment = -0.1 if zero_rate > 0.5 else -0.05
            logger.debug(
                f"RL: High zero-result rate ({zero_rate:.1%}) for {stats['context_key']} → "
                f"lower threshold by {abs(adjustment):.2f}"
            )

        # High false positive rate → Raise threshold (increase precision)
        elif low_quality_rate > 0.3:
            adjustment = 0.1 if low_quality_rate > 0.5 else 0.05
            logger.debug(
                f"RL: High low-quality rate ({low_quality_rate:.1%}) for {stats['context_key']} → "
                f"raise threshold by {adjustment:.2f}"
            )

        # Good balance → Minor adjustment toward optimal result count
        elif zero_rate < 0.1 and low_quality_rate < 0.1:
            if avg_results < 3:
                # Too few results on average → lower threshold slightly
                adjustment = -0.02
                logger.debug(
                    f"RL: Low avg results ({avg_results:.1f}) for {stats['context_key']} → "
                    f"lower threshold by {abs(adjustment):.2f}"
                )
            elif avg_results > 20:
                # Too many results on average → raise threshold slightly
                adjustment = 0.02
                logger.debug(
                    f"RL: High avg results ({avg_results:.1f}) for {stats['context_key']} → "
                    f"raise threshold by {adjustment:.2f}"
                )

        # Apply adjustment with bounds [0.1, 0.9]
        recommended = current + adjustment
        recommended = max(0.1, min(0.9, recommended))

        stats["recommended_threshold"] = recommended

        if abs(recommended - current) > 0.01:
            logger.info(
                f"RL: Updated semantic_threshold for {stats['context_key']}: "
                f"{current:.2f} → {recommended:.2f} "
                f"(zero_rate={zero_rate:.1%}, low_quality_rate={low_quality_rate:.1%})"
            )

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommended semantic similarity threshold.

        Note: For semantic threshold learner, provider/model params are
        actually embedding_model/tool_name, but we maintain signature
        compatibility with BaseLearner.

        Args:
            provider: Embedding model name (e.g., "all-MiniLM-L12-v2")
            model: Tool name (e.g., "code_search", "semantic_code_search")
            task_type: Task type

        Returns:
            Recommendation with threshold value and confidence, or None
        """
        embedding_model = provider
        tool_name = model

        context_key = self._get_context_key(embedding_model, task_type, tool_name)

        cursor = self.db.cursor()
        cursor.execute(
            f"SELECT * FROM {Tables.RL_SEMANTIC_STAT} WHERE context_key = ?",
            (context_key,),
        )
        row = cursor.fetchone()

        if not row:
            # No data yet - return None (caller will use default)
            return None

        # Convert row tuple to dict using column names from cursor description
        column_names = [description[0] for description in cursor.description]
        stats = dict(zip(column_names, row))

        # Need at least 5 searches for confidence
        if stats["total_searches"] < 5:
            return None

        # Calculate confidence based on sample size
        confidence = min(1.0, stats["total_searches"] / 30.0)

        # Reduce confidence if high false positive/negative rate
        zero_rate = stats["zero_result_count"] / stats["total_searches"]
        low_quality_rate = stats["low_quality_count"] / stats["total_searches"]

        if zero_rate > 0.3 or low_quality_rate > 0.3:
            confidence *= 0.7

        recommended = stats["recommended_threshold"] or stats["avg_threshold"]

        return RLRecommendation(
            value=recommended,
            confidence=confidence,
            reason=f"Learned from {stats['total_searches']} searches "
            f"(zero_rate={zero_rate:.1%}, low_quality_rate={low_quality_rate:.1%})",
            sample_size=stats["total_searches"],
            is_baseline=False,
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Reward based on:
        - Non-zero results: +0.5
        - Quality score: weighted component
        - False negatives: -0.5
        - False positives: -0.3
        - Result count in optimal range [3, 20]: +0.2

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        reward = 0.0

        # Non-zero results bonus
        results_count = outcome.metadata.get("results_count", 0)
        if results_count > 0:
            reward += 0.5

        # Quality component (weight 40%)
        reward += 0.4 * (outcome.quality_score * 2 - 1)

        # Penalties
        if outcome.metadata.get("false_negatives", False):
            reward -= 0.5

        if outcome.metadata.get("false_positives", False):
            reward -= 0.3

        # Optimal result count bonus
        if 3 <= results_count <= 20:
            reward += 0.2

        return max(-1.0, min(1.0, reward))
