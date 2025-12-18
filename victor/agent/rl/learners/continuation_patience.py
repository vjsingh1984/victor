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

"""RL learner for optimal continuation patience per provider:model:task.

Learns the optimal `continuation_patience` value - how many prompts without
tool calls before flagging as stuck loop.

Currently, this is a static value in provider adapters:
- DeepSeek: continuation_patience=5
- Claude: continuation_patience=3

This learner refines these baselines based on actual stuck loop outcomes.

Strategy:
- Track false_positive_rate: Model flagged as stuck but eventually made progress
- Track missed_stuck_loops: Model continued without making progress
- If FP rate > 20% → Increase patience (too eager to flag)
- If missed loops > 10% → Decrease patience (too lenient)
- Converge to optimal [1, 15] range
"""

import logging
from typing import Optional

from victor.agent.rl.base import BaseLearner, RLOutcome, RLRecommendation

logger = logging.getLogger(__name__)


class ContinuationPatienceLearner(BaseLearner):
    """Learn optimal continuation_patience per provider:model:task.

    This learns how patient we should be before flagging a continuation loop
    as stuck (i.e., how many prompts without tool calls are acceptable).

    Attributes:
        name: Always "continuation_patience"
        db: SQLite database connection
        learning_rate: Adjustment rate (0.0-1.0)
    """

    def _ensure_tables(self) -> None:
        """Create tables for continuation patience stats."""
        cursor = self.db.cursor()

        # Stats table: one row per provider:model:task_type
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS continuation_patience_stats (
                context_key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                current_patience INTEGER NOT NULL,
                total_sessions INTEGER DEFAULT 0,
                false_positives INTEGER DEFAULT 0,
                true_positives INTEGER DEFAULT 0,
                missed_stuck_loops INTEGER DEFAULT 0,
                last_updated TEXT
            )
            """
        )

        # Index for fast lookups
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cont_patience_provider
            ON continuation_patience_stats(provider, model, task_type)
            """
        )

        self.db.commit()
        logger.debug("RL: continuation_patience tables ensured")

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record continuation patience outcome.

        Expected metadata:
        - continuation_prompts: How many prompts were sent without tool calls
        - patience_threshold: What the patience was configured to
        - flagged_as_stuck: Was it flagged as stuck loop?
        - actually_stuck: Was it actually stuck (true positive)?
        - eventually_made_progress: Did model eventually recover and make progress?

        Args:
            outcome: Outcome with continuation patience data
        """
        context_key = self._get_context_key(
            outcome.provider, outcome.model, outcome.task_type
        )

        cursor = self.db.cursor()

        # Get or create stats
        cursor.execute(
            "SELECT * FROM continuation_patience_stats WHERE context_key = ?",
            (context_key,),
        )
        row = cursor.fetchone()

        if row:
            # Update existing
            stats = dict(row)
        else:
            # Initialize with provider baseline
            baseline = self._get_provider_baseline("continuation_patience") or 3
            stats = {
                "context_key": context_key,
                "provider": outcome.provider,
                "model": outcome.model,
                "task_type": outcome.task_type,
                "current_patience": baseline,
                "total_sessions": 0,
                "false_positives": 0,
                "true_positives": 0,
                "missed_stuck_loops": 0,
                "last_updated": outcome.timestamp,
            }

        # Update counts
        stats["total_sessions"] += 1

        flagged_as_stuck = outcome.metadata.get("flagged_as_stuck", False)
        actually_stuck = outcome.metadata.get("actually_stuck", False)
        made_progress = outcome.metadata.get("eventually_made_progress", False)

        if flagged_as_stuck and not actually_stuck:
            # False positive: Flagged as stuck but wasn't really stuck
            stats["false_positives"] += 1
        elif flagged_as_stuck and actually_stuck:
            # True positive: Correctly detected stuck loop
            stats["true_positives"] += 1
        elif not flagged_as_stuck and actually_stuck:
            # Missed stuck loop: Should have been flagged but wasn't
            stats["missed_stuck_loops"] += 1

        stats["last_updated"] = outcome.timestamp

        # Update recommendation if enough data
        if stats["total_sessions"] >= 5:
            self._update_patience(stats)

        # Upsert to database
        cursor.execute(
            """
            INSERT OR REPLACE INTO continuation_patience_stats
            (context_key, provider, model, task_type, current_patience,
             total_sessions, false_positives, true_positives, missed_stuck_loops,
             last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stats["context_key"],
                stats["provider"],
                stats["model"],
                stats["task_type"],
                stats["current_patience"],
                stats["total_sessions"],
                stats["false_positives"],
                stats["true_positives"],
                stats["missed_stuck_loops"],
                stats["last_updated"],
            ),
        )
        self.db.commit()

        logger.debug(
            f"RL: Recorded continuation_patience outcome for {context_key} "
            f"(sessions={stats['total_sessions']}, FP={stats['false_positives']}, "
            f"TP={stats['true_positives']}, missed={stats['missed_stuck_loops']})"
        )

    def _update_patience(self, stats: dict) -> None:
        """Update patience recommendation based on stats.

        Strategy:
        - High false positive rate → Increase patience
        - High missed stuck loop rate → Decrease patience
        - Balance between catching stuck loops and avoiding false alarms

        Args:
            stats: Statistics dictionary
        """
        total = stats["total_sessions"]
        fp_rate = stats["false_positives"] / total if total > 0 else 0.0
        missed_rate = stats["missed_stuck_loops"] / total if total > 0 else 0.0

        current_patience = stats["current_patience"]
        adjustment = 0

        # High false positive rate → Increase patience
        if fp_rate > 0.2:
            adjustment = 2 if fp_rate > 0.3 else 1
            logger.debug(
                f"RL: High FP rate ({fp_rate:.1%}) for {stats['context_key']} → "
                f"increase patience by {adjustment}"
            )

        # High missed stuck loop rate → Decrease patience
        elif missed_rate > 0.1:
            adjustment = -2 if missed_rate > 0.2 else -1
            logger.debug(
                f"RL: High missed loop rate ({missed_rate:.1%}) for {stats['context_key']} → "
                f"decrease patience by {abs(adjustment)}"
            )

        # Apply adjustment with bounds [1, 15]
        new_patience = max(1, min(15, current_patience + adjustment))

        if new_patience != current_patience:
            stats["current_patience"] = new_patience
            logger.info(
                f"RL: Updated continuation_patience for {stats['context_key']}: "
                f"{current_patience} → {new_patience} "
                f"(FP={fp_rate:.1%}, missed={missed_rate:.1%})"
            )

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommended continuation patience.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Recommendation with patience value and confidence, or None
        """
        context_key = self._get_context_key(provider, model, task_type)

        cursor = self.db.cursor()
        cursor.execute(
            "SELECT * FROM continuation_patience_stats WHERE context_key = ?",
            (context_key,),
        )
        row = cursor.fetchone()

        if not row:
            # No data yet - fall back to provider baseline
            baseline = self._get_provider_baseline("continuation_patience")
            if baseline is not None:
                return RLRecommendation(
                    value=baseline,
                    confidence=0.0,
                    reason="Provider baseline (no learning data yet)",
                    sample_size=0,
                    is_baseline=True,
                )
            return None

        stats = dict(row)

        # Need at least 5 sessions for confidence
        if stats["total_sessions"] < 5:
            baseline = self._get_provider_baseline("continuation_patience") or stats[
                "current_patience"
            ]
            return RLRecommendation(
                value=baseline,
                confidence=0.0,
                reason=f"Insufficient data ({stats['total_sessions']} sessions)",
                sample_size=stats["total_sessions"],
                is_baseline=True,
            )

        # Calculate confidence based on sample size and accuracy
        confidence = min(1.0, stats["total_sessions"] / 20.0)

        # Reduce confidence if high FP or missed rate
        fp_rate = stats["false_positives"] / stats["total_sessions"]
        missed_rate = stats["missed_stuck_loops"] / stats["total_sessions"]
        if fp_rate > 0.2 or missed_rate > 0.1:
            confidence *= 0.7

        return RLRecommendation(
            value=stats["current_patience"],
            confidence=confidence,
            reason=f"Learned from {stats['total_sessions']} sessions "
            f"(FP={fp_rate:.1%}, missed={missed_rate:.1%})",
            sample_size=stats["total_sessions"],
            is_baseline=False,
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        # Reward for correct stuck loop detection
        flagged = outcome.metadata.get("flagged_as_stuck", False)
        actually_stuck = outcome.metadata.get("actually_stuck", False)

        if flagged and actually_stuck:
            # True positive - good!
            return 1.0
        elif not flagged and not actually_stuck:
            # True negative - good!
            return 1.0
        elif flagged and not actually_stuck:
            # False positive - bad
            return -0.5
        elif not flagged and actually_stuck:
            # Missed stuck loop - bad
            return -1.0

        return 0.0
