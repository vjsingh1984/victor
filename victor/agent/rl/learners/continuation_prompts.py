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

"""RL learner for optimal continuation prompt limits per provider:model:task.

This learner determines the optimal max_continuation_prompts value - the total
budget for continuation prompts before forcing completion.

This is DIFFERENT from continuation_patience (learned by ContinuationPatienceLearner):
- max_continuation_prompts: Total budget (e.g., 10 prompts max)
- continuation_patience: No-tool tolerance (e.g., 5 prompts without tools)

Strategy:
- If stuck rate > 30% → Decrease max prompts (model gets stuck frequently)
- If quality < 0.6 and few prompts used → Increase max prompts (needs more time)
- If quality > 0.8 and many prompts used → Decrease max prompts (wasting time)
- Converge to optimal [1, 20] range

Migrated from: victor/agent/continuation_learner.py
"""

import logging
from typing import Optional

from victor.agent.rl.base import BaseLearner, RLOutcome, RLRecommendation

logger = logging.getLogger(__name__)


class ContinuationPromptLearner(BaseLearner):
    """Learn optimal max_continuation_prompts per provider:model:task.

    Uses multi-armed bandit approach with weighted moving average to find
    optimal continuation prompt limits for each provider:model:task_type combination.

    Attributes:
        name: Always "continuation_prompts"
        db: SQLite database connection
        learning_rate: Adjustment rate (0.0-1.0)
    """

    def _ensure_tables(self) -> None:
        """Create tables for continuation prompts stats."""
        cursor = self.db.cursor()

        # Stats table: one row per provider:model:task_type
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS continuation_prompts_stats (
                context_key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                total_sessions INTEGER DEFAULT 0,
                successful_sessions INTEGER DEFAULT 0,
                stuck_loop_count INTEGER DEFAULT 0,
                forced_completion_count INTEGER DEFAULT 0,
                avg_quality_score REAL DEFAULT 0.0,
                avg_prompts_used REAL DEFAULT 0.0,
                current_max_prompts INTEGER NOT NULL,
                recommended_max_prompts INTEGER,
                quality_sum REAL DEFAULT 0.0,
                prompts_sum REAL DEFAULT 0.0,
                last_updated TEXT
            )
            """
        )

        # Index for fast lookups
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cont_prompts_provider
            ON continuation_prompts_stats(provider, model, task_type)
            """
        )

        self.db.commit()
        logger.debug("RL: continuation_prompts tables ensured")

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record continuation prompts outcome.

        Expected metadata:
        - continuation_prompts_used: How many continuation prompts were sent
        - max_prompts_configured: The max limit that was configured
        - stuck_loop_detected: Was stuck continuation loop detected?
        - forced_completion: Was completion forced?
        - tool_calls_total: Total tool calls made

        Args:
            outcome: Outcome with continuation prompts data
        """
        context_key = self._get_context_key(outcome.provider, outcome.model, outcome.task_type)

        cursor = self.db.cursor()

        # Get or create stats
        cursor.execute(
            "SELECT * FROM continuation_prompts_stats WHERE context_key = ?",
            (context_key,),
        )
        row = cursor.fetchone()

        if row:
            # Update existing
            stats = dict(row)
        else:
            # Initialize new entry
            max_prompts = outcome.metadata.get("max_prompts_configured", 6)
            stats = {
                "context_key": context_key,
                "provider": outcome.provider,
                "model": outcome.model,
                "task_type": outcome.task_type,
                "total_sessions": 0,
                "successful_sessions": 0,
                "stuck_loop_count": 0,
                "forced_completion_count": 0,
                "avg_quality_score": 0.0,
                "avg_prompts_used": 0.0,
                "current_max_prompts": max_prompts,
                "recommended_max_prompts": None,
                "quality_sum": 0.0,
                "prompts_sum": 0.0,
                "last_updated": outcome.timestamp,
            }

        # Update counts
        stats["total_sessions"] += 1
        if outcome.success:
            stats["successful_sessions"] += 1

        stuck_loop = outcome.metadata.get("stuck_loop_detected", False)
        if stuck_loop:
            stats["stuck_loop_count"] += 1

        forced = outcome.metadata.get("forced_completion", False)
        if forced:
            stats["forced_completion_count"] += 1

        # Update weighted moving average (decay=0.9)
        decay = 0.9
        prompts_used = outcome.metadata.get("continuation_prompts_used", 0)

        stats["quality_sum"] = decay * stats["quality_sum"] + outcome.quality_score
        stats["prompts_sum"] = decay * stats["prompts_sum"] + prompts_used

        stats["avg_quality_score"] = stats["quality_sum"] / (
            1 + (stats["total_sessions"] - 1) * decay
        )
        stats["avg_prompts_used"] = stats["prompts_sum"] / (
            1 + (stats["total_sessions"] - 1) * decay
        )

        stats["current_max_prompts"] = outcome.metadata.get(
            "max_prompts_configured", stats["current_max_prompts"]
        )
        stats["last_updated"] = outcome.timestamp

        # Update recommendation if enough data
        if stats["total_sessions"] >= 3:
            self._update_max_prompts(stats)

        # Upsert to database
        cursor.execute(
            """
            INSERT OR REPLACE INTO continuation_prompts_stats
            (context_key, provider, model, task_type, total_sessions, successful_sessions,
             stuck_loop_count, forced_completion_count, avg_quality_score, avg_prompts_used,
             current_max_prompts, recommended_max_prompts, quality_sum, prompts_sum, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stats["context_key"],
                stats["provider"],
                stats["model"],
                stats["task_type"],
                stats["total_sessions"],
                stats["successful_sessions"],
                stats["stuck_loop_count"],
                stats["forced_completion_count"],
                stats["avg_quality_score"],
                stats["avg_prompts_used"],
                stats["current_max_prompts"],
                stats["recommended_max_prompts"],
                stats["quality_sum"],
                stats["prompts_sum"],
                stats["last_updated"],
            ),
        )
        self.db.commit()

        logger.debug(
            f"RL: Recorded continuation_prompts outcome for {context_key} "
            f"(sessions={stats['total_sessions']}, "
            f"stuck={stats['stuck_loop_count']}, quality={stats['avg_quality_score']:.2f})"
        )

    def _update_max_prompts(self, stats: dict) -> None:
        """Update max_prompts recommendation based on stats.

        Strategy:
        1. If stuck rate > 30% → Decrease (model gets stuck frequently)
        2. If quality < 0.6 and few prompts → Increase (needs more time)
        3. If quality > 0.8 and many prompts → Decrease (wasting time)
        4. If low success rate → Increase (model struggling)

        Args:
            stats: Statistics dictionary
        """
        total = stats["total_sessions"]
        stuck_rate = stats["stuck_loop_count"] / total if total > 0 else 0.0
        success_rate = stats["successful_sessions"] / total if total > 0 else 0.0
        quality = stats["avg_quality_score"]
        avg_prompts = stats["avg_prompts_used"]
        current = stats["current_max_prompts"]

        adjustment = 0

        # High stuck rate → Decrease
        if stuck_rate > 0.3:
            adjustment = -2 if stuck_rate > 0.5 else -1
            logger.debug(
                f"RL: High stuck rate ({stuck_rate:.1%}) for {stats['context_key']} → "
                f"decrease by {abs(adjustment)}"
            )

        # Low quality + few prompts → Increase
        elif quality < 0.6 and avg_prompts < current * 0.7:
            adjustment = 2 if quality < 0.4 else 1
            logger.debug(
                f"RL: Low quality ({quality:.2f}) + few prompts ({avg_prompts:.1f}/{current}) "
                f"for {stats['context_key']} → increase by {adjustment}"
            )

        # High quality + many prompts → Decrease
        elif quality > 0.8 and avg_prompts > current * 0.8:
            adjustment = -1
            logger.debug(
                f"RL: High quality ({quality:.2f}) + many prompts ({avg_prompts:.1f}/{current}) "
                f"for {stats['context_key']} → decrease by 1"
            )

        # Low success rate → Increase
        elif success_rate < 0.5 and total >= 5:
            adjustment = 1
            logger.debug(
                f"RL: Low success rate ({success_rate:.1%}) for {stats['context_key']} → "
                f"increase by 1"
            )

        # Apply adjustment with bounds [1, 20]
        new_max = max(1, min(20, current + adjustment))

        if new_max != current:
            stats["recommended_max_prompts"] = new_max
            logger.info(
                f"RL: Updated continuation_prompts for {stats['context_key']}: "
                f"{current} → {new_max} "
                f"(stuck={stuck_rate:.1%}, quality={quality:.2f}, success={success_rate:.1%})"
            )
        else:
            stats["recommended_max_prompts"] = current

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommended max continuation prompts.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Recommendation with max prompts value and confidence, or None
        """
        context_key = self._get_context_key(provider, model, task_type)

        cursor = self.db.cursor()
        cursor.execute(
            "SELECT * FROM continuation_prompts_stats WHERE context_key = ?",
            (context_key,),
        )
        row = cursor.fetchone()

        if not row:
            # No data yet - return None (caller will use default)
            return None

        stats = dict(row)

        # Need at least 3 sessions for confidence
        if stats["total_sessions"] < 3:
            return None

        # Calculate confidence based on sample size
        confidence = min(1.0, stats["total_sessions"] / 20.0)

        # Reduce confidence if high stuck rate or low success rate
        stuck_rate = stats["stuck_loop_count"] / stats["total_sessions"]
        success_rate = stats["successful_sessions"] / stats["total_sessions"]

        if stuck_rate > 0.3 or success_rate < 0.5:
            confidence *= 0.7

        recommended = stats["recommended_max_prompts"] or stats["current_max_prompts"]

        return RLRecommendation(
            value=recommended,
            confidence=confidence,
            reason=f"Learned from {stats['total_sessions']} sessions "
            f"(stuck={stuck_rate:.1%}, success={success_rate:.1%})",
            sample_size=stats["total_sessions"],
            is_baseline=False,
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Reward based on:
        - Success: +1.0
        - Quality: weighted component
        - Stuck loop: -0.5
        - Forced completion: -0.3

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        reward = 1.0 if outcome.success else -0.5

        # Quality component (weight 30%)
        reward = 0.7 * reward + 0.3 * (outcome.quality_score * 2 - 1)

        # Penalties
        if outcome.metadata.get("stuck_loop_detected", False):
            reward -= 0.5

        if outcome.metadata.get("forced_completion", False):
            reward -= 0.3

        # Tool usage bonus
        tool_calls = outcome.metadata.get("tool_calls_total", 0)
        if tool_calls > 5:
            reward += 0.1

        return max(-1.0, min(1.0, reward))
