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

"""RL learner for optimal context pruning to minimize token usage.

This learner uses Q-learning to learn when and how aggressively to prune
conversation context to reduce token consumption while preserving task success.

Strategy:
- State: (context_utilization_bucket, tool_call_count, task_complexity, provider_type)
- Action: (no_prune, light_prune, moderate_prune, aggressive_prune)
- Reward: token_efficiency_gain * task_success - context_loss_penalty

Design:
- Learns per-provider optimal pruning thresholds (cloud vs local models differ)
- Balances token savings against task success rate
- Uses Thompson Sampling for exploration to find optimal pruning levels
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


class PruningAction(str, Enum):
    """Possible context pruning actions."""

    NO_PRUNE = "no_prune"  # Keep all context
    LIGHT = "light"  # Prune 20% oldest tool results
    MODERATE = "moderate"  # Prune 40% oldest messages
    AGGRESSIVE = "aggressive"  # Prune to 50% target utilization


@dataclass
class PruningConfig:
    """Configuration for a pruning action."""

    action: PruningAction
    compaction_threshold: float  # When to trigger (0.0-1.0)
    compaction_target: float  # Target utilization after pruning (0.0-1.0)
    tool_result_max_chars: int  # Max chars per tool result
    min_messages_keep: int  # Minimum messages to preserve


# Default configs for each action
PRUNING_CONFIGS = {
    PruningAction.NO_PRUNE: PruningConfig(
        action=PruningAction.NO_PRUNE,
        compaction_threshold=0.95,
        compaction_target=0.90,
        tool_result_max_chars=12000,
        min_messages_keep=15,
    ),
    PruningAction.LIGHT: PruningConfig(
        action=PruningAction.LIGHT,
        compaction_threshold=0.80,
        compaction_target=0.65,
        tool_result_max_chars=8000,
        min_messages_keep=10,
    ),
    PruningAction.MODERATE: PruningConfig(
        action=PruningAction.MODERATE,
        compaction_threshold=0.70,
        compaction_target=0.50,
        tool_result_max_chars=5000,
        min_messages_keep=6,
    ),
    PruningAction.AGGRESSIVE: PruningConfig(
        action=PruningAction.AGGRESSIVE,
        compaction_threshold=0.60,
        compaction_target=0.40,
        tool_result_max_chars=3000,
        min_messages_keep=4,
    ),
}


class ContextPruningLearner(BaseLearner):
    """Learn optimal context pruning policy using Q-learning with Thompson Sampling.

    Uses Thompson Sampling for exploration to find optimal pruning levels
    that maximize token efficiency while maintaining task success.

    Attributes:
        name: Always "context_pruning"
        db: SQLite database connection
        learning_rate: Q-value update rate (alpha), default 0.15
        discount_factor: Future reward discount (gamma), default 0.90
    """

    # Q-learning parameters
    DEFAULT_LEARNING_RATE = 0.15
    DEFAULT_DISCOUNT_FACTOR = 0.90

    # Thompson Sampling priors (Beta distribution)
    PRIOR_ALPHA = 1.0
    PRIOR_BETA = 1.0

    # Minimum samples before confident recommendation
    MIN_SAMPLES_FOR_CONFIDENCE = 10

    # Valid actions
    ACTIONS = list(PruningAction)

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        provider_adapter: Optional[Any] = None,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
    ):
        """Initialize context pruning learner.

        Args:
            name: Learner name (should be "context_pruning")
            db_connection: SQLite database connection
            learning_rate: Q-value update rate (default 0.15)
            provider_adapter: Optional provider adapter
            discount_factor: Future reward discount (default 0.90)
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)
        self.discount_factor = discount_factor

    def _ensure_tables(self) -> None:
        """Create tables for context pruning stats."""
        cursor = self.db.cursor()

        # Q-values table: state-action pairs
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.RL_CONTEXT_PRUNING} (
                state_key TEXT NOT NULL,
                action TEXT NOT NULL,
                q_value REAL DEFAULT 0.0,
                visit_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                total_tokens_saved INTEGER DEFAULT 0,
                avg_token_savings REAL DEFAULT 0.0,
                last_updated TEXT,
                PRIMARY KEY (state_key, action)
            )
            """
        )

        # Stats table: overall pruning statistics
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.RL_CONTEXT_PRUNING}_stats (
                provider_type TEXT PRIMARY KEY,
                total_decisions INTEGER DEFAULT 0,
                total_tokens_saved INTEGER DEFAULT 0,
                avg_success_rate REAL DEFAULT 0.0,
                best_action TEXT,
                last_updated TEXT
            )
            """
        )

        self.db.commit()

    def _discretize_state(
        self,
        context_utilization: float,
        tool_call_count: int,
        task_complexity: str,
        provider_type: str,
    ) -> str:
        """Convert continuous state to discrete state key.

        Args:
            context_utilization: Current context usage (0.0-1.0)
            tool_call_count: Number of tool calls made
            task_complexity: Task complexity (simple, medium, complex)
            provider_type: Provider type (cloud, local)

        Returns:
            State key string for Q-table lookup
        """
        # Discretize context utilization into buckets
        if context_utilization < 0.3:
            util_bucket = "low"
        elif context_utilization < 0.6:
            util_bucket = "medium"
        elif context_utilization < 0.8:
            util_bucket = "high"
        else:
            util_bucket = "critical"

        # Discretize tool call count
        if tool_call_count < 3:
            tool_bucket = "few"
        elif tool_call_count < 7:
            tool_bucket = "moderate"
        else:
            tool_bucket = "many"

        return f"{provider_type}:{util_bucket}:{tool_bucket}:{task_complexity}"

    def get_recommendation(
        self,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[RLRecommendation]:
        """Get recommended pruning action for current state.

        Uses Thompson Sampling to balance exploration and exploitation.

        Args:
            provider: Provider name
            model: Model name (used to infer provider type)
            task_type: Task type (used to infer complexity)

        Returns:
            RLRecommendation with action and config
        """
        # Infer parameters from provider/model/task_type
        provider_type = "local" if provider in ["ollama", "lmstudio", "vllm"] else "cloud"
        task_complexity = "medium"  # Default

        # For simplicity, use moderate defaults
        context_utilization = 0.5
        tool_call_count = 5

        state_key = self._discretize_state(
            context_utilization, tool_call_count, task_complexity, provider_type
        )

        cursor = self.db.cursor()

        # Get Q-values and visit counts for all actions in this state
        cursor.execute(
            f"""
            SELECT action, q_value, visit_count, success_count
            FROM {Tables.RL_CONTEXT_PRUNING}
            WHERE state_key = ?
            """,
            (state_key,),
        )
        rows = cursor.fetchall()

        # Build action stats dict
        action_stats = {}
        for row in rows:
            action_stats[row[0]] = {
                "q_value": row[1],
                "visits": row[2],
                "successes": row[3],
            }

        # Thompson Sampling: sample from Beta distribution for each action
        import random

        best_action = None
        best_sample = -float("inf")

        for action in self.ACTIONS:
            stats = action_stats.get(
                action.value,
                {"q_value": 0.0, "visits": 0, "successes": 0},
            )

            # Beta distribution parameters
            alpha = self.PRIOR_ALPHA + stats["successes"]
            beta = self.PRIOR_BETA + stats["visits"] - stats["successes"]

            # Sample from Beta distribution
            sample = random.betavariate(alpha, max(1, beta))

            # Weight by Q-value if we have enough samples
            if stats["visits"] >= self.MIN_SAMPLES_FOR_CONFIDENCE:
                sample = 0.7 * sample + 0.3 * (stats["q_value"] / 100.0)

            if sample > best_sample:
                best_sample = sample
                best_action = action

        # Default to MODERATE if no data
        if best_action is None:
            best_action = PruningAction.MODERATE

        config = PRUNING_CONFIGS[best_action]

        total_visits = sum(s.get("visits", 0) for s in action_stats.values())
        confidence = min(1.0, total_visits / (self.MIN_SAMPLES_FOR_CONFIDENCE * len(self.ACTIONS)))

        # Store config in metadata for later retrieval
        metadata_value = {
            "state_key": state_key,
            "config": {
                "compaction_threshold": config.compaction_threshold,
                "compaction_target": config.compaction_target,
                "tool_result_max_chars": config.tool_result_max_chars,
                "min_messages_keep": config.min_messages_keep,
            },
            "exploration_sample": best_sample,
        }

        return RLRecommendation(
            value={
                "action": best_action.value,
                "config": metadata_value,
            },
            confidence=confidence,
            reason=f"Pruning recommendation: {best_action.value} (confidence={confidence:.2f})",
            sample_size=total_visits,
            is_baseline=total_visits < self.MIN_SAMPLES_FOR_CONFIDENCE,
        )

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record outcome of a pruning decision.

        Args:
            outcome: Outcome with pruning-specific metadata
        """
        # Extract parameters from outcome metadata
        context_utilization = outcome.metadata.get("context_utilization", 0.5)
        tool_call_count = outcome.metadata.get("tool_call_count", 5)
        action = outcome.metadata.get("action", "moderate")
        task_success = outcome.success
        tokens_saved = outcome.metadata.get("tokens_saved", 0)
        task_complexity = outcome.metadata.get("task_complexity", "medium")
        provider_type = outcome.metadata.get("provider_type", "cloud")

        state_key = self._discretize_state(
            context_utilization, tool_call_count, task_complexity, provider_type
        )

        # Calculate reward: balance token savings vs task success
        # Normalize tokens_saved to 0-1 range (assume max savings ~50K tokens)
        normalized_savings = min(1.0, tokens_saved / 50000.0)
        success_bonus = 1.0 if task_success else -0.5  # Penalty for failure
        reward = (0.4 * normalized_savings + 0.6 * success_bonus) * 100  # Scale to 0-100

        cursor = self.db.cursor()
        now = datetime.now(timezone.utc).isoformat()

        # Update Q-value using Q-learning update rule
        cursor.execute(
            f"""
            INSERT INTO {Tables.RL_CONTEXT_PRUNING}
                (state_key, action, q_value, visit_count, success_count,
                 total_tokens_saved, avg_token_savings, last_updated)
            VALUES (?, ?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(state_key, action) DO UPDATE SET
                q_value = q_value + ? * (? - q_value),
                visit_count = visit_count + 1,
                success_count = success_count + ?,
                total_tokens_saved = total_tokens_saved + ?,
                avg_token_savings = (total_tokens_saved + ?) / (visit_count + 1),
                last_updated = ?
            """,
            (
                state_key,
                action,
                reward,
                1 if task_success else 0,
                tokens_saved,
                float(tokens_saved),
                now,
                self.learning_rate,
                reward,
                1 if task_success else 0,
                tokens_saved,
                tokens_saved,
                now,
            ),
        )

        # Update provider stats
        cursor.execute(
            f"""
            INSERT INTO {Tables.RL_CONTEXT_PRUNING}_stats
                (provider_type, total_decisions, total_tokens_saved, last_updated)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(provider_type) DO UPDATE SET
                total_decisions = total_decisions + 1,
                total_tokens_saved = total_tokens_saved + ?,
                last_updated = ?
            """,
            (provider_type, tokens_saved, now, tokens_saved, now),
        )

        self.db.commit()
        logger.debug(
            f"Recorded pruning outcome: state={state_key}, action={action}, "
            f"success={task_success}, tokens_saved={tokens_saved}, reward={reward:.1f}"
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward for context pruning outcome.

        Args:
            outcome: Outcome with pruning metadata

        Returns:
            Reward value (-1.0 to 1.0)
        """
        # Extract parameters from outcome metadata
        tokens_saved = outcome.metadata.get("tokens_saved", 0)
        task_success = outcome.success

        # Calculate reward: balance token savings vs task success
        # Normalize tokens_saved to 0-1 range (assume max savings ~50K tokens)
        normalized_savings = min(1.0, tokens_saved / 50000.0)
        success_bonus = 1.0 if task_success else -0.5  # Penalty for failure
        reward = 0.4 * normalized_savings + 0.6 * success_bonus

        # Scale to -1.0 to 1.0 range
        return max(-1.0, min(1.0, reward * 2))

    def get_stats(self, provider_type: Optional[str] = None) -> Dict[str, Any]:
        """Get learner statistics.

        Args:
            provider_type: Optional filter by provider type

        Returns:
            Dictionary with learning statistics
        """
        cursor = self.db.cursor()

        # Get per-action stats
        if provider_type:
            cursor.execute(
                f"""
                SELECT action, SUM(visit_count), AVG(q_value), SUM(success_count)
                FROM {Tables.RL_CONTEXT_PRUNING}
                WHERE state_key LIKE ?
                GROUP BY action
                """,
                (f"{provider_type}:%",),
            )
        else:
            cursor.execute(
                f"""
                SELECT action, SUM(visit_count), AVG(q_value), SUM(success_count)
                FROM {Tables.RL_CONTEXT_PRUNING}
                GROUP BY action
                """
            )

        action_stats = {}
        for row in cursor.fetchall():
            visits = row[1] or 0
            action_stats[row[0]] = {
                "visits": visits,
                "avg_q_value": row[2] or 0.0,
                "success_rate": (row[3] or 0) / max(1, visits),
            }

        # Get overall stats
        cursor.execute(
            f"""
            SELECT SUM(total_decisions), SUM(total_tokens_saved)
            FROM {Tables.RL_CONTEXT_PRUNING}_stats
            """
        )
        row = cursor.fetchone()

        return {
            "total_decisions": row[0] or 0 if row else 0,
            "total_tokens_saved": row[1] or 0 if row else 0,
            "action_stats": action_stats,
            "best_action": max(
                action_stats.items(),
                key=lambda x: x[1].get("avg_q_value", 0),
                default=(None, {}),
            )[0],
        }

    def reset(self) -> None:
        """Reset learner state (clear Q-values)."""
        cursor = self.db.cursor()
        cursor.execute(f"DELETE FROM {Tables.RL_CONTEXT_PRUNING}")
        cursor.execute(f"DELETE FROM {Tables.RL_CONTEXT_PRUNING}_stats")
        self.db.commit()
        logger.info("Context pruning learner reset")
