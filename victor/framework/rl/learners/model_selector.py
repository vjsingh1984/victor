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

"""RL learner for adaptive model/provider selection using Q-learning.

This learner uses Q-learning with epsilon-greedy exploration to select optimal
LLM models/providers based on historical session performance.

Strategy:
- Q-learning with epsilon-greedy exploration
- Multiple selection strategies: EPSILON_GREEDY, UCB, EXPLOIT_ONLY
- Task-type aware Q-values (different optimal models per task type)
- Reward based on: success, latency, token throughput, tool usage
- Epsilon decay schedule for exploration → exploitation transition
- Warm-up phase with higher learning rate
- Range: [0.0, 1.0] for Q-values

Migrated from: victor/agent/rl_model_selector.py
"""

import json
import logging
import math
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Model selection strategy."""

    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson"
    EXPLOIT_ONLY = "exploit"


class ModelSelectorLearner(BaseLearner):
    """Learn optimal provider/model selection using Q-learning.

    Uses Q-learning with epsilon-greedy exploration to find optimal
    providers/models for each task type based on historical performance.

    Attributes:
        name: Always "model_selector"
        db: SQLite database connection
        learning_rate: Q-value update rate (alpha)
        epsilon: Exploration rate
        epsilon_decay: Decay rate per selection
        epsilon_min: Minimum exploration rate
        discount_factor: Future reward discount (gamma)
        ucb_c: UCB exploration constant
        strategy: Selection strategy
    """

    # Mock/test providers to exclude from Q-table
    MOCK_PROVIDERS = frozenset({"mock", "mock_provider", "dummy", "dummy-stream", "test"})

    # Warm-up threshold: use higher learning rate until this many real selections
    WARMUP_THRESHOLD = 100

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        discount_factor: float = 0.95,
        ucb_c: float = 2.0,
        strategy: SelectionStrategy = SelectionStrategy.EPSILON_GREEDY,
    ):
        """Initialize model selector learner.

        Args:
            name: Learner name
            db_connection: SQLite database connection
            learning_rate: Q-learning alpha parameter
            provider_adapter: Optional provider adapter
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per selection
            epsilon_min: Minimum epsilon value
            discount_factor: Q-learning gamma parameter
            ucb_c: UCB exploration constant
            strategy: Selection strategy to use
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor
        self.ucb_c = ucb_c
        self.strategy = strategy

        # In-memory caches
        self._q_table: Dict[str, float] = {}
        self._selection_counts: Dict[str, int] = {}
        self._total_selections: int = 0
        self._q_table_by_task: Dict[str, Dict[str, float]] = {}
        self._task_selection_counts: Dict[str, Dict[str, int]] = {}

        # Load state from database
        self._load_state()

    def _ensure_tables(self) -> None:
        """Create tables for model selector stats."""
        cursor = self.db.cursor()

        # Global Q-values table (uses Tables constants)
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.RL_MODEL_Q} (
                provider TEXT PRIMARY KEY,
                q_value REAL NOT NULL,
                selection_count INTEGER DEFAULT 0,
                last_updated TEXT
            )
            """
        )

        # Task-specific Q-values table
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.RL_MODEL_TASK} (
                provider TEXT NOT NULL,
                task_type TEXT NOT NULL,
                q_value REAL NOT NULL,
                selection_count INTEGER DEFAULT 0,
                last_updated TEXT,
                PRIMARY KEY (provider, task_type)
            )
            """
        )

        # State table (epsilon, total_selections)
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.RL_MODEL_STATE} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        # Indexes
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_rl_model_task
            ON {Tables.RL_MODEL_TASK}(provider, task_type)
            """
        )

        self.db.commit()
        logger.debug("RL: model_selector tables ensured")

    def _load_state(self) -> None:
        """Load state from database."""
        cursor = self.db.cursor()

        # Load global Q-values
        cursor.execute(f"SELECT * FROM {Tables.RL_MODEL_Q}")
        for row in cursor.fetchall():
            stats = dict(row)
            provider = stats["provider"]
            self._q_table[provider] = stats["q_value"]
            self._selection_counts[provider] = stats["selection_count"]
            self._total_selections += stats["selection_count"]

        # Load task-specific Q-values
        cursor.execute(f"SELECT * FROM {Tables.RL_MODEL_TASK}")
        for row in cursor.fetchall():
            stats = dict(row)
            provider = stats["provider"]
            task_type = stats["task_type"]

            if provider not in self._q_table_by_task:
                self._q_table_by_task[provider] = {}
                self._task_selection_counts[provider] = {}

            self._q_table_by_task[provider][task_type] = stats["q_value"]
            self._task_selection_counts[provider][task_type] = stats["selection_count"]

        # Load epsilon and total_selections
        cursor.execute(
            f"SELECT * FROM {Tables.RL_MODEL_STATE} WHERE key IN ('epsilon', 'total_selections')"
        )
        for row in cursor.fetchall():
            stats = dict(row)
            if stats["key"] == "epsilon":
                self.epsilon = float(stats["value"])
            elif stats["key"] == "total_selections":
                self._total_selections = int(stats["value"])

        if self._q_table:
            logger.info(f"RL: Loaded {len(self._q_table)} provider Q-values from database")

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record model selection outcome and update Q-values.

        Expected metadata:
        - latency_seconds: Session duration
        - token_count: Total tokens
        - tool_calls_made: Number of tool calls
        - user_satisfaction: Optional user feedback (0-1)

        Args:
            outcome: Outcome with model selection data
        """
        provider = outcome.provider

        # Skip mock providers
        if provider in self.MOCK_PROVIDERS:
            logger.debug(f"RL: Skipping mock provider '{provider}'")
            return

        # Compute reward
        reward = self._compute_reward(outcome)

        # Get old Q-value
        old_q = self._q_table.get(provider, 0.5)

        # Compute effective learning rate (higher during warm-up)
        effective_lr = self._get_effective_learning_rate(provider)

        # Update Q-value: Q(s,a) ← Q(s,a) + α[r - Q(s,a)]
        new_q = max(0.0, min(1.0, old_q + effective_lr * (reward - old_q)))

        # Update global Q-table
        self._q_table[provider] = new_q
        self._selection_counts[provider] = self._selection_counts.get(provider, 0) + 1
        self._total_selections += 1

        # Update task-specific Q-table
        task_type = outcome.task_type
        if task_type:
            if provider not in self._q_table_by_task:
                self._q_table_by_task[provider] = {}
                self._task_selection_counts[provider] = {}

            old_task_q = self._q_table_by_task[provider].get(task_type, 0.5)
            new_task_q = max(0.0, min(1.0, old_task_q + effective_lr * (reward - old_task_q)))

            self._q_table_by_task[provider][task_type] = new_task_q
            self._task_selection_counts[provider][task_type] = (
                self._task_selection_counts[provider].get(task_type, 0) + 1
            )

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Persist to database
        self._save_to_db(provider, task_type, outcome.timestamp)

        logger.debug(
            f"RL: Updated Q({provider}): {old_q:.3f} → {new_q:.3f} "
            f"(reward={reward:.3f}, count={self._selection_counts[provider]})"
        )

    def _save_to_db(self, provider: str, task_type: Optional[str], timestamp: str) -> None:
        """Save Q-values and state to database."""
        cursor = self.db.cursor()

        # Save global Q-value
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_MODEL_Q}
            (provider, q_value, selection_count, last_updated)
            VALUES (?, ?, ?, ?)
            """,
            (
                provider,
                self._q_table[provider],
                self._selection_counts[provider],
                timestamp,
            ),
        )

        # Save task-specific Q-value
        if task_type and provider in self._q_table_by_task:
            if task_type in self._q_table_by_task[provider]:
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {Tables.RL_MODEL_TASK}
                    (provider, task_type, q_value, selection_count, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        provider,
                        task_type,
                        self._q_table_by_task[provider][task_type],
                        self._task_selection_counts[provider][task_type],
                        timestamp,
                    ),
                )

        # Save epsilon and total_selections
        cursor.execute(
            f"INSERT OR REPLACE INTO {Tables.RL_MODEL_STATE} (key, value) VALUES ('epsilon', ?)",
            (str(self.epsilon),),
        )
        cursor.execute(
            f"INSERT OR REPLACE INTO {Tables.RL_MODEL_STATE} (key, value) VALUES ('total_selections', ?)",
            (str(self._total_selections),),
        )

        self.db.commit()

    def _get_effective_learning_rate(self, provider: str) -> float:
        """Get effective learning rate based on warm-up phase."""
        real_selections = sum(
            v for k, v in self._selection_counts.items() if k not in self.MOCK_PROVIDERS
        )
        provider_count = self._selection_counts.get(provider, 0)

        # Higher learning rate during warm-up
        if real_selections < self.WARMUP_THRESHOLD or provider_count < 10:
            return min(0.5, self.learning_rate * 3)
        return self.learning_rate

    def _get_q_value(self, provider: str, task_type: Optional[str] = None) -> float:
        """Get Q-value for a provider, optionally task-specific."""
        global_q = self._q_table.get(provider, 0.5)

        if not task_type or provider not in self._q_table_by_task:
            return global_q

        task_q = self._q_table_by_task[provider].get(task_type)
        # Blend: 70% task-specific, 30% global
        return 0.7 * task_q + 0.3 * global_q if task_q is not None else global_q

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommended provider for selection.

        Note: This method signature matches BaseLearner but for model selector,
        we use it differently - provider param is used to pass available providers list.

        Args:
            provider: JSON string of available providers list (e.g., '["anthropic", "openai"]')
            model: Not used (kept for signature compatibility)
            task_type: Task type for context-aware selection

        Returns:
            Recommendation with selected provider and confidence, or None
        """
        # Parse available providers from provider param
        try:
            available_providers = json.loads(provider) if isinstance(provider, str) else [provider]
        except (json.JSONDecodeError, TypeError):
            available_providers = [provider]

        if not available_providers:
            return None

        # Ensure all providers have Q-values
        for p in available_providers:
            if p not in self._q_table:
                self._q_table[p] = 0.5  # Optimistic initial value
                self._selection_counts[p] = 0

        # Select provider using configured strategy
        if self.strategy == SelectionStrategy.EPSILON_GREEDY:
            selected, reason = self._select_epsilon_greedy(available_providers, task_type)
        elif self.strategy == SelectionStrategy.UCB:
            selected, reason = self._select_ucb(available_providers, task_type)
        elif self.strategy == SelectionStrategy.EXPLOIT_ONLY:
            selected, reason = self._select_exploit(available_providers, task_type)
        else:
            selected, reason = self._select_epsilon_greedy(available_providers, task_type)

        q_value = self._get_q_value(selected, task_type)
        count = self._selection_counts.get(selected, 0)
        confidence = min(0.95, 0.5 + 0.45 * (1 - math.exp(-count / 20)))

        return RLRecommendation(
            value=selected,
            confidence=confidence,
            reason=f"{reason} (Q={q_value:.3f}, n={count})",
            sample_size=count,
            is_baseline=False,
        )

    def _select_epsilon_greedy(
        self, providers: List[str], task_type: Optional[str] = None
    ) -> Tuple[str, str]:
        """Select using epsilon-greedy strategy."""
        # Exploration
        if random.random() < self.epsilon:
            selected = random.choice(providers)
            reason = f"Exploration (ε={self.epsilon:.2f})"
            return selected, reason

        # Exploitation
        return self._select_exploit(providers, task_type)

    def _select_exploit(
        self, providers: List[str], task_type: Optional[str] = None
    ) -> Tuple[str, str]:
        """Select provider with highest Q-value."""
        best_provider = max(providers, key=lambda p: self._get_q_value(p, task_type))
        reason = "Best Q-value"
        if task_type:
            reason += f" [task={task_type}]"
        return best_provider, reason

    def _select_ucb(self, providers: List[str], task_type: Optional[str] = None) -> Tuple[str, str]:
        """Select using Upper Confidence Bound (UCB) strategy."""
        if self._total_selections == 0:
            return random.choice(providers), "No history, random selection"

        log_total = math.log(self._total_selections + 1)
        ucb_scores = {}

        for p in providers:
            q_value = self._get_q_value(p, task_type)
            count = self._selection_counts.get(p, 0)
            ucb_scores[p] = (
                float("inf") if count == 0 else q_value + self.ucb_c * math.sqrt(log_total / count)
            )

        best_provider = max(providers, key=lambda p: ucb_scores[p])
        ucb_score = ucb_scores[best_provider]

        if ucb_score == float("inf"):
            reason = "Unexplored (UCB=∞)"
        else:
            reason = f"UCB={ucb_score:.3f}"

        return best_provider, reason

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Reward based on:
        - Success: +1.0 / -0.5
        - Latency penalty (baseline 30s)
        - Tool usage bonus
        - Token throughput bonus (baseline 5 tok/s)
        - User satisfaction override (if available)

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        reward = 1.0 if outcome.success else -0.5

        # Latency penalty
        latency = outcome.metadata.get("latency_seconds", 0)
        if latency > 30:
            reward -= min(0.1 * (latency - 30) / 60, 0.5)

        # Tool usage bonus
        tool_calls = outcome.metadata.get("tool_calls_made", 0)
        if tool_calls > 0:
            reward += 0.1

        # Token throughput bonus
        token_count = outcome.metadata.get("token_count", 0)
        if latency > 0 and token_count > 0:
            tok_per_sec = token_count / latency
            if tok_per_sec > 5:
                reward += min(0.05 * (tok_per_sec - 5) / 10, 0.2)

        # User satisfaction override
        user_satisfaction = outcome.metadata.get("user_satisfaction")
        if user_satisfaction is not None:
            reward = 0.7 * reward + 0.3 * (user_satisfaction * 2 - 0.5)

        return max(-1.0, min(1.0, reward))

    def get_provider_rankings(self) -> List[Dict[str, Any]]:
        """Get all providers ranked by Q-value.

        Returns:
            List of provider stats sorted by Q-value descending
        """
        rankings = []
        log_total = math.log(self._total_selections + 1) if self._total_selections > 0 else 0

        for provider, q_value in self._q_table.items():
            count = self._selection_counts.get(provider, 0)

            # Compute UCB score
            if count == 0:
                ucb_score = float("inf")
            else:
                ucb_score = q_value + self.ucb_c * math.sqrt(log_total / count)

            # Confidence
            confidence = min(0.95, 0.5 + 0.45 * (1 - math.exp(-count / 20)))

            rankings.append(
                {
                    "provider": provider,
                    "q_value": q_value,
                    "session_count": count,
                    "ucb_score": ucb_score if ucb_score != float("inf") else None,
                    "confidence": confidence,
                }
            )

        return sorted(rankings, key=lambda x: x["q_value"], reverse=True)

    def get_exploration_rate(self) -> float:
        """Get the current exploration rate (epsilon).

        Returns:
            Current epsilon value for exploration
        """
        return self.epsilon

    def get_strategy(self) -> SelectionStrategy:
        """Get the current selection strategy.

        Returns:
            Current selection strategy enum
        """
        return self.strategy
