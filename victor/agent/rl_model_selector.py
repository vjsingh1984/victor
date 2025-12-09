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

"""Reinforcement Learning based model selector for adaptive model selection.

This module implements an epsilon-greedy Q-learning approach to select optimal
LLM models/providers based on historical session performance data stored in
ConversationStore.

The RL feedback loop:
1. ConversationStore tracks session metrics (provider, model, tool usage, duration)
2. RLModelSelector computes Q-values from aggregated statistics
3. Model selection uses epsilon-greedy (explore new models vs exploit best)
4. Session outcomes update Q-values via reward signal

Key Features:
- Uses normalized FK aggregation queries for efficient Q-value computation
- Epsilon-greedy exploration with decay schedule
- UCB (Upper Confidence Bound) for uncertainty-aware selection
- Multi-armed bandit formulation treating providers as arms
- Task-type aware selection (different optimal models per task type)
"""

from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Model selection strategy."""

    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson"
    EXPLOIT_ONLY = "exploit"


@dataclass
class ProviderQValue:
    """Q-value and statistics for a provider."""

    provider: str
    q_value: float
    session_count: int
    avg_messages: float
    tool_capable_pct: float
    ucb_score: float = 0.0
    confidence: float = 0.0


@dataclass
class ModelRecommendation:
    """Model recommendation with confidence and reasoning."""

    provider: str
    model: Optional[str]
    confidence: float
    q_value: float
    reason: str
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class SessionReward:
    """Reward signal from a completed session."""

    session_id: str
    provider: str
    model: str
    success: bool
    latency_seconds: float
    token_count: int
    tool_calls_made: int
    user_satisfaction: Optional[float] = None  # 0-1 if available
    task_type: Optional[str] = None  # Task type for context-aware Q-learning

    @property
    def reward(self) -> float:
        """Compute composite reward signal.

        Reward formula:
        - Base reward: 1.0 for success, -0.5 for failure
        - Latency penalty: -0.1 per 60 seconds over 30s baseline
        - Tool usage bonus: +0.1 if tools were used
        - Throughput bonus: +0.05 per 10 tok/s over 5 tok/s baseline
        """
        reward = 1.0 if self.success else -0.5

        # Latency penalty (baseline 30s)
        if self.latency_seconds > 30:
            penalty = 0.1 * ((self.latency_seconds - 30) / 60)
            reward -= min(penalty, 0.5)  # Cap penalty

        # Tool usage bonus
        if self.tool_calls_made > 0:
            reward += 0.1

        # Throughput bonus (baseline 5 tok/s)
        if self.latency_seconds > 0:
            tok_per_sec = self.token_count / self.latency_seconds
            if tok_per_sec > 5:
                bonus = 0.05 * ((tok_per_sec - 5) / 10)
                reward += min(bonus, 0.2)  # Cap bonus

        # User satisfaction override
        if self.user_satisfaction is not None:
            # Blend with computed reward
            reward = 0.7 * reward + 0.3 * (self.user_satisfaction * 2 - 0.5)

        return max(-1.0, min(1.0, reward))  # Clamp to [-1, 1]


class RLModelSelector:
    """Reinforcement Learning based model selector.

    Uses Q-learning with epsilon-greedy exploration to select optimal
    providers/models based on historical performance data.

    Attributes:
        epsilon: Exploration rate (probability of random selection)
        epsilon_decay: Decay rate per selection
        epsilon_min: Minimum exploration rate
        learning_rate: Q-value update rate (alpha)
        discount_factor: Future reward discount (gamma)
        ucb_c: UCB exploration constant

    Design Decisions:
        - Mock/test providers are filtered from persistence (MOCK_PROVIDERS set)
        - Warm-up phase uses higher learning rate for faster convergence
        - Q-values clamped to [0, 1] range
    """

    # Mock/test providers to exclude from Q-table persistence and selection
    MOCK_PROVIDERS = frozenset({
        "mock", "mock_provider", "dummy", "dummy-stream", "test"
    })

    # Warm-up threshold: use higher learning rate until this many real selections
    WARMUP_THRESHOLD = 100

    def __init__(
        self,
        db_path: Optional[Path] = None,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        ucb_c: float = 2.0,
        strategy: SelectionStrategy = SelectionStrategy.EPSILON_GREEDY,
    ):
        """Initialize the RL model selector.

        Args:
            db_path: Path to ConversationStore SQLite database
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per selection
            epsilon_min: Minimum epsilon value
            learning_rate: Q-learning alpha parameter
            discount_factor: Q-learning gamma parameter
            ucb_c: UCB exploration constant
            strategy: Selection strategy to use
        """
        self.db_path = db_path or Path.home() / ".victor" / "conversations.db"
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.ucb_c = ucb_c
        self.strategy = strategy

        # In-memory Q-table (loaded from DB aggregation)
        self._q_table: Dict[str, float] = {}
        self._selection_counts: Dict[str, int] = {}
        self._total_selections: int = 0

        # Task-type aware Q-tables (provider -> task_type -> Q-value)
        # This allows different optimal providers for different task types
        self._q_table_by_task: Dict[str, Dict[str, float]] = {}
        self._task_selection_counts: Dict[str, Dict[str, int]] = {}

        # Path for persisted Q-tables
        self._q_table_path = Path.home() / ".victor" / "rl_q_tables.json"

        # Initialize from stored data
        self._load_q_values()

    @property
    def q_table_path(self) -> Path:
        """Path to persisted Q-table JSON file."""
        return self._q_table_path

    def save_q_values(self) -> bool:
        """Persist Q-tables to JSON file.

        Saves both global Q-table and task-specific Q-tables for
        persistence across sessions. Mock/test providers are filtered out.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            self._q_table_path.parent.mkdir(parents=True, exist_ok=True)

            # Filter out mock providers from persistence
            filtered_q_table = {
                k: v for k, v in self._q_table.items()
                if k not in self.MOCK_PROVIDERS
            }
            filtered_selection_counts = {
                k: v for k, v in self._selection_counts.items()
                if k not in self.MOCK_PROVIDERS
            }
            filtered_q_by_task = {
                provider: tasks
                for provider, tasks in self._q_table_by_task.items()
                if provider not in self.MOCK_PROVIDERS
            }
            filtered_task_counts = {
                provider: tasks
                for provider, tasks in self._task_selection_counts.items()
                if provider not in self.MOCK_PROVIDERS
            }

            # Calculate real selections (excluding mocks)
            real_selections = sum(
                v for k, v in self._selection_counts.items()
                if k not in self.MOCK_PROVIDERS
            )

            data = {
                "version": 1,
                "epsilon": self.epsilon,
                "total_selections": self._total_selections,
                "real_selections": real_selections,  # New field for actual usage
                "q_table": filtered_q_table,
                "selection_counts": filtered_selection_counts,
                "q_table_by_task": filtered_q_by_task,
                "task_selection_counts": filtered_task_counts,
            }

            with open(self._q_table_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved Q-tables to {self._q_table_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to save Q-tables: {e}")
            return False

    def _load_persisted_q_values(self) -> bool:
        """Load Q-tables from persisted JSON file.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self._q_table_path.exists():
            logger.debug("No persisted Q-table found")
            return False

        try:
            with open(self._q_table_path, "r") as f:
                data = json.load(f)

            version = data.get("version", 0)
            if version != 1:
                logger.warning(f"Unknown Q-table version {version}, ignoring")
                return False

            # Restore state
            self.epsilon = data.get("epsilon", self.epsilon)
            self._total_selections = data.get("total_selections", 0)
            self._q_table = data.get("q_table", {})
            self._selection_counts = data.get("selection_counts", {})
            self._q_table_by_task = data.get("q_table_by_task", {})
            self._task_selection_counts = data.get("task_selection_counts", {})

            logger.info(
                f"Loaded Q-tables for {len(self._q_table)} providers "
                f"from {self._q_table_path}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to load persisted Q-tables: {e}")
            return False

    def _load_q_values(self) -> None:
        """Load Q-values from persisted file and ConversationStore.

        Loading priority:
        1. First try loading from persisted JSON file (learned Q-values)
        2. If not found, bootstrap from ConversationStore statistics
        3. Merge any new providers from DB not in persisted file
        """
        # First, try loading persisted Q-values
        loaded_persisted = self._load_persisted_q_values()

        # If persisted Q-values exist, optionally merge with new DB providers
        if loaded_persisted and not self.db_path.exists():
            return  # Nothing more to load

        if not self.db_path.exists():
            logger.debug("No conversation DB found, starting with empty Q-table")
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Load provider statistics from normalized schema
                # Schema: sessions(provider_id FK), providers(id, name), messages(tool_name)
                rows = conn.execute(
                    """
                    SELECT
                        p.name AS provider,
                        COUNT(DISTINCT s.session_id) AS session_count,
                        COUNT(m.id) AS total_messages,
                        ROUND(COUNT(m.id) * 1.0 / NULLIF(COUNT(DISTINCT s.session_id), 0), 1) AS avg_messages,
                        ROUND(
                            CAST(COUNT(CASE WHEN m.tool_name IS NOT NULL THEN 1 END) AS REAL) * 100 /
                            NULLIF(COUNT(m.id), 0),
                            1
                        ) AS tool_capable_pct
                    FROM sessions s
                    LEFT JOIN providers p ON s.provider_id = p.id
                    LEFT JOIN messages m ON s.session_id = m.session_id
                    WHERE p.name IS NOT NULL AND p.name != ''
                    GROUP BY p.name
                    """
                ).fetchall()

                new_providers = 0
                for row in rows:
                    provider = row["provider"]
                    session_count = row["session_count"] or 0
                    avg_messages = row["avg_messages"] or 0
                    tool_capable_pct = row["tool_capable_pct"] or 0

                    # Skip if already loaded from persisted file
                    if loaded_persisted and provider in self._q_table:
                        continue

                    # Compute initial Q-value from historical performance
                    # Higher is better: more messages = engagement, tool_capable = capability
                    q_value = self._compute_initial_q(
                        session_count=session_count,
                        avg_messages=avg_messages,
                        tool_capable_pct=tool_capable_pct,
                    )

                    self._q_table[provider] = q_value
                    self._selection_counts[provider] = session_count
                    self._total_selections += session_count
                    new_providers += 1

                if loaded_persisted:
                    if new_providers > 0:
                        logger.info(
                            f"Merged {new_providers} new providers from DB "
                            f"with persisted Q-tables"
                        )
                else:
                    logger.info(
                        f"Loaded Q-values for {len(self._q_table)} providers "
                        f"from {self._total_selections} total sessions"
                    )

        except Exception as e:
            logger.warning(f"Failed to load Q-values from DB: {e}")

    def _compute_initial_q(
        self,
        session_count: int,
        avg_messages: float,
        tool_capable_pct: float,
    ) -> float:
        """Compute initial Q-value from historical statistics.

        Args:
            session_count: Number of sessions with this provider
            avg_messages: Average messages per session
            tool_capable_pct: Percentage of sessions with tool capability

        Returns:
            Initial Q-value in range [0, 1]
        """
        # Normalize components
        # More sessions = more confidence, but diminishing returns
        session_score = 1 - math.exp(-session_count / 10)

        # More messages = better engagement (capped at 20)
        message_score = min(avg_messages / 20, 1.0)

        # Tool capability is important for agentic tasks
        tool_score = tool_capable_pct / 100

        # Weighted combination
        q_value = (
            0.3 * session_score + 0.4 * message_score + 0.3 * tool_score
        )

        return q_value

    def _get_q_value(
        self,
        provider: str,
        task_type: Optional[str] = None,
    ) -> float:
        """Get Q-value for a provider, optionally task-specific.

        If task_type is provided and task-specific Q-value exists, blend
        global Q-value with task-specific value (weighted average).

        Args:
            provider: Provider name
            task_type: Optional task type for context-aware lookup

        Returns:
            Q-value (task-specific if available, else global)
        """
        global_q = self._q_table.get(provider, 0.5)

        if not task_type:
            return global_q

        # Check for task-specific Q-value
        task_q_table = self._q_table_by_task.get(provider, {})
        if task_type not in task_q_table:
            return global_q

        task_q = task_q_table[task_type]

        # Blend global and task-specific (70% task, 30% global for context)
        # This provides stability while leveraging task-specific learning
        return 0.7 * task_q + 0.3 * global_q

    def select_provider(
        self,
        available_providers: Optional[List[str]] = None,
        task_type: Optional[str] = None,
    ) -> ModelRecommendation:
        """Select a provider using the configured strategy.

        Args:
            available_providers: List of available provider names
            task_type: Optional task type for context-aware selection

        Returns:
            ModelRecommendation with selected provider and reasoning
        """
        # Store task_type for internal methods
        self._current_task_type = task_type

        # Get all known providers if none specified
        if not available_providers:
            available_providers = list(self._q_table.keys()) or ["ollama"]

        # Ensure all providers have Q-values
        for provider in available_providers:
            if provider not in self._q_table:
                self._q_table[provider] = 0.5  # Optimistic initial value
                self._selection_counts[provider] = 0

        if self.strategy == SelectionStrategy.EPSILON_GREEDY:
            return self._select_epsilon_greedy(available_providers, task_type)
        elif self.strategy == SelectionStrategy.UCB:
            return self._select_ucb(available_providers, task_type)
        elif self.strategy == SelectionStrategy.EXPLOIT_ONLY:
            return self._select_exploit(available_providers, task_type)
        else:
            return self._select_epsilon_greedy(available_providers, task_type)

    def _select_epsilon_greedy(
        self,
        providers: List[str],
        task_type: Optional[str] = None,
    ) -> ModelRecommendation:
        """Select using epsilon-greedy strategy.

        With probability epsilon, select randomly (explore).
        Otherwise, select the provider with highest Q-value (exploit).
        """
        # Exploration
        if random.random() < self.epsilon:
            selected = random.choice(providers)
            reason = f"Exploration (epsilon={self.epsilon:.2f})"
            if task_type:
                reason += f" [task={task_type}]"

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            return ModelRecommendation(
                provider=selected,
                model=None,
                confidence=0.5,
                q_value=self._get_q_value(selected, task_type),
                reason=reason,
                alternatives=self._get_alternatives(providers, selected, task_type),
            )

        # Exploitation
        return self._select_exploit(providers, task_type)

    def _select_exploit(
        self,
        providers: List[str],
        task_type: Optional[str] = None,
    ) -> ModelRecommendation:
        """Select provider with highest Q-value."""
        best_provider = max(providers, key=lambda p: self._get_q_value(p, task_type))
        q_value = self._get_q_value(best_provider, task_type)

        # Confidence based on selection count
        count = self._selection_counts.get(best_provider, 0)
        confidence = min(0.95, 0.5 + 0.45 * (1 - math.exp(-count / 20)))

        reason = f"Best Q-value ({q_value:.3f}) from {count} sessions"
        if task_type:
            reason += f" [task={task_type}]"

        return ModelRecommendation(
            provider=best_provider,
            model=None,
            confidence=confidence,
            q_value=q_value,
            reason=reason,
            alternatives=self._get_alternatives(providers, best_provider, task_type),
        )

    def _select_ucb(
        self,
        providers: List[str],
        task_type: Optional[str] = None,
    ) -> ModelRecommendation:
        """Select using Upper Confidence Bound (UCB) strategy.

        UCB score = Q(a) + c * sqrt(ln(N) / n(a))

        Balances exploitation (high Q) with exploration (low selection count).
        """
        if self._total_selections == 0:
            # No history, random selection
            selected = random.choice(providers)
            return ModelRecommendation(
                provider=selected,
                model=None,
                confidence=0.3,
                q_value=0.5,
                reason="No history available, random selection",
                alternatives=[],
            )

        ucb_scores: Dict[str, float] = {}
        log_total = math.log(self._total_selections + 1)

        for provider in providers:
            q_value = self._get_q_value(provider, task_type)
            count = self._selection_counts.get(provider, 0)

            if count == 0:
                # Infinite UCB for unexplored providers
                ucb_scores[provider] = float("inf")
            else:
                exploration_bonus = self.ucb_c * math.sqrt(log_total / count)
                ucb_scores[provider] = q_value + exploration_bonus

        best_provider = max(providers, key=lambda p: ucb_scores.get(p, 0.0))
        q_value = self._get_q_value(best_provider, task_type)
        ucb_score = ucb_scores[best_provider]
        count = self._selection_counts.get(best_provider, 0)

        # Confidence based on UCB exploration component
        if ucb_score == float("inf"):
            confidence = 0.3
            reason = "Unexplored provider (UCB=infinity)"
        else:
            exploration_component = ucb_score - q_value
            confidence = max(0.4, 0.9 - exploration_component)
            reason = f"UCB score {ucb_score:.3f} (Q={q_value:.3f}, exploration={exploration_component:.3f})"

        if task_type:
            reason += f" [task={task_type}]"

        return ModelRecommendation(
            provider=best_provider,
            model=None,
            confidence=confidence,
            q_value=q_value,
            reason=reason,
            alternatives=self._get_alternatives(providers, best_provider, task_type),
        )

    def _get_alternatives(
        self,
        providers: List[str],
        selected: str,
        task_type: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Get alternative providers ranked by Q-value."""
        alternatives = [
            (p, self._get_q_value(p, task_type))
            for p in providers
            if p != selected
        ]
        return sorted(alternatives, key=lambda x: x[1], reverse=True)[:3]

    def _get_effective_learning_rate(self, provider: str) -> float:
        """Get effective learning rate based on warm-up phase.

        During warm-up (< WARMUP_THRESHOLD real selections), uses 3x higher
        learning rate for faster convergence on initial data.

        Args:
            provider: Provider name for per-provider warm-up tracking

        Returns:
            Effective learning rate (higher during warm-up)
        """
        # Count real selections (excluding mocks)
        real_selections = sum(
            v for k, v in self._selection_counts.items()
            if k not in self.MOCK_PROVIDERS
        )

        # Provider-specific selection count
        provider_count = self._selection_counts.get(provider, 0)

        # During warm-up or for new providers, use 3x learning rate
        if real_selections < self.WARMUP_THRESHOLD or provider_count < 10:
            return min(0.5, self.learning_rate * 3)  # Cap at 0.5

        return self.learning_rate

    def update_q_value(self, reward: SessionReward) -> float:
        """Update Q-value based on session reward.

        Uses Q-learning update rule:
        Q(s,a) <- Q(s,a) + alpha * (reward - Q(s,a))

        Uses adaptive learning rate (higher during warm-up phase).
        Also updates task-specific Q-table if task_type is provided.

        Mock/test providers are skipped entirely to avoid wasting storage
        and memory on data that will never be used for real selection.

        Args:
            reward: SessionReward with performance metrics

        Returns:
            New Q-value for the provider (global), or -1.0 if mock provider
        """
        provider = reward.provider
        r = reward.reward
        task_type = reward.task_type

        # Skip mock providers entirely - don't store at all (saves memory + storage)
        if provider in self.MOCK_PROVIDERS:
            logger.debug(f"Skipping mock provider '{provider}' - not tracking")
            return -1.0

        # Get current Q-value
        old_q = self._q_table.get(provider, 0.5)

        # Get adaptive learning rate (higher during warm-up)
        effective_lr = self._get_effective_learning_rate(provider)

        # Q-learning update (simplified without next state)
        new_q = old_q + effective_lr * (r - old_q)

        # Clamp to valid range
        new_q = max(0.0, min(1.0, new_q))

        # Update global Q-table
        self._q_table[provider] = new_q
        self._selection_counts[provider] = self._selection_counts.get(provider, 0) + 1
        self._total_selections += 1

        # Update task-specific Q-table if task_type provided
        if task_type:
            if provider not in self._q_table_by_task:
                self._q_table_by_task[provider] = {}
            if provider not in self._task_selection_counts:
                self._task_selection_counts[provider] = {}

            old_task_q = self._q_table_by_task[provider].get(task_type, 0.5)
            new_task_q = old_task_q + effective_lr * (r - old_task_q)  # Use same adaptive rate
            new_task_q = max(0.0, min(1.0, new_task_q))

            self._q_table_by_task[provider][task_type] = new_task_q
            self._task_selection_counts[provider][task_type] = (
                self._task_selection_counts[provider].get(task_type, 0) + 1
            )

            logger.debug(
                f"Updated Q({provider}, {task_type}): {old_task_q:.3f} -> {new_task_q:.3f}"
            )

        logger.debug(
            f"Updated Q({provider}): {old_q:.3f} -> {new_q:.3f} "
            f"(reward={r:.3f}, count={self._selection_counts[provider]})"
        )

        # Persist Q-tables after update
        self.save_q_values()

        return new_q

    def get_provider_rankings(self) -> List[ProviderQValue]:
        """Get all providers ranked by Q-value.

        Returns:
            List of ProviderQValue objects sorted by Q-value descending
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
                ProviderQValue(
                    provider=provider,
                    q_value=q_value,
                    session_count=count,
                    avg_messages=0,  # Would need DB query
                    tool_capable_pct=0,  # Would need DB query
                    ucb_score=ucb_score,
                    confidence=confidence,
                )
            )

        return sorted(rankings, key=lambda x: x.q_value, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get RL selector statistics.

        Returns:
            Dictionary with selector state and statistics
        """
        rankings = self.get_provider_rankings()

        return {
            "strategy": self.strategy.value,
            "epsilon": self.epsilon,
            "total_selections": self._total_selections,
            "num_providers": len(self._q_table),
            "top_provider": rankings[0].provider if rankings else None,
            "top_q_value": rankings[0].q_value if rankings else 0.0,
            "learning_rate": self.learning_rate,
            "ucb_c": self.ucb_c,
            "provider_rankings": [
                {
                    "provider": r.provider,
                    "q_value": r.q_value,
                    "sessions": r.session_count,
                    "confidence": r.confidence,
                }
                for r in rankings[:5]
            ],
        }

    def reset(self) -> None:
        """Reset the selector to initial state."""
        self._q_table.clear()
        self._selection_counts.clear()
        self._total_selections = 0
        self.epsilon = 0.3
        self._load_q_values()
        logger.info("RL model selector reset")


# Convenience function for global access
_global_selector: Optional[RLModelSelector] = None


def get_model_selector() -> RLModelSelector:
    """Get the global RL model selector.

    Returns:
        Singleton RLModelSelector instance
    """
    global _global_selector
    if _global_selector is None:
        _global_selector = RLModelSelector()
    return _global_selector


def reset_model_selector() -> None:
    """Reset the global model selector."""
    global _global_selector
    if _global_selector is not None:
        _global_selector.reset()
    _global_selector = None
