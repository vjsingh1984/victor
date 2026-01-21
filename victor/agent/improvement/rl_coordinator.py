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

"""Enhanced RL coordinator with reward shaping and policy optimization.

This module extends the framework RL coordinator with advanced features
for self-improvement:

- Reward shaping based on proficiency metrics
- Policy optimization using gradient-free methods
- Hyperparameter optimization for better performance
- Integration with ProficiencyTracker for feedback

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                    EnhancedRLCoordinator                         │
│  ├─ Reward shaping                                               │
│  ├─ Policy optimization                                          │
│  ├─ Hyperparameter tuning                                        │
│  └─ Proficiency integration                                      │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Framework RLCoordinator                       │
│  ├─ Learner management                                           │
│  ├─ Outcome recording                                            │
│  └─ Recommendations                                              │
└──────────────────────────────────────────────────────────────────┘

Key Components:
- EnhancedRLCoordinator: Enhanced coordinator with new features
- Reward: Structured reward signal
- Policy: Policy representation (Q-table or neural network)
- Hyperparameters: Optimizable hyperparameters
- Action: Available actions for state

Usage:
    coordinator = EnhancedRLCoordinator()

    # Record outcome with reward shaping
    reward = coordinator.reward_shaping(outcome)
    coordinator.update_policy("code_analysis", reward)

    # Select action
    action = coordinator.select_action(
        task="code_review",
        available_actions=["ast_analyzer", "semantic_search"]
    )

    # Optimize hyperparameters
    new_params = coordinator.optimize_hyperparameters(performance_data)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from victor.agent.improvement.proficiency_tracker import (
    ProficiencyTracker,
    TaskOutcome,
    TrendDirection,
)
from victor.framework.rl import RLCoordinator, RLOutcome

logger = logging.getLogger(__name__)


class RewardShapingStrategy(str, Enum):
    """Reward shaping strategies."""

    BASELINE = "baseline"
    """Simple reward based on success/failure."""

    QUALITY_ADJUSTED = "quality_adjusted"
    """Reward adjusted by quality score."""

    PROFICIENCY_AWARE = "proficiency_aware"
    """Reward adjusted by proficiency metrics."""

    TIME_PENALTY = "time_penalty"
    """Reward with time penalty."""

    COST_AWARE = "cost_aware"
    """Reward adjusted by cost."""


@dataclass
class Reward:
    """Structured reward signal.

    Attributes:
        value: Raw reward value
        shaped_value: Shaped reward value
        components: Dictionary of reward components
        timestamp: ISO timestamp
        metadata: Additional metadata
    """

    value: float
    shaped_value: float
    components: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "shaped_value": self.shaped_value,
            "components": self.components,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Policy:
    """Policy representation.

    Attributes:
        task_type: Task type
        q_table: Q-table (state-action values)
        action_space: Available actions
        last_updated: ISO timestamp of last update
        total_updates: Total number of updates
    """

    task_type: str
    q_table: Dict[str, Dict[str, float]]
    action_space: List[str]
    last_updated: str
    total_updates: int = 0

    def get_best_action(self, state: str) -> Optional[str]:
        """Get best action for state.

        Args:
            state: Current state

        Returns:
            Best action or None if no data
        """
        if state not in self.q_table:
            return None

        actions = self.q_table[state]
        if not actions:
            return None

        return max(actions.items(), key=lambda x: x[1])[0]

    def get_action_values(self, state: str) -> Dict[str, float]:
        """Get action values for state.

        Args:
            state: Current state

        Returns:
            Dictionary mapping action to value
        """
        return self.q_table.get(state, {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "q_table": self.q_table,
            "action_space": self.action_space,
            "last_updated": self.last_updated,
            "total_updates": self.total_updates,
        }


@dataclass
class Hyperparameters:
    """Optimizable hyperparameters.

    Attributes:
        learning_rate: Learning rate for Q-learning
        exploration_rate: Exploration rate (epsilon)
        discount_factor: Discount factor (gamma)
        reward_scale: Reward scaling factor
        penalty_scale: Penalty scaling factor
    """

    learning_rate: float = 0.1
    exploration_rate: float = 0.1
    discount_factor: float = 0.9
    reward_scale: float = 1.0
    penalty_scale: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "discount_factor": self.discount_factor,
            "reward_scale": self.reward_scale,
            "penalty_scale": self.penalty_scale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Hyperparameters":
        """Create from dictionary."""
        return cls(**data)

    def mutate(self, mutation_rate: float = 0.1) -> "Hyperparameters":
        """Mutate hyperparameters for optimization.

        Args:
            mutation_rate: Mutation rate (0.0-1.0)

        Returns:
            New hyperparameters with mutations
        """
        return Hyperparameters(
            learning_rate=np.clip(
                self.learning_rate + np.random.uniform(-mutation_rate, mutation_rate),
                0.01,
                1.0,
            ),
            exploration_rate=np.clip(
                self.exploration_rate + np.random.uniform(-mutation_rate, mutation_rate),
                0.01,
                1.0,
            ),
            discount_factor=np.clip(
                self.discount_factor + np.random.uniform(-mutation_rate, mutation_rate),
                0.5,
                0.99,
            ),
            reward_scale=max(0.1, self.reward_scale + np.random.uniform(-mutation_rate, mutation_rate)),
            penalty_scale=max(0.1, self.penalty_scale + np.random.uniform(-mutation_rate, mutation_rate)),
        )


@dataclass
class Action:
    """Available action.

    Attributes:
        name: Action name
        tool: Associated tool
        expected_reward: Expected reward
        confidence: Confidence in action
    """

    name: str
    tool: str
    expected_reward: float = 0.0
    confidence: float = 0.0


class EnhancedRLCoordinator:
    """Enhanced RL coordinator with reward shaping and policy optimization.

    This coordinator extends the framework RLCoordinator with advanced
    self-improvement capabilities.

    Attributes:
        base_coordinator: Framework RL coordinator
        proficiency_tracker: Proficiency tracker for metrics
        policies: Task-type policies
        hyperparameters: Optimizable hyperparameters
        reward_strategy: Current reward shaping strategy

    Example:
        coordinator = EnhancedRLCoordinator()

        # Record outcome
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3-opus",
            task_type="code_review",
            success=True,
            quality_score=0.9
        )
        reward = coordinator.reward_shaping(outcome)
        coordinator.update_policy("code_review", reward)

        # Select action
        action = coordinator.select_action(
            task="code_review",
            available_actions=[
                Action("ast_analyzer", "ast_analyzer"),
                Action("semantic_search", "semantic_search")
            ]
        )
    """

    def __init__(
        self,
        base_coordinator: Optional[RLCoordinator] = None,
        proficiency_tracker: Optional[ProficiencyTracker] = None,
        reward_strategy: RewardShapingStrategy = RewardShapingStrategy.PROFICIENCY_AWARE,
    ):
        """Initialize EnhancedRLCoordinator.

        Args:
            base_coordinator: Framework RL coordinator
            proficiency_tracker: Proficiency tracker
            reward_strategy: Reward shaping strategy
        """
        from victor.framework.rl import get_rl_coordinator

        self._base_coordinator = base_coordinator or get_rl_coordinator()
        self._proficiency_tracker = proficiency_tracker or ProficiencyTracker()
        self._reward_strategy = reward_strategy

        # Policy storage
        self._policies: Dict[str, Policy] = {}

        # Hyperparameters
        self._hyperparameters = Hyperparameters()

        # Load existing policies and hyperparameters
        self._load_policies()
        self._load_hyperparameters()

        logger.info("EnhancedRLCoordinator: Initialized with strategy=%s", reward_strategy.value)

    @property
    def base_coordinator(self) -> RLCoordinator:
        """Get base coordinator."""
        return self._base_coordinator

    @property
    def proficiency_tracker(self) -> ProficiencyTracker:
        """Get proficiency tracker."""
        return self._proficiency_tracker

    # =========================================================================
    # Reward Shaping
    # =========================================================================

    def reward_shaping(
        self,
        outcome: RLOutcome,
        strategy: Optional[RewardShapingStrategy] = None,
    ) -> Reward:
        """Shape reward signal from outcome.

        Args:
            outcome: RL outcome
            strategy: Reward shaping strategy (uses default if None)

        Returns:
            Shaped reward
        """
        strategy = strategy or self._reward_strategy

        # Base reward
        base_reward = 1.0 if outcome.success else -1.0

        # Apply strategy
        if strategy == RewardShapingStrategy.BASELINE:
            shaped_value, components = self._baseline_reward(outcome, base_reward)
        elif strategy == RewardShapingStrategy.QUALITY_ADJUSTED:
            shaped_value, components = self._quality_adjusted_reward(outcome, base_reward)
        elif strategy == RewardShapingStrategy.PROFICIENCY_AWARE:
            shaped_value, components = self._proficiency_aware_reward(outcome, base_reward)
        elif strategy == RewardShapingStrategy.TIME_PENALTY:
            shaped_value, components = self._time_penalty_reward(outcome, base_reward)
        elif strategy == RewardShapingStrategy.COST_AWARE:
            shaped_value, components = self._cost_aware_reward(outcome, base_reward)
        else:
            shaped_value, components = base_reward, {"base": base_reward}

        # Apply hyperparameter scaling
        if outcome.success:
            shaped_value *= self._hyperparameters.reward_scale
        else:
            shaped_value *= self._hyperparameters.penalty_scale

        return Reward(
            value=base_reward,
            shaped_value=shaped_value,
            components=components,
            metadata={
                "strategy": strategy.value,
                "task_type": outcome.task_type,
                "provider": outcome.provider,
                "model": outcome.model,
            },
        )

    def _baseline_reward(self, outcome: RLOutcome, base_reward: float) -> Tuple[float, Dict[str, float]]:
        """Baseline reward (success/failure only)."""
        return base_reward, {"base": base_reward}

    def _quality_adjusted_reward(
        self, outcome: RLOutcome, base_reward: float
    ) -> Tuple[float, Dict[str, float]]:
        """Quality-adjusted reward."""
        quality_multiplier = outcome.quality_score if outcome.success else (1.0 - outcome.quality_score)
        shaped_value = base_reward * quality_multiplier
        return shaped_value, {"base": base_reward, "quality_multiplier": quality_multiplier}

    def _proficiency_aware_reward(
        self, outcome: RLOutcome, base_reward: float
    ) -> Tuple[float, Dict[str, float]]:
        """Proficiency-aware reward.

        Adjusts reward based on tool proficiency. If tool is performing poorly,
        negative rewards are amplified and positive rewards are reduced.
        """
        components = {"base": base_reward}

        # Get tool from metadata
        tool = outcome.metadata.get("tool", "")
        if not tool:
            return base_reward, components

        # Get proficiency
        proficiency = self._proficiency_tracker.get_proficiency(tool)
        if not proficiency:
            return base_reward, components

        # Adjust based on success rate
        success_rate = proficiency.success_rate
        if success_rate < 0.5:
            # Low proficiency: amplify signal
            multiplier = 1.5 if not outcome.success else 0.5
        elif success_rate > 0.8:
            # High proficiency: reduce signal (already learned)
            multiplier = 0.8 if not outcome.success else 1.2
        else:
            # Medium proficiency: neutral
            multiplier = 1.0

        shaped_value = base_reward * multiplier
        components.update(
            {
                "proficiency_multiplier": multiplier,
                "success_rate": success_rate,
            }
        )

        return shaped_value, components

    def _time_penalty_reward(
        self, outcome: RLOutcome, base_reward: float
    ) -> Tuple[float, Dict[str, float]]:
        """Time-penalty reward."""
        duration = outcome.metadata.get("duration_ms", 0) / 1000.0  # Convert to seconds

        # Penalize long durations
        time_penalty = min(duration / 10.0, 1.0)  # Max penalty at 10 seconds
        shaped_value = base_reward - (0.1 * time_penalty if outcome.success else 0)

        return shaped_value, {"base": base_reward, "time_penalty": -0.1 * time_penalty}

    def _cost_aware_reward(
        self, outcome: RLOutcome, base_reward: float
    ) -> Tuple[float, Dict[str, float]]:
        """Cost-aware reward."""
        # Get cost from metadata
        cost = outcome.metadata.get("cost", 0.0)

        # Penalize high cost
        cost_penalty = min(cost / 0.01, 0.5)  # Max penalty at $0.01
        shaped_value = base_reward - (0.2 * cost_penalty if outcome.success else 0)

        return shaped_value, {"base": base_reward, "cost_penalty": -0.2 * cost_penalty}

    # =========================================================================
    # Policy Management
    # =========================================================================

    def update_policy(self, task_type: str, reward: Reward, action: Optional[str] = None) -> None:
        """Update policy for task type using reward.

        Args:
            task_type: Task type
            reward: Reward signal
            action: Action that generated reward
        """
        # Get or create policy
        if task_type not in self._policies:
            self._policies[task_type] = Policy(
                task_type=task_type,
                q_table={},
                action_space=[],
                last_updated=datetime.now().isoformat(),
            )

        policy = self._policies[task_type]

        # Default state (can be extended to support multiple states)
        state = "default"

        if state not in policy.q_table:
            policy.q_table[state] = {}

        # Update Q-value
        if action:
            if action not in policy.q_table[state]:
                policy.q_table[state][action] = 0.0

            # Q-learning update
            current_value = policy.q_table[state][action]
            learning_rate = self._hyperparameters.learning_rate
            new_value = current_value + learning_rate * (reward.shaped_value - current_value)
            policy.q_table[state][action] = new_value

        policy.last_updated = datetime.now().isoformat()
        policy.total_updates += 1

        # Persist policy
        self._save_policy(task_type)

        logger.debug(
            "EnhancedRLCoordinator: Updated policy for task_type=%s, reward=%s, action=%s",
            task_type,
            reward.shaped_value,
            action,
        )

    def get_policy(self, task_type: str) -> Optional[Policy]:
        """Get policy for task type.

        Args:
            task_type: Task type

        Returns:
            Policy or None if not found
        """
        return self._policies.get(task_type)

    def select_action(
        self, task: str, available_actions: List[Action], state: str = "default"
    ) -> Optional[Action]:
        """Select action using policy.

        Args:
            task: Task type
            available_actions: Available actions
            state: Current state

        Returns:
            Selected action or None
        """
        # Get policy
        policy = self.get_policy(task)
        if not policy:
            # No policy, return random action
            if available_actions:
                return np.random.choice(available_actions)
            return None

        # Epsilon-greedy selection
        if np.random.random() < self._hyperparameters.exploration_rate:
            # Explore: random action
            return np.random.choice(available_actions) if available_actions else None

        # Exploit: best action from policy
        action_values = policy.get_action_values(state)
        if not action_values:
            return np.random.choice(available_actions) if available_actions else None

        # Filter available actions
        available_names = {a.name for a in available_actions}
        best_action_name = None
        best_value = float("-inf")

        for name, value in action_values.items():
            if name in available_names and value > best_value:
                best_action_name = name
                best_value = value

        if best_action_name:
            for action in available_actions:
                if action.name == best_action_name:
                    action.expected_reward = best_value
                    return action

        # Fallback to random
        return np.random.choice(available_actions) if available_actions else None

    # =========================================================================
    # Hyperparameter Optimization
    # =========================================================================

    def optimize_hyperparameters(
        self,
        performance_data: List[Dict[str, Any]],
        optimization_iterations: int = 10,
    ) -> Hyperparameters:
        """Optimize hyperparameters using performance data.

        Args:
            performance_data: List of performance metrics
            optimization_iterations: Number of optimization iterations

        Returns:
            Optimized hyperparameters
        """
        if not performance_data:
            logger.warning("EnhancedRLCoordinator: No performance data for optimization")
            return self._hyperparameters

        # Evaluate current hyperparameters
        best_score = self._evaluate_hyperparameters(self._hyperparameters, performance_data)
        best_hyperparams = self._hyperparameters

        logger.info(
            "EnhancedRLCoordinator: Starting hyperparameter optimization, current_score=%s",
            best_score,
        )

        # Simple random search (can be replaced with more sophisticated methods)
        for i in range(optimization_iterations):
            # Mutate hyperparameters
            candidate = best_hyperparams.mutate(mutation_rate=0.2)

            # Evaluate
            score = self._evaluate_hyperparameters(candidate, performance_data)

            # Keep if better
            if score > best_score:
                best_score = score
                best_hyperparams = candidate
                logger.debug(
                    "EnhancedRLCoordinator: Iteration %d: New best score=%s",
                    i + 1,
                    best_score,
                )

        # Update hyperparameters
        self._hyperparameters = best_hyperparams
        self._save_hyperparameters()

        logger.info(
            "EnhancedRLCoordinator: Optimization complete, final_score=%s, hyperparams=%s",
            best_score,
            best_hyperparams.to_dict(),
        )

        return best_hyperparams

    def _evaluate_hyperparameters(
        self, hyperparams: Hyperparameters, performance_data: List[Dict[str, Any]]
    ) -> float:
        """Evaluate hyperparameters on performance data.

        Args:
            hyperparams: Hyperparameters to evaluate
            performance_data: Performance metrics

        Returns:
            Score (higher is better)
        """
        total_score = 0.0

        for data in performance_data:
            success = data.get("success", False)
            quality = data.get("quality_score", 0.0)

            # Simple scoring: success * quality
            score = 1.0 if success else 0.0
            score *= quality

            total_score += score

        return total_score / len(performance_data) if performance_data else 0.0

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load_policies(self) -> None:
        """Load policies from database."""
        cursor = self._base_coordinator.db.cursor()

        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rl_policies'")
            if not cursor.fetchone():
                # Table doesn't exist, create it
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rl_policies (
                        task_type TEXT PRIMARY KEY,
                        q_table TEXT,
                        action_space TEXT,
                        last_updated TEXT,
                        total_updates INTEGER
                    )
                """
                )
                self._base_coordinator.db.commit()
                return

            # Load policies
            cursor.execute("SELECT task_type, q_table, action_space, last_updated, total_updates FROM rl_policies")
            rows = cursor.fetchall()

            import json

            for row in rows:
                task_type, q_table_str, action_space_str, last_updated, total_updates = row
                self._policies[task_type] = Policy(
                    task_type=task_type,
                    q_table=json.loads(q_table_str) if q_table_str else {},
                    action_space=json.loads(action_space_str) if action_space_str else [],
                    last_updated=last_updated,
                    total_updates=total_updates or 0,
                )

            logger.debug("EnhancedRLCoordinator: Loaded %d policies", len(self._policies))

        except Exception as e:
            logger.error("EnhancedRLCoordinator: Failed to load policies: %e")

    def _save_policy(self, task_type: str) -> None:
        """Save policy to database.

        Args:
            task_type: Task type
        """
        if task_type not in self._policies:
            return

        policy = self._policies[task_type]
        cursor = self._base_coordinator.db.cursor()

        try:
            import json

            cursor.execute(
                """
                INSERT OR REPLACE INTO rl_policies
                (task_type, q_table, action_space, last_updated, total_updates)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    task_type,
                    json.dumps(policy.q_table),
                    json.dumps(policy.action_space),
                    policy.last_updated,
                    policy.total_updates,
                ),
            )
            self._base_coordinator.db.commit()

        except Exception as e:
            logger.error("EnhancedRLCoordinator: Failed to save policy for %s: %e", task_type, e)

    def _save_hyperparameters(self) -> None:
        """Save hyperparameters to database."""
        cursor = self._base_coordinator.db.cursor()

        try:
            import json

            # Ensure table exists
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rl_hyperparameters (
                    id INTEGER PRIMARY KEY,
                    hyperparams TEXT,
                    last_updated TEXT
                )
            """
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO rl_hyperparameters (id, hyperparams, last_updated)
                VALUES (1, ?, ?)
            """,
                (json.dumps(self._hyperparameters.to_dict()), datetime.now().isoformat()),
            )
            self._base_coordinator.db.commit()

        except Exception as e:
            logger.error("EnhancedRLCoordinator: Failed to save hyperparameters: %e")

    def _load_hyperparameters(self) -> None:
        """Load hyperparameters from database."""
        cursor = self._base_coordinator.db.cursor()

        try:
            import json

            # Ensure table exists
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rl_hyperparameters (
                    id INTEGER PRIMARY KEY,
                    hyperparams TEXT,
                    last_updated TEXT
                )
            """
            )

            cursor.execute("SELECT hyperparams FROM rl_hyperparameters WHERE id = 1")
            row = cursor.fetchone()

            if row and row[0]:
                params_dict = json.loads(row[0])
                self._hyperparameters = Hyperparameters.from_dict(params_dict)
                logger.debug("EnhancedRLCoordinator: Loaded hyperparameters from database")

        except Exception as e:
            logger.debug("EnhancedRLCoordinator: No saved hyperparameters found, using defaults")

    def export_metrics(self) -> Dict[str, Any]:
        """Export coordinator metrics.

        Returns:
            Dictionary with metrics
        """
        return {
            "policies": {name: p.to_dict() for name, p in self._policies.items()},
            "hyperparameters": self._hyperparameters.to_dict(),
            "reward_strategy": self._reward_strategy.value,
        }

    def get_proficiency_tracker(self) -> ProficiencyTracker:
        """Get proficiency tracker.

        Returns:
            Proficiency tracker instance
        """
        return self._proficiency_tracker
