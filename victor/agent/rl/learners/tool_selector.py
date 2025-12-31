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

"""RL learner for adaptive tool selection using contextual bandits.

This learner uses a contextual bandit approach with epsilon-greedy exploration
to learn optimal tool rankings based on historical execution performance.

Strategy:
- Contextual bandit with epsilon-greedy exploration (ε=0.1 default, conservative)
- Context: task_type, conversation_stage, recent_tools
- Reward based on: tool success, task completion, grounding score
- Q-values per (tool_name, task_type) for context-aware selection
- Slow, stable learning (learning_rate=0.05)
- Enabled by default for all users

HIGH PRIORITY: Sprint 1 of RL Enhancement Plan
"""

import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from victor.agent.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


class ToolSelectorLearner(BaseLearner):
    """Learn optimal tool selection using contextual bandits.

    Uses contextual bandit approach to learn tool rankings based on:
    - Task type context (analysis, action, create, edit, search)
    - Tool execution success rates
    - Task completion correlation

    Attributes:
        name: Always "tool_selector"
        db: SQLite database connection
        learning_rate: Q-value update rate (alpha), default 0.05 (conservative)
        epsilon: Exploration rate (0.1 = conservative)
        min_samples: Minimum samples before confident recommendation
    """

    # Default tool Q-value (optimistic for exploration)
    DEFAULT_Q_VALUE = 0.5

    # Minimum samples before providing confident recommendations
    MIN_SAMPLES_FOR_CONFIDENCE = 20

    # Conservative exploration (90% exploit, 10% explore)
    DEFAULT_EPSILON = 0.1

    # Slow, stable learning
    DEFAULT_LEARNING_RATE = 0.05

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        provider_adapter: Optional[Any] = None,
        epsilon: float = DEFAULT_EPSILON,
    ):
        """Initialize tool selector learner.

        Args:
            name: Learner name (should be "tool_selector")
            db_connection: SQLite database connection
            learning_rate: Q-value update rate (default 0.05, conservative)
            provider_adapter: Optional provider adapter
            epsilon: Exploration rate (default 0.1, conservative)
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        self.epsilon = epsilon

        # In-memory caches for fast access
        self._tool_q_values: Dict[str, float] = {}  # tool_name -> global Q-value
        self._tool_task_q_values: Dict[str, Dict[str, float]] = (
            {}
        )  # tool_name -> {task_type -> Q-value}
        self._tool_selection_counts: Dict[str, int] = {}  # tool_name -> count
        self._tool_success_counts: Dict[str, int] = {}  # tool_name -> success count
        self._total_selections: int = 0

        # Load state from database
        self._load_state()

    def _ensure_tables(self) -> None:
        """Create tables for tool selector stats."""
        cursor = self.db.cursor()

        # Global tool Q-values table (uses Tables constants)
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.RL_TOOL_Q} (
                tool_name TEXT PRIMARY KEY,
                q_value REAL NOT NULL,
                selection_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                last_updated TEXT
            )
            """
        )

        # Task-specific tool Q-values table
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.RL_TOOL_TASK} (
                tool_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                q_value REAL NOT NULL,
                selection_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                last_updated TEXT,
                PRIMARY KEY (tool_name, task_type)
            )
            """
        )

        # Tool execution outcomes table (for detailed analysis)
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.RL_TOOL_OUTCOME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                success INTEGER NOT NULL,
                quality_score REAL,
                reward REAL,
                metadata TEXT,
                timestamp TEXT
            )
            """
        )

        # Indexes
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_rl_tool_q_task
            ON {Tables.RL_TOOL_TASK}(tool_name, task_type)
            """
        )
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_rl_tool_outcome_tool
            ON {Tables.RL_TOOL_OUTCOME}(tool_name, task_type)
            """
        )

        self.db.commit()
        logger.debug("RL: tool_selector tables ensured")

    def _load_state(self) -> None:
        """Load state from database."""
        cursor = self.db.cursor()

        # Load global Q-values
        try:
            cursor.execute(f"SELECT * FROM {Tables.RL_TOOL_Q}")
            for row in cursor.fetchall():
                stats = dict(row)
                tool_name = stats["tool_name"]
                self._tool_q_values[tool_name] = stats["q_value"]
                self._tool_selection_counts[tool_name] = stats["selection_count"]
                self._tool_success_counts[tool_name] = stats["success_count"]
                self._total_selections += stats["selection_count"]
        except Exception as e:
            logger.debug(f"RL: Could not load global Q-values: {e}")

        # Load task-specific Q-values
        try:
            cursor.execute(f"SELECT * FROM {Tables.RL_TOOL_TASK}")
            for row in cursor.fetchall():
                stats = dict(row)
                tool_name = stats["tool_name"]
                task_type = stats["task_type"]

                if tool_name not in self._tool_task_q_values:
                    self._tool_task_q_values[tool_name] = {}

                self._tool_task_q_values[tool_name][task_type] = stats["q_value"]
        except Exception as e:
            logger.debug(f"RL: Could not load task Q-values: {e}")

        if self._tool_q_values:
            logger.info(f"RL: Loaded {len(self._tool_q_values)} tool Q-values from database")

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record tool execution outcome and update Q-values.

        Expected metadata:
        - tool_name: Name of the tool executed
        - tool_success: Whether tool execution succeeded
        - task_completed: Whether the overall task completed
        - grounding_score: Score from grounding verifier (0-1)
        - efficiency_score: Time/resource efficiency (0-1)

        Args:
            outcome: Outcome with tool execution data
        """
        tool_name = outcome.metadata.get("tool_name")
        if not tool_name:
            logger.debug("RL: tool_selector outcome missing tool_name, skipping")
            return

        task_type = outcome.task_type or "default"

        # Compute reward from implicit signals
        reward = self._compute_reward(outcome)

        # Get old Q-values
        old_global_q = self._tool_q_values.get(tool_name, self.DEFAULT_Q_VALUE)

        if tool_name not in self._tool_task_q_values:
            self._tool_task_q_values[tool_name] = {}
        old_task_q = self._tool_task_q_values[tool_name].get(task_type, self.DEFAULT_Q_VALUE)

        # Update Q-values: Q(s,a) <- Q(s,a) + alpha * (reward - Q(s,a))
        new_global_q = self._clamp_q_value(
            old_global_q + self.learning_rate * (reward - old_global_q)
        )
        new_task_q = self._clamp_q_value(old_task_q + self.learning_rate * (reward - old_task_q))

        # Update caches
        self._tool_q_values[tool_name] = new_global_q
        self._tool_task_q_values[tool_name][task_type] = new_task_q
        self._tool_selection_counts[tool_name] = self._tool_selection_counts.get(tool_name, 0) + 1
        self._total_selections += 1

        if outcome.success:
            self._tool_success_counts[tool_name] = self._tool_success_counts.get(tool_name, 0) + 1

        # Persist to database
        self._save_to_db(tool_name, task_type, outcome, reward)

        logger.debug(
            f"RL: Tool '{tool_name}' Q-value: {old_global_q:.3f} -> {new_global_q:.3f} "
            f"(reward={reward:.3f}, task={task_type})"
        )

    def _save_to_db(
        self,
        tool_name: str,
        task_type: str,
        outcome: RLOutcome,
        reward: float,
    ) -> None:
        """Save Q-values and outcome to database."""
        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        # Save global Q-value
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_TOOL_Q}
            (tool_name, q_value, selection_count, success_count, last_updated)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                tool_name,
                self._tool_q_values[tool_name],
                self._tool_selection_counts[tool_name],
                self._tool_success_counts.get(tool_name, 0),
                timestamp,
            ),
        )

        # Save task-specific Q-value
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_TOOL_TASK}
            (tool_name, task_type, q_value, selection_count, success_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                tool_name,
                task_type,
                self._tool_task_q_values[tool_name][task_type],
                self._tool_selection_counts[tool_name],
                self._tool_success_counts.get(tool_name, 0),
                timestamp,
            ),
        )

        # Save detailed outcome for analysis
        cursor.execute(
            f"""
            INSERT INTO {Tables.RL_TOOL_OUTCOME}
            (tool_name, task_type, success, quality_score, reward, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tool_name,
                task_type,
                1 if outcome.success else 0,
                outcome.quality_score,
                reward,
                json.dumps(outcome.metadata),
                timestamp,
            ),
        )

        self.db.commit()

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get tool ranking boost recommendation for given context.

        Note: This method is called per-tool to get a confidence boost.
        The provider param is overloaded to contain the tool_name.

        Args:
            provider: Tool name (overloaded parameter for API compatibility)
            model: Not used (for signature compatibility)
            task_type: Task type for context-aware selection

        Returns:
            Recommendation with confidence boost value, or None
        """
        tool_name = provider  # provider param overloaded to pass tool_name

        # Get blended Q-value (70% task-specific + 30% global)
        q_value = self._get_blended_q_value(tool_name, task_type)

        # Get selection count
        count = self._tool_selection_counts.get(tool_name, 0)

        # Compute confidence based on sample size
        if count < self.MIN_SAMPLES_FOR_CONFIDENCE:
            confidence = 0.3 + 0.2 * (count / self.MIN_SAMPLES_FOR_CONFIDENCE)
            is_baseline = True
            reason = f"Low confidence (n={count})"
        else:
            # Confidence increases with samples, caps at 0.95
            confidence = min(0.95, 0.5 + 0.45 * (1 - math.exp(-count / 50)))
            is_baseline = False

            # Success rate for explanation
            success_count = self._tool_success_counts.get(tool_name, 0)
            success_rate = success_count / count if count > 0 else 0
            reason = f"Q={q_value:.3f}, success_rate={success_rate:.1%}"

        return RLRecommendation(
            value=q_value,  # Q-value as confidence boost
            confidence=confidence,
            reason=reason,
            sample_size=count,
            is_baseline=is_baseline,
        )

    def get_tool_rankings(
        self, available_tools: List[str], task_type: str
    ) -> List[Tuple[str, float, float]]:
        """Get ranked tools with RL-boosted scores.

        Args:
            available_tools: List of available tool names
            task_type: Current task type

        Returns:
            List of (tool_name, q_value, confidence) tuples, sorted by Q-value desc
        """
        rankings = []

        for tool_name in available_tools:
            q_value = self._get_blended_q_value(tool_name, task_type)
            count = self._tool_selection_counts.get(tool_name, 0)

            # Confidence based on sample size
            if count < self.MIN_SAMPLES_FOR_CONFIDENCE:
                confidence = 0.3 + 0.2 * (count / self.MIN_SAMPLES_FOR_CONFIDENCE)
            else:
                confidence = min(0.95, 0.5 + 0.45 * (1 - math.exp(-count / 50)))

            rankings.append((tool_name, q_value, confidence))

        # Sort by Q-value descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def should_explore(self) -> bool:
        """Determine if we should explore (random tool boost) or exploit.

        Returns:
            True if should explore, False if should exploit
        """
        import random

        return random.random() < self.epsilon

    def _get_blended_q_value(self, tool_name: str, task_type: str) -> float:
        """Get blended Q-value (70% task-specific + 30% global).

        Args:
            tool_name: Name of the tool
            task_type: Task type

        Returns:
            Blended Q-value
        """
        global_q = self._tool_q_values.get(tool_name, self.DEFAULT_Q_VALUE)

        if tool_name not in self._tool_task_q_values:
            return global_q

        task_q = self._tool_task_q_values[tool_name].get(task_type)
        if task_q is None:
            return global_q

        # Blend: 70% task-specific, 30% global
        return 0.7 * task_q + 0.3 * global_q

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward from implicit signals.

        Reward based on (no user input needed):
        - Tool success (40%): Did tool execute without errors?
        - Task completion (30%): Did session complete?
        - Grounding score (20%): Did response pass verification?
        - Efficiency (10%): Was it faster than baseline?

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (0.0 to 1.0)
        """
        reward = 0.0

        # Tool success (40% weight)
        tool_success = outcome.metadata.get("tool_success", outcome.success)
        reward += 0.4 * (1.0 if tool_success else 0.0)

        # Task completion (30% weight)
        task_completed = outcome.metadata.get("task_completed", outcome.success)
        reward += 0.3 * (1.0 if task_completed else 0.0)

        # Grounding score (20% weight)
        grounding_score = outcome.metadata.get("grounding_score", outcome.quality_score)
        reward += 0.2 * grounding_score

        # Efficiency bonus (10% weight)
        efficiency = outcome.metadata.get("efficiency_score", 0.5)
        reward += 0.1 * efficiency

        return self._clamp_q_value(reward)

    def _clamp_q_value(self, value: float) -> float:
        """Clamp Q-value to valid range [0.0, 1.0].

        Args:
            value: Value to clamp

        Returns:
            Clamped value
        """
        return max(0.0, min(1.0, value))

    def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """Get statistics for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dictionary with tool statistics
        """
        count = self._tool_selection_counts.get(tool_name, 0)
        success_count = self._tool_success_counts.get(tool_name, 0)
        q_value = self._tool_q_values.get(tool_name, self.DEFAULT_Q_VALUE)

        return {
            "tool_name": tool_name,
            "q_value": q_value,
            "selection_count": count,
            "success_count": success_count,
            "success_rate": success_count / count if count > 0 else 0.0,
            "task_q_values": self._tool_task_q_values.get(tool_name, {}),
        }

    def export_metrics(self) -> Dict[str, Any]:
        """Export learner metrics for monitoring.

        Returns:
            Dictionary with learner stats
        """
        return {
            "learner": self.name,
            "total_tools_tracked": len(self._tool_q_values),
            "total_selections": self._total_selections,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "top_tools": self.get_tool_rankings(list(self._tool_q_values.keys())[:10], "default"),
        }
