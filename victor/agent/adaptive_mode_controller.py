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

"""Adaptive mode controller with reinforcement learning for optimal transitions.

Architecture (State Machine + Q-Learning patterns):
- Tracks conversation state (explore, plan, build, review)
- Learns optimal mode transitions from user feedback
- Optimizes tool budgets based on task type and success patterns
- Supports continuation context management

Key Features:
1. Q-Learning for Mode Transitions:
   - State: (current_mode, task_type, tool_usage_ratio, quality_score)
   - Action: (transition_to_mode, adjust_tool_budget)
   - Reward: User satisfaction, task completion, quality scores

2. Tool Budget Optimization:
   - Learns optimal budgets per task type
   - Adjusts based on success/failure patterns
   - Respects user-defined limits

3. Continuation Management:
   - Tracks context across continuations
   - Preserves learned state between sessions
   - Intelligent context summarization

4. Real-time Adaptation:
   - Adjusts during conversation based on patterns
   - Early termination detection
   - Loop detection and recovery

Usage:
    controller = AdaptiveModeController(profile_name="local-qwen")

    # Get recommended action
    action = controller.get_recommended_action(
        current_mode="explore",
        task_type="analysis",
        tool_calls_made=5,
        tool_budget=10,
        quality_score=0.7,
    )

    # Record outcome for learning
    controller.record_outcome(
        success=True,
        quality_score=0.85,
        user_satisfied=True,
    )
"""

import hashlib
import json
import logging
import math
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    """Agent operation modes."""

    EXPLORE = "explore"  # Understanding codebase
    PLAN = "plan"  # Creating implementation plan
    BUILD = "build"  # Writing/modifying code
    REVIEW = "review"  # Reviewing changes
    COMPLETE = "complete"  # Task finished


class TransitionTrigger(Enum):
    """Triggers for mode transitions."""

    USER_REQUEST = "user_request"  # User explicitly requested
    TASK_COMPLETE = "task_complete"  # Task objective achieved
    BUDGET_LOW = "budget_low"  # Tool/iteration budget low
    QUALITY_THRESHOLD = "quality_threshold"  # Quality score threshold met
    PATTERN_DETECTED = "pattern_detected"  # Learned pattern suggests transition
    LOOP_DETECTED = "loop_detected"  # Repetitive behavior detected
    ERROR_RECOVERY = "error_recovery"  # Recovering from error


@dataclass
class ModeState:
    """Current state for mode decision making."""

    mode: AgentMode
    task_type: str
    tool_calls_made: int
    tool_budget: int
    iteration_count: int
    iteration_budget: int
    quality_score: float
    grounding_score: float
    time_in_mode_seconds: float
    recent_tool_success_rate: float

    def to_state_key(self) -> str:
        """Convert to discrete state key for Q-learning."""
        # Discretize continuous values
        tool_ratio = self._discretize_ratio(self.tool_calls_made / max(self.tool_budget, 1))
        iter_ratio = self._discretize_ratio(self.iteration_count / max(self.iteration_budget, 1))
        quality_bucket = self._discretize_quality(self.quality_score)
        grounding_bucket = self._discretize_quality(self.grounding_score)

        return f"{self.mode.value}:{self.task_type}:{tool_ratio}:{iter_ratio}:{quality_bucket}:{grounding_bucket}"

    def _discretize_ratio(self, ratio: float) -> str:
        """Discretize a ratio to buckets."""
        if ratio < 0.25:
            return "low"
        elif ratio < 0.5:
            return "mid_low"
        elif ratio < 0.75:
            return "mid_high"
        else:
            return "high"

    def _discretize_quality(self, score: float) -> str:
        """Discretize quality score."""
        if score < 0.4:
            return "poor"
        elif score < 0.6:
            return "fair"
        elif score < 0.8:
            return "good"
        else:
            return "excellent"


@dataclass
class ModeAction:
    """Action to take in the current state."""

    target_mode: AgentMode
    adjust_tool_budget: int = 0  # +/- adjustment
    should_continue: bool = True
    reason: str = ""
    confidence: float = 0.5

    def __repr__(self) -> str:
        return (
            f"ModeAction(target={self.target_mode.value}, "
            f"budget_adj={self.adjust_tool_budget:+d}, "
            f"continue={self.should_continue}, conf={self.confidence:.2f})"
        )


@dataclass
class TransitionEvent:
    """Record of a mode transition."""

    from_mode: AgentMode
    to_mode: AgentMode
    trigger: TransitionTrigger
    state_before: ModeState
    action_taken: ModeAction
    timestamp: datetime = field(default_factory=datetime.now)

    # Outcome (filled in after transition)
    outcome_success: Optional[bool] = None
    outcome_quality: Optional[float] = None
    reward: Optional[float] = None


class QLearningStore:
    """SQLite-backed Q-learning storage for mode transitions.

    Stores Q-values for state-action pairs and learns from outcomes.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the Q-learning store."""
        if db_path is None:
            from victor.config.settings import get_project_paths

            paths = get_project_paths()
            db_path = paths.project_victor_dir / "mode_learning.db"

        self.db_path = db_path
        self._initialized = False

        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1  # Epsilon for epsilon-greedy

    def _ensure_initialized(self) -> None:
        """Ensure database tables exist."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Q-values table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS q_values (
                    state_key TEXT NOT NULL,
                    action_key TEXT NOT NULL,
                    q_value REAL NOT NULL DEFAULT 0.0,
                    visit_count INTEGER NOT NULL DEFAULT 0,
                    last_updated TEXT NOT NULL,
                    PRIMARY KEY (state_key, action_key)
                )
            """
            )

            # Transition history for analysis
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_name TEXT NOT NULL,
                    from_mode TEXT NOT NULL,
                    to_mode TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    state_key TEXT NOT NULL,
                    action_key TEXT NOT NULL,
                    reward REAL,
                    timestamp TEXT NOT NULL
                )
            """
            )

            # Task-type specific statistics
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_stats (
                    task_type TEXT PRIMARY KEY,
                    optimal_tool_budget INTEGER DEFAULT 10,
                    avg_quality_score REAL DEFAULT 0.5,
                    avg_completion_rate REAL DEFAULT 0.5,
                    sample_count INTEGER DEFAULT 0,
                    last_updated TEXT NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_q_state ON q_values(state_key)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_history_profile
                ON transition_history(profile_name, timestamp)
            """
            )

        self._initialized = True

    def get_q_value(self, state_key: str, action_key: str) -> float:
        """Get Q-value for a state-action pair."""
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT q_value FROM q_values WHERE state_key = ? AND action_key = ?",
                (state_key, action_key),
            ).fetchone()

            return row[0] if row else 0.0

    def get_all_actions(self, state_key: str) -> Dict[str, float]:
        """Get all Q-values for a state."""
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT action_key, q_value FROM q_values WHERE state_key = ?", (state_key,)
            ).fetchall()

            return {row["action_key"]: row["q_value"] for row in rows}

    def update_q_value(
        self,
        state_key: str,
        action_key: str,
        reward: float,
        next_state_key: Optional[str] = None,
    ) -> float:
        """Update Q-value using Q-learning update rule.

        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))

        Returns:
            New Q-value
        """
        self._ensure_initialized()

        # Get current Q-value
        current_q = self.get_q_value(state_key, action_key)

        # Get max Q-value for next state (if provided)
        max_next_q = 0.0
        if next_state_key:
            next_actions = self.get_all_actions(next_state_key)
            if next_actions:
                max_next_q = max(next_actions.values())

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        # Store updated Q-value
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO q_values (state_key, action_key, q_value, visit_count, last_updated)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(state_key, action_key) DO UPDATE SET
                    q_value = ?,
                    visit_count = visit_count + 1,
                    last_updated = ?
            """,
                (
                    state_key,
                    action_key,
                    new_q,
                    datetime.now().isoformat(),
                    new_q,
                    datetime.now().isoformat(),
                ),
            )

        return new_q

    def record_transition(
        self,
        profile_name: str,
        from_mode: AgentMode,
        to_mode: AgentMode,
        trigger: TransitionTrigger,
        state_key: str,
        action_key: str,
        reward: Optional[float] = None,
    ) -> None:
        """Record a transition for analysis."""
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO transition_history
                (profile_name, from_mode, to_mode, trigger, state_key, action_key, reward, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    profile_name,
                    from_mode.value,
                    to_mode.value,
                    trigger.value,
                    state_key,
                    action_key,
                    reward,
                    datetime.now().isoformat(),
                ),
            )

    def update_task_stats(
        self,
        task_type: str,
        tool_budget_used: int,
        quality_score: float,
        completed: bool,
    ) -> None:
        """Update task-type statistics."""
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            # Get current stats
            row = conn.execute(
                "SELECT * FROM task_stats WHERE task_type = ?", (task_type,)
            ).fetchone()

            if row:
                # Exponential moving average
                alpha = 0.1
                count = row[4] + 1
                new_budget = int((1 - alpha) * row[1] + alpha * tool_budget_used)
                new_quality = (1 - alpha) * row[2] + alpha * quality_score
                completion_rate = (1 - alpha) * row[3] + alpha * (1.0 if completed else 0.0)
            else:
                count = 1
                new_budget = tool_budget_used
                new_quality = quality_score
                completion_rate = 1.0 if completed else 0.0

            conn.execute(
                """
                INSERT INTO task_stats
                (task_type, optimal_tool_budget, avg_quality_score, avg_completion_rate, sample_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_type) DO UPDATE SET
                    optimal_tool_budget = ?,
                    avg_quality_score = ?,
                    avg_completion_rate = ?,
                    sample_count = ?,
                    last_updated = ?
            """,
                (
                    task_type,
                    new_budget,
                    new_quality,
                    completion_rate,
                    count,
                    datetime.now().isoformat(),
                    new_budget,
                    new_quality,
                    completion_rate,
                    count,
                    datetime.now().isoformat(),
                ),
            )

    def get_task_stats(self, task_type: str) -> Dict[str, Any]:
        """Get statistics for a task type."""
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM task_stats WHERE task_type = ?", (task_type,)
            ).fetchone()

            if row:
                return {
                    "task_type": row["task_type"],
                    "optimal_tool_budget": row["optimal_tool_budget"],
                    "avg_quality_score": row["avg_quality_score"],
                    "avg_completion_rate": row["avg_completion_rate"],
                    "sample_count": row["sample_count"],
                }

            # Default values
            return {
                "task_type": task_type,
                "optimal_tool_budget": 10,
                "avg_quality_score": 0.5,
                "avg_completion_rate": 0.5,
                "sample_count": 0,
            }


class AdaptiveModeController:
    """Controller for adaptive mode management with reinforcement learning.

    Uses Q-learning to optimize:
    - Mode transitions (explore → plan → build → review)
    - Tool budget allocation
    - Continuation decisions
    """

    # Valid mode transitions
    VALID_TRANSITIONS = {
        AgentMode.EXPLORE: [AgentMode.PLAN, AgentMode.BUILD, AgentMode.COMPLETE],
        AgentMode.PLAN: [AgentMode.BUILD, AgentMode.EXPLORE, AgentMode.COMPLETE],
        AgentMode.BUILD: [AgentMode.REVIEW, AgentMode.EXPLORE, AgentMode.COMPLETE],
        AgentMode.REVIEW: [AgentMode.BUILD, AgentMode.COMPLETE],
        AgentMode.COMPLETE: [],
    }

    # Default tool budgets by task type
    DEFAULT_TOOL_BUDGETS = {
        "code_generation": 3,
        "create_simple": 2,
        "create": 5,
        "edit": 5,
        "search": 6,
        "action": 10,
        "analysis_deep": 15,
        "analyze": 8,
        "design": 20,
        "general": 8,
    }

    def __init__(
        self,
        profile_name: str = "default",
        q_store: Optional[QLearningStore] = None,
    ):
        """Initialize the adaptive mode controller.

        Args:
            profile_name: Profile name for tracking
            q_store: Optional Q-learning store (creates one if not provided)
        """
        self.profile_name = profile_name
        self._q_store = q_store or QLearningStore()

        # Current state tracking
        self._current_state: Optional[ModeState] = None
        self._current_action: Optional[ModeAction] = None
        self._pending_transition: Optional[TransitionEvent] = None

        # Session tracking
        self._session_start = datetime.now()
        self._mode_history: List[Tuple[AgentMode, datetime]] = []
        self._total_reward = 0.0

    def get_recommended_action(
        self,
        current_mode: str,
        task_type: str,
        tool_calls_made: int,
        tool_budget: int,
        iteration_count: int = 0,
        iteration_budget: int = 20,
        quality_score: float = 0.5,
        grounding_score: float = 0.5,
        time_in_mode_seconds: float = 0.0,
        recent_tool_success_rate: float = 1.0,
    ) -> ModeAction:
        """Get recommended action based on current state.

        Uses epsilon-greedy policy: explore with probability epsilon,
        otherwise exploit best known action.

        Args:
            current_mode: Current agent mode
            task_type: Task type being processed
            tool_calls_made: Number of tool calls made
            tool_budget: Total tool call budget
            iteration_count: Current iteration count
            iteration_budget: Total iteration budget
            quality_score: Current quality score (0-1)
            grounding_score: Grounding verification score (0-1)
            time_in_mode_seconds: Time spent in current mode
            recent_tool_success_rate: Recent tool call success rate

        Returns:
            Recommended ModeAction
        """
        # Build current state
        # Ensure current_mode is a string (could be int from ConversationStage.value)
        try:
            mode_str = str(current_mode).lower() if current_mode else "explore"
            mode_enum = AgentMode(mode_str)
        except ValueError:
            mode_enum = AgentMode.EXPLORE

        state = ModeState(
            mode=mode_enum,
            task_type=task_type,
            tool_calls_made=tool_calls_made,
            tool_budget=tool_budget,
            iteration_count=iteration_count,
            iteration_budget=iteration_budget,
            quality_score=quality_score,
            grounding_score=grounding_score,
            time_in_mode_seconds=time_in_mode_seconds,
            recent_tool_success_rate=recent_tool_success_rate,
        )

        self._current_state = state
        state_key = state.to_state_key()

        # Get possible actions
        possible_actions = self._get_possible_actions(state)

        if not possible_actions:
            # No valid actions - stay in current mode
            action = ModeAction(
                target_mode=mode_enum,
                should_continue=True,
                reason="No valid transitions available",
                confidence=1.0,
            )
            self._current_action = action
            return action

        # Epsilon-greedy selection
        if random.random() < self._q_store.exploration_rate:
            # Explore: random action
            action = random.choice(possible_actions)
            action.reason = f"Exploring (ε={self._q_store.exploration_rate:.2f})"
        else:
            # Exploit: best known action
            action = self._select_best_action(state_key, possible_actions)

        self._current_action = action

        # Create pending transition event
        if action.target_mode != mode_enum:
            self._pending_transition = TransitionEvent(
                from_mode=mode_enum,
                to_mode=action.target_mode,
                trigger=self._infer_trigger(state, action),
                state_before=state,
                action_taken=action,
            )

        logger.debug(f"[AdaptiveModeController] State: {state_key}, " f"Recommended: {action}")

        return action

    def _get_possible_actions(self, state: ModeState) -> List[ModeAction]:
        """Get all possible actions from current state."""
        actions = []
        current_mode = state.mode

        # Stay in current mode (with budget adjustments)
        for budget_adj in [-2, 0, 2, 5]:
            actions.append(
                ModeAction(
                    target_mode=current_mode,
                    adjust_tool_budget=budget_adj,
                    should_continue=True,
                    confidence=0.5,
                )
            )

        # Transitions to valid modes
        for target_mode in self.VALID_TRANSITIONS.get(current_mode, []):
            actions.append(
                ModeAction(
                    target_mode=target_mode,
                    adjust_tool_budget=0,
                    should_continue=target_mode != AgentMode.COMPLETE,
                    confidence=0.5,
                )
            )

        # Check for termination conditions
        if state.quality_score > 0.8 and state.tool_calls_made > 2:
            actions.append(
                ModeAction(
                    target_mode=AgentMode.COMPLETE,
                    should_continue=False,
                    reason="High quality achieved",
                    confidence=0.8,
                )
            )

        if state.tool_calls_made >= state.tool_budget:
            actions.append(
                ModeAction(
                    target_mode=AgentMode.COMPLETE,
                    should_continue=False,
                    reason="Tool budget exhausted",
                    confidence=0.9,
                )
            )

        return actions

    def _select_best_action(
        self,
        state_key: str,
        possible_actions: List[ModeAction],
    ) -> ModeAction:
        """Select best action using Q-values."""
        q_values = self._q_store.get_all_actions(state_key)

        best_action = possible_actions[0]
        best_q = float("-inf")

        for action in possible_actions:
            action_key = self._action_to_key(action)
            q_value = q_values.get(action_key, 0.0)

            # Add heuristic bonuses
            q_value += self._get_heuristic_bonus(action)

            if q_value > best_q:
                best_q = q_value
                best_action = action

        best_action.confidence = self._q_to_confidence(best_q)
        best_action.reason = f"Best Q-value: {best_q:.2f}"

        return best_action

    def _action_to_key(self, action: ModeAction) -> str:
        """Convert action to string key."""
        return f"{action.target_mode.value}:{action.adjust_tool_budget:+d}"

    def _get_heuristic_bonus(self, action: ModeAction) -> float:
        """Get heuristic bonus for an action."""
        bonus = 0.0

        # Bonus for completing when quality is high
        if action.target_mode == AgentMode.COMPLETE:
            if self._current_state and self._current_state.quality_score > 0.8:
                bonus += 0.3

        # Penalty for unnecessary mode switches
        if self._current_state and action.target_mode != self._current_state.mode:
            if self._current_state.tool_calls_made < 2:
                bonus -= 0.2  # Don't switch too early

        return bonus

    def _q_to_confidence(self, q_value: float) -> float:
        """Convert Q-value to confidence score (0-1)."""
        # Sigmoid function to map Q-value to [0, 1]
        return 1.0 / (1.0 + math.exp(-q_value))

    def _infer_trigger(self, state: ModeState, action: ModeAction) -> TransitionTrigger:
        """Infer the trigger for a transition."""
        if action.target_mode == AgentMode.COMPLETE:
            if state.quality_score > 0.8:
                return TransitionTrigger.QUALITY_THRESHOLD
            if state.tool_calls_made >= state.tool_budget:
                return TransitionTrigger.BUDGET_LOW

        if state.recent_tool_success_rate < 0.5:
            return TransitionTrigger.ERROR_RECOVERY

        return TransitionTrigger.PATTERN_DETECTED

    def record_outcome(
        self,
        success: bool,
        quality_score: float,
        user_satisfied: bool = True,
        completed: bool = False,
    ) -> float:
        """Record outcome and update Q-values.

        Call this after the action has been executed to learn from the outcome.

        Args:
            success: Whether the action was successful
            quality_score: Quality score of the result
            user_satisfied: Whether user indicated satisfaction
            completed: Whether the task was completed

        Returns:
            Reward value used for learning
        """
        if not self._current_state or not self._current_action:
            return 0.0

        # Calculate reward
        reward = self._calculate_reward(
            success=success,
            quality_score=quality_score,
            user_satisfied=user_satisfied,
            completed=completed,
        )

        self._total_reward += reward

        # Update Q-value
        state_key = self._current_state.to_state_key()
        action_key = self._action_to_key(self._current_action)

        self._q_store.update_q_value(state_key, action_key, reward)

        # Record transition if there was one
        if self._pending_transition:
            self._pending_transition.outcome_success = success
            self._pending_transition.outcome_quality = quality_score
            self._pending_transition.reward = reward

            self._q_store.record_transition(
                profile_name=self.profile_name,
                from_mode=self._pending_transition.from_mode,
                to_mode=self._pending_transition.to_mode,
                trigger=self._pending_transition.trigger,
                state_key=state_key,
                action_key=action_key,
                reward=reward,
            )

            self._mode_history.append(
                (
                    self._pending_transition.to_mode,
                    datetime.now(),
                )
            )

            self._pending_transition = None

        # Update task stats
        self._q_store.update_task_stats(
            task_type=self._current_state.task_type,
            tool_budget_used=self._current_state.tool_calls_made,
            quality_score=quality_score,
            completed=completed,
        )

        logger.debug(
            f"[AdaptiveModeController] Recorded outcome: "
            f"success={success}, quality={quality_score:.2f}, reward={reward:.2f}"
        )

        return reward

    def _calculate_reward(
        self,
        success: bool,
        quality_score: float,
        user_satisfied: bool,
        completed: bool,
    ) -> float:
        """Calculate reward for reinforcement learning.

        Reward components:
        - Success: +1.0 if successful, -0.5 if failed
        - Quality: +quality_score (0-1)
        - User satisfaction: +0.5 if satisfied
        - Completion: +0.3 if completed task
        - Efficiency: Bonus for using fewer resources
        """
        reward = 0.0

        # Success component
        reward += 1.0 if success else -0.5

        # Quality component
        reward += quality_score

        # User satisfaction
        if user_satisfied:
            reward += 0.5

        # Completion bonus
        if completed:
            reward += 0.3

        # Efficiency bonus (less tool calls = more efficient)
        if self._current_state:
            budget_ratio = self._current_state.tool_calls_made / max(
                self._current_state.tool_budget, 1
            )
            if budget_ratio < 0.5 and success:
                reward += 0.2  # Efficient completion

        return reward

    def get_optimal_tool_budget(self, task_type: str) -> int:
        """Get learned optimal tool budget for a task type."""
        stats = self._q_store.get_task_stats(task_type)
        return stats.get("optimal_tool_budget", self.DEFAULT_TOOL_BUDGETS.get(task_type, 10))

    def should_continue(
        self,
        tool_calls_made: int,
        tool_budget: int,
        quality_score: float,
        iteration_count: int,
        iteration_budget: int,
    ) -> Tuple[bool, str]:
        """Determine if the agent should continue or stop.

        Returns:
            Tuple of (should_continue, reason)
        """
        # Hard limits
        if tool_calls_made >= tool_budget:
            return False, "Tool budget exhausted"

        if iteration_count >= iteration_budget:
            return False, "Iteration budget exhausted"

        # Quality threshold
        if quality_score > 0.85:
            return False, "High quality achieved"

        # Check for patterns suggesting completion
        if self._current_state:
            state_key = self._current_state.to_state_key()
            q_values = self._q_store.get_all_actions(state_key)

            # If completing has highest Q-value, suggest stopping
            complete_key = f"{AgentMode.COMPLETE.value}:+0"
            if complete_key in q_values:
                if all(q_values[complete_key] >= v for v in q_values.values()):
                    return False, "Learned pattern suggests completion"

        return True, "Continue processing"

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session."""
        session_duration = (datetime.now() - self._session_start).total_seconds()

        return {
            "profile_name": self.profile_name,
            "session_duration_seconds": session_duration,
            "total_reward": self._total_reward,
            "mode_transitions": len(self._mode_history),
            "modes_visited": [m.value for m, _ in self._mode_history],
            "exploration_rate": self._q_store.exploration_rate,
        }

    def reset_session(self) -> None:
        """Reset session tracking."""
        self._session_start = datetime.now()
        self._mode_history = []
        self._total_reward = 0.0
        self._current_state = None
        self._current_action = None
        self._pending_transition = None

    def adjust_exploration_rate(self, new_rate: float) -> None:
        """Adjust the exploration rate for Q-learning."""
        self._q_store.exploration_rate = max(0.0, min(1.0, new_rate))

    def decay_exploration_rate(self, decay_factor: float = 0.99) -> None:
        """Decay exploration rate over time."""
        self._q_store.exploration_rate *= decay_factor


# Module-level convenience function
def get_mode_controller(profile_name: str = "default") -> AdaptiveModeController:
    """Get or create a mode controller for a profile.

    Args:
        profile_name: Profile name

    Returns:
        AdaptiveModeController instance
    """
    return AdaptiveModeController(profile_name=profile_name)
