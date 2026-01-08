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

"""RL learner for workflow execution optimization.

This learner tracks which workflows succeed for different task types and
uses Q-learning to recommend optimal workflows.

Strategy:
- State: (task_type, complexity, mode)
- Action: workflow selection
- Reward: workflow success, duration, quality
- Q-learning updates for workflow-task associations

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                WorkflowExecutionLearner                          │
    ├─────────────────────────────────────────────────────────────────┤
    │  1. Track workflow executions with success/failure              │
    │  2. Build Q-table: (task_type) → workflow → Q-value             │
    │  3. Recommend best workflow for task type                       │
    │  4. Learn from execution outcomes via TD updates                │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from victor.agent.rl.learners.workflow_execution import WorkflowExecutionLearner

    learner = WorkflowExecutionLearner("workflow_execution", db_connection)

    # Get best workflow for task type
    best = learner.get_best_workflow(
        task_type="feature",
        available_workflows=["feature_implementation", "quick_feature"],
    )

    # Record workflow execution outcome
    learner.record_workflow_outcome(
        workflow_name="feature_implementation",
        task_type="feature",
        success=True,
        duration_seconds=45.0,
        quality_score=0.85,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.agent.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


class WorkflowExecutionLearner(BaseLearner):
    """Learns optimal workflows for task types using Q-learning.

    Tracks workflow execution outcomes and builds Q-values for
    (task_type, workflow) pairs to recommend optimal workflows.

    Attributes:
        name: Learner name (should be "workflow_execution")
        db: SQLite database connection
        learning_rate: Q-value update rate (alpha)
        discount_factor: Future reward discount (gamma) - not used for single-step
        epsilon: Exploration rate for ε-greedy selection
    """

    # Default Q-value for unseen workflow-task pairs
    DEFAULT_Q_VALUE = 0.5

    # Q-learning parameters
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_EPSILON = 0.1

    # Minimum samples for confident recommendation
    MIN_SAMPLES_FOR_CONFIDENCE = 5

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        provider_adapter: Optional[Any] = None,
        epsilon: float = DEFAULT_EPSILON,
    ):
        """Initialize workflow execution learner.

        Args:
            name: Learner name
            db_connection: SQLite database connection
            learning_rate: Q-value update rate
            provider_adapter: Optional provider adapter
            epsilon: Exploration rate
        """
        super().__init__(
            name=name,
            db_connection=db_connection,
            learning_rate=learning_rate,
            provider_adapter=provider_adapter,
        )
        self._epsilon = epsilon
        self._ensure_tables()

        logger.debug(
            f"WorkflowExecutionLearner initialized: "
            f"learning_rate={learning_rate}, epsilon={epsilon}"
        )

    def _ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        cursor = self.db.cursor()

        # Q-values table for workflow-task pairs
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.AGENT_WORKFLOW_Q} (
                workflow_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                q_value REAL NOT NULL DEFAULT 0.5,
                execution_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                avg_duration REAL DEFAULT 0,
                avg_quality REAL DEFAULT 0.5,
                last_updated TEXT,
                PRIMARY KEY (workflow_name, task_type)
            )
        """
        )

        # Execution history for detailed analysis
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.AGENT_WORKFLOW_RUN} (
                execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                success INTEGER NOT NULL,
                duration_seconds REAL,
                quality_score REAL,
                vertical TEXT,
                mode TEXT,
                executed_at TEXT NOT NULL
            )
        """
        )

        self.db.commit()

    def get_best_workflow(
        self,
        task_type: str,
        available_workflows: List[str],
    ) -> Optional[str]:
        """Get the best workflow for a task type using Q-values.

        Uses ε-greedy selection: with probability ε, explore randomly;
        otherwise, exploit the best known workflow.

        Args:
            task_type: Type of task
            available_workflows: List of available workflow names

        Returns:
            Best workflow name or None if no data
        """
        if not available_workflows:
            return None

        import random

        # ε-greedy: explore with probability epsilon
        if random.random() < self._epsilon:
            return random.choice(available_workflows)

        # Exploit: get workflow with highest Q-value
        cursor = self.db.cursor()

        placeholders = ",".join("?" * len(available_workflows))
        cursor.execute(
            f"""
            SELECT workflow_name, q_value
            FROM {Tables.AGENT_WORKFLOW_Q}
            WHERE task_type = ? AND workflow_name IN ({placeholders})
            ORDER BY q_value DESC
            LIMIT 1
        """,
            [task_type] + available_workflows,
        )

        row = cursor.fetchone()
        if row:
            return row[0]

        # No Q-values yet - return first available
        return available_workflows[0]

    def get_workflow_q_values(
        self,
        task_type: str,
        available_workflows: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Get Q-values for all workflows for a task type.

        Args:
            task_type: Type of task
            available_workflows: Optional filter for specific workflows

        Returns:
            Dict mapping workflow names to Q-values
        """
        cursor = self.db.cursor()

        if available_workflows:
            placeholders = ",".join("?" * len(available_workflows))
            cursor.execute(
                f"""
                SELECT workflow_name, q_value
                FROM {Tables.AGENT_WORKFLOW_Q}
                WHERE task_type = ? AND workflow_name IN ({placeholders})
            """,
                [task_type] + available_workflows,
            )
        else:
            cursor.execute(
                """
                SELECT workflow_name, q_value
                FROM {Tables.AGENT_WORKFLOW_Q}
                WHERE task_type = ?
            """,
                (task_type,),
            )

        return {row[0]: row[1] for row in cursor.fetchall()}

    def record_workflow_outcome(
        self,
        workflow_name: str,
        task_type: str,
        success: bool,
        duration_seconds: float = 0.0,
        quality_score: float = 0.5,
        vertical: str = "coding",
        mode: str = "build",
    ) -> None:
        """Record a workflow execution outcome.

        Updates Q-values using temporal difference learning.

        Args:
            workflow_name: Name of executed workflow
            task_type: Type of task
            success: Whether execution succeeded
            duration_seconds: Execution duration
            quality_score: Quality score (0.0-1.0)
            vertical: Vertical context
            mode: Agent mode
        """
        cursor = self.db.cursor()
        now = datetime.now().isoformat()

        # Record in execution history
        cursor.execute(
            f"""
            INSERT INTO {Tables.AGENT_WORKFLOW_RUN}
            (workflow_name, task_type, success, duration_seconds, quality_score,
             vertical, mode, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                workflow_name,
                task_type,
                1 if success else 0,
                duration_seconds,
                quality_score,
                vertical,
                mode,
                now,
            ),
        )

        # Compute reward
        reward = self._compute_reward(success, duration_seconds, quality_score)

        # Get current Q-value
        cursor.execute(
            f"""
            SELECT q_value, execution_count, success_count, avg_duration, avg_quality
            FROM {Tables.AGENT_WORKFLOW_Q}
            WHERE workflow_name = ? AND task_type = ?
        """,
            (workflow_name, task_type),
        )

        row = cursor.fetchone()
        if row:
            current_q = row[0]
            execution_count = row[1]
            success_count = row[2]
            avg_duration = row[3]
            avg_quality = row[4]
        else:
            current_q = self.DEFAULT_Q_VALUE
            execution_count = 0
            success_count = 0
            avg_duration = 0.0
            avg_quality = 0.5

        # TD update: Q(s,a) = Q(s,a) + α(r - Q(s,a))
        # Single-step, no future value since workflow is terminal action
        new_q = current_q + self.learning_rate * (reward - current_q)

        # Update aggregates
        new_execution_count = execution_count + 1
        new_success_count = success_count + (1 if success else 0)
        new_avg_duration = (avg_duration * execution_count + duration_seconds) / new_execution_count
        new_avg_quality = (avg_quality * execution_count + quality_score) / new_execution_count

        # Upsert Q-value
        cursor.execute(
            f"""
            INSERT INTO {Tables.AGENT_WORKFLOW_Q}
            (workflow_name, task_type, q_value, execution_count, success_count,
             avg_duration, avg_quality, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workflow_name, task_type) DO UPDATE SET
                q_value = excluded.q_value,
                execution_count = excluded.execution_count,
                success_count = excluded.success_count,
                avg_duration = excluded.avg_duration,
                avg_quality = excluded.avg_quality,
                last_updated = excluded.last_updated
        """,
            (
                workflow_name,
                task_type,
                new_q,
                new_execution_count,
                new_success_count,
                new_avg_duration,
                new_avg_quality,
                now,
            ),
        )

        self.db.commit()

        logger.debug(
            f"Recorded workflow outcome: workflow={workflow_name}, "
            f"task={task_type}, success={success}, "
            f"Q: {current_q:.3f} → {new_q:.3f}"
        )

    def _compute_reward(
        self,
        success: bool,
        duration_seconds: float,
        quality_score: float,
    ) -> float:
        """Compute reward for a workflow execution.

        Args:
            success: Whether execution succeeded
            duration_seconds: Execution duration
            quality_score: Quality score

        Returns:
            Reward value (0.0-1.0)
        """
        if not success:
            return 0.2  # Partial reward for attempting

        # Base reward for success
        reward = 0.6

        # Quality bonus
        reward += quality_score * 0.3

        # Speed bonus (reward faster executions)
        if duration_seconds < 30:
            reward += 0.1
        elif duration_seconds > 120:
            reward -= 0.05

        return min(1.0, max(0.0, reward))

    def get_recommendation(
        self,
        task_type: str,
    ) -> RLRecommendation:
        """Get workflow recommendation for a task type.

        Args:
            task_type: Type of task

        Returns:
            RLRecommendation with best workflow info
        """
        cursor = self.db.cursor()

        # Get best workflow
        cursor.execute(
            """
            SELECT workflow_name, q_value, execution_count, success_count, avg_quality
            FROM {Tables.AGENT_WORKFLOW_Q}
            WHERE task_type = ?
            ORDER BY q_value DESC
            LIMIT 1
        """,
            (task_type,),
        )

        row = cursor.fetchone()
        if not row:
            return RLRecommendation(
                value=0.5,
                confidence=0.1,
                reason=f"No workflow data for task type '{task_type}'",
                is_baseline=True,
            )

        workflow_name = row[0]
        q_value = row[1]
        execution_count = row[2]
        success_count = row[3]
        avg_quality = row[4]

        # Confidence based on execution count
        confidence = min(execution_count / (self.MIN_SAMPLES_FOR_CONFIDENCE * 2), 1.0)
        success_rate = success_count / execution_count if execution_count > 0 else 0.0

        return RLRecommendation(
            value=q_value,
            confidence=confidence,
            reason=(
                f"Workflow '{workflow_name}' recommended for {task_type} tasks: "
                f"Q={q_value:.2f}, success_rate={success_rate:.0%}, "
                f"avg_quality={avg_quality:.2f} ({execution_count} executions)"
            ),
            is_baseline=execution_count < self.MIN_SAMPLES_FOR_CONFIDENCE,
            formation=None,
            suggested_budget=None,
            role_distribution=None,
        )

    def get_workflow_stats(
        self,
        workflow_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics for workflows.

        Args:
            workflow_name: Optional filter for specific workflow

        Returns:
            Dict with workflow statistics
        """
        cursor = self.db.cursor()

        if workflow_name:
            cursor.execute(
                """
                SELECT
                    workflow_name,
                    task_type,
                    q_value,
                    execution_count,
                    success_count,
                    avg_duration,
                    avg_quality
                FROM {Tables.AGENT_WORKFLOW_Q}
                WHERE workflow_name = ?
                ORDER BY task_type
            """,
                (workflow_name,),
            )
        else:
            cursor.execute(
                """
                SELECT
                    workflow_name,
                    task_type,
                    q_value,
                    execution_count,
                    success_count,
                    avg_duration,
                    avg_quality
                FROM {Tables.AGENT_WORKFLOW_Q}
                ORDER BY workflow_name, task_type
            """
            )

        stats = {}
        for row in cursor.fetchall():
            wf_name = row[0]
            if wf_name not in stats:
                stats[wf_name] = {"task_types": {}, "total_executions": 0}

            task_type = row[1]
            execution_count = row[3]
            success_count = row[4]

            stats[wf_name]["task_types"][task_type] = {
                "q_value": row[2],
                "execution_count": execution_count,
                "success_rate": success_count / execution_count if execution_count > 0 else 0.0,
                "avg_duration": row[5],
                "avg_quality": row[6],
            }
            stats[wf_name]["total_executions"] += execution_count

        return stats

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record an outcome and update learned values.

        Extracts workflow information from metadata if available.

        Args:
            outcome: RL outcome to process
        """
        # Extract workflow name from metadata
        workflow_name = None
        if outcome.metadata:
            workflow_name = outcome.metadata.get("workflow_name")

        if not workflow_name:
            return

        self.record_workflow_outcome(
            workflow_name=workflow_name,
            task_type=outcome.task_type,
            success=outcome.success,
            duration_seconds=outcome.metadata.get("duration_seconds", 0.0),
            quality_score=outcome.quality_score,
            vertical=outcome.vertical or "coding",
            mode=outcome.metadata.get("mode", "build"),
        )


__all__ = [
    "WorkflowExecutionLearner",
]
