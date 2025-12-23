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

"""RL learner for quality dimension weight optimization.

This learner uses gradient-based learning to optimize the weights
for response quality dimensions based on task type and implicit feedback.

Problem:
- Fixed weights in ResponseQualityScorer don't account for task differences
- Different tasks may value different quality dimensions
- Weights should be learned from actual outcomes

Strategy:
- State: task_type
- Weights: [relevance, completeness, accuracy, conciseness, actionability, coherence, code_quality]
- Reward: Correlation between weighted quality score and task success

Algorithm: Online gradient descent with momentum

Sprint 4: Implicit Feedback Enhancement
"""

import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.agent.rl.base import BaseLearner, RLOutcome, RLRecommendation

logger = logging.getLogger(__name__)


class QualityDimension:
    """Quality dimension names matching response_quality.py."""

    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONCISENESS = "conciseness"
    ACTIONABILITY = "actionability"
    COHERENCE = "coherence"
    CODE_QUALITY = "code_quality"

    ALL = [
        RELEVANCE,
        COMPLETENESS,
        ACCURACY,
        CONCISENESS,
        ACTIONABILITY,
        COHERENCE,
        CODE_QUALITY,
    ]


class QualityWeightLearner(BaseLearner):
    """Learn optimal quality dimension weights per task type.

    Uses online gradient descent to update weights based on
    correlation between quality scores and task success.

    Attributes:
        name: Always "quality_weights"
        db: SQLite database connection
        learning_rate: Weight update rate (default 0.05)
        momentum: Momentum factor for smoother updates (default 0.9)
    """

    # Default weights (matching ResponseQualityScorer.DEFAULT_WEIGHTS)
    DEFAULT_WEIGHTS = {
        QualityDimension.RELEVANCE: 1.5,
        QualityDimension.COMPLETENESS: 1.2,
        QualityDimension.ACCURACY: 1.3,
        QualityDimension.CONCISENESS: 0.8,
        QualityDimension.ACTIONABILITY: 1.0,
        QualityDimension.COHERENCE: 0.9,
        QualityDimension.CODE_QUALITY: 1.1,
    }

    # Learning parameters
    DEFAULT_LEARNING_RATE = 0.05
    DEFAULT_MOMENTUM = 0.9
    MIN_WEIGHT = 0.1
    MAX_WEIGHT = 3.0

    # Minimum samples before confident recommendation
    MIN_SAMPLES_FOR_CONFIDENCE = 15

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        provider_adapter: Optional[Any] = None,
        momentum: float = DEFAULT_MOMENTUM,
    ):
        """Initialize quality weight learner.

        Args:
            name: Learner name (should be "quality_weights")
            db_connection: SQLite database connection
            learning_rate: Weight update rate (default 0.05)
            provider_adapter: Optional provider adapter
            momentum: Momentum factor (default 0.9)
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        self.momentum = momentum

        # Learned weights per task type
        # task_type -> {dimension: weight}
        self._weights: Dict[str, Dict[str, float]] = {}

        # Momentum velocities for gradient descent
        self._velocities: Dict[str, Dict[str, float]] = {}

        # Sample counts per task type
        self._sample_counts: Dict[str, int] = {}

        # Recent outcomes for gradient computation
        self._recent_outcomes: Dict[str, List[Dict[str, float]]] = {}

        # Load state from database
        self._load_state()

    def _ensure_tables(self) -> None:
        """Create tables for quality weight learning."""
        cursor = self.db.cursor()

        # Weights table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS quality_weights (
                task_type TEXT NOT NULL,
                dimension TEXT NOT NULL,
                weight REAL NOT NULL,
                velocity REAL NOT NULL DEFAULT 0.0,
                sample_count INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (task_type, dimension)
            )
            """
        )

        # Learning history
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS quality_weight_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                dimension_scores TEXT NOT NULL,
                overall_success REAL NOT NULL,
                weights_used TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )

        # Indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_quality_weights_task
            ON quality_weights(task_type)
            """
        )

        self.db.commit()
        logger.debug("RL: quality_weights tables ensured")

    def _load_state(self) -> None:
        """Load state from database."""
        cursor = self.db.cursor()

        try:
            cursor.execute("SELECT * FROM quality_weights")
            for row in cursor.fetchall():
                row_dict = dict(row)
                task_type = row_dict["task_type"]
                dimension = row_dict["dimension"]

                if task_type not in self._weights:
                    self._weights[task_type] = dict(self.DEFAULT_WEIGHTS)
                    self._velocities[task_type] = {d: 0.0 for d in QualityDimension.ALL}
                    self._sample_counts[task_type] = 0

                self._weights[task_type][dimension] = row_dict["weight"]
                self._velocities[task_type][dimension] = row_dict["velocity"]
                self._sample_counts[task_type] = max(
                    self._sample_counts[task_type], row_dict["sample_count"]
                )

        except Exception as e:
            logger.debug(f"RL: Could not load quality weights: {e}")

        if self._weights:
            logger.info(
                f"RL: Loaded quality weights for {len(self._weights)} task types from database"
            )

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record quality outcome and update weights.

        Expected metadata:
        - dimension_scores: Dict mapping dimension name to score (0-1)
        - overall_success: Task success indicator (0-1)

        Args:
            outcome: Outcome with quality scoring data
        """
        task_type = outcome.task_type
        dimension_scores = outcome.metadata.get("dimension_scores", {})
        overall_success = outcome.metadata.get("overall_success", outcome.quality_score)

        if not dimension_scores:
            logger.debug("RL: quality_weights outcome missing dimension_scores, skipping")
            return

        # Ensure task type exists
        if task_type not in self._weights:
            self._weights[task_type] = dict(self.DEFAULT_WEIGHTS)
            self._velocities[task_type] = {d: 0.0 for d in QualityDimension.ALL}
            self._sample_counts[task_type] = 0
            self._recent_outcomes[task_type] = []

        # Store outcome for batch gradient computation
        if task_type not in self._recent_outcomes:
            self._recent_outcomes[task_type] = []

        self._recent_outcomes[task_type].append(
            {
                "dimension_scores": dimension_scores,
                "success": overall_success,
            }
        )

        # Keep last 20 outcomes for gradient computation
        self._recent_outcomes[task_type] = self._recent_outcomes[task_type][-20:]

        # Update weights using gradient descent
        self._update_weights(task_type, dimension_scores, overall_success)

        # Save to database
        self._save_to_db(task_type, dimension_scores, overall_success)

        logger.debug(
            f"RL: Quality weights updated for task_type={task_type}, "
            f"success={overall_success:.2f}"
        )

    def _update_weights(
        self,
        task_type: str,
        dimension_scores: Dict[str, float],
        success: float,
    ) -> None:
        """Update weights using gradient descent with momentum.

        The gradient is computed as:
        ∂L/∂w_i = (predicted_quality - success) * dimension_score_i

        With momentum update:
        v_i = momentum * v_i + learning_rate * gradient
        w_i = w_i - v_i

        Args:
            task_type: Task type being updated
            dimension_scores: Scores for each dimension
            success: Overall task success (target)
        """
        weights = self._weights[task_type]
        velocities = self._velocities[task_type]

        # Compute predicted quality with current weights
        total_weight = sum(weights.values())
        predicted_quality = 0.0

        for dim in QualityDimension.ALL:
            if dim in dimension_scores:
                predicted_quality += weights[dim] * dimension_scores[dim]

        if total_weight > 0:
            predicted_quality /= total_weight

        # Compute error
        error = predicted_quality - success

        # Update each dimension weight
        for dim in QualityDimension.ALL:
            if dim in dimension_scores:
                # Gradient: error * dimension_score
                gradient = error * dimension_scores[dim]

                # Momentum update
                velocities[dim] = self.momentum * velocities[dim] + self.learning_rate * gradient

                # Weight update (gradient descent)
                new_weight = weights[dim] - velocities[dim]

                # Clip to valid range
                weights[dim] = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))

        # Normalize weights to sum to original total
        current_total = sum(weights.values())
        default_total = sum(self.DEFAULT_WEIGHTS.values())

        if current_total > 0:
            scale = default_total / current_total
            for dim in weights:
                weights[dim] *= scale
                # Re-clip after normalization to ensure bounds
                weights[dim] = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, weights[dim]))

        self._sample_counts[task_type] += 1

    def _save_to_db(
        self,
        task_type: str,
        dimension_scores: Dict[str, float],
        success: float,
    ) -> None:
        """Save weights and history to database."""
        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        # Save current weights
        for dim in QualityDimension.ALL:
            cursor.execute(
                """
                INSERT OR REPLACE INTO quality_weights
                (task_type, dimension, weight, velocity, sample_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task_type,
                    dim,
                    self._weights[task_type][dim],
                    self._velocities[task_type][dim],
                    self._sample_counts[task_type],
                    timestamp,
                ),
            )

        # Save history
        cursor.execute(
            """
            INSERT INTO quality_weight_history
            (task_type, dimension_scores, overall_success, weights_used, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                task_type,
                json.dumps(dimension_scores),
                success,
                json.dumps(self._weights[task_type]),
                timestamp,
            ),
        )

        self.db.commit()

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommended weights for a task type.

        Args:
            provider: Provider name (not used)
            model: Model name (not used)
            task_type: Task type to get weights for

        Returns:
            Recommendation with weight dictionary
        """
        if task_type not in self._weights:
            return RLRecommendation(
                value=dict(self.DEFAULT_WEIGHTS),
                confidence=0.3,
                reason="No learned data, using default weights",
                sample_size=0,
                is_baseline=True,
            )

        sample_count = self._sample_counts.get(task_type, 0)

        if sample_count < self.MIN_SAMPLES_FOR_CONFIDENCE:
            confidence = 0.3 + 0.2 * (sample_count / self.MIN_SAMPLES_FOR_CONFIDENCE)
            is_baseline = True
        else:
            confidence = min(0.95, 0.5 + 0.45 * (1 - math.exp(-sample_count / 30)))
            is_baseline = False

        return RLRecommendation(
            value=dict(self._weights[task_type]),
            confidence=confidence,
            reason=f"Learned from {sample_count} samples",
            sample_size=sample_count,
            is_baseline=is_baseline,
        )

    def get_weights(self, task_type: str) -> Dict[str, float]:
        """Get weights for a task type.

        Args:
            task_type: Task type

        Returns:
            Dictionary of dimension -> weight
        """
        if task_type in self._weights:
            return dict(self._weights[task_type])
        return dict(self.DEFAULT_WEIGHTS)

    def get_weight_adjustments(self, task_type: str) -> Dict[str, float]:
        """Get weight adjustments relative to defaults.

        Args:
            task_type: Task type

        Returns:
            Dictionary of dimension -> adjustment (positive = increased, negative = decreased)
        """
        if task_type not in self._weights:
            return {d: 0.0 for d in QualityDimension.ALL}

        return {
            dim: self._weights[task_type][dim] - self.DEFAULT_WEIGHTS[dim]
            for dim in QualityDimension.ALL
        }

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        For quality weights, reward is based on how well the weighted
        quality prediction matched the actual success.

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        # Get dimension scores from metadata
        dimension_scores = outcome.metadata.get("dimension_scores", {})
        if not dimension_scores:
            return 0.0

        # Compute prediction with current weights
        task_type = outcome.task_type
        weights = self._weights.get(task_type, self.DEFAULT_WEIGHTS)

        total_weight = sum(weights.values())
        predicted = 0.0

        for dim in QualityDimension.ALL:
            if dim in dimension_scores:
                predicted += weights[dim] * dimension_scores[dim]

        if total_weight > 0:
            predicted /= total_weight

        # Reward is negative of prediction error
        actual = outcome.metadata.get("overall_success", outcome.quality_score)
        error = abs(predicted - actual)

        # Map error [0, 1] to reward [-1, 1]
        # error=0 -> reward=1, error=1 -> reward=-1
        return 1.0 - 2.0 * error

    def get_correlation_analysis(self, task_type: str) -> Dict[str, float]:
        """Analyze correlation between dimensions and success.

        Uses recent outcomes to compute Pearson correlation.

        Args:
            task_type: Task type to analyze

        Returns:
            Dictionary of dimension -> correlation coefficient
        """
        if task_type not in self._recent_outcomes:
            return {d: 0.0 for d in QualityDimension.ALL}

        outcomes = self._recent_outcomes[task_type]
        if len(outcomes) < 5:
            return {d: 0.0 for d in QualityDimension.ALL}

        correlations = {}

        for dim in QualityDimension.ALL:
            # Extract dimension scores and success values
            dim_scores = []
            successes = []

            for o in outcomes:
                if dim in o["dimension_scores"]:
                    dim_scores.append(o["dimension_scores"][dim])
                    successes.append(o["success"])

            if len(dim_scores) < 5:
                correlations[dim] = 0.0
                continue

            # Compute Pearson correlation
            n = len(dim_scores)
            mean_dim = sum(dim_scores) / n
            mean_success = sum(successes) / n

            numerator = sum(
                (d - mean_dim) * (s - mean_success) for d, s in zip(dim_scores, successes)
            )

            var_dim = sum((d - mean_dim) ** 2 for d in dim_scores)
            var_success = sum((s - mean_success) ** 2 for s in successes)

            if var_dim > 0 and var_success > 0:
                correlations[dim] = numerator / math.sqrt(var_dim * var_success)
            else:
                correlations[dim] = 0.0

        return correlations

    def export_metrics(self) -> Dict[str, Any]:
        """Export learner metrics for monitoring.

        Returns:
            Dictionary with learner stats
        """
        total_samples = sum(self._sample_counts.values())

        # Get top adjustments across all task types
        all_adjustments = {}
        for task_type in self._weights:
            adjustments = self.get_weight_adjustments(task_type)
            for dim, adj in adjustments.items():
                key = f"{task_type}:{dim}"
                all_adjustments[key] = adj

        # Sort by absolute adjustment
        top_adjustments = dict(
            sorted(all_adjustments.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )

        return {
            "learner": self.name,
            "task_types_learned": len(self._weights),
            "total_samples": total_samples,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "default_weights": self.DEFAULT_WEIGHTS,
            "top_weight_adjustments": top_adjustments,
            "samples_per_task": dict(self._sample_counts),
        }
