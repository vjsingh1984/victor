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

"""ML-based formation prediction.

This module implements machine learning models for predicting the
optimal team formation pattern based on task characteristics and
team composition.

Example:
    from victor.teams.ml.formation_predictor import FormationPredictor

    predictor = FormationPredictor()

    # Predict optimal formation
    formation = predictor.predict_formation(
        task_features=task_features,
        team_features=team_features
    )

    # Get probabilities for all formations
    probs = predictor.predict_probabilities(...)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from victor.teams.team_predictor import TaskFeatures, TeamFeatures
    from victor.teams.types import TeamFormation

logger = logging.getLogger(__name__)


@dataclass
class FormationPrediction:
    """Prediction result for team formation.

    Attributes:
        formation: Predicted optimal formation
        confidence: Confidence in prediction (0.0-1.0)
        probabilities: Probabilities for all formations
        reasoning: Explanation for prediction
    """

    formation: "TeamFormation"
    confidence: float
    probabilities: Dict[str, float]
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "formation": self.formation.value,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "reasoning": self.reasoning,
        }


class FormationPredictor:
    """ML-based formation prediction.

    Predicts the optimal team formation pattern based on task
    characteristics, team composition, and historical performance.

    Example:
        predictor = FormationPredictor()

        # Train on historical data
        predictor.train(training_data)

        # Predict formation
        prediction = predictor.predict_formation(
            task_features=task_features,
            team_features=team_features
        )
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_heuristic: bool = True,
    ):
        """Initialize formation predictor.

        Args:
            model_path: Path to trained model
            use_heuristic: Use heuristic rules if no model loaded
        """
        self.model_path = model_path
        self.use_heuristic = use_heuristic

        # Model (lazy loaded)
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None

        # Formation labels
        self._formations = [
            "sequential",
            "parallel",
            "hierarchical",
            "pipeline",
            "consensus",
        ]

        if model_path and model_path.exists():
            self.load_model(model_path)

    def predict_formation(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
    ) -> FormationPrediction:
        """Predict optimal formation for task and team.

        Args:
            task_features: Features of the task
            team_features: Features of the team

        Returns:
            Formation prediction
        """
        if self._model is not None:
            return self._predict_with_model(task_features, team_features)
        elif self.use_heuristic:
            return self._predict_with_heuristic(task_features, team_features)
        else:
            # Default fallback
            from victor.teams.types import TeamFormation

            return FormationPrediction(
                formation=TeamFormation.SEQUENTIAL,
                confidence=0.0,
                probabilities=dict.fromkeys(self._formations, 0.2),
                reasoning="No prediction method available, using default",
            )

    def _predict_with_model(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
    ) -> FormationPrediction:
        """Predict using trained model.

        Args:
            task_features: Task features
            team_features: Team features

        Returns:
            Formation prediction
        """
        # Create feature vector
        feature_vector = np.concatenate(
            [
                task_features.to_feature_vector(),
                team_features.to_feature_vector(),
            ]
        )

        # Scale features
        if self._scaler:
            feature_vector = self._scaler.transform([feature_vector])[0]

        # Predict probabilities
        probs = self._model.predict_proba([feature_vector])[0]

        # Create probability dict
        prob_dict = dict(zip(self._formations, probs))

        # Get best formation
        best_idx = int(np.argmax(probs))
        from victor.teams.types import TeamFormation

        best_formation = TeamFormation(self._formations[best_idx])
        confidence = float(probs[best_idx])

        return FormationPrediction(
            formation=best_formation,
            confidence=confidence,
            probabilities=prob_dict,
            reasoning=f"Model predicts {best_formation.value} with {confidence:.1%} confidence",
        )

    def _predict_with_heuristic(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
    ) -> FormationPrediction:
        """Predict using heuristic rules.

        Args:
            task_features: Task features
            team_features: Team features

        Returns:
            Formation prediction
        """
        from victor.teams.types import TeamFormation

        # Score each formation
        scores = {}

        # SEQUENTIAL: Good for simple, single-file, low-complexity tasks
        sequential_score = (
            (1.0 - task_features.complexity) * 0.5
            + (1.0 / max(1, task_features.file_count)) * 0.3
            + (1.0 - task_features.urgency) * 0.2
        )
        scores["sequential"] = sequential_score

        # PARALLEL: Good for complex, multi-file, high-urgency tasks
        parallel_score = (
            task_features.complexity * 0.4
            + min(1.0, task_features.file_count / 5) * 0.4
            + task_features.urgency * 0.2
        )
        scores["parallel"] = parallel_score

        # HIERARCHICAL: Good for complex tasks with manager
        hierarchical_score = (
            task_features.complexity * 0.5
            + float(team_features.has_manager) * 0.3
            + (team_features.member_count / 10) * 0.2
        )
        scores["hierarchical"] = hierarchical_score

        # PIPELINE: Good for sequential dependencies, multi-stage tasks
        pipeline_score = (
            (min(1.0, task_features.file_count / 3)) * 0.5
            + (1.0 - task_features.urgency) * 0.3
            + task_features.novelty * 0.2
        )
        scores["pipeline"] = pipeline_score

        # CONSENSUS: Good for high-quality, low-urgency requirements
        consensus_score = (
            (1.0 - task_features.urgency) * 0.4
            + team_features.diversity * 0.4
            + (1.0 - task_features.complexity) * 0.2
        )
        scores["consensus"] = consensus_score

        # Normalize scores to probabilities
        total = sum(scores.values())
        probabilities = {k: v / total for k, v in scores.items()}

        # Select best formation
        best_formation_str = max(scores, key=scores.get)
        best_formation = TeamFormation(best_formation_str)
        confidence = probabilities[best_formation_str]

        # Generate reasoning
        reasoning = self._generate_heuristic_reasoning(best_formation, task_features, team_features)

        return FormationPrediction(
            formation=best_formation,
            confidence=confidence,
            probabilities=probabilities,
            reasoning=reasoning,
        )

    def _generate_heuristic_reasoning(
        self,
        formation: "TeamFormation",
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
    ) -> str:
        """Generate reasoning for heuristic prediction.

        Args:
            formation: Predicted formation
            task_features: Task features
            team_features: Team features

        Returns:
            Reasoning string
        """
        reasons = []

        if formation.value == "sequential":
            if task_features.complexity < 0.5:
                reasons.append("low complexity")
            if task_features.file_count <= 2:
                reasons.append("few files")
            if task_features.urgency < 0.5:
                reasons.append("not urgent")
            return f"Sequential recommended due to: {', '.join(reasons)}"

        elif formation.value == "parallel":
            if task_features.complexity > 0.6:
                reasons.append("high complexity")
            if task_features.file_count > 3:
                reasons.append("multiple files")
            if task_features.urgency > 0.6:
                reasons.append("high urgency")
            return f"Parallel recommended for: {', '.join(reasons)}"

        elif formation.value == "hierarchical":
            if task_features.complexity > 0.6:
                reasons.append("complex task")
            if team_features.has_manager:
                reasons.append("has manager")
            if team_features.member_count > 3:
                reasons.append("large team")
            return f"Hierarchical recommended for: {', '.join(reasons)}"

        elif formation.value == "pipeline":
            if task_features.file_count > 2:
                reasons.append("multi-file workflow")
            if task_features.urgency < 0.5:
                reasons.append("flexible timeline")
            if task_features.novelty > 0.6:
                reasons.append("novel approach")
            return f"Pipeline recommended for: {', '.join(reasons)}"

        elif formation.value == "consensus":
            if task_features.urgency < 0.5:
                reasons.append("quality over speed")
            if team_features.diversity > 0.6:
                reasons.append("diverse perspectives")
            if task_features.complexity < 0.7:
                reasons.append("manageable complexity")
            return f"Consensus recommended for: {', '.join(reasons)}"

        return f"Recommended {formation.value} based on task and team characteristics"

    def predict_probabilities(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
    ) -> Dict[str, float]:
        """Get probabilities for all formations.

        Args:
            task_features: Task features
            team_features: Team features

        Returns:
            Dictionary mapping formation to probability
        """
        prediction = self.predict_formation(task_features, team_features)
        return prediction.probabilities

    def train(
        self,
        training_data: List[Dict[str, Any]],
    ) -> None:
        """Train formation prediction model.

        Args:
            training_data: List of training examples with:
                - task_features: TaskFeatures
                - team_features: TeamFeatures
                - formation: TeamFormation (label)
                - success: bool (optional, for weighting)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        # Prepare features
        X = []
        y = []
        weights = []

        for example in training_data:
            task_features = example["task_features"]
            team_features = example["team_features"]
            formation = example["formation"]
            success = example.get("success", True)

            # Extract features
            feature_vector = np.concatenate(
                [
                    task_features.to_feature_vector(),
                    team_features.to_feature_vector(),
                ]
            )

            X.append(feature_vector)
            y.append(self._formations.index(formation.value))
            weights.append(2.0 if success else 1.0)

        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Train model
        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )
        self._model.fit(X_scaled, y, sample_weight=weights)

        logger.info(f"Trained formation predictor on {len(training_data)} examples")

    def save_model(self, path: Path) -> None:
        """Save trained model to file.

        Args:
            path: Path to save model
        """
        if self._model is None:
            logger.warning("No model to save")
            return

        try:
            model_data = {
                "model": self._model,
                "scaler": self._scaler,
                "formations": self._formations,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved model to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, path: Path) -> None:
        """Load trained model from file.

        Args:
            path: Path to load model from
        """
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self._model = model_data["model"]
            self._scaler = model_data.get("scaler")
            self._formations = model_data.get("formations", self._formations)

            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model.

        Returns:
            Dictionary mapping feature names to importance
        """
        if self._model is None or not hasattr(self._model, "feature_importances_"):
            return {}

        feature_names = [
            # Task features
            "task_complexity",
            "task_estimated_loc",
            "task_file_count",
            "task_urgency",
            "task_dependencies",
            "task_novelty",
            "task_required_expertise_count",
            # Team features
            "team_member_count",
            "team_total_tool_budget",
            "team_expertise_coverage",
            "team_diversity",
            "team_avg_confidence",
            "team_has_manager",
            "team_max_delegation_depth",
        ]

        importances = dict(zip(feature_names, self._model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


__all__ = [
    "FormationPrediction",
    "FormationPredictor",
]
