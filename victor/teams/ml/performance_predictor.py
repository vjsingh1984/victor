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

"""ML-based performance prediction for teams.

This module implements machine learning models for predicting various
team performance metrics including execution time, success rate,
quality score, and resource utilization.

Example:
    from victor.teams.ml.performance_predictor import PerformancePredictor

    predictor = PerformancePredictor()

    # Predict execution time
    time_pred = predictor.predict_execution_time(
        task_features=task_features,
        team_features=team_features,
        formation=formation
    )

    # Predict success rate
    success_pred = predictor.predict_success_rate(...)
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
class PerformancePrediction:
    """Prediction of a performance metric.

    Attributes:
        metric_name: Name of predicted metric
        predicted_value: Predicted value
        confidence_interval: 95% confidence interval (low, high)
        confidence: Overall confidence (0.0-1.0)
        factors: Key factors influencing prediction
    """

    metric_name: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence: float
    factors: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "predicted_value": self.predicted_value,
            "confidence_interval": self.confidence_interval,
            "confidence": self.confidence,
            "factors": self.factors,
        }


class PerformancePredictor:
    """ML-based performance prediction.

    Predicts various team performance metrics using machine learning
    models trained on historical execution data.

    Example:
        predictor = PerformancePredictor()

        # Train models
        predictor.train_execution_time_model(time_data)
        predictor.train_success_model(success_data)

        # Make predictions
        time_pred = predictor.predict_execution_time(...)
        success_pred = predictor.predict_success_rate(...)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_heuristic: bool = True,
    ):
        """Initialize performance predictor.

        Args:
            model_path: Path to trained models
            use_heuristic: Use heuristic prediction if no model loaded
        """
        self.model_path = model_path
        self.use_heuristic = use_heuristic

        # Models for different metrics
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}

        if model_path and model_path.exists():
            self.load_models(model_path)

    def predict_execution_time(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
        formation: "TeamFormation",
    ) -> PerformancePrediction:
        """Predict execution time.

        Args:
            task_features: Task features
            team_features: Team features
            formation: Formation pattern

        Returns:
            Execution time prediction
        """
        if "execution_time" in self._models:
            return self._predict_with_model(
                "execution_time", task_features, team_features, formation
            )
        elif self.use_heuristic:
            return self._predict_time_heuristic(task_features, team_features, formation)
        else:
            return PerformancePrediction(
                metric_name="execution_time",
                predicted_value=60.0,
                confidence_interval=(30.0, 120.0),
                confidence=0.0,
                factors={},
            )

    def predict_success_rate(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
        formation: "TeamFormation",
    ) -> PerformancePrediction:
        """Predict success rate.

        Args:
            task_features: Task features
            team_features: Team features
            formation: Formation pattern

        Returns:
            Success rate prediction
        """
        if "success_rate" in self._models:
            return self._predict_with_model(
                "success_rate", task_features, team_features, formation
            )
        elif self.use_heuristic:
            return self._predict_success_heuristic(task_features, team_features, formation)
        else:
            return PerformancePrediction(
                metric_name="success_rate",
                predicted_value=0.8,
                confidence_interval=(0.6, 1.0),
                confidence=0.0,
                factors={},
            )

    def predict_quality_score(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
        formation: "TeamFormation",
    ) -> PerformancePrediction:
        """Predict quality score.

        Args:
            task_features: Task features
            team_features: Team features
            formation: Formation pattern

        Returns:
            Quality score prediction
        """
        if "quality_score" in self._models:
            return self._predict_with_model(
                "quality_score", task_features, team_features, formation
            )
        elif self.use_heuristic:
            return self._predict_quality_heuristic(task_features, team_features, formation)
        else:
            return PerformancePrediction(
                metric_name="quality_score",
                predicted_value=0.75,
                confidence_interval=(0.5, 1.0),
                confidence=0.0,
                factors={},
            )

    def predict_tool_usage(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
        formation: "TeamFormation",
    ) -> PerformancePrediction:
        """Predict tool call usage.

        Args:
            task_features: Task features
            team_features: Team features
            formation: Formation pattern

        Returns:
            Tool usage prediction
        """
        if "tool_usage" in self._models:
            return self._predict_with_model(
                "tool_usage", task_features, team_features, formation
            )
        elif self.use_heuristic:
            return self._predict_tool_usage_heuristic(task_features, team_features, formation)
        else:
            return PerformancePrediction(
                metric_name="tool_usage",
                predicted_value=50,
                confidence_interval=(25, 75),
                confidence=0.0,
                factors={},
            )

    def _predict_with_model(
        self,
        metric_name: str,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
        formation: "TeamFormation",
    ) -> PerformancePrediction:
        """Predict using trained model.

        Args:
            metric_name: Name of metric
            task_features: Task features
            team_features: Team features
            formation: Formation pattern

        Returns:
            Performance prediction
        """
        model = self._models.get(metric_name)
        scaler = self._scalers.get(metric_name)

        if model is None:
            raise ValueError(f"No model available for metric: {metric_name}")

        # Create feature vector
        feature_vector = self._create_feature_vector(
            task_features, team_features, formation
        )

        # Scale features
        if scaler:
            feature_vector = scaler.transform([feature_vector])[0]

        # Predict
        if hasattr(model, "predict"):
            prediction = model.predict([feature_vector])[0]
        else:
            prediction = model([feature_vector])[0]

        # Estimate confidence interval (simplified)
        std_dev = abs(prediction) * 0.2  # 20% variance
        confidence_interval = (
            max(0, prediction - 1.96 * std_dev),
            prediction + 1.96 * std_dev
        )

        return PerformancePrediction(
            metric_name=metric_name,
            predicted_value=float(prediction),
            confidence_interval=confidence_interval,
            confidence=0.75,  # Would be computed from model uncertainty
            factors={},  # Would compute SHAP values or similar
        )

    def _create_feature_vector(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
        formation: "TeamFormation",
    ) -> np.ndarray:
        """Create feature vector for prediction.

        Args:
            task_features: Task features
            team_features: Team features
            formation: Formation pattern

        Returns:
            Feature vector
        """
        # Task features
        task_vec = task_features.to_feature_vector()

        # Team features
        team_vec = team_features.to_feature_vector()

        # Formation encoding (one-hot)
        formations = ["sequential", "parallel", "hierarchical", "pipeline", "consensus"]
        formation_vec = [1.0 if formation.value == f else 0.0 for f in formations]

        return np.concatenate([task_vec, team_vec, formation_vec])

    # =========================================================================
    # Heuristic Predictions
    # =========================================================================

    def _predict_time_heuristic(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
        formation: "TeamFormation",
    ) -> PerformancePrediction:
        """Predict execution time using heuristics.

        Args:
            task_features: Task features
            team_features: Team features
            formation: Formation

        Returns:
            Time prediction
        """
        base_time = 30.0

        # Complexity factor
        complexity_factor = 1.0 + task_features.complexity * 2.0

        # LOC factor
        loc_factor = 1.0 + (task_features.estimated_lines_of_code / 500) * 0.5

        # File count factor
        file_factor = 1.0 + (task_features.file_count - 1) * 0.3

        # Team size factor
        team_factor = max(0.5, 1.0 - (team_features.member_count - 1) * 0.1)

        # Formation factor
        formation_multipliers = {
            "sequential": 1.2,
            "parallel": 0.7,
            "hierarchical": 0.9,
            "pipeline": 1.0,
            "consensus": 1.5,
        }
        formation_factor = formation_multipliers.get(formation.value, 1.0)

        predicted_time = (
            base_time *
            complexity_factor *
            loc_factor *
            file_factor *
            team_factor *
            formation_factor
        )

        # Confidence interval
        std_dev = predicted_time * 0.25
        confidence_interval = (
            max(0, predicted_time - 1.96 * std_dev),
            predicted_time + 1.96 * std_dev
        )

        factors = {
            "complexity": complexity_factor,
            "loc": loc_factor,
            "files": file_factor,
            "team_size": team_factor,
            "formation": formation_factor,
        }

        return PerformancePrediction(
            metric_name="execution_time",
            predicted_value=round(predicted_time, 2),
            confidence_interval=tuple(round(x, 2) for x in confidence_interval),
            confidence=0.65,
            factors=factors,
        )

    def _predict_success_heuristic(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
        formation: "TeamFormation",
    ) -> PerformancePrediction:
        """Predict success rate using heuristics.

        Args:
            task_features: Task features
            team_features: Team features
            formation: Formation

        Returns:
            Success rate prediction
        """
        base_rate = 0.8

        # Expertise coverage
        expertise_bonus = team_features.expertise_coverage * 0.1

        # Team diversity
        diversity_bonus = team_features.diversity * 0.05

        # Complexity penalty
        complexity_penalty = task_features.complexity * 0.1

        # Formation bonus
        formation_bonuses = {
            "sequential": 0.0,
            "parallel": 0.05,
            "hierarchical": 0.03,
            "pipeline": 0.02,
            "consensus": 0.08,
        }
        formation_bonus = formation_bonuses.get(formation.value, 0.0)

        predicted_rate = (
            base_rate +
            expertise_bonus +
            diversity_bonus +
            formation_bonus -
            complexity_penalty
        )
        predicted_rate = max(0.0, min(1.0, predicted_rate))

        # Confidence interval
        std_dev = 0.15
        confidence_interval = (
            max(0, predicted_rate - 1.96 * std_dev),
            min(1, predicted_rate + 1.96 * std_dev)
        )

        factors = {
            "expertise_coverage": team_features.expertise_coverage,
            "diversity": team_features.diversity,
            "complexity": task_features.complexity,
            "formation_bonus": formation_bonus,
        }

        return PerformancePrediction(
            metric_name="success_rate",
            predicted_value=round(predicted_rate, 3),
            confidence_interval=tuple(round(x, 3) for x in confidence_interval),
            confidence=0.60,
            factors=factors,
        )

    def _predict_quality_heuristic(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
        formation: "TeamFormation",
    ) -> PerformancePrediction:
        """Predict quality score using heuristics.

        Args:
            task_features: Task features
            team_features: Team features
            formation: Formation

        Returns:
            Quality score prediction
        """
        base_score = 0.75

        # Expertise coverage
        expertise_bonus = team_features.expertise_coverage * 0.15

        # Diversity bonus
        diversity_bonus = team_features.diversity * 0.1

        # Formation quality bonus
        formation_bonuses = {
            "sequential": 0.0,
            "parallel": 0.02,
            "hierarchical": 0.03,
            "pipeline": 0.05,
            "consensus": 0.12,
        }
        formation_bonus = formation_bonuses.get(formation.value, 0.0)

        # Complexity penalty
        complexity_penalty = task_features.complexity * 0.05

        predicted_score = (
            base_score +
            expertise_bonus +
            diversity_bonus +
            formation_bonus -
            complexity_penalty
        )
        predicted_score = max(0.0, min(1.0, predicted_score))

        # Confidence interval
        std_dev = 0.12
        confidence_interval = (
            max(0, predicted_score - 1.96 * std_dev),
            min(1, predicted_score + 1.96 * std_dev)
        )

        factors = {
            "expertise_coverage": team_features.expertise_coverage,
            "diversity": team_features.diversity,
            "formation": formation_bonus,
            "complexity": task_features.complexity,
        }

        return PerformancePrediction(
            metric_name="quality_score",
            predicted_value=round(predicted_score, 3),
            confidence_interval=tuple(round(x, 3) for x in confidence_interval),
            confidence=0.62,
            factors=factors,
        )

    def _predict_tool_usage_heuristic(
        self,
        task_features: "TaskFeatures",
        team_features: "TeamFeatures",
        formation: "TeamFormation",
    ) -> PerformancePrediction:
        """Predict tool usage using heuristics.

        Args:
            task_features: Task features
            team_features: Team features
            formation: Formation

        Returns:
            Tool usage prediction
        """
        base_calls = 25

        # Complexity factor
        complexity_factor = 1.0 + task_features.complexity * 1.5

        # File count factor
        file_factor = 1.0 + task_features.file_count * 0.4

        # LOC factor
        loc_factor = 1.0 + (task_features.estimated_lines_of_code / 200) * 0.5

        # Team size factor (more members = more tool calls)
        team_factor = 1.0 + (team_features.member_count - 1) * 0.2

        predicted_calls = int(
            base_calls *
            complexity_factor *
            file_factor *
            loc_factor *
            team_factor
        )

        # Confidence interval
        std_dev = predicted_calls * 0.3
        confidence_interval = (
            max(0, int(predicted_calls - 1.96 * std_dev)),
            int(predicted_calls + 1.96 * std_dev)
        )

        factors = {
            "complexity": complexity_factor,
            "files": file_factor,
            "loc": loc_factor,
            "team_size": team_factor,
        }

        return PerformancePrediction(
            metric_name="tool_usage",
            predicted_value=predicted_calls,
            confidence_interval=confidence_interval,
            confidence=0.58,
            factors=factors,
        )

    # =========================================================================
    # Model Training
    # =========================================================================

    def train_execution_time_model(
        self,
        training_data: List[Dict[str, Any]],
    ) -> None:
        """Train execution time prediction model.

        Args:
            training_data: List of training examples
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        self._train_model(
            metric_name="execution_time",
            training_data=training_data,
            model_class=RandomForestRegressor,
            model_params={"n_estimators": 100, "max_depth": 10},
        )

    def train_success_model(
        self,
        training_data: List[Dict[str, Any]],
    ) -> None:
        """Train success rate prediction model.

        Args:
            training_data: List of training examples
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        self._train_model(
            metric_name="success_rate",
            training_data=training_data,
            model_class=RandomForestRegressor,
            model_params={"n_estimators": 100, "max_depth": 10},
        )

    def _train_model(
        self,
        metric_name: str,
        training_data: List[Dict[str, Any]],
        model_class: Any,
        model_params: Dict[str, Any],
    ) -> None:
        """Train a prediction model.

        Args:
            metric_name: Name of metric
            training_data: Training examples
            model_class: Model class to use
            model_params: Parameters for model
        """
        from sklearn.preprocessing import StandardScaler

        # Prepare features
        X = []
        y = []

        for example in training_data:
            task_features = example["task_features"]
            team_features = example["team_features"]
            formation = example["formation"]
            value = example["value"]

            feature_vector = self._create_feature_vector(
                task_features, team_features, formation
            )

            X.append(feature_vector)
            y.append(value)

        X = np.array(X)
        y = np.array(y)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = model_class(**model_params, random_state=42)
        model.fit(X_scaled, y)

        # Store
        self._models[metric_name] = model
        self._scalers[metric_name] = scaler

        logger.info(f"Trained {metric_name} model on {len(training_data)} examples")

    def save_models(self, path: Path) -> None:
        """Save trained models to file.

        Args:
            path: Path to save models
        """
        if not self._models:
            logger.warning("No models to save")
            return

        try:
            model_data = {
                "models": self._models,
                "scalers": self._scalers,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved models to {path}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def load_models(self, path: Path) -> None:
        """Load trained models from file.

        Args:
            path: Path to load models from
        """
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self._models = model_data.get("models", {})
            self._scalers = model_data.get("scalers", {})

            logger.info(f"Loaded models from {path}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")


__all__ = [
    "PerformancePrediction",
    "PerformancePredictor",
]
