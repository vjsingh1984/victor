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

"""ML-based team performance prediction system.

This module provides machine learning models for predicting team performance,
execution time, success probability, and optimal team configurations.

Example:
    from victor.teams.team_predictor import TeamPredictor

    predictor = TeamPredictor()
    predictor.load_model("team_performance_model")

    # Predict execution time
    predicted_time = predictor.predict_execution_time(
        team_config=team_config,
        task=task,
        historical_data=history
    )

    # Predict success probability
    success_prob = predictor.predict_success_probability(
        formation=TeamFormation.PARALLEL,
        members=members,
        task=task
    )
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from victor.teams.types import TeamConfig, TeamFormation, TeamMember

logger = logging.getLogger(__name__)


# =============================================================================
# Prediction Types
# =============================================================================


class PredictionMetric(str, Enum):
    """Types of predictions available."""

    EXECUTION_TIME = "execution_time"
    SUCCESS_PROBABILITY = "success_probability"
    TOOL_CALLS = "tool_calls"
    QUALITY_SCORE = "quality_score"
    ITERATION_COUNT = "iteration_count"
    FORMATION_SUITABILITY = "formation_suitability"


@dataclass
class TaskFeatures:
    """Feature representation of a task for ML prediction.

    Attributes:
        complexity: Task complexity (0.0-1.0)
        estimated_lines_of_code: Estimated LOC to change
        file_count: Number of files involved
        domain: Domain area (e.g., "security", "ui", "database")
        required_expertise: List of required expertise areas
        urgency: Urgency level (0.0-1.0)
        dependencies: Number of dependencies
        novelty: How novel is the task (0.0-1.0)
    """

    complexity: float = 0.5
    estimated_lines_of_code: int = 100
    file_count: int = 1
    domain: str = "general"
    required_expertise: List[str] = field(default_factory=list)
    urgency: float = 0.5
    dependencies: int = 0
    novelty: float = 0.5

    def to_feature_vector(self) -> np.ndarray:
        """Convert to numpy feature vector for ML.

        Returns:
            Feature vector
        """
        return np.array([
            self.complexity,
            self.estimated_lines_of_code,
            self.file_count,
            self.urgency,
            self.dependencies,
            self.novelty,
            len(self.required_expertise),
        ])

    @classmethod
    def from_task(cls, task: str, context: Dict[str, Any]) -> "TaskFeatures":
        """Extract features from task description and context.

        Args:
            task: Task description
            context: Execution context

        Returns:
            TaskFeatures instance
        """
        # Simple heuristic-based feature extraction
        # In production, this would use NLP models

        # Estimate complexity from task length and keywords
        complexity = min(len(task) / 1000, 1.0)

        # Extract domain from context
        domain = context.get("domain", "general")

        # Extract required expertise
        required_expertise = context.get("required_expertise", [])
        if isinstance(required_expertise, str):
            required_expertise = [required_expertise]

        # Estimate file count from context
        file_count = context.get("file_count", 1)

        return cls(
            complexity=complexity,
            estimated_lines_of_code=context.get("estimated_loc", 100),
            file_count=file_count,
            domain=domain,
            required_expertise=required_expertise,
            urgency=context.get("urgency", 0.5),
            dependencies=context.get("dependencies", 0),
            novelty=context.get("novelty", 0.5),
        )


@dataclass
class TeamFeatures:
    """Feature representation of a team for ML prediction.

    Attributes:
        member_count: Number of team members
        formation: Formation pattern
        total_tool_budget: Total tool budget
        expertise_coverage: Coverage of required expertise (0.0-1.0)
        diversity: Diversity of team composition (0.0-1.0)
        avg_confidence: Average member confidence
        has_manager: Whether team has a manager
        max_delegation_depth: Maximum delegation depth
    """

    member_count: int
    formation: str
    total_tool_budget: int
    expertise_coverage: float = 0.5
    diversity: float = 0.5
    avg_confidence: float = 0.5
    has_manager: bool = False
    max_delegation_depth: int = 0

    def to_feature_vector(self) -> np.ndarray:
        """Convert to numpy feature vector for ML.

        Returns:
            Feature vector
        """
        return np.array([
            self.member_count,
            self.total_tool_budget,
            self.expertise_coverage,
            self.diversity,
            self.avg_confidence,
            float(self.has_manager),
            self.max_delegation_depth,
        ])

    @classmethod
    def from_team_config(cls, team_config: "TeamConfig", task_features: TaskFeatures) -> "TeamFeatures":
        """Extract features from team configuration.

        Args:
            team_config: Team configuration
            task_features: Task features for expertise matching

        Returns:
            TeamFeatures instance
        """
        # Calculate expertise coverage
        if task_features.required_expertise:
            team_expertise = set()
            for member in team_config.members:
                team_expertise.update(member.expertise)

            covered = len(set(task_features.required_expertise) & team_expertise)
            expertise_coverage = covered / len(task_features.required_expertise)
        else:
            expertise_coverage = 1.0

        # Calculate diversity (variety of roles)
        roles = set(m.role.value for m in team_config.members)
        diversity = len(roles) / len(team_config.members) if team_config.members else 0.0

        # Check for manager
        has_manager = any(m.is_manager for m in team_config.members)

        # Max delegation depth
        max_delegation_depth = max((m.max_delegation_depth for m in team_config.members), default=0)

        return cls(
            member_count=len(team_config.members),
            formation=team_config.formation.value,
            total_tool_budget=team_config.total_tool_budget,
            expertise_coverage=expertise_coverage,
            diversity=diversity,
            avg_confidence=0.5,  # Could be computed from historical performance
            has_manager=has_manager,
            max_delegation_depth=max_delegation_depth,
        )


@dataclass
class PredictionResult:
    """Result of a prediction.

    Attributes:
        metric: Type of prediction
        predicted_value: Predicted value
        confidence: Confidence in prediction (0.0-1.0)
        explanation: Human-readable explanation
        metadata: Additional metadata
        timestamp: When prediction was made
    """

    metric: PredictionMetric
    predicted_value: Union[float, int, str]
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric.value,
            "predicted_value": self.predicted_value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Historical Data Storage
# =============================================================================


@dataclass
class TeamExecutionRecord:
    """Record of a team execution for training/improving predictions.

    Attributes:
        team_config_hash: Hash of team configuration
        task_features: Features of the task
        team_features: Features of the team
        formation: Formation used
        execution_time: Actual execution time in seconds
        success: Whether execution succeeded
        tool_calls_used: Actual tool calls used
        quality_score: Quality score achieved
        iteration_count: Iterations used
        timestamp: When execution occurred
        metadata: Additional metadata
    """

    team_config_hash: str
    task_features: TaskFeatures
    team_features: TeamFeatures
    formation: str
    execution_time: float
    success: bool
    tool_calls_used: int
    quality_score: float
    iteration_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "team_config_hash": self.team_config_hash,
            "task_features": self.task_features.__dict__,
            "team_features": self.team_features.__dict__,
            "formation": self.formation,
            "execution_time": self.execution_time,
            "success": self.success,
            "tool_calls_used": self.tool_calls_used,
            "quality_score": self.quality_score,
            "iteration_count": self.iteration_count,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamExecutionRecord":
        """Create from dictionary."""
        return cls(
            team_config_hash=data["team_config_hash"],
            task_features=TaskFeatures(**data["task_features"]),
            team_features=TeamFeatures(**data["team_features"]),
            formation=data["formation"],
            execution_time=data["execution_time"],
            success=data["success"],
            tool_calls_used=data["tool_calls_used"],
            quality_score=data["quality_score"],
            iteration_count=data["iteration_count"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Team Predictor
# =============================================================================


class TeamPredictor:
    """ML-based team performance predictor.

    Uses machine learning models to predict various team performance metrics
    based on historical execution data and task/team features.

    Example:
        predictor = TeamPredictor()

        # Load historical data
        predictor.load_historical_data("team_executions.jsonl")

        # Train model
        predictor.train_model()

        # Make predictions
        result = predictor.predict(
            metric=PredictionMetric.EXECUTION_TIME,
            team_config=team_config,
            task="Implement authentication"
        )
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        historical_data_path: Optional[Path] = None,
    ):
        """Initialize team predictor.

        Args:
            model_path: Path to saved model (for loading)
            historical_data_path: Path to historical data file
        """
        self.model_path = model_path
        self.historical_data_path = historical_data_path

        # Models (lazy loaded)
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}

        # Historical data
        self._historical_records: List[TeamExecutionRecord] = []

        # Load if paths provided
        if historical_data_path and historical_data_path.exists():
            self.load_historical_data(historical_data_path)

        if model_path and model_path.exists():
            self.load_model(model_path)

    def predict(
        self,
        metric: PredictionMetric,
        team_config: "TeamConfig",
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """Make a prediction for the given metric.

        Args:
            metric: Type of prediction
            team_config: Team configuration
            task: Task description
            context: Additional context

        Returns:
            Prediction result
        """
        context = context or {}

        # Extract features
        task_features = TaskFeatures.from_task(task, context)
        team_features = TeamFeatures.from_team_config(team_config, task_features)

        # Route to specific prediction method
        if metric == PredictionMetric.EXECUTION_TIME:
            return self._predict_execution_time(task_features, team_features)
        elif metric == PredictionMetric.SUCCESS_PROBABILITY:
            return self._predict_success_probability(task_features, team_features)
        elif metric == PredictionMetric.TOOL_CALLS:
            return self._predict_tool_calls(task_features, team_features)
        elif metric == PredictionMetric.QUALITY_SCORE:
            return self._predict_quality_score(task_features, team_features)
        elif metric == PredictionMetric.ITERATION_COUNT:
            return self._predict_iteration_count(task_features, team_features)
        elif metric == PredictionMetric.FORMATION_SUITABILITY:
            return self._predict_formation_suitability(task_features, team_features)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _predict_execution_time(
        self, task_features: TaskFeatures, team_features: TeamFeatures
    ) -> PredictionResult:
        """Predict execution time.

        Args:
            task_features: Task features
            team_features: Team features

        Returns:
            Prediction result
        """
        # Simple heuristic-based prediction (in production, use trained ML model)
        base_time = 30.0  # Base time in seconds

        # Adjust based on features
        complexity_factor = 1.0 + task_features.complexity * 2.0
        loc_factor = 1.0 + (task_features.estimated_lines_of_code / 1000) * 0.5
        file_factor = 1.0 + (task_features.file_count - 1) * 0.2
        member_factor = max(0.5, 1.0 - (team_features.member_count - 1) * 0.1)
        budget_factor = 1.0 + (team_features.total_tool_budget / 100) * 0.3

        predicted_time = base_time * complexity_factor * loc_factor * file_factor * member_factor * budget_factor

        # Confidence based on historical data
        confidence = min(0.9, 0.5 + len(self._historical_records) / 1000)

        return PredictionResult(
            metric=PredictionMetric.EXECUTION_TIME,
            predicted_value=round(predicted_time, 2),
            confidence=confidence,
            explanation=(
                f"Predicted execution time based on task complexity ({task_features.complexity:.2f}), "
                f"estimated LOC ({task_features.estimated_lines_of_code}), "
                f"file count ({task_features.file_count}), "
                f"and team size ({team_features.member_count})"
            ),
            metadata={
                "base_time": base_time,
                "complexity_factor": complexity_factor,
                "loc_factor": loc_factor,
                "file_factor": file_factor,
                "member_factor": member_factor,
            },
        )

    def _predict_success_probability(
        self, task_features: TaskFeatures, team_features: TeamFeatures
    ) -> PredictionResult:
        """Predict success probability.

        Args:
            task_features: Task features
            team_features: Team features

        Returns:
            Prediction result
        """
        # Heuristic-based prediction
        base_prob = 0.8

        # Adjust based on expertise coverage
        expertise_bonus = team_features.expertise_coverage * 0.15

        # Adjust based on complexity
        complexity_penalty = task_features.complexity * 0.1

        # Adjust based on team size
        team_bonus = min(0.1, (team_features.member_count - 1) * 0.03)

        # Adjust based on budget
        budget_bonus = min(0.05, team_features.total_tool_budget / 2000 * 0.05)

        predicted_prob = base_prob + expertise_bonus - complexity_penalty + team_bonus + budget_bonus
        predicted_prob = max(0.0, min(1.0, predicted_prob))

        confidence = min(0.95, 0.6 + len(self._historical_records) / 500)

        return PredictionResult(
            metric=PredictionMetric.SUCCESS_PROBABILITY,
            predicted_value=round(predicted_prob, 3),
            confidence=confidence,
            explanation=(
                f"Predicted success probability based on expertise coverage "
                f"({team_features.expertise_coverage:.2f}), "
                f"task complexity ({task_features.complexity:.2f}), "
                f"and team size ({team_features.member_count})"
            ),
            metadata={
                "base_prob": base_prob,
                "expertise_bonus": expertise_bonus,
                "complexity_penalty": complexity_penalty,
            },
        )

    def _predict_tool_calls(
        self, task_features: TaskFeatures, team_features: TeamFeatures
    ) -> PredictionResult:
        """Predict tool calls needed.

        Args:
            task_features: Task features
            team_features: Team features

        Returns:
            Prediction result
        """
        # Heuristic-based prediction
        base_calls = 20

        # Adjust based on features
        complexity_factor = 1.0 + task_features.complexity * 1.5
        file_factor = 1.0 + task_features.file_count * 0.3
        loc_factor = 1.0 + (task_features.estimated_lines_of_code / 500) * 0.5

        predicted_calls = int(base_calls * complexity_factor * file_factor * loc_factor)
        predicted_calls = min(predicted_calls, team_features.total_tool_budget)

        confidence = min(0.85, 0.5 + len(self._historical_records) / 800)

        return PredictionResult(
            metric=PredictionMetric.TOOL_CALLS,
            predicted_value=predicted_calls,
            confidence=confidence,
            explanation=(
                f"Predicted tool calls based on complexity ({task_features.complexity:.2f}), "
                f"files ({task_features.file_count}), and LOC ({task_features.estimated_lines_of_code})"
            ),
        )

    def _predict_quality_score(
        self, task_features: TaskFeatures, team_features: TeamFeatures
    ) -> PredictionResult:
        """Predict quality score.

        Args:
            task_features: Task features
            team_features: Team features

        Returns:
            Prediction result
        """
        # Heuristic-based prediction
        base_score = 0.75

        # Adjust based on expertise coverage
        expertise_bonus = team_features.expertise_coverage * 0.15

        # Adjust based on diversity
        diversity_bonus = team_features.diversity * 0.1

        # Adjust based on complexity (harder tasks may have lower quality)
        complexity_penalty = task_features.complexity * 0.05

        predicted_score = base_score + expertise_bonus + diversity_bonus - complexity_penalty
        predicted_score = max(0.0, min(1.0, predicted_score))

        confidence = min(0.8, 0.5 + len(self._historical_records) / 600)

        return PredictionResult(
            metric=PredictionMetric.QUALITY_SCORE,
            predicted_value=round(predicted_score, 3),
            confidence=confidence,
            explanation=(
                f"Predicted quality score based on expertise coverage "
                f"({team_features.expertise_coverage:.2f}) and team diversity "
                f"({team_features.diversity:.2f})"
            ),
        )

    def _predict_iteration_count(
        self, task_features: TaskFeatures, team_features: TeamFeatures
    ) -> PredictionResult:
        """Predict iteration count.

        Args:
            task_features: Task features
            team_features: Team features

        Returns:
            Prediction result
        """
        # Heuristic-based prediction
        base_iterations = 5

        # Adjust based on complexity
        complexity_factor = 1.0 + task_features.complexity * 2.0

        # Adjust based on file count
        file_factor = 1.0 + task_features.file_count * 0.2

        predicted_iterations = int(base_iterations * complexity_factor * file_factor)

        confidence = min(0.8, 0.5 + len(self._historical_records) / 700)

        return PredictionResult(
            metric=PredictionMetric.ITERATION_COUNT,
            predicted_value=predicted_iterations,
            confidence=confidence,
            explanation=f"Predicted iterations based on complexity ({task_features.complexity:.2f})",
        )

    def _predict_formation_suitability(
        self, task_features: TaskFeatures, team_features: TeamFeatures
    ) -> PredictionResult:
        """Predict formation suitability for task.

        Args:
            task_features: Task features
            team_features: Team features

        Returns:
            Prediction result with formation recommendation
        """
        # Score each formation
        formation_scores = {}

        # Sequential: good for simple, single-file tasks
        sequential_score = (
            1.0 - task_features.complexity
        ) * 0.7 + (1.0 / max(1, task_features.file_count)) * 0.3
        formation_scores["sequential"] = sequential_score

        # Parallel: good for multi-file, complex tasks
        parallel_score = (
            task_features.complexity * 0.5 + min(1.0, task_features.file_count / 3) * 0.5
        )
        formation_scores["parallel"] = parallel_score

        # Hierarchical: good for complex tasks with clear manager
        hierarchical_score = (
            task_features.complexity * 0.6 + float(team_features.has_manager) * 0.4
        )
        formation_scores["hierarchical"] = hierarchical_score

        # Pipeline: good for sequential dependencies
        pipeline_score = (
            1.0 / max(1, task_features.file_count)
        ) * 0.6 + task_features.complexity * 0.4
        formation_scores["pipeline"] = pipeline_score

        # Consensus: good for high-quality requirements
        consensus_score = (1.0 - task_features.urgency) * 0.6 + team_features.diversity * 0.4
        formation_scores["consensus"] = consensus_score

        # Select best formation
        best_formation = max(formation_scores, key=formation_scores.get)
        best_score = formation_scores[best_formation]

        confidence = min(0.85, 0.6 + len(self._historical_records) / 900)

        return PredictionResult(
            metric=PredictionMetric.FORMATION_SUITABILITY,
            predicted_value=best_formation,
            confidence=confidence * best_score,
            explanation=(
                f"Recommended '{best_formation}' formation (score: {best_score:.2f}) "
                f"based on task complexity ({task_features.complexity:.2f}), "
                f"file count ({task_features.file_count}), and team composition"
            ),
            metadata={
                "formation_scores": formation_scores,
                "has_manager": team_features.has_manager,
                "diversity": team_features.diversity,
            },
        )

    def record_execution(
        self,
        team_config: "TeamConfig",
        task: str,
        context: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> None:
        """Record a team execution for future predictions.

        Args:
            team_config: Team configuration used
            task: Task description
            context: Execution context
            execution_result: Result from execution
        """
        import hashlib

        # Extract features
        task_features = TaskFeatures.from_task(task, context)
        team_features = TeamFeatures.from_team_config(team_config, task_features)

        # Create config hash
        config_str = json.dumps(team_config.to_dict(), sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        # Create record
        record = TeamExecutionRecord(
            team_config_hash=config_hash,
            task_features=task_features,
            team_features=team_features,
            formation=team_config.formation.value,
            execution_time=execution_result.get("total_duration", 0.0),
            success=execution_result.get("success", False),
            tool_calls_used=execution_result.get("total_tool_calls", 0),
            quality_score=execution_result.get("quality_score", 0.0),
            iteration_count=execution_result.get("iteration_count", 0),
            metadata={
                "task": task[:500],  # Truncate long tasks
                "member_count": len(team_config.members),
            },
        )

        self._historical_records.append(record)

    def load_historical_data(self, path: Path) -> None:
        """Load historical execution data.

        Args:
            path: Path to JSONL file with historical data
        """
        self._historical_records.clear()

        try:
            with open(path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    record = TeamExecutionRecord.from_dict(data)
                    self._historical_records.append(record)

            logger.info(f"Loaded {len(self._historical_records)} historical records from {path}")
        except Exception as e:
            logger.error(f"Failed to load historical data from {path}: {e}")

    def save_historical_data(self, path: Path) -> None:
        """Save historical execution data.

        Args:
            path: Path to save JSONL file
        """
        try:
            with open(path, "w") as f:
                for record in self._historical_records:
                    f.write(json.dumps(record.to_dict()) + "\n")

            logger.info(f"Saved {len(self._historical_records)} historical records to {path}")
        except Exception as e:
            logger.error(f"Failed to save historical data to {path}: {e}")

    def train_model(self) -> None:
        """Train prediction models on historical data.

        This is a placeholder for future ML model training.
        In production, this would train scikit-learn or PyTorch models.
        """
        if len(self._historical_records) < 10:
            logger.warning("Not enough historical data to train models (need at least 10 records)")
            return

        # Placeholder for ML training
        # In production, this would:
        # 1. Prepare feature matrices from historical records
        # 2. Train separate models for each prediction metric
        # 3. Validate and save models

        logger.info(f"Training models on {len(self._historical_records)} records (placeholder)")

    def load_model(self, path: Path) -> None:
        """Load trained model.

        Args:
            path: Path to saved model file
        """
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)
                self._models = model_data.get("models", {})
                self._scalers = model_data.get("scalers", {})

            logger.info(f"Loaded models from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")

    def save_model(self, path: Path) -> None:
        """Save trained model.

        Args:
            path: Path to save model file
        """
        try:
            model_data = {"models": self._models, "scalers": self._scalers}

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved models to {path}")
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {e}")

    def get_historical_stats(self) -> Dict[str, Any]:
        """Get statistics about historical data.

        Returns:
            Dictionary with statistics
        """
        if not self._historical_records:
            return {
                "total_records": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "avg_tool_calls": 0.0,
            }

        successful = [r for r in self._historical_records if r.success]
        success_rate = len(successful) / len(self._historical_records)

        avg_execution_time = sum(r.execution_time for r in self._historical_records) / len(
            self._historical_records
        )
        avg_tool_calls = sum(r.tool_calls_used for r in self._historical_records) / len(
            self._historical_records
        )

        return {
            "total_records": len(self._historical_records),
            "success_rate": round(success_rate, 3),
            "avg_execution_time": round(avg_execution_time, 2),
            "avg_tool_calls": round(avg_tool_calls, 1),
        }


__all__ = [
    "PredictionMetric",
    "TaskFeatures",
    "TeamFeatures",
    "PredictionResult",
    "TeamExecutionRecord",
    "TeamPredictor",
]
