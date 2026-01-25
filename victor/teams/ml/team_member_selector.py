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

"""ML-based team member selection.

This module implements machine learning models for optimally selecting
team members based on task requirements, expertise matching, and
historical performance data.

Example:
    from victor.teams.ml.team_member_selector import TeamMemberSelector

    selector = TeamMemberSelector()
    selector.load_model("member_selector.pkl")

    # Select optimal members
    selected = selector.select_members(
        task="Implement authentication",
        available_members=member_pool,
        task_features=task_features,
        top_k=3
    )
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from victor.teams.team_predictor import TaskFeatures
    from victor.teams.types import TeamMember

logger = logging.getLogger(__name__)


@dataclass
class MemberScore:
    """Score for a team member.

    Attributes:
        member: Team member
        score: Selection score
        confidence: Confidence in score
        reasons: Reasons for score
    """

    member: "TeamMember"
    score: float
    confidence: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "member_id": self.member.id,
            "member_name": self.member.name,
            "score": self.score,
            "confidence": self.confidence,
            "reasons": self.reasons,
        }


class TeamMemberSelector:
    """ML-based team member selection.

    Uses machine learning to score and rank team members based on
    their fit for a given task.

    Example:
        selector = TeamMemberSelector()

        # Train on historical data
        selector.train(training_data)

        # Select members for new task
        selected = selector.select_members(
            task="Implement OAuth",
            available_members=pool,
            task_features=features,
            top_k=5
        )
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_heuristic: bool = True,
    ):
        """Initialize team member selector.

        Args:
            model_path: Path to trained model
            use_heuristic: Use heuristic scoring if no model loaded
        """
        self.model_path = model_path
        self.use_heuristic = use_heuristic

        # Model (lazy loaded)
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None

        if model_path and model_path.exists():
            self.load_model(model_path)

    def select_members(
        self,
        task: str,
        available_members: List["TeamMember"],
        task_features: "TaskFeatures",
        top_k: int = 5,
    ) -> List[MemberScore]:
        """Select optimal members for task.

        Args:
            task: Task description
            available_members: Pool of available members
            task_features: Features of the task
            top_k: Number of members to select

        Returns:
            List of scored members, sorted by score
        """
        scored_members = []

        for member in available_members:
            score, confidence, reasons = self._score_member(member, task, task_features)
            scored_members.append(
                MemberScore(
                    member=member,
                    score=score,
                    confidence=confidence,
                    reasons=reasons,
                )
            )

        # Sort by score
        scored_members.sort(key=lambda m: m.score, reverse=True)

        return scored_members[:top_k]

    def _score_member(
        self,
        member: "TeamMember",
        task: str,
        task_features: "TaskFeatures",
    ) -> Tuple[float, float, List[str]]:
        """Score a member for the task.

        Args:
            member: Team member to score
            task: Task description
            task_features: Task features

        Returns:
            Tuple of (score, confidence, reasons)
        """
        if self._model is not None:
            return self._score_with_model(member, task_features)
        elif self.use_heuristic:
            return self._score_with_heuristic(member, task, task_features)
        else:
            return 0.5, 0.0, ["No scoring method available"]

    def _score_with_model(
        self,
        member: "TeamMember",
        task_features: "TaskFeatures",
    ) -> Tuple[float, float, List[str]]:
        """Score member using trained model.

        Args:
            member: Team member
            task_features: Task features

        Returns:
            Tuple of (score, confidence, reasons)
        """
        # Extract features
        member_features = self._extract_member_features(member, task_features)

        # Create feature vector
        feature_vector = np.concatenate(
            [
                task_features.to_feature_vector(),
                member_features,
            ]
        )

        # Scale features
        if self._scaler:
            feature_vector = self._scaler.transform([feature_vector])[0]

        # Predict score
        if self._model is not None:
            score = float(self._model.predict_proba([feature_vector])[0][1])
        else:
            raise RuntimeError("Model not trained")

        return score, 0.8, [f"Model-based score: {score:.3f}"]

    def _score_with_heuristic(
        self,
        member: "TeamMember",
        task: str,
        task_features: "TaskFeatures",
    ) -> Tuple[float, float, List[str]]:
        """Score member using heuristic rules.

        Args:
            member: Team member
            task: Task description
            task_features: Task features

        Returns:
            Tuple of (score, confidence, reasons)
        """
        score = 0.0
        reasons = []

        # Expertise matching
        if task_features.required_expertise:
            match_count = len(set(member.expertise) & set(task_features.required_expertise))
            expertise_score = match_count / len(task_features.required_expertise)
            score += expertise_score * 0.4
            if match_count > 0:
                reasons.append(f"Has {match_count} required expertise areas")
        else:
            score += 0.2  # Base score for expertise

        # Role appropriateness
        role_scores = {
            "planner": 0.15,
            "researcher": 0.15,
            "executor": 0.20,
            "reviewer": 0.10,
            "tester": 0.10,
        }
        role_score = role_scores.get(member.role.value, 0.1)
        score += role_score
        reasons.append(f"Role {member.role.value} contributes {role_score:.2f}")

        # Tool budget adequacy
        budget_score = min(1.0, member.tool_budget / 50) * 0.15
        score += budget_score
        if member.tool_budget >= 50:
            reasons.append("Adequate tool budget")

        # Experience level (based on backstory length as proxy)
        experience_score = min(0.1, len(member.backstory) / 1000) * 0.1
        score += experience_score
        if member.backstory:
            reasons.append("Has domain experience")

        # Delegation capability
        if member.can_delegate:
            score += 0.05
            reasons.append("Can delegate tasks")

        # Memory capability
        if member.memory_enabled:
            score += 0.05
            reasons.append("Has persistent memory")

        return min(1.0, score), 0.6, reasons

    def _extract_member_features(
        self, member: "TeamMember", task_features: "TaskFeatures"
    ) -> np.ndarray:
        """Extract numerical features from member.

        Args:
            member: Team member
            task_features: Task features for context

        Returns:
            Feature vector
        """
        features = []

        # Basic features
        features.append(int(member.tool_budget))
        features.append(1.0 if member.can_delegate else 0.0)
        features.append(1.0 if member.is_manager else 0.0)
        features.append(int(member.max_delegation_depth))
        features.append(1.0 if member.memory_enabled else 0.0)
        features.append(int(len(member.expertise)))
        features.append(int(len(member.backstory)))

        # Expertise overlap
        if task_features.required_expertise:
            overlap = len(set(member.expertise) & set(task_features.required_expertise))
            features.append(int(overlap / len(task_features.required_expertise)))
        else:
            features.append(0)

        # Role encoding (one-hot)
        roles = ["planner", "researcher", "executor", "reviewer", "tester"]
        for role in roles:
            features.append(1.0 if member.role.value == role else 0.0)

        return np.array(features)

    def train(
        self,
        training_data: List[Dict[str, Any]],
    ) -> None:
        """Train member selection model.

        Args:
            training_data: List of training examples with:
                - member: TeamMember
                - task_features: TaskFeatures
                - success: bool (whether member was successful)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        # Prepare features
        X = []
        y = []

        for example in training_data:
            member = example["member"]
            task_features = example["task_features"]
            success = example["success"]

            # Extract features
            member_features = self._extract_member_features(member, task_features)
            feature_vector = np.concatenate(
                [
                    task_features.to_feature_vector(),
                    member_features,
                ]
            )

            X.append(feature_vector)
            y.append(int(success))

        X_array = np.array(X)
        y_array = np.array(y)

        X_list: list[Any] = X_array.tolist()
        y_list: list[int] = y_array.tolist()

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_list)

        # Train model
        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )
        self._model.fit(X_scaled, y_list)

        logger.info(f"Trained member selection model on {len(training_data)} examples")

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
            "complexity",
            "estimated_loc",
            "file_count",
            "urgency",
            "dependencies",
            "novelty",
            "required_expertise_count",
            # Member features
            "tool_budget",
            "can_delegate",
            "is_manager",
            "max_delegation_depth",
            "memory_enabled",
            "expertise_count",
            "backstory_length",
            "expertise_overlap",
            # Role one-hot
            "role_planner",
            "role_researcher",
            "role_executor",
            "role_reviewer",
            "role_tester",
        ]

        importances = dict(zip(feature_names, self._model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


__all__ = [
    "MemberScore",
    "TeamMemberSelector",
]
