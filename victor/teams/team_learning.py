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

"""Team learning and adaptation system.

This module implements learning mechanisms that allow teams to improve
over time through experience, feedback, and adaptation.

Example:
    from victor.teams.team_learning import TeamLearningSystem

    learner = TeamLearningSystem()

    # Record experience
    learner.record_experience(
        team_config=team_config,
        task=task,
        result=result
    )

    # Get recommendations
    recommendations = learner.get_improvement_recommendations(team_id="my_team")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from victor.teams.team_analytics import ExecutionRecord
    from victor.teams.types import TeamConfig, TeamFormation

logger = logging.getLogger(__name__)


# =============================================================================
# Learning Types
# =============================================================================


class LearningStrategy(str, Enum):
    """Learning strategies for team adaptation."""

    REINFORCEMENT = "reinforcement"  # Learn from rewards/penalties
    SUPERVISED = "supervised"  # Learn from labeled examples
    UNSUPERVISED = "unsupervised"  # Learn patterns from data
    TRANSFER = "transfer"  # Transfer knowledge from similar tasks
    META_LEARNING = "meta_learning"  # Learn how to learn


@dataclass
class TeamExperience:
    """A single experience for learning.

    Attributes:
        experience_id: Unique identifier
        team_id: Team identifier
        task: Task description
        team_config: Team configuration used
        result: Execution result
        reward: Reward value (for reinforcement learning)
        timestamp: When experience occurred
        metadata: Additional metadata
    """

    experience_id: str
    team_id: str
    task: str
    team_config: "TeamConfig"
    result: Dict[str, Any]
    reward: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experience_id": self.experience_id,
            "team_id": self.team_id,
            "task": self.task,
            "team_config": self.team_config.to_dict(),
            "result": self.result,
            "reward": self.reward,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AdaptationRecommendation:
    """Recommendation for team adaptation.

    Attributes:
        recommendation_type: Type of adaptation
        description: Description of recommendation
        expected_improvement: Expected improvement (0.0-1.0)
        confidence: Confidence in recommendation (0.0-1.0)
        changes: Specific changes to make
        rationale: Why this recommendation is made
    """

    recommendation_type: str
    description: str
    expected_improvement: float
    confidence: float
    changes: Dict[str, Any]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommendation_type": self.recommendation_type,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "changes": self.changes,
            "rationale": self.rationale,
        }


@dataclass
class LearningProgress:
    """Progress of team learning over time.

    Attributes:
        team_id: Team identifier
        total_experiences: Number of experiences
        avg_reward: Average reward
        reward_trend: Trend in rewards (improving/stable/declining)
        skill_level: Current skill level (0.0-1.0)
        learning_rate: How fast team is learning
        adaptation_count: Number of adaptations made
    """

    team_id: str
    total_experiences: int
    avg_reward: float
    reward_trend: str
    skill_level: float
    learning_rate: float
    adaptation_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "team_id": self.team_id,
            "total_experiences": self.total_experiences,
            "avg_reward": self.avg_reward,
            "reward_trend": self.reward_trend,
            "skill_level": self.skill_level,
            "learning_rate": self.learning_rate,
            "adaptation_count": self.adaptation_count,
        }


# =============================================================================
# Team Learning System
# =============================================================================


class TeamLearningSystem:
    """Learning and adaptation system for teams.

    Tracks team experiences, learns from outcomes, and provides
    recommendations for team improvement.

    Example:
        learner = TeamLearningSystem()

        # Record successful execution
        learner.record_experience(
            team_config=team_config,
            task="Implement feature",
            result={"success": True, "quality": 0.9},
            reward=1.0
        )

        # Get learning progress
        progress = learner.get_progress(team_id="my_team")

        # Get recommendations
        recommendations = learner.get_recommendations(team_id="my_team")
    """

    def __init__(
        self,
        learning_strategy: LearningStrategy = LearningStrategy.REINFORCEMENT,
        storage_path: Optional[Path] = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
    ):
        """Initialize team learning system.

        Args:
            learning_strategy: Learning strategy to use
            storage_path: Path to store learning data
            learning_rate: Learning rate for updates
            discount_factor: Discount factor for future rewards
        """
        self.learning_strategy = learning_strategy
        self.storage_path = storage_path
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Knowledge storage
        self._experiences: Dict[str, TeamExperience] = {}
        self._team_experiences: Dict[str, List[str]] = defaultdict(list)
        self._q_table: Dict[Tuple[str, str], float] = {}  # (team_id, state) -> value
        self._formation_performance: Dict[str, Dict[str, List[float]]] = {}
        self._member_performance: Dict[str, Dict[str, List[float]]] = {}

        # Load existing data
        if storage_path and storage_path.exists():
            self.load_data(storage_path)

    def record_experience(
        self,
        team_config: "TeamConfig",
        task: str,
        result: Dict[str, Any],
        team_id: str = "default",
        reward: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a team experience for learning.

        Args:
            team_config: Team configuration used
            task: Task description
            result: Execution result
            team_id: Team identifier
            reward: Reward value (computed if not provided)
            metadata: Additional metadata

        Returns:
            Experience ID
        """
        import uuid

        experience_id = f"exp_{uuid.uuid4().hex[:8]}"

        # Compute reward if not provided
        if reward is None:
            reward = self._compute_reward(result)

        # Create experience
        experience = TeamExperience(
            experience_id=experience_id,
            team_id=team_id,
            task=task,
            team_config=team_config,
            result=result,
            reward=reward,
            metadata=metadata or {},
        )

        # Store
        self._experiences[experience_id] = experience
        self._team_experiences[team_id].append(experience_id)

        # Update knowledge
        self._update_from_experience(experience)

        return experience_id

    def _compute_reward(self, result: Dict[str, Any]) -> float:
        """Compute reward from execution result.

        Args:
            result: Execution result

        Returns:
            Reward value (typically -1.0 to 1.0)
        """
        reward = 0.0

        # Success reward
        if result.get("success", False):
            reward += 0.5

        # Quality reward
        quality = result.get("quality_score", 0.5)
        reward += (quality - 0.5) * 0.5

        # Speed reward (faster is better)
        execution_time = result.get("total_duration", 100)
        if execution_time > 0:
            reward += min(0.2, (100 / execution_time - 1) * 0.1)

        # Efficiency reward (fewer tool calls is better)
        tool_calls = result.get("total_tool_calls", 50)
        budget = result.get("tool_budget", 100)
        if budget > 0:
            efficiency = 1.0 - (tool_calls / budget)
            reward += (efficiency - 0.5) * 0.2

        return float(max(-1.0, min(1.0, reward)))

    def _update_from_experience(self, experience: TeamExperience) -> None:
        """Update knowledge from experience.

        Args:
            experience: Team experience
        """
        # Update Q-table (for reinforcement learning)
        state = self._get_state_key(experience)
        key = (experience.team_id, state)

        if key not in self._q_table:
            self._q_table[key] = 0.0

        # Q-learning update
        current_q = self._q_table[key]
        self._q_table[key] = current_q + self.learning_rate * (experience.reward - current_q)

        # Update formation performance
        formation = experience.team_config.formation.value
        if formation not in self._formation_performance:
            self._formation_performance[formation] = {}
        if "rewards" not in self._formation_performance[formation]:
            self._formation_performance[formation]["rewards"] = []
        if "success" not in self._formation_performance[formation]:
            self._formation_performance[formation]["success"] = []
        self._formation_performance[formation]["rewards"].append(experience.reward)
        self._formation_performance[formation]["success"].append(
            experience.result.get("success", False)
        )

        # Update member performance
        for member_id, member_result in experience.result.get("member_results", {}).items():
            if member_id not in self._member_performance:
                self._member_performance[member_id] = {}
            if "rewards" not in self._member_performance[member_id]:
                self._member_performance[member_id]["rewards"] = []
            if "success" not in self._member_performance[member_id]:
                self._member_performance[member_id]["success"] = []
            self._member_performance[member_id]["rewards"].append(experience.reward)
            self._member_performance[member_id]["success"].append(
                member_result.get("success", False)
            )

    def _get_state_key(self, experience: TeamExperience) -> str:
        """Get state key for experience.

        Args:
            experience: Team experience

        Returns:
            State key string
        """
        # Simple state representation based on task features
        task_complexity = "high" if len(experience.task) > 200 else "low"
        formation = experience.team_config.formation.value
        member_count = len(experience.team_config.members)

        return f"{formation}_{member_count}_{task_complexity}"

    def get_progress(self, team_id: str) -> LearningProgress:
        """Get learning progress for a team.

        Args:
            team_id: Team identifier

        Returns:
            Learning progress
        """
        experience_ids = self._team_experiences.get(team_id, [])

        if not experience_ids:
            return LearningProgress(
                team_id=team_id,
                total_experiences=0,
                avg_reward=0.0,
                reward_trend="stable",
                skill_level=0.0,
                learning_rate=0.0,
                adaptation_count=0,
            )

        experiences = [self._experiences[eid] for eid in experience_ids if eid in self._experiences]

        # Compute metrics
        rewards = [e.reward for e in experiences]
        avg_reward = np.mean(rewards) if rewards else 0.0

        # Compute trend
        if len(rewards) >= 10:
            recent_avg = np.mean(rewards[-10:])
            old_avg = np.mean(rewards[:-10])
            if recent_avg > old_avg * 1.1:
                trend = "improving"
            elif recent_avg < old_avg * 0.9:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Compute skill level (based on recent success rate and reward)
        recent_experiences = experiences[-20:] if len(experiences) >= 20 else experiences
        success_rate = np.mean([e.result.get("success", False) for e in recent_experiences])
        skill_level = (success_rate + (avg_reward + 1) / 2) / 2

        # Learning rate (how fast rewards are improving)
        if len(rewards) >= 20:
            first_half = rewards[: len(rewards) // 2]
            second_half = rewards[len(rewards) // 2 :]
            learning_rate = (np.mean(second_half) - np.mean(first_half)) / len(rewards)
        else:
            learning_rate = 0.0

        return LearningProgress(
            team_id=team_id,
            total_experiences=len(experiences),
            avg_reward=round(avg_reward, 3),
            reward_trend=trend,
            skill_level=round(skill_level, 3),
            learning_rate=round(learning_rate, 4),
            adaptation_count=0,  # Would track actual adaptations
        )

    def get_recommendations(self, team_id: str, top_k: int = 5) -> List[AdaptationRecommendation]:
        """Get adaptation recommendations for a team.

        Args:
            team_id: Team identifier
            top_k: Number of top recommendations

        Returns:
            List of recommendations
        """
        recommendations: List[str] = []
        experience_ids = self._team_experiences.get(team_id, [])
        experiences = [self._experiences[eid] for eid in experience_ids if eid in self._experiences]

        if not experiences:
            return recommendations

        # Analyze formation performance
        formation_rec = self._recommend_formation(experiences)
        if formation_rec:
            recommendations.append(formation_rec)

        # Analyze member performance
        member_rec = self._recommend_member_changes(experiences)
        if member_rec:
            recommendations.append(member_rec)

        # Analyze budget
        budget_rec = self._recommend_budget_changes(experiences)
        if budget_rec:
            recommendations.append(budget_rec)

        # Sort by expected improvement
        recommendations.sort(key=lambda r: r.expected_improvement, reverse=True)

        return recommendations[:top_k]

    def _recommend_formation(
        self, experiences: List[TeamExperience]
    ) -> Optional[AdaptationRecommendation]:
        """Recommend formation changes.

        Args:
            experiences: Team experiences

        Returns:
            Formation recommendation or None
        """
        # Compute average reward per formation
        formation_rewards = defaultdict(list)
        for exp in experiences:
            formation = exp.team_config.formation.value
            formation_rewards[formation].append(exp.reward)

        if len(formation_rewards) < 2:
            return None

        # Find best formation
        avg_rewards = {
            formation: np.mean(rewards) for formation, rewards in formation_rewards.items()
        }
        best_formation = max(avg_rewards.keys(), key=lambda k: avg_rewards[k])

        # Check if current formation is suboptimal
        current_formation = experiences[-1].team_config.formation.value
        if best_formation == current_formation:
            return None

        improvement = (avg_rewards[best_formation] - avg_rewards[current_formation]) / abs(
            avg_rewards[current_formation]
        )

        return AdaptationRecommendation(
            recommendation_type="formation_change",
            description=f"Switch from {current_formation} to {best_formation}",
            expected_improvement=min(1.0, abs(improvement)),
            confidence=min(1.0, len(formation_rewards[current_formation]) / 20),
            changes={"formation": best_formation},
            rationale=f"{best_formation} has {avg_rewards[best_formation]:.3f} avg reward vs "
            f"{avg_rewards[current_formation]:.3f} for {current_formation}",
        )

    def _recommend_member_changes(
        self, experiences: List[TeamExperience]
    ) -> Optional[AdaptationRecommendation]:
        """Recommend member changes.

        Args:
            experiences: Team experiences

        Returns:
            Member recommendation or None
        """
        # Find underperforming members
        member_rewards = defaultdict(list)
        for exp in experiences:
            for member_id, member_result in exp.result.get("member_results", {}).items():
                if member_result.get("success", False):
                    member_rewards[member_id].append(exp.reward)
                else:
                    member_rewards[member_id].append(-0.5)  # Penalty for failure

        if not member_rewards:
            return None

        avg_rewards = {member_id: np.mean(rewards) for member_id, rewards in member_rewards.items()}

        # Find worst performing member
        worst_member = min(avg_rewards.keys(), key=lambda k: avg_rewards[k])
        worst_reward = avg_rewards[worst_member]

        if worst_reward > 0:
            return None  # All members performing adequately

        return AdaptationRecommendation(
            recommendation_type="member_replacement",
            description=f"Consider replacing or retraining {worst_member}",
            expected_improvement=min(1.0, abs(worst_reward)),
            confidence=min(1.0, len(member_rewards[worst_member]) / 10),
            changes={"member_to_replace": worst_member},
            rationale=f"{worst_member} has average reward of {worst_reward:.3f}, below threshold",
        )

    def _recommend_budget_changes(
        self, experiences: List[TeamExperience]
    ) -> Optional[AdaptationRecommendation]:
        """Recommend budget changes.

        Args:
            experiences: Team experiences

        Returns:
            Budget recommendation or None
        """
        # Analyze budget utilization
        tool_calls_list = []
        budgets = []

        for exp in experiences:
            tool_calls = exp.result.get("total_tool_calls", 0)
            budget = exp.team_config.total_tool_budget
            tool_calls_list.append(tool_calls)
            budgets.append(budget)

        if not tool_calls_list:
            return None

        avg_utilization = np.mean([tc / b for tc, b in zip(tool_calls_list, budgets)])

        if avg_utilization > 0.9:
            # Frequently running out of budget
            return AdaptationRecommendation(
                recommendation_type="budget_increase",
                description="Increase tool budget to avoid exhaustion",
                expected_improvement=(avg_utilization - 0.9) * 2,
                confidence=min(1.0, len(experiences) / 15),
                changes={"budget_multiplier": 1.2},
                rationale=f"Average budget utilization is {avg_utilization:.1%}, frequently exhausted",
            )
        elif avg_utilization < 0.5:
            # Underutilizing budget
            return AdaptationRecommendation(
                recommendation_type="budget_decrease",
                description="Decrease tool budget for efficiency",
                expected_improvement=(0.5 - avg_utilization) * 0.5,
                confidence=min(1.0, len(experiences) / 15),
                changes={"budget_multiplier": 0.8},
                rationale=f"Average budget utilization is {avg_utilization:.1%}, underutilized",
            )

        return None

    def get_optimal_formation(self, task: str, team_id: str) -> Optional[str]:
        """Get optimal formation for task based on learning.

        Args:
            task: Task description
            team_id: Team identifier

        Returns:
            Optimal formation or None
        """
        # Get experiences for similar tasks
        experiences = [
            exp
            for eid in self._team_experiences.get(team_id, [])
            if (exp := self._experiences.get(eid)) is not None
        ]

        if not experiences:
            return None

        # Group by formation and compute average reward
        formation_rewards = defaultdict(list)
        for exp in experiences:
            formation = exp.team_config.formation.value
            formation_rewards[formation].append(exp.reward)

        # Return best formation
        avg_rewards = {
            formation: np.mean(rewards) for formation, rewards in formation_rewards.items()
        }

        if avg_rewards:
            return max(avg_rewards.keys(), key=lambda k: avg_rewards[k])

        return None

    def save_data(self, path: Path) -> None:
        """Save learning data to file.

        Args:
            path: Path to save data
        """
        try:
            data = {
                "experiences": {eid: exp.to_dict() for eid, exp in self._experiences.items()},
                "team_experiences": dict(self._team_experiences),
                "q_table": {f"{k[0]}_{k[1]}": v for k, v in self._q_table.items()},
                "formation_performance": dict(self._formation_performance),
                "member_performance": dict(self._member_performance),
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved learning data to {path}")
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")

    def load_data(self, path: Path) -> None:
        """Load learning data from file.

        Args:
            path: Path to load data from
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Load Q-table
            q_table = {}
            for key_str, value in data.get("q_table", {}).items():
                parts = key_str.split("_")
                team_id = "_".join(parts[:-1])
                state = parts[-1]
                q_table[(team_id, state)] = value
            self._q_table = q_table

            logger.info(f"Loaded learning data from {path}")
        except Exception as e:
            logger.error(f"Failed to load learning data: {e}")


__all__ = [
    "LearningStrategy",
    "TeamExperience",
    "AdaptationRecommendation",
    "LearningProgress",
    "TeamLearningSystem",
]
