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

"""Mode-Workflow-Team Coordinator for intelligent task coordination.

This module provides the main coordination component that bridges:
1. Agent modes (explore/plan/build)
2. Team specifications (from verticals)
3. Workflow definitions (from YAML)
4. Task analysis results (from TaskAnalyzer)

The coordinator uses a strategy pattern for team and workflow selection,
enabling pluggable selection algorithms (rule-based, learning-based, hybrid).

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │              ModeWorkflowTeamCoordinator                         │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌───────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
    │  │ TeamSelection │  │ WorkflowSelection│  │ ConfigProvider  │   │
    │  │   Strategy    │  │    Strategy      │  │                 │   │
    │  └───────────────┘  └──────────────────┘  └─────────────────┘   │
    │           │                  │                     │             │
    │           ▼                  ▼                     ▼             │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │              VerticalContext (team specs, workflows)      │  │
    │  └───────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from victor.agent.mode_workflow_team_coordinator import (
        ModeWorkflowTeamCoordinator,
        create_coordinator,
    )

    # Create with default strategies
    coordinator = create_coordinator(vertical_context)

    # Get suggestions for a task
    suggestion = coordinator.suggest_for_task(
        task_type="feature",
        complexity="high",
        mode="build",
    )

    if suggestion.should_spawn_team:
        team = suggestion.primary_team
        print(f"Auto-spawning: {team.team_name}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from victor.protocols.coordination import (
    CoordinationSuggestion,
    ComplexityLevel,
    ModeWorkflowTeamCoordinatorProtocol,
    TeamRecommendation,
    TeamSelectionStrategyProtocol,
    TeamSuggestionAction,
    WorkflowRecommendation,
    WorkflowSelectionStrategyProtocol,
)

if TYPE_CHECKING:
    from victor.agent.vertical_context import VerticalContext
    from victor.agent.teams.learner import TeamCompositionLearner

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Data Classes
# =============================================================================


@dataclass
class ModeCoordinationConfig:
    """Configuration for a specific mode.

    Attributes:
        mode_name: Mode identifier
        default_workflows: Workflow names to suggest for this mode
        default_teams: Team names to suggest for this mode
        team_suggestion_enabled: Whether team suggestions are enabled
        complexity_thresholds: Map complexity levels to actions
        tool_priorities: Tool priority adjustments for this mode
        system_prompt_addition: Additional prompt content
    """

    mode_name: str
    default_workflows: List[str] = field(default_factory=list)
    default_teams: List[str] = field(default_factory=list)
    team_suggestion_enabled: bool = True
    complexity_thresholds: Dict[str, TeamSuggestionAction] = field(
        default_factory=lambda: {
            "trivial": TeamSuggestionAction.NONE,
            "low": TeamSuggestionAction.NONE,
            "medium": TeamSuggestionAction.SUGGEST,
            "high": TeamSuggestionAction.SUGGEST,
            "extreme": TeamSuggestionAction.AUTO_SPAWN,
        }
    )
    tool_priorities: Dict[str, float] = field(default_factory=dict)
    system_prompt_addition: str = ""


# Default configurations for each mode
DEFAULT_MODE_CONFIGS: Dict[str, ModeCoordinationConfig] = {
    "explore": ModeCoordinationConfig(
        mode_name="explore",
        default_workflows=[],  # No workflows in explore mode
        default_teams=["research_team"],
        team_suggestion_enabled=False,  # Explore is for understanding, not teams
        complexity_thresholds={
            "trivial": TeamSuggestionAction.NONE,
            "low": TeamSuggestionAction.NONE,
            "medium": TeamSuggestionAction.NONE,
            "high": TeamSuggestionAction.SUGGEST,
            "extreme": TeamSuggestionAction.SUGGEST,
        },
    ),
    "plan": ModeCoordinationConfig(
        mode_name="plan",
        default_workflows=["planning_workflow"],
        default_teams=["research_team", "analysis_team"],
        team_suggestion_enabled=True,
        complexity_thresholds={
            "trivial": TeamSuggestionAction.NONE,
            "low": TeamSuggestionAction.NONE,
            "medium": TeamSuggestionAction.SUGGEST,
            "high": TeamSuggestionAction.SUGGEST,
            "extreme": TeamSuggestionAction.SUGGEST,
        },
    ),
    "build": ModeCoordinationConfig(
        mode_name="build",
        default_workflows=["feature_implementation", "bug_fix"],
        default_teams=["feature_team", "bug_fix_team", "refactoring_team"],
        team_suggestion_enabled=True,
        complexity_thresholds={
            "trivial": TeamSuggestionAction.NONE,
            "low": TeamSuggestionAction.NONE,
            "medium": TeamSuggestionAction.SUGGEST,
            "high": TeamSuggestionAction.AUTO_SPAWN,
            "extreme": TeamSuggestionAction.AUTO_SPAWN,
        },
    ),
}


# =============================================================================
# Team Selection Strategies
# =============================================================================


class RuleBasedTeamSelector:
    """Rule-based team selection using keyword matching.

    Matches task types to team names using configurable patterns.
    Fast and predictable, good for common task types.
    """

    # Task type to team name mapping patterns
    TASK_TEAM_PATTERNS: Dict[str, List[str]] = {
        "feature": ["feature_team", "implementation_team"],
        "bugfix": ["bug_fix_team", "debugging_team"],
        "bug": ["bug_fix_team", "debugging_team"],
        "refactor": ["refactoring_team", "code_review_team"],
        "review": ["review_team", "code_review_team"],
        "test": ["testing_team", "qa_team"],
        "documentation": ["documentation_team", "docs_team"],
        "research": ["research_team", "deep_research_team"],
        "analysis": ["analysis_team", "eda_team"],
        "deployment": ["deployment_team", "devops_team"],
        "container": ["container_team", "docker_team"],
        "monitoring": ["monitoring_team", "observability_team"],
        "visualization": ["visualization_team", "charting_team"],
        "ml": ["ml_team", "machine_learning_team"],
        "data": ["data_team", "cleaning_team", "eda_team"],
    }

    def select(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
    ) -> Optional[str]:
        """Select a team based on task type matching."""
        task_lower = task_type.lower()

        # Find matching patterns
        for pattern, team_names in self.TASK_TEAM_PATTERNS.items():
            if pattern in task_lower:
                # Check if any matching team is available
                for team_name in team_names:
                    if team_name in available_teams:
                        return team_name

        # Fallback: look for team name containing task type
        for team_name in available_teams:
            if task_lower in team_name.lower():
                return team_name

        return None

    def recommend(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
        top_k: int = 3,
    ) -> List[TeamRecommendation]:
        """Recommend teams with confidence scores."""
        recommendations = []
        task_lower = task_type.lower()

        for team_name, spec in available_teams.items():
            confidence = 0.0
            reason = ""

            # Check pattern matches
            for pattern, team_names in self.TASK_TEAM_PATTERNS.items():
                if pattern in task_lower and team_name in team_names:
                    confidence = max(confidence, 0.8)
                    reason = f"Task type '{task_type}' matches pattern '{pattern}'"
                    break

            # Check name similarity
            if task_lower in team_name.lower():
                confidence = max(confidence, 0.6)
                reason = reason or "Team name contains task type"

            # Check spec description if available
            if hasattr(spec, "description") and task_lower in spec.description.lower():
                confidence = max(confidence, 0.5)
                reason = reason or "Team description matches task"

            if confidence > 0:
                recommendations.append(
                    TeamRecommendation(
                        team_name=team_name,
                        confidence=confidence,
                        reason=reason,
                        formation=getattr(spec, "formation", None),
                        source="rule",
                    )
                )

        # Sort by confidence and return top_k
        recommendations.sort(reverse=True)
        return recommendations[:top_k]


class LearningBasedTeamSelector:
    """Learning-based team selection using TeamCompositionLearner.

    Uses Q-learning recommendations from historical executions
    to suggest optimal team compositions.
    """

    def __init__(self, learner: Optional["TeamCompositionLearner"] = None):
        """Initialize with optional learner.

        Args:
            learner: TeamCompositionLearner instance
        """
        self._learner = learner

    def set_learner(self, learner: "TeamCompositionLearner") -> None:
        """Set the learner instance.

        Args:
            learner: TeamCompositionLearner to use
        """
        self._learner = learner

    def select(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
    ) -> Optional[str]:
        """Select team using learner recommendations."""
        if self._learner is None:
            return None

        try:
            recommendation = self._learner.get_recommendation(task_type)
            if recommendation and not recommendation.is_baseline:
                # Map formation to team name
                return self._find_team_for_recommendation(recommendation, available_teams)
        except Exception as e:
            logger.debug(f"Learner selection failed: {e}")

        return None

    def recommend(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
        top_k: int = 3,
    ) -> List[TeamRecommendation]:
        """Recommend teams using learner."""
        if self._learner is None:
            return []

        try:
            recommendation = self._learner.get_recommendation(task_type)
            if recommendation:
                return [
                    TeamRecommendation(
                        team_name=self._find_team_for_recommendation(
                            recommendation, available_teams
                        )
                        or "feature_team",
                        confidence=recommendation.confidence,
                        reason=recommendation.reason,
                        formation=(
                            recommendation.formation.value if recommendation.formation else None
                        ),
                        suggested_budget=recommendation.suggested_budget,
                        role_distribution=recommendation.role_distribution,
                        source="learning",
                    )
                ]
        except Exception as e:
            logger.debug(f"Learner recommendation failed: {e}")

        return []

    def _find_team_for_recommendation(
        self,
        recommendation: Any,
        available_teams: Dict[str, Any],
    ) -> Optional[str]:
        """Find team matching learner recommendation.

        Args:
            recommendation: TeamRecommendation from learner
            available_teams: Available team specs

        Returns:
            Matching team name or None
        """
        # Try to match by formation if available
        if hasattr(recommendation, "formation") and recommendation.formation:
            formation_name = (
                recommendation.formation.value
                if hasattr(recommendation.formation, "value")
                else str(recommendation.formation)
            )
            for team_name, spec in available_teams.items():
                if hasattr(spec, "formation"):
                    spec_formation = (
                        spec.formation.value
                        if hasattr(spec.formation, "value")
                        else str(spec.formation)
                    )
                    if spec_formation == formation_name:
                        return team_name

        # Return first available team
        if available_teams:
            return next(iter(available_teams))

        return None


class HybridTeamSelector:
    """Hybrid team selection combining rules and learning.

    Blends rule-based matching with learning-based recommendations
    using configurable weights.
    """

    def __init__(
        self,
        rule_selector: RuleBasedTeamSelector,
        learning_selector: LearningBasedTeamSelector,
        rule_weight: float = 0.4,
        learning_weight: float = 0.6,
    ):
        """Initialize hybrid selector.

        Args:
            rule_selector: Rule-based selector
            learning_selector: Learning-based selector
            rule_weight: Weight for rule-based recommendations
            learning_weight: Weight for learning-based recommendations
        """
        self._rule_selector = rule_selector
        self._learning_selector = learning_selector
        self._rule_weight = rule_weight
        self._learning_weight = learning_weight

    def select(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
    ) -> Optional[str]:
        """Select team using hybrid approach."""
        # Try learning first (if learner has data)
        learning_pick = self._learning_selector.select(task_type, complexity, available_teams)
        if learning_pick:
            return learning_pick

        # Fall back to rules
        return self._rule_selector.select(task_type, complexity, available_teams)

    def recommend(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
        top_k: int = 3,
    ) -> List[TeamRecommendation]:
        """Recommend teams using hybrid approach."""
        # Get recommendations from both
        rule_recs = self._rule_selector.recommend(task_type, complexity, available_teams, top_k)
        learning_recs = self._learning_selector.recommend(
            task_type, complexity, available_teams, top_k
        )

        # Blend recommendations
        combined: Dict[str, TeamRecommendation] = {}

        for rec in rule_recs:
            combined[rec.team_name] = TeamRecommendation(
                team_name=rec.team_name,
                confidence=rec.confidence * self._rule_weight,
                reason=rec.reason,
                formation=rec.formation,
                source="hybrid-rule",
            )

        for rec in learning_recs:
            if rec.team_name in combined:
                # Blend confidences
                existing = combined[rec.team_name]
                combined[rec.team_name] = TeamRecommendation(
                    team_name=rec.team_name,
                    confidence=(existing.confidence + rec.confidence * self._learning_weight),
                    reason=f"{existing.reason}; {rec.reason}",
                    formation=rec.formation or existing.formation,
                    suggested_budget=rec.suggested_budget,
                    role_distribution=rec.role_distribution,
                    source="hybrid",
                )
            else:
                combined[rec.team_name] = TeamRecommendation(
                    team_name=rec.team_name,
                    confidence=rec.confidence * self._learning_weight,
                    reason=rec.reason,
                    formation=rec.formation,
                    suggested_budget=rec.suggested_budget,
                    role_distribution=rec.role_distribution,
                    source="hybrid-learning",
                )

        # Sort and return top_k
        result = sorted(combined.values(), reverse=True)
        return result[:top_k]


# =============================================================================
# Workflow Selection Strategy
# =============================================================================


class RuleBasedWorkflowSelector:
    """Rule-based workflow selection.

    Matches task types and modes to workflow names.
    """

    # Task type to workflow mapping
    TASK_WORKFLOW_PATTERNS: Dict[str, List[str]] = {
        "feature": ["feature_implementation", "quick_feature"],
        "bugfix": ["bug_fix", "quick_fix"],
        "bug": ["bug_fix", "quick_fix"],
        "refactor": ["refactoring", "code_review"],
        "review": ["code_review", "pr_review", "quick_review"],
        "test": ["testing", "test_coverage"],
    }

    def select(
        self,
        task_type: str,
        mode: str,
        available_workflows: Dict[str, Any],
    ) -> Optional[str]:
        """Select workflow based on task type and mode."""
        task_lower = task_type.lower()

        # Find matching patterns
        for pattern, workflow_names in self.TASK_WORKFLOW_PATTERNS.items():
            if pattern in task_lower:
                for name in workflow_names:
                    if name in available_workflows:
                        return name

        return None

    def recommend(
        self,
        task_type: str,
        mode: str,
        available_workflows: Dict[str, Any],
        top_k: int = 3,
    ) -> List[WorkflowRecommendation]:
        """Recommend workflows with confidence."""
        recommendations = []
        task_lower = task_type.lower()

        for name, workflow in available_workflows.items():
            confidence = 0.0
            reason = ""

            # Check pattern matches
            for pattern, workflow_names in self.TASK_WORKFLOW_PATTERNS.items():
                if pattern in task_lower and name in workflow_names:
                    confidence = max(confidence, 0.8)
                    reason = f"Task type '{task_type}' matches workflow pattern"
                    break

            # Check name similarity
            if task_lower in name.lower():
                confidence = max(confidence, 0.6)
                reason = reason or "Workflow name matches task"

            if confidence > 0:
                recommendations.append(
                    WorkflowRecommendation(
                        workflow_name=name,
                        confidence=confidence,
                        reason=reason,
                    )
                )

        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        return recommendations[:top_k]


# =============================================================================
# Main Coordinator
# =============================================================================


class ModeWorkflowTeamCoordinator(ModeWorkflowTeamCoordinatorProtocol):
    """Main coordinator for mode-workflow-team integration.

    Coordinates between:
    1. Agent modes (explore/plan/build)
    2. Team specifications (from VerticalContext)
    3. Workflow definitions (from VerticalContext)
    4. Task analysis results

    Uses pluggable strategies for team and workflow selection.

    Example:
        coordinator = ModeWorkflowTeamCoordinator(vertical_context)

        suggestion = coordinator.suggest_for_task(
            task_type="feature",
            complexity="high",
            mode="build",
        )

        if suggestion.should_spawn_team:
            print(f"Spawning: {suggestion.primary_team.team_name}")
    """

    def __init__(
        self,
        vertical_context: Optional["VerticalContext"] = None,
        team_selector: Optional[Any] = None,
        workflow_selector: Optional[Any] = None,
        mode_configs: Optional[Dict[str, ModeCoordinationConfig]] = None,
    ):
        """Initialize coordinator.

        Args:
            vertical_context: Context with team specs and workflows
            team_selector: Strategy for team selection
            workflow_selector: Strategy for workflow selection
            mode_configs: Mode-specific configurations
        """
        self._vertical_context = vertical_context
        self._team_selector = team_selector or HybridTeamSelector(
            RuleBasedTeamSelector(),
            LearningBasedTeamSelector(),
        )
        self._workflow_selector = workflow_selector or RuleBasedWorkflowSelector()
        self._mode_configs = mode_configs or DEFAULT_MODE_CONFIGS.copy()

        logger.debug(
            f"ModeWorkflowTeamCoordinator initialized: "
            f"team_selector={type(self._team_selector).__name__}, "
            f"workflow_selector={type(self._workflow_selector).__name__}"
        )

    def set_vertical_context(self, context: "VerticalContext") -> None:
        """Set the vertical context.

        Args:
            context: VerticalContext with team specs and workflows
        """
        self._vertical_context = context
        logger.debug(f"Coordinator vertical context set: {context.vertical_name}")

    def set_team_learner(self, learner: "TeamCompositionLearner") -> None:
        """Set the team composition learner.

        Args:
            learner: Learner for learning-based selection
        """
        if isinstance(self._team_selector, HybridTeamSelector):
            self._team_selector._learning_selector.set_learner(learner)
        elif isinstance(self._team_selector, LearningBasedTeamSelector):
            self._team_selector.set_learner(learner)

    def suggest_for_task(
        self,
        task_type: str,
        complexity: str,
        mode: str,
    ) -> CoordinationSuggestion:
        """Suggest teams and workflows for a task.

        Args:
            task_type: Classified task type
            complexity: Complexity level string
            mode: Current agent mode

        Returns:
            CoordinationSuggestion with recommendations
        """
        complexity_level = ComplexityLevel.from_string(complexity)
        mode_config = self._mode_configs.get(mode, self._mode_configs["build"])

        # Determine action based on complexity and mode
        action = self.get_action_for_complexity(complexity, mode)

        # Get team recommendations if enabled
        team_recommendations: List[TeamRecommendation] = []
        if mode_config.team_suggestion_enabled:
            team_recommendations = self.get_suggested_teams(task_type, complexity)

        # Get workflow recommendations
        workflow_recommendations = self._get_workflow_recommendations(task_type, mode)

        suggestion = CoordinationSuggestion(
            mode=mode,
            task_type=task_type,
            complexity=complexity_level,
            action=action,
            team_recommendations=team_recommendations,
            workflow_recommendations=workflow_recommendations,
            system_prompt_additions=mode_config.system_prompt_addition,
            tool_priorities=mode_config.tool_priorities,
            metadata={
                "mode_config": mode_config.mode_name,
                "team_suggestion_enabled": mode_config.team_suggestion_enabled,
            },
        )

        logger.info(
            f"Coordination suggestion: mode={mode}, task={task_type}, "
            f"complexity={complexity}, action={action.value}, "
            f"teams={len(team_recommendations)}, workflows={len(workflow_recommendations)}"
        )

        return suggestion

    def get_default_workflow(self, mode: str) -> Optional[str]:
        """Get the default workflow for a mode.

        Args:
            mode: Agent mode name

        Returns:
            Default workflow name or None
        """
        mode_config = self._mode_configs.get(mode)
        if mode_config and mode_config.default_workflows:
            # Check if any default workflow exists in context
            if self._vertical_context:
                for name in mode_config.default_workflows:
                    if self._vertical_context.get_workflow(name):
                        return name
            return mode_config.default_workflows[0]
        return None

    def get_suggested_teams(
        self,
        task_type: str,
        complexity: str,
    ) -> List[TeamRecommendation]:
        """Get team suggestions for a task.

        Args:
            task_type: Classified task type
            complexity: Complexity level string

        Returns:
            List of team recommendations
        """
        available_teams = self._get_available_teams()
        if not available_teams:
            return []

        return self._team_selector.recommend(
            task_type=task_type,
            complexity=complexity,
            available_teams=available_teams,
            top_k=3,
        )

    def get_action_for_complexity(
        self,
        complexity: str,
        mode: str,
    ) -> TeamSuggestionAction:
        """Determine action based on complexity and mode.

        Args:
            complexity: Complexity level string
            mode: Agent mode name

        Returns:
            TeamSuggestionAction to take
        """
        mode_config = self._mode_configs.get(mode, self._mode_configs["build"])
        complexity_lower = complexity.lower()

        # Check explicit thresholds
        if complexity_lower in mode_config.complexity_thresholds:
            return mode_config.complexity_thresholds[complexity_lower]

        # Default mapping
        complexity_level = ComplexityLevel.from_string(complexity)
        default_actions = {
            ComplexityLevel.TRIVIAL: TeamSuggestionAction.NONE,
            ComplexityLevel.LOW: TeamSuggestionAction.NONE,
            ComplexityLevel.MEDIUM: TeamSuggestionAction.SUGGEST,
            ComplexityLevel.HIGH: TeamSuggestionAction.SUGGEST,
            ComplexityLevel.EXTREME: TeamSuggestionAction.AUTO_SPAWN,
        }

        return default_actions.get(complexity_level, TeamSuggestionAction.NONE)

    def _get_available_teams(self) -> Dict[str, Any]:
        """Get available team specs from context.

        Returns:
            Dict mapping team names to specs
        """
        if self._vertical_context:
            return self._vertical_context.team_specs or {}
        return {}

    def _get_available_workflows(self) -> Dict[str, Any]:
        """Get available workflows from context.

        Returns:
            Dict mapping workflow names to definitions
        """
        if self._vertical_context:
            return self._vertical_context.workflows or {}
        return {}

    def _get_workflow_recommendations(
        self,
        task_type: str,
        mode: str,
    ) -> List[WorkflowRecommendation]:
        """Get workflow recommendations.

        Args:
            task_type: Classified task type
            mode: Current agent mode

        Returns:
            List of workflow recommendations
        """
        available_workflows = self._get_available_workflows()
        if not available_workflows:
            return []

        return self._workflow_selector.recommend(
            task_type=task_type,
            mode=mode,
            available_workflows=available_workflows,
            top_k=3,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_coordinator(
    vertical_context: Optional["VerticalContext"] = None,
    team_learner: Optional["TeamCompositionLearner"] = None,
    selection_strategy: str = "hybrid",
) -> ModeWorkflowTeamCoordinator:
    """Create a ModeWorkflowTeamCoordinator.

    Args:
        vertical_context: Context with team specs and workflows
        team_learner: Optional learner for learning-based selection
        selection_strategy: Strategy type ("rule", "learning", "hybrid")

    Returns:
        Configured coordinator
    """
    # Create team selector based on strategy
    team_selector: TeamSelectionStrategyProtocol
    if selection_strategy == "rule":
        team_selector = RuleBasedTeamSelector()
    elif selection_strategy == "learning":
        team_selector = LearningBasedTeamSelector(team_learner)
    else:  # hybrid
        learning_selector = LearningBasedTeamSelector(team_learner)
        team_selector = HybridTeamSelector(
            RuleBasedTeamSelector(),
            learning_selector,
        )

    coordinator = ModeWorkflowTeamCoordinator(
        vertical_context=vertical_context,
        team_selector=team_selector,
        workflow_selector=RuleBasedWorkflowSelector(),
    )

    return coordinator


__all__ = [
    # Main coordinator
    "ModeWorkflowTeamCoordinator",
    "ModeCoordinationConfig",
    # Strategies
    "RuleBasedTeamSelector",
    "LearningBasedTeamSelector",
    "HybridTeamSelector",
    "RuleBasedWorkflowSelector",
    # Factory
    "create_coordinator",
    # Configuration
    "DEFAULT_MODE_CONFIGS",
]
