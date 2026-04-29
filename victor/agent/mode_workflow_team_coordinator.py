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

"""Agent-facing coordinator wrapper over shared framework recommendation logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.framework.coordination_runtime import (
    DEFAULT_MODE_CONFIGS,
    HybridTeamSelector,
    LearningBasedTeamSelector,
    ModeCoordinationConfig,
    RuleBasedTeamSelector,
    RuleBasedWorkflowSelector,
    build_coordination_suggestion,
    get_action_for_complexity,
    recommend_teams_for_catalog,
    recommend_workflows_for_catalog,
    resolve_default_workflow_for_mode,
)
from victor.protocols.coordination import CoordinationSuggestion, ModeWorkflowTeamCoordinatorProtocol

if TYPE_CHECKING:
    from victor.agent.teams.learner import TeamCompositionLearner
    from victor.agent.vertical_context import VerticalContext

logger = logging.getLogger(__name__)


class ModeWorkflowTeamCoordinator(ModeWorkflowTeamCoordinatorProtocol):
    """Agent wrapper that delegates recommendation logic to framework helpers."""

    def __init__(
        self,
        vertical_context: Optional["VerticalContext"] = None,
        team_selector: Optional[Any] = None,
        workflow_selector: Optional[Any] = None,
        mode_configs: Optional[Dict[str, ModeCoordinationConfig]] = None,
    ):
        self._vertical_context = vertical_context
        self._team_selector = team_selector or HybridTeamSelector(
            RuleBasedTeamSelector(),
            LearningBasedTeamSelector(),
        )
        self._workflow_selector = workflow_selector or RuleBasedWorkflowSelector()
        self._mode_configs = mode_configs or DEFAULT_MODE_CONFIGS.copy()

        logger.debug(
            "ModeWorkflowTeamCoordinator initialized: team_selector=%s, workflow_selector=%s",
            type(self._team_selector).__name__,
            type(self._workflow_selector).__name__,
        )

    def set_vertical_context(self, context: "VerticalContext") -> None:
        """Set the vertical context."""
        self._vertical_context = context
        logger.debug("Coordinator vertical context set: %s", context.vertical_name)

    def set_team_learner(self, learner: "TeamCompositionLearner") -> None:
        """Set the team composition learner for learning-aware selectors."""
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
        """Suggest teams and workflows for a task."""
        suggestion = build_coordination_suggestion(
            task_type=task_type,
            complexity=complexity,
            mode=mode,
            coordination_catalog=self._get_coordination_catalog(),
            team_selector=self._team_selector,
            workflow_selector=self._workflow_selector,
            mode_configs=self._mode_configs,
        )

        logger.info(
            "Coordination suggestion: mode=%s, task=%s, complexity=%s, action=%s, "
            "teams=%s, workflows=%s",
            mode,
            task_type,
            complexity,
            suggestion.action.value,
            len(suggestion.team_recommendations),
            len(suggestion.workflow_recommendations),
        )
        return suggestion

    def get_default_workflow(self, mode: str) -> Optional[str]:
        """Get the default workflow for a mode."""
        return resolve_default_workflow_for_mode(
            mode,
            coordination_catalog=self._get_coordination_catalog(),
            mode_configs=self._mode_configs,
        )

    def get_suggested_teams(
        self,
        task_type: str,
        complexity: str,
    ) -> List[Any]:
        """Get team suggestions for a task."""
        return recommend_teams_for_catalog(
            task_type=task_type,
            complexity=complexity,
            coordination_catalog=self._get_coordination_catalog(),
            team_selector=self._team_selector,
        )

    def get_action_for_complexity(
        self,
        complexity: str,
        mode: str,
    ) -> Any:
        """Determine action based on complexity and mode."""
        return get_action_for_complexity(
            complexity,
            mode,
            mode_configs=self._mode_configs,
        )

    def _get_coordination_catalog(self) -> Any:
        """Resolve the shared team/workflow catalog for the current vertical context."""
        from victor.framework.team_runtime import resolve_vertical_coordination_catalog

        return resolve_vertical_coordination_catalog(self._vertical_context)

    def _get_workflow_recommendations(
        self,
        task_type: str,
        mode: str,
    ) -> List[Any]:
        """Get workflow recommendations for a task."""
        return recommend_workflows_for_catalog(
            task_type=task_type,
            mode=mode,
            coordination_catalog=self._get_coordination_catalog(),
            workflow_selector=self._workflow_selector,
        )


def create_coordinator(
    vertical_context: Optional["VerticalContext"] = None,
    team_learner: Optional["TeamCompositionLearner"] = None,
    selection_strategy: str = "hybrid",
) -> ModeWorkflowTeamCoordinator:
    """Create a ModeWorkflowTeamCoordinator."""
    if selection_strategy == "rule":
        team_selector = RuleBasedTeamSelector()
    elif selection_strategy == "learning":
        team_selector = LearningBasedTeamSelector(team_learner)
    else:
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
    "DEFAULT_MODE_CONFIGS",
    "HybridTeamSelector",
    "LearningBasedTeamSelector",
    "ModeCoordinationConfig",
    "ModeWorkflowTeamCoordinator",
    "RuleBasedTeamSelector",
    "RuleBasedWorkflowSelector",
    "create_coordinator",
]
