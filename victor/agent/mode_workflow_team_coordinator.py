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
    VerticalCoordinationAdvisor,
    create_vertical_coordination_advisor,
)
from victor.protocols.coordination import (
    CoordinationSuggestion,
    ModeWorkflowTeamCoordinatorProtocol,
)

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
        advisor: Optional[ModeWorkflowTeamCoordinatorProtocol] = None,
    ):
        self._advisor = advisor or VerticalCoordinationAdvisor(
            vertical_context=vertical_context,
            team_selector=team_selector,
            workflow_selector=workflow_selector,
            mode_configs=mode_configs,
        )

        logger.debug(
            "ModeWorkflowTeamCoordinator initialized: advisor=%s",
            type(self._advisor).__name__,
        )

    def set_vertical_context(self, context: "VerticalContext") -> None:
        """Set the vertical context."""
        self._advisor.set_vertical_context(context)
        logger.debug("Coordinator vertical context set: %s", context.vertical_name)

    def set_team_learner(self, learner: "TeamCompositionLearner") -> None:
        """Set the team composition learner for learning-aware selectors."""
        self._advisor.set_team_learner(learner)

    def suggest_for_task(
        self,
        task_type: str,
        complexity: str,
        mode: str,
    ) -> CoordinationSuggestion:
        """Suggest teams and workflows for a task."""
        suggestion = self._advisor.suggest_for_task(task_type, complexity, mode)

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
        return self._advisor.get_default_workflow(mode)

    def get_suggested_teams(
        self,
        task_type: str,
        complexity: str,
    ) -> List[Any]:
        """Get team suggestions for a task."""
        return self._advisor.get_suggested_teams(task_type, complexity)

    def get_action_for_complexity(
        self,
        complexity: str,
        mode: str,
    ) -> Any:
        """Determine action based on complexity and mode."""
        return self._advisor.get_action_for_complexity(complexity, mode)

    def _get_workflow_recommendations(
        self,
        task_type: str,
        mode: str,
    ) -> List[Any]:
        """Get workflow recommendations for a task."""
        return self._advisor.get_workflow_recommendations(task_type, mode)


def create_coordinator(
    vertical_context: Optional["VerticalContext"] = None,
    team_learner: Optional["TeamCompositionLearner"] = None,
    selection_strategy: str = "hybrid",
) -> ModeWorkflowTeamCoordinator:
    """Create a ModeWorkflowTeamCoordinator."""
    advisor = create_vertical_coordination_advisor(
        vertical_context=vertical_context,
        team_learner=team_learner,
        selection_strategy=selection_strategy,
    )
    coordinator = ModeWorkflowTeamCoordinator(advisor=advisor)
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
