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

"""Team coordinator for multi-agent coordination.

This coordinator handles team specifications and suggestions as part of
Track 4 orchestrator extraction (Phase 1).

Responsibilities:
- Manage team specifications for vertical integration
- Provide team formation suggestions based on task characteristics

Thread Safety:
- All public methods are thread-safe
- Team specs stored in orchestrator instance (not shared state)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TeamCoordinator:
    """Coordinator for team specification and suggestion management.

    This coordinator manages team specifications for multi-agent
    coordination and provides team formation suggestions through
    the ModeWorkflowTeamCoordinator.

    Design Principles:
    - Single Responsibility: Only handles team spec and suggestion operations
    - Dependency Injection: All dependencies injected via constructor
    - Thread Safety: All operations are thread-safe

    Attributes:
        orchestrator: Reference to parent orchestrator for delegation
        mode_coordinator: ModeCoordinator for accessing current mode
        mode_workflow_team_coordinator: ModeWorkflowTeamCoordinator for suggestions
    """

    def __init__(
        self,
        orchestrator: Any,
        mode_coordinator: Any,
        mode_workflow_team_coordinator: Any,
    ):
        """Initialize the team coordinator.

        Args:
            orchestrator: Reference to parent orchestrator
            mode_coordinator: ModeCoordinator for accessing current mode
            mode_workflow_team_coordinator: ModeWorkflowTeamCoordinator for suggestions
        """
        self._orchestrator = orchestrator
        self._mode_coordinator = mode_coordinator
        self._mode_workflow_team_coordinator = mode_workflow_team_coordinator

    def get_team_suggestions(self, task_type: str, complexity: str) -> Any:
        """Get team and workflow suggestions for a task.

        Queries the ModeWorkflowTeamCoordinator to get recommendations for
        teams and workflows based on task classification and current mode.

        Args:
            task_type: Classified task type (e.g., "feature", "bugfix", "refactor")
            complexity: Complexity level (e.g., "low", "medium", "high", "extreme")

        Returns:
            CoordinationSuggestion with team and workflow recommendations

        Example:
            >>> coordinator = TeamCoordinator(orchestrator, mode_coordinator, mw_coordinator)
            >>> suggestions = coordinator.get_team_suggestions("feature", "high")
            >>> print(suggestions.recommended_team)
            "parallel_review_team"
        """
        # Get current mode via ModeCoordinator
        current_mode = self._mode_coordinator.current_mode_name

        # Delegate to ModeWorkflowTeamCoordinator
        result = self._mode_workflow_team_coordinator.coordination.suggest_for_task(
            task_type=task_type,
            complexity=complexity,
            mode=current_mode,
        )

        logger.debug(
            f"Got team suggestions: task_type={task_type}, "
            f"complexity={complexity}, mode={current_mode}"
        )

        return result

    def set_team_specs(self, specs: dict[str, Any]) -> None:
        """Store team specifications.

        Provides a clean public interface for setting team specs,
        replacing direct private attribute access.

        Implements VerticalStorageProtocol.set_team_specs().

        Args:
            specs: Dictionary mapping team names to TeamSpec instances

        Example:
            >>> coordinator = TeamCoordinator(orchestrator, mode_coordinator, mw_coordinator)
            >>> specs = {"review_team": TeamSpec(...)}
            >>> coordinator.set_team_specs(specs)
        """
        # Store in orchestrator instance
        self._orchestrator._team_specs = specs

        logger.debug(f"Set team specs: {len(specs)} teams")

    def get_team_specs(self) -> dict[str, Any]:
        """Retrieve team specifications.

        Returns the dictionary of team specs configured by vertical integration.

        Implements VerticalStorageProtocol.get_team_specs().

        Returns:
            Dictionary of team specs, or empty dict if not set

        Example:
            >>> coordinator = TeamCoordinator(orchestrator, mode_coordinator, mw_coordinator)
            >>> specs = coordinator.get_team_specs()
            >>> print(list(specs.keys()))
            ["review_team", "refactor_team"]
        """
        # Get from orchestrator instance
        specs = getattr(self._orchestrator, "_team_specs", {})
        if specs:
            return specs

        # Fall back to vertical context if available
        vertical_context = getattr(self._orchestrator, "vertical_context", None)
        if vertical_context is not None:
            context_specs: dict[str, Any] = getattr(vertical_context, "team_specs", None) or {}
            if context_specs:
                return context_specs

        # Final fallback: global team registry by vertical name
        vertical_name = None
        if vertical_context is not None:
            vertical_name = getattr(vertical_context, "vertical_name", None)
        if vertical_name:
            from victor.framework.team_registry import get_team_registry

            registry_specs = get_team_registry().find_by_vertical(vertical_name)
            # Strip namespace prefix for compatibility with local team names
            if not isinstance(registry_specs, dict):
                registry_specs = {}
            normalized: dict[str, Any] = registry_specs
            for name, spec in registry_specs.items():
                short_name = name.split(":", 1)[-1] if ":" in name else name
                normalized[short_name] = spec
            if normalized:
                return normalized

        return {}
