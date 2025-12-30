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

"""Teams integration for coding vertical.

This package provides team specifications for common software development
tasks, including pre-configured teams for feature implementation,
bug fixing, refactoring, code review, and testing.

Example:
    from victor.coding.teams import (
        get_team_for_task,
        CODING_TEAM_SPECS,
    )

    # Get team for a task type
    team_spec = get_team_for_task("feature")
    print(f"Team: {team_spec.name}")
    print(f"Formation: {team_spec.formation}")
    print(f"Members: {len(team_spec.members)}")

    # Or access directly
    feature_team = CODING_TEAM_SPECS["feature_team"]

Teams are auto-registered with the global TeamSpecRegistry on import,
enabling cross-vertical team discovery via:
    from victor.framework.team_registry import get_team_registry
    registry = get_team_registry()
    coding_teams = registry.find_by_vertical("coding")
"""

import logging

from victor.coding.teams.specs import (
    # Types
    CodingRoleConfig,
    CodingTeamSpec,
    # Role configurations
    CODING_ROLES,
    # Team specifications
    CODING_TEAM_SPECS,
    # Helper functions
    get_team_for_task,
    get_role_config,
    list_team_types,
    list_roles,
)

from victor.coding.teams.personas import (
    # Types
    ExpertiseCategory,
    CommunicationStyle,
    DecisionStyle,
    PersonaTraits,
    CodingPersona,
    # Pre-defined personas
    CODING_PERSONAS,
    # Helper functions
    get_persona,
    get_personas_for_role,
    get_persona_by_expertise,
    apply_persona_to_spec,
    list_personas,
)

__all__ = [
    # Types from specs
    "CodingRoleConfig",
    "CodingTeamSpec",
    # Provider
    "CodingTeamSpecProvider",
    # Role configurations
    "CODING_ROLES",
    # Team specifications
    "CODING_TEAM_SPECS",
    # Helper functions from specs
    "get_team_for_task",
    "get_role_config",
    "list_team_types",
    "list_roles",
    # Types from personas
    "ExpertiseCategory",
    "CommunicationStyle",
    "DecisionStyle",
    "PersonaTraits",
    "CodingPersona",
    # Pre-defined personas
    "CODING_PERSONAS",
    # Helper functions from personas
    "get_persona",
    "get_personas_for_role",
    "get_persona_by_expertise",
    "apply_persona_to_spec",
    "list_personas",
]

logger = logging.getLogger(__name__)


from typing import Dict, List, Optional


class CodingTeamSpecProvider:
    """Team specification provider for Coding vertical.

    Implements TeamSpecProviderProtocol interface for consistent
    ISP compliance across all verticals.
    """

    def get_team_specs(self) -> Dict[str, CodingTeamSpec]:
        """Get all Coding team specifications.

        Returns:
            Dictionary mapping team names to CodingTeamSpec instances
        """
        return CODING_TEAM_SPECS

    def get_team_for_task(self, task_type: str) -> Optional[CodingTeamSpec]:
        """Get appropriate team for a task type.

        Args:
            task_type: Type of task

        Returns:
            CodingTeamSpec or None if no matching team
        """
        return get_team_for_task(task_type)

    def list_team_types(self) -> List[str]:
        """List all available team types.

        Returns:
            List of team type names
        """
        return list_team_types()


def _auto_register_teams() -> int:
    """Auto-register coding teams with global registry.

    Returns:
        Number of teams registered.
    """
    try:
        from victor.framework.team_registry import get_team_registry

        registry = get_team_registry()
        count = registry.register_from_vertical("coding", CODING_TEAM_SPECS)
        logger.debug(f"Auto-registered {count} coding teams")
        return count
    except Exception as e:
        logger.warning(f"Failed to auto-register coding teams: {e}")
        return 0


# Auto-register on import
_auto_register_teams()
