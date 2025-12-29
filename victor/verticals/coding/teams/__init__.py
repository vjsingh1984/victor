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
    from victor.verticals.coding.teams import (
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
"""

from victor.verticals.coding.teams.specs import (
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

__all__ = [
    # Types
    "CodingRoleConfig",
    "CodingTeamSpec",
    # Role configurations
    "CODING_ROLES",
    # Team specifications
    "CODING_TEAM_SPECS",
    # Helper functions
    "get_team_for_task",
    "get_role_config",
    "list_team_types",
    "list_roles",
]
