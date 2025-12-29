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

"""Teams integration for DevOps vertical."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from victor.framework.teams import TeamFormation, TeamMemberSpec


@dataclass
class DevOpsTeamSpec:
    """Specification for a DevOps team."""

    name: str
    description: str
    formation: TeamFormation
    members: List[TeamMemberSpec]
    total_tool_budget: int = 100
    max_iterations: int = 50


DEVOPS_TEAM_SPECS: Dict[str, DevOpsTeamSpec] = {
    "deployment_team": DevOpsTeamSpec(
        name="Deployment Team",
        description="Infrastructure deployment with validation",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Assess current infrastructure state",
                name="Infrastructure Assessor",
                tool_budget=20,
            ),
            TeamMemberSpec(
                role="planner",
                goal="Plan deployment strategy",
                name="Deployment Planner",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Implement infrastructure changes",
                name="Infrastructure Engineer",
                tool_budget=35,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Validate and deploy changes",
                name="Deployment Validator",
                tool_budget=25,
            ),
        ],
        total_tool_budget=95,
    ),
    "container_team": DevOpsTeamSpec(
        name="Container Team",
        description="Docker container setup and management",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze containerization requirements",
                name="Container Analyst",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Create container configurations",
                name="Container Engineer",
                tool_budget=30,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Build and test containers",
                name="Container Tester",
                tool_budget=20,
            ),
        ],
        total_tool_budget=65,
    ),
    "monitoring_team": DevOpsTeamSpec(
        name="Monitoring Team",
        description="Observability and monitoring setup",
        formation=TeamFormation.SEQUENTIAL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze monitoring requirements",
                name="Monitoring Analyst",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Configure monitoring stack",
                name="Monitoring Engineer",
                tool_budget=30,
            ),
        ],
        total_tool_budget=45,
    ),
}


def get_team_for_task(task_type: str) -> Optional[DevOpsTeamSpec]:
    """Get appropriate team for task type."""
    mapping = {
        "deploy": "deployment_team",
        "deployment": "deployment_team",
        "infrastructure": "deployment_team",
        "container": "container_team",
        "docker": "container_team",
        "containerization": "container_team",
        "monitor": "monitoring_team",
        "monitoring": "monitoring_team",
        "observability": "monitoring_team",
    }
    spec_name = mapping.get(task_type.lower())
    if spec_name:
        return DEVOPS_TEAM_SPECS.get(spec_name)
    return None


def list_team_types() -> List[str]:
    return list(DEVOPS_TEAM_SPECS.keys())


__all__ = [
    "DevOpsTeamSpec",
    "DEVOPS_TEAM_SPECS",
    "get_team_for_task",
    "list_team_types",
]
