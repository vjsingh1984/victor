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

"""DevOps vertical workflows.

This package provides workflow definitions for common DevOps tasks:
- Infrastructure deployment
- Container management
- CI/CD pipeline setup
- Monitoring configuration
"""

from typing import Dict, List, Optional, Tuple, Type

from victor.verticals.protocols import WorkflowProviderProtocol
from victor.workflows.definition import (
    WorkflowBuilder,
    WorkflowDefinition,
    workflow,
)


@workflow("deploy_infrastructure", "Infrastructure deployment with validation")
def deploy_infrastructure_workflow() -> WorkflowDefinition:
    """Create infrastructure deployment workflow."""
    return (
        WorkflowBuilder("deploy_infrastructure")
        .set_metadata("category", "devops")
        .set_metadata("complexity", "high")
        # Assess current state
        .add_agent(
            "assess",
            role="researcher",
            goal="Assess current infrastructure state and requirements",
            tool_budget=20,
            allowed_tools=["read_file", "bash", "grep", "git_status"],
            output_key="assessment",
        )
        # Plan deployment
        .add_agent(
            "plan",
            role="planner",
            goal="Plan infrastructure deployment strategy",
            tool_budget=15,
            allowed_tools=["read_file", "grep", "web_search"],
            input_mapping={"current_state": "assessment"},
            output_key="deployment_plan",
        )
        # Implement infrastructure
        .add_agent(
            "implement",
            role="executor",
            goal="Create or update infrastructure configurations",
            tool_budget=35,
            allowed_tools=["read_file", "write_file", "edit_files", "bash"],
            input_mapping={"plan": "deployment_plan"},
            output_key="implementation",
        )
        # Validate configurations
        .add_agent(
            "validate",
            role="reviewer",
            goal="Validate configurations and run infrastructure tests",
            tool_budget=20,
            allowed_tools=["bash", "read_file", "docker"],
            output_key="validation_result",
        )
        # Deploy
        .add_agent(
            "deploy",
            role="executor",
            goal="Deploy infrastructure changes",
            tool_budget=15,
            allowed_tools=["bash", "docker", "git_status", "git_diff"],
            next_nodes=[],
        )
        .build()
    )


@workflow("container_setup", "Docker container setup and configuration")
def container_setup_workflow() -> WorkflowDefinition:
    """Create container setup workflow."""
    return (
        WorkflowBuilder("container_setup")
        .set_metadata("category", "devops")
        .set_metadata("complexity", "medium")
        # Analyze requirements
        .add_agent(
            "analyze",
            role="researcher",
            goal="Analyze application requirements for containerization",
            tool_budget=15,
            allowed_tools=["read_file", "grep", "bash"],
            output_key="requirements",
        )
        # Create Dockerfile and configs
        .add_agent(
            "configure",
            role="executor",
            goal="Create Dockerfile and container configurations",
            tool_budget=25,
            allowed_tools=["read_file", "write_file", "edit_files"],
            input_mapping={"reqs": "requirements"},
            output_key="configs",
        )
        # Build and test
        .add_agent(
            "build",
            role="executor",
            goal="Build and test container",
            tool_budget=20,
            allowed_tools=["bash", "docker"],
            next_nodes=[],
        )
        .build()
    )


@workflow("cicd_pipeline", "CI/CD pipeline configuration")
def cicd_pipeline_workflow() -> WorkflowDefinition:
    """Create CI/CD pipeline workflow."""
    return (
        WorkflowBuilder("cicd_pipeline")
        .set_metadata("category", "devops")
        .set_metadata("complexity", "high")
        # Research existing setup
        .add_agent(
            "research",
            role="researcher",
            goal="Analyze existing CI/CD setup and requirements",
            tool_budget=20,
            allowed_tools=["read_file", "grep", "git_log", "bash"],
            output_key="analysis",
        )
        # Design pipeline
        .add_agent(
            "design",
            role="planner",
            goal="Design CI/CD pipeline stages and configurations",
            tool_budget=10,
            allowed_tools=["read_file", "web_search"],
            input_mapping={"setup": "analysis"},
            output_key="design",
        )
        # Implement pipeline
        .add_agent(
            "implement",
            role="executor",
            goal="Create CI/CD configuration files",
            tool_budget=30,
            allowed_tools=["read_file", "write_file", "edit_files", "bash"],
            input_mapping={"plan": "design"},
            output_key="pipeline",
        )
        # Test pipeline
        .add_agent(
            "test",
            role="reviewer",
            goal="Validate pipeline configuration",
            tool_budget=15,
            allowed_tools=["bash", "read_file"],
            next_nodes=[],
        )
        .build()
    )


class DevOpsWorkflowProvider(WorkflowProviderProtocol):
    """Provides DevOps-specific workflows."""

    def __init__(self) -> None:
        self._workflows: Optional[Dict[str, WorkflowDefinition]] = None

    def _load_workflows(self) -> Dict[str, WorkflowDefinition]:
        if self._workflows is None:
            self._workflows = {
                "deploy_infrastructure": deploy_infrastructure_workflow(),
                "container_setup": container_setup_workflow(),
                "cicd_pipeline": cicd_pipeline_workflow(),
            }
        return self._workflows

    def get_workflows(self) -> Dict[str, Type]:
        return self._load_workflows()  # type: ignore

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        return self._load_workflows().get(name)

    def get_workflow_names(self) -> List[str]:
        return list(self._load_workflows().keys())

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        return [
            (r"deploy\s+infrastructure", "deploy_infrastructure"),
            (r"terraform\s+apply", "deploy_infrastructure"),
            (r"container(ize)?", "container_setup"),
            (r"docker(file)?", "container_setup"),
            (r"ci/?cd", "cicd_pipeline"),
            (r"pipeline", "cicd_pipeline"),
            (r"github\s+actions", "cicd_pipeline"),
        ]

    def __repr__(self) -> str:
        return f"DevOpsWorkflowProvider(workflows={len(self._load_workflows())})"


__all__ = [
    "DevOpsWorkflowProvider",
    "deploy_infrastructure_workflow",
    "container_setup_workflow",
    "cicd_pipeline_workflow",
]
