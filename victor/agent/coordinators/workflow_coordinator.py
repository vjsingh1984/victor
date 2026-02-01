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

"""Workflow coordinator for agent orchestration.

This module provides the WorkflowCoordinator which handles workflow
registration, discovery, and execution for the orchestrator.

Extracted from AgentOrchestrator as part of SOLID refactoring
to improve modularity and testability.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.workflows.registry import WorkflowRegistry

logger = logging.getLogger(__name__)


class WorkflowCoordinator:
    """Coordinates workflow registration and execution for the orchestrator.

    This coordinator wraps the WorkflowRegistry and provides a clean
    interface for:
    - Workflow registration and discovery
    - Workflow execution via WorkflowEngine
    - Listing available workflows

    The coordinator uses callback functions for workflow operations to
    maintain loose coupling with the orchestrator.

    Example:
        coordinator = WorkflowCoordinator(
            workflow_registry=registry,
        )

        # Register default workflows
        count = coordinator.register_default_workflows()

        # List available workflows
        workflows = coordinator.list_workflows()
    """

    def __init__(
        self,
        workflow_registry: "WorkflowRegistry",
    ) -> None:
        """Initialize the workflow coordinator.

        Args:
            workflow_registry: The workflow registry instance
        """
        self._workflow_registry = workflow_registry

    @property
    def workflow_registry(self) -> "WorkflowRegistry":
        """Get the underlying workflow registry.

        Returns:
            WorkflowRegistry instance
        """
        return self._workflow_registry

    def register_default_workflows(self) -> int:
        """Register default workflows via dynamic discovery.

        Registers core mode workflows (explore/plan/build) from YAML.

        Returns:
            Number of workflows registered
        """
        from victor.workflows.mode_workflows import get_mode_workflow_provider

        provider = get_mode_workflow_provider()
        workflows = provider.get_workflow_definitions()

        count = 0
        for workflow in workflows.values():
            try:
                self._workflow_registry.register(workflow, replace=True)
                count += 1
            except Exception as e:
                logger.debug(f"Failed to register mode workflow '{workflow.name}': {e}")

        logger.debug(f"Registered {count} mode workflows via WorkflowCoordinator")
        return count

    def list_workflows(self) -> list[str]:
        """List all registered workflow names.

        Returns:
            List of workflow names
        """
        return self._workflow_registry.list_workflows()

    def has_workflow(self, name: str) -> bool:
        """Check if a workflow is registered.

        Args:
            name: Workflow name

        Returns:
            True if workflow exists
        """
        return self._workflow_registry.get(name) is not None

    def get_workflow_count(self) -> int:
        """Get the number of registered workflows.

        Returns:
            Number of workflows
        """
        return len(self._workflow_registry.list_workflows())
