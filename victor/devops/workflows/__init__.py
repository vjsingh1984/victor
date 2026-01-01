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

Uses YAML-first architecture with Python escape hatches for complex conditions
and transforms that cannot be expressed in YAML.

Example:
    provider = DevOpsWorkflowProvider()

    # Standard execution
    executor = provider.create_executor(orchestrator)
    result = await executor.execute(workflow, context)

    # Streaming execution
    async for chunk in provider.astream("deploy", orchestrator, context):
        if chunk.event_type == WorkflowEventType.NODE_COMPLETE:
            print(f"Completed: {chunk.node_name}")

Available workflows (all YAML-defined):
- deploy: Safe deployment with validation and rollback
- cicd: CI/CD pipeline with security scanning
- container_setup: Container setup with Dockerfile optimization
- container_quick: Quick container build
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple

from victor.core.verticals.protocols import WorkflowProviderProtocol
from victor.workflows.definition import WorkflowDefinition

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator
    from victor.workflows.executor import WorkflowExecutor
    from victor.workflows.streaming import WorkflowStreamChunk
    from victor.workflows.streaming_executor import StreamingWorkflowExecutor
    from victor.workflows.yaml_loader import YAMLWorkflowConfig


class DevOpsWorkflowProvider(WorkflowProviderProtocol):
    """Provides DevOps-specific workflows.

    Uses YAML-first architecture with Python escape hatches for complex
    conditions and transforms that cannot be expressed in YAML.

    Includes support for streaming execution via StreamingWorkflowExecutor
    for real-time progress updates during long-running DevOps workflows.

    Example:
        provider = DevOpsWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Stream deployment execution
        async for chunk in provider.astream("deploy", orchestrator, {}):
            print(f"[{chunk.progress:.0f}%] {chunk.event_type.value}")
    """

    def __init__(self) -> None:
        self._workflows: Optional[Dict[str, WorkflowDefinition]] = None
        self._config: Optional["YAMLWorkflowConfig"] = None

    def _get_config(self) -> "YAMLWorkflowConfig":
        """Get YAML workflow config with escape hatches registered.

        Returns:
            YAMLWorkflowConfig with DevOps conditions and transforms
        """
        if self._config is None:
            from victor.workflows.yaml_loader import YAMLWorkflowConfig
            from victor.devops.escape_hatches import CONDITIONS, TRANSFORMS

            self._config = YAMLWorkflowConfig(
                base_dir=Path(__file__).parent,
                condition_registry=CONDITIONS,
                transform_registry=TRANSFORMS,
            )
        return self._config

    def _load_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Lazy load all YAML workflows.

        Uses escape hatches for complex conditions that can't be expressed in YAML.

        Returns:
            Dict mapping workflow names to definitions
        """
        if self._workflows is None:
            try:
                from victor.workflows.yaml_loader import load_workflows_from_directory

                # Load from the workflows directory with escape hatches
                workflows_dir = Path(__file__).parent
                config = self._get_config()
                self._workflows = load_workflows_from_directory(
                    workflows_dir,
                    pattern="*.yaml",
                    config=config,
                )
                logger.debug(
                    f"Loaded {len(self._workflows)} YAML workflows from {workflows_dir}"
                )
            except Exception as e:
                logger.warning(f"Failed to load YAML workflows: {e}")
                self._workflows = {}

        return self._workflows

    def get_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Get workflow definitions for this vertical."""
        return self._load_workflows()

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

    def create_executor(
        self,
        orchestrator: "AgentOrchestrator",
    ) -> "WorkflowExecutor":
        """Create a standard workflow executor.

        Args:
            orchestrator: Agent orchestrator instance

        Returns:
            WorkflowExecutor for running workflows
        """
        from victor.workflows.executor import WorkflowExecutor

        return WorkflowExecutor(orchestrator)

    def create_streaming_executor(
        self,
        orchestrator: "AgentOrchestrator",
    ) -> "StreamingWorkflowExecutor":
        """Create a streaming workflow executor.

        Args:
            orchestrator: Agent orchestrator instance

        Returns:
            StreamingWorkflowExecutor for real-time progress streaming
        """
        from victor.workflows.streaming_executor import StreamingWorkflowExecutor

        return StreamingWorkflowExecutor(orchestrator)

    async def astream(
        self,
        workflow_name: str,
        orchestrator: "AgentOrchestrator",
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator["WorkflowStreamChunk"]:
        """Stream workflow execution with real-time events.

        Convenience method that creates a streaming executor and
        streams the specified workflow.

        Args:
            workflow_name: Name of the workflow to execute
            orchestrator: Agent orchestrator instance
            context: Initial context data for the workflow

        Yields:
            WorkflowStreamChunk events during execution

        Raises:
            ValueError: If workflow_name is not found

        Example:
            provider = DevOpsWorkflowProvider()
            async for chunk in provider.astream("container_setup", orchestrator, {}):
                if chunk.event_type == WorkflowEventType.NODE_START:
                    print(f"Starting: {chunk.node_name}")
        """
        workflow = self.get_workflow(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        executor = self.create_streaming_executor(orchestrator)
        async for chunk in executor.astream(workflow, context or {}):
            yield chunk

    def __repr__(self) -> str:
        return f"DevOpsWorkflowProvider(workflows={len(self._load_workflows())})"


# Register DevOps domain handlers when this module is loaded
from victor.devops.handlers import register_handlers as _register_handlers

_register_handlers()

__all__ = [
    # YAML-first workflow provider
    "DevOpsWorkflowProvider",
]
