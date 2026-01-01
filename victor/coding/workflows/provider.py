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

"""Coding workflow provider.

Implements WorkflowProviderProtocol to provide coding-specific workflows
to the framework, with support for streaming execution via
StreamingWorkflowExecutor.

Supports hybrid loading:
- Python workflows (inline @workflow definitions)
- YAML workflows (external files in workflows/*.yaml)

YAML workflows override Python workflows when names collide,
allowing customization without code changes.

Example:
    provider = CodingWorkflowProvider()

    # Standard execution
    executor = provider.create_executor(orchestrator)
    result = await executor.execute(workflow, context)

    # Streaming execution
    async for chunk in provider.astream("code_review", orchestrator, context):
        if chunk.event_type == WorkflowEventType.NODE_COMPLETE:
            print(f"Completed: {chunk.node_name}")
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple, Type

from victor.core.verticals.protocols import WorkflowProviderProtocol
from victor.workflows.definition import WorkflowDefinition

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator
    from victor.workflows.executor import WorkflowExecutor
    from victor.workflows.streaming import WorkflowStreamChunk
    from victor.workflows.streaming_executor import StreamingWorkflowExecutor


class CodingWorkflowProvider(WorkflowProviderProtocol):
    """Provides coding-specific workflows.

    This provider implements WorkflowProviderProtocol to expose all
    coding workflows to the framework. Workflows can be retrieved
    by name and auto-triggered based on task patterns.

    Example:
        provider = CodingWorkflowProvider()
        workflows = provider.get_workflows()
        feature_wf = provider.get_workflow("feature_implementation")
    """

    def __init__(self) -> None:
        """Initialize the provider."""
        self._workflows: Optional[Dict[str, WorkflowDefinition]] = None

    def _load_python_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Load Python-defined workflows.

        Returns:
            Dict mapping workflow names to definitions
        """
        from victor.coding.workflows.bugfix import (
            bug_fix_workflow,
            quick_fix_workflow,
        )
        from victor.coding.workflows.feature import (
            feature_implementation_workflow,
            quick_feature_workflow,
        )
        from victor.coding.workflows.review import (
            code_review_workflow,
            pr_review_workflow,
            quick_review_workflow,
        )

        return {
            # Feature workflows
            "feature_implementation": feature_implementation_workflow(),
            "quick_feature": quick_feature_workflow(),
            # Bug fix workflows
            "bug_fix": bug_fix_workflow(),
            "quick_fix": quick_fix_workflow(),
            # Review workflows
            "code_review": code_review_workflow(),
            "quick_review": quick_review_workflow(),
            "pr_review": pr_review_workflow(),
        }

    def _load_yaml_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Load YAML-defined workflows from workflows/*.yaml.

        Returns:
            Dict mapping workflow names to definitions
        """
        try:
            from victor.workflows.yaml_loader import load_workflows_from_directory

            # Load from the workflows directory (same as this file)
            workflows_dir = Path(__file__).parent
            yaml_workflows = load_workflows_from_directory(workflows_dir)
            logger.debug(f"Loaded {len(yaml_workflows)} YAML workflows from {workflows_dir}")
            return yaml_workflows
        except Exception as e:
            logger.warning(f"Failed to load YAML workflows: {e}")
            return {}

    def _load_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Lazy load all workflows with hybrid Python/YAML support.

        YAML workflows override Python workflows when names collide,
        allowing external customization without code changes.

        Returns:
            Dict mapping workflow names to definitions
        """
        if self._workflows is None:
            # Start with Python workflows as base
            python_workflows = self._load_python_workflows()

            # Override with YAML workflows (external overrides inline)
            yaml_workflows = self._load_yaml_workflows()

            # Merge: YAML takes precedence
            self._workflows = {**python_workflows, **yaml_workflows}

            logger.debug(
                f"Loaded {len(python_workflows)} Python + {len(yaml_workflows)} YAML workflows "
                f"= {len(self._workflows)} total"
            )
        return self._workflows

    def get_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Get workflow definitions for this vertical.

        Returns:
            Dict mapping workflow names to WorkflowDefinition instances
        """
        return self._load_workflows()

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        """Get a specific workflow by name.

        Args:
            name: Workflow name

        Returns:
            WorkflowDefinition or None if not found
        """
        workflows = self._load_workflows()
        return workflows.get(name)

    def get_workflow_names(self) -> List[str]:
        """Get all available workflow names.

        Returns:
            List of workflow names
        """
        return list(self._load_workflows().keys())

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatically triggered workflows.

        Returns patterns that can trigger workflows automatically
        based on user input.

        Returns:
            List of (pattern, workflow_name) tuples
        """
        return [
            # Feature patterns
            (r"implement\s+.+feature", "feature_implementation"),
            (r"add\s+.+feature", "feature_implementation"),
            (r"create\s+.+feature", "feature_implementation"),
            (r"build\s+new\s+", "feature_implementation"),
            (r"quick(ly)?\s+implement", "quick_feature"),
            (r"simple\s+feature", "quick_feature"),
            # Bug fix patterns
            (r"fix\s+.+bug", "bug_fix"),
            (r"debug\s+", "bug_fix"),
            (r"investigate\s+.+issue", "bug_fix"),
            (r"quick(ly)?\s+fix", "quick_fix"),
            (r"simple\s+fix", "quick_fix"),
            # Review patterns
            (r"review\s+.+code", "code_review"),
            (r"code\s+review", "code_review"),
            (r"comprehensive\s+review", "code_review"),
            (r"quick(ly)?\s+review", "quick_review"),
            (r"review\s+.+pr", "pr_review"),
            (r"review\s+pull\s+request", "pr_review"),
            (r"pr\s+review", "pr_review"),
        ]

    def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
        """Get recommended workflow for a task type.

        Args:
            task_type: Type of task (feature, bugfix, review, etc.)

        Returns:
            Workflow name or None
        """
        task_mapping = {
            "feature": "feature_implementation",
            "implement": "feature_implementation",
            "new_feature": "feature_implementation",
            "simple_feature": "quick_feature",
            "quick_feature": "quick_feature",
            "bug": "bug_fix",
            "bugfix": "bug_fix",
            "debug": "bug_fix",
            "investigation": "bug_fix",
            "simple_bug": "quick_fix",
            "quick_fix": "quick_fix",
            "review": "code_review",
            "code_review": "code_review",
            "security_review": "code_review",
            "quick_review": "quick_review",
            "pr": "pr_review",
            "pull_request": "pr_review",
            "pr_review": "pr_review",
        }
        return task_mapping.get(task_type.lower())

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
            provider = CodingWorkflowProvider()
            async for chunk in provider.astream("code_review", orchestrator, {"files": ["main.py"]}):
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
        return f"CodingWorkflowProvider(workflows={len(self._load_workflows())})"


# Register Coding domain handlers when this module is loaded
from victor.coding.handlers import register_handlers as _register_handlers
_register_handlers()

__all__ = [
    "CodingWorkflowProvider",
]
