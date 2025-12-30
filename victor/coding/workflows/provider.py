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
to the framework.
"""

from typing import Dict, List, Optional, Tuple, Type

from victor.core.verticals.protocols import WorkflowProviderProtocol
from victor.workflows.definition import WorkflowDefinition


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

    def _load_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Lazy load all workflows.

        Returns:
            Dict mapping workflow names to definitions
        """
        if self._workflows is None:
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

            self._workflows = {
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

    def __repr__(self) -> str:
        return f"CodingWorkflowProvider(workflows={len(self._load_workflows())})"


__all__ = [
    "CodingWorkflowProvider",
]
