# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Workflow provider protocol for ISP compliance.

This protocol defines the minimal interface for workflow providers,
enabling type-safe workflow access without hasattr() checks.
"""

from typing import Protocol, runtime_checkable, Any, Optional
from victor.framework.graph import StateGraph


@runtime_checkable
class WorkflowProviderProtocol(Protocol):
    """Protocol for objects that provide workflows.

    This protocol replaces hasattr() checks for workflow-related methods,
    enabling type-safe workflow access.

    Example:
        ```python
        @runtime_checkable
        class MyWorkflowProvider(WorkflowProviderProtocol, Protocol):
            def get_workflows(self) -> Dict[str, StateGraph[Any]]:
                return self._workflows

            def get_workflow_provider(self) -> Optional[WorkflowProviderProtocol]:
                return self
        ```
    """

    def get_workflows(self) -> dict[str, "StateGraph[Any]"]:
        """Get all available workflows.

        Returns:
            Dictionary mapping workflow names to StateGraph instances
        """
        ...

    def get_workflow_provider(self) -> Optional["WorkflowProviderProtocol"]:
        """Get the workflow provider instance.

        Returns:
            The workflow provider, or None if not available
        """
        ...
