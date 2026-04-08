"""Workflow compiler protocol definitions.

These protocols define how verticals interact with the workflow engine.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional

from victor_sdk.core.types import StageDefinition


@runtime_checkable
class WorkflowCompiler(Protocol):
    """Protocol for workflow compilation and execution.

    Workflow compilers transform workflow specifications into
    executable state graphs.
    """

    def compile(self, workflow_spec: Dict[str, Any]) -> Any:
        """Compile a workflow specification into an executable graph.

        Args:
            workflow_spec: Workflow specification dictionary

        Returns:
            Compiled workflow graph
        """
        ...

    def validate(self, workflow_spec: Dict[str, Any]) -> List[str]:
        """Validate a workflow specification.

        Args:
            workflow_spec: Workflow specification to validate

        Returns:
            List of validation errors (empty if valid)
        """
        ...

    def get_stage_definitions(
        self, workflow_spec: Dict[str, Any]
    ) -> Dict[str, StageDefinition]:
        """Extract stage definitions from workflow spec.

        Args:
            workflow_spec: Workflow specification

        Returns:
            Dictionary of stage definitions
        """
        ...

    def get_entry_point(self, workflow_spec: Dict[str, Any]) -> Optional[str]:
        """Get the entry point stage for a workflow.

        Args:
            workflow_spec: Workflow specification

        Returns:
            Entry point stage name or None
        """
        ...
