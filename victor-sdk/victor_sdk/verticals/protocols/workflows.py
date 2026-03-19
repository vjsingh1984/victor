"""Workflow-related protocol definitions.

These protocols define how verticals provide workflow and handler configurations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, Any, Optional

from victor_sdk.core.types import StageDefinition


@runtime_checkable
class WorkflowProvider(Protocol):
    """Protocol for providing workflow configurations.

    Workflow providers define multi-stage workflows that agents can
    execute to accomplish complex tasks.
    """

    def get_workflow_spec(self) -> Dict[str, Any]:
        """Return workflow specification.

        Returns:
            Dictionary representing the workflow structure
        """
        ...

    def get_stage_definitions(self) -> Dict[str, StageDefinition]:
        """Return stage definitions for this workflow.

        Returns:
            Dictionary mapping stage names to StageDefinition objects
        """
        ...

    def get_initial_stage(self) -> str:
        """Return the name of the initial stage.

        Returns:
            Name of the starting stage
        """
        ...


@runtime_checkable
class HandlerProvider(Protocol):
    """Protocol for providing handler configurations.

    Handlers are functions that can be called at specific points
    in a workflow or in response to events.
    """

    def get_handlers(self) -> Dict[str, Any]:
        """Return handler functions.

        Returns:
            Dictionary mapping handler names to handler functions
        """
        ...

    def get_handler_for_event(self, event_type: str) -> Optional[Any]:
        """Get handler for a specific event type.

        Args:
            event_type: Type of event (e.g., "tool_call", "error")

        Returns:
            Handler function or None if not found
        """
        ...

    def get_pre_execution_handlers(self) -> List[Any]:
        """Return handlers to run before tool execution.

        Returns:
            List of handler functions
        """
        ...

    def get_post_execution_handlers(self) -> List[Any]:
        """Return handlers to run after tool execution.

        Returns:
            List of handler functions
        """
        ...
