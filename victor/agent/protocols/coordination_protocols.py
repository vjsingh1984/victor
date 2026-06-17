"""Protocol definitions for coordination protocols."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from victor.agent.services.protocols.runtime_support import (
    CoordinationAdvisorRuntimeProtocol,
    PromptRuntimeProtocol as PromptCoordinatorProtocol,
    StateRuntimeProtocol as StateCoordinatorProtocol,
    TaskRuntimeProtocol as TaskCoordinatorProtocol,
    ToolPlanningRuntimeProtocol as ToolPlannerProtocol,
)

__all__ = [
    "CoordinationAdvisorRuntimeProtocol",
    "ToolPlannerProtocol",
    "TaskCoordinatorProtocol",
    "ToolCoordinatorProtocol",
    "StateCoordinatorProtocol",
    "PromptCoordinatorProtocol",
    "UnifiedMemoryCoordinatorProtocol",
]


@runtime_checkable
class ToolCoordinatorProtocol(Protocol):
    """[DEPRECATED] Compatibility protocol for legacy tool coordination.

    This protocol remains importable for backward compatibility, but the
    canonical runtime path should depend on ToolService instead.
    """

    async def select_tools(self, context: Any) -> List[Any]:
        """Select appropriate tools for the current context.

        Args:
            context: TaskContext with message, task_type, etc.

        Returns:
            List of selected tool definitions
        """
        ...

    def get_remaining_budget(self) -> int:
        """Get remaining tool call budget.

        Returns:
            Number of tool calls remaining
        """
        ...

    def consume_budget(self, amount: int = 1) -> None:
        """Consume tool call budget.

        Args:
            amount: Number of budget units to consume
        """
        ...

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Any] = None,
    ) -> Any:
        """Execute tool calls through the pipeline.

        Args:
            tool_calls: List of tool calls to execute
            context: Optional task context

        Returns:
            PipelineExecutionResult with execution details
        """
        ...

    def reset_budget(self, new_budget: Optional[int] = None) -> None:
        """Reset the tool budget.

        Args:
            new_budget: New budget to set, or use default
        """
        ...

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted.

        Returns:
            True if no budget remaining
        """
        ...


@runtime_checkable
class UnifiedMemoryCoordinatorProtocol(Protocol):
    """Protocol for unified memory coordinator.

    Provides federated search across multiple memory backends (entity,
    conversation, graph, embeddings) with pluggable ranking strategies.
    """

    async def search_all(
        self,
        query: str,
        limit: int = 20,
        memory_types: Optional[List[Any]] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_relevance: float = 0.0,
    ) -> List[Any]:
        """Search across all registered memory providers.

        Args:
            query: Search query string
            limit: Maximum results to return
            memory_types: Optional filter for specific memory types
            session_id: Optional session ID for context
            filters: Additional provider-specific filters
            min_relevance: Minimum relevance threshold

        Returns:
            Ranked list of memory results from all providers
        """
        ...

    async def search_type(
        self,
        memory_type: Any,
        query: str,
        limit: int = 20,
        **kwargs: Any,
    ) -> List[Any]:
        """Search a specific memory type.

        Args:
            memory_type: Type of memory to search
            query: Search query string
            limit: Maximum results to return
            **kwargs: Additional search parameters

        Returns:
            List of memory results from the specified type
        """
        ...

    async def store(
        self,
        memory_type: Any,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a value in a specific memory type.

        Args:
            memory_type: Type of memory to store in
            key: Storage key
            value: Value to store
            metadata: Optional metadata

        Returns:
            True if stored successfully
        """
        ...

    async def get(
        self,
        memory_type: Any,
        key: str,
    ) -> Optional[Any]:
        """Get a value from a specific memory type.

        Args:
            memory_type: Type of memory to retrieve from
            key: Storage key

        Returns:
            Memory result or None if not found
        """
        ...

    def register_provider(self, provider: Any) -> None:
        """Register a memory provider.

        Args:
            provider: Provider implementing MemoryProviderProtocol
        """
        ...

    def unregister_provider(self, memory_type: Any) -> bool:
        """Unregister a memory provider.

        Args:
            memory_type: Type of memory provider to remove

        Returns:
            True if provider was removed
        """
        ...

    def get_registered_types(self) -> List[Any]:
        """Get list of registered memory types.

        Returns:
            List of registered MemoryType values
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics.

        Returns:
            Dictionary with query counts, errors, registered providers
        """
        ...
