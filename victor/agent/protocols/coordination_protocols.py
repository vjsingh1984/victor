"""Protocol definitions for coordination protocols."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    TYPE_CHECKING,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.conversation_state import ConversationStage


__all__ = [
    "ToolPlannerProtocol",
    "TaskCoordinatorProtocol",
    "ToolCoordinatorProtocol",
    "StateCoordinatorProtocol",
    "PromptCoordinatorProtocol",
    "UnifiedMemoryCoordinatorProtocol",
]

@runtime_checkable
class ToolPlannerProtocol(Protocol):
    """Protocol for tool planning and intent-based filtering.

    Centralizes all tool planning operations, including:
    - Tool sequence planning using dependency graph
    - Goal inference from user messages
    - Intent-based tool filtering

    Extracted from CRITICAL-001 Phase 2C.
    """

    def plan_tools(
        self, goals: List[str], available_inputs: Optional[List[str]] = None
    ) -> List[Any]:
        """Plan a sequence of tools to satisfy goals.

        Args:
            goals: List of desired outputs
            available_inputs: Optional list of inputs already available

        Returns:
            List of ToolDefinition objects for the planned sequence
        """
        ...

    def infer_goals_from_message(self, user_message: str) -> List[str]:
        """Infer planning goals from user request.

        Args:
            user_message: The user's input message

        Returns:
            List of inferred goal outputs
        """
        ...

    def filter_tools_by_intent(
        self, tools: List[Any], current_intent: Optional[Any] = None
    ) -> List[Any]:
        """Filter tools based on detected user intent.

        Args:
            tools: List of tool definitions
            current_intent: The detected user intent (if None, no filtering)

        Returns:
            Filtered list of tools
        """
        ...


@runtime_checkable
class TaskCoordinatorProtocol(Protocol):
    """Protocol for task coordination and guidance.

    Centralizes all task coordination operations, including:
    - Task preparation with complexity detection
    - Intent-based prompt guards
    - Task-specific guidance and budget adjustments

    Extracted from CRITICAL-001 Phase 2D.
    """

    def prepare_task(
        self, user_message: str, unified_task_type: Any, conversation_controller: Any
    ) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments.

        Args:
            user_message: The user's input message
            unified_task_type: Unified task type classification
            conversation_controller: Conversation controller for message injection

        Returns:
            Tuple of (task_classification, complexity_tool_budget)
        """
        ...

    def apply_intent_guard(self, user_message: str, conversation_controller: Any) -> None:
        """Detect intent and inject prompt guards for read-only tasks.

        Args:
            user_message: The user's input message
            conversation_controller: Conversation controller for message injection
        """
        ...

    def apply_task_guidance(
        self,
        user_message: str,
        unified_task_type: Any,
        is_analysis_task: bool,
        is_action_task: bool,
        needs_execution: bool,
        max_exploration_iterations: int,
        conversation_controller: Any,
    ) -> None:
        """Apply guidance and budget tweaks for analysis/action tasks.

        Args:
            user_message: The user's input message
            unified_task_type: Unified task type classification
            is_analysis_task: Whether this is an analysis task
            is_action_task: Whether this is an action-oriented task
            needs_execution: Whether the task requires execution
            max_exploration_iterations: Maximum exploration iterations allowed
            conversation_controller: Conversation controller for message injection
        """
        ...

    @property
    def current_intent(self) -> Any:
        """Get the current detected intent."""
        ...

    @property
    def temperature(self) -> float:
        """Get the current temperature setting."""
        ...

    @property
    def tool_budget(self) -> int:
        """Get the current tool budget."""
        ...

    @property
    def observed_files(self) -> list:
        """Get the list of observed files."""
        ...


@runtime_checkable
class ToolCoordinatorProtocol(Protocol):
    """Protocol for tool coordination operations.

    Coordinates tool selection, budgeting, and execution through a unified
    interface. Consolidates tool-related operations from AgentOrchestrator.
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
class StateCoordinatorProtocol(Protocol):
    """Protocol for state coordination operations.

    Coordinates conversation state and stage transitions through a unified
    interface. Consolidates state management from AgentOrchestrator.
    """

    def get_current_stage(self) -> Any:
        """Get the current conversation stage.

        Returns:
            Current ConversationStage
        """
        ...

    def transition_to(
        self,
        stage: Any,
        reason: str = "",
        tool_name: Optional[str] = None,
    ) -> bool:
        """Transition to a new conversation stage.

        Args:
            stage: Target stage to transition to
            reason: Reason for the transition
            tool_name: Tool that triggered the transition

        Returns:
            True if transition was successful
        """
        ...

    def get_message_history(self) -> List[Any]:
        """Get the full message history.

        Returns:
            List of Message objects
        """
        ...

    def get_recent_messages(
        self,
        limit: int = 10,
        include_system: bool = False,
    ) -> List[Any]:
        """Get recent messages from history.

        Args:
            limit: Maximum messages to return
            include_system: Whether to include system messages

        Returns:
            List of recent Message objects
        """
        ...

    def is_in_exploration_phase(self) -> bool:
        """Check if currently in exploration phase.

        Returns:
            True if in exploration phase
        """
        ...

    def is_in_execution_phase(self) -> bool:
        """Check if currently in execution phase.

        Returns:
            True if in execution phase
        """
        ...


@runtime_checkable
class PromptCoordinatorProtocol(Protocol):
    """Protocol for prompt coordination operations.

    Coordinates system prompt assembly through a unified interface.
    Consolidates prompt building from AgentOrchestrator.
    """

    def build_system_prompt(
        self,
        context: Any,
        include_hints: bool = True,
    ) -> str:
        """Build the complete system prompt.

        Args:
            context: TaskContext for prompt building
            include_hints: Whether to include task hints

        Returns:
            Complete system prompt string
        """
        ...

    def add_task_hint(self, task_type: str, hint: str) -> None:
        """Add or update a task-type hint.

        Args:
            task_type: Task type (e.g., "edit", "debug")
            hint: Hint text for this task type
        """
        ...

    def get_task_hint(self, task_type: str) -> Optional[str]:
        """Get the hint for a task type.

        Args:
            task_type: Task type to get hint for

        Returns:
            Hint string or None
        """
        ...

    def add_section(
        self,
        name: str,
        content: str,
        priority: Optional[int] = None,
    ) -> None:
        """Add a runtime section to be included in prompts.

        Args:
            name: Section name (unique identifier)
            content: Section content
            priority: Optional priority
        """
        ...

    def set_grounding_mode(self, mode: str) -> None:
        """Set the grounding rules mode.

        Args:
            mode: "minimal" or "extended"
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

