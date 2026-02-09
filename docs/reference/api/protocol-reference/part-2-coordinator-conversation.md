# Protocol Reference - Part 2

**Part 2 of 4:** Coordinator and Conversation Protocols

---

## Navigation

- [Part 1: Factory, Provider, Tool](part-1-factory-provider-tool.md)
- **[Part 2: Coordinator & Conversation](#)** (Current)
- [Part 3: Streaming, Observability, Recovery](part-3-streaming-observability-recovery.md)
- [Part 4: Utility, Access, Budget](part-4-utility-access-budget.md)
- [**Complete Reference**](../PROTOCOL_REFERENCE.md)

---

### IToolSelectionCoordinator

Intelligent tool selection and routing.

**Purpose:** Extracts ~650 lines of tool selection logic from orchestrator following SRP.

```python
@runtime_checkable
class IToolSelectionCoordinator(Protocol):
    def get_recommended_search_tool(
        self,
        query: str,
        context: Optional["AgentToolSelectionContext"] = None,
    ) -> Optional[str]:
        """Get recommended search tool for a query.

        Analyzes the query to determine which search tool (semantic, grep,
        code_search, etc.) would be most appropriate.

        Args:
            query: Search query string
            context: Optional selection context (stage, task type, history)

        Returns:
            Recommended tool name or None if no recommendation
        """
        ...

    def route_search_query(
        self,
        query: str,
        available_tools: Set[str],
    ) -> str:
        """Route a search query to the appropriate tool.

        Determines the best search tool based on query characteristics
        and available tools.

        Args:
            query: Search query string
            available_tools: Set of available tool names

        Returns:
            Selected tool name
        """
        ...

    def detect_mentioned_tools(
        self,
        prompt: str,
        available_tools: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Detect tools mentioned in a prompt.

        Scans the prompt for explicit tool mentions (e.g., "use grep to
        find...", "run the web_search tool").

        Args:
            prompt: Prompt text to scan
            available_tools: Optional set of available tools (defaults to all)

        Returns:
            Set of detected tool names
        """
        ...

    def classify_task_keywords(
        self,
        task: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Classify task type using keyword analysis.

        Determines if a task is primarily analysis, action, or creation
        based on keyword presence.

        Args:
            task: Task description
            conversation_history: Optional conversation history for context

        Returns:
            Task type: "analysis", "action", or "creation"
        """
        ...

    def classify_task_with_context(
        self,
        task: str,
        context: Optional["AgentToolSelectionContext"] = None,
    ) -> str:
        """Classify task type with full context.

        Enhanced task classification using conversation stage, recent tools,
        and other context information.

        Args:
            task: Task description
            context: Selection context with stage, history, recent tools

        Returns:
            Task type: "analysis", "action", or "creation"
        """
        ...

    def should_use_tools(
        self,
        message: str,
        model_supports_tools: bool = True,
    ) -> bool:
        """Determine if tools should be used for a message.

        Analyzes the message to determine if tool use is appropriate.

        Args:
            message: User message
            model_supports_tools: Whether the model supports tool calling

        Returns:
            True if tools should be used
        """
        ...

    def extract_required_files(
        self,
        prompt: str,
    ) -> Set[str]:
        """Extract required files from a prompt.

        Parses the prompt to find file paths that are explicitly mentioned
        or implied as dependencies.

        Args:
            prompt: Prompt text to parse

        Returns:
            Set of required file paths
        """
        ...

    def extract_required_outputs(
        self,
        prompt: str,
    ) -> Set[str]:
        """Extract required outputs from a prompt.

        Parses the prompt to find output specifications (file paths,
        variable names, etc.) that the task should produce.

        Args:
            prompt: Prompt text to parse

        Returns:
            Set of required output identifiers
        """
        ...
```text

---

### ToolCoordinatorProtocol

Tool coordination operations (WS-D refactoring).

```python
@runtime_checkable
class ToolCoordinatorProtocol(Protocol):
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
```

---

### StateCoordinatorProtocol

State coordination operations.

```python
@runtime_checkable
class StateCoordinatorProtocol(Protocol):
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
```text

---

### PromptCoordinatorProtocol

Prompt coordination operations.

```python
@runtime_checkable
class PromptCoordinatorProtocol(Protocol):
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
```


**Reading Time:** 4 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


