# Protocol Reference - Part 4

**Part 4 of 4:** Utility, Tool Access Control, and Budget Management Protocols

---

## Navigation

- [Part 1: Factory, Provider, Tool](part-1-factory-provider-tool.md)
- [Part 2: Coordinator & Conversation](part-2-coordinator-conversation.md)
- [Part 3: Streaming, Observability, Recovery](part-3-streaming-observability-recovery.md)
- **[Part 4: Utility, Access, Budget](#)** (Current)
- [**Complete Reference**](../PROTOCOL_REFERENCE.md)

---

### RecoveryHandlerProtocol

Model failure recovery.

```python
@runtime_checkable
class RecoveryHandlerProtocol(Protocol):
    def detect_failure(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        mentioned_tools: Optional[List[str]],
        elapsed_time: float,
        session_idle_timeout: float,
        quality_score: float,
        consecutive_failures: int,
        recent_responses: Optional[List[str]],
        context_utilization: Optional[float],
    ) -> Optional[Any]:
        """Detect failure type from response characteristics.

        Returns:
            FailureType if failure detected, None otherwise
        """
        ...

    async def recover(
        self,
        failure_type: Any,
        provider: str,
        model: str,
        content: str,
        tool_calls_made: int,
        tool_budget: int,
        iteration_count: int,
        max_iterations: int,
        elapsed_time: float,
        session_idle_timeout: float,
        current_temperature: float,
        consecutive_failures: int,
        mentioned_tools: Optional[List[str]],
        recent_responses: Optional[List[str]],
        quality_score: float,
        task_type: str,
        is_analysis_task: bool,
        is_action_task: bool,
        session_id: Optional[str],
    ) -> Any:
        """Attempt recovery using appropriate strategy.

        Returns:
            RecoveryOutcome with action to take
        """
        ...

    def record_outcome(
        self,
        success: bool,
        quality_improvement: float
    ) -> None:
        """Record recovery outcome for Q-learning."""
        ...

    def reset_session(self, session_id: str) -> None:
        """Reset recovery state for a new session."""
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about recovery system."""
        ...
```

---

## Utility Protocols

### ToolCacheProtocol

Tool result caching.

```python
@runtime_checkable
class ToolCacheProtocol(Protocol):
    def get(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached result for a tool call."""
        ...

    def set(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any
    ) -> None:
        """Cache a tool result."""
        ...

    def invalidate(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> None:
        """Invalidate a cached result."""
        ...
```

---

### TaskTrackerProtocol

Task tracking.

```python
@runtime_checkable
class TaskTrackerProtocol(Protocol):
    def start_task(self, task_id: str, description: str) -> None:
        """Start tracking a task."""
        ...

    def complete_task(self, task_id: str) -> None:
        """Mark task as complete."""
        ...

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all active tasks."""
        ...

    def is_loop_detected(self) -> bool:
        """Check if execution loop is detected."""
        ...
```

---

### ToolOutputFormatterProtocol

Tool output formatting.

```python
@runtime_checkable
class ToolOutputFormatterProtocol(Protocol):
    def format(
        self,
        tool_name: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format tool output for LLM consumption.

        Args:
            tool_name: Name of the tool
            result: Raw tool result
            context: Optional formatting context

        Returns:
            Formatted output string
        """
        ...
```

---

### ArgumentNormalizerProtocol

Argument normalization.

```python
@runtime_checkable
class ArgumentNormalizerProtocol(Protocol):
    def normalize(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize tool arguments.

        Handles malformed arguments, type coercion, etc.
        """
        ...
```

---

## Tool Access Control Protocols

### IToolAccessController

Unified tool access control.

```python
@runtime_checkable
class IToolAccessController(Protocol):
    """Protocol for unified tool access control.

    The controller applies layers in precedence order:
    Safety (L0) > Mode (L1) > Session (L2) > Vertical (L3) > Stage (L4) > Intent (L5)

    A tool is blocked if ANY layer denies it.
    """

    def check_access(
        self,
        tool_name: str,
        context: Optional[ToolAccessContext] = None
    ) -> ToolAccessDecision:
        """Check if a tool is allowed in the given context.

        Args:
            tool_name: Name of the tool to check
            context: Access context (mode, stage, intent, etc.)

        Returns:
            ToolAccessDecision with result and explanation
        """
        ...

    def filter_tools(
        self,
        tools: List[str],
        context: Optional[ToolAccessContext] = None
    ) -> Tuple[List[str], List[ToolAccessDecision]]:
        """Filter a list of tools to only allowed ones.

        Args:
            tools: List of tool names to filter
            context: Access context

        Returns:
            Tuple of (allowed_tools, denial_decisions)
        """
        ...

    def get_allowed_tools(
        self,
        context: Optional[ToolAccessContext] = None
    ) -> Set[str]:
        """Get all tools allowed in the given context.

        Args:
            context: Access context

        Returns:
            Set of allowed tool names
        """
        ...

    def explain_decision(
        self,
        tool_name: str,
        context: Optional[ToolAccessContext] = None
    ) -> str:
        """Get detailed explanation for a tool access decision.

        Args:
            tool_name: Name of the tool
            context: Access context

        Returns:
            Human-readable explanation
        """
        ...
```

---

## Budget Management Protocols

### IBudgetManager

Unified budget management.

```python
@runtime_checkable
class IBudgetManager(Protocol):
    """Protocol for unified budget management.

    Centralizes all budget tracking with consistent multiplier composition:
    effective_max = base × model_multiplier × mode_multiplier × productivity_multiplier
    """

    def get_status(self, budget_type: BudgetType) -> BudgetStatus:
        """Get current status of a budget.

        Args:
            budget_type: Type of budget to check

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 3 min
**Last Updated:** February 08, 2026**
