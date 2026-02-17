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

"""Tool Coordinator - Coordinates tool selection, budgeting, and execution.

This module extracts tool-related coordination logic from AgentOrchestrator,
providing a focused interface for:
- Tool selection (semantic + keyword-based)
- Tool budget management and enforcement
- Tool access control and enable/disable
- Tool execution coordination via ToolPipeline
- Tool alias resolution and shell variant handling
- Tool call parsing and validation

Design Philosophy:
- Single Responsibility: Coordinates all tool-related operations
- Composable: Works with existing ToolPipeline, ToolSelector, etc.
- Observable: Provides budget/execution metrics
- Backward Compatible: Maintains API compatibility with orchestrator

Usage:
    coordinator = ToolCoordinator(
        tool_pipeline=pipeline,
        tool_selector=selector,
        tool_registry=registry,
        budget_manager=budget_mgr,
    )

    # Select tools for current context
    tools = await coordinator.select_tools(context)

    # Check budget
    remaining = coordinator.get_remaining_budget()

    # Execute tool calls
    result = await coordinator.execute_tool_calls(tool_calls)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.tool_pipeline import ToolPipeline, PipelineExecutionResult
    from victor.agent.tool_selection import ToolSelector
    from victor.agent.budget_manager import BudgetManager
    from victor.agent.tool_executor import ToolExecutionResult
    from victor.storage.cache.tool_cache import ToolCache
    from victor.tools.base import BaseTool, ToolRegistry
    from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
    from victor.agent.tool_calling import ToolCallingAdapter, ToolCallParseResult
    from victor.agent.protocols import ToolAccessContext
    from victor.tools.tool_names import ToolNames

logger = logging.getLogger(__name__)


@dataclass
class TaskContext:
    """Context for tool selection and execution.

    Attributes:
        message: The user's current message/query
        task_type: Detected task type (e.g., "edit", "analyze", "debug")
        complexity: Task complexity level (e.g., "simple", "medium", "complex")
        goals: Detected goals or intents
        stage: Current conversation stage
        observed_files: Files already observed in this session
        executed_tools: Tools already executed in this session
        conversation_depth: Number of messages in conversation
        conversation_history: Full conversation history for context
    """

    message: str
    task_type: str = "unknown"
    complexity: str = "medium"
    goals: Optional[Any] = None
    stage: Optional[str] = None
    observed_files: Set[str] = field(default_factory=set)
    executed_tools: Set[str] = field(default_factory=set)
    conversation_depth: int = 0
    conversation_history: Optional[List[Dict[str, Any]]] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCoordinatorConfig:
    """Configuration for ToolCoordinator.

    Attributes:
        default_budget: Default tool call budget
        budget_multiplier: Multiplier for budget based on complexity
        enable_caching: Whether to use tool result caching
        enable_sequence_tracking: Whether to track tool sequences
        max_tools_per_selection: Maximum tools to select per turn
        selection_threshold: Minimum similarity threshold for tool selection
        retry_enabled: Whether to retry failed tool calls
        max_retry_attempts: Maximum retry attempts for failed tools
        retry_base_delay: Base delay for exponential backoff (seconds)
        retry_max_delay: Maximum delay for retry (seconds)
    """

    default_budget: int = 25
    budget_multiplier: float = 1.0
    enable_caching: bool = True
    enable_sequence_tracking: bool = True
    max_tools_per_selection: int = 15
    selection_threshold: float = 0.3
    retry_enabled: bool = True
    max_retry_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 10.0


# ToolExecutionResult is now canonical in victor.agent.tool_executor
# Import from there: from victor.agent.tool_executor import ToolExecutionResult


@runtime_checkable
class IToolCoordinator(Protocol):
    """Protocol for tool coordination operations."""

    async def select_tools(self, context: TaskContext) -> List[Any]: ...

    def get_remaining_budget(self) -> int: ...

    def consume_budget(self, amount: int) -> None: ...

    def is_tool_enabled(self, tool_name: str) -> bool: ...

    def get_enabled_tools(self) -> Set[str]: ...

    def set_enabled_tools(self, tools: Set[str]) -> None: ...

    async def execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], context: Optional[TaskContext] = None
    ) -> Any: ...

    def resolve_tool_alias(self, tool_name: str) -> str: ...

    def get_tool_usage_stats(self) -> Dict[str, Any]: ...


class ToolCoordinator:
    """Coordinates tool selection, budgeting, and execution.

    This class consolidates tool-related operations that were spread across
    the orchestrator, providing a unified interface for:

    1. Tool Selection: Combines semantic and keyword-based selection
    2. Budget Management: Tracks and enforces tool call budgets
    3. Access Control: Manages enabled/disabled tools
    4. Alias Resolution: Handles tool aliases and shell variants
    5. Execution: Delegates to ToolPipeline for actual execution
    6. Caching: Coordinates tool result caching

    Example:
        coordinator = ToolCoordinator(
            tool_pipeline=pipeline,
            tool_selector=selector,
            tool_registry=registry,
            config=ToolCoordinatorConfig(default_budget=30),
        )

        # In orchestrator:
        context = TaskContext(message=user_query, task_type="edit")
        tools = await coordinator.select_tools(context)

        # Check if we can execute more tools
        if coordinator.get_remaining_budget() > 0:
            result = await coordinator.execute_tool_calls(tool_calls)
    """

    def __init__(
        self,
        tool_pipeline: "ToolPipeline",
        tool_registry: "ToolRegistry",
        tool_selector: Optional["ToolSelector"] = None,
        budget_manager: Optional["BudgetManager"] = None,
        tool_cache: Optional["ToolCache"] = None,
        argument_normalizer: Optional["ArgumentNormalizer"] = None,
        tool_adapter: Optional["ToolCallingAdapter"] = None,
        tool_access_controller: Optional[Any] = None,
        config: Optional[ToolCoordinatorConfig] = None,
        on_selection_complete: Optional[Callable[[str, int], None]] = None,  # method, count
        on_budget_warning: Optional[Callable[[int, int], None]] = None,  # remaining, total
        on_tool_complete: Optional[Callable[[ToolExecutionResult], None]] = None,
    ) -> None:
        """Initialize the ToolCoordinator.

        Args:
            tool_pipeline: Pipeline for tool execution
            tool_registry: Tool registry for tool access control
            tool_selector: Optional selector for tool selection
            budget_manager: Optional manager for budget tracking
            tool_cache: Optional cache for tool results
            argument_normalizer: Optional normalizer for tool arguments
            tool_adapter: Optional tool calling adapter
            tool_access_controller: Optional access controller for layered checks
            config: Configuration options
            on_selection_complete: Callback when tool selection completes
            on_budget_warning: Callback when budget is running low
            on_tool_complete: Callback when tool execution completes
        """
        self._pipeline = tool_pipeline
        self._registry = tool_registry
        self._selector = tool_selector
        self._budget_manager = budget_manager
        self._cache = tool_cache
        self._argument_normalizer = argument_normalizer
        self._tool_adapter = tool_adapter
        self._tool_access_controller = tool_access_controller
        self._config = config or ToolCoordinatorConfig()

        # Callbacks
        self._on_selection_complete = on_selection_complete
        self._on_budget_warning = on_budget_warning
        self._on_tool_complete = on_tool_complete

        # Internal state
        self._budget_used: int = 0
        self._total_budget: int = self._config.default_budget
        self._selection_history: List[Tuple[str, int]] = []  # (method, count)
        self._execution_count: int = 0
        self._executed_tools: List[str] = []
        self._failed_tool_signatures: Set[Tuple[str, str]] = set()
        self._enabled_tools: Optional[Set[str]] = None

        # Tool access dependencies (injected after init)
        self._mode_controller: Optional[Any] = None
        self._tool_planner: Optional[Any] = None

        logger.debug(
            f"ToolCoordinator initialized with budget={self._total_budget}, "
            f"caching={self._config.enable_caching}"
        )

    # =====================================================================
    # Budget Management
    # =====================================================================

    @property
    def budget(self) -> int:
        """Get the total tool budget."""
        return self._total_budget

    @budget.setter
    def budget(self, value: int) -> None:
        """Set the total tool budget."""
        self._total_budget = max(0, value)

    @property
    def budget_used(self) -> int:
        """Get the number of budget units used."""
        return self._budget_used

    @property
    def execution_count(self) -> int:
        """Get the total number of tool executions."""
        return self._execution_count

    def get_remaining_budget(self) -> int:
        """Get remaining tool call budget.

        Returns:
            Number of tool calls remaining
        """
        if self._budget_manager:
            return self._budget_manager.get_remaining_tool_calls()
        return max(0, self._total_budget - self._budget_used)

    def consume_budget(self, amount: int = 1) -> None:
        """Consume tool call budget.

        Args:
            amount: Number of budget units to consume
        """
        self._budget_used += amount

        if self._budget_manager:
            self._budget_manager.consume_tool_call(amount)

        remaining = self.get_remaining_budget()
        total = self._total_budget

        # Warn when budget is low (less than 20% remaining)
        if remaining < total * 0.2 and self._on_budget_warning:
            self._on_budget_warning(remaining, total)

        logger.debug(f"Budget consumed: {amount}, remaining: {remaining}/{total}")

    def reset_budget(self, new_budget: Optional[int] = None) -> None:
        """Reset the tool budget.

        Args:
            new_budget: New budget to set, or use default
        """
        self._budget_used = 0
        if new_budget is not None:
            self._total_budget = new_budget
        else:
            self._total_budget = self._config.default_budget

        if self._budget_manager:
            self._budget_manager.reset(new_budget or self._config.default_budget)

        logger.debug(f"Budget reset to {self._total_budget}")

    def set_budget_multiplier(self, multiplier: float) -> None:
        """Set budget multiplier for complexity adjustments.

        Args:
            multiplier: Multiplier to apply (e.g., 2.0 for complex tasks)
        """
        self._config.budget_multiplier = multiplier
        effective_budget = int(self._config.default_budget * multiplier)
        self._total_budget = effective_budget

        logger.debug(f"Budget multiplier set to {multiplier}, effective budget: {effective_budget}")

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted.

        Returns:
            True if no budget remaining
        """
        return self.get_remaining_budget() <= 0

    # =====================================================================
    # Tool Selection
    # =====================================================================

    async def select_tools(
        self,
        context: TaskContext,
        max_tools: Optional[int] = None,
    ) -> List[Any]:
        """Select appropriate tools for the current context.

        Uses the configured ToolSelector to determine which tools are
        relevant for the current task.

        Args:
            context: Task context for selection
            max_tools: Maximum tools to select (overrides config)

        Returns:
            List of selected tool definitions
        """
        if not self._selector:
            logger.warning("No tool selector configured, returning empty list")
            return []

        max_count = max_tools or self._config.max_tools_per_selection
        threshold = self._config.selection_threshold

        try:
            # Build context dict for selector
            selector_context = {
                "message": context.message,
                "task_type": context.task_type,
                "observed_files": context.observed_files,
                "executed_tools": context.executed_tools,
                "conversation_depth": context.conversation_depth,
                "conversation_history": context.conversation_history,
            }

            # Use selector's async select_tools method
            tools = await self._selector.select_tools(
                message=context.message,
                task_type=context.task_type,
                max_tools=max_count,
                threshold=threshold,
                context=selector_context,
            )

            # Track selection
            method = getattr(self._selector, "last_selection_method", "unknown")
            self._selection_history.append((method, len(tools)))

            if self._on_selection_complete:
                self._on_selection_complete(method, len(tools))

            logger.debug(
                f"Selected {len(tools)} tools via {method} for task_type={context.task_type}"
            )
            return tools

        except Exception as e:
            logger.warning(f"Tool selection failed: {e}, returning empty list")
            return []

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get tool selection statistics.

        Returns:
            Dict with selection method distribution and counts
        """
        method_counts: Dict[str, int] = {}
        total_selected = 0

        for method, count in self._selection_history:
            method_counts[method] = method_counts.get(method, 0) + 1
            total_selected += count

        return {
            "total_selections": len(self._selection_history),
            "total_tools_selected": total_selected,
            "method_distribution": method_counts,
            "avg_tools_per_selection": (
                total_selected / len(self._selection_history) if self._selection_history else 0
            ),
        }

    # =====================================================================
    # Tool Access Control
    # =====================================================================

    def get_enabled_tools(self) -> Set[str]:
        """Get currently enabled tool names.

        Returns:
            Set of enabled tool names for this session
        """
        # Use ToolAccessController if available
        if self._tool_access_controller:
            context = self._build_tool_access_context()
            return self._tool_access_controller.get_allowed_tools(context)

        # Check mode controller for BUILD mode (allows all tools)
        if self._mode_controller:
            config = self._mode_controller.config
            if config.allow_all_tools:
                all_tools = self.get_available_tools()
                enabled = all_tools - config.disallowed_tools
                return enabled

        # Return framework-set tools
        if self._enabled_tools:
            return self._enabled_tools

        # Fall back to all available tools
        return self.get_available_tools()

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set which tools are enabled for this session.

        Args:
            tools: Set of tool names to enable
        """
        self._enabled_tools = tools

        # Propagate to tool_selector if available
        if self._selector and hasattr(self._selector, "set_enabled_tools"):
            self._selector.set_enabled_tools(tools)
            logger.info(f"Enabled tools filter propagated to selector: {sorted(tools)}")

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is enabled
        """
        # Use ToolAccessController if available
        if self._tool_access_controller:
            context = self._build_tool_access_context()
            decision = self._tool_access_controller.check_access(tool_name, context)
            return decision.allowed

        # Check mode controller restrictions
        if self._mode_controller:
            config = self._mode_controller.config
            if tool_name in config.disallowed_tools:
                return False
            if config.allow_all_tools:
                if self._registry and tool_name in self._registry.list_tools():
                    return True

        # Fall back to session/vertical restrictions
        enabled = self.get_enabled_tools()
        return tool_name in enabled

    def get_available_tools(self) -> Set[str]:
        """Get all registered tool names.

        Returns:
            Set of tool names available in registry
        """
        if self._registry:
            return set(self._registry.list_tools())
        return set()

    def _build_tool_access_context(self) -> "ToolAccessContext":
        """Build ToolAccessContext for unified access control checks.

        Returns:
            ToolAccessContext with session tools and current mode
        """
        from victor.agent.protocols import ToolAccessContext

        return ToolAccessContext(
            session_enabled_tools=self._enabled_tools,
            current_mode=(self._mode_controller.config.name if self._mode_controller else None),
        )

    # =====================================================================
    # Tool Alias Resolution
    # =====================================================================

    def resolve_tool_alias(self, tool_name: str) -> str:
        """Resolve tool alias to canonical name.

        Handles shell variants and other tool aliases.

        Args:
            tool_name: Original tool name (may be alias)

        Returns:
            Canonical tool name
        """
        # Resolve shell aliases to appropriate enabled variant
        shell_aliases = {
            "run",
            "bash",
            "execute",
            "cmd",
            "execute_bash",
            "shell_readonly",
            "shell",
        }

        if tool_name not in shell_aliases:
            return tool_name

        # Check mode controller for BUILD mode (allows all tools including shell)
        if self._mode_controller:
            config = self._mode_controller.config
            if config.allow_all_tools and "shell" not in config.disallowed_tools:
                logger.debug(f"Resolved '{tool_name}' to 'shell' (BUILD mode allows all tools)")
                return ToolNames.SHELL

        # Check if full shell is enabled first
        if self._registry and self._registry.is_tool_enabled(ToolNames.SHELL):
            logger.debug(f"Resolved '{tool_name}' to 'shell' (shell enabled)")
            return ToolNames.SHELL

        # Fall back to shell_readonly if enabled
        if self._registry and self._registry.is_tool_enabled(ToolNames.SHELL_READONLY):
            logger.debug(f"Resolved '{tool_name}' to 'shell_readonly' (readonly mode)")
            return ToolNames.SHELL_READONLY

        # Neither enabled - return canonical name (will fail validation)
        from victor.tools.tool_names import get_canonical_name

        canonical = get_canonical_name(tool_name)
        logger.debug(f"No shell variant enabled for '{tool_name}', using canonical '{canonical}'")
        return canonical

    # =====================================================================
    # Tool Execution
    # =====================================================================

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[TaskContext] = None,
    ) -> "PipelineExecutionResult":
        """Execute tool calls through the pipeline.

        Delegates to ToolPipeline for actual execution, handling budget
        tracking and caching coordination.

        Args:
            tool_calls: List of tool calls to execute
            context: Optional task context

        Returns:
            PipelineExecutionResult with execution details
        """
        # Check budget before execution
        remaining = self.get_remaining_budget()
        call_count = len(tool_calls)

        if remaining < call_count:
            logger.warning(f"Insufficient budget: {remaining} remaining, {call_count} requested")

        # Build execution context
        execution_context = {}
        if context:
            execution_context = {
                "task_type": context.task_type,
                "complexity": context.complexity,
                "observed_files": context.observed_files,
                "executed_tools": context.executed_tools,
            }

        # Execute through pipeline
        result = await self._pipeline.execute_tool_calls(
            tool_calls=tool_calls,
            context=execution_context,
        )

        # Update tracking
        successful = result.successful_calls if hasattr(result, "successful_calls") else 0
        self._execution_count += successful
        self.consume_budget(successful)

        logger.debug(
            f"Executed {successful}/{call_count} tool calls, "
            f"budget remaining: {self.get_remaining_budget()}"
        )

        return result

    async def execute_tool_with_retry(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Optional[Any], bool, Optional[str]]:
        """Execute a tool with retry logic and exponential backoff.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            context: Execution context

        Returns:
            Tuple of (result, success, error_message or None)
        """
        # Try cache first for allowlisted tools
        if self._cache:
            cached = self._cache.get(tool_name, tool_args)
            if cached is not None:
                logger.debug(f"Cache hit for tool '{tool_name}'")
                return cached, True, None

        retry_enabled = self._config.retry_enabled
        max_attempts = self._config.max_retry_attempts if retry_enabled else 1
        base_delay = self._config.retry_base_delay
        max_delay = self._config.retry_max_delay

        last_error = None
        for attempt in range(max_attempts):
            try:
                result = await self._pipeline._execute_single_tool(tool_name, tool_args, context)

                if result.success:
                    # Cache successful result
                    if self._cache:
                        self._cache.set(tool_name, tool_args, result)
                        # Invalidate related cache entries
                        invalidating_tools = {
                            "write_file",
                            "edit_files",
                            "execute_bash",
                            "git",
                            "docker",
                        }
                        if tool_name in invalidating_tools:
                            touched_paths = []
                            if "path" in tool_args:
                                touched_paths.append(tool_args["path"])
                            if "paths" in tool_args and isinstance(tool_args["paths"], list):
                                touched_paths.extend(tool_args["paths"])
                            if touched_paths:
                                self._cache.invalidate_paths(touched_paths)
                            else:
                                namespaces_to_clear = [
                                    "code_search",
                                    "semantic_code_search",
                                    "list_directory",
                                ]
                                self._cache.clear_namespaces(namespaces_to_clear)

                    if attempt > 0:
                        logger.info(
                            f"Tool '{tool_name}' succeeded on retry attempt {attempt + 1}/{max_attempts}"
                        )

                    return result, True, None
                else:
                    # Tool returned failure - check if retryable
                    error_msg = result.error or "Unknown error"

                    # Don't retry validation errors or permanent failures
                    non_retryable_errors = [
                        "Invalid",
                        "Missing required",
                        "Not found",
                        "disabled",
                    ]
                    if any(err in error_msg for err in non_retryable_errors):
                        logger.debug(
                            f"Tool '{tool_name}' failed with non-retryable error: {error_msg}"
                        )
                        return result, False, error_msg

                    last_error = error_msg
                    if attempt < max_attempts - 1:
                        # Calculate exponential backoff delay
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            f"Tool '{tool_name}' failed (attempt {attempt + 1}/{max_attempts}): {error_msg}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Tool '{tool_name}' failed after {max_attempts} attempts: {error_msg}"
                        )
                        return result, False, error_msg

            except Exception as e:
                # Check for non-retryable errors
                from victor.core.errors import ToolNotFoundError, ToolValidationError

                if isinstance(e, (ToolNotFoundError, ToolValidationError, PermissionError)):
                    logger.error(f"Tool '{tool_name}' permanent failure: {e}")
                    return None, False, str(e)

                # Retryable transient errors
                last_error = str(e)
                if attempt < max_attempts - 1:
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"Tool '{tool_name}' transient error (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Tool '{tool_name}' failed after {max_attempts} attempts: {e}")
                    return None, False, last_error

        # Should not reach here, but handle it anyway
        return None, False, last_error or "Unknown error"

    # =====================================================================
    # Tool Call Parsing
    # =====================================================================

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> "ToolCallParseResult":
        """Parse tool calls using the tool calling adapter.

        Handles:
        1. Native tool calls from provider
        2. JSON fallback parsing
        3. XML fallback parsing
        4. Tool name validation

        Args:
            content: Response content text
            raw_tool_calls: Native tool_calls from provider (if any)

        Returns:
            ToolCallParseResult with parsed tool calls and metadata
        """
        if not self._tool_adapter:
            from victor.agent.tool_calling import ToolCallParseResult

            return ToolCallParseResult(
                tool_calls=[],
                parse_method="none",
                confidence=0.0,
                warnings=["No tool adapter configured"],
            )

        result = self._tool_adapter.parse_tool_calls(content, raw_tool_calls)

        # Log any warnings
        for warning in result.warnings:
            logger.warning(f"Tool call parse warning: {warning}")

        # Log parse method for debugging
        if result.tool_calls:
            logger.debug(
                f"Parsed {len(result.tool_calls)} tool calls via {result.parse_method} "
                f"(confidence={result.confidence})"
            )

        return result

    def normalize_tool_arguments(
        self,
        tool_args: Dict[str, Any],
        tool_name: str,
    ) -> Tuple[Dict[str, Any], "NormalizationStrategy"]:
        """Normalize tool arguments to handle malformed JSON.

        Args:
            tool_args: Raw arguments from tool call
            tool_name: Name of the tool being called

        Returns:
            Tuple of (normalized_args, strategy_used)
        """
        if not self._argument_normalizer:
            return tool_args, NormalizationStrategy.DIRECT

        return self._argument_normalizer.normalize_arguments(tool_args, tool_name)

    # =====================================================================
    # Statistics and Tracking
    # =====================================================================

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics.

        Returns:
            Dict with execution counts and budget usage
        """
        return {
            "total_executions": self._execution_count,
            "budget_used": self._budget_used,
            "budget_total": self._total_budget,
            "budget_remaining": self.get_remaining_budget(),
            "budget_utilization": (
                self._budget_used / self._total_budget if self._total_budget > 0 else 0
            ),
            "executed_tools": list(self._executed_tools),
            "failed_signatures_count": len(self._failed_tool_signatures),
        }

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool usage statistics.

        Returns:
            Dictionary with usage analytics including:
            - Selection stats (semantic/keyword/fallback counts)
            - Per-tool execution stats (calls, success rate, timing)
            - Cost tracking (by tier and total)
            - Overall metrics
        """
        return {
            "selection": self.get_selection_stats(),
            "execution": self.get_execution_stats(),
            "budget": {
                "total": self._total_budget,
                "used": self._budget_used,
                "remaining": self.get_remaining_budget(),
            },
        }

    def clear_selection_history(self) -> None:
        """Clear the selection history."""
        self._selection_history.clear()

    def clear_failed_signatures(self) -> None:
        """Clear the failed tool signatures cache."""
        self._failed_tool_signatures.clear()

    # =====================================================================
    # Dependency Injection
    # =====================================================================

    def set_mode_controller(self, mode_controller: Any) -> None:
        """Set the mode controller for tool access control.

        Args:
            mode_controller: ModeController instance
        """
        self._mode_controller = mode_controller

    def set_tool_planner(self, tool_planner: Any) -> None:
        """Set the tool planner for goal-based tool selection.

        Args:
            tool_planner: ToolPlanner instance
        """
        self._tool_planner = tool_planner

    def set_orchestrator_reference(self, orchestrator: Any) -> None:
        """Set orchestrator reference for callback context.

        Args:
            orchestrator: AgentOrchestrator instance
        """
        self._orchestrator = orchestrator

    # =====================================================================
    # Tool Call Validation & Normalization
    # =====================================================================

    def validate_tool_call(
        self,
        tool_call: Any,
        sanitizer: Any,
        is_tool_enabled_fn: Optional[Callable[[str], bool]] = None,
    ) -> "ToolCallValidation":
        """Validate a single tool call structure, name, and enabled status.

        Encapsulates structure checks, name format validation, alias resolution,
        and enabled-tool checking.

        Args:
            tool_call: Raw tool call dict from the model
            sanitizer: ResponseSanitizer with is_valid_tool_name()
            is_tool_enabled_fn: Optional callback to check if tool is enabled.
                Defaults to self.is_tool_enabled.

        Returns:
            ToolCallValidation with validation result
        """
        _is_enabled = is_tool_enabled_fn or self.is_tool_enabled
        # Structure check
        if not isinstance(tool_call, dict):
            return ToolCallValidation(
                valid=False,
                skip_reason=f"Skipping invalid tool call (not a dict): {tool_call}",
            )

        tool_name = tool_call.get("name")

        # Missing name
        if not tool_name:
            return ToolCallValidation(
                valid=False,
                skip_reason="Skipping tool call without name",
                error_result={
                    "tool_name": "",
                    "success": False,
                    "result": None,
                    "error": (
                        "Tool call missing name. Each tool call must include a 'name' field. "
                        "Please specify which tool you want to use."
                    ),
                },
            )

        # Format validation (reject hallucinated/malformed names)
        if not sanitizer.is_valid_tool_name(tool_name):
            return ToolCallValidation(
                valid=False,
                original_name=tool_name,
                skip_reason=f"Skipping invalid/hallucinated tool name: {tool_name}",
                error_result={
                    "tool_name": tool_name,
                    "success": False,
                    "result": None,
                    "error": (
                        f"Invalid tool name '{tool_name}'. This tool does not exist. "
                        "Use only tools from the provided tool list. "
                        "Check for typos or hallucinated tool names."
                    ),
                },
            )

        # Resolve legacy/alias names to canonical form
        try:
            from victor.tools.decorators import resolve_tool_name

            canonical = resolve_tool_name(tool_name)
        except Exception:
            canonical = tool_name

        # Check if enabled
        if not _is_enabled(canonical):
            return ToolCallValidation(
                valid=False,
                original_name=tool_name,
                canonical_name=canonical,
                skip_reason=(
                    f"Skipping unknown or disabled tool: {tool_name} (resolved: {canonical})"
                ),
                error_result={
                    "tool_name": tool_name,
                    "success": False,
                    "result": None,
                    "error": (
                        f"Tool '{tool_name}' is not available. It may be disabled, not registered, "
                        "or not included in the current tool selection. "
                        "Use only the tools listed in your available tools."
                    ),
                },
            )

        return ToolCallValidation(
            valid=True,
            original_name=tool_name,
            canonical_name=canonical,
        )

    def normalize_arguments_full(
        self,
        tool_name: str,
        original_name: str,
        raw_args: Any,
        argument_normalizer: Any,
        tool_adapter: Any,
        failed_signatures: Optional[Set[Tuple[str, str]]] = None,
    ) -> "NormalizedArgs":
        """Normalize tool arguments through all stages.

        Encapsulates: JSON string parsing, ArgumentNormalizer, ToolCallingAdapter,
        git operation inference, and dedup signature computation.

        Args:
            tool_name: Canonical tool name
            original_name: Original (alias) tool name from the model
            raw_args: Raw arguments from tool call
            argument_normalizer: ArgumentNormalizer instance
            tool_adapter: ToolCallingAdapter instance
            failed_signatures: Set of previously failed (name, args_json) tuples.
                Defaults to self._failed_tool_signatures.

        Returns:
            NormalizedArgs with normalized arguments and metadata
        """
        _failed = (
            failed_signatures if failed_signatures is not None else self._failed_tool_signatures
        )
        import ast as _ast

        from victor.agent.argument_normalizer import NormalizationStrategy
        from victor.agent.orchestrator_utils import infer_git_operation

        # Stage 1: Parse JSON strings to dicts
        tool_args = raw_args
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except Exception:
                try:
                    tool_args = _ast.literal_eval(tool_args)
                except Exception:
                    tool_args = {"value": tool_args}
        elif tool_args is None:
            tool_args = {}

        # Stage 2: ArgumentNormalizer (handles malformed JSON syntax)
        normalized_args, strategy = argument_normalizer.normalize_arguments(tool_args, tool_name)

        # Stage 3: ToolCallingAdapter (handles missing required params)
        normalized_args = tool_adapter.normalize_arguments(normalized_args, tool_name)

        # Stage 4: Git operation inference from alias
        normalized_args = infer_git_operation(original_name, tool_name, normalized_args)

        # Stage 5: Compute dedup signature
        try:
            signature = (tool_name, json.dumps(normalized_args, sort_keys=True, default=str))
        except Exception:
            signature = (tool_name, str(normalized_args))

        is_repeated = signature in _failed

        return NormalizedArgs(
            args=normalized_args,
            strategy=strategy,
            signature=signature,
            is_repeated_failure=is_repeated,
        )


@dataclass
class ToolCallValidation:
    """Result of validating a single tool call.

    Attributes:
        valid: Whether the tool call passed all validation checks
        original_name: Original tool name from the model (if present)
        canonical_name: Resolved canonical tool name (if valid)
        skip_reason: Human-readable reason for skipping (for console output)
        error_result: Error dict to feed back to the model (for learning)
    """

    valid: bool
    original_name: Optional[str] = None
    canonical_name: Optional[str] = None
    skip_reason: Optional[str] = None
    error_result: Optional[Dict[str, Any]] = None


@dataclass
class NormalizedArgs:
    """Result of full argument normalization.

    Attributes:
        args: The normalized argument dict
        strategy: The NormalizationStrategy used
        signature: Dedup signature tuple (tool_name, args_json)
        is_repeated_failure: Whether this signature already failed
    """

    args: Dict[str, Any]
    strategy: Any  # NormalizationStrategy
    signature: Tuple[str, str]
    is_repeated_failure: bool = False


def create_tool_coordinator(
    tool_pipeline: "ToolPipeline",
    tool_registry: "ToolRegistry",
    tool_selector: Optional["ToolSelector"] = None,
    budget_manager: Optional["BudgetManager"] = None,
    config: Optional[ToolCoordinatorConfig] = None,
) -> ToolCoordinator:
    """Factory function to create a ToolCoordinator.

    Args:
        tool_pipeline: Pipeline for tool execution
        tool_registry: Tool registry for tool access
        tool_selector: Optional selector for tool selection
        budget_manager: Optional manager for budget tracking
        config: Configuration options

    Returns:
        Configured ToolCoordinator instance
    """
    return ToolCoordinator(
        tool_pipeline=tool_pipeline,
        tool_registry=tool_registry,
        tool_selector=tool_selector,
        budget_manager=budget_manager,
        config=config,
    )


__all__ = [
    "ToolCoordinator",
    "ToolCoordinatorConfig",
    "TaskContext",
    "ToolCallValidation",
    "NormalizedArgs",
    "ToolExecutionResult",
    "IToolCoordinator",
    "create_tool_coordinator",
]
