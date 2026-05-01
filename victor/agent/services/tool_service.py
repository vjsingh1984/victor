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

"""Tool service implementation.

Extracts tool operations from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Tool selection based on context
- Tool execution with budgeting
- Tool usage tracking
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from victor.tools.core_tool_aliases import canonicalize_core_tool_name

if TYPE_CHECKING:
    from victor.agent.services.protocols.tool_service import ToolSelectionContext

logger = logging.getLogger(__name__)


@dataclass
class ToolResultContext:
    """State-passed context for post-processing tool pipeline results.

    This is the canonical service-layer carrier for mutable state and callbacks
    needed by ``ToolService.process_tool_results``.
    """

    executed_tools: List[str]
    observed_files: Set[str]
    failed_tool_signatures: Set[str]
    shown_tool_errors: Set[str]
    continuation_prompts: int = 0
    asking_input_prompts: int = 0
    tool_calls_used: int = 0
    record_tool_execution: Optional[Any] = None
    conversation_state: Optional[Any] = None
    unified_tracker: Optional[Any] = None
    usage_logger: Optional[Any] = None
    add_message: Optional[Any] = None
    format_tool_output: Optional[Any] = None
    console: Optional[Any] = None
    presentation: Optional[Any] = None
    stream_context: Optional[Any] = None
    task_type: str = "unknown"


def normalize_tool_result_arguments(arguments: Any) -> Dict[str, Any]:
    """Normalize tool arguments for formatting/pruning decisions."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


_PREVIEW_ONLY_TOOL_RESULT_FIELDS = frozenset(
    {
        "formatted_output",
        "formatted_summary",
        "formatted_results",
        "contains_markup",
    }
)


def _strip_preview_only_tool_fields(output: Any) -> Any:
    """Remove preview-only fields from tool results before LLM injection.

    Rich preview payloads improve CLI/TUI UX, but they duplicate structured data
    and add token cost when sent back to the model. Keep them for display paths
    while excluding them from the LLM-facing serialization.
    """
    if isinstance(output, dict):
        stripped = {
            key: _strip_preview_only_tool_fields(value)
            for key, value in output.items()
            if key not in _PREVIEW_ONLY_TOOL_RESULT_FIELDS
        }
        # Preserve original content if preview fields were the only payload.
        return stripped or output
    if isinstance(output, list):
        return [_strip_preview_only_tool_fields(value) for value in output]
    return output


def format_and_prune_tool_output(
    tool_name: str,
    arguments: Any,
    output: Any,
    *,
    task_type: str = "unknown",
    formatter: Optional[Any] = None,
) -> tuple[str, str, str, bool, Optional[Any]]:
    """Format tool output for display and LLM injection.

    CRITICAL: LLM receives FULL output for accuracy. Pruning only affects user preview.
    """
    normalized_args = normalize_tool_result_arguments(arguments)
    llm_safe_output = _strip_preview_only_tool_fields(output)
    llm_output_changed = llm_safe_output != output

    if formatter is not None:
        formatted_output = formatter(tool_name, normalized_args, output)
        llm_output = (
            formatter(tool_name, normalized_args, llm_safe_output)
            if llm_output_changed
            else formatted_output
        )
    else:
        from victor.agent.tool_output_formatter import format_tool_output as default_formatter
        from victor.agent.tool_output_formatter import ToolOutputFormatterConfig

        # CRITICAL: Use very high max_output_chars for LLM input (no truncation)
        # This ensures LLM receives complete tool output for accurate decisions
        llm_formatter_config = ToolOutputFormatterConfig(
            max_output_chars=1_000_000,  # 1MB - effectively no truncation for LLM
            file_structure_threshold=30000,
            min_savings_threshold=0.15,
            max_classes_shown=15,
            max_functions_shown=20,
            sample_lines_start=20,
            sample_lines_end=15,
        )

        # Format with NO truncation for LLM input
        formatted_output = default_formatter(
            tool_name,
            normalized_args,
            output,
            config=llm_formatter_config,
        )
        llm_output = (
            default_formatter(
                tool_name,
                normalized_args,
                llm_safe_output,
                config=llm_formatter_config,
            )
            if llm_output_changed
            else formatted_output
        )

    if not isinstance(formatted_output, str):
        formatted_output = str(formatted_output)
    if not isinstance(llm_output, str):
        llm_output = str(llm_output)

    # Generate preview for user display only (does not affect LLM)
    preview_output = formatted_output
    was_pruned = False
    pruning_info = None

    from victor.config.tool_settings import get_tool_settings
    from victor.tools.output_pruner import get_output_pruner

    tool_settings = get_tool_settings()
    if tool_settings.tool_output_preview_enabled and tool_settings.tool_output_pruning_enabled:
        pruner = get_output_pruner()
        # Prune only for user preview, NOT for LLM input
        preview_output, pruning_info = pruner.prune(
            tool_output=formatted_output,
            task_type=task_type,
            tool_name=tool_name,
            context={
                "formatted_output": True,
                "safe_only": tool_settings.tool_output_pruning_safe_only,
                "tool_args": normalized_args,
            },
        )
        was_pruned = pruning_info.was_pruned

    if not isinstance(preview_output, str):
        preview_output = str(preview_output)

    # Return:
    # - preview_output: user-facing preview payload (may be pruned)
    # - llm_output: full output sent to the model
    # - formatted_output: full display output retained for expansion/debug
    # - was_pruned / pruning_info: preview pruning metadata
    return preview_output, llm_output, formatted_output, was_pruned, pruning_info


def process_tool_results_with_context(
    pipeline_result: Any,
    ctx: ToolResultContext,
) -> List[Dict[str, Any]]:
    """Canonical service-layer implementation for tool result post-processing."""
    results: List[Dict[str, Any]] = []

    for call_result in pipeline_result.results:
        # Invalid tools with tool_call_id must still produce a response per OpenAI spec.
        if call_result.skipped and not call_result.tool_call_id:
            continue

        tool_name = call_result.tool_name
        normalized_args = call_result.arguments or {}
        output = call_result.result
        success = call_result.success
        error_msg = call_result.error
        elapsed_ms = call_result.execution_time_ms
        skipped = bool(getattr(call_result, "skipped", False))
        skip_reason = getattr(call_result, "skip_reason", None)
        outcome_kind = getattr(call_result, "outcome_kind", None)
        block_source = getattr(call_result, "block_source", None)
        retryable = getattr(call_result, "retryable", None)
        user_message = getattr(call_result, "user_message", None)

        ctx.executed_tools.append(tool_name)
        if tool_name == "read" and "path" in normalized_args:
            ctx.observed_files.add(str(normalized_args.get("path")))

        if ctx.stream_context is not None:
            ctx.stream_context.reset_activity_timer()

        if success:
            if ctx.continuation_prompts > 0:
                ctx.continuation_prompts = 0
            if ctx.asking_input_prompts > 0:
                ctx.asking_input_prompts = 0

        error_type = type(error_msg).__name__ if error_msg and not success else None
        if ctx.record_tool_execution:
            ctx.record_tool_execution(tool_name, success, elapsed_ms, error_type=error_type)
        if ctx.conversation_state:
            ctx.conversation_state.record_tool_execution(tool_name, normalized_args)

        result_dict: Dict[str, Any] = {"success": success}
        if output is not None:
            result_dict["result"] = output
        if ctx.unified_tracker:
            ctx.unified_tracker.update_from_tool_call(tool_name, normalized_args, result_dict)

        follow_up_suggestions = None
        semantic_success = success
        if isinstance(output, dict):
            metadata = output.get("metadata")
            if isinstance(metadata, dict):
                suggestions = metadata.get("follow_up_suggestions")
                if isinstance(suggestions, list) and suggestions:
                    follow_up_suggestions = suggestions

        if success and isinstance(output, dict) and output.get("success") is False:
            semantic_success = False
            error_msg = output.get("error", "Operation returned success=False")

        if semantic_success:
            error_display = None
        else:
            error_display = user_message or error_msg or skip_reason or "Unknown error"

        if ctx.usage_logger and hasattr(ctx.usage_logger, "set_duration_context"):
            ctx.usage_logger.set_duration_context(elapsed_ms)
        if ctx.usage_logger:
            ctx.usage_logger.log_event(
                "tool_result",
                {
                    "tool_name": tool_name,
                    "success": semantic_success,
                    "skipped": skipped,
                    "outcome_kind": outcome_kind,
                    "block_source": block_source,
                    "retryable": retryable,
                    "result": output,
                    "error": error_display,
                },
            )

        if semantic_success:
            preview_output, llm_output, full_output, was_pruned, pruning_info = (
                format_and_prune_tool_output(
                    tool_name=tool_name,
                    arguments=normalized_args,
                    output=output,
                    task_type=ctx.task_type,
                    formatter=ctx.format_tool_output,
                )
            )
            if ctx.add_message:
                # CRITICAL: Send FULL output to LLM for accuracy
                # llm_output is always the full formatted output (not pruned)
                ctx.add_message(
                    "tool",
                    llm_output,  # Full output - LLM needs complete context
                    name=tool_name,
                    tool_call_id=call_result.tool_call_id,
                )
            results.append(
                {
                    "name": tool_name,
                    "success": True,
                    "elapsed": elapsed_ms / 1000,
                    "args": normalized_args,
                    "result": preview_output,  # Pruned preview for display when applicable
                    "full_result": full_output,  # Full display output for expansion/debug
                    "follow_up_suggestions": follow_up_suggestions,
                    "was_pruned": was_pruned,  # Indicates user preview was truncated (not LLM input)
                    "pruning_info": pruning_info,
                    "tool_call_id": call_result.tool_call_id,
                    "content": llm_output,  # Full output sent to LLM
                    "skipped": skipped,
                    "outcome_kind": outcome_kind,
                    "block_source": block_source,
                    "retryable": retryable,
                    "user_message": user_message,
                }
            )
            continue

        if not skipped:
            sig = f"{tool_name}:{hash(str(sorted(normalized_args.items())))}"
            ctx.failed_tool_signatures.add(sig)

        _not_found = "not found" in str(error_display).lower()
        _shown_key = f"notfound:{tool_name}" if _not_found else None
        if not (_shown_key and _shown_key in ctx.shown_tool_errors):
            if _shown_key and len(ctx.shown_tool_errors) < 500:
                ctx.shown_tool_errors.add(_shown_key)
            if ctx.console and ctx.presentation:
                prefix = "Tool call skipped" if skipped else "Tool execution failed"
                ctx.console.print(
                    f"[red]{ctx.presentation.icon('error', with_color=False)} "
                    f"{prefix}: {error_display}[/] "
                    f"[dim]({elapsed_ms:.0f}ms)[/dim]"
                )

        if isinstance(output, dict):
            error_output = dict(output)
        else:
            error_output = {"error": error_display}
        if skipped:
            error_output["skipped"] = True
            if skip_reason:
                error_output["skip_reason"] = skip_reason
        if outcome_kind:
            error_output["outcome_kind"] = outcome_kind
        if block_source:
            error_output["block_source"] = block_source
        if retryable is not None:
            error_output["retryable"] = retryable
        if user_message:
            error_output["user_message"] = user_message
        if ctx.format_tool_output:
            formatted_error = ctx.format_tool_output(tool_name, normalized_args, error_output)
        else:
            formatted_error, _, _, _, _ = format_and_prune_tool_output(
                tool_name=tool_name,
                arguments=normalized_args,
                output=error_output,
                task_type=ctx.task_type,
            )
        if ctx.add_message:
            ctx.add_message(
                "tool",
                formatted_error,
                name=tool_name,
                tool_call_id=call_result.tool_call_id,
            )
        results.append(
            {
                "name": tool_name,
                "success": False,
                "elapsed": elapsed_ms / 1000,
                "error": error_display,
                "result": formatted_error,
                "full_result": formatted_error,
                "follow_up_suggestions": follow_up_suggestions,
                "was_pruned": False,
                "tool_call_id": call_result.tool_call_id,
                "content": formatted_error,
                "skipped": skipped,
                "outcome_kind": outcome_kind,
                "block_source": block_source,
                "retryable": retryable,
                "user_message": user_message,
            }
        )

    return results


class ToolServiceConfig:
    """Configuration for ToolService.

    Attributes:
        default_max_tools: Default maximum tools per selection
        default_tool_budget: Default tool budget per session
        enable_parallel_execution: Enable parallel tool execution
        enable_caching: Enable tool result caching
        cache_ttl: Cache TTL in seconds
    """

    def __init__(
        self,
        default_max_tools: int = 10,
        default_tool_budget: int = 100,
        enable_parallel_execution: bool = True,
        enable_caching: bool = True,
        cache_ttl: int = 600,
        retry_enabled: bool = True,
        max_retry_attempts: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 10.0,
    ):
        self.default_max_tools = default_max_tools
        self.default_tool_budget = default_tool_budget
        self.enable_parallel_execution = enable_parallel_execution
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.retry_enabled = retry_enabled
        self.max_retry_attempts = max_retry_attempts
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay


class BudgetManager:
    """Manages executable tool budget for the current turn.

    Tracks tool calls and enforces limits to prevent excessive
    tool usage inside a single agentic turn or prompt cycle.

    Attributes:
        max_budget: Maximum tool calls allowed
        calls_made: Number of tool calls made
    """

    def __init__(self, max_budget: int = 100):
        self.max_budget = max_budget
        self.calls_made = 0

    def is_exhausted(self) -> bool:
        """Check if budget is exhausted.

        Returns:
            True if no more tool calls allowed
        """
        return self.calls_made >= self.max_budget

    def record_usage(self, count: int = 1) -> None:
        """Record tool usage.

        Args:
            count: Number of tool calls to record
        """
        self.calls_made += count

    def get_remaining(self) -> int:
        """Get remaining budget.

        Returns:
            Number of tool calls remaining
        """
        return max(0, self.max_budget - self.calls_made)

    def reset(self) -> None:
        """Reset budget to initial state."""
        self.calls_made = 0


class ToolService:
    """[CANONICAL] Service for managing tool operations.

    The target implementation for tool operations following the
    state-passed architectural pattern. Supersedes ToolCoordinator.

    This service follows SOLID principles:
    - SRP: Only handles tool operations
    - OCP: Extensible through strategy pattern
    - LSP: Implements ToolServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on abstractions

    Example:
        config = ToolServiceConfig()
        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        tools = await service.select_tools(context)
        result = await service.execute_tool("read", {"path": "file.txt"})
    """

    def __init__(
        self,
        config: ToolServiceConfig,
        tool_selector: Any,
        tool_executor: Any,
        tool_registrar: Any,
    ):
        """Initialize the tool service.

        Args:
            config: Service configuration
            tool_selector: Tool selection component
            tool_executor: Tool execution component
            tool_registrar: Tool registration component
        """
        self._config = config
        self._selector = tool_selector
        self._executor = tool_executor
        self._registrar = tool_registrar
        self._budget_manager = BudgetManager(config.default_tool_budget)
        self._usage_stats: Dict[str, int] = {}
        self._enabled_tools: Optional[set[str]] = None  # None means all tools enabled
        self._mode_controller: Optional[Any] = None
        self._tool_planner: Optional[Any] = None
        self._argument_normalizer: Optional[Any] = None
        self._tool_pipeline: Optional[Any] = None
        self._tool_cache: Optional[Any] = None
        self._retry_executor: Optional[Any] = None
        from victor.tools.tool_call_parser import ToolCallParser
        from victor.tools.tool_call_validator import ToolCallValidator

        self._tool_call_parser: Optional[Any] = ToolCallParser()
        self._tool_call_validator: Optional[Any] = ToolCallValidator()
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

    def _get_budget_limit(self) -> int:
        """Return the active tool budget ceiling."""
        tool_pipeline = self._tool_pipeline
        if tool_pipeline is not None:
            pipeline_budget = getattr(tool_pipeline, "tool_budget", None)
            if isinstance(pipeline_budget, int):
                return max(0, pipeline_budget)
            pipeline_config = getattr(tool_pipeline, "config", None)
            budget = getattr(pipeline_config, "tool_budget", None)
            if isinstance(budget, int):
                return max(0, budget)
        return max(0, self._budget_manager.max_budget)

    def _get_budget_used(self) -> int:
        """Return the active count of tool calls already spent this turn."""
        tool_pipeline = self._tool_pipeline
        if tool_pipeline is not None:
            pipeline_used = getattr(tool_pipeline, "calls_used", None)
            if isinstance(pipeline_used, int):
                return max(0, pipeline_used)
        return max(0, self._budget_manager.calls_made)

    def _sync_budget_manager_from_runtime(self) -> None:
        """Mirror the active runtime budget into the local compatibility manager."""
        self._budget_manager.max_budget = self._get_budget_limit()
        self._budget_manager.calls_made = self._get_budget_used()

    def _set_budget_limit(self, budget: int) -> None:
        """Set the active tool budget ceiling across bound runtime owners."""
        if budget < 0:
            raise ValueError(f"Tool budget must be non-negative: {budget}")

        self._budget_manager.max_budget = budget

        tool_pipeline = self._tool_pipeline
        if tool_pipeline is not None:
            set_tool_budget = getattr(tool_pipeline, "set_tool_budget", None)
            if callable(set_tool_budget):
                set_tool_budget(budget)
            else:
                pipeline_config = getattr(tool_pipeline, "config", None)
                if pipeline_config is not None and hasattr(pipeline_config, "tool_budget"):
                    pipeline_config.tool_budget = budget

    def _consume_budget(self, amount: int = 1) -> None:
        """Record tool usage against the active runtime budget."""
        if amount < 0:
            raise ValueError(f"Cannot consume negative budget: {amount}")

        tool_pipeline = self._tool_pipeline
        if tool_pipeline is not None:
            consume_budget = getattr(tool_pipeline, "consume_budget", None)
            if callable(consume_budget):
                consume_budget(amount)
            else:
                tool_pipeline._calls_used = self._get_budget_used() + amount

        self._budget_manager.record_usage(amount)

    async def select_tools(
        self,
        context: "ToolSelectionContext",
        max_tools: int = 10,
    ) -> List[str]:
        """Select tools based on context.

        Uses the tool selector to analyze the context and select
        the most relevant tools for the task.

        Args:
            context: Tool selection context
            max_tools: Maximum number of tools to select

        Returns:
            List of selected tool names, ordered by relevance

        Raises:
            ToolSelectionError: If tool selection fails
        """
        self._logger.debug(f"Selecting tools (max={max_tools})")

        try:
            # Use selector to choose tools
            selected = await self._selector.select(context, max_tools)

            self._logger.debug(f"Selected {len(selected)} tools: {selected}")
            return selected

        except Exception as e:
            self._logger.error(f"Tool selection failed: {e}")
            # Return empty list on failure for resilience
            return []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """Execute a single tool with validation and budgeting.

        Validates arguments, checks budget, executes the tool,
        and tracks usage.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            ToolResult with execution outcome

        Raises:
            ToolBudgetExceededError: If tool budget is exhausted
            ToolNotFoundError: If tool is not registered
            ToolValidationError: If arguments are invalid
            ToolExecutionError: If tool execution fails
        """
        self._logger.debug(f"Executing tool: {tool_name}")

        # Check budget
        if self.is_budget_exhausted():
            self._logger.warning("Tool budget exhausted")
            raise ToolBudgetExceededError(
                f"Tool budget exhausted ({self.budget_used} / {self.budget})"
            )

        # Check cache if enabled
        if self._config.enable_caching:
            cached = await self._get_cached_result(tool_name, arguments)
            if cached is not None:
                self._logger.debug(f"Using cached result for {tool_name}")
                return cached

        # Execute tool
        try:
            result = await self._executor.execute(tool_name, arguments)

            # Track usage
            self._consume_budget()
            self._track_tool_usage(tool_name, success=True)

            # Cache result if enabled and successful
            if self._config.enable_caching and result.success:
                await self._cache_result(tool_name, arguments, result)

            return result

        except Exception as e:
            self._logger.error(f"Tool execution failed: {tool_name}: {e}")
            self._track_tool_usage(tool_name, success=False)
            raise

    async def execute_tools_parallel(
        self,
        tool_calls: List[tuple[str, Dict[str, Any]]],
        max_parallel: int = 5,
    ) -> AsyncIterator[Any]:
        """Execute multiple tools in parallel.

        Executes independent tools concurrently for improved performance.

        Args:
            tool_calls: List of (tool_name, arguments) tuples
            max_parallel: Maximum number of concurrent executions

        Yields:
            ToolResult objects as they complete

        Raises:
            ToolExecutionError: If any critical tool execution fails
        """
        if not self._config.enable_parallel_execution:
            # Execute sequentially
            for tool_name, arguments in tool_calls:
                yield await self.execute_tool(tool_name, arguments)
            return

        self._logger.debug(f"Executing {len(tool_calls)} tools in parallel")

        # Create semaphore for parallelism limit
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_with_limit(tool_name: str, arguments: Dict[str, Any]):
            async with semaphore:
                return await self.execute_tool(tool_name, arguments)

        # Execute all tools in parallel
        tasks = [execute_with_limit(name, args) for name, args in tool_calls]

        for task in asyncio.as_completed(tasks):
            result = await task
            yield result

    def get_tool_budget(self) -> int:
        """Get the remaining executable tool budget for the current turn.

        Returns:
            Number of remaining tool calls allowed
        """
        return self.get_remaining_budget()

    def set_tool_budget(self, budget: int) -> None:
        """Set the tool budget limit.

        Args:
            budget: Maximum number of tool calls allowed

        Raises:
            ValueError: If budget is negative
        """
        old_max = self.budget
        self._set_budget_limit(budget)
        self._logger.info(f"Tool budget updated: {old_max} -> {budget}")

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        total_calls = sum(self._usage_stats.values())
        successful_calls = sum(
            count for tool, count in self._usage_stats.items() if not tool.startswith("error:")
        )

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 1.0,
            "by_tool": self._usage_stats.copy(),
            "budget_remaining": self.get_tool_budget(),
            "budget_used": self.budget_used,
        }

    def start_new_turn(self) -> None:
        """Reset per-turn budget while preserving cumulative usage stats.

        Interactive chat sessions should replenish executable tool budget for
        each new user prompt, but cross-turn analytics should remain intact.
        """
        tool_pipeline = self._tool_pipeline
        if tool_pipeline is not None:
            start_new_turn = getattr(tool_pipeline, "start_new_turn", None)
            if callable(start_new_turn):
                start_new_turn()
            else:
                tool_pipeline._calls_used = 0
        self._budget_manager.reset()
        self._sync_budget_manager_from_runtime()
        self._logger.debug("Tool turn budget reset")

    def reset_tool_budget(self) -> None:
        """Fully reset tool budget and usage stats to initial state.

        Useful for starting a fresh session, test isolation, or explicit
        operator-driven recovery where cumulative usage analytics should be
        cleared along with the budget.
        """
        self.start_new_turn()
        self._usage_stats.clear()
        self._logger.info("Tool budget reset")

    def process_tool_results(
        self,
        pipeline_result: Any,
        ctx: ToolResultContext,
    ) -> List[Dict[str, Any]]:
        """Post-process tool pipeline results using the canonical service path."""
        return process_tool_results_with_context(pipeline_result, ctx)

    def bind_runtime_components(
        self,
        *,
        tool_registry: Optional[Any] = None,
        tool_pipeline: Optional[Any] = None,
        tool_cache: Optional[Any] = None,
        mode_controller: Optional[Any] = None,
        tool_planner: Optional[Any] = None,
        argument_normalizer: Optional[Any] = None,
        retry_executor: Optional[Any] = None,
        tool_call_parser: Optional[Any] = None,
    ) -> None:
        """Bind live runtime collaborators after bootstrap.

        The service bootstrap path creates the canonical ToolService before
        the full orchestrator runtime is assembled. This hook upgrades the
        service with the real runtime collaborators once they exist.
        """
        if tool_registry is not None:
            self._registrar = tool_registry
        if tool_pipeline is not None:
            self._tool_pipeline = tool_pipeline
            self._sync_budget_manager_from_runtime()
        if tool_cache is not None:
            self._tool_cache = tool_cache
        if mode_controller is not None:
            self._mode_controller = mode_controller
        if tool_planner is not None:
            self._tool_planner = tool_planner
        if argument_normalizer is not None:
            self._argument_normalizer = argument_normalizer
        if retry_executor is not None:
            self._retry_executor = retry_executor
        elif self._tool_pipeline is not None and self._retry_executor is None:
            from victor.agent.services.tool_retry import ToolRetryExecutor

            self._retry_executor = ToolRetryExecutor(
                self._config,
                self._tool_pipeline,
                self._tool_cache,
            )
        if tool_call_parser is not None:
            self._tool_call_parser = tool_call_parser

    # ==========================================================================
    # Budget Properties (for parity with ToolCoordinator)
    # ==========================================================================

    @property
    def budget(self) -> int:
        """Get the current tool budget limit.

        Returns:
            Maximum number of tool calls allowed
        """
        return self._get_budget_limit()

    @budget.setter
    def budget(self, value: int) -> None:
        """Set the tool budget limit.

        Args:
            value: Maximum number of tool calls allowed
        """
        self.set_tool_budget(value)

    @property
    def budget_used(self) -> int:
        """Get the number of tool calls used.

        Returns:
            Number of tool calls made so far
        """
        return self._get_budget_used()

    @property
    def execution_count(self) -> int:
        """Get the number of tool executions.

        Returns:
            Number of tool calls executed
        """
        return self.budget_used

    def set_budget_multiplier(self, multiplier: float) -> None:
        """Set a budget multiplier for scaling the budget.

        Adjusts the budget limit by multiplying the current budget
        by the given multiplier. Useful for scaling operations up or down.

        Args:
            multiplier: Multiplier to apply to current budget
                        (e.g., 2.0 doubles the budget)

        Raises:
            ValueError: If multiplier is negative

        Example:
            service.set_budget_multiplier(2.0)  # Double the budget
            service.set_budget_multiplier(0.5)  # Halve the budget
        """
        if multiplier < 0:
            raise ValueError(f"Budget multiplier must be non-negative: {multiplier}")

        current_budget = self.budget
        new_budget = int(current_budget * multiplier)
        self.budget = new_budget
        self._logger.info(f"Budget multiplied by {multiplier}: {current_budget} -> {new_budget}")

    def consume_budget(self, amount: int = 1) -> None:
        """Consume budget for tool calls.

        Records the specified number of tool calls against the budget.

        Args:
            amount: Number of tool calls to record (default: 1)

        Raises:
            BudgetExhaustedError: If budget is insufficient

        Example:
            # Record 3 tool calls
            service.consume_budget(3)

            # Check if budget exhausted
            if service.is_budget_exhausted():
                # Handle budget exhaustion
                pass
        """
        if amount < 0:
            raise ValueError(f"Cannot consume negative budget: {amount}")

        remaining = self.get_remaining_budget()
        if amount > remaining:
            from victor.core.errors import BudgetExhaustedError

            raise BudgetExhaustedError(
                f"Insufficient budget: need {amount}, have {remaining}",
                budget=self.budget,
                used=self.budget_used,
            )

        self._consume_budget(amount)

    def on_tool_complete(
        self,
        result: Any,
        metrics_collector: Optional[Any] = None,
        *,
        read_files_session: Optional[Set[str]] = None,
        required_files: Optional[List[str]] = None,
        required_outputs: Optional[List[str]] = None,
        nudge_sent_flag: Optional[List[bool]] = None,
        add_message: Optional[Callable[[str, str], None]] = None,
        observability: Optional[Any] = None,
        iteration_count: int = 0,
        tool_name: Optional[str] = None,
        elapsed: float = 0.0,
        session_id: Optional[str] = None,
    ) -> None:
        """Handle tool completion through the canonical service path.

        When ``metrics_collector`` is provided this uses the existing shared
        ``ToolObservabilityHandler`` implementation, which preserves file-read
        nudges, tool completion events, and UI preview behavior. When only the
        lightweight hook arguments are provided, it degrades to the narrower
        observability callback for backward compatibility.
        """
        if metrics_collector is not None:
            from victor.agent.services.tool_observability import ToolObservabilityHandler

            ToolObservabilityHandler(self).on_tool_complete(
                result=result,
                metrics_collector=metrics_collector,
                read_files_session=read_files_session,
                required_files=required_files,
                required_outputs=required_outputs,
                nudge_sent_flag=nudge_sent_flag,
                add_message=add_message,
                observability=observability,
                iteration_count=iteration_count,
            )
            return

        effective_observability = observability or getattr(self, "_observability", None)
        effective_tool_name = tool_name or getattr(result, "tool_name", "")
        if effective_observability is not None and hasattr(
            effective_observability,
            "on_tool_complete",
        ):
            try:
                effective_observability.on_tool_complete(
                    effective_tool_name,
                    result,
                    elapsed,
                    session_id,
                )
            except Exception:
                logger.debug("on_tool_complete observability hook failed", exc_info=True)

    def is_healthy(self) -> bool:
        """Check if the tool service is healthy.

        Returns:
            True if the service is healthy
        """
        return (
            self._selector is not None
            and self._executor is not None
            and not self._budget_manager.is_exhausted()
        )

    # ==========================================================================
    # Tool Enable/Disable Management
    # ==========================================================================

    def set_enabled_tools(self, tools: set[str]) -> None:
        """Set which tools are enabled for this session.

        When tools are explicitly set, only those tools can be used.
        Setting to None or empty set enables all available tools.

        Args:
            tools: Set of tool names to enable, or None for all tools

        Example:
            service.set_enabled_tools({"read", "write", "search"})
            # Only these tools can be used

            service.set_enabled_tools(set())
            # All tools enabled (default)
        """
        self._enabled_tools = tools if tools else None
        self._logger.info(f"Enabled tools: {sorted(tools) if tools else 'all'}")

        # Propagate to selector if it supports filtering
        if self._selector and hasattr(self._selector, "set_enabled_tools"):
            self._selector.set_enabled_tools(tools or set())
            self._logger.debug("Propagated enabled tools to selector")

    def get_enabled_tools(self) -> set[str]:
        """Get currently enabled tool names.

        Returns:
            Set of enabled tool names. If None or empty set returned,
            all tools are considered enabled.

        Example:
            enabled = service.get_enabled_tools()
            if "shell" in enabled:
                # Shell tool is available
        """
        if self._enabled_tools is not None:
            return self._enabled_tools.copy()

        return self.get_available_tools()

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is enabled, False otherwise

        Example:
            if service.is_tool_enabled("shell"):
                # Shell tool can be used
        """
        # If no explicit filter, all tools are enabled
        if self._enabled_tools is None:
            return True

        # Check against explicit filter
        return tool_name in self._enabled_tools

    # ==========================================================================
    # Tool Alias Resolution
    # ==========================================================================

    def resolve_tool_alias(self, tool_name: str) -> str:
        """Resolve tool alias to canonical name.

        Handles shell variants and other tool aliases by:
        1. Converting to canonical name using tool_names registry
        2. Checking if shell is enabled
        3. Returning the canonical name for non-shell tools

        Args:
            tool_name: Original tool name (may be alias)

        Returns:
            Canonical tool name that should be used

        Example:
            canonical = service.resolve_tool_alias("bash")
            # Returns "shell" if enabled

            canonical = service.resolve_tool_alias("execute_bash")
            # Returns "shell" if enabled
        """
        from victor.tools.tool_names import ToolNames, get_canonical_name

        # Step 1: Get canonical name from registry
        canonical = get_canonical_name(tool_name)

        # Step 2: Define shell aliases that need special handling
        shell_aliases = {
            "shell",
            "run",
            "bash",
            "execute",
            "cmd",
            "execute_bash",
        }

        # Also check ToolNames constants if available
        try:
            if hasattr(ToolNames, "SHELL"):
                shell_aliases.add(str(ToolNames.SHELL))
        except Exception:
            pass  # ToolNames not fully available, use string aliases

        # Step 3: If not a shell alias, return canonical name directly
        if canonical not in shell_aliases and tool_name not in shell_aliases:
            if canonical != tool_name:
                self._logger.debug(f"Resolved '{tool_name}' to canonical '{canonical}'")
            return canonical

        # Step 4: Handle shell aliases - check if shell is enabled
        try:
            shell_canonical = str(ToolNames.SHELL) if hasattr(ToolNames, "SHELL") else "shell"
        except Exception:
            shell_canonical = "shell"

        # Check if shell is enabled
        if self.is_tool_enabled(shell_canonical):
            self._logger.debug(f"Resolved '{tool_name}' to '{shell_canonical}' (shell enabled)")
            return shell_canonical

        # Shell not enabled - return canonical name (will fail validation later)
        self._logger.debug(
            f"Shell tool not enabled for '{tool_name}', using canonical '{canonical}'"
        )
        return canonical

    # ==========================================================================
    # Tool Statistics and Budget Queries
    # ==========================================================================

    def get_tool_call_count(self, tool_name: str) -> int:
        """Get the number of times a specific tool was called.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of successful calls (errors counted separately)

        Example:
            count = service.get_tool_call_count("read")
            # Returns: 10
        """
        return self._usage_stats.get(tool_name, 0)

    def get_tool_error_count(self, tool_name: str) -> int:
        """Get the number of errors for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of error occurrences

        Example:
            errors = service.get_tool_error_count("shell")
            # Returns: 2
        """
        error_key = f"error:{tool_name}"
        return self._usage_stats.get(error_key, 0)

    def get_remaining_budget(self) -> int:
        """Get remaining tool budget.

        Returns the number of tool calls remaining before budget exhaustion.

        Returns:
            Remaining budget count

        Example:
            remaining = service.get_remaining_budget()
            # Returns: 85 (out of 100)
        """
        return max(0, self.budget - self.budget_used)

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted.

        Returns:
            True if no more tool calls allowed, False otherwise

        Example:
            if service.is_budget_exhausted():
                # Stop tool execution
        """
        return self.get_remaining_budget() <= 0

    def get_budget_info(self) -> Dict[str, int]:
        """Get detailed budget information.

        Returns:
            Dictionary with budget details (max, used, remaining)

        Example:
            info = service.get_budget_info()
            # {"max": 100, "used": 15, "remaining": 85}
        """
        return {
            "max": self.budget,
            "used": self.budget_used,
            "remaining": self.get_remaining_budget(),
        }

    def get_available_tools(self) -> set[str]:
        """Get all available tools from the registrar.

        Returns a set of tool names that are registered and available.
        Returns empty set if registrar not available.

        Returns:
            Set of available tool names

        Example:
            tools = service.get_available_tools()
            # {"read", "write", "search", "shell", ...}
        """
        if not self._registrar:
            return set()

        try:
            if hasattr(self._registrar, "get_registered_tools"):
                return set(self._registrar.get_registered_tools())
            if hasattr(self._registrar, "get_tool_names"):
                return set(self._registrar.get_tool_names())
            if hasattr(self._registrar, "list_tools"):
                names: set[str] = set()
                for tool in self._registrar.list_tools():
                    if isinstance(tool, str):
                        names.add(tool)
                        continue
                    tool_name = getattr(tool, "name", None)
                    if isinstance(tool_name, str) and tool_name:
                        names.add(tool_name)
                return names
        except Exception as e:
            self._logger.warning(f"Failed to get available tools: {e}")
        return set()

    def build_tool_access_context(self) -> Any:
        """Build the canonical access-control context for tool gating."""
        from victor.agent.protocols import ToolAccessContext

        return ToolAccessContext(
            session_enabled_tools=self._enabled_tools,
            current_mode=(self._mode_controller.config.name if self._mode_controller else None),
        )

    def validate_tool_call(
        self,
        tool_call: Any,
        sanitizer: Any,
        is_tool_enabled_fn: Optional[Callable[[str], bool]] = None,
    ) -> Any:
        """Validate a tool call's structure, name, and enabled status."""
        from victor.agent.services.tool_contracts import ToolCallValidation

        _is_enabled = is_tool_enabled_fn or self.is_tool_enabled
        if not isinstance(tool_call, dict):
            return ToolCallValidation(
                valid=False,
                skip_reason="Skipping malformed tool call payload",
            )

        tool_name = str(tool_call.get("name", "")).strip()
        if not tool_name:
            return ToolCallValidation(
                valid=False,
                skip_reason="Skipping tool call with no name",
            )

        canonical = self.resolve_tool_alias(tool_name)
        available_tools = self.get_available_tools()
        enabled = _is_enabled(canonical)

        if not sanitizer.is_valid_tool_name(tool_name) and not (
            enabled and canonical in available_tools
        ):
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

        if not enabled:
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
                        f"Tool '{tool_name}' is not available. It may be disabled, not "
                        "registered, or not included in the current tool selection. "
                        "Use only the tools listed in your available tools."
                    ),
                },
            )

        validator = self._tool_call_validator
        if validator is not None and canonical in validator._tool_schemas:
            schema_result = validator.validate(canonical, tool_call.get("arguments", {}))
            if not schema_result.valid:
                self._logger.warning("Schema validation: %s: %s", canonical, schema_result.errors)

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
    ) -> Any:
        """Normalize tool arguments through parsing, repair, adapter defaults, and dedup."""
        from victor.agent.services.tool_contracts import NormalizedArgs
        from victor.agent.orchestrator_utils import infer_git_operation

        _failed = failed_signatures if failed_signatures is not None else set()

        tool_args = raw_args
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except Exception:
                try:
                    tool_args = ast.literal_eval(tool_args)
                except Exception:
                    tool_args = {"value": tool_args}
        elif tool_args is None:
            tool_args = {}

        normalized_args, strategy = argument_normalizer.normalize_arguments(tool_args, tool_name)
        normalized_args = tool_adapter.normalize_arguments(normalized_args, tool_name)
        normalized_args = infer_git_operation(original_name, tool_name, normalized_args)

        try:
            signature = (
                tool_name,
                json.dumps(normalized_args, sort_keys=True, default=str),
            )
        except Exception:
            signature = (tool_name, str(normalized_args))

        return NormalizedArgs(
            args=normalized_args,
            strategy=strategy,
            signature=signature,
            is_repeated_failure=signature in _failed,
        )

    # ==========================================================================
    # Tool Call Validation
    # ==========================================================================

    def _validate_tool_call(
        self,
        tool_call: Dict[str, Any],
        available_tools: Optional[set[str]] = None,
    ) -> tuple[bool, Optional[str]]:
        """Internal: Validate a single tool call.

        Checks if the tool call is valid by verifying:
        - Tool name is present
        - Tool is available/enabled
        - Required fields are present

        Args:
            tool_call: Tool call dictionary with 'name' and 'arguments'
            available_tools: Set of available tool names (uses get_available_tools if None)

        Returns:
            Tuple of (is_valid, error_message)
        """

        def _normalize_tools(tool_names: set[str]) -> set[str]:
            return {canonicalize_core_tool_name(name) for name in tool_names}

        # Check required fields
        if not isinstance(tool_call, dict):
            return False, "Tool call must be a dictionary"

        tool_name = tool_call.get("name") or tool_call.get("tool")
        if not tool_name:
            return False, "Tool call missing 'name' field"

        canonical_name = canonicalize_core_tool_name(self.resolve_tool_alias(str(tool_name)))

        # Check if tool is available
        if available_tools is None:
            available_tools = self.get_enabled_tools()

        normalized_available = _normalize_tools(available_tools)

        if canonical_name not in normalized_available:
            registered_tools = _normalize_tools(self.get_available_tools())
            available_preview = ", ".join(sorted(normalized_available)[:8]) or "(none)"

            if canonical_name not in registered_tools:
                return (
                    False,
                    f"Tool '{tool_name}' is not available because it is not registered. "
                    f"Available tools: {available_preview}",
                )

            if self._mode_controller is not None:
                mode_name = self._mode_controller.current_mode.value.upper()
                sandbox_dir = getattr(self._mode_controller.config, "sandbox_dir", None)
                if (
                    canonical_name in {"edit", "write"}
                    and not getattr(self._mode_controller.config, "allow_all_tools", False)
                    and getattr(self._mode_controller.config, "allow_sandbox_edits", False)
                    and sandbox_dir
                ):
                    return (
                        False,
                        f"Tool '{tool_name}' is not available because it resolves to "
                        f"'{canonical_name}', which is limited to {sandbox_dir} in {mode_name} "
                        f"mode. Available tools: {available_preview}",
                    )
                if not self._mode_controller.is_tool_allowed(canonical_name):
                    return (
                        False,
                        f"Tool '{tool_name}' is not available because it resolves to "
                        f"'{canonical_name}', which is not allowed in {mode_name} mode. "
                        f"Available tools: {available_preview}",
                    )

            if self._enabled_tools is not None and canonical_name not in _normalize_tools(
                self._enabled_tools
            ):
                return (
                    False,
                    f"Tool '{tool_name}' is not available because it resolves to "
                    f"'{canonical_name}', which is not enabled for this session. "
                    f"Available tools: {available_preview}",
                )

            return (
                False,
                f"Tool '{tool_name}' is not available because it resolves to "
                f"'{canonical_name}', which is not enabled in the current runtime state. "
                f"Available tools: {available_preview}",
            )

        # Check arguments field
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            return False, "Tool arguments must be a dictionary"

        return True, None

    def validate_tool_calls(
        self,
        tool_calls: Union[Dict[str, Any], List[Dict[str, Any]]],
        available_tools: Optional[set[str]] = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate tool calls (single or multiple).

        This is the canonical method for tool validation. It automatically handles
        both single tool calls and batches of tool calls.

        Splits tool calls into valid and invalid groups. Invalid calls include
        a '_validation_error' field with the error message.

        Args:
            tool_calls: A single tool call dict or a list of tool call dicts.
                Each dict should have 'name' and 'arguments' fields.
            available_tools: Set of available tool names (uses get_available_tools if None)

        Returns:
            Tuple of (valid_calls, invalid_calls). Both are lists, even for single input.

        Examples:
            # Validate a single tool call
            valid, invalid = service.validate_tool_calls(
                {"name": "code_search", "arguments": {"query": "foo"}}
            )
            # valid = [{"name": "code_search", ...}]
            # invalid = []

            # Validate multiple tool calls
            valid, invalid = service.validate_tool_calls([
                {"name": "code_search", "arguments": {"query": "foo"}},
                {"name": "invalid_tool", "arguments": {}}
            ])
            # valid = [{"name": "code_search", ...}]
            # invalid = [{"name": "invalid_tool", ..., "_validation_error": "..."}]
        """
        # Normalize single tool call to list
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        elif not isinstance(tool_calls, list):
            return [], [{"_validation_error": "tool_calls must be a dict or list"}]

        valid_calls = []
        invalid_calls = []

        for call in tool_calls:
            is_valid, error = self._validate_tool_call(call, available_tools)
            if is_valid:
                valid_calls.append(call)
            else:
                # Add error message to invalid call
                invalid_call = dict(call)
                invalid_call["_validation_error"] = error
                invalid_calls.append(invalid_call)

        return valid_calls, invalid_calls

    def filter_hallucinated_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        available_tools: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter out hallucinated (non-existent) tool calls.

        Removes tool calls for tools that don't exist in the available tool set.
        This is useful when LLMs invent tool names.

        Args:
            tool_calls: List of tool call dictionaries
            available_tools: Set of available tool names (uses get_available_tools if None)

        Returns:
            List of valid tool calls (hallucinated ones removed)

        Example:
            filtered = service.filter_hallucinated_tools(tool_calls)
            print(f"Filtered from {len(tool_calls)} to {len(filtered)} calls")
        """
        if available_tools is None:
            available_tools = self.get_available_tools()

        valid_calls = []
        for call in tool_calls:
            tool_name = call.get("name") or call.get("tool")
            if tool_name and tool_name in available_tools:
                valid_calls.append(call)
            else:
                self._logger.debug(f"Filtered hallucinated tool: {tool_name}")

        return valid_calls

    def check_tool_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check if tool arguments are valid.

        Performs basic validation of tool arguments:
        - Checks for None values
        - Checks for empty strings
        - Validates argument types if schema available

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments dictionary

        Returns:
            Tuple of (is_valid, error_message)

        Example:
            is_valid, error = service.check_tool_arguments(
                "code_search", {"query": "test"}
            )
        """
        if not isinstance(arguments, dict):
            return False, "Arguments must be a dictionary"

        # Check for None values in arguments
        for key, value in arguments.items():
            if value is None:
                return False, f"Argument '{key}' is None"

            # Check for empty strings (often indicates missing required param)
            if isinstance(value, str) and not value.strip():
                return False, f"Argument '{key}' is empty"

        # Schema-aware validation can be layered in later. For now we keep
        # validation intentionally conservative to avoid rejecting valid calls.
        _ = tool_name
        _ = schema

        return True, None

    def get_tool_schema(
        self,
        tool_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the schema for a tool.

        Returns the JSON schema for a tool's arguments if available.
        Useful for validation and documentation.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema dictionary or None if not available

        Example:
            schema = service.get_tool_schema("code_search")
            if schema:
                print(f"Required: {schema.get('required', [])}")
        """
        if not self._registrar:
            return None

        try:
            tool = self._registrar.get_tool(tool_name)
            if tool and hasattr(tool, "get_schema"):
                return tool.get_schema()
        except Exception as e:
            self._logger.warning(f"Failed to get schema for {tool_name}: {e}")

        return None

    def parse_tool_calls(
        self,
        response_content: str,
        tool_call_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Parse tool calls from response content.

        Extracts tool calls from LLM response text, handling various
        formats (XML tags, JSON blocks, etc.).

        Args:
            response_content: Response text containing tool calls
            tool_call_id: Optional tool call ID for association

        Returns:
            List of parsed tool call dictionaries

        Example:
            tool_calls = service.parse_tool_calls(
                '<function_calls>...</function_calls>'
            )
        """
        import re
        import json

        tool_calls = []

        # Try XML format: <function_calls><invoke name="tool">...</invoke></function_calls>
        xml_pattern = r'<invoke\s+name="([^"]+)">(.*?)</invoke>'
        for match in re.finditer(xml_pattern, response_content, re.DOTALL):
            tool_name = match.group(1)
            params_xml = match.group(2)

            # Extract parameters
            params = {}
            param_pattern = r'<parameter\s+name="([^"]+)">(.*?)</parameter>'
            for param_match in re.finditer(param_pattern, params_xml, re.DOTALL):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                params[param_name] = param_value

            tool_call = {"name": tool_name, "arguments": params}
            if tool_call_id:
                tool_call["id"] = tool_call_id
            tool_calls.append(tool_call)

        # Try JSON format: [{"name": "tool", "arguments": {...}}]
        if not tool_calls:
            try:
                json_match = re.search(r"\[.*\]", response_content, re.DOTALL)
                if json_match:
                    json_data = json.loads(json_match.group(0))
                    if isinstance(json_data, list):
                        for item in json_data:
                            if isinstance(item, dict) and "name" in item:
                                tool_call = {
                                    "name": item.get("name"),
                                    "arguments": item.get("arguments", item.get("parameters", {})),
                                }
                                if tool_call_id:
                                    tool_call["id"] = tool_call_id
                                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def parse_and_validate_tool_calls(
        self,
        tool_calls: Optional[List[Dict[str, Any]]],
        full_content: str,
        tool_adapter: Any,
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        """Parse, validate, normalize, and filter tool calls from provider output."""
        if not tool_calls and full_content:
            self._logger.debug(
                "No native tool_calls, attempting fallback parsing on content len=%s",
                len(full_content),
            )
            parse_result = tool_adapter.parse_tool_calls(full_content, tool_calls)
            for warning in parse_result.warnings:
                self._logger.warning(f"Tool call parse warning: {warning}")
            if parse_result.tool_calls:
                tool_calls = [tc.to_dict() for tc in parse_result.tool_calls]
                self._logger.debug(
                    "Fallback parser found %s tool calls: %s",
                    len(tool_calls),
                    [tc.get("name") for tc in tool_calls],
                )
                full_content = parse_result.remaining_content
            else:
                self._logger.debug("Fallback parser found no tool calls")

        if tool_calls:
            normalized_tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
            if len(normalized_tool_calls) != len(tool_calls):
                self._logger.warning(f"Dropped non-dict tool_calls: {tool_calls}")
            tool_calls = normalized_tool_calls or None
            self._logger.debug("After normalization: %s tool_calls", len(tool_calls or []))

        if tool_calls:
            valid_tool_calls = []
            invalid_count = 0
            for tc in tool_calls:
                name = tc.get("name", "")
                resolved_name = self.resolve_tool_alias(name)
                if resolved_name != name:
                    tc["name"] = resolved_name
                    name = resolved_name
                is_enabled = self.is_tool_enabled(name)
                self._logger.debug("Tool '%s' enabled=%s", name, is_enabled)
                if is_enabled:
                    valid_tool_calls.append(tc)
                else:
                    invalid_count += 1
                    tc["_invalid"] = True
                    tc["_error"] = f"Unknown tool '{name}'. Use one of the available tools."
                    valid_tool_calls.append(tc)
                    self._logger.debug("Marked invalid tool for error response: %s", name)
            if invalid_count:
                self._logger.warning(
                    "Marked %s invalid tool calls for error responses", invalid_count
                )
            tool_calls = valid_tool_calls or None
            self._logger.debug("After filtering: %s valid tool_calls", len(tool_calls or []))

        parser = self._tool_call_parser
        if parser is None:
            from victor.tools.tool_call_parser import ToolCallParser

            parser = ToolCallParser()
            self._tool_call_parser = parser

        if tool_calls:
            for tc in tool_calls:
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        tc["arguments"] = json.loads(args)
                    except Exception:
                        try:
                            tc["arguments"] = ast.literal_eval(args)
                        except Exception:
                            tc["arguments"] = {"value": args}
                elif args is None:
                    tc["arguments"] = {}

                parsed_args = tc.get("arguments")
                if isinstance(parsed_args, dict):
                    tc["arguments"] = parser.normalize_args(tc.get("name", ""), parsed_args)

        return tool_calls, full_content

    def normalize_tool_call(
        self,
        tool_call: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize a tool call to have consistent structure.

        Ensures tool call has consistent structure:
        - Has 'name' field (not 'tool')
        - Has 'arguments' field (not 'parameters')
        - Removes None values from arguments

        Args:
            tool_call: Raw tool call dictionary

        Returns:
            Normalized tool call dictionary

        Example:
            normalized = service.normalize_tool_call({
                "tool": "code_search",
                "parameters": {"query": "test"}
            })
            # {"name": "code_search", "arguments": {"query": "test"}}
        """
        normalized = {}

        # Normalize tool name field
        tool_name = tool_call.get("name") or tool_call.get("tool")
        if tool_name:
            normalized["name"] = tool_name

        # Normalize arguments field
        arguments = tool_call.get("arguments") or tool_call.get("parameters", {})
        if isinstance(arguments, dict):
            # Remove None values
            normalized["arguments"] = {k: v for k, v in arguments.items() if v is not None}

        # Preserve any other fields
        for key, value in tool_call.items():
            if key not in ("name", "tool", "arguments", "parameters"):
                normalized[key] = value

        return normalized

    # ==========================================================================
    # Mode Controller and Tool Planner
    # ==========================================================================

    def set_mode_controller(self, mode_controller: Any) -> None:
        """Set the mode controller for tool access control.

        The mode controller can restrict tool access based on the current
        mode (e.g., safe mode, research mode, etc.).

        Args:
            mode_controller: Mode controller instance

        Example:
            service.set_mode_controller(mode_controller)
        """
        self._mode_controller = mode_controller
        self._logger.debug(f"Mode controller set: {type(mode_controller).__name__}")

    def set_tool_planner(self, tool_planner: Any) -> None:
        """Set the tool planner for goal-based tool selection.

        The tool planner analyzes tasks and infers goals to guide
        intelligent tool selection.

        Args:
            tool_planner: Tool planner instance

        Example:
            service.set_tool_planner(tool_planner)
        """
        self._tool_planner = tool_planner
        self._logger.debug(f"Tool planner set: {type(tool_planner).__name__}")

    def _pre_filter_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Pre-filter tool calls, removing hallucinated tool names.

        Filters out tool calls that reference tools that don't exist
        in the registry. This prevents execution failures from
        hallucinated tool names.

        Args:
            tool_calls: Raw tool calls from the model

        Returns:
            Tuple of (valid_calls, filtered_names)

        Example:
            valid, filtered = service._pre_filter_tool_calls(tool_calls)
            # valid: List of tool calls with known tool names
            # filtered: List of hallucinated tool names that were filtered
        """
        # Get known tool names
        known = self._get_known_tool_names()
        if not known:
            # No filtering if we can't determine known tools
            return tool_calls, []

        valid = []
        filtered_names = []

        for call in tool_calls:
            name = call.get("name", "")
            if name in known:
                valid.append(call)
            else:
                filtered_names.append(name)
                self._logger.debug(f"Filtered hallucinated tool: {name}")

        if filtered_names:
            self._logger.warning(
                f"Filtered {len(filtered_names)} hallucinated tool(s): {filtered_names}"
            )

        return valid, filtered_names

    def _get_known_tool_names(self) -> set[str]:
        """Get set of known tool names from registrar.

        Returns:
            Set of known tool names, or empty set if registrar unavailable
        """
        if self._registrar is None:
            return set()

        try:
            # Try to get all registered tools
            if hasattr(self._registrar, "get_all_tools"):
                all_tools = self._registrar.get_all_tools()
                return {tool.name for tool in all_tools if hasattr(tool, "name")}
            if hasattr(self._registrar, "get_tool_names"):
                return set(self._registrar.get_tool_names())
            elif hasattr(self._registrar, "list_tools"):
                tool_names = set()
                for tool in self._registrar.list_tools():
                    if isinstance(tool, str):
                        tool_names.add(tool)
                    elif hasattr(tool, "name"):
                        tool_names.add(tool.name)
                return tool_names
            else:
                return set()
        except Exception as e:
            self._logger.debug(f"Failed to get known tool names: {e}")
            return set()

    def set_argument_normalizer(self, normalizer: Any) -> None:
        """Set the argument normalizer for tool arguments.

        Args:
            normalizer: Argument normalizer instance

        Example:
            service.set_argument_normalizer(normalizer)
        """
        self._argument_normalizer = normalizer
        self._logger.debug(f"Argument normalizer set: {type(normalizer).__name__}")

    def normalize_tool_arguments(
        self,
        tool_args: Dict[str, Any],
        tool_name: str,
    ) -> tuple[Dict[str, Any], str]:
        """Normalize tool arguments to handle malformed JSON.

        Args:
            tool_args: Raw arguments from tool call
            tool_name: Name of the tool being called

        Returns:
            Tuple of (normalized_args, strategy_used)

        Example:
            args, strategy = service.normalize_tool_arguments(
                {"query": "test"}, "code_search"
            )
        """
        if not self._argument_normalizer:
            return tool_args, "direct"

        try:
            # Try to use the normalizer
            result = self._argument_normalizer.normalize_arguments(tool_args, tool_name)
            if isinstance(result, tuple):
                return result
            return result, "normalized"
        except Exception as e:
            self._logger.debug(f"Argument normalization failed: {e}")
            return tool_args, "direct"

    async def execute_tool_with_retry(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any],
        tool_executor: Optional[Callable[..., Awaitable[Any]]] = None,
        cache: Optional[Any] = None,
        on_success: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], bool, Optional[str]]:
        """Execute a tool with retry logic and exponential backoff."""
        if self._retry_executor is not None:
            return await self._retry_executor.execute_tool_with_retry(
                tool_name=tool_name,
                tool_args=tool_args,
                context=context,
                tool_executor=tool_executor,
                cache=cache,
                on_success=on_success,
                retry_config=retry_config,
            )

        try:
            effective_executor = tool_executor
            if effective_executor is None and hasattr(self._executor, "execute"):

                async def _default_execute(
                    name: str,
                    arguments: Dict[str, Any],
                    _context: Dict[str, Any],
                ) -> Any:
                    return await self._executor.execute(name, arguments)

                effective_executor = _default_execute

            if effective_executor is None:
                raise RuntimeError("Tool retry execution requires a bound executor")

            result = await effective_executor(tool_name, tool_args, context)
            success = bool(getattr(result, "success", False))
            error = None if success else str(getattr(result, "error", "Unknown error"))
            if success and on_success is not None:
                on_success(tool_name, tool_args, result)
            return result, success, error
        except Exception as exc:
            return None, False, str(exc)

    # ==========================================================================
    # Tool Selection Convenience Methods
    # ==========================================================================

    def can_select_tools(self) -> bool:
        """Check if tool selection is available.

        Returns True if a tool selector is configured and ready.

        Returns:
            True if tool selection is available

        Example:
            if service.can_select_tools():
                tools = await service.select_tools(context)
        """
        return self._selector is not None

    def get_selection_config(self) -> Dict[str, Any]:
        """Get tool selection configuration.

        Returns the current configuration for tool selection
        including max tools, thresholds, etc.

        Returns:
            Dictionary with selection configuration

        Example:
            config = service.get_selection_config()
            print(f"Max tools: {config['max_tools']}")
        """
        return {
            "max_tools": self._config.default_max_tools,
            "has_selector": self._selector is not None,
            "selector_type": type(self._selector).__name__ if self._selector else None,
        }

    async def select_tools_sync(
        self,
        message: str,
        task_type: str = "unknown",
        max_tools: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Select tools with simplified synchronous-style interface.

        Convenience wrapper around select_tools that provides a simpler
        interface for basic tool selection.

        Args:
            message: User message/query
            task_type: Task type (e.g., "edit", "analyze", "debug")
            max_tools: Maximum tools to select (uses config default if None)
            context: Additional context for selection

        Returns:
            List of selected tool names

        Example:
            tools = await service.select_tools_sync(
                message="Search for foo function",
                task_type="analyze",
                max_tools=5
            )
        """
        if not self._selector:
            self._logger.warning("No tool selector configured")
            return []

        # Build context dict
        selection_context: "ToolSelectionContext" = {
            "message": message,
            "task_type": task_type,
            "max_tools": max_tools or self._config.default_max_tools,
        }

        if context:
            selection_context.update(context)

        # Call the async select_tools method
        return await self.select_tools(
            selection_context, max_tools or self._config.default_max_tools
        )

    def estimate_tools_needed(
        self,
        message: str,
        task_type: str = "unknown",
    ) -> int:
        """Estimate number of tools needed for a task.

        Provides a quick heuristic estimate without full selection.
        Useful for budget planning and early filtering.

        Args:
            message: User message/query
            task_type: Task type

        Returns:
            Estimated number of tools needed

        Example:
            estimated = service.estimate_tools_needed(message, "edit")
            if estimated > remaining_budget:
                # Handle budget constraint
        """
        # Simple heuristic based on message length and task type
        word_count = len(message.split())

        # Base estimate on task type
        base_estimate = {
            "edit": 3,
            "analyze": 4,
            "debug": 5,
            "search": 2,
            "refactor": 4,
            "test": 3,
            "unknown": 3,
        }.get(task_type, 3)

        # Adjust based on message complexity
        if word_count > 50:
            base_estimate += 1
        if word_count > 100:
            base_estimate += 1

        # Cap at configured max
        return min(base_estimate, self._config.default_max_tools)

    def get_recommended_tools(
        self,
        task_type: str = "unknown",
    ) -> List[str]:
        """Get recommended tools for a task type.

        Returns a list of commonly used tools for a given task type.
        This is a static recommendation, not based on message content.

        Args:
            task_type: Task type (e.g., "edit", "analyze", "debug")

        Returns:
            List of recommended tool names

        Example:
            tools = service.get_recommended_tools("analyze")
            # ["code_search", "file_read", "grep", ...]
        """
        # Static recommendations based on task type
        recommendations = {
            "edit": ["code_search", "file_read", "file_write", "edit"],
            "analyze": ["code_search", "file_read", "grep", "syntax_check"],
            "debug": ["code_search", "file_read", "grep", "test_runner"],
            "search": ["code_search", "grep", "file_search"],
            "refactor": ["code_search", "file_read", "edit", "syntax_check"],
            "test": ["test_runner", "code_search", "file_read"],
            "build": ["shell", "file_read"],
            "deploy": ["shell", "file_read", "file_write"],
            "unknown": ["code_search", "file_read"],
        }

        # Filter to only available tools
        all_tools = self.get_available_tools()
        recommended = recommendations.get(task_type, recommendations["unknown"])

        return [tool for tool in recommended if tool in all_tools]

    # ==========================================================================
    # Tool Execution Convenience Methods
    # ==========================================================================

    def can_execute_tools(self) -> bool:
        """Check if tool execution is available.

        Returns True if a tool executor is configured and ready.

        Returns:
            True if tool execution is available

        Example:
            if service.can_execute_tools():
                result = await service.execute_tool_call(call)
        """
        return self._executor is not None

    async def execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        validate: bool = True,
        check_budget: bool = True,
    ) -> Dict[str, Any]:
        """Execute a single tool call with validation and budgeting.

        Convenience wrapper that validates, checks budget, and executes
        a single tool call.

        Args:
            tool_call: Tool call dictionary with 'name' and 'arguments'
            validate: Whether to validate tool call before execution
            check_budget: Whether to check budget before execution

        Returns:
            Execution result dictionary with 'result', 'error', 'success' fields

        Example:
            result = await service.execute_tool_call(
                {"name": "code_search", "arguments": {"query": "foo"}}
            )
            if result["success"]:
                print(result["result"])
        """
        result = {
            "tool": tool_call.get("name", "unknown"),
            "success": False,
            "result": None,
            "error": None,
        }

        try:
            # Normalize tool call
            normalized = self.normalize_tool_call(tool_call)
            tool_name = normalized.get("name")
            arguments = normalized.get("arguments", {})

            if not tool_name:
                result["error"] = "Tool name missing"
                return result

            # Validate if requested
            if validate:
                is_valid, error = self._validate_tool_call(normalized)
                if not is_valid:
                    result["error"] = error
                    return result

                # Check arguments
                args_valid, args_error = self.check_tool_arguments(tool_name, arguments)
                if not args_valid:
                    result["error"] = args_error
                    return result

            # Check budget if requested
            if check_budget:
                if self.is_budget_exhausted():
                    result["error"] = "Tool budget exhausted"
                    return result

            # Execute the tool
            if not self._executor:
                result["error"] = "No tool executor configured"
                return result

            # Call the executor's execute_tool method
            execution_result = await self._executor.execute_tool(
                tool_name=tool_name,
                arguments=arguments,
            )

            # Process result
            if execution_result:
                result["success"] = True
                result["result"] = execution_result

                # Track usage
                self._budget_manager.record_usage(1)
                self._track_tool_usage(tool_name, success=True)
            else:
                result["error"] = "Execution returned no result"
                self._track_tool_usage(tool_name, success=False)

        except Exception as e:
            result["error"] = str(e)
            self._track_tool_usage(result["tool"], success=False)

        return result

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        validate: bool = True,
        check_budget: bool = True,
        parallel: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls with validation and budgeting.

        Convenience wrapper that executes multiple tool calls with
        optional validation, budget checking, and parallel execution.

        Args:
            tool_calls: List of tool call dictionaries
            validate: Whether to validate tool calls before execution
            check_budget: Whether to check budget before execution
            parallel: Whether to execute tools in parallel (if budget allows)

        Returns:
            List of execution result dictionaries

        Example:
            results = await service.execute_tool_calls([
                {"name": "code_search", "arguments": {"query": "foo"}},
                {"name": "file_read", "arguments": {"path": "/tmp/test"}},
            ])
            for result in results:
                if result["success"]:
                    print(f"{result['tool']}: {result['result']}")
        """
        if not tool_calls:
            return []

        # Validate all calls first
        valid_calls = []
        validation_errors = []

        if validate:
            valid_calls, invalid_calls = self.validate_tool_calls(tool_calls)
            validation_errors = [
                {
                    "tool": call.get("name", "unknown"),
                    "error": call.get("_validation_error", "Unknown error"),
                }
                for call in invalid_calls
            ]
        else:
            valid_calls = tool_calls

        # Check budget
        if check_budget:
            remaining = self.get_remaining_budget()
            if remaining < len(valid_calls):
                self._logger.warning(
                    f"Insufficient budget: {len(valid_calls)} calls, {remaining} remaining"
                )
                # Truncate to available budget
                valid_calls = valid_calls[:remaining]

        # Execute tools
        if parallel and self._config.enable_parallel_execution:
            # Execute in parallel
            results = await asyncio.gather(
                *[
                    self.execute_tool_call(call, validate=False, check_budget=False)
                    for call in valid_calls
                ],
                return_exceptions=True,
            )

            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(
                        {
                            "tool": valid_calls[i].get("name", "unknown"),
                            "success": False,
                            "result": None,
                            "error": str(result),
                        }
                    )
                else:
                    processed_results.append(result)
            results = processed_results
        else:
            # Execute sequentially
            results = []
            for call in valid_calls:
                result = await self.execute_tool_call(call, validate=False, check_budget=False)
                results.append(result)

        # Add validation errors to results
        all_results = []
        all_results.extend(validation_errors)
        all_results.extend(results)

        return all_results

    async def execute_tool_with_validation(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """Execute a tool with comprehensive validation.

        High-level execution method that validates the tool name,
        normalizes arguments, checks budget, and executes.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            normalize: Whether to normalize arguments

        Returns:
            Execution result dictionary

        Example:
            result = await service.execute_tool_with_validation(
                "code_search",
                {"query": "foo", "path": "/tmp"}
            )
        """
        # Build tool call
        tool_call = {"name": tool_name, "arguments": arguments}

        # Execute with full validation
        return await self.execute_tool_call(
            tool_call,
            validate=True,
            check_budget=True,
        )

    def get_execution_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of tool execution results.

        Analyzes a list of execution results and provides summary
        statistics including success rate, errors, etc.

        Args:
            results: List of execution result dictionaries

        Returns:
            Summary dictionary with statistics

        Example:
            results = await service.execute_tool_calls(calls)
            summary = service.get_execution_summary(results)
            print(f"Success: {summary['success_count']}/{summary['total_count']}")
        """
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        failed = total - successful

        errors = {}
        for result in results:
            error = result.get("error")
            if error:
                errors[error] = errors.get(error, 0) + 1

        return {
            "total_count": total,
            "success_count": successful,
            "failed_count": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "errors": errors,
            "has_errors": failed > 0,
        }

    async def safe_execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        default: Any = None,
    ) -> Any:
        """Execute a tool safely with error handling.

        Executes a tool and returns a default value on error,
        never raises exceptions.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            default: Default value to return on error

        Returns:
            Tool result or default value on error

        Example:
            result = await service.safe_execute_tool(
                "code_search",
                {"query": "foo"},
                default=[]
            )
        """
        try:
            result = await self.execute_tool_with_validation(tool_name, arguments)
            if result.get("success"):
                return result.get("result", default)
            return default
        except Exception:
            return default

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    def _track_tool_usage(self, tool_name: str, success: bool) -> None:
        """Track tool usage for statistics.

        Args:
            tool_name: Name of the tool
            success: Whether execution was successful
        """
        key = tool_name if success else f"error:{tool_name}"
        self._usage_stats[key] = self._usage_stats.get(key, 0) + 1

    async def _get_cached_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[Any]:
        """Get cached tool result if available.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Cached result if available and valid, None otherwise
        """
        # This would integrate with the caching service
        # For now, return None to indicate no cache
        return None

    async def _cache_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
    ) -> None:
        """Cache a tool result.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Result to cache
        """
        # This would integrate with the caching service
        # For now, just log
        self._logger.debug(f"Caching result for {tool_name}")


class ToolBudgetExceededError(Exception):
    """Raised when tool budget is exhausted."""

    pass
