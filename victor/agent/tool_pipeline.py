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

"""Tool Execution Pipeline - Coordinates tool execution flow.

This module extracts tool execution coordination from AgentOrchestrator:
- Tool call validation
- Argument normalization
- Execution via ToolExecutor
- Result tracking and analytics
- Failed signature tracking to avoid loops

Design Principles:
- Single Responsibility: Coordinates tool execution only
- Composable: Works with existing ToolExecutor and ToolRegistry
- Observable: Emits events for monitoring
- Configurable: Budget, retry, and caching settings
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.tool_executor import ToolExecutor
from victor.agent.parallel_executor import (
    ParallelToolExecutor,
    ParallelExecutionConfig,
)

if TYPE_CHECKING:
    from victor.tools.base import ToolRegistry
    from victor.cache.tool_cache import ToolCache
    from victor.agent.code_correction_middleware import CodeCorrectionMiddleware
    from victor.agent.signature_store import SignatureStore

logger = logging.getLogger(__name__)


@dataclass
class ToolPipelineConfig:
    """Configuration for tool pipeline."""

    tool_budget: int = 25
    enable_caching: bool = True
    enable_analytics: bool = True
    enable_failed_signature_tracking: bool = True
    max_tool_name_length: int = 64

    # Code correction settings
    enable_code_correction: bool = False
    code_correction_auto_fix: bool = True

    # Parallel execution settings
    enable_parallel_execution: bool = True
    max_concurrent_tools: int = 5
    parallel_batch_size: int = 10
    parallel_timeout_per_tool: float = 60.0


@dataclass
class ToolCallResult:
    """Result of a single tool call through the pipeline."""

    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    cached: bool = False
    normalization_applied: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None

    # Code correction tracking
    code_corrected: bool = False
    code_validation_errors: Optional[List[str]] = None


@dataclass
class PipelineExecutionResult:
    """Result of executing multiple tool calls."""

    results: List[ToolCallResult] = field(default_factory=list)
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    skipped_calls: int = 0
    total_time_ms: float = 0.0
    budget_exhausted: bool = False
    # Parallel execution metrics
    parallel_execution_used: bool = False
    parallel_speedup: float = 1.0


class ToolPipeline:
    """Coordinates tool execution flow.

    This class handles the complete lifecycle of tool calls:
    1. Validation (name format, tool exists, budget check)
    2. Argument normalization
    3. Deduplication (skip repeated failures)
    4. Execution via ToolExecutor
    5. Result tracking and analytics

    Example:
        pipeline = ToolPipeline(
            tool_registry=registry,
            tool_executor=executor,
            config=ToolPipelineConfig(tool_budget=20),
        )

        result = await pipeline.execute_tool_calls(tool_calls, context)
        for call_result in result.results:
            if call_result.success:
                print(f"{call_result.tool_name}: {call_result.result}")
    """

    # Regex pattern for valid tool names
    VALID_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        tool_executor: ToolExecutor,
        config: Optional[ToolPipelineConfig] = None,
        tool_cache: Optional["ToolCache"] = None,
        argument_normalizer: Optional[ArgumentNormalizer] = None,
        code_correction_middleware: Optional["CodeCorrectionMiddleware"] = None,
        signature_store: Optional["SignatureStore"] = None,
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_complete: Optional[Callable[[ToolCallResult], None]] = None,
    ):
        """Initialize tool pipeline.

        Args:
            tool_registry: Registry of available tools
            tool_executor: Executor for running tools
            config: Pipeline configuration
            tool_cache: Optional cache for tool results
            argument_normalizer: Optional argument normalizer
            code_correction_middleware: Optional middleware for code validation/fixing
            signature_store: Optional persistent storage for failed signatures (cross-session learning)
            on_tool_start: Callback when tool execution starts
            on_tool_complete: Callback when tool execution completes
        """
        self.tools = tool_registry
        self.executor = tool_executor
        self.config = config or ToolPipelineConfig()
        self.tool_cache = tool_cache
        self.normalizer = argument_normalizer or ArgumentNormalizer()
        self.code_correction_middleware = code_correction_middleware
        self.signature_store = signature_store

        # Callbacks
        self.on_tool_start = on_tool_start
        self.on_tool_complete = on_tool_complete

        # State tracking
        self._calls_used = 0
        # In-memory failed signatures for session-only tracking (backward compatible)
        # When signature_store is provided, it's used for persistent cross-session tracking
        self._failed_signatures: Set[tuple] = set()
        self._executed_tools: List[str] = []

        # Analytics
        self._tool_stats: Dict[str, Dict[str, Any]] = {}

        # Parallel executor (lazy initialized)
        self._parallel_executor: Optional[ParallelToolExecutor] = None

    @property
    def calls_used(self) -> int:
        """Number of tool calls used."""
        return self._calls_used

    @property
    def calls_remaining(self) -> int:
        """Number of tool calls remaining in budget."""
        return max(0, self.config.tool_budget - self._calls_used)

    @property
    def executed_tools(self) -> List[str]:
        """List of executed tool names."""
        return list(self._executed_tools)

    @property
    def parallel_executor(self) -> ParallelToolExecutor:
        """Get or create the parallel executor (lazy initialization)."""
        if self._parallel_executor is None:
            parallel_config = ParallelExecutionConfig(
                max_concurrent=self.config.max_concurrent_tools,
                enable_parallel=self.config.enable_parallel_execution,
                batch_size=self.config.parallel_batch_size,
                timeout_per_tool=self.config.parallel_timeout_per_tool,
            )
            self._parallel_executor = ParallelToolExecutor(
                tool_executor=self.executor,
                config=parallel_config,
                progress_callback=self._parallel_progress_callback,
            )
        return self._parallel_executor

    def _parallel_progress_callback(self, tool_name: str, status: str, success: bool) -> None:
        """Progress callback for parallel execution."""
        if status == "started" and self.on_tool_start:
            try:
                self.on_tool_start(tool_name, {})
            except Exception as e:
                logger.warning(f"on_tool_start callback failed: {e}")

    def reset(self) -> None:
        """Reset pipeline state for new conversation."""
        self._calls_used = 0
        self._failed_signatures.clear()
        self._executed_tools.clear()

    def is_valid_tool_name(self, name: str) -> bool:
        """Check if tool name is valid format.

        Args:
            name: Tool name to validate

        Returns:
            True if valid format
        """
        if not name or not isinstance(name, str):
            return False
        if len(name) > self.config.max_tool_name_length:
            return False
        return bool(self.VALID_TOOL_NAME_PATTERN.match(name))

    def _normalize_arguments(
        self, tool_name: str, arguments: Any
    ) -> tuple[Dict[str, Any], NormalizationStrategy]:
        """Normalize tool arguments.

        Args:
            tool_name: Name of the tool
            arguments: Raw arguments (may be string, dict, or None)

        Returns:
            Tuple of (normalized_args, strategy_used)
        """
        # Handle string arguments (from streaming)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                try:
                    arguments = ast.literal_eval(arguments)
                except Exception:
                    arguments = {"value": arguments}
        elif arguments is None:
            arguments = {}

        # Apply normalizer
        return self.normalizer.normalize_arguments(arguments, tool_name)

    def _get_call_signature(self, tool_name: str, args: Dict[str, Any]) -> tuple:
        """Generate signature for deduplication.

        Args:
            tool_name: Tool name
            args: Tool arguments

        Returns:
            Hashable signature tuple
        """
        try:
            args_str = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            args_str = str(args)
        return (tool_name, args_str)

    def is_known_failure(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if a tool call is known to fail.

        Uses persistent SignatureStore when available for cross-session learning,
        falls back to in-memory set for session-only tracking.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            True if this call has failed before
        """
        if not self.config.enable_failed_signature_tracking:
            return False

        # Check persistent store first (cross-session)
        if self.signature_store is not None:
            try:
                if self.signature_store.is_known_failure(tool_name, args):
                    logger.debug(f"Tool call is known failure (persistent): {tool_name}")
                    return True
            except Exception as e:
                logger.warning(f"Signature store check failed: {e}")

        # Fall back to in-memory check (session-only)
        signature = self._get_call_signature(tool_name, args)
        return signature in self._failed_signatures

    def record_failure(self, tool_name: str, args: Dict[str, Any], error_message: str) -> None:
        """Record a failed tool call.

        Uses persistent SignatureStore when available for cross-session learning,
        also records in-memory for current session.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            error_message: Error message from the failure
        """
        if not self.config.enable_failed_signature_tracking:
            return

        # Record in persistent store (cross-session)
        if self.signature_store is not None:
            try:
                self.signature_store.record_failure(tool_name, args, error_message)
                logger.debug(f"Recorded failure to persistent store: {tool_name}")
            except Exception as e:
                logger.warning(f"Failed to record to signature store: {e}")

        # Also record in-memory (session-only, backward compatible)
        signature = self._get_call_signature(tool_name, args)
        self._failed_signatures.add(signature)

    def clear_failure(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Clear a specific failure signature (e.g., after a fix).

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            True if signature was found and cleared
        """
        cleared = False

        # Clear from persistent store
        if self.signature_store is not None:
            try:
                if self.signature_store.clear_signature(tool_name, args):
                    cleared = True
                    logger.debug(f"Cleared failure from persistent store: {tool_name}")
            except Exception as e:
                logger.warning(f"Failed to clear from signature store: {e}")

        # Clear from in-memory
        signature = self._get_call_signature(tool_name, args)
        if signature in self._failed_signatures:
            self._failed_signatures.discard(signature)
            cleared = True

        return cleared

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineExecutionResult:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of tool call requests
            context: Execution context passed to tools

        Returns:
            PipelineExecutionResult with all results
        """
        context = context or {}
        result = PipelineExecutionResult(total_calls=len(tool_calls))
        start_time = time.monotonic()

        for tool_call in tool_calls:
            call_result = await self._execute_single_call(tool_call, context)
            result.results.append(call_result)

            if call_result.skipped:
                result.skipped_calls += 1
            elif call_result.success:
                result.successful_calls += 1
            else:
                result.failed_calls += 1

            # Check if budget exhausted
            if self._calls_used >= self.config.tool_budget:
                result.budget_exhausted = True
                # Skip remaining calls
                break

        result.total_time_ms = (time.monotonic() - start_time) * 1000
        return result

    async def execute_tool_calls_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        force_parallel: bool = False,
    ) -> PipelineExecutionResult:
        """Execute tool calls with parallelization when beneficial.

        Automatically decides whether to use parallel execution based on:
        - Number of tool calls (>1 for parallel)
        - Tool categories (read-only tools can parallelize)
        - Configuration settings

        Args:
            tool_calls: List of tool call requests
            context: Execution context passed to tools
            force_parallel: Override automatic decision (for testing)

        Returns:
            PipelineExecutionResult with parallel metrics
        """
        context = context or {}
        start_time = time.monotonic()

        # Check if parallelization is worthwhile
        should_parallelize = self.config.enable_parallel_execution and (
            force_parallel or len(tool_calls) > 1
        )

        if not should_parallelize:
            # Fall back to sequential execution
            return await self.execute_tool_calls(tool_calls, context)

        # Pre-validate and normalize tool calls
        validated_calls = []
        skipped_results = []

        for tc in tool_calls:
            tool_name = tc.get("name", "")

            # Quick validation checks
            if not tool_name or not self.is_valid_tool_name(tool_name):
                skipped_results.append(
                    ToolCallResult(
                        tool_name=tool_name or "unknown",
                        arguments={},
                        success=False,
                        skipped=True,
                        skip_reason=f"Invalid tool name: {tool_name}",
                    )
                )
                continue

            if not self.tools.is_tool_enabled(tool_name):
                skipped_results.append(
                    ToolCallResult(
                        tool_name=tool_name,
                        arguments={},
                        success=False,
                        skipped=True,
                        skip_reason=f"Unknown or disabled tool: {tool_name}",
                    )
                )
                continue

            # Normalize arguments
            raw_args = tc.get("arguments", {})
            normalized_args, _ = self._normalize_arguments(tool_name, raw_args)

            # Check for repeated failures
            if self.config.enable_failed_signature_tracking:
                signature = self._get_call_signature(tool_name, normalized_args)
                if signature in self._failed_signatures:
                    skipped_results.append(
                        ToolCallResult(
                            tool_name=tool_name,
                            arguments=normalized_args,
                            success=False,
                            skipped=True,
                            skip_reason="Repeated failing call with same arguments",
                        )
                    )
                    continue

            validated_calls.append({"name": tool_name, "arguments": normalized_args})

        # Execute validated calls in parallel
        parallel_result = await self.parallel_executor.execute_parallel(validated_calls, context)

        # Build pipeline result
        result = PipelineExecutionResult(
            total_calls=len(tool_calls),
            parallel_execution_used=True,
            parallel_speedup=parallel_result.parallel_speedup,
        )

        # Add skipped results first
        for skipped in skipped_results:
            result.results.append(skipped)
            result.skipped_calls += 1

        # Convert parallel results to pipeline results
        for exec_result in parallel_result.results:
            call_result = ToolCallResult(
                tool_name=exec_result.tool_name,
                arguments={},  # Arguments already logged
                success=exec_result.success,
                result=exec_result.result,
                error=exec_result.error,
                execution_time_ms=exec_result.execution_time * 1000,
            )
            result.results.append(call_result)

            if exec_result.success:
                result.successful_calls += 1
                self._executed_tools.append(exec_result.tool_name)
            else:
                result.failed_calls += 1
                # Track failed signature
                if self.config.enable_failed_signature_tracking:
                    # We need to get arguments back - use tool name + error as key
                    self._failed_signatures.add((exec_result.tool_name, exec_result.error or ""))

            # Update call count
            self._calls_used += 1

            # Check budget
            if self._calls_used >= self.config.tool_budget:
                result.budget_exhausted = True
                break

        result.total_time_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            f"Parallel pipeline: {len(validated_calls)} tools, "
            f"speedup={parallel_result.parallel_speedup:.2f}x, "
            f"time={result.total_time_ms:.1f}ms"
        )

        return result

    async def _execute_single_call(
        self,
        tool_call: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ToolCallResult:
        """Execute a single tool call.

        Args:
            tool_call: Tool call request
            context: Execution context

        Returns:
            ToolCallResult
        """
        # Validate structure
        if not isinstance(tool_call, dict):
            return ToolCallResult(
                tool_name="unknown",
                arguments={},
                success=False,
                skipped=True,
                skip_reason="Invalid tool call structure (not a dict)",
            )

        tool_name = tool_call.get("name", "")
        raw_args = tool_call.get("arguments", {})

        # Validate tool name
        if not tool_name:
            return ToolCallResult(
                tool_name="unknown",
                arguments={},
                success=False,
                skipped=True,
                skip_reason="Tool call missing name",
            )

        if not self.is_valid_tool_name(tool_name):
            return ToolCallResult(
                tool_name=tool_name,
                arguments={},
                success=False,
                skipped=True,
                skip_reason=f"Invalid tool name format: {tool_name}",
            )

        # Check if tool exists
        if not self.tools.is_tool_enabled(tool_name):
            return ToolCallResult(
                tool_name=tool_name,
                arguments={},
                success=False,
                skipped=True,
                skip_reason=f"Unknown or disabled tool: {tool_name}",
            )

        # Check budget
        if self._calls_used >= self.config.tool_budget:
            return ToolCallResult(
                tool_name=tool_name,
                arguments={},
                success=False,
                skipped=True,
                skip_reason="Tool budget exhausted",
            )

        # Normalize arguments
        normalized_args, strategy = self._normalize_arguments(tool_name, raw_args)
        normalization_applied = None if strategy == NormalizationStrategy.DIRECT else strategy.value

        # Code correction middleware - validate and fix code arguments
        code_corrected = False
        code_validation_errors: Optional[List[str]] = None

        if (
            self.config.enable_code_correction
            and self.code_correction_middleware is not None
            and self.code_correction_middleware.should_validate(tool_name)
        ):
            try:
                correction_result = self.code_correction_middleware.validate_and_fix(
                    tool_name, normalized_args
                )

                if correction_result.was_corrected:
                    # Apply the correction
                    normalized_args = self.code_correction_middleware.apply_correction(
                        normalized_args, correction_result
                    )
                    code_corrected = True
                    logger.info(
                        f"Code auto-corrected for tool '{tool_name}': "
                        f"{len(correction_result.validation.errors)} issues fixed"
                    )

                if not correction_result.validation.valid:
                    # Collect validation errors for feedback
                    code_validation_errors = list(correction_result.validation.errors)
                    logger.warning(
                        f"Code validation errors for tool '{tool_name}': "
                        f"{code_validation_errors}"
                    )
            except Exception as e:
                logger.warning(f"Code correction middleware failed: {e}")

        # Check for repeated failures
        if self.config.enable_failed_signature_tracking:
            signature = self._get_call_signature(tool_name, normalized_args)
            if signature in self._failed_signatures:
                return ToolCallResult(
                    tool_name=tool_name,
                    arguments=normalized_args,
                    success=False,
                    skipped=True,
                    skip_reason="Repeated failing call with same arguments",
                    normalization_applied=normalization_applied,
                )

        # Notify start
        if self.on_tool_start:
            try:
                self.on_tool_start(tool_name, normalized_args)
            except Exception as e:
                logger.warning(f"on_tool_start callback failed: {e}")

        # Execute
        start_time = time.monotonic()
        exec_result = await self.executor.execute(
            tool_name=tool_name,
            arguments=normalized_args,
            context=context,
        )
        execution_time_ms = (time.monotonic() - start_time) * 1000

        # Update state
        self._calls_used += 1
        self._executed_tools.append(tool_name)

        # Build result
        call_result = ToolCallResult(
            tool_name=tool_name,
            arguments=normalized_args,
            success=exec_result.success,
            result=exec_result.result,
            error=exec_result.error,
            execution_time_ms=execution_time_ms,
            normalization_applied=normalization_applied,
            code_corrected=code_corrected,
            code_validation_errors=code_validation_errors,
        )

        # Track failed signatures
        if not exec_result.success and self.config.enable_failed_signature_tracking:
            signature = self._get_call_signature(tool_name, normalized_args)
            self._failed_signatures.add(signature)

        # Update analytics
        if self.config.enable_analytics:
            self._update_analytics(call_result)

        # Notify complete
        if self.on_tool_complete:
            try:
                self.on_tool_complete(call_result)
            except Exception as e:
                logger.warning(f"on_tool_complete callback failed: {e}")

        return call_result

    def _update_analytics(self, result: ToolCallResult) -> None:
        """Update tool analytics.

        Args:
            result: Tool call result
        """
        tool_name = result.tool_name
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time_ms": 0.0,
            }

        stats = self._tool_stats[tool_name]
        stats["calls"] += 1
        if result.success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        stats["total_time_ms"] += result.execution_time_ms

    def get_analytics(self) -> Dict[str, Any]:
        """Get tool execution analytics.

        Returns:
            Dictionary with analytics data
        """
        return {
            "total_calls": self._calls_used,
            "budget": self.config.tool_budget,
            "remaining": self.calls_remaining,
            "tools": dict(self._tool_stats),
        }

    def clear_failed_signatures(self) -> None:
        """Clear the failed signature cache."""
        self._failed_signatures.clear()
