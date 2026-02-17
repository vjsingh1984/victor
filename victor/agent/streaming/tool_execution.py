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

"""Tool execution handler for streaming chat.

This module provides the ToolExecutionHandler class which encapsulates
the tool execution logic extracted from _stream_chat_impl (P0 SRP refactor).

The handler manages:
- Budget warnings and exhaustion
- Progress checking and force completion
- Tool call filtering and truncation
- Tool execution and result generation
- Reminder injection

Design Pattern: Command + Facade
================================
The handler acts as a facade over the recovery coordinator, chunk generator,
and tool executor, providing a single entry point for tool execution.

Usage:
    handler = ToolExecutionHandler(
        recovery_coordinator=self._recovery_coordinator,
        chunk_generator=self._chunk_generator,
        ...
    )

    result = await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=tool_calls,
        user_message=user_message,
        full_content=full_content,
    )

    for chunk in result.chunks:
        yield chunk

    if result.should_return:
        return
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from victor.agent.streaming.context import StreamingChatContext
from victor.providers.base import StreamChunk

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols for Dependencies
# =============================================================================


class RecoveryCoordinatorProtocol(Protocol):
    """Protocol for recovery coordination."""

    def check_tool_budget(
        self, recovery_ctx: Any, warning_threshold: int
    ) -> Optional[StreamChunk]: ...

    def truncate_tool_calls(
        self, recovery_ctx: Any, tool_calls: List[Dict], remaining: int
    ) -> Tuple[List[Dict], Any]: ...

    def filter_blocked_tool_calls(
        self, recovery_ctx: Any, tool_calls: List[Dict]
    ) -> Tuple[List[Dict], List[StreamChunk], int]: ...

    def check_blocked_threshold(
        self, recovery_ctx: Any, all_blocked: bool
    ) -> Optional[Tuple[StreamChunk, bool]]: ...


class ChunkGeneratorProtocol(Protocol):
    """Protocol for chunk generation."""

    def generate_tool_start_chunk(
        self, tool_name: str, tool_args: Dict, status_msg: str
    ) -> StreamChunk: ...

    def generate_tool_result_chunks(self, result: Dict[str, Any]) -> List[StreamChunk]: ...

    def generate_thinking_status_chunk(self) -> StreamChunk: ...


class MessageAdderProtocol(Protocol):
    """Protocol for adding messages to conversation."""

    def add_message(self, role: str, content: str) -> None: ...


class ReminderManagerProtocol(Protocol):
    """Protocol for reminder management."""

    def update_state(
        self,
        observed_files: Set[str],
        executed_tool: Optional[str],
        tool_calls: int,
    ) -> None: ...

    def get_consolidated_reminder(self) -> Optional[str]: ...


class UnifiedTrackerProtocol(Protocol):
    """Protocol for unified task tracking."""

    @property
    def unique_resources(self) -> Set[str]: ...


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class ToolExecutionResult:
    """Result of tool execution phase.

    Attributes:
        chunks: List of chunks to yield to the stream.
        should_return: Whether the main loop should return (exit).
        tool_results: Results from tool execution.
        tool_calls_executed: Number of tool calls that were executed.
        last_tool_name: Name of the last executed tool (for reminder tracking).
    """

    chunks: List[StreamChunk] = field(default_factory=list)
    should_return: bool = False
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_executed: int = 0
    last_tool_name: Optional[str] = None

    def add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to yield."""
        self.chunks.append(chunk)

    def add_chunks(self, chunks: List[StreamChunk]) -> None:
        """Add multiple chunks to yield."""
        self.chunks.extend(chunks)


# =============================================================================
# Tool Execution Handler
# =============================================================================


class ToolExecutionHandler:
    """Handler for tool execution in streaming chat.

    This class encapsulates the tool execution logic extracted from
    _stream_chat_impl, providing better testability and separation of concerns.

    The handler manages:
    - Budget checks and warnings
    - Progress tracking and force completion
    - Tool call filtering and truncation
    - Actual tool execution
    - Result chunk generation
    - Reminder injection

    Example:
        handler = ToolExecutionHandler(...)
        result = await handler.execute_tools(stream_ctx, tool_calls, ...)

        for chunk in result.chunks:
            yield chunk
        if result.should_return:
            return
    """

    def __init__(
        self,
        recovery_coordinator: RecoveryCoordinatorProtocol,
        chunk_generator: ChunkGeneratorProtocol,
        message_adder: MessageAdderProtocol,
        reminder_manager: ReminderManagerProtocol,
        unified_tracker: UnifiedTrackerProtocol,
        settings: "Settings",
        create_recovery_context: Callable[[StreamingChatContext], Any],
        check_progress_with_handler: Callable[[StreamingChatContext], None],
        handle_force_completion_with_handler: Callable[
            [StreamingChatContext], Optional[StreamChunk]
        ],
        handle_budget_exhausted: Callable[[StreamingChatContext], AsyncIterator[StreamChunk]],
        handle_force_final_response: Callable[[StreamingChatContext], AsyncIterator[StreamChunk]],
        handle_tool_calls: Callable[[List[Dict]], Any],
        get_tool_status_message: Callable[[str, Dict], str],
        observed_files: Optional[Set[str]] = None,
    ):
        """Initialize the tool execution handler.

        Args:
            recovery_coordinator: Coordinator for recovery actions.
            chunk_generator: Generator for stream chunks.
            message_adder: Object that can add messages to conversation.
            reminder_manager: Manager for context reminders.
            unified_tracker: Unified task tracker.
            settings: Application settings.
            create_recovery_context: Factory for recovery context.
            check_progress_with_handler: Callback for progress checking.
            handle_force_completion_with_handler: Callback for force completion.
            handle_budget_exhausted: Async generator for budget exhausted handling.
            handle_force_final_response: Async generator for force final response.
            handle_tool_calls: Callback to execute tool calls.
            get_tool_status_message: Function to generate tool status messages.
            observed_files: Set of observed files (for reminder tracking).
        """
        self._recovery_coordinator = recovery_coordinator
        self._chunk_generator = chunk_generator
        self._message_adder = message_adder
        self._reminder_manager = reminder_manager
        self._unified_tracker = unified_tracker
        self._settings = settings
        self._create_recovery_context = create_recovery_context
        self._check_progress_with_handler = check_progress_with_handler
        self._handle_force_completion_with_handler = handle_force_completion_with_handler
        self._handle_budget_exhausted = handle_budget_exhausted
        self._handle_force_final_response = handle_force_final_response
        self._handle_tool_calls = handle_tool_calls
        self._get_tool_status_message = get_tool_status_message
        self._observed_files = observed_files or set()

    def update_observed_files(self, files: Set[str]) -> None:
        """Update the set of observed files."""
        self._observed_files = files

    async def execute_tools(
        self,
        stream_ctx: StreamingChatContext,
        tool_calls: Optional[List[Dict[str, Any]]],
        user_message: str,
        full_content: str,
        tool_calls_used: int,
        tool_budget: int,
    ) -> ToolExecutionResult:
        """Execute tools and return results.

        This is the main entry point that handles the complete tool execution
        phase including budget checks, filtering, execution, and result generation.

        Args:
            stream_ctx: The streaming context.
            tool_calls: List of tool calls to execute.
            user_message: The original user message.
            full_content: Full content from model response.
            tool_calls_used: Current count of tool calls used.
            tool_budget: Maximum tool call budget.

        Returns:
            ToolExecutionResult with chunks and control flags.
        """
        result = ToolExecutionResult()

        # Sync tool tracking to context
        stream_ctx.tool_calls_used = tool_calls_used
        stream_ctx.tool_budget = tool_budget

        remaining = stream_ctx.get_remaining_budget()

        logger.debug(
            f"Entering tool execution: tool_calls={len(tool_calls) if tool_calls else 0}, "
            f"tool_calls_used={tool_calls_used}/{tool_budget}"
        )

        # Check budget warning
        budget_warning = await self._check_budget_warning(stream_ctx)
        if budget_warning:
            result.add_chunk(budget_warning)

        # Check budget exhausted
        if remaining <= 0:
            exhausted_result = await self._handle_budget_exhausted_phase(stream_ctx)
            result.add_chunks(exhausted_result.chunks)
            result.should_return = True
            return result

        # Sync unique_resources and check progress
        stream_ctx.unique_resources = self._unified_tracker.unique_resources
        self._check_progress_with_handler(stream_ctx)

        # Check force completion
        force_result = await self._check_force_completion(stream_ctx)
        if force_result:
            result.add_chunks(force_result.chunks)
            result.should_return = force_result.should_return
            if result.should_return:
                return result

        # Filter and truncate tool calls
        tool_calls = await self._filter_and_truncate_tools(stream_ctx, tool_calls, result)

        # Execute tools if any remain
        if tool_calls:
            await self._execute_tool_calls(stream_ctx, tool_calls, result)

        # Update context for next iteration
        stream_ctx.update_context_message(full_content or user_message)

        return result

    async def _check_budget_warning(
        self, stream_ctx: StreamingChatContext
    ) -> Optional[StreamChunk]:
        """Check if budget warning should be shown."""
        recovery_ctx = self._create_recovery_context(stream_ctx)
        warning_threshold = getattr(self._settings, "tool_call_budget_warning_threshold", 250)
        return self._recovery_coordinator.check_tool_budget(recovery_ctx, warning_threshold)

    async def _handle_budget_exhausted_phase(
        self, stream_ctx: StreamingChatContext
    ) -> ToolExecutionResult:
        """Handle budget exhausted condition."""
        result = ToolExecutionResult(should_return=True)
        async for chunk in self._handle_budget_exhausted(stream_ctx):
            result.add_chunk(chunk)
        return result

    async def _check_force_completion(
        self, stream_ctx: StreamingChatContext
    ) -> Optional[ToolExecutionResult]:
        """Check if force completion should be triggered."""
        force_chunk = self._handle_force_completion_with_handler(stream_ctx)
        if force_chunk:
            result = ToolExecutionResult(should_return=True)
            result.add_chunk(force_chunk)
            async for final_chunk in self._handle_force_final_response(stream_ctx):
                result.add_chunk(final_chunk)
            return result
        return None

    async def _filter_and_truncate_tools(
        self,
        stream_ctx: StreamingChatContext,
        tool_calls: Optional[List[Dict[str, Any]]],
        result: ToolExecutionResult,
    ) -> List[Dict[str, Any]]:
        """Filter and truncate tool calls based on budget and blocking rules."""
        if not tool_calls:
            return []

        # Truncate to remaining budget
        recovery_ctx = self._create_recovery_context(stream_ctx)
        remaining = stream_ctx.get_remaining_budget()
        tool_calls, _ = self._recovery_coordinator.truncate_tool_calls(
            recovery_ctx, tool_calls, remaining
        )

        # Filter blocked tool calls
        filtered_calls, blocked_chunks, blocked_count = (
            self._recovery_coordinator.filter_blocked_tool_calls(recovery_ctx, tool_calls)
        )
        result.add_chunks(blocked_chunks)

        # Check blocked threshold
        all_blocked = blocked_count > 0 and not filtered_calls
        recovery_ctx = self._create_recovery_context(stream_ctx)
        threshold_result = self._recovery_coordinator.check_blocked_threshold(
            recovery_ctx, all_blocked
        )
        if threshold_result:
            chunk, should_clear = threshold_result
            result.add_chunk(chunk)
            if should_clear:
                filtered_calls = []

        return filtered_calls

    async def _execute_tool_calls(
        self,
        stream_ctx: StreamingChatContext,
        tool_calls: List[Dict[str, Any]],
        result: ToolExecutionResult,
    ) -> None:
        """Execute tool calls and generate result chunks."""
        last_tool_name = None

        # Generate start chunks for each tool
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "tool")
            tool_args = tool_call.get("arguments", {})
            status_msg = self._get_tool_status_message(tool_name, tool_args)
            result.add_chunk(
                self._chunk_generator.generate_tool_start_chunk(tool_name, tool_args, status_msg)
            )
            last_tool_name = tool_name

        # Execute all tool calls
        tool_results = await self._handle_tool_calls(tool_calls)
        result.tool_results = tool_results
        result.tool_calls_executed = len(tool_calls)
        result.last_tool_name = last_tool_name

        # Generate result chunks
        for tool_result in tool_results:
            tool_name = tool_result.get("name", "tool")
            for chunk in self._chunk_generator.generate_tool_result_chunks(tool_result):
                result.add_chunk(chunk)

        # Generate thinking status chunk
        result.add_chunk(self._chunk_generator.generate_thinking_status_chunk())

        # Update reminder manager
        self._reminder_manager.update_state(
            observed_files=self._observed_files,
            executed_tool=last_tool_name,
            tool_calls=stream_ctx.tool_calls_used + len(tool_calls),
        )

        # Get and inject consolidated reminder
        reminder = self._reminder_manager.get_consolidated_reminder()
        if reminder:
            self._message_adder.add_message("system", reminder)


# =============================================================================
# Factory Function
# =============================================================================


def _noop_check_progress(stream_ctx: StreamingChatContext) -> None:
    """No-op fallback for progress checking."""
    pass


def _noop_force_completion(stream_ctx: StreamingChatContext) -> Optional[StreamChunk]:
    """No-op fallback for force completion (returns None = no force)."""
    return None


async def _noop_async_generator(stream_ctx: StreamingChatContext):
    """No-op async generator fallback that yields nothing."""
    return
    yield  # Make it an async generator


def create_tool_execution_handler(
    orchestrator: "AgentOrchestrator",
) -> ToolExecutionHandler:
    """Factory function to create a ToolExecutionHandler from an orchestrator.

    Args:
        orchestrator: The AgentOrchestrator instance.

    Returns:
        Configured ToolExecutionHandler.
    """
    from victor.agent.orchestrator_utils import get_tool_status_message

    return ToolExecutionHandler(
        recovery_coordinator=orchestrator._recovery_coordinator,
        chunk_generator=orchestrator._chunk_generator,
        message_adder=orchestrator,
        reminder_manager=orchestrator.reminder_manager,
        unified_tracker=orchestrator.unified_tracker,
        settings=orchestrator.settings,
        create_recovery_context=orchestrator._create_recovery_context,
        check_progress_with_handler=getattr(
            orchestrator, "_check_progress_with_handler", _noop_check_progress
        ),
        handle_force_completion_with_handler=getattr(
            orchestrator, "_handle_force_completion_with_handler", _noop_force_completion
        ),
        handle_budget_exhausted=getattr(
            orchestrator, "_handle_budget_exhausted", _noop_async_generator
        ),
        handle_force_final_response=getattr(
            orchestrator, "_handle_force_final_response", _noop_async_generator
        ),
        handle_tool_calls=orchestrator._handle_tool_calls,
        get_tool_status_message=get_tool_status_message,
        observed_files=set(orchestrator.observed_files) if orchestrator.observed_files else set(),
    )
