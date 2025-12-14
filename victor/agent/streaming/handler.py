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

"""Streaming chat handler.

Main handler for streaming chat iterations, extracted from AgentOrchestrator
for better testability and separation of concerns.
"""

import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING

from victor.agent.streaming.context import StreamingChatContext
from victor.agent.streaming.iteration import (
    IterationAction,
    IterationResult,
    ProviderResponseResult,
    ToolExecutionResult,
    create_break_result,
    create_continue_result,
    create_force_completion_result,
)
from victor.providers.base import StreamChunk

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class MessageAdder(Protocol):
    """Protocol for adding messages to conversation."""

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        ...


class ToolExecutor(Protocol):
    """Protocol for executing tools."""

    async def execute_tools(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        ...


class StreamingChatHandler:
    """Handler for streaming chat iterations.

    This class encapsulates the core streaming loop logic, making it
    testable independently of the full orchestrator. Dependencies are
    injected via protocols, enabling mocking in tests.
    """

    def __init__(
        self,
        settings: "Settings",
        message_adder: MessageAdder,
        tool_executor: Optional[ToolExecutor] = None,
        session_time_limit: float = 240.0,
    ):
        """Initialize the streaming handler.

        Args:
            settings: Application settings
            message_adder: Object that can add messages to conversation
            tool_executor: Optional tool executor for running tools
            session_time_limit: Maximum session duration in seconds
        """
        self.settings = settings
        self.message_adder = message_adder
        self.tool_executor = tool_executor
        self.session_time_limit = session_time_limit

    def check_time_limit(self, ctx: StreamingChatContext) -> Optional[IterationResult]:
        """Check if session time limit has been exceeded.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult if limit exceeded, None otherwise
        """
        if ctx.is_over_time_limit(self.session_time_limit):
            logger.warning(
                f"Session time limit ({self.session_time_limit}s) reached - "
                f"elapsed={ctx.elapsed_time():.1f}s, forcing completion"
            )
            result = create_force_completion_result(
                f"Session time limit ({self.session_time_limit}s) reached. "
                "Providing summary of progress so far."
            )
            # Add message to force summary
            self.message_adder.add_message(
                "user",
                "TIME LIMIT REACHED. Provide a summary of your findings NOW. "
                "Do NOT call any more tools.",
            )
            ctx.force_completion = True
            return result
        return None

    def check_iteration_limit(
        self, ctx: StreamingChatContext
    ) -> Optional[IterationResult]:
        """Check if iteration limit has been exceeded.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult if limit exceeded, None otherwise
        """
        if ctx.is_over_iteration_limit():
            logger.warning(
                f"Max iterations ({ctx.max_total_iterations}) reached - forcing completion"
            )
            return create_break_result(
                f"\n\n⚠️ Maximum iterations ({ctx.max_total_iterations}) reached.\n"
            )
        return None

    def check_force_completion(
        self, ctx: StreamingChatContext
    ) -> Optional[IterationResult]:
        """Check if force completion conditions are met.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult if should force, None otherwise
        """
        if ctx.should_force_completion():
            logger.info("Force completion triggered")
            return create_force_completion_result("Forcing completion due to constraints.")
        return None

    def handle_blocked_attempts(
        self, ctx: StreamingChatContext
    ) -> Optional[IterationResult]:
        """Handle consecutive blocked tool attempts.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult if force triggered, None otherwise
        """
        force_triggered = ctx.record_blocked_attempt()
        if force_triggered:
            logger.warning(
                f"Multiple blocked attempts ({ctx.consecutive_blocked_attempts}) - "
                "forcing completion"
            )
            result = IterationResult(action=IterationAction.YIELD_AND_CONTINUE)
            result.add_chunk(
                StreamChunk(
                    content="\n[loop] ⚠️ Multiple blocked attempts - forcing completion\n"
                )
            )
            # Add strong instruction to stop tool use
            self.message_adder.add_message(
                "user",
                "⚠️ STOP: You have attempted blocked operations too many times. "
                "You MUST now provide your final response WITHOUT any tool calls. "
                "Summarize what you found and answer the user's question based on "
                "the information you have already gathered. DO NOT call any more tools.",
            )
            ctx.reset_blocked_attempts()
            return result
        return None

    def process_provider_response(
        self, response: ProviderResponseResult, ctx: StreamingChatContext
    ) -> IterationResult:
        """Process the response from the LLM provider.

        Args:
            response: The provider's response
            ctx: The streaming context

        Returns:
            IterationResult with appropriate action
        """
        # Update context with response content
        ctx.accumulate_content(response.content)
        ctx.update_context_message(response.content)

        # Handle garbage detection
        if response.garbage_detected and not response.has_tool_calls:
            ctx.force_completion = True
            logger.info("Setting force_completion due to garbage detection")

        # Create result based on response
        if response.has_tool_calls:
            return create_continue_result(
                content=response.content,
                tool_calls=response.tool_calls,
                tokens=response.tokens_used,
            )
        elif response.has_content:
            # Content without tool calls - may be final or need continuation
            result = IterationResult(
                action=IterationAction.YIELD_AND_CONTINUE,
                content=response.content,
                tokens_used=response.tokens_used,
            )
            result.add_chunk(StreamChunk(content=response.content))
            return result
        else:
            # Empty response
            return create_continue_result()

    def process_tool_results(
        self, execution: ToolExecutionResult, ctx: StreamingChatContext
    ) -> List[StreamChunk]:
        """Process tool execution results and generate status chunks.

        Args:
            execution: The tool execution results
            ctx: The streaming context

        Returns:
            List of StreamChunk objects for tool result status
        """
        chunks = []
        for result in execution.results:
            tool_name = result.get("name", "tool")
            elapsed = result.get("elapsed", 0.0)
            tool_args = result.get("args", {})
            success = result.get("success", False)

            if success:
                chunks.append(
                    StreamChunk(
                        content="",
                        metadata={
                            "tool_result": {
                                "name": tool_name,
                                "success": True,
                                "elapsed": elapsed,
                                "arguments": tool_args,
                            }
                        },
                    )
                )
            else:
                error_msg = result.get("error", "failed")
                chunks.append(
                    StreamChunk(
                        content="",
                        metadata={
                            "tool_result": {
                                "name": tool_name,
                                "success": False,
                                "elapsed": elapsed,
                                "arguments": tool_args,
                                "error": error_msg,
                            }
                        },
                    )
                )

        # Add thinking status
        chunks.append(StreamChunk(content="", metadata={"status": "💭 Thinking..."}))
        return chunks

    def generate_tool_start_chunk(
        self, tool_name: str, tool_args: Dict[str, Any], status_msg: str
    ) -> StreamChunk:
        """Generate a chunk indicating tool execution start.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            status_msg: Status message to display

        Returns:
            StreamChunk with tool start metadata
        """
        return StreamChunk(
            content="",
            metadata={
                "tool_start": {
                    "name": tool_name,
                    "arguments": tool_args,
                    "status_msg": status_msg,
                }
            },
        )

    def should_continue_loop(
        self, result: IterationResult, ctx: StreamingChatContext
    ) -> bool:
        """Determine if the streaming loop should continue.

        Args:
            result: The current iteration result
            ctx: The streaming context

        Returns:
            True if loop should continue, False otherwise
        """
        if result.should_break:
            return False
        if ctx.is_over_iteration_limit():
            return False
        if ctx.should_force_completion() and not result.has_tool_calls:
            return False
        return True

    def check_natural_completion(
        self, ctx: StreamingChatContext, has_tool_calls: bool, content_length: int
    ) -> Optional[IterationResult]:
        """Check if the response represents natural completion (substantial content, no tools).

        When the model has provided substantial content without tool calls after
        an empty response, treat it as natural completion rather than continuing
        to attempt recovery.

        Args:
            ctx: The streaming context
            has_tool_calls: Whether there are pending tool calls
            content_length: Length of current response content

        Returns:
            IterationResult to break if natural completion, None otherwise
        """
        if not has_tool_calls and ctx.has_substantial_content():
            logger.info(
                f"Model returned empty after {ctx.total_accumulated_chars} chars of content - "
                "treating as natural completion (skipping recovery)"
            )
            return create_break_result("")
        return None

    def handle_empty_response(
        self, ctx: StreamingChatContext
    ) -> Optional[IterationResult]:
        """Handle an empty response from the model.

        Tracks consecutive empty responses and forces summary if threshold exceeded.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult if threshold exceeded and summary forced, None otherwise
        """
        threshold_exceeded = ctx.record_empty_response()
        if threshold_exceeded:
            logger.warning(
                f"Model stuck with {ctx.consecutive_empty_responses} consecutive "
                "empty responses - forcing summary"
            )
            result = IterationResult(action=IterationAction.YIELD_AND_CONTINUE)
            result.add_chunk(
                StreamChunk(
                    content="\n[recovery] Forcing summary after repeated empty responses\n"
                )
            )
            # Add strong instruction to summarize
            self.message_adder.add_message(
                "user",
                "You seem to be stuck. Please provide a summary of what you have found so far. "
                "DO NOT call any more tools - just summarize the information you have already gathered.",
            )
            ctx.reset_empty_responses()
            ctx.force_completion = True
            return result
        return None

    def handle_blocked_tool_call(
        self,
        ctx: StreamingChatContext,
        tool_name: str,
        tool_args: Dict[str, Any],
        block_reason: str,
    ) -> StreamChunk:
        """Handle a blocked tool call by recording it and generating feedback.

        Args:
            ctx: The streaming context
            tool_name: Name of the blocked tool
            tool_args: Arguments that were passed
            block_reason: Reason the tool was blocked

        Returns:
            StreamChunk with block notification
        """
        ctx.record_tool_blocked()
        logger.debug(f"BLOCKED tool call: {tool_name}({tool_args}) - {block_reason}")

        # Add tool result feedback
        args_summary = ", ".join(f"{k}={repr(v)[:30]}" for k, v in tool_args.items())
        self.message_adder.add_message(
            "user",
            f"⛔ TOOL BLOCKED: {tool_name}({args_summary})\n\n"
            f"Reason: {block_reason}\n\n"
            "This operation was permanently blocked because you already tried it multiple times. "
            "You MUST use a DIFFERENT approach - this exact operation will NEVER work again.",
        )

        return StreamChunk(content=f"\n[loop] ⛔ {block_reason}\n")

    def check_blocked_threshold(
        self,
        ctx: StreamingChatContext,
        all_blocked: bool,
        consecutive_limit: int = 4,
        total_limit: int = 6,
    ) -> Optional[IterationResult]:
        """Check if blocked tool attempts have exceeded thresholds.

        Args:
            ctx: The streaming context
            all_blocked: Whether all tool calls in this iteration were blocked
            consecutive_limit: Threshold for consecutive blocked attempts
            total_limit: Threshold for total blocked attempts

        Returns:
            IterationResult if force completion triggered, None otherwise
        """
        if all_blocked:
            ctx.consecutive_blocked_attempts += 1
            if ctx.consecutive_blocked_attempts >= consecutive_limit:
                logger.warning(
                    f"Model stuck after {ctx.consecutive_blocked_attempts} consecutive "
                    f"blocked attempts (limit: {consecutive_limit}) - forcing completion"
                )
                return self._create_blocked_force_result(ctx)
        else:
            ctx.reset_blocked_attempts()

        if ctx.total_blocked_attempts >= total_limit:
            logger.warning(
                f"Model stuck after {ctx.total_blocked_attempts} total blocked attempts "
                f"(limit: {total_limit}) - forcing completion"
            )
            return self._create_blocked_force_result(ctx)

        return None

    def _create_blocked_force_result(
        self, ctx: StreamingChatContext
    ) -> IterationResult:
        """Create a force completion result due to blocked attempts.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult with force completion
        """
        result = IterationResult(action=IterationAction.YIELD_AND_CONTINUE)
        result.add_chunk(
            StreamChunk(
                content="\n[loop] ⚠️ Multiple blocked attempts - forcing completion\n"
            )
        )
        self.message_adder.add_message(
            "user",
            "⚠️ STOP: You have attempted blocked operations too many times. "
            "You MUST now provide your final response WITHOUT any tool calls. "
            "Summarize what you found and answer the user's question based on "
            "the information you have already gathered. DO NOT call any more tools.",
        )
        ctx.reset_blocked_attempts()
        # Signal that tool_calls should be cleared
        result.clear_tool_calls = True
        return result

    def handle_force_tool_execution(
        self,
        ctx: StreamingChatContext,
        mentioned_tools: List[str],
        force_message: Optional[str] = None,
    ) -> Optional[IterationResult]:
        """Handle forcing tool execution when model mentions tools without calling them.

        Args:
            ctx: The streaming context
            mentioned_tools: Tools that were mentioned but not executed
            force_message: Optional pre-crafted message to use instead of default

        Returns:
            IterationResult with appropriate action, None if not applicable
        """
        attempt_count = ctx.record_force_tool_attempt()

        if attempt_count >= 3:
            logger.warning(
                "Giving up on forced tool execution after 3 attempts - requesting summary"
            )
            self.message_adder.add_message(
                "user",
                "You are unable to make tool calls. Please provide your response "
                "NOW based on what you know. Do not mention any tools.",
            )
            ctx.reset_force_tool_attempts()
            return create_continue_result()

        # Use provided message or fall back to default
        if force_message:
            message = force_message
        else:
            tools_str = ", ".join(mentioned_tools)
            message = (
                f"You mentioned using {tools_str} but did not actually call the tool(s). "
                "Please make the actual tool call now, or provide your final answer without "
                "mentioning tools you cannot use."
            )

        self.message_adder.add_message("user", message)
        return create_continue_result()

    def check_tool_budget(
        self, ctx: StreamingChatContext, warning_threshold: int = 250
    ) -> Optional[IterationResult]:
        """Check tool budget and generate warning if approaching limit.

        Args:
            ctx: The streaming context
            warning_threshold: Number of tool calls before warning

        Returns:
            IterationResult with warning chunk if approaching limit, None otherwise
        """
        if ctx.is_approaching_budget_limit(warning_threshold):
            result = IterationResult(action=IterationAction.YIELD_AND_CONTINUE)
            result.add_chunk(
                StreamChunk(
                    content=f"[tool] ⚠ Approaching tool budget limit: {ctx.tool_calls_used}/{ctx.tool_budget} calls used\n"
                )
            )
            return result
        return None

    def check_budget_exhausted(self, ctx: StreamingChatContext) -> bool:
        """Check if tool budget is exhausted.

        Args:
            ctx: The streaming context

        Returns:
            True if budget exhausted, False otherwise
        """
        return ctx.is_budget_exhausted()

    def check_progress_and_force(
        self, ctx: StreamingChatContext, base_max_consecutive: int = 8
    ) -> bool:
        """Check progress and set force_completion if stuck.

        Args:
            ctx: The streaming context
            base_max_consecutive: Base limit for consecutive tool calls

        Returns:
            True if force_completion was set, False otherwise
        """
        if ctx.force_completion:
            return False  # Already forcing

        if not ctx.check_progress(base_max_consecutive):
            logger.warning(
                f"Forcing completion: {ctx.tool_calls_used} tool calls but only "
                f"{len(ctx.unique_resources)} unique resources"
            )
            ctx.force_completion = True
            return True
        return False

    def get_budget_exhausted_chunks(self, ctx: StreamingChatContext) -> List[StreamChunk]:
        """Generate chunks for budget exhausted state.

        Args:
            ctx: The streaming context

        Returns:
            List of StreamChunks for budget exhausted warning
        """
        return [
            StreamChunk(
                content=f"[tool] ⚠ Tool budget reached ({ctx.tool_budget}); skipping tool calls.\n"
            ),
            StreamChunk(content="Generating final summary...\n"),
        ]

    def truncate_tool_calls(
        self, tool_calls: List[Dict[str, Any]], ctx: StreamingChatContext
    ) -> List[Dict[str, Any]]:
        """Truncate tool calls to fit remaining budget.

        Args:
            tool_calls: List of tool calls to truncate
            ctx: The streaming context

        Returns:
            Truncated list of tool calls
        """
        remaining = ctx.get_remaining_budget()
        if not tool_calls:
            return []
        return tool_calls[:remaining]


def create_streaming_handler(
    settings: "Settings",
    orchestrator: "AgentOrchestrator",
    session_time_limit: Optional[float] = None,
) -> StreamingChatHandler:
    """Factory function to create a StreamingChatHandler.

    Args:
        settings: Application settings
        orchestrator: The orchestrator (used as message adder)
        session_time_limit: Optional time limit override

    Returns:
        Configured StreamingChatHandler
    """
    limit = session_time_limit or getattr(settings, "session_time_limit", 240.0)
    return StreamingChatHandler(
        settings=settings,
        message_adder=orchestrator,
        tool_executor=None,  # Tool execution stays in orchestrator for now
        session_time_limit=limit,
    )
