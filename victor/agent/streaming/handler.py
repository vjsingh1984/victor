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
