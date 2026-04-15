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

"""Chat service implementation.

Extracts chat flow coordination from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Chat flow coordination
- Streaming response processing
- Response aggregation
- Integration with tool, context, provider, and recovery services
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

if TYPE_CHECKING:
    from victor.agent.services.protocols import (
        ToolServiceProtocol,
        ContextServiceProtocol,
        ProviderServiceProtocol,
        RecoveryServiceProtocol,
    )
    from victor.providers.base import CompletionResponse, StreamChunk

logger = logging.getLogger(__name__)


class ChatServiceConfig:
    """Configuration for ChatService.

    Attributes:
        max_iterations: Maximum agentic loop iterations
        max_continuation_prompts: Maximum continuation prompts
        stream_chunk_size: Size of stream chunks
        enable_response_caching: Enable response caching
    """

    def __init__(
        self,
        max_iterations: int = 200,
        max_continuation_prompts: int = 3,
        stream_chunk_size: int = 100,
        enable_response_caching: bool = True,
    ):
        self.max_iterations = max_iterations
        self.max_continuation_prompts = max_continuation_prompts
        self.stream_chunk_size = stream_chunk_size
        self.enable_response_caching = enable_response_caching


class ChatService:
    """Service for managing chat operations.

    Extracted from AgentOrchestrator to handle:
    - Chat flow coordination
    - Streaming response processing
    - Response aggregation
    - Agentic loop management

    This service follows SOLID principles:
    - SRP: Only handles chat operations
    - OCP: Extensible through composition
    - LSP: Implements ChatServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on service protocols, not concretions

    Example:
        config = ChatServiceConfig()
        service = ChatService(
            config=config,
            provider_service=provider_service,
            tool_service=tool_service,
            context_service=context_service,
            recovery_service=recovery_service,
            conversation_controller=conversation,
            streaming_coordinator=streaming,
        )

        response = await service.chat("Hello, world!")
    """

    def __init__(
        self,
        config: ChatServiceConfig,
        provider_service: "ProviderServiceProtocol",
        tool_service: "ToolServiceProtocol",
        context_service: "ContextServiceProtocol",
        recovery_service: "RecoveryServiceProtocol",
        conversation_controller: Any,
        streaming_coordinator: Any,
    ):
        """Initialize the chat service.

        Args:
            config: Service configuration
            provider_service: Provider management service
            tool_service: Tool operations service
            context_service: Context management service
            recovery_service: Error recovery service
            conversation_controller: Conversation state controller
            streaming_coordinator: Streaming response coordinator
        """
        self._config = config
        self._provider = provider_service
        self._tools = tool_service
        self._context = context_service
        self._recovery = recovery_service
        self._conversation = conversation_controller
        self._streaming = streaming_coordinator
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

    async def chat(
        self, user_message: str, *, stream: bool = False, **kwargs
    ) -> "CompletionResponse":
        """Process chat message with agentic loop.

        This is the main entry point for chat operations. It coordinates
        the agentic loop which:
        1. Processes the user message
        2. Selects appropriate tools
        3. Executes tools as needed
        4. Generates the final response
        5. Handles recovery from errors

        Args:
            user_message: The user's input message
            stream: If True, return a streaming response
            **kwargs: Additional options (temperature, max_tokens, etc.)

        Returns:
            CompletionResponse with the generated response

        Raises:
            ProviderError: If the provider fails critically
            ToolExecutionError: If tool execution fails critically
            ContextOverflowError: If context exceeds limits
        """
        self._logger.debug(f"Starting chat for message: {user_message[:50]}...")

        try:
            # Add user message to context
            self._add_user_message_to_context(user_message)

            # Check context overflow before processing
            if await self._context.check_context_overflow():
                await self._context.compact_context()

            # Run agentic loop
            if stream:
                # For streaming, we aggregate the stream into a final response
                chunks = []
                async for chunk in self.stream_chat(user_message, **kwargs):
                    chunks.append(chunk)

                # Aggregate chunks into completion response
                return self._aggregate_chunks(chunks)

            # Non-streaming path
            response = await self._run_agentic_loop(user_message, **kwargs)

            # Add assistant response to context
            self._add_assistant_message_to_context(response)

            return response

        except Exception as e:
            self._logger.error(f"Chat failed: {e}")

            # Attempt recovery
            recovery_context = self._create_recovery_context(e)
            if await self._recovery.execute_recovery(recovery_context):
                # Retry after recovery
                return await self.chat(user_message, stream=stream, **kwargs)

            # Recovery failed, re-raise
            raise

    async def stream_chat(self, user_message: str, **kwargs) -> AsyncIterator["StreamChunk"]:
        """Stream chat response in chunks.

        Provides incremental response chunks as they're generated,
        enabling real-time feedback and better UX.

        Args:
            user_message: The user's input message
            **kwargs: Additional options for the chat

        Yields:
            StreamChunk objects with incremental response content

        Raises:
            ProviderError: If the provider fails during streaming
            ToolExecutionError: If tool execution fails critically
        """
        self._logger.debug(f"Starting stream chat for: {user_message[:50]}...")

        try:
            # Run agentic loop with streaming
            async for chunk in self._run_agentic_loop_streaming(user_message, **kwargs):
                yield chunk

        except Exception as e:
            self._logger.error(f"Stream chat failed: {e}")

            # Attempt recovery
            recovery_context = self._create_recovery_context(e)
            if await self._recovery.execute_recovery(recovery_context):
                # Retry after recovery
                async for chunk in self.stream_chat(user_message, **kwargs):
                    yield chunk
                return

            # Recovery failed, re-raise
            raise

    def reset_conversation(self) -> None:
        """Reset the conversation history and state.

        Clears all conversation context, effectively starting
        a new conversation session.
        """
        self._logger.debug("Resetting conversation")
        self._context.clear_messages(retain_system=True)
        self._conversation.reset()

    def is_healthy(self) -> bool:
        """Check if the chat service is healthy.

        A healthy chat service requires:
        - Valid provider connection
        - Tool service available
        - Context service available

        Returns:
            True if the service is healthy, False otherwise
        """
        if not self._provider.is_healthy():
            self._logger.warning("Provider service is unhealthy")
            return False

        if not self._tools.is_healthy():
            self._logger.warning("Tool service is unhealthy")
            return False

        if not self._context.is_healthy():
            self._logger.warning("Context service is unhealthy")
            return False

        return True

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    async def _run_agentic_loop(self, user_message: str, **kwargs) -> "CompletionResponse":
        """Run the agentic loop for chat processing.

        The agentic loop handles tool execution and response generation
        until completion or max iterations is reached.

        Args:
            user_message: The user's input message
            **kwargs: Additional options

        Returns:
            CompletionResponse with the final response
        """
        from victor.providers.base import CompletionResponse

        iterations = 0
        continuation_count = 0

        while iterations < self._config.max_iterations:
            iterations += 1

            # Get messages from context
            messages = self._context.get_messages()

            # Get completion from provider
            response = await self._get_completion(messages, **kwargs)

            # Check if response is complete
            if self._is_response_complete(response):
                return response

            # Check for tool calls
            if self._has_tool_calls(response):
                # Execute tools
                await self._execute_tool_calls(response.tool_calls)

                # Add assistant message with tool calls to context
                self._add_assistant_message_to_context(response)

                # Continue loop for next iteration
                continue

            # Check for continuation needed
            if self._needs_continuation(response):
                continuation_count += 1
                if continuation_count >= self._config.max_continuation_prompts:
                    # Force completion after max continuations
                    break

                # Add continuation prompt
                continuation = await self._create_continuation_prompt(response)
                self._add_user_message_to_context(continuation)
                continue

            # Response is complete
            return response

        # Max iterations reached, return last response
        self._logger.warning(f"Max iterations ({self._config.max_iterations}) reached")
        return response

    # NOTE: _run_agentic_loop_streaming and _get_completion have been removed.
    # The ChatServiceAdapter wraps ChatCoordinator which uses StreamingChatPipeline
    # with full AgenticLoop integration (perception, fulfillment, progress tracking).
    # Raw ChatService methods are not the primary execution path.

    def _is_response_complete(self, response: "CompletionResponse") -> bool:
        """Check if response is complete.

        A response is complete if:
        - It has a stop finish reason
        - It has no tool calls
        - It has substantial content

        Args:
            response: Response to check

        Returns:
            True if response is complete
        """
        if response.stop_reason == "stop":
            return True

        if response.content and len(response.content) > 50:
            return True

        return False

    def _has_tool_calls(self, response: "CompletionResponse") -> bool:
        """Check if response has tool calls.

        Args:
            response: Response to check

        Returns:
            True if response has tool calls
        """
        return bool(response.tool_calls)

    def _needs_continuation(self, response: "CompletionResponse") -> bool:
        """Check if response needs continuation.

        Args:
            response: Response to check

        Returns:
            True if continuation is needed
        """
        return response.stop_reason == "length"

    async def _execute_tool_calls(self, tool_calls: List[Any]) -> None:
        """Execute tool calls from response.

        Args:
            tool_calls: Tool calls to execute
        """
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            result = await self._tools.execute_tool(tool_name, arguments)

            # Add tool result to context
            self._add_tool_result_to_context(tool_name, result)

    async def _create_continuation_prompt(self, response: "CompletionResponse") -> str:
        """Create continuation prompt for incomplete response.

        Args:
            response: Incomplete response

        Returns:
            Continuation prompt
        """
        return "Please continue."

    def _add_user_message_to_context(self, message: str) -> None:
        """Add user message to context.

        Args:
            message: Message content
        """
        msg = {"role": "user", "content": message}
        self._context.add_message(msg)

    def _add_assistant_message_to_context(self, response: "CompletionResponse") -> None:
        """Add assistant message to context.

        Args:
            response: Response to add
        """
        msg = {
            "role": "assistant",
            "content": response.content,
        }
        if hasattr(response, "tool_calls") and response.tool_calls:
            msg["tool_calls"] = response.tool_calls
        self._context.add_message(msg)

    def _add_tool_result_to_context(self, tool_name: str, result: Any) -> None:
        """Add tool result to context.

        Args:
            tool_name: Name of tool that was executed
            result: Tool result
        """
        content = str(result.output) if result.output else str(result.error)
        msg = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_name,
        }
        self._context.add_message(msg)

    def _aggregate_chunks(self, chunks: List["StreamChunk"]) -> "CompletionResponse":
        """Aggregate stream chunks into completion response.

        Args:
            chunks: Chunks to aggregate

        Returns:
            Aggregated CompletionResponse
        """
        from victor.providers.base import CompletionResponse

        content = "".join(chunk.content for chunk in chunks)
        # StreamChunk has optional usage field, handle safely
        total_tokens = sum(
            (chunk.usage.get("total_tokens", 0) if chunk.usage else 0) for chunk in chunks
        )

        return CompletionResponse(
            content=content,
            stop_reason="stop",
            usage={"total_tokens": total_tokens},
        )

    def _create_recovery_context(self, error: Exception) -> Any:
        """Create recovery context from error.

        Args:
            error: Error that occurred

        Returns:
            RecoveryContextImpl with error details
        """
        from victor.agent.services.recovery_service import RecoveryContextImpl

        return RecoveryContextImpl(
            error=error,
            error_type=type(error).__name__,
            attempt_count=1,
            state={},
            metadata={},
        )
