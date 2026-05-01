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

"""Chat flow service implementation.

Handles message routing, flow control, and request/response lifecycle
management. This service coordinates the chat flow but delegates
streaming, aggregation, and continuation logic to other services.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from victor.agent.services.chat.protocols import (
        ResponseAggregationServiceProtocol,
        StreamingServiceProtocol,
        ContinuationServiceProtocol,
    )
    from victor.agent.services.protocols import (
        ToolServiceProtocol,
        ContextServiceProtocol,
        ProviderServiceProtocol,
        RecoveryServiceProtocol,
    )
    from victor.agent.conversation.controller import ConversationController
    from victor.providers.base import CompletionResponse

logger = logging.getLogger(__name__)


class ChatFlowServiceConfig:
    """Configuration for ChatFlowService.

    Attributes:
        max_iterations: Maximum agentic loop iterations
        enable_planning: Enable planning mode
        stream_chunk_size: Size of stream chunks
    """

    def __init__(
        self,
        max_iterations: int = 200,
        enable_planning: bool = False,
        stream_chunk_size: int = 100,
    ):
        self.max_iterations = max_iterations
        self.enable_planning = enable_planning
        self.stream_chunk_size = stream_chunk_size


class ChatFlowService:
    """Service for chat flow coordination.

    Responsible for:
    - Message routing through the agentic loop
    - Flow control and iteration management
    - Request/response lifecycle coordination
    - Conversation state management

    This service does NOT handle:
    - Streaming (delegated to StreamingService)
    - Response aggregation (delegated to ResponseAggregationService)
    - Continuation logic (delegated to ContinuationService)

    Example:
        config = ChatFlowServiceConfig()
        service = ChatFlowService(
            config=config,
            provider_service=provider_service,
            tool_service=tool_service,
            context_service=context_service,
            recovery_service=recovery_service,
            conversation_controller=conversation,
        )

        response = await service.chat("Hello, world!")
    """

    def __init__(
        self,
        config: ChatFlowServiceConfig,
        provider_service: "ProviderServiceProtocol",
        tool_service: "ToolServiceProtocol",
        context_service: "ContextServiceProtocol",
        recovery_service: "RecoveryServiceProtocol",
        conversation_controller: "ConversationController",
        response_aggregation_service: Optional["ResponseAggregationServiceProtocol"] = None,
        streaming_service: Optional["StreamingServiceProtocol"] = None,
        continuation_service: Optional["ContinuationServiceProtocol"] = None,
    ):
        """Initialize ChatFlowService.

        Args:
            config: Service configuration
            provider_service: Provider service for LLM calls
            tool_service: Tool service for tool execution
            context_service: Context service for message management
            recovery_service: Recovery service for error handling
            conversation_controller: Conversation controller for state
            response_aggregation_service: Optional response aggregation
            streaming_service: Optional streaming service
            continuation_service: Optional continuation service
        """
        self.config = config
        self.provider_service = provider_service
        self.tool_service = tool_service
        self.context_service = context_service
        self.recovery_service = recovery_service
        self.conversation_controller = conversation_controller
        self.response_aggregation_service = response_aggregation_service
        self.streaming_service = streaming_service
        self.continuation_service = continuation_service

        # Health tracking
        self._healthy = True
        self._error_count = 0
        self._max_errors = 10

    async def chat(self, user_message: str, **kwargs: Any) -> "CompletionResponse":
        """Process a chat message through the agentic loop.

        Args:
            user_message: The user's message
            **kwargs: Additional parameters (tool_budget, temperature, etc.)

        Returns:
            CompletionResponse with the final response

        Raises:
            RuntimeError: If service is unhealthy
            ValueError: If message is invalid
        """
        if not self.is_healthy():
            raise RuntimeError("ChatFlowService is unhealthy and cannot process requests")

        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty")

        try:
            # Add user message to conversation
            self._add_user_message_to_context(user_message, **kwargs)

            # Run agentic loop
            response = await self._run_agentic_loop(user_message, **kwargs)

            # Add assistant response to conversation
            if response.content:
                self._add_assistant_message_to_context(response.content, response)

            self._error_count = 0  # Reset error count on success
            return response

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in chat flow: {e}", exc_info=True)
            raise

    async def _run_agentic_loop(self, user_message: str, **kwargs: Any) -> "CompletionResponse":
        """Run the agentic loop for message processing.

        Args:
            user_message: The user's message
            **kwargs: Additional parameters

        Returns:
            Final CompletionResponse
        """
        from victor.providers.base import CompletionResponse

        iteration = 0
        last_response: Optional[CompletionResponse] = None

        while iteration < self.config.max_iterations:
            iteration += 1

            # Check for continuation logic
            if last_response and self.continuation_service:
                needs_continuation = self.continuation_service.needs_continuation(last_response)
                should_stop = self.continuation_service.should_stop_continuation(
                    iteration, self.config.max_iterations
                )

                if needs_continuation and not should_stop:
                    # Create continuation prompt
                    continuation_prompt = (
                        await self.continuation_service.create_continuation_prompt(last_response)
                    )
                    user_message = continuation_prompt

            # Check context overflow
            if await self.context_service.check_context_overflow():
                await self.context_service.compact_context()

            # Execute turn with tools
            response = await self._execute_turn_with_tools(user_message, **kwargs)

            # Check if response is complete
            if self._is_response_complete(response):
                return response

            last_response = response

        # Return last response if max iterations reached
        return last_response or CompletionResponse(
            content="Max iterations reached without completion",
            stop_reason="max_iterations",
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )

    async def _execute_turn_with_tools(
        self, user_message: str, **kwargs: Any
    ) -> "CompletionResponse":
        """Execute a single turn with potential tool execution.

        Args:
            user_message: The message to process
            **kwargs: Additional parameters

        Returns:
            CompletionResponse from the turn
        """
        # Get messages from context
        messages = self.context_service.get_messages()

        # Call provider
        response = await self.provider_service.chat_completion(messages, **kwargs)

        # Execute tool calls if present
        if self._has_tool_calls(response):
            await self._execute_tool_calls(response.tool_calls)

        return response

    async def _execute_tool_calls(self, tool_calls: list) -> None:
        """Execute tool calls from the response.

        Args:
            tool_calls: List of tool calls to execute
        """
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            try:
                result = await self.tool_service.execute_tool(tool_name, arguments)
                self._add_tool_result_to_context(tool_name, result, tool_call)
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                self._add_tool_result_to_context(tool_name, {"error": str(e)}, tool_call)

    def _is_response_complete(self, response: "CompletionResponse") -> bool:
        """Check if the response is complete.

        Args:
            response: The response to check

        Returns:
            True if response is complete, False otherwise
        """
        # Check stop reason
        if response.stop_reason == "stop":
            return True

        # Check for tool calls (incomplete if tools present)
        if self._has_tool_calls(response):
            return False

        # Check for incomplete content
        if not response.content or len(response.content.strip()) == 0:
            return False

        return True

    def _has_tool_calls(self, response: "CompletionResponse") -> bool:
        """Check if the response has tool calls.

        Args:
            response: The response to check

        Returns:
            True if response has tool calls, False otherwise
        """
        return bool(getattr(response, "tool_calls", None))

    def _add_user_message_to_context(self, message: str, **kwargs: Any) -> None:
        """Add user message to conversation context.

        Args:
            message: The user message
            **kwargs: Additional metadata
        """
        self.context_service.add_message({"role": "user", "content": message, **kwargs})

    def _add_assistant_message_to_context(
        self, content: str, response: "CompletionResponse"
    ) -> None:
        """Add assistant message to conversation context.

        Args:
            content: The assistant's response content
            response: The full completion response
        """
        message = {"role": "assistant", "content": content}

        # Add tool calls if present
        if hasattr(response, "tool_calls") and response.tool_calls:
            message["tool_calls"] = response.tool_calls

        self.context_service.add_message(message)

    def _add_tool_result_to_context(self, tool_name: str, result: Any, tool_call: dict) -> None:
        """Add tool result to conversation context.

        Args:
            tool_name: Name of the tool that was executed
            result: Result from tool execution
            tool_call: Original tool call metadata
        """
        tool_id = tool_call.get("id", "unknown")

        self.context_service.add_message(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": str(result),
            }
        )

    def is_conversation_empty(self) -> bool:
        """Check if the conversation has any messages.

        Returns:
            True if conversation is empty, False otherwise
        """
        messages = self.context_service.get_messages()
        return len(messages) == 0

    def get_message_count(self) -> int:
        """Get the number of messages in the conversation.

        Returns:
            Number of messages
        """
        messages = self.context_service.get_messages()
        return len(messages)

    def get_conversation_stats(self) -> Dict[str, int]:
        """Get conversation statistics.

        Returns:
            Dictionary with conversation metrics
        """
        messages = self.context_service.get_messages()

        stats = {
            "total_messages": len(messages),
            "user_messages": sum(1 for m in messages if m.get("role") == "user"),
            "assistant_messages": sum(1 for m in messages if m.get("role") == "assistant"),
            "system_messages": sum(1 for m in messages if m.get("role") == "system"),
            "tool_messages": sum(1 for m in messages if m.get("role") == "tool"),
        }

        return stats

    def reset_conversation(self) -> None:
        """Reset the conversation state.

        Clears messages while preserving system prompt if configured.
        """
        self.context_service.clear_messages(retain_system=True)

    def is_healthy(self) -> bool:
        """Check if the service is healthy.

        Returns:
            True if service is operational, False otherwise
        """
        if self._error_count >= self._max_errors:
            self._healthy = False
            return False

        # Check dependency services
        if not self.provider_service.is_healthy():
            return False

        if not self.tool_service.is_healthy():
            return False

        if not self.context_service.is_healthy():
            return False

        if not self.recovery_service.is_healthy():
            return False

        return True
