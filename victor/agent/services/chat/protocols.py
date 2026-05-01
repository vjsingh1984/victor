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

"""Protocol definitions for decomposed chat services.

Following the Dependency Inversion Principle, these protocols define
the contracts that each decomposed service must implement. This enables:
- Strong typing with mypy
- Dependency injection
- Testability with mocks
- Extensibility through protocol compliance
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional, Protocol

if TYPE_CHECKING:
    from victor.providers.base import CompletionResponse, StreamChunk


class ChatFlowServiceProtocol(Protocol):
    """Protocol for chat flow coordination service.

    Responsible for message routing, flow control, and request/response
    lifecycle management. Does NOT handle streaming or continuation logic.
    """

    async def chat(self, user_message: str, **kwargs: Any) -> CompletionResponse:
        """Process a chat message through the agentic loop.

        Args:
            user_message: The user's message
            **kwargs: Additional parameters (tool_budget, temperature, etc.)

        Returns:
            CompletionResponse with the final response
        """
        ...

    def is_conversation_empty(self) -> bool:
        """Check if the conversation has any messages.

        Returns:
            True if conversation is empty, False otherwise
        """
        ...

    def get_message_count(self) -> int:
        """Get the number of messages in the conversation.

        Returns:
            Number of messages
        """
        ...

    def get_conversation_stats(self) -> Dict[str, int]:
        """Get conversation statistics.

        Returns:
            Dictionary with conversation metrics
        """
        ...

    def reset_conversation(self) -> None:
        """Reset the conversation state.

        Clears messages while preserving system prompt if configured.
        """
        ...

    def is_healthy(self) -> bool:
        """Check if the service is healthy.

        Returns:
            True if service is operational, False otherwise
        """
        ...


class ResponseAggregationServiceProtocol(Protocol):
    """Protocol for response aggregation service.

    Responsible for chunk aggregation, response formatting, and
    multi-provider response consolidation.
    """

    def aggregate_chunks(self, chunks: List[StreamChunk]) -> CompletionResponse:
        """Aggregate stream chunks into a complete response.

        Args:
            chunks: List of stream chunks to aggregate

        Returns:
            CompletionResponse with aggregated content
        """
        ...

    def format_response(self, response: CompletionResponse) -> str:
        """Format a completion response for display.

        Args:
            response: The completion response to format

        Returns:
            Formatted response string
        """
        ...

    def normalize_response(self, response: CompletionResponse) -> CompletionResponse:
        """Normalize response across different providers.

        Args:
            response: The response to normalize

        Returns:
            Normalized CompletionResponse
        """
        ...


class StreamingServiceProtocol(Protocol):
    """Protocol for streaming service.

    Responsible for stream processing, real-time chunk delivery,
    and stream state management.
    """

    async def stream_chat(self, user_message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream a chat response in real-time.

        Args:
            user_message: The user's message
            **kwargs: Additional parameters

        Yields:
            StreamChunk objects as they arrive
        """
        ...

    async def stream_with_callback(
        self,
        user_message: str,
        callback: Callable[[StreamChunk], None],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Stream with a callback function for each chunk.

        Args:
            user_message: The user's message
            callback: Function to call with each chunk
            **kwargs: Additional parameters

        Returns:
            Final aggregated CompletionResponse
        """
        ...

    def get_streaming_metrics(self) -> Dict[str, Any]:
        """Get streaming performance metrics.

        Returns:
            Dictionary with streaming metrics (chunks received, latency, etc.)
        """
        ...


class ContinuationServiceProtocol(Protocol):
    """Protocol for continuation service.

    Responsible for continuation prompt generation, continuation
    decision logic, and multi-turn coordination.
    """

    def needs_continuation(self, response: CompletionResponse) -> bool:
        """Determine if a response needs continuation.

        Args:
            response: The response to evaluate

        Returns:
            True if continuation is needed, False otherwise
        """
        ...

    async def create_continuation_prompt(
        self, response: CompletionResponse, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a continuation prompt for incomplete responses.

        Args:
            response: The incomplete response
            context: Optional context for continuation

        Returns:
            Continuation prompt string
        """
        ...

    def should_stop_continuation(self, iteration: int, max_iterations: int) -> bool:
        """Determine if continuation should stop.

        Args:
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations

        Returns:
            True if should stop, False to continue
        """
        ...

    def get_continuation_count(self) -> int:
        """Get the number of continuation prompts used.

        Returns:
            Number of continuation prompts
        """
        ...
