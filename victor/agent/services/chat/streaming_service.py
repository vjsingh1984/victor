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

"""Streaming service implementation.

Handles stream processing, real-time chunk delivery, and stream
state management.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from victor.agent.services.chat.protocols import (
        ResponseAggregationServiceProtocol,
    )
    from victor.agent.services.protocols import (
        ToolServiceProtocol,
        ContextServiceProtocol,
        ProviderServiceProtocol,
    )
    from victor.providers.base import CompletionResponse, StreamChunk

logger = logging.getLogger(__name__)


class StreamingServiceConfig:
    """Configuration for StreamingService.

    Attributes:
        chunk_timeout: Timeout for receiving chunks (seconds)
        enable_chunk_callbacks: Enable callbacks for each chunk
        max_concurrent_streams: Maximum concurrent streaming operations
    """

    def __init__(
        self,
        chunk_timeout: float = 30.0,
        enable_chunk_callbacks: bool = True,
        max_concurrent_streams: int = 5,
    ):
        self.chunk_timeout = chunk_timeout
        self.enable_chunk_callbacks = enable_chunk_callbacks
        self.max_concurrent_streams = max_concurrent_streams


class StreamingService:
    """Service for streaming chat responses.

    Responsible for:
    - Stream processing and real-time chunk delivery
    - Stream state management
    - Chunk callbacks and event handling
    - Stream metrics collection

    This service does NOT handle:
    - Chat flow control (delegated to ChatFlowService)
    - Response aggregation (delegated to ResponseAggregationService)
    - Continuation logic (delegated to ContinuationService)

    Example:
        config = StreamingServiceConfig()
        service = StreamingService(
            config=config,
            provider_service=provider_service,
            context_service=context_service,
            response_aggregation_service=aggregation_service,
        )

        # Stream response
        async for chunk in service.stream_chat("Hello, world!"):
            print(chunk.content)
    """

    def __init__(
        self,
        config: StreamingServiceConfig,
        provider_service: "ProviderServiceProtocol",
        context_service: "ContextServiceProtocol",
        response_aggregation_service: Optional["ResponseAggregationServiceProtocol"] = None,
    ):
        """Initialize StreamingService.

        Args:
            config: Service configuration
            provider_service: Provider service for LLM streaming
            context_service: Context service for message management
            response_aggregation_service: Optional aggregation service
        """
        self.config = config
        self.provider_service = provider_service
        self.context_service = context_service
        self.response_aggregation_service = response_aggregation_service

        # Stream state tracking
        self._active_streams: Dict[str, asyncio.Task] = {}
        self._stream_count = 0
        self._total_chunks_delivered = 0
        self._total_stream_time = 0.0

    async def stream_chat(self, user_message: str, **kwargs: Any) -> AsyncIterator["StreamChunk"]:
        """Stream a chat response in real-time.

        Args:
            user_message: The user's message
            **kwargs: Additional parameters

        Yields:
            StreamChunk objects as they arrive

        Raises:
            RuntimeError: If max concurrent streams exceeded
        """
        if len(self._active_streams) >= self.config.max_concurrent_streams:
            raise RuntimeError(
                f"Maximum concurrent streams ({self.config.max_concurrent_streams}) exceeded"
            )

        # Add user message to context
        self.context_service.add_message({"role": "user", "content": user_message})

        # Get messages for provider
        messages = self.context_service.get_messages()

        # Track stream
        stream_id = f"stream_{self._stream_count}"
        self._stream_count += 1

        start_time = time.time()
        chunks_delivered = 0

        try:
            # Stream from provider
            async for chunk in self.provider_service.stream_completion(messages, **kwargs):
                chunks_delivered += 1
                self._total_chunks_delivered += 1

                # Yield chunk to caller
                yield chunk

                # Handle chunk callback if enabled
                if self.config.enable_chunk_callbacks:
                    await self._handle_stream_chunk(chunk)

        except Exception as e:
            logger.error(f"Error in stream {stream_id}: {e}", exc_info=True)
            raise

        finally:
            # Update metrics
            stream_time = time.time() - start_time
            self._total_stream_time += stream_time

            logger.debug(
                f"Stream {stream_id} completed: {chunks_delivered} chunks " f"in {stream_time:.2f}s"
            )

    async def stream_with_callback(
        self,
        user_message: str,
        callback: Callable[["StreamChunk"], None],
        **kwargs: Any,
    ) -> "CompletionResponse":
        """Stream with a callback function for each chunk.

        Args:
            user_message: The user's message
            callback: Function to call with each chunk
            **kwargs: Additional parameters

        Returns:
            Final aggregated CompletionResponse
        """
        chunks: List["StreamChunk"] = []

        # Stream and collect chunks
        async for chunk in self.stream_chat(user_message, **kwargs):
            chunks.append(chunk)

            # Call user callback
            try:
                callback(chunk)
            except Exception as e:
                logger.error(f"Error in chunk callback: {e}", exc_info=True)

        # Aggregate chunks into response
        if self.response_aggregation_service:
            response = self.response_aggregation_service.aggregate_chunks(chunks)
        else:
            # Fallback aggregation
            from victor.providers.base import CompletionResponse

            content = "".join(c.content for c in chunks if hasattr(c, "content"))
            response = CompletionResponse(
                content=content,
                stop_reason=chunks[-1].stop_reason if chunks else "stop",
                usage=None,
            )

        # Add assistant response to context
        if response.content:
            self.context_service.add_message({"role": "assistant", "content": response.content})

        return response

    async def _handle_stream_chunk(self, chunk: "StreamChunk") -> None:
        """Handle a stream chunk (internal callback).

        Args:
            chunk: The chunk to handle
        """
        # Can be extended for custom chunk handling
        # Currently just logs chunk arrival
        if hasattr(chunk, "content") and chunk.content:
            logger.debug(f"Stream chunk received: {len(chunk.content)} chars")

    def get_streaming_metrics(self) -> Dict[str, Any]:
        """Get streaming performance metrics.

        Returns:
            Dictionary with streaming metrics
        """
        avg_stream_time = (
            self._total_stream_time / self._stream_count if self._stream_count > 0 else 0
        )

        avg_chunks_per_stream = (
            self._total_chunks_delivered / self._stream_count if self._stream_count > 0 else 0
        )

        return {
            "total_streams": self._stream_count,
            "active_streams": len(self._active_streams),
            "total_chunks_delivered": self._total_chunks_delivered,
            "average_chunks_per_stream": avg_chunks_per_stream,
            "total_stream_time": self._total_stream_time,
            "average_stream_time": avg_stream_time,
        }

    def reset_metrics(self) -> None:
        """Reset streaming metrics."""
        self._stream_count = 0
        self._total_chunks_delivered = 0
        self._total_stream_time = 0.0

    async def close_all_streams(self) -> None:
        """Close all active streams.

        Cancels all active streaming tasks.
        """
        for stream_id, task in self._active_streams.items():
            if not task.done():
                task.cancel()
                logger.debug(f"Cancelled stream {stream_id}")

        # Wait for cancellation
        if self._active_streams:
            await asyncio.gather(*self._active_streams.values(), return_exceptions=True)

        self._active_streams.clear()
