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

"""Response aggregation service implementation.

Handles chunk aggregation from streaming responses, response formatting,
and multi-provider response normalization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.providers.base import CompletionResponse, StreamChunk

logger = logging.getLogger(__name__)


class ResponseAggregationServiceConfig:
    """Configuration for ResponseAggregationService.

    Attributes:
        enable_content_filtering: Enable filtering of duplicate content
        chunk_timeout: Timeout for chunk aggregation (seconds)
        max_chunk_size: Maximum size of aggregated chunks
    """

    def __init__(
        self,
        enable_content_filtering: bool = True,
        chunk_timeout: float = 5.0,
        max_chunk_size: int = 10000,
    ):
        self.enable_content_filtering = enable_content_filtering
        self.chunk_timeout = chunk_timeout
        self.max_chunk_size = max_chunk_size


class ResponseAggregationService:
    """Service for response aggregation and formatting.

    Responsible for:
    - Aggregating stream chunks into complete responses
    - Formatting responses for display
    - Normalizing responses across different providers
    - Multi-provider response consolidation

    This service does NOT handle:
    - Stream processing (delegated to StreamingService)
    - Chat flow control (delegated to ChatFlowService)
    - Continuation logic (delegated to ContinuationService)

    Example:
        config = ResponseAggregationServiceConfig()
        service = ResponseAggregationService(config=config)

        # Aggregate chunks
        chunks = [chunk1, chunk2, chunk3]
        response = service.aggregate_chunks(chunks)

        # Format response
        formatted = service.format_response(response)
    """

    def __init__(self, config: ResponseAggregationServiceConfig):
        """Initialize ResponseAggregationService.

        Args:
            config: Service configuration
        """
        self.config = config
        self._aggregation_count = 0
        self._total_chunks_processed = 0

    def aggregate_chunks(self, chunks: List["StreamChunk"]) -> "CompletionResponse":
        """Aggregate stream chunks into a complete response.

        Args:
            chunks: List of stream chunks to aggregate

        Returns:
            CompletionResponse with aggregated content

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Cannot aggregate empty chunk list")

        self._aggregation_count += 1
        self._total_chunks_processed += len(chunks)

        # Aggregate content from all chunks
        aggregated_content = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        for chunk in chunks:
            # Accumulate content
            if hasattr(chunk, "content") and chunk.content:
                aggregated_content.append(chunk.content)

            # Accumulate usage if present
            if hasattr(chunk, "usage") and chunk.usage:
                total_usage["prompt_tokens"] += chunk.usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += chunk.usage.get("completion_tokens", 0)

        # Join content
        final_content = "".join(aggregated_content)

        # Apply content filtering if enabled
        if self.config.enable_content_filtering:
            final_content = self._filter_duplicate_content(final_content)

        # Get stop reason from last chunk
        stop_reason = "stop"
        if chunks and hasattr(chunks[-1], "stop_reason"):
            stop_reason = chunks[-1].stop_reason or "stop"

        # Create completion response
        from victor.providers.base import CompletionResponse

        response = CompletionResponse(
            content=final_content,
            stop_reason=stop_reason,
            usage=total_usage if total_usage["completion_tokens"] > 0 else None,
        )

        # Copy additional metadata from last chunk
        if chunks and hasattr(chunks[-1], "model"):
            response.model = chunks[-1].model

        if chunks and hasattr(chunks[-1], "id"):
            response.id = chunks[-1].id

        logger.debug(
            f"Aggregated {len(chunks)} chunks into response "
            f"({len(final_content)} chars, {total_usage['completion_tokens']} tokens)"
        )

        return response

    def format_response(self, response: "CompletionResponse") -> str:
        """Format a completion response for display.

        Args:
            response: The completion response to format

        Returns:
            Formatted response string
        """
        if not response or not response.content:
            return ""

        # Basic formatting (can be extended)
        formatted = response.content.strip()

        # Remove trailing whitespace
        formatted = formatted.rstrip()

        return formatted

    def normalize_response(self, response: "CompletionResponse") -> "CompletionResponse":
        """Normalize response across different providers.

        Ensures consistent response structure regardless of which
        provider generated the response.

        Args:
            response: The response to normalize

        Returns:
            Normalized CompletionResponse
        """
        if not response:
            from victor.providers.base import CompletionResponse

            return CompletionResponse(content="", stop_reason="stop", usage=None)

        # Ensure content is string
        if not hasattr(response, "content") or response.content is None:
            response.content = ""

        # Ensure stop_reason is set
        if not hasattr(response, "stop_reason") or response.stop_reason is None:
            response.stop_reason = "stop"

        # Ensure usage is dict
        if not hasattr(response, "usage") or response.usage is None:
            response.usage = {"prompt_tokens": 0, "completion_tokens": 0}

        # Ensure usage has required keys
        if "prompt_tokens" not in response.usage:
            response.usage["prompt_tokens"] = 0
        if "completion_tokens" not in response.usage:
            response.usage["completion_tokens"] = 0

        return response

    def _filter_duplicate_content(self, content: str) -> str:
        """Filter duplicate content from aggregated response.

        Some providers may repeat content in streaming. This method
        removes duplicates while preserving unique content.

        Args:
            content: The content to filter

        Returns:
            Filtered content
        """
        # Simple deduplication (can be enhanced)
        lines = content.split("\n")
        seen = set()
        filtered_lines = []

        for line in lines:
            if line not in seen:
                seen.add(line)
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def get_aggregation_metrics(self) -> Dict[str, Any]:
        """Get aggregation performance metrics.

        Returns:
            Dictionary with aggregation metrics
        """
        return {
            "aggregation_count": self._aggregation_count,
            "total_chunks_processed": self._total_chunks_processed,
            "average_chunks_per_aggregation": (
                self._total_chunks_processed / self._aggregation_count
                if self._aggregation_count > 0
                else 0
            ),
        }

    def reset_metrics(self) -> None:
        """Reset aggregation metrics."""
        self._aggregation_count = 0
        self._total_chunks_processed = 0
