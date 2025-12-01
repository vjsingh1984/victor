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

"""Stream handling for processing LLM streaming responses."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from victor.providers.base import StreamChunk

logger = logging.getLogger(__name__)


@dataclass
class StreamMetrics:
    """Metrics collected during streaming."""

    start_time: float = 0.0
    first_token_time: Optional[float] = None
    end_time: float = 0.0
    total_chunks: int = 0
    total_content_length: int = 0
    tool_calls_count: int = 0

    @property
    def time_to_first_token(self) -> Optional[float]:
        """Time from start to first token (TTFT)."""
        if self.first_token_time and self.start_time:
            return self.first_token_time - self.start_time
        return None

    @property
    def total_duration(self) -> float:
        """Total streaming duration."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def tokens_per_second(self) -> float:
        """Estimated tokens per second (rough, based on content length)."""
        duration = self.total_duration
        if duration > 0:
            # Rough estimate: ~4 chars per token
            estimated_tokens = self.total_content_length / 4
            return estimated_tokens / duration
        return 0.0


@dataclass
class StreamResult:
    """Result from processing a stream."""

    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    error: Optional[str] = None


class StreamHandler:
    """Handles streaming responses from LLM providers.

    Responsibilities:
    - Process stream chunks and accumulate content
    - Handle tool calls in streaming responses
    - Collect streaming metrics (TTFT, throughput)
    - Support callbacks for real-time updates
    - Handle stream interruption and cleanup
    """

    def __init__(
        self,
        on_content: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_complete: Optional[Callable[[StreamResult], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        timeout: float = 300.0,
    ):
        """Initialize stream handler.

        Args:
            on_content: Callback for each content chunk
            on_tool_call: Callback when a tool call is detected
            on_complete: Callback when stream completes
            on_error: Callback on stream error
            timeout: Maximum time to wait for stream (seconds)
        """
        self.on_content = on_content
        self.on_tool_call = on_tool_call
        self.on_complete = on_complete
        self.on_error = on_error
        self.timeout = timeout

        # State
        self._cancelled = False
        self._current_content = ""
        self._tool_calls: List[Dict[str, Any]] = []
        self._pending_tool_call: Dict[str, Any] = {}

    async def process_stream(
        self,
        stream: AsyncIterator[StreamChunk],
    ) -> StreamResult:
        """Process a stream of chunks and return accumulated result.

        Args:
            stream: Async iterator of StreamChunk objects

        Returns:
            StreamResult with accumulated content and metrics
        """
        metrics = StreamMetrics(start_time=time.time())
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        stop_reason: Optional[str] = None
        error: Optional[str] = None

        try:
            # Use wait_for pattern for Python 3.9/3.10 compatibility
            async def _process_chunks() -> None:
                nonlocal stop_reason
                async for chunk in stream:
                    if self._cancelled:
                        logger.info("Stream cancelled by user")
                        break

                    metrics.total_chunks += 1

                    # Handle content
                    if chunk.content:
                        if metrics.first_token_time is None:
                            metrics.first_token_time = time.time()

                        content_parts.append(chunk.content)
                        metrics.total_content_length += len(chunk.content)

                        if self.on_content:
                            try:
                                self.on_content(chunk.content)
                            except Exception as e:
                                logger.warning(f"Content callback error: {e}")

                    # Handle tool calls
                    if chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            tool_calls.append(tc)
                            metrics.tool_calls_count += 1

                            if self.on_tool_call:
                                try:
                                    self.on_tool_call(tc)
                                except Exception as e:
                                    logger.warning(f"Tool call callback error: {e}")

                    # Check for completion
                    if chunk.stop_reason:
                        stop_reason = chunk.stop_reason

                    if chunk.is_final:
                        break

            await asyncio.wait_for(_process_chunks(), timeout=self.timeout)

        except asyncio.TimeoutError:
            error = f"Stream timed out after {self.timeout}s"
            logger.error(error)
            if self.on_error:
                self.on_error(TimeoutError(error))

        except Exception as e:
            error = str(e)
            logger.error(f"Stream error: {e}")
            if self.on_error:
                self.on_error(e)

        finally:
            metrics.end_time = time.time()

        result = StreamResult(
            content="".join(content_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            metrics=metrics,
            error=error,
        )

        if self.on_complete:
            try:
                self.on_complete(result)
            except Exception as e:
                logger.warning(f"Complete callback error: {e}")

        return result

    def cancel(self) -> None:
        """Cancel the current stream processing."""
        self._cancelled = True

    def reset(self) -> None:
        """Reset handler state for reuse."""
        self._cancelled = False
        self._current_content = ""
        self._tool_calls.clear()
        self._pending_tool_call.clear()


class StreamBuffer:
    """Buffer for accumulating streamed tool call arguments.

    Some providers send tool call arguments in chunks. This buffer
    accumulates them until the tool call is complete.
    """

    def __init__(self) -> None:
        self._buffers: Dict[str, Dict[str, Any]] = {}

    def add_chunk(
        self,
        tool_call_id: str,
        chunk: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Add a chunk to the buffer.

        Args:
            tool_call_id: ID of the tool call
            chunk: Partial tool call data

        Returns:
            Complete tool call if finished, None otherwise
        """
        if tool_call_id not in self._buffers:
            self._buffers[tool_call_id] = {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            }

        buffer = self._buffers[tool_call_id]

        # Update function name if provided
        if "function" in chunk:
            func_data = chunk["function"]
            if "name" in func_data and func_data["name"]:
                buffer["function"]["name"] = func_data["name"]
            if "arguments" in func_data:
                buffer["function"]["arguments"] += func_data["arguments"]

        # Check if complete (has name and arguments ends with valid JSON)
        if buffer["function"]["name"] and buffer["function"]["arguments"]:
            args = buffer["function"]["arguments"].strip()
            if args.endswith("}") or args.endswith("]"):
                # Likely complete
                complete = buffer.copy()
                del self._buffers[tool_call_id]
                return complete

        return None

    def flush(self) -> List[Dict[str, Any]]:
        """Flush all buffered tool calls (may be incomplete)."""
        results = list(self._buffers.values())
        self._buffers.clear()
        return results

    def clear(self) -> None:
        """Clear all buffers."""
        self._buffers.clear()
