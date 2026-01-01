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

"""Streaming Tool Adapter - Unified streaming interface for ToolPipeline.

This module provides a streaming adapter that wraps ToolPipeline, enabling
real-time tool execution updates while preserving ALL ToolPipeline features:
- Caching and deduplication
- Middleware chain integration
- Budget management
- Parallel execution
- Output aggregation
- Analytics/metrics
- Before/after callbacks
- Verification steps

SOLID Principles Applied:
- SRP: Adapter handles streaming concerns only; ToolPipeline handles execution
- OCP: Extends behavior without modifying ToolPipeline
- LSP: Can substitute direct tool calls anywhere
- ISP: Minimal StreamingToolAdapterProtocol interface
- DIP: Depends on ToolPipeline abstraction (via protocol)

This solves the dual execution path problem where:
- Batch path: Used ToolPipeline with full feature support
- Streaming path: Previously bypassed ToolPipeline using self.tools.execute()

Now BOTH paths route through: StreamingToolAdapter -> ToolPipeline
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, TYPE_CHECKING

from victor.agent.protocols import StreamingToolChunk, StreamingToolAdapterProtocol

if TYPE_CHECKING:
    from victor.agent.tool_pipeline import ToolPipeline, ToolCallResult

logger = logging.getLogger(__name__)


class StreamingToolAdapter:
    """Adapter that wraps ToolPipeline for streaming tool execution.

    Provides an async generator interface for tool execution while delegating
    all actual execution to ToolPipeline, preserving:
    - Caching (idempotent and tool-specific)
    - Middleware (code correction, etc.)
    - Callbacks (on_tool_start, on_tool_complete)
    - Budget enforcement
    - Deduplication
    - Analytics

    Usage:
        adapter = StreamingToolAdapter(tool_pipeline)

        async for chunk in adapter.execute_streaming(tool_calls, context):
            if chunk.chunk_type == "start":
                print(f"Starting {chunk.tool_name}...")
            elif chunk.chunk_type == "result":
                print(f"Result: {chunk.content}")
            elif chunk.chunk_type == "error":
                print(f"Error: {chunk.content}")

    The adapter emits StreamingToolChunk for each execution phase:
    - "start": Tool execution beginning (with args in content)
    - "cache_hit": Result served from cache (with cached result)
    - "result": Successful completion (with ToolCallResult)
    - "error": Execution failure (with error message)
    """

    def __init__(
        self,
        tool_pipeline: "ToolPipeline",
        on_chunk: Optional[Callable[[StreamingToolChunk], None]] = None,
    ):
        """Initialize the streaming adapter.

        Args:
            tool_pipeline: ToolPipeline instance to wrap
            on_chunk: Optional callback for each chunk (for observability hooks)
        """
        self._pipeline = tool_pipeline
        self._on_chunk = on_chunk

    @property
    def calls_used(self) -> int:
        """Number of tool calls used (delegates to ToolPipeline)."""
        return self._pipeline.calls_used

    @property
    def calls_remaining(self) -> int:
        """Number of tool calls remaining in budget."""
        return self._pipeline.calls_remaining

    @property
    def budget(self) -> int:
        """Maximum tool calls allowed."""
        return self._pipeline.config.tool_budget

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted."""
        return self._pipeline.calls_used >= self._pipeline.config.tool_budget

    def reset(self) -> None:
        """Reset adapter state (delegates to pipeline)."""
        self._pipeline.reset()

    def _emit_chunk(self, chunk: StreamingToolChunk) -> None:
        """Emit a chunk and call optional callback."""
        if self._on_chunk:
            try:
                self._on_chunk(chunk)
            except Exception as e:
                logger.warning(f"on_chunk callback failed: {e}")

    async def execute_streaming(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamingToolChunk]:
        """Execute tools with streaming output.

        Yields StreamingToolChunk for each execution phase:
        1. "start" - Tool execution beginning
        2. "cache_hit" - Result served from cache (skips execution)
        3. "result" - Successful completion with result
        4. "error" - Execution failure

        Args:
            tool_calls: List of tool calls to execute
            context: Optional execution context

        Yields:
            StreamingToolChunk for each execution event
        """
        context = context or {}

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("arguments", {})
            tool_call_id = tool_call.get("id") or str(uuid.uuid4())[:8]

            async for chunk in self.execute_streaming_single(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
                context=context,
            ):
                yield chunk

                # Check if budget exhausted after each tool
                if chunk.is_final and self.is_budget_exhausted():
                    # Emit budget exhausted notification
                    budget_chunk = StreamingToolChunk(
                        tool_name="",
                        tool_call_id="budget",
                        chunk_type="error",
                        content="Tool budget exhausted",
                        is_final=True,
                        metadata={"budget_exhausted": True},
                    )
                    self._emit_chunk(budget_chunk)
                    yield budget_chunk
                    return

    async def execute_streaming_single(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamingToolChunk]:
        """Execute a single tool with streaming output.

        Convenience method for single tool execution.

        Args:
            tool_name: Name of tool to execute
            tool_args: Tool arguments
            tool_call_id: Optional identifier for tracking
            context: Optional execution context

        Yields:
            StreamingToolChunk for each execution event
        """
        tool_call_id = tool_call_id or str(uuid.uuid4())[:8]
        context = context or {}
        start_time = time.monotonic()

        # Emit start chunk
        start_chunk = StreamingToolChunk(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            chunk_type="start",
            content=tool_args,
            is_final=False,
            metadata={"start_time": start_time},
        )
        self._emit_chunk(start_chunk)
        yield start_chunk

        try:
            # Create tool call dict for pipeline
            tool_call = {
                "name": tool_name,
                "arguments": tool_args,
                "id": tool_call_id,
            }

            # Execute through ToolPipeline (preserves all features)
            result = await self._pipeline.execute_tool_calls([tool_call], context)

            if not result.results:
                # No results - should not happen but handle gracefully
                error_chunk = StreamingToolChunk(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    chunk_type="error",
                    content="No result returned from pipeline",
                    is_final=True,
                    metadata={"execution_time_ms": (time.monotonic() - start_time) * 1000},
                )
                self._emit_chunk(error_chunk)
                yield error_chunk
                return

            # Get the tool call result
            tool_result = result.results[0]
            execution_time_ms = (time.monotonic() - start_time) * 1000

            # Check if it was a cache hit
            if tool_result.cached or (
                tool_result.skipped and "cache" in (tool_result.skip_reason or "").lower()
            ):
                cache_chunk = StreamingToolChunk(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    chunk_type="cache_hit",
                    content=tool_result.result,
                    is_final=True,
                    metadata={
                        "execution_time_ms": execution_time_ms,
                        "cached": True,
                        "skip_reason": tool_result.skip_reason,
                    },
                )
                self._emit_chunk(cache_chunk)
                yield cache_chunk
                return

            # Check if it was skipped for other reasons
            if tool_result.skipped:
                skip_chunk = StreamingToolChunk(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    chunk_type="error",
                    content=tool_result.skip_reason or "Skipped",
                    is_final=True,
                    metadata={
                        "execution_time_ms": execution_time_ms,
                        "skipped": True,
                        "skip_reason": tool_result.skip_reason,
                    },
                )
                self._emit_chunk(skip_chunk)
                yield skip_chunk
                return

            # Check success/failure
            if tool_result.success:
                result_chunk = StreamingToolChunk(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    chunk_type="result",
                    content=tool_result,  # Full ToolCallResult for downstream processing
                    is_final=True,
                    metadata={
                        "execution_time_ms": execution_time_ms,
                        "success": True,
                        "normalization_applied": tool_result.normalization_applied,
                    },
                )
                self._emit_chunk(result_chunk)
                yield result_chunk
            else:
                error_chunk = StreamingToolChunk(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    chunk_type="error",
                    content=tool_result.error or "Execution failed",
                    is_final=True,
                    metadata={
                        "execution_time_ms": execution_time_ms,
                        "success": False,
                        "result": tool_result.result,
                    },
                )
                self._emit_chunk(error_chunk)
                yield error_chunk

        except Exception as e:
            # Handle unexpected exceptions
            execution_time_ms = (time.monotonic() - start_time) * 1000
            error_chunk = StreamingToolChunk(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                chunk_type="error",
                content=str(e),
                is_final=True,
                metadata={
                    "execution_time_ms": execution_time_ms,
                    "exception": type(e).__name__,
                },
            )
            self._emit_chunk(error_chunk)
            yield error_chunk
            logger.exception(f"Exception during streaming tool execution: {e}")


def create_streaming_tool_adapter(
    tool_pipeline: "ToolPipeline",
    on_chunk: Optional[Callable[[StreamingToolChunk], None]] = None,
) -> StreamingToolAdapter:
    """Factory function to create a StreamingToolAdapter.

    Args:
        tool_pipeline: ToolPipeline instance to wrap
        on_chunk: Optional callback for each chunk

    Returns:
        Configured StreamingToolAdapter
    """
    return StreamingToolAdapter(tool_pipeline=tool_pipeline, on_chunk=on_chunk)


# Type assertion for protocol conformance
def _assert_protocol_conformance() -> None:
    """Compile-time check that StreamingToolAdapter implements the protocol."""
    adapter: StreamingToolAdapterProtocol = StreamingToolAdapter(None)  # type: ignore
    _ = adapter  # Silence unused variable warning
