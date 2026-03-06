"""Streaming loop handler extracted from ChatCoordinator.

Manages the streaming response accumulation loop, chunk processing,
and continuation decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    """Result from a streaming loop iteration."""

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    should_continue: bool = False


class StreamingLoopHandler:
    """Handles the streaming response accumulation loop.

    Extracted from ChatCoordinator to reduce its size and isolate
    the streaming logic for independent testing.
    """

    def __init__(
        self,
        max_iterations: int = 50,
        on_chunk: Optional[Callable] = None,
        on_tool_call: Optional[Callable] = None,
    ):
        self._max_iterations = max_iterations
        self._on_chunk = on_chunk
        self._on_tool_call = on_tool_call
        self._iteration = 0

    async def run_streaming_loop(
        self,
        provider_stream: AsyncIterator,
        on_chunk: Optional[Callable] = None,
        on_tool_call: Optional[Callable] = None,
    ) -> StreamingResult:
        """Run the streaming accumulation loop.

        Args:
            provider_stream: Async iterator yielding chunks from provider.
            on_chunk: Callback for each content chunk.
            on_tool_call: Callback for each tool call detected.

        Returns:
            StreamingResult with accumulated content and tool calls.
        """
        chunk_handler = on_chunk or self._on_chunk
        tool_handler = on_tool_call or self._on_tool_call

        result = StreamingResult()
        chunks: list[str] = []

        async for chunk in provider_stream:
            if isinstance(chunk, dict):
                # Handle structured chunks
                if "content" in chunk:
                    content = chunk["content"]
                    chunks.append(content)
                    if chunk_handler:
                        await self._safe_callback(chunk_handler, content)

                if "tool_calls" in chunk:
                    for tc in chunk["tool_calls"]:
                        result.tool_calls.append(tc)
                        if tool_handler:
                            await self._safe_callback(tool_handler, tc)

                if "finish_reason" in chunk:
                    result.finish_reason = chunk["finish_reason"]

                if "usage" in chunk:
                    result.usage = chunk["usage"]
            elif isinstance(chunk, str):
                chunks.append(chunk)
                if chunk_handler:
                    await self._safe_callback(chunk_handler, chunk)

        result.content = "".join(chunks)
        self._iteration += 1
        result.should_continue = self.should_continue()

        return result

    async def accumulate_chunks(self, stream: AsyncIterator) -> str:
        """Simple accumulation without callbacks."""
        chunks: list[str] = []
        async for chunk in stream:
            if isinstance(chunk, dict) and "content" in chunk:
                chunks.append(chunk["content"])
            elif isinstance(chunk, str):
                chunks.append(chunk)
        return "".join(chunks)

    def should_continue(self) -> bool:
        """Whether the loop should continue for another iteration."""
        return self._iteration < self._max_iterations

    def reset(self) -> None:
        """Reset iteration counter."""
        self._iteration = 0

    async def _safe_callback(self, callback: Callable, *args: Any) -> None:
        """Safely invoke a callback, handling both sync and async."""
        import asyncio

        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning(f"Streaming callback error: {e}")
