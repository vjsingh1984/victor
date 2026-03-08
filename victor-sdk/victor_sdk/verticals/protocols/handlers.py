"""Handler-related protocol definitions.

These protocols define how verticals provide input/output handlers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, Any, List, Callable, Awaitable


@runtime_checkable
class HandlerProvider(Protocol):
    """Protocol for providing input/output handlers.

    Handlers transform inputs and outputs between the agent and
    external systems.
    """

    def get_input_handlers(self) -> Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]:
        """Return input transformation handlers.

        Returns:
            Dictionary mapping handler names to async functions
        """
        ...

    def get_output_handlers(self) -> Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]:
        """Return output transformation handlers.

        Returns:
            Dictionary mapping handler names to async functions
        """
        ...

    def get_handler_chain(self) -> List[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]:
        """Return ordered chain of handlers to apply.

        Returns:
            List of handler functions in execution order
        """
        ...


@runtime_checkable
class StreamHandler(Protocol):
    """Protocol for streaming response handlers.

    Stream handlers process streaming responses from the agent.
    """

    async def handle_chunk(self, chunk: Dict[str, Any]) -> None:
        """Handle a streaming response chunk.

        Args:
            chunk: Streaming chunk data
        """
        ...

    async def handle_complete(self) -> None:
        """Handle stream completion."""
        ...

    async def handle_error(self, error: Exception) -> None:
        """Handle stream error.

        Args:
            error: Exception that occurred
        """
        ...
