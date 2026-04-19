"""Core protocols for the Victor SDK.

Pure protocol definitions that external packages can use for type checking
without importing from victor.core.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, List, Protocol, runtime_checkable


@runtime_checkable
class OrchestratorProtocol(Protocol):
    """Protocol for agent orchestrator functionality.

    Allows external verticals and evaluation modules to depend on the
    orchestrator interface without importing the concrete implementation.
    """

    @property
    def model(self) -> str:
        """Current model identifier."""
        ...

    @property
    def provider_name(self) -> str:
        """Current provider name."""
        ...

    @property
    def tool_budget(self) -> int:
        """Maximum allowed tool calls for session."""
        ...

    @property
    def tool_calls_used(self) -> int:
        """Number of tool calls used in current session."""
        ...

    @property
    def messages(self) -> List[Any]:
        """Current conversation message history."""
        ...

    async def chat(self, user_message: str) -> Any:
        """Process a user message and return completion response."""
        ...

    async def stream_chat(self, user_message: str) -> AsyncIterator[Any]:
        """Process a user message and yield streaming response chunks."""
        ...

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        ...


__all__ = ["OrchestratorProtocol"]
