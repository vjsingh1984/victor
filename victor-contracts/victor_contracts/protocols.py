"""Core protocols for the Victor Contracts.

Pure protocol definitions that external packages can use for type checking
without importing from victor.core.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Protocol, runtime_checkable


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


@runtime_checkable
class SandboxProtocol(Protocol):
    """Protocol for stateful code execution sandboxes.

    Allows core executors (e.g. sandbox_executor) to manage sandboxes
    without knowing the concrete container implementation.
    """

    def __enter__(self) -> "SandboxProtocol": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    async def __aenter__(self) -> "SandboxProtocol": ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

    def start(self) -> None:
        """Start the sandbox environment."""
        ...

    def stop(self) -> None:
        """Stop and cleanup the sandbox environment."""
        ...

    def execute(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """Execute code inside the sandbox."""
        ...

    def put_files(self, file_paths: List[str]) -> None:
        """Upload files from the host to the sandbox."""
        ...

    def get_file(self, remote_path: str) -> bytes:
        """Retrieve a file from the sandbox."""
        ...


__all__ = ["OrchestratorProtocol", "SandboxProtocol"]
