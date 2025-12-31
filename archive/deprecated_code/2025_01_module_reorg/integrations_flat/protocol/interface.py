"""Victor Protocol Interface - Unified API contract.

This module defines the abstract interface that all Victor clients
use to interact with the core engine. Both CLI and VS Code extension
implement this protocol to ensure feature parity.

Usage:
    # CLI uses DirectProtocolAdapter
    from victor.protocol import DirectProtocolAdapter
    protocol = await DirectProtocolAdapter.create()
    response = await protocol.chat([ChatMessage(role="user", content="Hello")])

    # VS Code uses HTTPProtocolAdapter
    from victor.protocol import HTTPProtocolAdapter
    protocol = HTTPProtocolAdapter("http://localhost:8765")
    response = await protocol.chat([ChatMessage(role="user", content="Hello")])
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Any
from datetime import datetime


class AgentMode(str, Enum):
    """Agent operation modes."""

    BUILD = "build"  # Full implementation - all tools available
    PLAN = "plan"  # Read-only analysis mode
    EXPLORE = "explore"  # Code navigation and understanding


@dataclass
class ChatMessage:
    """A message in a conversation."""

    role: str  # "user", "assistant", or "system"
    content: str
    tool_calls: list["ToolCall"] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        return cls(
            role=data["role"],
            content=data["content"],
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            timestamp=(
                datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
            ),
        )


@dataclass
class ToolCall:
    """A tool invocation by the agent."""

    id: str
    name: str
    arguments: dict[str, Any]
    result: "ToolResult | None" = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result.to_dict() if self.result else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data["arguments"],
            result=ToolResult.from_dict(data["result"]) if data.get("result") else None,
        )


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        return cls(
            success=data["success"],
            output=data["output"],
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChatResponse:
    """Response from a chat request."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatResponse":
        return cls(
            content=data["content"],
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            finish_reason=data.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
        )


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""

    content: str
    tool_call: ToolCall | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "finish_reason": self.finish_reason,
        }


@dataclass
class SearchResult:
    """Result from a code search."""

    file: str
    line: int
    content: str
    score: float
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "content": self.content,
            "score": self.score,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchResult":
        return cls(
            file=data["file"],
            line=data["line"],
            content=data["content"],
            score=data["score"],
            context=data.get("context", ""),
        )


@dataclass
class UndoRedoResult:
    """Result from an undo/redo operation."""

    success: bool
    message: str
    files_modified: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "files_modified": self.files_modified,
        }


@dataclass
class AgentStatus:
    """Current agent status."""

    provider: str
    model: str
    mode: AgentMode
    connected: bool
    tools_available: int
    conversation_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode.value,
            "connected": self.connected,
            "tools_available": self.tools_available,
            "conversation_length": self.conversation_length,
        }


class VictorProtocol(ABC):
    """Abstract protocol interface for Victor clients.

    All clients (CLI, VS Code, JetBrains, MCP) implement this interface
    through appropriate adapters. This ensures feature parity across
    all integration points.
    """

    # =========================================================================
    # Chat Operations
    # =========================================================================

    @abstractmethod
    async def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        """Send messages and get a response.

        Args:
            messages: Conversation history

        Returns:
            Chat response with content and tool calls
        """
        ...

    @abstractmethod
    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Stream a chat response.

        Args:
            messages: Conversation history

        Yields:
            Stream chunks with content and tool calls
        """
        ...

    @abstractmethod
    async def reset_conversation(self) -> None:
        """Clear conversation history."""
        ...

    # =========================================================================
    # Search Operations
    # =========================================================================

    @abstractmethod
    async def semantic_search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search code by semantic meaning.

        Args:
            query: Natural language query
            max_results: Maximum results to return

        Returns:
            List of search results ranked by relevance
        """
        ...

    @abstractmethod
    async def code_search(
        self,
        query: str,
        regex: bool = False,
        case_sensitive: bool = False,
        file_pattern: str | None = None,
    ) -> list[SearchResult]:
        """Search code by pattern.

        Args:
            query: Search pattern (literal or regex)
            regex: Treat query as regex
            case_sensitive: Match case
            file_pattern: Glob pattern to filter files

        Returns:
            List of search results
        """
        ...

    # =========================================================================
    # Model/Mode Management
    # =========================================================================

    @abstractmethod
    async def switch_model(self, provider: str, model: str) -> None:
        """Switch to a different model.

        Args:
            provider: Provider name (anthropic, openai, ollama, etc.)
            model: Model identifier
        """
        ...

    @abstractmethod
    async def switch_mode(self, mode: AgentMode) -> None:
        """Switch agent mode.

        Args:
            mode: New agent mode
        """
        ...

    @abstractmethod
    async def get_status(self) -> AgentStatus:
        """Get current agent status.

        Returns:
            Current agent status including provider, model, mode
        """
        ...

    # =========================================================================
    # Undo/Redo Operations
    # =========================================================================

    @abstractmethod
    async def undo(self) -> UndoRedoResult:
        """Undo the last change.

        Returns:
            Result with success status and affected files
        """
        ...

    @abstractmethod
    async def redo(self) -> UndoRedoResult:
        """Redo the last undone change.

        Returns:
            Result with success status and affected files
        """
        ...

    @abstractmethod
    async def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get change history.

        Args:
            limit: Maximum entries to return

        Returns:
            List of history entries
        """
        ...

    # =========================================================================
    # Patch Operations
    # =========================================================================

    @abstractmethod
    async def apply_patch(self, patch: str, dry_run: bool = False) -> dict[str, Any]:
        """Apply a unified diff patch.

        Args:
            patch: Unified diff patch content
            dry_run: If True, preview without applying

        Returns:
            Result with success status and affected files
        """
        ...

    # =========================================================================
    # LSP Operations (optional, for IDE integrations)
    # =========================================================================

    async def get_definition(self, file: str, line: int, character: int) -> list[dict[str, Any]]:
        """Get definition locations for symbol at position.

        Args:
            file: File path
            line: Line number (0-indexed)
            character: Character position

        Returns:
            List of definition locations
        """
        return []

    async def get_references(self, file: str, line: int, character: int) -> list[dict[str, Any]]:
        """Get reference locations for symbol at position.

        Args:
            file: File path
            line: Line number (0-indexed)
            character: Character position

        Returns:
            List of reference locations
        """
        return []

    async def get_hover(self, file: str, line: int, character: int) -> str | None:
        """Get hover information for symbol at position.

        Args:
            file: File path
            line: Line number (0-indexed)
            character: Character position

        Returns:
            Hover content or None
        """
        return None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    @abstractmethod
    async def close(self) -> None:
        """Close the connection and clean up resources."""
        ...

    async def check_health(self) -> bool:
        """Check if the connection is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self.get_status()
            return True
        except Exception:
            return False
