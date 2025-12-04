"""Victor Protocol - Unified API contract for all clients.

This module defines the protocol interface that all Victor clients
(CLI, VS Code, JetBrains, MCP) use to communicate with the core engine.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     CLIENTS (Layer 3)                        │
    ├──────────┬──────────┬───────────┬──────────┬───────────────┤
    │   CLI    │  VS Code │ JetBrains │   MCP    │   HTTP API    │
    └────┬─────┴────┬─────┴─────┬─────┴────┬─────┴───────┬───────┘
         │          │           │          │             │
         │ VictorProtocol (shared interface)             │
         │          │           │          │             │
    ┌────▼──────────▼───────────▼──────────▼─────────────▼───────┐
    │               PROTOCOL ADAPTERS (Layer 2)                   │
    ├─────────────────────────────────────────────────────────────┤
    │  DirectAdapter  │  HTTPAdapter  │  MCPAdapter  │  WSAdapter │
    └───────────────────────────┬─────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────┐
    │                    CORE ENGINE (Layer 1)                     │
    ├─────────────────────────────────────────────────────────────┤
    │    AgentOrchestrator  │  Tools  │  Providers  │  State      │
    └─────────────────────────────────────────────────────────────┘
"""

from victor.protocol.interface import (
    VictorProtocol,
    ChatMessage,
    ChatResponse,
    StreamChunk,
    SearchResult,
    ToolCall,
    ToolResult,
    UndoRedoResult,
    AgentMode,
    AgentStatus,
)
from victor.protocol.adapters import (
    DirectProtocolAdapter,
    HTTPProtocolAdapter,
)

__all__ = [
    # Protocol interface
    "VictorProtocol",
    # Data types
    "ChatMessage",
    "ChatResponse",
    "StreamChunk",
    "SearchResult",
    "ToolCall",
    "ToolResult",
    "UndoRedoResult",
    "AgentMode",
    "AgentStatus",
    # Adapters
    "DirectProtocolAdapter",
    "HTTPProtocolAdapter",
]
