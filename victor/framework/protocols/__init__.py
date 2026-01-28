"""Stable protocols for framework-orchestrator integration.

These protocols define the contract that any orchestrator implementation
must satisfy. Framework code uses these protocols, never direct attribute access.

Design Pattern: Protocol-First Architecture
- Eliminates duck-typing (hasattr/getattr calls)
- Provides type safety via Protocol structural subtyping
- Enables clean mocking for tests
- Documents the exact interface contract

Usage:
    # Type hint with protocol instead of concrete class
    def process(orchestrator: OrchestratorProtocol) -> None:
        stage = orchestrator.get_stage()  # Type-safe method call
        tools = orchestrator.get_available_tools()

Package Structure:
    This package organizes protocols into logical modules:
    - exceptions: Protocol-related exceptions
    - streaming: Streaming types and chunks
    - component: Component protocols (state, provider, tools, etc.)
    - orchestrator: Main orchestrator protocol
    - capability: Capability discovery and versioning
    - utils: Protocol verification utilities
    - chat: Phase 1 workflow chat protocols and implementations
"""

from __future__ import annotations

# Re-export all protocols for backward compatibility
from victor.framework.protocols.exceptions import IncompatibleVersionError
from victor.framework.protocols.streaming import ChunkType, OrchestratorStreamChunk
from victor.framework.protocols.component import (
    ConversationStateProtocol,
    ProviderProtocol,
    ToolsProtocol,
    SystemPromptProtocol,
    MessagesProtocol,
    StreamingProtocol,
)
from victor.framework.protocols.orchestrator import OrchestratorProtocol
from victor.framework.protocols.capability import (
    CapabilityType,
    OrchestratorCapability,
    CapabilityRegistryProtocol,
)
from victor.framework.protocols.utils import verify_protocol_conformance
from victor.framework.protocols.chat import (
    ChatStateProtocol,
    ChatResultProtocol,
    WorkflowChatProtocol,
    ChatResult,
    MutableChatState,
)

__all__ = [
    # Exceptions
    "IncompatibleVersionError",
    # Streaming
    "ChunkType",
    "OrchestratorStreamChunk",
    # Component Protocols
    "ConversationStateProtocol",
    "ProviderProtocol",
    "ToolsProtocol",
    "SystemPromptProtocol",
    "MessagesProtocol",
    "StreamingProtocol",
    # Main Protocol
    "OrchestratorProtocol",
    # Capability Discovery
    "CapabilityType",
    "OrchestratorCapability",
    "CapabilityRegistryProtocol",
    # Utilities
    "verify_protocol_conformance",
    # Phase 1: Workflow Chat
    "ChatStateProtocol",
    "ChatResultProtocol",
    "WorkflowChatProtocol",
    "ChatResult",
    "MutableChatState",
]
