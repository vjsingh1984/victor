"""Service adapters that bridge coordinators to service protocols.

These adapters implement the Strangler Fig pattern: they wrap existing
coordinator implementations behind service protocol interfaces, enabling
gradual migration from coordinator-based to service-based architecture.

Feature-flagged via USE_SERVICE_LAYER — when disabled, the orchestrator
continues using coordinators directly.
"""

from victor.agent.services.adapters.chat_adapter import ChatServiceAdapter
from victor.agent.services.adapters.tool_adapter import ToolServiceAdapter
from victor.agent.services.adapters.context_adapter import ContextServiceAdapter
from victor.agent.services.adapters.session_adapter import SessionServiceAdapter

__all__ = [
    "ChatServiceAdapter",
    "ToolServiceAdapter",
    "ContextServiceAdapter",
    "SessionServiceAdapter",
]
