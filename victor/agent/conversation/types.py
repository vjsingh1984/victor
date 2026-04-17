# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Canonical conversation types.

Single source of truth for message representation. ConversationMessage
is a superset of provider Message — it adds priority, token tracking,
timestamps, and metadata needed for context management and persistence.

All conversation modules import types from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from victor.providers.base import Message


class MessageRole(Enum):
    """Message roles in conversation.

    Values match the OpenAI Chat Completions API spec.
    Non-OpenAI providers should adapt at the provider layer.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # OpenAI spec: role=tool with tool_call_id
    TOOL_CALL = "tool_call"  # Internal: assistant requesting tool execution


class MessagePriority(Enum):
    """Priority levels for context management.

    Higher values indicate higher priority (kept longer during compaction).
    """

    CRITICAL = 100  # System prompts, current task
    HIGH = 75  # Recent tool results, code context
    MEDIUM = 50  # Previous exchanges
    LOW = 25  # Old context, summaries
    EPHEMERAL = 0  # Can be dropped immediately


@dataclass
class ConversationMessage:
    """Canonical message type for the conversation system.

    Superset of provider Message — adds priority, token tracking,
    timestamps, and metadata for context management and persistence.

    Usage:
        # Create from scratch
        msg = ConversationMessage(role="user", content="Fix the bug")

        # Create from provider Message
        msg = ConversationMessage.from_provider_message(provider_msg)

        # Convert back to provider Message for LLM calls
        provider_msg = msg.to_provider_message()
    """

    role: str
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    token_count: int = 0
    priority: MessagePriority = MessagePriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    @property
    def role_enum(self) -> MessageRole:
        """Get role as MessageRole enum."""
        try:
            return MessageRole(self.role)
        except ValueError:
            return MessageRole.ASSISTANT

    def to_provider_message(self) -> "Message":
        """Convert to provider Message for LLM API calls."""
        from victor.providers.base import Message

        return Message(
            role=self._normalize_role(),
            content=self.content,
            name=self.name,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
        )

    @classmethod
    def from_provider_message(
        cls,
        msg: "Message",
        *,
        priority: MessagePriority = MessagePriority.MEDIUM,
        token_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMessage:
        """Create from provider Message."""
        return cls(
            role=msg.role,
            content=msg.content,
            name=msg.name,
            tool_calls=msg.tool_calls,
            tool_call_id=msg.tool_call_id,
            priority=priority,
            token_count=token_count,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "priority": self.priority.value,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationMessage:
        """Deserialize from dictionary."""
        priority_val = data.get("priority", 50)
        try:
            priority = MessagePriority(priority_val)
        except ValueError:
            priority = MessagePriority.MEDIUM

        return cls(
            id=data.get("id", str(uuid4())),
            role=data["role"],
            content=data["content"],
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now(tz=timezone.utc)
            ),
            token_count=data.get("token_count", 0),
            priority=priority,
            tool_name=data.get("tool_name"),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            metadata=data.get("metadata", {}),
        )

    def _normalize_role(self) -> str:
        """Normalize role for provider API (only system/user/assistant/tool)."""
        if self.role in ("system", "user", "assistant", "tool"):
            return self.role
        if self.role == "tool_result":
            return "tool"
        return "assistant"


__all__ = [
    "ConversationMessage",
    "MessagePriority",
    "MessageRole",
]
