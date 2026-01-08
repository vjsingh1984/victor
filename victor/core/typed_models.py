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

"""Typed dataclasses for domain objects.

This module provides type-safe alternatives to getattr() for accessing
attributes on domain objects. Part of HIGH-003: Unsafe Attribute Access migration.

Design Principles:
    - All fields are explicitly typed
    - Optional fields use Optional[T] with None defaults
    - Immutable by default (frozen=True where appropriate)
    - No dynamic attribute access via getattr()

Usage:
    # OLD (unsafe)
    cost_tier = getattr(tool, "cost_tier", None)

    # NEW (type-safe)
    metadata = ToolMetadata.from_tool(tool)
    cost_tier = metadata.cost_tier  # IDE autocomplete, type checking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# Tool Metadata
# =============================================================================


class Priority(str, Enum):
    """Tool priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessMode(str, Enum):
    """Tool access modes."""

    READONLY = "readonly"
    WRITE = "write"
    READWRITE = "readwrite"


class DangerLevel(str, Enum):
    """Tool danger levels."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionCategory(str, Enum):
    """Tool execution categories."""

    READ_ONLY = "read_only"
    WRITE = "write"
    NETWORK = "network"
    COMPUTE = "compute"


@dataclass(frozen=True)
class ToolMetadata:
    """Type-safe metadata for a tool.

    Replaces getattr() calls on tool objects with typed access.
    All tools inherit from BaseTool which has these attributes.
    """

    name: str
    description: str
    priority: Priority = Priority.MEDIUM
    access_mode: AccessMode = AccessMode.READONLY
    danger_level: DangerLevel = DangerLevel.SAFE
    execution_category: ExecutionCategory = ExecutionCategory.READ_ONLY
    aliases: Set[str] = field(default_factory=set)
    category: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    stages: List[str] = field(default_factory=list)
    mandatory_keywords: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)
    progress_params: List[str] = field(default_factory=list)
    cost_tier: Optional[str] = None

    @classmethod
    def from_tool(cls, tool: Any) -> ToolMetadata:
        """Extract metadata from a tool instance.

        Args:
            tool: Tool instance (BaseTool subclass)

        Returns:
            ToolMetadata with safe defaults for missing attributes
        """
        return cls(
            name=getattr(tool, "name", "unknown"),
            description=getattr(tool, "description", ""),
            priority=getattr(tool, "priority", Priority.MEDIUM),
            access_mode=getattr(tool, "access_mode", AccessMode.READONLY),
            danger_level=getattr(tool, "danger_level", DangerLevel.SAFE),
            execution_category=getattr(tool, "execution_category", ExecutionCategory.READ_ONLY),
            aliases=getattr(tool, "aliases", set()),
            category=getattr(tool, "category", None),
            keywords=getattr(tool, "keywords", []),
            stages=getattr(tool, "stages", []),
            mandatory_keywords=getattr(tool, "mandatory_keywords", []),
            task_types=getattr(tool, "task_types", []),
            progress_params=getattr(tool, "progress_params", []),
            cost_tier=getattr(tool, "cost_tier", None),
        )


# =============================================================================
# Provider Response Models
# =============================================================================


@dataclass
class ProviderUsage:
    """Token usage information from provider response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ProviderMessage:
    """Message in provider response."""

    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class StreamDelta:
    """Delta update in streaming response."""

    content: str = ""
    role: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None


@dataclass
class TypedStreamChunk:
    """Single chunk from streaming response with type-safe nested structure.

    Replaces getattr(delta, "content", "") pattern. Uses nested StreamDelta
    for structured access to streaming delta updates.

    Renamed from StreamChunk to be semantically distinct from other streaming types:
    - StreamChunk (victor.providers.base): Provider-level raw streaming
    - OrchestratorStreamChunk: Orchestrator protocol with typed ChunkType
    - TypedStreamChunk: Safe typed accessor with nested StreamDelta
    - ClientStreamChunk: Protocol interface for clients (CLI/VS Code)
    """

    delta: StreamDelta
    finish_reason: Optional[str] = None
    index: int = 0

    @property
    def content(self) -> str:
        """Convenience accessor for delta content."""
        return self.delta.content

    @property
    def is_tool_call(self) -> bool:
        """Check if chunk contains tool call."""
        return self.delta.tool_calls is not None and len(self.delta.tool_calls) > 0


# =============================================================================
# Agent Component Models
# =============================================================================


@dataclass
class AgentInfo:
    """Type-safe facade for accessing agent/orchestrator components.

    Replaces getattr(self.agent, "component", None) pattern.
    """

    provider: Any
    mcp_client: Optional[Any] = None
    codebase_index: Optional[Any] = None
    conversation_state: Optional[Any] = None
    intelligent_integration: Optional[Any] = None

    @classmethod
    def from_agent(cls, agent: Any) -> AgentInfo:
        """Extract component references from agent."""
        return cls(
            provider=agent.provider,
            mcp_client=getattr(agent, "mcp_client", None),
            codebase_index=getattr(agent, "codebase_index", None),
            conversation_state=getattr(agent, "conversation_state", None),
            intelligent_integration=getattr(agent, "intelligent_integration", None),
        )


# =============================================================================
# Metrics Models
# =============================================================================


@dataclass
class ModelMetrics:
    """Performance metrics for a model execution.

    Replaces getattr(m, "tokens_per_second", 0) pattern.
    """

    provider: str = "unknown"
    model: str = "unknown"
    tokens_per_second: float = 0.0
    time_to_first_token: float = 0.0
    total_duration: float = 0.0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetrics:
        """Create from dictionary with safe defaults."""
        return cls(
            provider=data.get("provider", "unknown"),
            model=data.get("model", "unknown"),
            tokens_per_second=data.get("tokens_per_second", 0.0),
            time_to_first_token=data.get("time_to_first_token", 0.0),
            total_duration=data.get("total_duration", 0.0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class UsageSummary:
    """Summary of usage analytics.

    Replaces getattr(summary, "total_requests", 0) pattern.
    """

    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> UsageSummary:
        """Create from optional dictionary."""
        if data is None:
            return cls()
        return cls(
            total_requests=data.get("total_requests", 0),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            avg_latency=data.get("avg_latency", 0.0),
        )


# =============================================================================
# CQRS Event Models
# =============================================================================


@dataclass
class CQRSEventMetadata:
    """Metadata for CQRS events."""

    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    correlation_id: Optional[str] = None


@dataclass
class CQRSEvent:
    """Base CQRS event with typed fields.

    Replaces getattr(cqrs_event, "tool_name", "") pattern.
    """

    event_type: str
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: CQRSEventMetadata = field(default_factory=CQRSEventMetadata)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CQRSEvent:
        """Create from dictionary with safe extraction."""
        metadata_dict = data.get("metadata", {})
        metadata = CQRSEventMetadata(
            data=metadata_dict.get("data", {}),
            timestamp=metadata_dict.get("timestamp"),
            correlation_id=metadata_dict.get("correlation_id"),
        )

        return cls(
            event_type=data.get("event_type", "unknown"),
            tool_name=data.get("tool_name", ""),
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            error=data.get("error"),
            metadata=metadata,
        )


__all__ = [
    # Enums
    "Priority",
    "AccessMode",
    "DangerLevel",
    "ExecutionCategory",
    # Tool models
    "ToolMetadata",
    # Provider models
    "ProviderUsage",
    "ProviderMessage",
    "StreamDelta",
    "TypedStreamChunk",
    # Agent models
    "AgentInfo",
    # Metrics models
    "ModelMetrics",
    "UsageSummary",
    # CQRS models
    "CQRSEventMetadata",
    "CQRSEvent",
]
