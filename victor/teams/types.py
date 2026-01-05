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

"""Canonical types for multi-agent team coordination.

This module is the SINGLE SOURCE OF TRUTH for all team-related types.
Import from here, not from victor.framework.agent_protocols or victor.agent.teams.

Example:
    from victor.teams.types import TeamFormation, MessageType, AgentMessage

Type Consolidation:
    - TeamFormation: Unified from framework (5 values) and agent/teams (4 values)
    - MessageType: Unified from framework (5 values) and agent/teams (7 values)
    - AgentMessage: Superset of both implementations
    - MemberResult: Superset of both implementations
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TeamFormation(str, Enum):
    """Team organization patterns for multi-agent coordination.

    Unified superset combining:
    - Framework formations: SEQUENTIAL, PARALLEL, HIERARCHICAL, PIPELINE, CONSENSUS
    - Agent/teams formations: SEQUENTIAL, PARALLEL, HIERARCHICAL, PIPELINE

    Attributes:
        SEQUENTIAL: Execute members one after another, context chaining
        PARALLEL: Execute all members simultaneously, independent work
        HIERARCHICAL: Manager delegates to workers, synthesizes results
        PIPELINE: Output of one member feeds into the next
        CONSENSUS: All members must agree (multiple rounds if needed)
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PIPELINE = "pipeline"
    CONSENSUS = "consensus"


class MessageType(str, Enum):
    """Types of messages for inter-agent communication.

    Unified superset combining:
    - Framework types: TASK, RESULT, QUERY, FEEDBACK, DELEGATION
    - Agent/teams types: DISCOVERY, REQUEST, RESPONSE, STATUS, ALERT, HANDOFF, RESULT

    Attributes:
        TASK: Task assignment message
        RESULT: Result from task execution
        QUERY: Information request
        FEEDBACK: Feedback on work
        DELEGATION: Task delegation request
        DISCOVERY: Discovery/finding announcement
        REQUEST: Generic request
        RESPONSE: Response to a request
        STATUS: Status update
        ALERT: Alert/warning message
        HANDOFF: Task handoff between agents
    """

    # From framework/agent_protocols.py
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    FEEDBACK = "feedback"
    DELEGATION = "delegation"

    # From agent/teams/communication.py
    DISCOVERY = "discovery"
    REQUEST = "request"
    RESPONSE = "response"
    STATUS = "status"
    ALERT = "alert"
    HANDOFF = "handoff"


class MessagePriority(int, Enum):
    """Priority levels for agent messages."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class AgentMessage:
    """Unified agent message for inter-agent communication.

    Superset combining:
    - Framework AgentMessage: sender_id, recipient_id, content, message_type, metadata
    - Agent/teams AgentMessage: type, from_agent, to_agent, content, data, timestamp,
                                reply_to, priority, id

    This canonical version uses consistent naming and supports all features.

    Attributes:
        sender_id: ID of the sending agent
        recipient_id: ID of the recipient (None for broadcast)
        content: Message content string
        message_type: Type of message (MessageType enum)
        data: Structured data payload
        timestamp: Unix timestamp when created
        reply_to: ID of message this is replying to
        priority: Message priority level
        id: Unique message identifier
    """

    sender_id: str
    recipient_id: Optional[str]  # None = broadcast to all
    content: str
    message_type: MessageType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Compatibility aliases for framework AgentMessage
    @property
    def metadata(self) -> Dict[str, Any]:
        """Alias for data (framework compatibility)."""
        return self.data

    # Compatibility aliases for agent/teams AgentMessage
    @property
    def from_agent(self) -> str:
        """Alias for sender_id (agent/teams compatibility)."""
        return self.sender_id

    @property
    def to_agent(self) -> Optional[str]:
        """Alias for recipient_id (agent/teams compatibility)."""
        return self.recipient_id

    @property
    def type(self) -> MessageType:
        """Alias for message_type (agent/teams compatibility)."""
        return self.message_type

    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message."""
        return self.recipient_id is None

    def is_reply(self) -> bool:
        """Check if this is a reply to another message."""
        return self.reply_to is not None

    def to_context_string(self) -> str:
        """Format message for inclusion in agent context."""
        header = f"[{self.message_type.value.upper()}] {self.sender_id}"
        if self.recipient_id:
            header += f" → {self.recipient_id}"
        return f"{header}: {self.content}"


@dataclass
class MemberResult:
    """Result from a team member's task execution.

    Superset combining:
    - Framework MemberResult: member_id, success, output, error, metadata
    - Agent/teams MemberResult: member_id, success, output, tool_calls_used,
                                duration_seconds, discoveries, error

    Attributes:
        member_id: ID of the team member
        success: Whether execution succeeded
        output: Output/result string
        error: Error message if failed
        metadata: Additional metadata
        tool_calls_used: Number of tool calls made
        duration_seconds: Execution duration
        discoveries: Key findings/discoveries made
    """

    member_id: str
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls_used: int = 0
    duration_seconds: float = 0.0
    discoveries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "member_id": self.member_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "tool_calls_used": self.tool_calls_used,
            "duration_seconds": self.duration_seconds,
            "discoveries": self.discoveries,
        }


@dataclass
class TeamResult:
    """Result from team task execution.

    Attributes:
        success: Whether the team succeeded overall
        final_output: Synthesized final output
        member_results: Results from each member
        formation: Formation pattern used
        total_tool_calls: Total tool calls across all members
        total_duration: Total execution duration
        communication_log: Log of inter-agent messages
        shared_context: Final shared context state
        consensus_achieved: Whether consensus was reached (for CONSENSUS formation)
        consensus_rounds: Number of rounds needed for consensus
        error: Error message if failed
    """

    success: bool
    final_output: str
    member_results: Dict[str, MemberResult]
    formation: TeamFormation
    total_tool_calls: int = 0
    total_duration: float = 0.0
    communication_log: List[AgentMessage] = field(default_factory=list)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    consensus_achieved: Optional[bool] = None
    consensus_rounds: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "final_output": self.final_output,
            "member_results": {
                k: v.to_dict() for k, v in self.member_results.items()
            },
            "formation": self.formation.value,
            "total_tool_calls": self.total_tool_calls,
            "total_duration": self.total_duration,
            "consensus_achieved": self.consensus_achieved,
            "consensus_rounds": self.consensus_rounds,
            "error": self.error,
        }


# Type aliases for backward compatibility
TeamFormationType = TeamFormation
MessageTypeEnum = MessageType
