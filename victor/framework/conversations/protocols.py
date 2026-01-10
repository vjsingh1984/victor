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

"""Focused protocols for conversational agent coordination.

This module defines ISP-compliant protocols for multi-turn conversations
between agents. Each protocol has a single, well-defined responsibility.

Protocol Separation:
- ConversationProtocol: Core conversation lifecycle (turn management)
- MessageFormatterProtocol: Message formatting and parsing
- ConversationResultProtocol: Result extraction and aggregation
- ConversationHistoryProtocol: History tracking and retrieval
- ConversationRoutingProtocol: Dynamic routing between agents
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from victor.framework.protocols import OrchestratorProtocol


# =============================================================================
# Data Classes
# =============================================================================


class ConversationParticipant:
    """A participant in a multi-agent conversation."""

    def __init__(
        self,
        agent_id: str,
        role: str,
        capabilities: List[str],
        persona: Optional[str] = None,
    ) -> None:
        """Initialize conversation participant.

        Args:
            agent_id: Unique agent identifier
            role: Participant role (e.g., "moderator", "expert", "critic")
            capabilities: List of agent capabilities/skills
            persona: Optional persona description
        """
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.persona = persona


class ConversationContext:
    """Context for conversation execution."""

    def __init__(
        self,
        conversation_id: str,
        topic: str,
        protocol: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize conversation context.

        Args:
            conversation_id: Unique conversation identifier
            topic: Conversation topic/goal
            protocol: Conversation protocol (e.g., "request_response", "debate")
            metadata: Optional additional context
        """
        self.conversation_id = conversation_id
        self.topic = topic
        self.protocol = protocol
        self.metadata = metadata or {}
        self.turn_count = 0
        self.current_speaker: Optional[str] = None


class ConversationalMessage:
    """A message in a multi-agent conversation.

    Extends canonical AgentMessage from victor/teams/types.py for compatibility.
    """

    def __init__(
        self,
        sender: str,
        content: str,
        message_type: str = "conversation",
        recipients: Optional[List[str]] = None,
        conversation_id: str = "",
        turn_number: int = 0,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize conversational message.

        Args:
            sender: Sender agent ID
            content: Message content
            message_type: Type of message
            recipients: Optional list of recipient IDs
            conversation_id: Conversation identifier
            turn_number: Turn number in conversation
            reply_to: Optional message ID being replied to
            metadata: Optional message metadata
        """
        self.sender = sender
        self.content = content
        self.message_type = message_type
        self.recipients = recipients
        self.conversation_id = conversation_id
        self.turn_number = turn_number
        self.reply_to = reply_to
        self.metadata = metadata or {}
        self.id = f"{conversation_id}_t{turn_number}_{sender}"

    def to_agent_message(self) -> Dict[str, Any]:
        """Convert to canonical AgentMessage format.

        Returns:
            Dictionary compatible with AgentMessage structure
        """
        return {
            "sender_id": self.sender,
            "content": self.content,
            "message_type": self.message_type,
            "recipient_id": self.recipients[0] if self.recipients else None,
            "data": self.metadata,
            "timestamp": None,  # Added by EventBus
        }


class ConversationalTurn:
    """A single turn in a conversation."""

    def __init__(
        self,
        turn_number: int,
        speaker: str,
        message: ConversationalMessage,
        responses: List[ConversationalMessage],
    ) -> None:
        """Initialize conversational turn.

        Args:
            turn_number: Turn number
            speaker: Speaking agent ID
            message: Message from speaker
            responses: Responses from other agents
        """
        self.turn_number = turn_number
        self.speaker = speaker
        self.message = message
        self.responses = responses


# =============================================================================
# Core Protocol: Conversation Lifecycle
# =============================================================================


@runtime_checkable
class ConversationProtocol(Protocol):
    """Core conversation lifecycle management.

    Single Responsibility: Manage conversation flow and turn-taking.

    Key Methods:
    - initialize: Set up conversation with participants
    - get_next_speaker: Determine who speaks next
    - should_continue: Check if conversation should continue
    - execute_turn: Execute a single conversational turn

    This protocol focuses ONLY on turn management and conversation flow.
    Message formatting, result extraction, and history are separate protocols.
    """

    async def initialize(
        self,
        participants: List[ConversationParticipant],
        context: ConversationContext,
    ) -> None:
        """Initialize conversation with participants.

        Args:
            participants: List of conversation participants
            context: Conversation context

        Raises:
            ValueError: If participants or context invalid
        """
        ...

    async def get_next_speaker(
        self,
        context: ConversationContext,
    ) -> Optional[str]:
        """Get next speaker based on conversation protocol.

        Args:
            context: Current conversation context

        Returns:
            Next speaker agent_id, or None if conversation complete
        """
        ...

    async def should_continue(
        self,
        context: ConversationContext,
    ) -> tuple[bool, Optional[str]]:
        """Check if conversation should continue.

        Args:
            context: Current conversation context

        Returns:
            Tuple of (should_continue, reason)
        """
        ...

    async def execute_turn(
        self,
        speaker: str,
        context: ConversationContext,
        orchestrator: OrchestratorProtocol,
    ) -> ConversationalTurn:
        """Execute a single conversational turn.

        Args:
            speaker: Speaking agent ID
            context: Conversation context
            orchestrator: Orchestrator for executing agent logic

        Returns:
            Completed conversational turn
        """
        ...


# =============================================================================
# Message Formatting Protocol
# =============================================================================


@runtime_checkable
class MessageFormatterProtocol(Protocol):
    """Message formatting and parsing.

    Single Responsibility: Format messages for agents and parse responses.

    Key Methods:
    - format_message: Format message for specific recipient
    - parse_response: Parse agent response into ConversationalMessage

    This protocol is SEPARATE from ConversationProtocol because:
    - Different conversation types may need different formatting
    - Formatting can be swapped independently from turn management
    - Testing is easier with separate protocol
    """

    async def format_message(
        self,
        message: ConversationalMessage,
        recipient: str,
        context: ConversationContext,
    ) -> str:
        """Format message for specific recipient.

        Args:
            message: Message to format
            recipient: Recipient agent ID
            context: Conversation context

        Returns:
            Formatted message string
        """
        ...

    async def parse_response(
        self,
        response: str,
        sender: str,
        context: ConversationContext,
    ) -> ConversationalMessage:
        """Parse agent response into ConversationalMessage.

        Args:
            response: Raw response string from agent
            sender: Sender agent ID
            context: Conversation context

        Returns:
            Parsed ConversationalMessage
        """
        ...


# =============================================================================
# Result Extraction Protocol
# =============================================================================


@runtime_checkable
class ConversationResultProtocol(Protocol):
    """Result extraction and aggregation.

    Single Responsibility: Extract final results from conversation.

    Key Methods:
    - get_result: Extract aggregated result from conversation context
    - format_result: Format result for presentation

    This protocol is SEPARATE from ConversationProtocol because:
    - Different result formats may be needed
    - Aggregation logic is independent of turn management
    - Allows different result strategies for different protocols
    """

    async def get_result(
        self,
        context: ConversationContext,
    ) -> Dict[str, Any]:
        """Extract result from conversation context.

        Args:
            context: Conversation context with history

        Returns:
            Aggregated result dictionary
        """
        ...

    async def format_result(
        self,
        result: Dict[str, Any],
        format: str = "summary",
    ) -> str:
        """Format result for presentation.

        Args:
            result: Result dictionary
            format: Format type (e.g., "summary", "detailed", "json")

        Returns:
            Formatted result string
        """
        ...


# =============================================================================
# History Management Protocol
# =============================================================================


@runtime_checkable
class ConversationHistoryProtocol(Protocol):
    """Conversation history tracking and retrieval.

    Single Responsibility: Track and retrieve conversation history.

    Key Methods:
    - add_turn: Add turn to history
    - get_history: Retrieve conversation history
    - get_last_n_turns: Get last N turns
    - clear_history: Clear conversation history

    This protocol is SEPARATE from ConversationProtocol because:
    - History storage can be implemented independently (in-memory, DB, etc.)
    - History queries don't affect conversation flow
    - Different storage strategies can be swapped
    """

    async def add_turn(
        self,
        context: ConversationContext,
        turn: ConversationalTurn,
    ) -> None:
        """Add turn to conversation history.

        Args:
            context: Conversation context
            turn: Turn to add
        """
        ...

    async def get_history(
        self,
        context: ConversationContext,
    ) -> List[ConversationalTurn]:
        """Get full conversation history.

        Args:
            context: Conversation context

        Returns:
            List of all turns
        """
        ...

    async def get_last_n_turns(
        self,
        context: ConversationContext,
        n: int,
    ) -> List[ConversationalTurn]:
        """Get last N turns from conversation.

        Args:
            context: Conversation context
            n: Number of turns to retrieve

        Returns:
            List of last N turns
        """
        ...

    async def clear_history(
        self,
        context: ConversationContext,
    ) -> None:
        """Clear conversation history.

        Args:
            context: Conversation context
        """
        ...


# =============================================================================
# Routing Protocol
# =============================================================================


@runtime_checkable
class ConversationRoutingProtocol(Protocol):
    """Dynamic routing between conversation participants.

    Single Responsibility: Route messages between participants.

    Key Methods:
    - route_message: Route message to appropriate recipients
    - get_recipients: Determine recipients for message
    - should_reply: Check if agent should reply to message

    This protocol is SEPARATE from ConversationProtocol because:
    - Routing logic can be complex and independent
    - Different routing strategies for different conversation types
    - Testing is easier with isolated routing logic
    """

    async def route_message(
        self,
        message: ConversationalMessage,
        participants: List[ConversationParticipant],
        context: ConversationContext,
    ) -> List[str]:
        """Route message to appropriate recipients.

        Args:
            message: Message to route
            participants: All conversation participants
            context: Conversation context

        Returns:
            List of recipient agent IDs
        """
        ...

    async def get_recipients(
        self,
        speaker: str,
        participants: List[ConversationParticipant],
        context: ConversationContext,
    ) -> List[str]:
        """Get recipients for speaker's message.

        Args:
            speaker: Speaking agent ID
            participants: All conversation participants
            context: Conversation context

        Returns:
            List of recipient agent IDs
        """
        ...

    async def should_reply(
        self,
        agent: str,
        message: ConversationalMessage,
        context: ConversationContext,
    ) -> bool:
        """Check if agent should reply to message.

        Args:
            agent: Agent ID
            message: Message to check
            context: Conversation context

        Returns:
            True if agent should reply
        """
        ...
