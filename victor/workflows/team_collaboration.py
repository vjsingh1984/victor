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

"""Advanced collaboration features for team members.

This module provides sophisticated collaboration capabilities for multi-agent teams,
including member-to-member communication, shared context management, and negotiation
frameworks for consensus building.

Key Features:
- TeamCommunicationProtocol: Message passing with request/response patterns
- SharedTeamContext: Shared memory with conflict resolution
- NegotiationFramework: Voting and consensus building mechanisms

Example:
    from victor.workflows.team_collaboration import (
        TeamCommunicationProtocol,
        SharedTeamContext,
        NegotiationFramework,
    )

    # Enable communication
    comm = TeamCommunicationProtocol(members=[member1, member2])
    response = await comm.send_request(
        sender_id="member1",
        recipient_id="member2",
        content="Please analyze this code",
    )

    # Share context with conflict resolution
    context = SharedTeamContext(
        keys=["findings", "decisions"],
        conflict_resolution="merge"
    )
    await context.set("findings", {"bugs": ["bug1", "bug2"]}, member_id="member1")
    await context.merge("findings", {"bugs": ["bug3"]}, member_id="member2")

    # Negotiate decisions
    negotiation = NegotiationFramework(
        members=members,
        voting_strategy="weighted_by_expertise"
    )
    result = await negotiation.negotiate(
        proposal="Use Python for implementation",
        max_rounds=3
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from victor.teams.types import AgentMessage, TeamMember

logger = logging.getLogger(__name__)


# =============================================================================
# Team Communication Protocol
# =============================================================================


class CommunicationType(str, Enum):
    """Communication patterns for team messaging."""

    REQUEST_RESPONSE = "request_response"  # Direct request-response
    BROADCAST = "broadcast"  # One-to-all messaging
    MULTICAST = "multicast"  # One-to-many messaging
    PUBSUB = "pubsub"  # Publish-subscribe pattern


@dataclass
class CommunicationLog:
    """Log entry for inter-team communication.

    Attributes:
        timestamp: When the communication occurred
        message_type: Type of message (MessageType enum)
        sender_id: ID of the sender
        recipient_id: ID of the recipient (None for broadcast)
        content: Message content
        communication_type: Type of communication pattern
        metadata: Additional metadata
        response_id: ID of the response message (if any)
        duration_ms: Time to receive response (for request-response)
    """

    timestamp: float
    message_type: str
    sender_id: str
    recipient_id: Optional[str]
    content: str
    communication_type: CommunicationType
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_id: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "message_type": self.message_type,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "content": self.content,
            "communication_type": self.communication_type.value,
            "metadata": self.metadata,
            "response_id": self.response_id,
            "duration_ms": self.duration_ms,
        }


class TeamCommunicationProtocol:
    """Enable member-to-member communication within teams.

    This protocol provides multiple communication patterns:
    - Request/Response: Direct messaging with replies
    - Broadcast: One-to-all messaging
    - Multicast: One-to-many messaging
    - Pub/Sub: Publish-subscribe pattern

    All communications are logged for observability and debugging.

    Attributes:
        _members: List of team members
        _communication_type: Default communication pattern
        _log_messages: Whether to log all messages
        _communication_log: List of all communications
        _message_handlers: Message handlers for pub/sub
        _pending_requests: Pending request-response messages

    Example:
        # Request-response pattern
        protocol = TeamCommunicationProtocol(
            members=[member1, member2],
            communication_type="request_response",
            log_messages=True
        )

        response = await protocol.send_request(
            sender_id="member1",
            recipient_id="member2",
            content="Please review this code"
        )

        # Broadcast pattern
        await protocol.broadcast(
            sender_id="coordinator",
            content="Team meeting in 5 minutes"
        )

        # Pub/sub pattern
        protocol.subscribe("security_alerts", member1)
        await protocol.publish(
            topic="security_alerts",
            message="Critical vulnerability found"
        )
    """

    def __init__(
        self,
        members: List[Any],
        communication_type: CommunicationType = CommunicationType.REQUEST_RESPONSE,
        log_messages: bool = True,
    ):
        """Initialize communication protocol.

        Args:
            members: List of team members (must have receive_message method)
            communication_type: Default communication pattern
            log_messages: Whether to log all communications
        """
        self._members = {m.id: m for m in members}
        self._communication_type = communication_type
        self._log_messages = log_messages
        self._communication_log: List[CommunicationLog] = []
        self._message_handlers: Dict[str, List[Any]] = {}  # topic -> subscribers
        self._pending_requests: Dict[str, asyncio.Future] = {}

    async def send_request(
        self,
        sender_id: str,
        recipient_id: str,
        content: str,
        message_type: str = "request",
        timeout: float = 30.0,
        **metadata,
    ) -> Optional[AgentMessage]:
        """Send a direct request and wait for response.

        Args:
            sender_id: ID of the sender
            recipient_id: ID of the recipient
            content: Request content
            message_type: Type of message
            timeout: Maximum time to wait for response (seconds)
            **metadata: Additional metadata

        Returns:
            Response message, or None if timeout/error

        Example:
            response = await protocol.send_request(
                sender_id="member1",
                recipient_id="member2",
                content="What's your analysis?",
                timeout=10.0
            )
        """
        from victor.teams.types import AgentMessage, MessageType

        start_time = time.time()

        # Create request message
        request = AgentMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            message_type=MessageType.REQUEST,
            data=metadata,
        )

        # Find recipient
        recipient = self._members.get(recipient_id)
        if not recipient:
            logger.warning(f"Recipient {recipient_id} not found")
            return None

        try:
            # Send request and wait for response
            response = await asyncio.wait_for(
                recipient.receive_message(request),
                timeout=timeout,
            )

            # Log communication
            if self._log_messages:
                duration_ms = (time.time() - start_time) * 1000
                self._communication_log.append(
                    CommunicationLog(
                        timestamp=start_time,
                        message_type=message_type,
                        sender_id=sender_id,
                        recipient_id=recipient_id,
                        content=content,
                        communication_type=self._communication_type,
                        metadata=metadata,
                        response_id=response.id if response else None,
                        duration_ms=duration_ms,
                    )
                )

            return response

        except asyncio.TimeoutError:
            logger.warning(f"Request from {sender_id} to {recipient_id} timed out")
            if self._log_messages:
                duration_ms = (time.time() - start_time) * 1000
                self._communication_log.append(
                    CommunicationLog(
                        timestamp=start_time,
                        message_type=message_type,
                        sender_id=sender_id,
                        recipient_id=recipient_id,
                        content=content,
                        communication_type=self._communication_type,
                        metadata=metadata,
                        duration_ms=duration_ms,
                    )
                )
            return None

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    async def broadcast(
        self,
        sender_id: str,
        content: str,
        message_type: str = "broadcast",
        exclude_sender: bool = True,
        **metadata,
    ) -> List[Optional[AgentMessage]]:
        """Broadcast message to all team members.

        Args:
            sender_id: ID of the sender
            content: Message content
            message_type: Type of message
            exclude_sender: Whether to exclude sender from recipients
            **metadata: Additional metadata

        Returns:
            List of responses from all members

        Example:
            responses = await protocol.broadcast(
                sender_id="coordinator",
                content="Update on project status"
            )
        """
        from victor.teams.types import AgentMessage, MessageType

        message = AgentMessage(
            sender_id=sender_id,
            recipient_id=None,  # None = broadcast
            content=content,
            message_type=MessageType.STATUS,
            data=metadata,
        )

        responses: List[Optional[AgentMessage]] = []

        for member_id, member in self._members.items():
            if exclude_sender and member_id == sender_id:
                continue

            try:
                response = await member.receive_message(message)
                responses.append(response)

                # Log communication
                if self._log_messages:
                    self._communication_log.append(
                        CommunicationLog(
                            timestamp=time.time(),
                            message_type=message_type,
                            sender_id=sender_id,
                            recipient_id=member_id,
                            content=content,
                            communication_type=CommunicationType.BROADCAST,
                            metadata=metadata,
                            response_id=response.id if response else None,
                        )
                    )

            except Exception as e:
                logger.warning(f"Broadcast to {member_id} failed: {e}")
                responses.append(None)

        return responses

    async def multicast(
        self,
        sender_id: str,
        recipient_ids: List[str],
        content: str,
        message_type: str = "multicast",
        **metadata,
    ) -> Dict[str, Optional[AgentMessage]]:
        """Send message to multiple specific recipients.

        Args:
            sender_id: ID of the sender
            recipient_ids: List of recipient IDs
            content: Message content
            message_type: Type of message
            **metadata: Additional metadata

        Returns:
            Dictionary mapping recipient_id to response

        Example:
            responses = await protocol.multicast(
                sender_id="manager",
                recipient_ids=["member1", "member2"],
                content="Please submit your reports"
            )
        """
        from victor.teams.types import AgentMessage, MessageType

        message = AgentMessage(
            sender_id=sender_id,
            recipient_id=None,  # Multicast
            content=content,
            message_type=MessageType.QUERY,
            data=metadata,
        )

        responses: Dict[str, Optional[AgentMessage]] = {}

        tasks = []
        for recipient_id in recipient_ids:
            recipient = self._members.get(recipient_id)
            if not recipient:
                responses[recipient_id] = None
                continue

            async def send_and_log(rec_id, rec):
                try:
                    response = await rec.receive_message(message)
                    if self._log_messages:
                        self._communication_log.append(
                            CommunicationLog(
                                timestamp=time.time(),
                                message_type=message_type,
                                sender_id=sender_id,
                                recipient_id=rec_id,
                                content=content,
                                communication_type=CommunicationType.MULTICAST,
                                metadata=metadata,
                                response_id=response.id if response else None,
                            )
                        )
                    return rec_id, response
                except Exception as e:
                    logger.warning(f"Multicast to {rec_id} failed: {e}")
                    return rec_id, None

            tasks.append(send_and_log(recipient_id, recipient))

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                continue
            recipient_id, response = result
            responses[recipient_id] = response

        return responses

    def subscribe(self, topic: str, member: Any) -> None:
        """Subscribe a member to a topic (pub/sub pattern).

        Args:
            topic: Topic to subscribe to
            member: Member to subscribe

        Example:
            protocol.subscribe("security_alerts", security_expert)
        """
        if topic not in self._message_handlers:
            self._message_handlers[topic] = []
        self._message_handlers[topic].append(member)

    def unsubscribe(self, topic: str, member: Any) -> None:
        """Unsubscribe a member from a topic.

        Args:
            topic: Topic to unsubscribe from
            member: Member to unsubscribe
        """
        if topic in self._message_handlers:
            self._message_handlers[topic] = [
                m for m in self._message_handlers[topic] if m.id != member.id
            ]

    async def publish(self, topic: str, message: str, sender_id: str = "system") -> int:
        """Publish message to all subscribers of a topic.

        Args:
            topic: Topic to publish to
            message: Message content
            sender_id: ID of the sender

        Returns:
            Number of subscribers notified

        Example:
            count = await protocol.publish(
                topic="security_alerts",
                message="New vulnerability found",
                sender_id="scanner"
            )
        """
        from victor.teams.types import AgentMessage, MessageType

        subscribers = self._message_handlers.get(topic, [])
        if not subscribers:
            return 0

        agent_message = AgentMessage(
            sender_id=sender_id,
            content=message,
            message_type=MessageType.ALERT,
            data={"topic": topic},
        )

        # Send to all subscribers
        tasks = [sub.receive_message(agent_message) for sub in subscribers]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Log communication
        if self._log_messages:
            for sub in subscribers:
                self._communication_log.append(
                    CommunicationLog(
                        timestamp=time.time(),
                        message_type="publish",
                        sender_id=sender_id,
                        recipient_id=sub.id,
                        content=message,
                        communication_type=CommunicationType.PUBSUB,
                        metadata={"topic": topic},
                    )
                )

        return len(subscribers)

    def get_communication_log(self) -> List[CommunicationLog]:
        """Get all communication logs.

        Returns:
            List of all logged communications

        Example:
            logs = protocol.get_communication_log()
            for log in logs:
                print(f"{log.sender_id} -> {log.recipient_id}: {log.content}")
        """
        return self._communication_log.copy()

    def clear_communication_log(self) -> None:
        """Clear all communication logs."""
        self._communication_log.clear()

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get statistics about team communications.

        Returns:
            Dictionary with communication statistics

        Example:
            stats = protocol.get_communication_stats()
            print(f"Total messages: {stats['total_messages']}")
            print(f"Average response time: {stats['avg_response_time_ms']}ms")
        """
        if not self._communication_log:
            return {
                "total_messages": 0,
                "by_type": {},
                "by_sender": {},
                "avg_response_time_ms": 0,
            }

        by_type: Dict[str, int] = {}
        by_sender: Dict[str, int] = {}
        total_response_time = 0
        response_count = 0

        for log in self._communication_log:
            by_type[log.message_type] = by_type.get(log.message_type, 0) + 1
            by_sender[log.sender_id] = by_sender.get(log.sender_id, 0) + 1
            if log.duration_ms:
                total_response_time += log.duration_ms
                response_count += 1

        return {
            "total_messages": len(self._communication_log),
            "by_type": by_type,
            "by_sender": by_sender,
            "avg_response_time_ms": (
                total_response_time / response_count if response_count > 0 else 0
            ),
        }


# =============================================================================
# Shared Team Context
# =============================================================================


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts in shared context."""

    LAST_WRITE_WINS = "last_write_wins"  # Most recent write wins
    MERGE = "merge"  # Merge dictionaries and lists
    FIRST_WRITE_WINS = "first_write_wins"  # First write wins
    CUSTOM = "custom"  # Use custom resolver function


@dataclass
class ContextUpdate:
    """Record of a context update.

    Attributes:
        key: Context key that was updated
        value: New value
        member_id: ID of member who made the update
        timestamp: When the update occurred
        operation: Type of operation (set, merge, delete)
        previous_value: Previous value (if any)
    """

    key: str
    value: Any
    member_id: str
    timestamp: float
    operation: str  # "set", "merge", "delete"
    previous_value: Any = None


class SharedTeamContext:
    """Shared memory/workspace for team collaboration.

    Provides a shared key-value store with:
    - State synchronization across members
    - Conflict resolution strategies
    - Persistence across member executions
    - Update history and rollback

    Attributes:
        _keys: Allowed keys in shared context
        _conflict_resolution: Strategy for resolving conflicts
        _state: Current state
        _update_history: History of all updates
        _custom_resolver: Optional custom conflict resolver

    Example:
        # Create shared context with merge conflict resolution
        context = SharedTeamContext(
            keys=["findings", "decisions", "status"],
            conflict_resolution="merge"
        )

        # Member 1 adds findings
        await context.set(
            "findings",
            {"bugs": ["bug1", "bug2"]},
            member_id="member1"
        )

        # Member 2 merges additional findings
        await context.merge(
            "findings",
            {"bugs": ["bug3"], "performance": ["slow_query"]},
            member_id="member2"
        )

        # Get current state
        findings = await context.get("findings")
        # Returns: {"bugs": ["bug1", "bug2", "bug3"], "performance": ["slow_query"]}

        # Get update history
        history = context.get_update_history()
    """

    def __init__(
        self,
        keys: Optional[List[str]] = None,
        conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS,
        custom_resolver: Optional[Callable[[str, Any, Any, str], Any]] = None,
    ):
        """Initialize shared team context.

        Args:
            keys: Allowed keys (None = any key allowed)
            conflict_resolution: Strategy for resolving conflicts
            custom_resolver: Custom conflict resolution function
                Signature: (key, existing_value, new_value, member_id) -> resolved_value
        """
        self._keys = set(keys) if keys else None
        self._conflict_resolution = conflict_resolution
        self._custom_resolver = custom_resolver
        self._state: Dict[str, Any] = {}
        self._update_history: List[ContextUpdate] = []
        self._lock = asyncio.Lock()

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared context.

        Args:
            key: Key to retrieve
            default: Default value if key doesn't exist

        Returns:
            Value associated with key, or default

        Example:
            findings = await context.get("findings", {})
        """
        async with self._lock:
            return self._state.get(key, default)

    async def set(self, key: str, value: Any, member_id: str) -> bool:
        """Set a value in shared context.

        Args:
            key: Key to set
            value: Value to store
            member_id: ID of member making the update

        Returns:
            True if successful, False otherwise

        Example:
            await context.set("findings", {"bugs": ["bug1"]}, member_id="member1")
        """
        async with self._lock:
            # Check if key is allowed
            if self._keys and key not in self._keys:
                logger.warning(f"Key {key} not in allowed keys")
                return False

            # Get previous value
            previous_value = self._state.get(key)

            # Resolve conflict based on strategy
            if (
                previous_value is not None
                and self._conflict_resolution != ConflictResolutionStrategy.LAST_WRITE_WINS
            ):
                resolved_value = await self._resolve_conflict(key, previous_value, value, member_id)
            else:
                resolved_value = value

            # Set value
            self._state[key] = resolved_value

            # Record update
            self._update_history.append(
                ContextUpdate(
                    key=key,
                    value=resolved_value,
                    member_id=member_id,
                    timestamp=time.time(),
                    operation="set",
                    previous_value=previous_value,
                )
            )

            return True

    async def merge(self, key: str, value: Any, member_id: str) -> bool:
        """Merge a value into existing value (for dicts and lists).

        Args:
            key: Key to merge into
            value: Value to merge
            member_id: ID of member making the update

        Returns:
            True if successful, False otherwise

        Example:
            await context.merge(
                "findings",
                {"bugs": ["bug3"]},
                member_id="member2"
            )
        """
        async with self._lock:
            # Check if key is allowed
            if self._keys and key not in self._keys:
                logger.warning(f"Key {key} not in allowed keys")
                return False

            # Get existing value
            existing = self._state.get(key)
            if existing is None:
                # No existing value, just set
                return await self.set(key, value, member_id)

            # Perform merge
            merged_value = await self._merge_values(existing, value)

            # Set merged value
            self._state[key] = merged_value

            # Record update
            self._update_history.append(
                ContextUpdate(
                    key=key,
                    value=merged_value,
                    member_id=member_id,
                    timestamp=time.time(),
                    operation="merge",
                    previous_value=existing,
                )
            )

            return True

    async def delete(self, key: str, member_id: str) -> bool:
        """Delete a key from shared context.

        Args:
            key: Key to delete
            member_id: ID of member making the update

        Returns:
            True if successful, False otherwise

        Example:
            await context.delete("temp_data", member_id="member1")
        """
        async with self._lock:
            if key not in self._state:
                return False

            # Get previous value
            previous_value = self._state.get(key)

            # Delete
            del self._state[key]

            # Record update
            self._update_history.append(
                ContextUpdate(
                    key=key,
                    value=None,
                    member_id=member_id,
                    timestamp=time.time(),
                    operation="delete",
                    previous_value=previous_value,
                )
            )

            return True

    async def _resolve_conflict(self, key: str, existing: Any, new: Any, member_id: str) -> Any:
        """Resolve conflict based on strategy.

        Args:
            key: Key being updated
            existing: Existing value
            new: New value
            member_id: ID of member making the update

        Returns:
            Resolved value
        """
        if self._conflict_resolution == ConflictResolutionStrategy.FIRST_WRITE_WINS:
            # Keep existing value
            return existing

        elif self._conflict_resolution == ConflictResolutionStrategy.MERGE:
            # Try to merge
            return await self._merge_values(existing, new)

        elif self._conflict_resolution == ConflictResolutionStrategy.CUSTOM:
            # Use custom resolver
            if self._custom_resolver:
                return self._custom_resolver(key, existing, new, member_id)
            return new

        else:  # LAST_WRITE_WINS
            return new

    async def _merge_values(self, existing: Any, new: Any) -> Any:
        """Merge two values intelligently.

        Args:
            existing: Existing value
            new: New value to merge

        Returns:
            Merged value
        """
        # If both are dicts, merge them
        if isinstance(existing, dict) and isinstance(new, dict):
            merged = existing.copy()
            for k, v in new.items():
                if k in merged and isinstance(merged[k], list) and isinstance(v, list):
                    # Merge lists
                    merged[k] = list(set(merged[k] + v))
                elif k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                    # Recursively merge dicts
                    merged[k] = await self._merge_values(merged[k], v)
                else:
                    # Overwrite
                    merged[k] = v
            return merged

        # If both are lists, merge them
        elif isinstance(existing, list) and isinstance(new, list):
            return list(set(existing + new))

        # Otherwise, new value wins
        return new

    def get_state(self) -> Dict[str, Any]:
        """Get current state (snapshot).

        Returns:
            Copy of current state

        Example:
            state = context.get_state()
        """
        return self._state.copy()

    def get_update_history(self, key: Optional[str] = None) -> List[ContextUpdate]:
        """Get update history.

        Args:
            key: Optional key to filter by

        Returns:
            List of updates

        Example:
            all_updates = context.get_update_history()
            findings_updates = context.get_update_history("findings")
        """
        if key:
            return [u for u in self._update_history if u.key == key]
        return self._update_history.copy()

    async def rollback(self, to_timestamp: float) -> bool:
        """Rollback state to a specific timestamp.

        Args:
            to_timestamp: Timestamp to rollback to

        Returns:
            True if successful, False otherwise

        Example:
            # Rollback to 5 minutes ago
            import time
            await context.rollback(time.time() - 300)
        """
        async with self._lock:
            # Find all updates before timestamp
            relevant_updates = [u for u in self._update_history if u.timestamp <= to_timestamp]

            # Rebuild state
            self._state.clear()
            for update in relevant_updates:
                if update.operation == "delete":
                    if update.key in self._state:
                        del self._state[update.key]
                else:
                    self._state[update.key] = update.value

            # Trim history
            self._update_history = relevant_updates

            return True

    def clear(self) -> None:
        """Clear all state and history."""
        self._state.clear()
        self._update_history.clear()


# =============================================================================
# Negotiation Framework
# =============================================================================


class VotingStrategy(str, Enum):
    """Voting strategies for negotiation."""

    MAJORITY = "majority"  # Simple majority (>50%)
    WEIGHTED_BY_EXPERTISE = "weighted_by_expertise"  # Weighted by expertise
    UNANIMOUS = "unanimous"  # All must agree
    CONSENSUS = "consensus"  # Build consensus through discussion


class NegotiationType(str, Enum):
    """Types of negotiation."""

    VOTING = "voting"  # Direct voting
    COMPROMISE = "compromise"  # Find middle ground
    RANKED_CHOICE = "ranked_choice"  # Ranked choice voting


@dataclass
class Proposal:
    """A proposal for team negotiation.

    Attributes:
        id: Unique proposal ID
        content: Proposal description
        proposer_id: ID of member proposing
        timestamp: When proposed
        metadata: Additional metadata
        votes: Votes received (for voting)
        rank: Rank in ranked choice (for ranked choice)
    """

    id: str
    content: str
    proposer_id: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    votes: Dict[str, Any] = field(default_factory=dict)
    rank: Optional[int] = None


@dataclass
class NegotiationResult:
    """Result of a negotiation.

    Attributes:
        success: Whether negotiation succeeded
        agreed_proposal: The proposal that was agreed upon
        votes: Final vote counts
        rounds: Number of rounds needed
        consensus_achieved: Whether full consensus was achieved
        metadata: Additional metadata
    """

    success: bool
    agreed_proposal: Optional[Proposal]
    votes: Dict[str, Any]
    rounds: int
    consensus_achieved: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class NegotiationFramework:
    """Member negotiation and consensus building framework.

    Provides multiple mechanisms for teams to reach decisions:
    - Voting: Majority, weighted, or unanimous
    - Compromise: Find middle ground between proposals
    - Ranked choice: Iterative elimination

    Attributes:
        _members: List of team members
        _voting_strategy: How to weight votes
        _negotiation_type: Type of negotiation
        _max_rounds: Maximum negotiation rounds
        _expertise_weights: Member expertise weights

    Example:
        # Weighted voting setup
        negotiation = NegotiationFramework(
            members=[member1, member2, member3],
            voting_strategy="weighted_by_expertise",
            max_rounds=3
        )

        # Set expertise weights
        negotiation.set_expertise_weights({
            "senior_dev": 3.0,
            "dev": 2.0,
            "junior_dev": 1.0
        })

        # Run negotiation
        result = await negotiation.negotiate(
            proposals=["Use Python", "Use JavaScript", "Use Go"],
            topic="Implementation language"
        )

        if result.success:
            print(f"Agreed on: {result.agreed_proposal.content}")
            print(f"Rounds: {result.rounds}")
    """

    def __init__(
        self,
        members: List[Any],
        voting_strategy: VotingStrategy = VotingStrategy.MAJORITY,
        negotiation_type: NegotiationType = NegotiationType.VOTING,
        max_rounds: int = 3,
    ):
        """Initialize negotiation framework.

        Args:
            members: List of team members
            voting_strategy: How to weight votes
            negotiation_type: Type of negotiation to use
            max_rounds: Maximum negotiation rounds before giving up
        """
        self._members = {m.id: m for m in members}
        self._voting_strategy = voting_strategy
        self._negotiation_type = negotiation_type
        self._max_rounds = max_rounds
        self._expertise_weights: Dict[str, float] = {}

    def set_expertise_weights(self, weights: Dict[str, float]) -> None:
        """Set expertise weights for members.

        Args:
            weights: Dictionary mapping member_id to weight

        Example:
            negotiation.set_expertise_weights({
                "senior_dev": 3.0,
                "dev": 2.0,
                "junior_dev": 1.0
            })
        """
        self._expertise_weights = weights.copy()

    async def negotiate(
        self,
        proposals: List[str],
        topic: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> NegotiationResult:
        """Run a negotiation process.

        Args:
            proposals: List of proposal descriptions
            topic: Topic being negotiated
            context: Additional context for members

        Returns:
            NegotiationResult with outcome

        Example:
            result = await negotiation.negotiate(
                proposals=["Use Python", "Use JavaScript"],
                topic="Implementation language"
            )
        """
        context = context or {}

        # Create proposal objects
        proposal_objs = [
            Proposal(
                id=uuid.uuid4().hex[:8],
                content=prop,
                proposer_id="system",
                timestamp=time.time(),
            )
            for prop in proposals
        ]

        # Run negotiation based on type
        if self._negotiation_type == NegotiationType.VOTING:
            return await self._voting_negotiation(proposal_objs, topic, context)
        elif self._negotiation_type == NegotiationType.COMPROMISE:
            return await self._compromise_negotiation(proposal_objs, topic, context)
        elif self._negotiation_type == NegotiationType.RANKED_CHOICE:
            return await self._ranked_choice_negotiation(proposal_objs, topic, context)
        else:
            # Default to voting
            return await self._voting_negotiation(proposal_objs, topic, context)

    async def _voting_negotiation(
        self,
        proposals: List[Proposal],
        topic: str,
        context: Dict[str, Any],
    ) -> NegotiationResult:
        """Run voting-based negotiation."""
        from victor.teams.types import AgentMessage, MessageType

        rounds = 0
        max_rounds = self._max_rounds

        while rounds < max_rounds:
            rounds += 1

            # Collect votes
            votes: Dict[str, Dict[str, Any]] = {prop.id: {} for prop in proposals}

            for member_id, member in self._members.items():
                # Send voting request
                message = AgentMessage(
                    sender_id="negotiator",
                    recipient_id=member_id,
                    content=f"Please vote on: {topic}\nProposals:\n"
                    + "\n".join(f"{i+1}. {prop.content}" for i, prop in enumerate(proposals)),
                    message_type=MessageType.QUERY,
                    data={"round": rounds, "topic": topic},
                )

                try:
                    response = await member.receive_message(message)
                    if response and response.content:
                        # Parse vote (simple format: "1" or "proposal text")
                        vote = self._parse_vote(response.content, proposals)
                        if vote:
                            # Get weight
                            weight = self._expertise_weights.get(member_id, 1.0)
                            votes[vote.id][member_id] = weight
                except Exception as e:
                    logger.warning(f"Failed to get vote from {member_id}: {e}")

            # Tally votes
            winner_id = self._tally_votes(votes)

            # Check if we have a winner
            if winner_id:
                winner = next((p for p in proposals if p.id == winner_id), None)
                if winner:
                    return NegotiationResult(
                        success=True,
                        agreed_proposal=winner,
                        votes=votes,
                        rounds=rounds,
                        consensus_achieved=self._check_consensus(votes, winner_id),
                        metadata={"voting_strategy": self._voting_strategy.value},
                    )

        # No agreement reached
        return NegotiationResult(
            success=False,
            agreed_proposal=None,
            votes=votes,
            rounds=rounds,
            consensus_achieved=False,
            metadata={"reason": "max_rounds_exceeded"},
        )

    async def _compromise_negotiation(
        self,
        proposals: List[Proposal],
        topic: str,
        context: Dict[str, Any],
    ) -> NegotiationResult:
        """Run compromise-based negotiation."""
        # Get preferences from all members
        preferences: Dict[str, List[str]] = {}

        for member_id, member in self._members.items():
            from victor.teams.types import AgentMessage, MessageType

            message = AgentMessage(
                sender_id="negotiator",
                recipient_id=member_id,
                content=f"Rank proposals for: {topic}\n"
                + "\n".join(f"{i+1}. {prop.content}" for i, prop in enumerate(proposals)),
                message_type=MessageType.QUERY,
                data={"topic": topic},
            )

            try:
                response = await member.receive_message(message)
                if response and response.content:
                    # Parse rankings
                    rankings = self._parse_rankings(response.content, proposals)
                    preferences[member_id] = rankings
            except Exception as e:
                logger.warning(f"Failed to get preferences from {member_id}: {e}")

        # Find compromise (proposal with highest average rank)
        if not preferences:
            return NegotiationResult(
                success=False,
                agreed_proposal=None,
                votes={},
                rounds=1,
                consensus_achieved=False,
                metadata={"reason": "no_preferences"},
            )

        # Calculate average rank for each proposal
        avg_ranks: Dict[str, float] = {}
        for prop in proposals:
            ranks = [
                preferences[m].index(prop.content) + 1
                for m in preferences
                if prop.content in preferences[m]
            ]
            if ranks:
                avg_ranks[prop.id] = sum(ranks) / len(ranks)

        # Find best (lowest average rank)
        best_id = min(avg_ranks.keys(), key=lambda k: avg_ranks[k])
        best_proposal = next((p for p in proposals if p.id == best_id), None)

        if best_proposal:
            return NegotiationResult(
                success=True,
                agreed_proposal=best_proposal,
                votes=avg_ranks,
                rounds=1,
                consensus_achieved=False,  # Compromise, not true consensus
                metadata={"negotiation_type": "compromise", "avg_ranks": avg_ranks},
            )

        return NegotiationResult(
            success=False,
            agreed_proposal=None,
            votes={},
            rounds=1,
            consensus_achieved=False,
            metadata={"reason": "no_compromise"},
        )

    async def _ranked_choice_negotiation(
        self,
        proposals: List[Proposal],
        topic: str,
        context: Dict[str, Any],
    ) -> NegotiationResult:
        """Run ranked choice negotiation."""
        from victor.teams.types import AgentMessage, MessageType

        rounds = 0
        active_proposals = proposals.copy()

        while len(active_proposals) > 1 and rounds < self._max_rounds:
            rounds += 1

            # Get rankings for active proposals
            rankings: Dict[str, List[str]] = {}

            for member_id, member in self._members.items():
                message = AgentMessage(
                    sender_id="negotiator",
                    recipient_id=member_id,
                    content=f"Rank remaining proposals for: {topic}\n"
                    + "\n".join(
                        f"{i+1}. {prop.content}" for i, prop in enumerate(active_proposals)
                    ),
                    message_type=MessageType.QUERY,
                    data={"round": rounds, "topic": topic},
                )

                try:
                    response = await member.receive_message(message)
                    if response and response.content:
                        member_rankings = self._parse_rankings(response.content, active_proposals)
                        rankings[member_id] = member_rankings
                except Exception as e:
                    logger.warning(f"Failed to get rankings from {member_id}: {e}")

            # Count first choices
            first_choices: Dict[str, int] = {}
            for member_id, member_rankings in rankings.items():
                if member_rankings:
                    first_choice = member_rankings[0]
                    first_choices[first_choice] = first_choices.get(first_choice, 0) + 1

            # Check for majority
            total_votes = len(rankings)
            for prop_id, count in first_choices.items():
                if count > total_votes / 2:
                    # Majority winner
                    winner = next((p for p in active_proposals if p.content == prop_id), None)
                    if winner:
                        return NegotiationResult(
                            success=True,
                            agreed_proposal=winner,
                            votes=first_choices,
                            rounds=rounds,
                            consensus_achieved=count == total_votes,
                            metadata={"negotiation_type": "ranked_choice"},
                        )

            # Eliminate lowest-ranked proposal
            if first_choices:
                eliminated_id = min(first_choices.keys(), key=lambda k: first_choices[k])
                active_proposals = [p for p in active_proposals if p.content != eliminated_id]

        # Return remaining proposal
        if active_proposals:
            return NegotiationResult(
                success=True,
                agreed_proposal=active_proposals[0],
                votes={},
                rounds=rounds,
                consensus_achieved=len(active_proposals) == 1,
                metadata={"negotiation_type": "ranked_choice"},
            )

        return NegotiationResult(
            success=False,
            agreed_proposal=None,
            votes={},
            rounds=rounds,
            consensus_achieved=False,
            metadata={"reason": "no_proposals_remaining"},
        )

    def _parse_vote(self, content: str, proposals: List[Proposal]) -> Optional[Proposal]:
        """Parse vote from member response.

        Args:
            content: Response content
            proposals: Available proposals

        Returns:
            Selected proposal, or None
        """
        content = content.strip().lower()

        # Try to match by number
        if content.isdigit():
            idx = int(content) - 1
            if 0 <= idx < len(proposals):
                return proposals[idx]

        # Try to match by text
        for prop in proposals:
            if prop.content.lower() in content or content in prop.content.lower():
                return prop

        return None

    def _parse_rankings(self, content: str, proposals: List[Proposal]) -> List[str]:
        """Parse rankings from member response.

        Args:
            content: Response content
            proposals: Available proposals

        Returns:
            List of proposal contents in ranked order
        """
        lines = content.strip().split("\n")
        rankings = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to extract ranking (e.g., "1. Python" or "Python")
            for prop in proposals:
                if prop.content.lower() in line.lower():
                    rankings.append(prop.content)
                    break

        # If no rankings found, return all proposals
        if not rankings:
            return [p.content for p in proposals]

        return rankings

    def _tally_votes(self, votes: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Tally votes and determine winner.

        Args:
            votes: Dictionary mapping proposal_id to {member_id: weight}

        Returns:
            Winning proposal_id, or None if no winner
        """
        if not votes:
            return None

        # Calculate totals
        totals: Dict[str, float] = {}
        for prop_id, member_votes in votes.items():
            totals[prop_id] = sum(member_votes.values())

        # Check voting strategy
        if self._voting_strategy == VotingStrategy.MAJORITY:
            # Simple majority (highest total wins)
            max_total = max(totals.values())
            winners = [pid for pid, total in totals.items() if total == max_total]

            if len(winners) == 1:
                return winners[0]

        elif self._voting_strategy == VotingStrategy.WEIGHTED_BY_EXPERTISE:
            # Already weighted, just pick highest
            max_total = max(totals.values())
            winners = [pid for pid, total in totals.items() if total == max_total]

            if len(winners) == 1:
                return winners[0]

        elif self._voting_strategy == VotingStrategy.UNANIMOUS:
            # Check if all members voted for same proposal
            if len(totals) == 1:
                return list(totals.keys())[0]

        return None

    def _check_consensus(self, votes: Dict[str, Dict[str, Any]], winner_id: str) -> bool:
        """Check if consensus was achieved (all voted for winner).

        Args:
            votes: Vote dictionary
            winner_id: Winning proposal ID

        Returns:
            True if consensus achieved
        """
        winner_votes = votes.get(winner_id, {})
        total_members = len(self._members)

        return len(winner_votes) == total_members


__all__ = [
    # Communication
    "CommunicationType",
    "CommunicationLog",
    "TeamCommunicationProtocol",
    # Shared Context
    "ConflictResolutionStrategy",
    "ContextUpdate",
    "SharedTeamContext",
    # Negotiation
    "VotingStrategy",
    "NegotiationType",
    "Proposal",
    "NegotiationResult",
    "NegotiationFramework",
]
