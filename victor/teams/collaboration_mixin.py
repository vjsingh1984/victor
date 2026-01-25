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

"""Collaboration mixin for UnifiedTeamCoordinator.

This mixin adds advanced collaboration features to team coordinators,
including member-to-member communication, shared context, and negotiation
capabilities.

The mixin is backward compatible and opt-in via configuration.

Example:
    from victor.teams import UnifiedTeamCoordinator
    from victor.teams.collaboration_mixin import CollaborationMixin

    # Create coordinator with collaboration
    class CollaborativeCoordinator(CollaborationMixin, UnifiedTeamCoordinator):
        pass

    coordinator = CollaborativeCoordinator(orchestrator)
    coordinator.enable_collaboration({
        "communication": {
            "type": "request_response",
            "log_messages": True
        },
        "shared_context": {
            "keys": ["findings", "decisions"],
            "conflict_resolution": "merge"
        },
        "negotiation": {
            "type": "voting",
            "voting_strategy": "weighted_by_expertise",
            "max_rounds": 3
        }
    })

    # Use collaboration features
    await coordinator.collaborative_broadcast(
        sender_id="manager",
        content="Team status update"
    )

    await coordinator.shared_context_set("findings", {"bugs": ["bug1"]}, "member1")

    result = await coordinator.negotiate(
        proposals=["Option A", "Option B"],
        topic="Implementation approach"
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

if TYPE_CHECKING:
    from victor.teams.types import AgentMessage
    from victor.workflows.team_collaboration import (
        CommunicationType,
        ConflictResolutionStrategy,
        NegotiationResult,
        NegotiationType,
        VotingStrategy,
    )

logger = logging.getLogger(__name__)


class CollaborationMixin:
    """Mixin to add collaboration features to team coordinators.

    This mixin provides:
    - Team communication protocol
    - Shared team context
    - Negotiation framework

    It is designed to be mixed into UnifiedTeamCoordinator or other
    coordinator implementations.

    Attributes:
        _collaboration_enabled: Whether collaboration features are enabled
        _communication_protocol: TeamCommunicationProtocol instance
        _shared_context: SharedTeamContext instance
        _negotiation_framework: NegotiationFramework instance

    Example:
        class MyCoordinator(CollaborationMixin, UnifiedTeamCoordinator):
            pass

        coordinator = MyCoordinator(orchestrator)
        coordinator.enable_collaboration(config)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize collaboration mixin."""
        super().__init__(*args, **kwargs)

        # Collaboration state
        self._collaboration_enabled: bool = False
        self._communication_protocol: Optional[Any] = None
        self._shared_context: Optional[Any] = None
        self._negotiation_framework: Optional[Any] = None
        self._collaboration_config: Dict[str, Any] = {}

    # ========================================================================
    # Collaboration Configuration
    # ========================================================================

    def enable_collaboration(self, config: Dict[str, Any]) -> None:
        """Enable collaboration features with configuration.

        Args:
            config: Collaboration configuration dictionary with optional keys:
                - communication: {type, log_messages}
                - shared_context: {keys, conflict_resolution}
                - negotiation: {type, voting_strategy, max_rounds}

        Example:
            coordinator.enable_collaboration({
                "communication": {
                    "type": "request_response",
                    "log_messages": True
                },
                "shared_context": {
                    "keys": ["findings", "decisions"],
                    "conflict_resolution": "merge"
                },
                "negotiation": {
                    "type": "voting",
                    "voting_strategy": "weighted_by_expertise",
                    "max_rounds": 3
                }
            })
        """
        self._collaboration_config = config
        self._collaboration_enabled = True

        # Initialize communication protocol
        comm_config = config.get("communication", {})
        if comm_config.get("enabled", True):
            self._init_communication_protocol(comm_config)

        # Initialize shared context
        context_config = config.get("shared_context", {})
        if context_config.get("enabled", True):
            self._init_shared_context(context_config)

        # Initialize negotiation framework
        neg_config = config.get("negotiation", {})
        if neg_config.get("enabled", True):
            self._init_negotiation_framework(neg_config)

        logger.info("Collaboration features enabled")

    def disable_collaboration(self) -> None:
        """Disable collaboration features."""
        self._collaboration_enabled = False
        self._communication_protocol = None
        self._shared_context = None
        self._negotiation_framework = None
        logger.info("Collaboration features disabled")

    @property
    def collaboration_enabled(self) -> bool:
        """Check if collaboration features are enabled.

        Returns:
            True if collaboration is enabled
        """
        return self._collaboration_enabled

    # ========================================================================
    # Initialization Helpers
    # ========================================================================

    def _init_communication_protocol(self, config: Dict[str, Any]) -> None:
        """Initialize communication protocol.

        Args:
            config: Communication configuration
        """
        from victor.workflows.team_collaboration import CommunicationType, TeamCommunicationProtocol

        # Get communication type
        comm_type_str = config.get("type", "request_response")
        try:
            comm_type = CommunicationType(comm_type_str)
        except ValueError:
            logger.warning(f"Invalid communication type: {comm_type_str}, using request_response")
            comm_type = CommunicationType.REQUEST_RESPONSE

        # Create protocol
        log_messages = config.get("log_messages", True)
        self._communication_protocol = TeamCommunicationProtocol(
            members=getattr(self, 'members', []),
            communication_type=comm_type,
            log_messages=log_messages,
        )

    def _init_shared_context(self, config: Dict[str, Any]) -> None:
        """Initialize shared context.

        Args:
            config: Shared context configuration
        """
        from victor.workflows.team_collaboration import (
            ConflictResolutionStrategy,
            SharedTeamContext,
        )

        # Get conflict resolution strategy
        resolution_str = config.get("conflict_resolution", "last_write_wins")
        try:
            resolution = ConflictResolutionStrategy(resolution_str)
        except ValueError:
            logger.warning(f"Invalid conflict resolution: {resolution_str}, using last_write_wins")
            resolution = ConflictResolutionStrategy.LAST_WRITE_WINS

        # Create shared context
        keys = config.get("keys")
        self._shared_context = SharedTeamContext(
            keys=keys,
            conflict_resolution=resolution,
        )

    def _init_negotiation_framework(self, config: Dict[str, Any]) -> None:
        """Initialize negotiation framework.

        Args:
            config: Negotiation configuration
        """
        from victor.workflows.team_collaboration import (
            NegotiationFramework,
            NegotiationType,
            VotingStrategy,
        )

        # Get voting strategy
        voting_str = config.get("voting_strategy", "majority")
        try:
            voting = VotingStrategy(voting_str)
        except ValueError:
            logger.warning(f"Invalid voting strategy: {voting_str}, using majority")
            voting = VotingStrategy.MAJORITY

        # Get negotiation type
        neg_type_str = config.get("type", "voting")
        try:
            neg_type = NegotiationType(neg_type_str)
        except ValueError:
            logger.warning(f"Invalid negotiation type: {neg_type_str}, using voting")
            neg_type = NegotiationType.VOTING

        # Create framework
        max_rounds = config.get("max_rounds", 3)
        self._negotiation_framework = NegotiationFramework(
            members=getattr(self, 'members', []),
            voting_strategy=voting,
            negotiation_type=neg_type,
            max_rounds=max_rounds,
        )

        # Set expertise weights if provided
        expertise = config.get("expertise_weights")
        if expertise:
            self._negotiation_framework.set_expertise_weights(expertise)

    # ========================================================================
    # Communication Methods
    # ========================================================================

    async def collaborative_send_request(
        self,
        sender_id: str,
        recipient_id: str,
        content: str,
        message_type: str = "request",
        timeout: float = 30.0,
        **metadata: Any,
    ) -> Optional[AgentMessage]:
        """Send a collaborative request between members.

        Args:
            sender_id: ID of the sender
            recipient_id: ID of the recipient
            content: Request content
            message_type: Type of message
            timeout: Maximum time to wait for response
            **metadata: Additional metadata

        Returns:
            Response message, or None if timeout/error

        Example:
            response = await coordinator.collaborative_send_request(
                sender_id="member1",
                recipient_id="member2",
                content="Please analyze this code",
                timeout=10.0
            )
        """
        if not self._collaboration_enabled or not self._communication_protocol:
            logger.warning("Collaboration not enabled")
            return None

        return cast(
            "Optional[AgentMessage]",
            await self._communication_protocol.send_request(
                sender_id=sender_id,
                recipient_id=recipient_id,
                content=content,
                message_type=message_type,
                timeout=timeout,
                **metadata,
            ),
        )

    async def collaborative_broadcast(
        self,
        sender_id: str,
        content: str,
        message_type: str = "broadcast",
        exclude_sender: bool = True,
        **metadata: Any,
    ) -> List[Optional[AgentMessage]]:
        """Broadcast a message to all team members.

        Args:
            sender_id: ID of the sender
            content: Message content
            message_type: Type of message
            exclude_sender: Whether to exclude sender from recipients
            **metadata: Additional metadata

        Returns:
            List of responses from all members

        Example:
            responses = await coordinator.collaborative_broadcast(
                sender_id="manager",
                content="Team meeting in 5 minutes"
            )
        """
        if not self._collaboration_enabled or not self._communication_protocol:
            logger.warning("Collaboration not enabled")
            return []

        return cast(
            "List[Optional[AgentMessage]]",
            await self._communication_protocol.broadcast(
                sender_id=sender_id,
                content=content,
                message_type=message_type,
                exclude_sender=exclude_sender,
                **metadata,
            ),
        )

    async def collaborative_multicast(
        self,
        sender_id: str,
        recipient_ids: List[str],
        content: str,
        message_type: str = "multicast",
        **metadata: Any,
    ) -> Dict[str, Optional[AgentMessage]]:
        """Send a message to multiple specific recipients.

        Args:
            sender_id: ID of the sender
            recipient_ids: List of recipient IDs
            content: Message content
            message_type: Type of message
            **metadata: Additional metadata

        Returns:
            Dictionary mapping recipient_id to response

        Example:
            responses = await coordinator.collaborative_multicast(
                sender_id="manager",
                recipient_ids=["member1", "member2"],
                content="Please submit your reports"
            )
        """
        if not self._collaboration_enabled or not self._communication_protocol:
            logger.warning("Collaboration not enabled")
            return {}

        return cast(
            "Dict[str, Optional[AgentMessage]]",
            await self._communication_protocol.multicast(
                sender_id=sender_id,
                recipient_ids=recipient_ids,
                content=content,
                message_type=message_type,
                **metadata,
            ),
        )

    def subscribe_to_topic(self, topic: str, member: Any) -> None:
        """Subscribe a member to a topic (pub/sub pattern).

        Args:
            topic: Topic to subscribe to
            member: Member to subscribe

        Example:
            coordinator.subscribe_to_topic("security_alerts", security_expert)
        """
        if not self._collaboration_enabled or not self._communication_protocol:
            logger.warning("Collaboration not enabled")
            return

        self._communication_protocol.subscribe(topic, member)

    async def publish_to_topic(self, topic: str, message: str, sender_id: str = "system") -> int:
        """Publish message to all subscribers of a topic.

        Args:
            topic: Topic to publish to
            message: Message content
            sender_id: ID of the sender

        Returns:
            Number of subscribers notified

        Example:
            count = await coordinator.publish_to_topic(
                topic="security_alerts",
                message="New vulnerability found"
            )
        """
        if not self._collaboration_enabled or not self._communication_protocol:
            logger.warning("Collaboration not enabled")
            return 0

        return cast(int, await self._communication_protocol.publish(topic, message, sender_id))

    def get_communication_log(self) -> List[Any]:
        """Get all communication logs.

        Returns:
            List of all logged communications

        Example:
            logs = coordinator.get_communication_log()
            for log in logs:
                print(f"{log.sender_id} -> {log.recipient_id}: {log.content}")
        """
        if not self._collaboration_enabled or not self._communication_protocol:
            return []

        return cast("List[Any]", self._communication_protocol.get_communication_log())

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get statistics about team communications.

        Returns:
            Dictionary with communication statistics

        Example:
            stats = coordinator.get_communication_stats()
            print(f"Total messages: {stats['total_messages']}")
        """
        if not self._collaboration_enabled or not self._communication_protocol:
            return {}

        return cast("Dict[str, Any]", self._communication_protocol.get_communication_stats())

    # ========================================================================
    # Shared Context Methods
    # ========================================================================

    async def shared_context_get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared context.

        Args:
            key: Key to retrieve
            default: Default value if key doesn't exist

        Returns:
            Value associated with key, or default

        Example:
            findings = await coordinator.shared_context_get("findings", {})
        """
        if not self._collaboration_enabled or not self._shared_context:
            logger.warning("Collaboration not enabled")
            return default

        return await self._shared_context.get(key, default)

    async def shared_context_set(self, key: str, value: Any, member_id: str) -> bool:
        """Set a value in shared context.

        Args:
            key: Key to set
            value: Value to store
            member_id: ID of member making the update

        Returns:
            True if successful, False otherwise

        Example:
            await coordinator.shared_context_set(
                "findings",
                {"bugs": ["bug1"]},
                member_id="member1"
            )
        """
        if not self._collaboration_enabled or not self._shared_context:
            logger.warning("Collaboration not enabled")
            return False

        return cast(bool, await self._shared_context.set(key, value, member_id))

    async def shared_context_merge(self, key: str, value: Any, member_id: str) -> bool:
        """Merge a value into existing value.

        Args:
            key: Key to merge into
            value: Value to merge
            member_id: ID of member making the update

        Returns:
            True if successful, False otherwise

        Example:
            await coordinator.shared_context_merge(
                "findings",
                {"bugs": ["bug2"]},
                member_id="member2"
            )
        """
        if not self._collaboration_enabled or not self._shared_context:
            logger.warning("Collaboration not enabled")
            return False

        return cast(bool, await self._shared_context.merge(key, value, member_id))

    async def shared_context_delete(self, key: str, member_id: str) -> bool:
        """Delete a key from shared context.

        Args:
            key: Key to delete
            member_id: ID of member making the update

        Returns:
            True if successful, False otherwise

        Example:
            await coordinator.shared_context_delete("temp_data", member_id="member1")
        """
        if not self._collaboration_enabled or not self._shared_context:
            logger.warning("Collaboration not enabled")
            return False

        return cast(bool, await self._shared_context.delete(key, member_id))

    def get_shared_context_state(self) -> Dict[str, Any]:
        """Get current shared context state.

        Returns:
            Copy of current state

        Example:
            state = coordinator.get_shared_context_state()
        """
        if not self._collaboration_enabled or not self._shared_context:
            return {}

        return cast("Dict[str, Any]", self._shared_context.get_state())

    def get_shared_context_history(self, key: Optional[str] = None) -> List[Any]:
        """Get shared context update history.

        Args:
            key: Optional key to filter by

        Returns:
            List of updates

        Example:
            all_updates = coordinator.get_shared_context_history()
            findings_updates = coordinator.get_shared_context_history("findings")
        """
        if not self._collaboration_enabled or not self._shared_context:
            return []

        return cast("List[Any]", self._shared_context.get_update_history(key))

    async def shared_context_rollback(self, to_timestamp: float) -> bool:
        """Rollback shared context to a specific timestamp.

        Args:
            to_timestamp: Timestamp to rollback to

        Returns:
            True if successful, False otherwise

        Example:
            import time
            await coordinator.shared_context_rollback(time.time() - 300)
        """
        if not self._collaboration_enabled or not self._shared_context:
            logger.warning("Collaboration not enabled")
            return False

        return cast(bool, await self._shared_context.rollback(to_timestamp))

    # ========================================================================
    # Negotiation Methods
    # ========================================================================

    async def negotiate(
        self,
        proposals: List[str],
        topic: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> NegotiationResult:
        """Run a team negotiation.

        Args:
            proposals: List of proposal descriptions
            topic: Topic being negotiated
            context: Additional context for members

        Returns:
            NegotiationResult with outcome

        Example:
            result = await coordinator.negotiate(
                proposals=["Use Python", "Use JavaScript"],
                topic="Implementation language"
            )

            if result.success:
                print(f"Agreed on: {result.agreed_proposal.content}")
        """
        if not self._collaboration_enabled or not self._negotiation_framework:
            logger.warning("Collaboration not enabled")
            from victor.workflows.team_collaboration import NegotiationResult

            return NegotiationResult(
                success=False,
                agreed_proposal=None,
                votes={},
                rounds=0,
                consensus_achieved=False,
                metadata={"reason": "collaboration_not_enabled"},
            )

        return cast(
            "NegotiationResult",
            await self._negotiation_framework.negotiate(proposals, topic, context),
        )

    def set_expertise_weights(self, weights: Dict[str, float]) -> None:
        """Set expertise weights for weighted voting.

        Args:
            weights: Dictionary mapping member_id to weight

        Example:
            coordinator.set_expertise_weights({
                "senior_dev": 3.0,
                "dev": 2.0,
                "junior_dev": 1.0
            })
        """
        if not self._collaboration_enabled or not self._negotiation_framework:
            logger.warning("Collaboration not enabled")
            return

        self._negotiation_framework.set_expertise_weights(weights)

    # ========================================================================
    # Integration with Team Execution
    # ========================================================================

    async def execute_task_with_collaboration(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute task with collaboration features enabled.

        This is a convenience method that combines collaboration setup
        with task execution.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Result dictionary with collaboration metadata

        Example:
            result = await coordinator.execute_task_with_collaboration(
                task="Implement authentication",
                context={"shared_context": {"requirements": [...]} }
            )

            # Access collaboration metadata
            print(f"Messages exchanged: {result['communication_stats']['total_messages']}")
            print(f"Shared context: {result['shared_context']}")
        """
        if not self._collaboration_enabled:
            # Fall back to regular execution
            return cast("Dict[str, Any]", await getattr(self, 'execute_task')(task, context))

        # Execute task
        result = cast("Dict[str, Any]", await getattr(self, 'execute_task')(task, context))

        # Add collaboration metadata
        if self._communication_protocol:
            result["communication_stats"] = self._communication_protocol.get_communication_stats()
        if self._shared_context:
            result["shared_context"] = self._shared_context.get_state()

        return result


__all__ = ["CollaborationMixin"]
