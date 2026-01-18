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

"""Conversation coordinator for message management.

This coordinator handles conversation history operations as part of
Track 4 orchestrator extraction (Phase 1).

Responsibilities:
- Retrieve conversation messages
- Add new messages to conversation
- Reset conversation state

Thread Safety:
- All public methods are thread-safe
- Delegates to thread-safe components (MessageHistory, LifecycleManager)
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class ConversationCoordinator:
    """Coordinator for conversation message management.

    This coordinator provides a clean interface for conversation operations,
    delegating to MessageHistory for message storage and LifecycleManager
    for session reset operations.

    Design Principles:
    - Single Responsibility: Only handles conversation message operations
    - Dependency Injection: All dependencies injected via constructor
    - Thread Safety: All operations are thread-safe

    Attributes:
        conversation: MessageHistory instance for message storage
        lifecycle_manager: LifecycleManager instance for session management
        memory_manager_wrapper: Optional memory manager for persistence
        usage_logger: Optional usage logger for analytics
    """

    def __init__(
        self,
        conversation: Any,
        lifecycle_manager: Any,
        memory_manager_wrapper: Optional[Any] = None,
        usage_logger: Optional[Any] = None,
    ):
        """Initialize the conversation coordinator.

        Args:
            conversation: MessageHistory instance for message storage
            lifecycle_manager: LifecycleManager instance for session management
            memory_manager_wrapper: Optional memory manager for persistence
            usage_logger: Optional usage logger for analytics
        """
        self._conversation = conversation
        self._lifecycle_manager = lifecycle_manager
        self._memory_manager_wrapper = memory_manager_wrapper
        self._usage_logger = usage_logger

    @property
    def messages(self) -> List[Any]:
        """Get conversation messages.

        Provides backward-compatible property access to message history.

        Returns:
            List of messages in conversation history
        """
        return self._conversation.messages

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history.

        Delegates to MessageHistory for storage and optionally persists
        to memory manager if enabled.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        # Add to conversation history
        self._conversation.add_message(role, content)

        # Persist to memory manager wrapper if available
        if self._memory_manager_wrapper and self._memory_manager_wrapper.is_enabled:
            self._memory_manager_wrapper.add_message(role, content)

        # Log to usage analytics if available
        if self._usage_logger:
            if role == "user":
                self._usage_logger.log_event("user_prompt", {"content": content})
            elif role == "assistant":
                self._usage_logger.log_event("assistant_response", {"content": content})

        logger.debug(f"Added message: role={role}, content_length={len(content)}")

    def reset_conversation(self) -> None:
        """Clear conversation history and session state.

        Delegates to LifecycleManager for core reset logic, which handles:
        - Conversation history reset
        - Tool call counter reset
        - Failed tool signatures cache clear
        - Observed files list clear
        - Executed tools list clear
        - Conversation state machine reset
        - Context reminder manager reset
        - Metrics collector stats reset
        - Context compactor statistics reset
        - Sequence tracker history reset (preserves learned patterns)
        - Usage analytics session reset (ends current, starts fresh)
        """
        self._lifecycle_manager.reset_conversation()
        logger.debug("Conversation and session state reset (via ConversationCoordinator)")
