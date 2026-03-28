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

"""Chat domain facade for orchestrator decomposition.

Groups conversation, memory, dialogue state, embedding, intent classification,
and response completion components behind a single interface.

This facade wraps already-initialized components from the orchestrator,
providing a coherent grouping without changing initialization ordering.
The orchestrator delegates property access through this facade.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ChatFacade:
    """Groups conversation, memory, and dialogue state components.

    Satisfies ``ChatFacadeProtocol`` structurally.  The orchestrator creates
    this facade after all chat-domain components are initialized, passing
    references to the already-created instances.

    Components managed:
        - conversation: MessageHistory for message storage
        - conversation_controller: ConversationController for dialogue state
        - conversation_state: ConversationStateMachine for stage transitions
        - memory_manager: Optional conversation memory store
        - memory_session_id: Session identifier for memory operations
        - embedding_store: Optional conversation embedding store
        - intent_classifier: Optional intent classifier
        - intent_detector: Action authorizer / intent detection
        - reminder_manager: Context reminder manager
        - system_prompt: Current system prompt text
        - response_completer: Response completion after tool calls
        - context_compactor: Context compaction for long conversations
        - task_completion_detector: Signal-based completion detection
    """

    def __init__(
        self,
        *,
        conversation: Any,
        conversation_controller: Any,
        conversation_state: Any,
        memory_manager: Optional[Any] = None,
        memory_session_id: Optional[str] = None,
        embedding_store: Optional[Any] = None,
        intent_classifier: Optional[Any] = None,
        intent_detector: Optional[Any] = None,
        reminder_manager: Optional[Any] = None,
        system_prompt: str = "",
        response_completer: Optional[Any] = None,
        context_compactor: Optional[Any] = None,
        task_completion_detector: Optional[Any] = None,
    ) -> None:
        self._conversation = conversation
        self._conversation_controller = conversation_controller
        self._conversation_state = conversation_state
        self._memory_manager = memory_manager
        self._memory_session_id = memory_session_id
        self._embedding_store = embedding_store
        self._intent_classifier = intent_classifier
        self._intent_detector = intent_detector
        self._reminder_manager = reminder_manager
        self._system_prompt = system_prompt
        self._response_completer = response_completer
        self._context_compactor = context_compactor
        self._task_completion_detector = task_completion_detector

        logger.debug(
            "ChatFacade initialized (memory=%s, embeddings=%s, intent=%s)",
            memory_manager is not None,
            embedding_store is not None,
            intent_classifier is not None,
        )

    # ------------------------------------------------------------------
    # Properties (satisfy ChatFacadeProtocol)
    # ------------------------------------------------------------------

    @property
    def conversation(self) -> Any:
        """Message history (MessageHistory instance)."""
        return self._conversation

    @property
    def conversation_controller(self) -> Any:
        """ConversationController for dialogue state management."""
        return self._conversation_controller

    @property
    def conversation_state(self) -> Any:
        """ConversationStateMachine for stage transitions."""
        return self._conversation_state

    @property
    def memory_manager(self) -> Optional[Any]:
        """Optional conversation memory store."""
        return self._memory_manager

    @property
    def memory_session_id(self) -> Optional[str]:
        """Session identifier for memory operations."""
        return self._memory_session_id

    @property
    def embedding_store(self) -> Optional[Any]:
        """Optional conversation embedding store."""
        return self._embedding_store

    @embedding_store.setter
    def embedding_store(self, value: Any) -> None:
        """Update the embedding store (set during lazy init)."""
        self._embedding_store = value

    @property
    def intent_classifier(self) -> Optional[Any]:
        """Optional intent classifier."""
        return self._intent_classifier

    @property
    def intent_detector(self) -> Optional[Any]:
        """Action authorizer / intent detection."""
        return self._intent_detector

    @property
    def reminder_manager(self) -> Optional[Any]:
        """Context reminder manager."""
        return self._reminder_manager

    @property
    def system_prompt(self) -> str:
        """Current system prompt text."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Update the system prompt."""
        self._system_prompt = value

    @property
    def response_completer(self) -> Optional[Any]:
        """Response completer for ensuring complete responses after tool calls."""
        return self._response_completer

    @property
    def context_compactor(self) -> Optional[Any]:
        """Context compaction for long conversations."""
        return self._context_compactor

    @context_compactor.setter
    def context_compactor(self, value: Any) -> None:
        """Update the context compactor reference."""
        self._context_compactor = value

    @property
    def task_completion_detector(self) -> Optional[Any]:
        """Signal-based completion detection."""
        return self._task_completion_detector
