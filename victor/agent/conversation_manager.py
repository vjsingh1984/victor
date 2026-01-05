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

"""Conversation management facade composing controller, store, and embedding services.

This module provides a unified interface for conversation management by composing:
- ConversationController: In-memory message handling and context management
- ConversationStore: SQLite persistence for session recovery
- ConversationEmbeddingStore: LanceDB-based semantic search

The ConversationManager acts as a Facade, simplifying the interface for
conversation operations while delegating to specialized components.

Part of TD-002: AgentOrchestrator god class refactoring.

Usage:
    from victor.agent.conversation_manager import ConversationManager
    from victor.config.settings import Settings

    manager = ConversationManager(settings, provider, model, system_prompt)
    manager.add_user_message("Hello!")
    manager.add_assistant_message("Hi there!")

    # Get context metrics
    metrics = manager.get_context_metrics()

    # Session persistence
    session_id = manager.session_id
    sessions = manager.get_recent_sessions(limit=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from victor.agent.conversation_controller import (
    ConversationController,
    ConversationConfig,
    ContextMetrics,
)
from victor.agent.conversation_state import ConversationStage

if TYPE_CHECKING:
    from victor.agent.conversation_memory import ConversationStore, ConversationSession
    from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
    from victor.config.settings import Settings
    from victor.storage.embeddings.service import EmbeddingService
    from victor.providers.base import Message


logger = logging.getLogger(__name__)


@dataclass
class ConversationManagerConfig:
    """Configuration for ConversationManager.

    Attributes:
        enable_persistence: Whether to persist conversations to SQLite
        enable_embeddings: Whether to enable LanceDB embedding store
        max_context_chars: Maximum context size in characters
        chars_per_token_estimate: Estimate for token calculation
        enable_stage_tracking: Whether to track conversation stages
        auto_compaction: Whether to automatically compact on overflow
    """

    enable_persistence: bool = True
    enable_embeddings: bool = True
    max_context_chars: int = 200000
    chars_per_token_estimate: int = 3
    enable_stage_tracking: bool = True
    auto_compaction: bool = True


class ConversationManager:
    """Facade for unified conversation management.

    Composes ConversationController, ConversationStore, and ConversationEmbeddingStore
    to provide a simplified interface for conversation operations.

    This class is part of the TD-002 refactoring effort to extract conversation
    management concerns from AgentOrchestrator.

    Responsibilities:
    - Message addition with persistence
    - Context metrics and overflow detection
    - Session management (create, recover, list)
    - Stage tracking delegation
    - Lazy embedding store initialization

    Example:
        manager = ConversationManager(
            settings=settings,
            provider="anthropic",
            model="claude-3-sonnet",
            system_prompt="You are a helpful assistant.",
        )

        manager.add_user_message("Hello!")
        response = await some_llm_call(manager.messages)
        manager.add_assistant_message(response.content)

        if manager.check_context_overflow():
            manager.handle_compaction(current_message)
    """

    def __init__(
        self,
        settings: Optional["Settings"] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        config: Optional[ConversationManagerConfig] = None,
        controller: Optional[ConversationController] = None,
        store: Optional["ConversationStore"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize ConversationManager.

        Args:
            settings: Application settings
            provider: LLM provider name (e.g., "anthropic", "openai")
            model: Model identifier (e.g., "claude-3-sonnet")
            system_prompt: System prompt to initialize conversation with
            config: Manager configuration options
            controller: Optional pre-configured ConversationController
            store: Optional pre-configured ConversationStore
            embedding_service: Optional embedding service for semantic features
            session_id: Optional session ID to recover/continue
        """
        self._settings = settings
        self._provider = provider
        self._model = model
        self._config = config or ConversationManagerConfig()

        # Build conversation config from manager config
        conv_config = ConversationConfig(
            max_context_chars=self._config.max_context_chars,
            chars_per_token_estimate=self._config.chars_per_token_estimate,
            enable_stage_tracking=self._config.enable_stage_tracking,
        )

        # Initialize controller (required component)
        self._controller = controller or ConversationController(config=conv_config)

        # Set system prompt if provided
        if system_prompt:
            self._controller.set_system_prompt(system_prompt)

        # Optional persistence components (lazy initialization)
        self._store: Optional["ConversationStore"] = store
        self._embedding_store: Optional["ConversationEmbeddingStore"] = None
        self._embedding_service = embedding_service
        self._session_id = session_id
        self._session: Optional["ConversationSession"] = None

        # Initialize persistence if enabled and store provided
        if self._store and self._config.enable_persistence:
            self._initialize_session()

        logger.debug(
            f"ConversationManager initialized: provider={provider}, model={model}, "
            f"persistence={self._config.enable_persistence}"
        )

    def _initialize_session(self) -> None:
        """Initialize or recover conversation session."""
        if not self._store:
            return

        if self._session_id:
            # Try to recover existing session
            self._session = self._store.get_session(self._session_id)
            if self._session:
                logger.info(f"Recovered session: {self._session_id}")
                return

        # Create new session
        project_path = None
        if self._settings:
            project_path = str(getattr(self._settings, "project_path", None) or "")

        self._session = self._store.create_session(
            session_id=self._session_id,
            project_path=project_path,
            provider=self._provider,
            model=self._model,
        )
        self._session_id = self._session.session_id
        logger.info(f"Created new session: {self._session_id}")

    # =========================================================================
    # MESSAGE MANAGEMENT
    # =========================================================================

    def add_message(self, role: str, content: str) -> "Message":
        """Add a message with specified role.

        Adds the message to the in-memory controller and optionally persists
        to SQLite if persistence is enabled.

        Args:
            role: Message role (user, assistant, system)
            content: Message content

        Returns:
            The created Message object
        """
        message = self._controller.add_message(role, content)
        self._persist_message(role, content)
        return message

    def add_user_message(self, content: str) -> "Message":
        """Add a user message.

        Args:
            content: User message content

        Returns:
            The created Message object
        """
        message = self._controller.add_user_message(content)
        self._persist_message("user", content)
        return message

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> "Message":
        """Add an assistant message with optional tool calls.

        Args:
            content: Assistant message content
            tool_calls: Optional list of tool calls made by the assistant

        Returns:
            The created Message object
        """
        message = self._controller.add_assistant_message(content, tool_calls=tool_calls)
        self._persist_message("assistant", content)
        return message

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> "Message":
        """Add a tool result message.

        Args:
            tool_call_id: ID of the tool call being responded to
            tool_name: Name of the tool that was executed
            result: Tool execution result

        Returns:
            The created Message object
        """
        message = self._controller.add_tool_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=result,
        )
        self._persist_message("tool", result, tool_name=tool_name)
        return message

    def _persist_message(
        self,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
    ) -> None:
        """Persist message to SQLite store if enabled."""
        if not self._store or not self._session_id or not self._config.enable_persistence:
            return

        try:
            from victor.agent.conversation_memory import MessageRole

            # Map role strings to MessageRole enum
            role_map = {
                "user": MessageRole.USER,
                "assistant": MessageRole.ASSISTANT,
                "system": MessageRole.SYSTEM,
                "tool": MessageRole.TOOL_RESULT,
            }
            msg_role = role_map.get(role, MessageRole.USER)

            self._store.add_message(
                session_id=self._session_id,
                role=msg_role,
                content=content,
                tool_name=tool_name,
            )
        except Exception as e:
            logger.warning(f"Failed to persist message: {e}")

    # =========================================================================
    # MESSAGE ACCESS
    # =========================================================================

    @property
    def messages(self) -> List["Message"]:
        """Get all messages in the conversation.

        Returns:
            List of Message objects
        """
        return self._controller.messages

    def message_count(self) -> int:
        """Get the number of messages in the conversation.

        Returns:
            Number of messages
        """
        return self._controller.message_count

    def get_last_user_message(self) -> Optional[str]:
        """Get the content of the last user message.

        Returns:
            Last user message content or None
        """
        return self._controller.get_last_user_message()

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the content of the last assistant message.

        Returns:
            Last assistant message content or None
        """
        return self._controller.get_last_assistant_message()

    # =========================================================================
    # CONTEXT MANAGEMENT
    # =========================================================================

    def get_context_metrics(self) -> ContextMetrics:
        """Get current context metrics.

        Returns:
            ContextMetrics with size and overflow information
        """
        return self._controller.get_context_metrics()

    def check_context_overflow(self) -> bool:
        """Check if context is at risk of overflow.

        Returns:
            True if context is dangerously large
        """
        return self._controller.check_context_overflow()

    def handle_compaction(
        self,
        user_message: Optional[str] = None,
        target_messages: Optional[int] = None,
    ) -> int:
        """Trigger context compaction.

        Uses smart compaction to reduce context size while preserving
        the most important information.

        Args:
            user_message: Current user message for semantic relevance scoring
            target_messages: Target number of messages to keep

        Returns:
            Number of messages removed
        """
        return self._controller.smart_compact_history(
            target_messages=target_messages,
            current_query=user_message,
        )

    def get_memory_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get token-aware context for LLM calls.

        If persistence is enabled, returns formatted messages within token budget.
        Otherwise, returns messages from in-memory controller.

        Args:
            max_tokens: Maximum tokens for context

        Returns:
            List of messages in provider format
        """
        if self._store and self._session_id:
            return self._store.get_context_messages(
                session_id=self._session_id,
                max_tokens=max_tokens,
            )

        # Fall back to in-memory messages
        return [{"role": m.role, "content": m.content} for m in self._controller.messages]

    # =========================================================================
    # STAGE TRACKING
    # =========================================================================

    @property
    def stage(self) -> ConversationStage:
        """Get current conversation stage.

        Returns:
            Current ConversationStage
        """
        return self._controller.stage

    def get_stage_recommended_tools(self) -> Set[str]:
        """Get tools recommended for current conversation stage.

        Returns:
            Set of recommended tool names
        """
        return self._controller.get_stage_recommended_tools()

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID.

        Returns:
            Session ID or None if no session
        """
        return self._session_id

    def get_recent_sessions(
        self,
        limit: int = 10,
        project_path: Optional[str] = None,
    ) -> List["ConversationSession"]:
        """List recent conversation sessions.

        Args:
            limit: Maximum sessions to return
            project_path: Optional filter by project path

        Returns:
            List of ConversationSession objects
        """
        if not self._store:
            return []

        return self._store.list_sessions(project_path=project_path, limit=limit)

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous session.

        Loads session from persistent store and restores messages
        to the controller.

        Args:
            session_id: Session ID to recover

        Returns:
            True if recovery successful, False otherwise
        """
        if not self._store:
            logger.warning("Cannot recover session: no store configured")
            return False

        session = self._store.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        # Reset controller
        self._controller.reset()

        # Restore messages to controller
        # Note: ConversationSession.messages contains ConversationMessage objects
        for msg in session.messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            self._controller.add_message(role, msg.content)

        self._session = session
        self._session_id = session_id
        logger.info(f"Recovered session {session_id} with {len(session.messages)} messages")
        return True

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session.

        Returns:
            Dictionary of session statistics
        """
        if not self._store or not self._session_id:
            return {
                "session_id": self._session_id,
                "message_count": self.message_count(),
                "stage": self.stage.value,
            }

        return self._store.get_session_stats(self._session_id)

    # =========================================================================
    # EMBEDDING STORE
    # =========================================================================

    async def initialize_embedding_store(self) -> bool:
        """Lazy initialize LanceDB embedding store.

        Call this method when semantic search capabilities are needed.
        The embedding store is not initialized by default to reduce startup overhead.

        Returns:
            True if initialization successful, False otherwise
        """
        if not self._config.enable_embeddings:
            return False

        if self._embedding_store is not None:
            return True

        try:
            from victor.agent.conversation_embedding_store import ConversationEmbeddingStore

            # Get or create embedding service
            if self._embedding_service is None:
                from victor.storage.embeddings.service import EmbeddingService

                self._embedding_service = EmbeddingService.get_instance()

            # Get SQLite DB path from store if available
            sqlite_path = None
            if self._store:
                sqlite_path = self._store.db_path

            self._embedding_store = ConversationEmbeddingStore(
                embedding_service=self._embedding_service,
                sqlite_db_path=sqlite_path,
            )
            await self._embedding_store.initialize()

            # Wire up embedding store to controller and store
            if self._embedding_service:
                self._controller.set_embedding_service(self._embedding_service)
            if self._store and self._embedding_store:
                self._store.set_embedding_store(self._embedding_store)

            logger.info("Embedding store initialized")
            return True

        except ImportError as e:
            logger.warning(f"Failed to initialize embedding store: {e}")
            return False
        except Exception as e:
            logger.error(f"Embedding store initialization error: {e}")
            return False

    async def search_similar_messages(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search for semantically similar messages.

        Requires embedding store to be initialized.

        Args:
            query: Query text to search for
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of matching messages with similarity scores
        """
        if not self._embedding_store:
            logger.warning(
                "Embedding store not initialized. Call initialize_embedding_store() first."
            )
            return []

        try:
            results = await self._embedding_store.search_similar(
                query=query,
                session_id=self._session_id,
                limit=limit,
                min_similarity=min_similarity,
            )

            # Fetch full message content from store if available
            if self._store:
                enriched = []
                for result in results:
                    enriched.append(
                        {
                            "message_id": result.message_id,
                            "session_id": result.session_id,
                            "similarity": result.similarity,
                            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
                        }
                    )
                return enriched

            return [
                {
                    "message_id": r.message_id,
                    "session_id": r.session_id,
                    "similarity": r.similarity,
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def reset(self) -> None:
        """Reset conversation to initial state.

        Clears messages and resets stage tracking.
        Does not delete persisted session data.
        """
        self._controller.reset()
        logger.info("Conversation reset")

    def set_system_prompt(self, prompt: str) -> None:
        """Set or update the system prompt.

        Args:
            prompt: New system prompt
        """
        self._controller.set_system_prompt(prompt)

    def to_dict(self) -> Dict[str, Any]:
        """Export conversation state as dictionary.

        Returns:
            Dictionary representation of conversation
        """
        return {
            **self._controller.to_dict(),
            "session_id": self._session_id,
            "provider": self._provider,
            "model": self._model,
            "persistence_enabled": self._config.enable_persistence,
            "embeddings_enabled": self._config.enable_embeddings,
        }

    async def close(self) -> None:
        """Clean up resources.

        Should be called when the conversation manager is no longer needed.
        """
        if self._embedding_store:
            await self._embedding_store.close()
            self._embedding_store = None

        logger.info("ConversationManager closed")


def create_conversation_manager(
    settings: Optional["Settings"] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    enable_persistence: bool = True,
    enable_embeddings: bool = True,
) -> ConversationManager:
    """Factory function to create a ConversationManager.

    This is a convenience function that handles common initialization patterns.

    Args:
        settings: Application settings
        provider: LLM provider name
        model: Model identifier
        system_prompt: System prompt for the conversation
        enable_persistence: Whether to enable SQLite persistence
        enable_embeddings: Whether to enable embedding store

    Returns:
        Configured ConversationManager instance
    """
    config = ConversationManagerConfig(
        enable_persistence=enable_persistence,
        enable_embeddings=enable_embeddings,
    )

    # Create store if persistence enabled
    store = None
    if enable_persistence:
        try:
            from victor.agent.conversation_memory import ConversationStore

            store = ConversationStore()
        except Exception as e:
            logger.warning(f"Failed to create ConversationStore: {e}")

    return ConversationManager(
        settings=settings,
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        config=config,
        store=store,
    )
