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

"""Tests for ChatFacade domain facade."""

import pytest
from unittest.mock import MagicMock

from victor.agent.facades.chat_facade import ChatFacade
from victor.agent.facades.protocols import ChatFacadeProtocol


class TestChatFacadeInit:
    """Tests for ChatFacade initialization."""

    def test_init_with_all_components(self):
        """ChatFacade initializes with all components provided."""
        conversation = MagicMock()
        controller = MagicMock()
        state = MagicMock()
        memory = MagicMock()
        embedding = MagicMock()
        intent_cls = MagicMock()

        facade = ChatFacade(
            conversation=conversation,
            conversation_controller=controller,
            conversation_state=state,
            memory_manager=memory,
            memory_session_id="session-123",
            embedding_store=embedding,
            intent_classifier=intent_cls,
            intent_detector=MagicMock(),
            reminder_manager=MagicMock(),
            system_prompt="You are a helpful assistant.",
            response_completer=MagicMock(),
            context_compactor=MagicMock(),
            task_completion_detector=MagicMock(),
        )

        assert facade.conversation is conversation
        assert facade.conversation_controller is controller
        assert facade.conversation_state is state
        assert facade.memory_manager is memory
        assert facade.memory_session_id == "session-123"
        assert facade.embedding_store is embedding
        assert facade.intent_classifier is intent_cls

    def test_init_with_minimal_components(self):
        """ChatFacade initializes with only required components."""
        conversation = MagicMock()
        controller = MagicMock()
        state = MagicMock()

        facade = ChatFacade(
            conversation=conversation,
            conversation_controller=controller,
            conversation_state=state,
        )

        assert facade.conversation is conversation
        assert facade.conversation_controller is controller
        assert facade.conversation_state is state
        assert facade.memory_manager is None
        assert facade.memory_session_id is None
        assert facade.embedding_store is None
        assert facade.intent_classifier is None
        assert facade.intent_detector is None
        assert facade.reminder_manager is None
        assert facade.system_prompt == ""
        assert facade.response_completer is None
        assert facade.context_compactor is None
        assert facade.task_completion_detector is None


class TestChatFacadeProperties:
    """Tests for ChatFacade property access."""

    @pytest.fixture
    def facade(self):
        """Create a ChatFacade with mock components."""
        return ChatFacade(
            conversation=MagicMock(name="conversation"),
            conversation_controller=MagicMock(name="controller"),
            conversation_state=MagicMock(name="state"),
            memory_manager=MagicMock(name="memory"),
            memory_session_id="test-session",
            embedding_store=MagicMock(name="embedding"),
            intent_classifier=MagicMock(name="intent"),
            intent_detector=MagicMock(name="detector"),
            reminder_manager=MagicMock(name="reminder"),
            system_prompt="Test prompt",
            response_completer=MagicMock(name="completer"),
            context_compactor=MagicMock(name="compactor"),
            task_completion_detector=MagicMock(name="detector"),
        )

    def test_conversation_property(self, facade):
        """Conversation property returns the MessageHistory."""
        assert facade.conversation._mock_name == "conversation"

    def test_conversation_controller_property(self, facade):
        """ConversationController property returns the controller."""
        assert facade.conversation_controller._mock_name == "controller"

    def test_conversation_state_property(self, facade):
        """ConversationState property returns the state machine."""
        assert facade.conversation_state._mock_name == "state"

    def test_memory_manager_property(self, facade):
        """MemoryManager property returns the memory store."""
        assert facade.memory_manager._mock_name == "memory"

    def test_embedding_store_property(self, facade):
        """EmbeddingStore property returns the embedding store."""
        assert facade.embedding_store._mock_name == "embedding"

    def test_intent_classifier_property(self, facade):
        """IntentClassifier property returns the classifier."""
        assert facade.intent_classifier._mock_name == "intent"

    def test_system_prompt_property(self, facade):
        """SystemPrompt property returns the prompt text."""
        assert facade.system_prompt == "Test prompt"

    def test_system_prompt_setter(self, facade):
        """SystemPrompt setter updates the prompt."""
        facade.system_prompt = "New prompt"
        assert facade.system_prompt == "New prompt"

    def test_embedding_store_setter(self, facade):
        """EmbeddingStore setter updates the store."""
        new_store = MagicMock(name="new_store")
        facade.embedding_store = new_store
        assert facade.embedding_store is new_store

    def test_context_compactor_setter(self, facade):
        """ContextCompactor setter updates the compactor."""
        new_compactor = MagicMock(name="new_compactor")
        facade.context_compactor = new_compactor
        assert facade.context_compactor is new_compactor


class TestChatFacadeProtocolConformance:
    """Tests that ChatFacade satisfies ChatFacadeProtocol."""

    def test_satisfies_protocol(self):
        """ChatFacade structurally conforms to ChatFacadeProtocol."""
        facade = ChatFacade(
            conversation=MagicMock(),
            conversation_controller=MagicMock(),
            conversation_state=MagicMock(),
        )
        assert isinstance(facade, ChatFacadeProtocol)

    def test_protocol_properties_present(self):
        """All protocol-required properties are present on ChatFacade."""
        required = [
            "conversation",
            "conversation_controller",
            "conversation_state",
            "memory_manager",
            "embedding_store",
            "intent_classifier",
        ]
        facade = ChatFacade(
            conversation=MagicMock(),
            conversation_controller=MagicMock(),
            conversation_state=MagicMock(),
        )
        for prop in required:
            assert hasattr(facade, prop), f"Missing protocol property: {prop}"
