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

"""Integration tests: orchestrator creates facades and delegates through them."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.facades.chat_facade import ChatFacade
from victor.agent.facades.tool_facade import ToolFacade
from victor.config.settings import Settings


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock()
    provider.name = "mock_provider"
    provider.supports_tools.return_value = True
    provider.get_context_window.return_value = 100000
    provider.chat = AsyncMock(return_value=MagicMock(content="Response", tool_calls=[]))
    return provider


@pytest.fixture
def orchestrator_settings():
    """Create settings for testing."""
    return Settings(
        analytics_enabled=False,
        use_semantic_tool_selection=False,
        use_mcp_tools=False,
    )


@pytest.fixture
def orchestrator(mock_provider, orchestrator_settings):
    """Create an orchestrator for testing."""
    with patch("victor.agent.orchestrator.UsageLogger"), \
         patch("victor.core.bootstrap_services.bootstrap_new_services"):
        return AgentOrchestrator(
            settings=orchestrator_settings,
            provider=mock_provider,
            model="test-model",
        )


class TestFacadeCreation:
    """Tests that orchestrator creates facade instances."""

    def test_chat_facade_created(self, orchestrator):
        """Orchestrator creates a ChatFacade instance."""
        assert hasattr(orchestrator, "_chat_facade")
        assert isinstance(orchestrator._chat_facade, ChatFacade)

    def test_tool_facade_created(self, orchestrator):
        """Orchestrator creates a ToolFacade instance."""
        assert hasattr(orchestrator, "_tool_facade")
        assert isinstance(orchestrator._tool_facade, ToolFacade)


class TestChatFacadeDelegation:
    """Tests that orchestrator properties delegate through ChatFacade."""

    def test_conversation_controller_delegates(self, orchestrator):
        """conversation_controller property returns same object via facade."""
        direct = orchestrator._conversation_controller
        via_facade = orchestrator._chat_facade.conversation_controller
        via_property = orchestrator.conversation_controller
        assert direct is via_facade
        assert via_property is via_facade

    def test_context_compactor_delegates(self, orchestrator):
        """context_compactor property returns same object via facade."""
        direct = orchestrator._context_compactor
        via_facade = orchestrator._chat_facade.context_compactor
        via_property = orchestrator.context_compactor
        assert direct is via_facade
        assert via_property is via_facade

    def test_chat_facade_has_conversation(self, orchestrator):
        """ChatFacade exposes the conversation (MessageHistory)."""
        assert orchestrator._chat_facade.conversation is orchestrator.conversation

    def test_chat_facade_has_conversation_state(self, orchestrator):
        """ChatFacade exposes the conversation state machine."""
        assert orchestrator._chat_facade.conversation_state is orchestrator.conversation_state

    def test_chat_facade_has_memory_manager(self, orchestrator):
        """ChatFacade exposes the memory manager."""
        assert orchestrator._chat_facade.memory_manager is orchestrator.memory_manager

    def test_chat_facade_has_intent_classifier(self, orchestrator):
        """ChatFacade exposes the intent classifier."""
        assert orchestrator._chat_facade.intent_classifier is orchestrator.intent_classifier

    def test_chat_facade_has_reminder_manager(self, orchestrator):
        """ChatFacade exposes the reminder manager."""
        assert orchestrator._chat_facade.reminder_manager is orchestrator.reminder_manager


class TestToolFacadeDelegation:
    """Tests that orchestrator properties delegate through ToolFacade."""

    def test_tool_pipeline_delegates(self, orchestrator):
        """tool_pipeline property returns same object via facade."""
        direct = orchestrator._tool_pipeline
        via_facade = orchestrator._tool_facade.tool_pipeline
        via_property = orchestrator.tool_pipeline
        assert direct is via_facade
        assert via_property is via_facade

    def test_tool_output_formatter_delegates(self, orchestrator):
        """tool_output_formatter property returns same object via facade."""
        direct = orchestrator._tool_output_formatter
        via_facade = orchestrator._tool_facade.tool_output_formatter
        via_property = orchestrator.tool_output_formatter
        assert direct is via_facade
        assert via_property is via_facade

    def test_sequence_tracker_delegates(self, orchestrator):
        """sequence_tracker property returns same object via facade."""
        direct = orchestrator._sequence_tracker
        via_facade = orchestrator._tool_facade.sequence_tracker
        via_property = orchestrator.sequence_tracker
        assert direct is via_facade
        assert via_property is via_facade

    def test_code_correction_middleware_delegates(self, orchestrator):
        """code_correction_middleware property returns same object via facade."""
        direct = orchestrator._code_correction_middleware
        via_facade = orchestrator._tool_facade.code_correction_middleware
        via_property = orchestrator.code_correction_middleware
        assert direct is via_facade
        assert via_property is via_facade

    def test_tool_facade_has_tools(self, orchestrator):
        """ToolFacade exposes the tool registry."""
        assert orchestrator._tool_facade.tools is orchestrator.tools

    def test_tool_facade_has_tool_executor(self, orchestrator):
        """ToolFacade exposes the tool executor."""
        assert orchestrator._tool_facade.tool_executor is orchestrator.tool_executor

    def test_tool_facade_has_tool_selector(self, orchestrator):
        """ToolFacade exposes the tool selector."""
        assert orchestrator._tool_facade.tool_selector is orchestrator.tool_selector

    def test_tool_facade_has_tool_cache(self, orchestrator):
        """ToolFacade exposes the tool cache."""
        assert orchestrator._tool_facade.tool_cache is orchestrator.tool_cache

    def test_tool_facade_has_tool_budget(self, orchestrator):
        """ToolFacade exposes the tool budget."""
        assert orchestrator._tool_facade.tool_budget == orchestrator.tool_budget

    def test_tool_facade_has_safety_checker(self, orchestrator):
        """ToolFacade exposes the safety checker."""
        assert orchestrator._tool_facade.safety_checker is orchestrator._safety_checker

    def test_tool_facade_has_search_router(self, orchestrator):
        """ToolFacade exposes the search router."""
        assert orchestrator._tool_facade.search_router is orchestrator.search_router

    def test_tool_facade_has_semantic_selector(self, orchestrator):
        """ToolFacade exposes the semantic selector."""
        assert orchestrator._tool_facade.semantic_selector is orchestrator.semantic_selector


class TestBackwardCompatibility:
    """Tests that direct attribute access still works after facade introduction."""

    def test_direct_attribute_access_conversation(self, orchestrator):
        """Direct self.conversation still works."""
        assert orchestrator.conversation is not None

    def test_direct_attribute_access_tools(self, orchestrator):
        """Direct self.tools still works."""
        assert orchestrator.tools is not None

    def test_direct_attribute_access_tool_executor(self, orchestrator):
        """Direct self.tool_executor still works."""
        assert orchestrator.tool_executor is not None

    def test_direct_attribute_access_tool_budget(self, orchestrator):
        """Direct self.tool_budget still works."""
        assert orchestrator.tool_budget > 0

    def test_direct_attribute_access_conversation_state(self, orchestrator):
        """Direct self.conversation_state still works."""
        assert orchestrator.conversation_state is not None

    def test_direct_attribute_access_tool_registrar(self, orchestrator):
        """Direct self.tool_registrar still works."""
        assert orchestrator.tool_registrar is not None

    def test_direct_attribute_access_argument_normalizer(self, orchestrator):
        """Direct self.argument_normalizer still works."""
        assert orchestrator.argument_normalizer is not None

    def test_direct_attribute_access_intent_detector(self, orchestrator):
        """Direct self.intent_detector still works."""
        assert orchestrator.intent_detector is not None
