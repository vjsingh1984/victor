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

"""Unit tests for ChatCoordinator protocol decoupling (Phase 6 DIP Hardening).

Tests:
- Protocol importability and structure
- runtime_checkable isinstance checks
- ChatCoordinator instantiation with mock orchestrator
- Protocol completeness (key attributes declared)
"""

import pytest
from typing import Protocol
from unittest.mock import MagicMock

# =============================================================================
# Protocol importability and structure
# =============================================================================


class TestProtocolImports:
    """Verify all protocols are importable and are Protocol subclasses."""

    def test_chat_context_protocol_importable(self):
        from victor.agent.coordinators.chat_protocols import ChatContextProtocol

        assert issubclass(ChatContextProtocol, Protocol)

    def test_tool_context_protocol_importable(self):
        from victor.agent.coordinators.chat_protocols import ToolContextProtocol

        assert issubclass(ToolContextProtocol, Protocol)

    def test_provider_context_protocol_importable(self):
        from victor.agent.coordinators.chat_protocols import ProviderContextProtocol

        assert issubclass(ProviderContextProtocol, Protocol)

    def test_chat_orchestrator_protocol_importable(self):
        from victor.agent.coordinators.chat_protocols import ChatOrchestratorProtocol

        assert issubclass(ChatOrchestratorProtocol, Protocol)

    def test_chat_orchestrator_protocol_is_runtime_checkable(self):
        from victor.agent.coordinators.chat_protocols import ChatOrchestratorProtocol

        # runtime_checkable sets _is_runtime_protocol on the class
        assert getattr(ChatOrchestratorProtocol, "_is_runtime_protocol", False)

    def test_importable_from_coordinators_package(self):
        from victor.agent.coordinators import (
            ChatContextProtocol,
            ChatOrchestratorProtocol,
            ProviderContextProtocol,
            ToolContextProtocol,
        )

        assert ChatContextProtocol is not None
        assert ChatOrchestratorProtocol is not None
        assert ProviderContextProtocol is not None
        assert ToolContextProtocol is not None


# =============================================================================
# Protocol completeness checks
# =============================================================================


class TestProtocolCompleteness:
    """Verify protocols declare the expected key attributes."""

    def test_chat_context_declares_conversation(self):
        from victor.agent.coordinators.chat_protocols import ChatContextProtocol

        annotations = {}
        for cls in ChatContextProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "conversation" in annotations
        assert "messages" in annotations
        assert "settings" in annotations
        assert "_system_added" in annotations
        assert "_cumulative_token_usage" in annotations

    def test_chat_context_declares_add_message(self):
        from victor.agent.coordinators.chat_protocols import ChatContextProtocol

        assert hasattr(ChatContextProtocol, "add_message")

    def test_tool_context_declares_tool_members(self):
        from victor.agent.coordinators.chat_protocols import ToolContextProtocol

        annotations = {}
        for cls in ToolContextProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "tool_selector" in annotations
        assert "tool_budget" in annotations
        assert "tool_calls_used" in annotations
        assert "use_semantic_selection" in annotations

    def test_tool_context_declares_handle_tool_calls(self):
        from victor.agent.coordinators.chat_protocols import ToolContextProtocol

        assert hasattr(ToolContextProtocol, "_handle_tool_calls")

    def test_provider_context_declares_provider_members(self):
        from victor.agent.coordinators.chat_protocols import ProviderContextProtocol

        annotations = {}
        for cls in ProviderContextProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "provider" in annotations
        assert "model" in annotations
        assert "temperature" in annotations
        assert "max_tokens" in annotations

    def test_provider_context_declares_check_cancellation(self):
        from victor.agent.coordinators.chat_protocols import ProviderContextProtocol

        assert hasattr(ProviderContextProtocol, "_check_cancellation")

    def test_composite_protocol_inherits_all_sub_protocols(self):
        from victor.agent.coordinators.chat_protocols import (
            ChatContextProtocol,
            ChatOrchestratorProtocol,
            ProviderContextProtocol,
            ToolContextProtocol,
        )

        # Check MRO for sub-protocol inheritance (sub-protocols are not
        # runtime_checkable, so issubclass() won't work)
        mro = ChatOrchestratorProtocol.__mro__
        assert ChatContextProtocol in mro
        assert ToolContextProtocol in mro
        assert ProviderContextProtocol in mro

    def test_composite_protocol_declares_task_members(self):
        from victor.agent.coordinators.chat_protocols import ChatOrchestratorProtocol

        annotations = {}
        for cls in ChatOrchestratorProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "task_classifier" in annotations
        assert "task_coordinator" in annotations
        assert "unified_tracker" in annotations

    def test_composite_protocol_declares_recovery_members(self):
        from victor.agent.coordinators.chat_protocols import ChatOrchestratorProtocol

        annotations = {}
        for cls in ChatOrchestratorProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "_recovery_coordinator" in annotations
        assert "_recovery_integration" in annotations
        assert hasattr(ChatOrchestratorProtocol, "_record_intelligent_outcome")

    def test_composite_protocol_declares_presentation_members(self):
        from victor.agent.coordinators.chat_protocols import ChatOrchestratorProtocol

        annotations = {}
        for cls in ChatOrchestratorProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "sanitizer" in annotations
        assert "response_completer" in annotations
        assert "debug_logger" in annotations
        assert "_chunk_generator" in annotations


# =============================================================================
# ChatCoordinator instantiation with mock
# =============================================================================


class TestChatCoordinatorWithMock:
    """Verify ChatCoordinator can be instantiated with a mock orchestrator."""

    def test_instantiation_with_magicmock(self):
        from victor.agent.coordinators.chat_coordinator import ChatCoordinator

        mock_orch = MagicMock()
        coordinator = ChatCoordinator(mock_orch)
        assert coordinator._orchestrator is mock_orch

    def test_lazy_handlers_are_none_initially(self):
        from victor.agent.coordinators.chat_coordinator import ChatCoordinator

        mock_orch = MagicMock()
        coordinator = ChatCoordinator(mock_orch)
        assert coordinator._intent_classification_handler is None
        assert coordinator._continuation_handler is None
        assert coordinator._tool_execution_handler is None

    def test_mock_missing_attribute_fails_isinstance(self):
        """A mock missing a protocol attribute fails isinstance."""
        from victor.agent.coordinators.chat_protocols import ChatOrchestratorProtocol

        class Incomplete:
            pass

        obj = Incomplete()
        assert not isinstance(obj, ChatOrchestratorProtocol)
