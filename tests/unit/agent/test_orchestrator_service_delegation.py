"""Tests for mandatory service layer in the orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.orchestrator import AgentOrchestrator


class TestServiceInitialization:
    """Service layer initialization is mandatory."""

    def _make_init_stub(self):
        obj = object.__new__(AgentOrchestrator)
        obj._container = MagicMock()
        obj._container.get_optional.return_value = None
        obj._container.is_registered.return_value = False
        obj._conversation_controller = MagicMock()
        obj._streaming_controller = MagicMock()
        return obj

    def test_services_resolved_from_container(self):
        obj = self._make_init_stub()
        obj._initialize_services()
        assert obj._container.get_optional.call_count >= 4

    def test_raises_without_container(self):
        obj = object.__new__(AgentOrchestrator)
        with pytest.raises(RuntimeError, match="ServiceContainer required"):
            obj._initialize_services()

    def test_all_service_attributes_set(self):
        obj = self._make_init_stub()
        mock_svc = object()
        obj._container.get_optional.return_value = mock_svc
        obj._initialize_services()
        assert obj._chat_service is mock_svc
        assert obj._tool_service is mock_svc
        assert obj._session_service is mock_svc
        assert obj._context_service is mock_svc


class TestServiceDelegation:
    """Methods delegate directly to services (no coordinator fallback)."""

    def _make_stub(self):
        obj = object.__new__(AgentOrchestrator)
        obj._chat_service = MagicMock()
        obj._tool_service = MagicMock()
        obj._session_service = MagicMock()
        obj._context_service = MagicMock()
        obj._provider_service = MagicMock()
        obj._recovery_service = MagicMock()
        return obj

    @pytest.mark.asyncio
    async def test_chat_delegates_to_service(self):
        obj = self._make_stub()
        obj._chat_service.chat = AsyncMock(return_value="response")
        with patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ff:
            mock_ff.return_value.is_enabled.return_value = False
            result = await obj.chat("hello")
        obj._chat_service.chat.assert_called_once_with("hello", use_planning=False)

    @pytest.mark.asyncio
    async def test_chat_with_planning_delegates(self):
        obj = self._make_stub()
        obj._chat_service.chat_with_planning = AsyncMock(return_value="response")
        result = await obj.chat_with_planning("plan this")
        obj._chat_service.chat_with_planning.assert_called_once()

    def test_get_available_tools_delegates(self):
        obj = self._make_stub()
        obj._tool_service.get_available_tools.return_value = {"read", "write"}
        result = obj.get_available_tools()
        assert result == {"read", "write"}

    def test_get_enabled_tools_delegates(self):
        obj = self._make_stub()
        obj._tool_service.get_enabled_tools.return_value = {"read"}
        result = obj.get_enabled_tools()
        assert result == {"read"}

    def test_is_tool_enabled_delegates(self):
        obj = self._make_stub()
        obj._tool_service.is_tool_enabled.return_value = True
        assert obj.is_tool_enabled("read") is True

    def test_set_enabled_tools_delegates(self):
        obj = self._make_stub()
        obj._enabled_tools = set()
        obj._apply_vertical_tools = MagicMock()
        obj.tool_selector = None
        obj.set_enabled_tools({"read", "write"})
        obj._tool_service.set_enabled_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_checkpoint_delegates(self):
        obj = self._make_stub()
        obj._session_service.save_checkpoint = AsyncMock(return_value="cp1")
        result = await obj.save_checkpoint("test")
        obj._session_service.save_checkpoint.assert_called_once()

    def test_get_context_metrics_delegates(self):
        obj = self._make_stub()
        obj._context_service.get_context_metrics.return_value = {"tokens": 100}
        result = obj.get_context_metrics()
        assert result == {"tokens": 100}


class TestNoCoordinatorFallback:
    """Structural test: no coordinator fallback patterns remain."""

    def test_no_service_none_guards(self):
        import inspect

        source = inspect.getsource(AgentOrchestrator)
        # Count patterns like "if self._chat_service:" followed by coordinator
        # After migration, there should be ZERO such guards
        import re

        guards = re.findall(r"if self\._\w+_service:", source)
        assert len(guards) == 0, (
            f"Found {len(guards)} service None-guards. "
            f"Services are mandatory — remove coordinator fallbacks."
        )

    def test_no_use_service_layer_flag(self):
        import inspect

        source = inspect.getsource(AgentOrchestrator)
        assert "_use_service_layer" not in source
