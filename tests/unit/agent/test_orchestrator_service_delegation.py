"""Tests for orchestrator service layer delegation (Strangler Fig pattern).

Verifies that:
1. Service attributes are None when USE_SERVICE_LAYER flag is off (default)
2. _initialize_services runs without error
3. When flag is on but no services bootstrapped, coordinator fallback works
4. Dual delegation pattern correctly checks flag + service availability
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.core.feature_flags import (
    FeatureFlag,
    FeatureFlagConfig,
    FeatureFlagManager,
    reset_feature_flag_manager,
)


@pytest.fixture(autouse=True)
def _reset_flags():
    """Reset feature flags before and after each test."""
    reset_feature_flag_manager()
    yield
    reset_feature_flag_manager()


class TestServiceInitialization:
    """Service layer initialization based on feature flag."""

    def test_services_none_when_flag_off(self):
        """When USE_SERVICE_LAYER is off, all service attributes are None."""
        from victor.agent.orchestrator import AgentOrchestrator

        # Explicitly disable the flag (FeatureFlagConfig defaults all flags on)
        config = FeatureFlagConfig(default_enabled=False)
        manager = FeatureFlagManager(config)

        obj = object.__new__(AgentOrchestrator)
        obj._container = MagicMock()

        with patch(
            "victor.core.feature_flags.get_feature_flag_manager",
            return_value=manager,
        ):
            obj._initialize_services()

        assert obj._use_service_layer is False
        assert obj._chat_service is None
        assert obj._tool_service is None
        assert obj._session_service is None
        assert obj._context_service is None

    def test_services_resolved_when_flag_on(self):
        """When USE_SERVICE_LAYER is on, services are resolved from container."""
        from victor.agent.orchestrator import AgentOrchestrator

        # Enable the flag
        config = FeatureFlagConfig(default_enabled=False)
        manager = FeatureFlagManager(config)
        manager.enable(FeatureFlag.USE_SERVICE_LAYER)

        obj = object.__new__(AgentOrchestrator)
        mock_container = MagicMock()
        mock_container.get_optional.return_value = None  # No services registered
        obj._container = mock_container

        with patch(
            "victor.core.feature_flags.get_feature_flag_manager",
            return_value=manager,
        ):
            obj._initialize_services()

        assert obj._use_service_layer is True
        # Services are None because container has nothing registered
        assert obj._chat_service is None
        assert obj._tool_service is None

    def test_services_none_without_container(self):
        """When container is not available, services are None even if flag is on."""
        from victor.agent.orchestrator import AgentOrchestrator

        config = FeatureFlagConfig(default_enabled=False)
        manager = FeatureFlagManager(config)
        manager.enable(FeatureFlag.USE_SERVICE_LAYER)

        obj = object.__new__(AgentOrchestrator)
        # No _container attribute

        with patch(
            "victor.core.feature_flags.get_feature_flag_manager",
            return_value=manager,
        ):
            obj._initialize_services()

        # Flag reflects feature flag state, but services degrade gracefully
        assert obj._use_service_layer is True
        assert obj._chat_service is None
        assert obj._tool_service is None
        assert obj._session_service is None
        assert obj._context_service is None


class TestDualDelegation:
    """Service layer delegates correctly when enabled, falls back to coordinator."""

    def _make_orchestrator_stub(self, use_service_layer=False):
        """Create a minimal stub with coordinator and optional service mocks."""
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj._use_service_layer = use_service_layer

        # Mock coordinators
        obj._chat_coordinator = MagicMock()
        obj._tool_coordinator = MagicMock()
        obj._session_coordinator = MagicMock()

        # Mock services (None by default)
        obj._chat_service = None
        obj._tool_service = None
        obj._session_service = None
        obj._context_service = None

        return obj

    # --- Chat delegation ---

    async def test_chat_uses_coordinator_when_flag_off(self):
        obj = self._make_orchestrator_stub(use_service_layer=False)
        obj._chat_coordinator.chat = AsyncMock(return_value="coordinator_response")

        result = await obj.chat("hello")
        assert result == "coordinator_response"
        obj._chat_coordinator.chat.assert_called_once_with("hello", use_planning=False)

    async def test_chat_uses_service_when_flag_on(self):
        obj = self._make_orchestrator_stub(use_service_layer=True)
        obj._chat_service = MagicMock()
        obj._chat_service.chat = AsyncMock(return_value="service_response")

        result = await obj.chat("hello")
        assert result == "service_response"
        obj._chat_service.chat.assert_called_once_with("hello")

    async def test_chat_falls_back_to_coordinator_when_service_unavailable(self):
        obj = self._make_orchestrator_stub(use_service_layer=True)
        obj._chat_service = None  # Service not bootstrapped
        obj._chat_coordinator.chat = AsyncMock(return_value="coordinator_fallback")

        result = await obj.chat("hello")
        assert result == "coordinator_fallback"

    # --- chat_with_planning delegation ---

    async def test_chat_with_planning_uses_service(self):
        obj = self._make_orchestrator_stub(use_service_layer=True)
        obj._chat_service = MagicMock()
        obj._chat_service.chat_with_planning = AsyncMock(return_value="planned")

        result = await obj.chat_with_planning("complex task", use_planning=True)
        assert result == "planned"

    async def test_chat_with_planning_falls_back(self):
        obj = self._make_orchestrator_stub(use_service_layer=False)
        obj._chat_coordinator.chat_with_planning = AsyncMock(
            return_value="coord_planned"
        )

        result = await obj.chat_with_planning("complex task")
        assert result == "coord_planned"

    # --- Tool delegation ---

    def test_get_available_tools_uses_coordinator_by_default(self):
        obj = self._make_orchestrator_stub(use_service_layer=False)
        obj._tool_coordinator.get_available_tools.return_value = {"tool_a", "tool_b"}

        result = obj.get_available_tools()
        assert result == {"tool_a", "tool_b"}

    def test_get_available_tools_uses_service(self):
        obj = self._make_orchestrator_stub(use_service_layer=True)
        obj._tool_service = MagicMock()
        obj._tool_service.get_available_tools.return_value = {"svc_tool"}

        result = obj.get_available_tools()
        assert result == {"svc_tool"}

    def test_get_enabled_tools_uses_service(self):
        obj = self._make_orchestrator_stub(use_service_layer=True)
        obj._tool_service = MagicMock()
        obj._tool_service.get_enabled_tools.return_value = {"enabled_tool"}

        result = obj.get_enabled_tools()
        assert result == {"enabled_tool"}

    def test_is_tool_enabled_uses_service(self):
        obj = self._make_orchestrator_stub(use_service_layer=True)
        obj._tool_service = MagicMock()
        obj._tool_service.is_tool_enabled.return_value = True

        assert obj.is_tool_enabled("my_tool") is True
        obj._tool_service.is_tool_enabled.assert_called_once_with("my_tool")

    def test_is_tool_enabled_falls_back(self):
        obj = self._make_orchestrator_stub(use_service_layer=False)
        obj._tool_coordinator.is_tool_enabled.return_value = False

        assert obj.is_tool_enabled("my_tool") is False

    def test_set_enabled_tools_uses_service(self):
        obj = self._make_orchestrator_stub(use_service_layer=True)
        obj._tool_service = MagicMock()
        obj._enabled_tools = set()
        obj._vertical_context = MagicMock()
        obj.tool_selector = None

        obj.set_enabled_tools({"a", "b"})
        obj._tool_service.set_enabled_tools.assert_called_once_with({"a", "b"})

    # --- Session delegation ---

    async def test_save_checkpoint_uses_service(self):
        obj = self._make_orchestrator_stub(use_service_layer=True)
        obj._session_service = MagicMock()
        obj._session_service.save_checkpoint = AsyncMock(return_value="cp-123")

        result = await obj.save_checkpoint("test checkpoint")
        assert result == "cp-123"

    async def test_save_checkpoint_falls_back(self):
        obj = self._make_orchestrator_stub(use_service_layer=False)
        obj._session_coordinator.save_checkpoint = AsyncMock(return_value="cp-456")

        result = await obj.save_checkpoint("test")
        assert result == "cp-456"

    async def test_restore_checkpoint_uses_service(self):
        obj = self._make_orchestrator_stub(use_service_layer=True)
        obj._session_service = MagicMock()
        obj._session_service.restore_checkpoint = AsyncMock(return_value=True)

        result = await obj.restore_checkpoint("cp-123")
        assert result is True

    def test_get_recent_sessions_uses_service(self):
        obj = self._make_orchestrator_stub(use_service_layer=True)
        obj._session_service = MagicMock()
        obj._session_service.get_recent_sessions.return_value = [{"id": "s1"}]

        result = obj.get_recent_sessions(5)
        assert result == [{"id": "s1"}]
        obj._session_service.get_recent_sessions.assert_called_once_with(5)

    def test_get_session_stats_uses_service(self):
        obj = self._make_orchestrator_stub(use_service_layer=True)
        obj._session_service = MagicMock()
        obj._session_service.get_session_stats.return_value = {"messages": 10}

        result = obj.get_session_stats()
        assert result == {"messages": 10}

    def test_get_session_stats_falls_back(self):
        obj = self._make_orchestrator_stub(use_service_layer=False)
        obj._session_coordinator.get_session_stats.return_value = {"messages": 5}

        result = obj.get_session_stats()
        assert result == {"messages": 5}
