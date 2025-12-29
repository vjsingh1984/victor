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

"""Tests for FrameworkShim backward compatibility layer.

Tests the FrameworkShim class which bridges the legacy CLI path
(AgentOrchestrator.from_settings) to the new Framework API.

Test Categories:
1. Basic orchestrator creation without vertical
2. Observability wiring (enabled/disabled)
3. Vertical configuration application
4. Session ID generation and propagation
5. Lifecycle event emission
6. Helper functions (get_vertical, list_verticals)
"""

import uuid
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.shim import FrameworkShim, get_vertical, list_verticals
from victor.observability.integration import ObservabilityIntegration
from victor.verticals.base import VerticalBase, VerticalRegistry


class MockVertical(VerticalBase):
    """Mock vertical for testing."""

    name = "test_vertical"
    description = "Test vertical for unit tests"
    version = "1.0.0"

    @classmethod
    def get_tools(cls):
        return ["read", "write", "edit"]

    @classmethod
    def get_system_prompt(cls):
        return "You are a test assistant."

    @classmethod
    def get_stages(cls):
        return {
            "INITIAL": {"allowed_tools": ["read"], "next": ["PLANNING"]},
            "PLANNING": {"allowed_tools": ["read", "write"], "next": ["EXECUTION"]},
            "EXECUTION": {"allowed_tools": ["read", "write", "edit"], "next": ["INITIAL"]},
        }


class TestFrameworkShimBasic:
    """Basic FrameworkShim tests."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.provider = "anthropic"
        settings.model = "claude-3-5-sonnet"
        settings.airgapped_mode = False
        return settings

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orch = MagicMock()
        orch.prompt_builder = MagicMock()
        orch.prompt_builder.set_custom_prompt = MagicMock()
        return orch

    @pytest.mark.asyncio
    async def test_create_orchestrator_without_vertical(self, mock_settings, mock_orchestrator):
        """Test basic orchestrator creation without vertical."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, "default")
            result = await shim.create_orchestrator()

            assert result == mock_orchestrator
            mock_from_settings.assert_called_once_with(
                mock_settings,
                profile_name="default",
                thinking=False,
            )

    @pytest.mark.asyncio
    async def test_create_orchestrator_with_thinking(self, mock_settings, mock_orchestrator):
        """Test orchestrator creation with thinking enabled."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, "default", thinking=True)
            await shim.create_orchestrator()

            mock_from_settings.assert_called_once_with(
                mock_settings,
                profile_name="default",
                thinking=True,
            )

    @pytest.mark.asyncio
    async def test_session_id_auto_generated(self, mock_settings, mock_orchestrator):
        """Test that session ID is auto-generated if not provided."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings)
            await shim.create_orchestrator()

            # Session ID should be a valid UUID
            assert shim.session_id is not None
            try:
                uuid.UUID(shim.session_id)
            except ValueError:
                pytest.fail("session_id should be a valid UUID")

    @pytest.mark.asyncio
    async def test_session_id_provided(self, mock_settings, mock_orchestrator):
        """Test that provided session ID is used."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, session_id="custom-session-123")
            await shim.create_orchestrator()

            assert shim.session_id == "custom-session-123"


class TestFrameworkShimObservability:
    """Tests for observability wiring."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.provider = "anthropic"
        settings.model = "claude-3-5-sonnet"
        return settings

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orch = MagicMock()
        orch.conversation_state = MagicMock()
        return orch

    @pytest.mark.asyncio
    async def test_observability_wired_by_default(self, mock_settings, mock_orchestrator):
        """Test that observability is wired by default."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, enable_observability=True)
            await shim.create_orchestrator()

            assert shim.observability is not None
            assert isinstance(shim.observability, ObservabilityIntegration)
            # Orchestrator should have observability attached
            assert hasattr(mock_orchestrator, "observability")
            assert mock_orchestrator.observability == shim.observability

    @pytest.mark.asyncio
    async def test_observability_disabled(self, mock_settings, mock_orchestrator):
        """Test that observability can be disabled."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, enable_observability=False)
            await shim.create_orchestrator()

            assert shim.observability is None

    @pytest.mark.asyncio
    async def test_observability_session_id_propagated(self, mock_settings, mock_orchestrator):
        """Test that session ID is propagated to observability."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(
                mock_settings,
                session_id="test-session-456",
                enable_observability=True,
            )
            await shim.create_orchestrator()

            assert shim.observability._session_id == "test-session-456"


class TestFrameworkShimVertical:
    """Tests for vertical configuration."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.provider = "anthropic"
        settings.model = "claude-3-5-sonnet"
        return settings

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orch = MagicMock()
        orch.prompt_builder = MagicMock()
        orch.prompt_builder.set_custom_prompt = MagicMock()
        # Add protocol methods for tools
        orch._enabled_tools = set()
        orch.set_enabled_tools = MagicMock(
            side_effect=lambda tools: setattr(orch, "_enabled_tools", tools)
        )
        orch.get_enabled_tools = MagicMock(side_effect=lambda: orch._enabled_tools)
        # Add vertical context storage (set via pipeline)
        orch._vertical_context = None

        def set_context(context):
            orch._vertical_context = context
        orch.set_vertical_context = MagicMock(side_effect=set_context)
        return orch

    @pytest.fixture(autouse=True)
    def register_mock_vertical(self):
        """Register mock vertical for tests."""
        VerticalRegistry.register(MockVertical)
        yield
        VerticalRegistry.unregister("test_vertical")

    @pytest.mark.asyncio
    async def test_vertical_tools_applied(self, mock_settings, mock_orchestrator):
        """Test that vertical tools are applied to orchestrator."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, vertical=MockVertical)
            await shim.create_orchestrator()

            # Check that set_enabled_tools was called with the correct tools
            mock_orchestrator.set_enabled_tools.assert_called_once()
            enabled_tools = mock_orchestrator._enabled_tools
            assert "read" in enabled_tools
            assert "write" in enabled_tools
            assert "edit" in enabled_tools

    @pytest.mark.asyncio
    async def test_vertical_system_prompt_applied(self, mock_settings, mock_orchestrator):
        """Test that vertical system prompt is applied.

        With capability-based invocation, the orchestrator's set_custom_prompt
        method is called directly (via _invoke_capability mapping).
        """
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, vertical=MockVertical)
            await shim.create_orchestrator()

            # New capability-based approach calls orchestrator.set_custom_prompt directly
            mock_orchestrator.set_custom_prompt.assert_called_once_with(
                "You are a test assistant."
            )

    @pytest.mark.asyncio
    async def test_vertical_stages_applied(self, mock_settings, mock_orchestrator):
        """Test that vertical stages are applied via vertical context."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, vertical=MockVertical)
            await shim.create_orchestrator()

            # Stages are applied via vertical context, not _vertical_stages
            assert hasattr(mock_orchestrator, "_vertical_context")
            context = mock_orchestrator._vertical_context
            assert context is not None
            # Check that stages were applied to context
            stages = context.stages
            assert "INITIAL" in stages
            assert "PLANNING" in stages
            assert "EXECUTION" in stages

    @pytest.mark.asyncio
    async def test_vertical_by_name(self, mock_settings, mock_orchestrator):
        """Test vertical lookup by string name."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, vertical="test_vertical")
            await shim.create_orchestrator()

            assert shim.vertical == MockVertical
            # Check that set_enabled_tools was called with the correct tools
            enabled_tools = mock_orchestrator._enabled_tools
            assert "read" in enabled_tools
            assert "write" in enabled_tools
            assert "edit" in enabled_tools

    @pytest.mark.asyncio
    async def test_vertical_not_found(self, mock_settings, mock_orchestrator):
        """Test handling of non-existent vertical name."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, vertical="nonexistent_vertical")
            await shim.create_orchestrator()

            # Should not crash, vertical should be None
            assert shim.vertical is None

    @pytest.mark.asyncio
    async def test_vertical_config_accessible(self, mock_settings, mock_orchestrator):
        """Test that vertical config is accessible after creation."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, vertical=MockVertical)
            await shim.create_orchestrator()

            config = shim.vertical_config
            assert config is not None
            assert config.system_prompt == "You are a test assistant."


class TestFrameworkShimLifecycle:
    """Tests for lifecycle event emission."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.provider = "anthropic"
        return settings

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orch = MagicMock()
        orch.conversation_state = MagicMock()
        return orch

    @pytest.mark.asyncio
    async def test_emit_session_start(self, mock_settings, mock_orchestrator):
        """Test session start event emission."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, enable_observability=True)
            await shim.create_orchestrator()

            # Mock the observability methods
            shim._observability.on_session_start = MagicMock()

            shim.emit_session_start({"mode": "cli"})
            shim._observability.on_session_start.assert_called_once_with({"mode": "cli"})

    @pytest.mark.asyncio
    async def test_emit_session_end(self, mock_settings, mock_orchestrator):
        """Test session end event emission."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, enable_observability=True)
            await shim.create_orchestrator()

            # Mock the observability methods
            shim._observability.on_session_end = MagicMock()

            shim.emit_session_end(tool_calls=5, duration_seconds=10.5, success=True)
            shim._observability.on_session_end.assert_called_once_with(
                tool_calls=5,
                duration_seconds=10.5,
                success=True,
            )

    @pytest.mark.asyncio
    async def test_lifecycle_events_noop_without_observability(
        self, mock_settings, mock_orchestrator
    ):
        """Test that lifecycle events are no-ops when observability disabled."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, enable_observability=False)
            await shim.create_orchestrator()

            # Should not raise
            shim.emit_session_start({"mode": "cli"})
            shim.emit_session_end(tool_calls=0)


class TestGetVerticalFunction:
    """Tests for get_vertical helper function."""

    @pytest.fixture(autouse=True)
    def register_mock_vertical(self):
        """Register mock vertical for tests."""
        VerticalRegistry.register(MockVertical)
        yield
        VerticalRegistry.unregister("test_vertical")

    def test_get_vertical_by_exact_name(self):
        """Test getting vertical by exact name."""
        vertical = get_vertical("test_vertical")
        assert vertical == MockVertical

    def test_get_vertical_case_insensitive(self):
        """Test case-insensitive vertical lookup."""
        vertical = get_vertical("TEST_VERTICAL")
        assert vertical == MockVertical

        vertical = get_vertical("Test_Vertical")
        assert vertical == MockVertical

    def test_get_vertical_none(self):
        """Test get_vertical with None."""
        vertical = get_vertical(None)
        assert vertical is None

    def test_get_vertical_not_found(self):
        """Test get_vertical with non-existent name."""
        vertical = get_vertical("nonexistent")
        assert vertical is None


class TestListVerticalsFunction:
    """Tests for list_verticals helper function."""

    @pytest.fixture(autouse=True)
    def register_mock_vertical(self):
        """Register mock vertical for tests and ensure built-ins are present."""
        # Ensure built-in verticals are registered (they may have been cleared by other tests)
        from victor.verticals.coding import CodingAssistant
        from victor.verticals.devops import DevOpsAssistant
        from victor.verticals.research import ResearchAssistant

        # Register built-ins if not already present
        if not VerticalRegistry.get("coding"):
            VerticalRegistry.register(CodingAssistant)
        if not VerticalRegistry.get("devops"):
            VerticalRegistry.register(DevOpsAssistant)
        if not VerticalRegistry.get("research"):
            VerticalRegistry.register(ResearchAssistant)

        # Register mock vertical
        VerticalRegistry.register(MockVertical)
        yield
        VerticalRegistry.unregister("test_vertical")

    def test_list_verticals_includes_registered(self):
        """Test that list_verticals includes registered verticals."""
        names = list_verticals()
        assert "test_vertical" in names

    def test_list_verticals_includes_builtins(self):
        """Test that list_verticals includes built-in verticals."""
        names = list_verticals()
        # Built-in verticals registered in victor/verticals/__init__.py
        assert "coding" in names


class TestFrameworkShimProperties:
    """Tests for FrameworkShim properties."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        return MagicMock()

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_orchestrator_property_before_create(self, mock_settings):
        """Test orchestrator property before creation."""
        shim = FrameworkShim(mock_settings)
        assert shim.orchestrator is None

    @pytest.mark.asyncio
    async def test_orchestrator_property_after_create(self, mock_settings, mock_orchestrator):
        """Test orchestrator property after creation."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings)
            await shim.create_orchestrator()

            assert shim.orchestrator == mock_orchestrator

    @pytest.mark.asyncio
    async def test_observability_property(self, mock_settings, mock_orchestrator):
        """Test observability property."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings, enable_observability=True)
            await shim.create_orchestrator()

            assert shim.observability is not None
            assert isinstance(shim.observability, ObservabilityIntegration)
