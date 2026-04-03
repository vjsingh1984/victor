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

"""Tests for ResilienceFacade domain facade."""

import asyncio
import pytest
from unittest.mock import MagicMock

from victor.agent.facades.resilience_facade import ResilienceFacade
from victor.agent.facades.protocols import ResilienceFacadeProtocol


class TestResilienceFacadeInit:
    """Tests for ResilienceFacade initialization."""

    def test_init_with_all_components(self):
        """ResilienceFacade initializes with all components provided."""
        recovery = MagicMock()
        code = MagicMock()

        facade = ResilienceFacade(
            resilience_runtime=MagicMock(),
            recovery_handler=recovery,
            recovery_integration=MagicMock(),
            recovery_coordinator=MagicMock(),
            chunk_generator=MagicMock(),
            context_manager=MagicMock(),
            rl_coordinator=MagicMock(),
            code_manager=code,
            background_tasks=set(),
            cancel_event=None,
            is_streaming=True,
        )

        assert facade.recovery_handler is recovery
        assert facade.code_manager is code
        assert facade.is_streaming is True

    def test_init_with_minimal_components(self):
        """ResilienceFacade initializes with no required components (all optional)."""
        facade = ResilienceFacade()

        assert facade.resilience_runtime is None
        assert facade.recovery_handler is None
        assert facade.recovery_integration is None
        assert facade.recovery_coordinator is None
        assert facade.chunk_generator is None
        assert facade.context_manager is None
        assert facade.rl_coordinator is None
        assert facade.code_manager is None
        assert facade.background_tasks == set()
        assert facade.cancel_event is None
        assert facade.is_streaming is False


class TestResilienceFacadeProperties:
    """Tests for ResilienceFacade property access."""

    @pytest.fixture
    def facade(self):
        """Create a ResilienceFacade with mock components."""
        return ResilienceFacade(
            resilience_runtime=MagicMock(name="runtime"),
            recovery_handler=MagicMock(name="handler"),
            recovery_integration=MagicMock(name="integration"),
            recovery_coordinator=MagicMock(name="coordinator"),
            chunk_generator=MagicMock(name="chunks"),
            context_manager=MagicMock(name="context"),
            rl_coordinator=MagicMock(name="rl"),
            code_manager=MagicMock(name="code"),
            is_streaming=False,
        )

    def test_resilience_runtime_property(self, facade):
        """ResilienceRuntime property returns the runtime."""
        assert facade.resilience_runtime._mock_name == "runtime"

    def test_recovery_handler_property(self, facade):
        """RecoveryHandler property returns the handler."""
        assert facade.recovery_handler._mock_name == "handler"

    def test_recovery_handler_setter(self, facade):
        """RecoveryHandler setter updates the handler."""
        new_handler = MagicMock(name="new_handler")
        facade.recovery_handler = new_handler
        assert facade.recovery_handler is new_handler

    def test_recovery_integration_setter(self, facade):
        """RecoveryIntegration setter updates the integration."""
        new_integration = MagicMock(name="new_integration")
        facade.recovery_integration = new_integration
        assert facade.recovery_integration is new_integration

    def test_recovery_coordinator_property(self, facade):
        """RecoveryCoordinator property returns the coordinator."""
        assert facade.recovery_coordinator._mock_name == "coordinator"

    def test_chunk_generator_property(self, facade):
        """ChunkGenerator property returns the generator."""
        assert facade.chunk_generator._mock_name == "chunks"

    def test_code_manager_property(self, facade):
        """CodeManager property returns the manager."""
        assert facade.code_manager._mock_name == "code"

    def test_is_streaming_property(self, facade):
        """IsStreaming property returns the streaming flag."""
        assert facade.is_streaming is False

    def test_is_streaming_setter(self, facade):
        """IsStreaming setter updates the flag."""
        facade.is_streaming = True
        assert facade.is_streaming is True

    def test_cancel_event_setter(self, facade):
        """CancelEvent setter updates the event."""
        event = asyncio.Event()
        facade.cancel_event = event
        assert facade.cancel_event is event


class TestResilienceFacadeProtocolConformance:
    """Tests that ResilienceFacade satisfies ResilienceFacadeProtocol."""

    def test_satisfies_protocol(self):
        """ResilienceFacade structurally conforms to ResilienceFacadeProtocol."""
        facade = ResilienceFacade()
        assert isinstance(facade, ResilienceFacadeProtocol)

    def test_protocol_properties_present(self):
        """All protocol-required properties are present on ResilienceFacade."""
        required = [
            "recovery_handler",
            "context_manager",
        ]
        facade = ResilienceFacade()
        for prop_name in required:
            assert hasattr(facade, prop_name), f"Missing protocol property: {prop_name}"
