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

"""Tests for ProviderFacade domain facade."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.facades.provider_facade import ProviderFacade
from victor.agent.facades.protocols import ProviderFacadeProtocol


class TestProviderFacadeInit:
    """Tests for ProviderFacade initialization."""

    def test_init_with_all_components(self):
        """ProviderFacade initializes with all components provided."""
        provider = MagicMock()
        manager = MagicMock()

        facade = ProviderFacade(
            provider=provider,
            model="claude-3-sonnet",
            provider_name="anthropic",
            temperature=0.5,
            max_tokens=8192,
            thinking=True,
            provider_manager=manager,
            provider_runtime=MagicMock(),
            provider_coordinator=MagicMock(),
            provider_switch_coordinator=MagicMock(),
        )

        assert facade.provider is provider
        assert facade.model == "claude-3-sonnet"
        assert facade.provider_name == "anthropic"
        assert facade.temperature == 0.5
        assert facade.max_tokens == 8192
        assert facade.thinking is True
        assert facade.provider_manager is manager

    def test_init_with_minimal_components(self):
        """ProviderFacade initializes with only required components."""
        provider = MagicMock()
        manager = MagicMock()

        facade = ProviderFacade(
            provider=provider,
            model="gpt-4",
            provider_manager=manager,
        )

        assert facade.provider is provider
        assert facade.model == "gpt-4"
        assert facade.provider_name is None
        assert facade.temperature == 0.7
        assert facade.max_tokens == 4096
        assert facade.thinking is False
        assert facade.provider_manager is manager
        assert facade.provider_runtime is None
        assert facade.provider_coordinator is None
        assert facade.provider_switch_coordinator is None

    def test_init_derives_deprecated_coordinators_from_runtime(self):
        """Deprecated coordinator accessors derive from provider runtime by default."""
        provider = MagicMock()
        manager = MagicMock()
        provider_coordinator = MagicMock(name="provider_coordinator")
        switch_coordinator = MagicMock(name="switch_coordinator")
        runtime = SimpleNamespace(
            provider_coordinator=provider_coordinator,
            provider_switch_coordinator=switch_coordinator,
        )

        facade = ProviderFacade(
            provider=provider,
            model="gpt-4",
            provider_manager=manager,
            provider_runtime=runtime,
        )

        assert facade.provider_runtime is runtime
        assert facade.provider_coordinator is provider_coordinator
        assert facade.provider_switch_coordinator is switch_coordinator

    def test_init_prefers_explicit_deprecated_coordinator_over_runtime(self):
        """Explicit compatibility slots still override runtime-derived values."""
        provider = MagicMock()
        manager = MagicMock()
        explicit_provider_coordinator = MagicMock(name="explicit_provider_coordinator")
        explicit_switch_coordinator = MagicMock(name="explicit_switch_coordinator")
        runtime = SimpleNamespace(
            provider_coordinator=MagicMock(name="runtime_provider_coordinator"),
            provider_switch_coordinator=MagicMock(name="runtime_switch_coordinator"),
        )

        facade = ProviderFacade(
            provider=provider,
            model="gpt-4",
            provider_manager=manager,
            provider_runtime=runtime,
            provider_coordinator=explicit_provider_coordinator,
            provider_switch_coordinator=explicit_switch_coordinator,
        )

        assert facade.provider_coordinator is explicit_provider_coordinator
        assert facade.provider_switch_coordinator is explicit_switch_coordinator


class TestProviderFacadeProperties:
    """Tests for ProviderFacade property access."""

    @pytest.fixture
    def facade(self):
        """Create a ProviderFacade with mock components."""
        return ProviderFacade(
            provider=MagicMock(name="provider"),
            model="test-model",
            provider_name="test-provider",
            temperature=0.8,
            max_tokens=2048,
            thinking=False,
            provider_manager=MagicMock(name="manager"),
            provider_runtime=MagicMock(name="runtime"),
            provider_coordinator=MagicMock(name="coordinator"),
            provider_switch_coordinator=MagicMock(name="switch"),
        )

    def test_provider_property(self, facade):
        """Provider property returns the active provider."""
        assert facade.provider._mock_name == "provider"

    def test_model_property(self, facade):
        """Model property returns the model identifier."""
        assert facade.model == "test-model"

    def test_provider_name_property(self, facade):
        """ProviderName property returns the label."""
        assert facade.provider_name == "test-provider"

    def test_temperature_property(self, facade):
        """Temperature property returns the sampling temperature."""
        assert facade.temperature == 0.8

    def test_max_tokens_property(self, facade):
        """MaxTokens property returns the maximum tokens."""
        assert facade.max_tokens == 2048

    def test_thinking_property(self, facade):
        """Thinking property returns the thinking mode flag."""
        assert facade.thinking is False

    def test_provider_setter(self, facade):
        """Provider setter updates the active provider."""
        new_provider = MagicMock(name="new_provider")
        facade.provider = new_provider
        assert facade.provider is new_provider

    def test_model_setter(self, facade):
        """Model setter updates the model identifier."""
        facade.model = "new-model"
        assert facade.model == "new-model"

    def test_temperature_setter(self, facade):
        """Temperature setter updates the sampling temperature."""
        facade.temperature = 0.3
        assert facade.temperature == 0.3

    def test_max_tokens_setter(self, facade):
        """MaxTokens setter updates the maximum tokens."""
        facade.max_tokens = 16384
        assert facade.max_tokens == 16384

    def test_thinking_setter(self, facade):
        """Thinking setter updates the thinking mode flag."""
        facade.thinking = True
        assert facade.thinking is True


class TestProviderFacadeProtocolConformance:
    """Tests that ProviderFacade satisfies ProviderFacadeProtocol."""

    def test_satisfies_protocol(self):
        """ProviderFacade structurally conforms to ProviderFacadeProtocol."""
        facade = ProviderFacade(
            provider=MagicMock(),
            model="test",
            provider_manager=MagicMock(),
        )
        assert isinstance(facade, ProviderFacadeProtocol)

    def test_protocol_properties_present(self):
        """All protocol-required properties are present on ProviderFacade."""
        required = [
            "provider",
            "model",
            "provider_manager",
        ]
        facade = ProviderFacade(
            provider=MagicMock(),
            model="test",
            provider_manager=MagicMock(),
        )
        for prop in required:
            assert hasattr(facade, prop), f"Missing protocol property: {prop}"
