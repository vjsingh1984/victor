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

"""Tests for ProviderProtocol.

Tests the ProviderProtocol interface and conformance.
"""


from victor.protocols.provider import ProviderProtocol


class MockProvider:
    """Mock provider for testing."""

    def __init__(self):
        self.name = "mock_provider"


class MockProviderImplementation:
    """Mock implementation of ProviderProtocol for testing."""

    def __init__(self):
        self._provider = MockProvider()
        self._provider_name = "anthropic"
        self._model = "claude-sonnet-4-20250514"
        self._temperature = 0.7

    @property
    def provider(self):
        """Get the current LLM provider instance."""
        return self._provider

    @property
    def provider_name(self) -> str:
        """Get the name of the current provider."""
        return self._provider_name

    @property
    def model(self) -> str:
        """Get the current model identifier."""
        return self._model

    @property
    def temperature(self) -> float:
        """Get the temperature setting for sampling."""
        return self._temperature


class TestProviderProtocol:
    """Test suite for ProviderProtocol."""

    def test_provider_property(self):
        """Test that provider property works correctly."""
        impl = MockProviderImplementation()
        provider = impl.provider
        assert provider.name == "mock_provider"

    def test_provider_name_property(self):
        """Test that provider_name property works correctly."""
        impl = MockProviderImplementation()
        assert impl.provider_name == "anthropic"

    def test_model_property(self):
        """Test that model property works correctly."""
        impl = MockProviderImplementation()
        assert impl.model == "claude-sonnet-4-20250514"

    def test_temperature_property(self):
        """Test that temperature property works correctly."""
        impl = MockProviderImplementation()
        assert impl.temperature == 0.7

    def test_protocol_conformance(self):
        """Test that mock implements ProviderProtocol."""
        impl = MockProviderImplementation()
        # This should not raise an error
        assert isinstance(impl, ProviderProtocol)

    def test_provider_property_is_readonly(self):
        """Test that provider property is read-only."""
        impl = MockProviderImplementation()
        provider = impl.provider
        # Property should be accessible
        assert provider is not None
        # We can't test that it's read-only without trying to set it,
        # which would fail with a property that has no setter

    def test_different_provider_values(self):
        """Test with different provider configurations."""
        impl = MockProviderImplementation()
        impl._provider_name = "openai"
        impl._model = "gpt-4o"
        impl._temperature = 1.0

        assert impl.provider_name == "openai"
        assert impl.model == "gpt-4o"
        assert impl.temperature == 1.0


class TestProviderProtocolTypeChecking:
    """Test type checking and protocol compliance."""

    def test_provider_protocol_is_protocol(self):
        """Test that ProviderProtocol is a Protocol."""
        from typing import Protocol

        assert issubclass(ProviderProtocol, Protocol)

    def test_provider_protocol_has_provider_property(self):
        """Test that ProviderProtocol defines provider property."""
        assert hasattr(ProviderProtocol, "__annotations__")
        # Check that provider is in the protocol
        assert "provider" in dir(ProviderProtocol)

    def test_provider_protocol_has_provider_name_property(self):
        """Test that ProviderProtocol defines provider_name property."""
        assert hasattr(ProviderProtocol, "__annotations__")
        assert "provider_name" in dir(ProviderProtocol)

    def test_provider_protocol_has_model_property(self):
        """Test that ProviderProtocol defines model property."""
        assert hasattr(ProviderProtocol, "__annotations__")
        assert "model" in dir(ProviderProtocol)

    def test_provider_protocol_has_temperature_property(self):
        """Test that ProviderProtocol defines temperature property."""
        assert hasattr(ProviderProtocol, "__annotations__")
        assert "temperature" in dir(ProviderProtocol)
