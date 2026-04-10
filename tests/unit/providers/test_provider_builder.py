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

"""Tests for ProviderBuilder.

Part of HIGH-005: Initialization Complexity reduction.
"""

import pytest
from unittest.mock import Mock, MagicMock
from victor.agent.builders.provider_builder import ProviderBuilder
from victor.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    return Mock(spec=Settings)


@pytest.fixture
def mock_factory():
    """Create mock factory with required default return values."""
    factory = MagicMock()
    # Set up default return values for methods that return tuples
    factory.create_tool_calling_matrix.return_value = ({}, {})
    factory.create_provider_manager_with_adapter.return_value = (
        MagicMock(),  # provider_manager
        MagicMock(),  # provider_instance
        "test-model",  # model_id
        "test-provider",  # provider_label
        MagicMock(),  # tool_adapter
        MagicMock(),  # tool_calling_caps
    )
    return factory


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    provider = Mock()
    provider.__class__.__name__ = "MockProvider"
    return provider


def test_provider_builder_initialization(mock_settings):
    """Test ProviderBuilder initialization."""
    builder = ProviderBuilder(mock_settings)

    assert builder.settings is mock_settings
    assert builder._factory is None


def test_provider_builder_initialization_with_factory(mock_settings, mock_factory):
    """Test ProviderBuilder initialization with factory."""
    builder = ProviderBuilder(mock_settings, factory=mock_factory)

    assert builder.settings is mock_settings
    assert builder._factory is mock_factory


def test_provider_builder_build_without_factory_raises_error(mock_settings):
    """Test that build() without factory and provider raises error."""
    builder = ProviderBuilder(mock_settings)

    with pytest.raises(ValueError, match="provider and model are required"):
        builder.build()


def test_provider_builder_build_creates_tool_calling_matrix(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates tool calling matrix."""
    mock_models = {"model1": True, "model2": False}
    mock_capabilities = {"model1": Mock()}
    mock_factory.create_tool_calling_matrix.return_value = (
        mock_models,
        mock_capabilities,
    )

    builder = ProviderBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "tool_calling_models" in components
    assert "tool_capabilities" in components
    assert components["tool_calling_models"] is mock_models
    assert components["tool_capabilities"] is mock_capabilities


def test_provider_builder_build_creates_provider_manager(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates provider manager."""
    mock_manager = Mock(name="provider_manager")
    mock_adapter = Mock(name="tool_adapter")
    mock_caps = Mock(name="tool_calling_caps")

    mock_factory.create_provider_manager_with_adapter.return_value = (
        mock_manager,
        mock_provider,
        "test-model",
        "test-provider",
        mock_adapter,
        mock_caps,
    )

    builder = ProviderBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "provider_manager" in components
    assert "provider" in components
    assert "model" in components
    assert "provider_name" in components
    assert "tool_adapter" in components
    assert "tool_calling_caps" in components


def test_provider_builder_build_creates_prompt_builder(mock_settings, mock_factory, mock_provider):
    """Test that build() creates system prompt builder."""
    mock_prompt_builder = Mock(name="prompt_builder")
    mock_factory.create_system_prompt_builder.return_value = mock_prompt_builder

    mock_factory.create_provider_manager_with_adapter.return_value = (
        Mock(),
        mock_provider,
        "test-model",
        "test-provider",
        Mock(),
        Mock(),
    )

    builder = ProviderBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "prompt_builder" in components
    assert components["prompt_builder"] is mock_prompt_builder


def test_provider_builder_build_registers_components(mock_settings, mock_factory, mock_provider):
    """Test that build() registers all built components."""
    mock_factory.create_provider_manager_with_adapter.return_value = (
        Mock(),
        mock_provider,
        "test-model",
        "test-provider",
        Mock(),
        Mock(),
    )

    builder = ProviderBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    # Check that components are registered
    for name, component in components.items():
        if component is not None:
            assert builder.has_component(name)
            assert builder.get_component(name) is component


def test_provider_builder_get_tool_adapter(mock_settings, mock_factory, mock_provider):
    """Test getting tool adapter from built components."""
    mock_adapter = Mock(name="tool_adapter")
    mock_factory.create_provider_manager_with_adapter.return_value = (
        Mock(),
        mock_provider,
        "test-model",
        "test-provider",
        mock_adapter,
        Mock(),
    )

    builder = ProviderBuilder(mock_settings, factory=mock_factory)
    builder.build(provider=mock_provider, model="test-model")

    assert builder.get_tool_adapter() is mock_adapter


def test_provider_builder_get_tool_calling_caps(mock_settings, mock_factory, mock_provider):
    """Test getting tool calling capabilities from built components."""
    mock_caps = Mock(name="tool_calling_caps")
    mock_factory.create_provider_manager_with_adapter.return_value = (
        Mock(),
        mock_provider,
        "test-model",
        "test-provider",
        Mock(),
        mock_caps,
    )

    builder = ProviderBuilder(mock_settings, factory=mock_factory)
    builder.build(provider=mock_provider, model="test-model")

    assert builder.get_tool_calling_caps() is mock_caps


def test_provider_builder_get_provider(mock_settings, mock_factory, mock_provider):
    """Test getting provider from built components."""
    mock_factory.create_provider_manager_with_adapter.return_value = (
        Mock(),
        mock_provider,
        "test-model",
        "test-provider",
        Mock(),
        Mock(),
    )

    builder = ProviderBuilder(mock_settings, factory=mock_factory)
    builder.build(provider=mock_provider, model="test-model")

    assert builder.get_provider() is mock_provider


def test_provider_builder_get_provider_manager(mock_settings, mock_factory, mock_provider):
    """Test getting provider manager from built components."""
    mock_manager = Mock(name="provider_manager")
    mock_factory.create_provider_manager_with_adapter.return_value = (
        mock_manager,
        mock_provider,
        "test-model",
        "test-provider",
        Mock(),
        Mock(),
    )

    builder = ProviderBuilder(mock_settings, factory=mock_factory)
    builder.build(provider=mock_provider, model="test-model")

    assert builder.get_provider_manager() is mock_manager
