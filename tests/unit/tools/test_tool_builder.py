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

"""Tests for ToolBuilder.

Part of HIGH-005: Initialization Complexity reduction.
"""

import pytest
from unittest.mock import Mock, MagicMock
from victor.agent.builders.tool_builder import ToolBuilder
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
    factory.create_middleware_chain.return_value = (MagicMock(), MagicMock())
    factory.setup_semantic_selection.return_value = (False, None)
    return factory


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    provider = Mock()
    provider.__class__.__name__ = "MockProvider"
    return provider


def test_tool_builder_initialization(mock_settings):
    """Test ToolBuilder initialization."""
    builder = ToolBuilder(mock_settings)

    assert builder.settings is mock_settings
    assert builder._factory is None


def test_tool_builder_initialization_with_factory(mock_settings, mock_factory):
    """Test ToolBuilder initialization with factory."""
    builder = ToolBuilder(mock_settings, factory=mock_factory)

    assert builder.settings is mock_settings
    assert builder._factory is mock_factory


def test_tool_builder_build_without_factory_raises_error(mock_settings):
    """Test that build() without factory and provider raises error."""
    builder = ToolBuilder(mock_settings)

    with pytest.raises(ValueError, match="provider and model are required"):
        builder.build()


def test_tool_builder_build_creates_tool_budget(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates tool budget."""
    mock_caps = Mock()
    mock_caps.recommended_tool_budget = 15
    mock_factory.initialize_tool_budget.return_value = 15

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(
        provider=mock_provider, model="test-model", tool_calling_caps=mock_caps
    )

    assert "tool_budget" in components
    assert components["tool_budget"] == 15


def test_tool_builder_build_creates_tool_cache(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates tool cache."""
    mock_cache = Mock(name="tool_cache")
    mock_factory.create_tool_cache.return_value = mock_cache

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "tool_cache" in components
    assert components["tool_cache"] is mock_cache


def test_tool_builder_build_creates_tool_graph(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates tool dependency graph."""
    mock_graph = Mock(name="tool_graph")
    mock_factory.create_tool_dependency_graph.return_value = mock_graph

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "tool_graph" in components
    assert components["tool_graph"] is mock_graph


def test_tool_builder_build_creates_tool_registry(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates tool registry."""
    mock_registry = Mock(name="tools")
    mock_factory.create_tool_registry.return_value = mock_registry

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "tools" in components
    assert components["tools"] is mock_registry


def test_tool_builder_build_creates_tool_registrar(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates tool registrar."""
    mock_registrar = Mock(name="tool_registrar")
    mock_factory.create_tool_registrar.return_value = mock_registrar

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "tool_registrar" in components
    assert components["tool_registrar"] is mock_registrar


def test_tool_builder_build_creates_argument_normalizer(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates argument normalizer."""
    mock_normalizer = Mock(name="argument_normalizer")
    mock_factory.create_argument_normalizer.return_value = mock_normalizer

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "argument_normalizer" in components
    assert components["argument_normalizer"] is mock_normalizer


def test_tool_builder_build_creates_middleware_chain(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates middleware chain."""
    mock_chain = [Mock(name="middleware1"), Mock(name="middleware2")]
    mock_correction = Mock(name="code_correction")
    mock_factory.create_middleware_chain.return_value = (mock_chain, mock_correction)

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "middleware_chain" in components
    assert "code_correction_middleware" in components
    assert components["middleware_chain"] is mock_chain
    assert components["code_correction_middleware"] is mock_correction


def test_tool_builder_build_creates_safety_checker(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates safety checker."""
    mock_checker = Mock(name="safety_checker")
    mock_factory.create_safety_checker.return_value = mock_checker

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "safety_checker" in components
    assert components["safety_checker"] is mock_checker


def test_tool_builder_build_creates_tool_executor(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates tool executor."""
    mock_executor = Mock(name="tool_executor")
    mock_factory.create_tool_executor.return_value = mock_executor

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "tool_executor" in components
    assert components["tool_executor"] is mock_executor


def test_tool_builder_build_creates_parallel_executor(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates parallel executor."""
    mock_parallel = Mock(name="parallel_executor")
    mock_factory.create_parallel_executor.return_value = mock_parallel

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "parallel_executor" in components
    assert components["parallel_executor"] is mock_parallel


def test_tool_builder_build_creates_tool_selector(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates tool selector."""
    mock_selector = Mock(name="tool_selector")
    mock_factory.create_tool_selector.return_value = mock_selector

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "tool_selector" in components
    assert components["tool_selector"] is mock_selector


def test_tool_builder_build_creates_tool_pipeline(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates tool pipeline."""
    mock_pipeline = Mock(name="tool_pipeline")
    mock_factory.create_tool_pipeline.return_value = mock_pipeline

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(
        provider=mock_provider,
        model="test-model",
        on_tool_start=Mock(),
        on_tool_complete=Mock(),
    )

    assert "tool_pipeline" in components
    assert components["tool_pipeline"] is mock_pipeline


def test_tool_builder_build_registers_components(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() registers all built components."""
    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    # Check that components are registered
    for name, component in components.items():
        if component is not None:
            assert builder.has_component(name)
            assert builder.get_component(name) is component


def test_tool_builder_get_tools(mock_settings, mock_factory, mock_provider):
    """Test getting tools from built components."""
    mock_tools = Mock(name="tools")
    mock_factory.create_tool_registry.return_value = mock_tools

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    builder.build(provider=mock_provider, model="test-model")

    assert builder.get_tools() is mock_tools


def test_tool_builder_get_tool_pipeline(mock_settings, mock_factory, mock_provider):
    """Test getting tool pipeline from built components."""
    mock_pipeline = Mock(name="tool_pipeline")
    mock_factory.create_tool_pipeline.return_value = mock_pipeline

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    builder.build(provider=mock_provider, model="test-model")

    assert builder.get_tool_pipeline() is mock_pipeline


def test_tool_builder_get_tool_executor(mock_settings, mock_factory, mock_provider):
    """Test getting tool executor from built components."""
    mock_executor = Mock(name="tool_executor")
    mock_factory.create_tool_executor.return_value = mock_executor

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    builder.build(provider=mock_provider, model="test-model")

    assert builder.get_tool_executor() is mock_executor


def test_tool_builder_get_tool_selector(mock_settings, mock_factory, mock_provider):
    """Test getting tool selector from built components."""
    mock_selector = Mock(name="tool_selector")
    mock_factory.create_tool_selector.return_value = mock_selector

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    builder.build(provider=mock_provider, model="test-model")

    assert builder.get_tool_selector() is mock_selector


def test_tool_builder_build_with_callbacks(mock_settings, mock_factory, mock_provider):
    """Test that build() properly passes callbacks."""
    mock_on_start = Mock()
    mock_on_complete = Mock()
    mock_background_callback = Mock()
    mock_registrar = Mock()
    mock_factory.create_tool_registrar.return_value = mock_registrar

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    builder.build(
        provider=mock_provider,
        model="test-model",
        on_tool_start=mock_on_start,
        on_tool_complete=mock_on_complete,
        background_task_callback=mock_background_callback,
    )

    # Verify callback was set on registrar
    mock_registrar.set_background_task_callback.assert_called_once_with(
        mock_background_callback
    )


def test_tool_builder_build_creates_workflow_components(
    mock_settings, mock_factory, mock_provider
):
    """Test that build() creates workflow and code execution components."""
    mock_workflow_registry = Mock(name="workflow_registry")
    mock_code_manager = Mock(name="code_manager")
    mock_factory.create_workflow_registry.return_value = mock_workflow_registry
    mock_factory.create_code_execution_manager.return_value = mock_code_manager

    builder = ToolBuilder(mock_settings, factory=mock_factory)
    components = builder.build(provider=mock_provider, model="test-model")

    assert "workflow_registry" in components
    assert "code_manager" in components
    assert components["workflow_registry"] is mock_workflow_registry
    assert components["code_manager"] is mock_code_manager
