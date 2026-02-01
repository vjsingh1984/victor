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

"""Test FactoryAwareBuilder class."""

import pytest
from unittest.mock import Mock

from victor.agent.builders.base import FactoryAwareBuilder, ComponentBuilder
from victor.config.settings import Settings


class ConcreteFactoryAwareBuilder(FactoryAwareBuilder):
    """Concrete implementation for testing."""

    def build(self, **kwargs):
        """Simple build implementation for testing."""
        return {"test_component": Mock()}


class TestFactoryAwareBuilder:
    """Test FactoryAwareBuilder functionality."""

    def test_is_subclass_of_component_builder(self):
        """Test that FactoryAwareBuilder is a proper subclass."""
        assert issubclass(FactoryAwareBuilder, ComponentBuilder)
        assert isinstance(ConcreteFactoryAwareBuilder(Settings()), ComponentBuilder)

    def test_initialization_without_factory(self):
        """Test initialization without factory parameter."""
        builder = ConcreteFactoryAwareBuilder(Settings())
        assert builder.factory is None
        assert builder._factory is None

    def test_initialization_with_factory(self):
        """Test initialization with factory parameter."""
        mock_factory = Mock()
        builder = ConcreteFactoryAwareBuilder(Settings(), factory=mock_factory)
        assert builder.factory is mock_factory
        assert builder._factory is mock_factory

    def test_ensure_factory_creates_on_first_call(self):
        """Test that _ensure_factory stores factory on first call."""
        builder = ConcreteFactoryAwareBuilder(Settings())
        mock_provider = Mock()
        mock_model = "test-model"

        # Create a mock factory to inject
        mock_factory = Mock()

        # Manually set the factory to test behavior
        builder._factory = mock_factory

        factory = builder._ensure_factory(provider=mock_provider, model=mock_model)

        # Verify it returns the stored factory
        assert factory is mock_factory
        assert builder._factory is mock_factory

    def test_ensure_factory_reuses_existing_factory(self):
        """Test that _ensure_factory reuses existing factory."""
        mock_factory = Mock()
        builder = ConcreteFactoryAwareBuilder(Settings(), factory=mock_factory)

        # Call _ensure_factory multiple times
        factory1 = builder._ensure_factory(provider=Mock(), model="test-model")
        factory2 = builder._ensure_factory(provider=Mock(), model="test-model")

        # Should return the same factory instance
        assert factory1 is mock_factory
        assert factory2 is mock_factory
        assert factory1 is factory2

    def test_ensure_factory_raises_error_without_provider_and_model(self):
        """Test that _ensure_factory raises ValueError when provider/model missing."""
        builder = ConcreteFactoryAwareBuilder(Settings())

        with pytest.raises(ValueError) as exc_info:
            builder._ensure_factory()

        assert "provider and model are required" in str(exc_info.value)

    def test_ensure_factory_raises_error_without_provider(self):
        """Test that _ensure_factory raises ValueError when provider missing."""
        builder = ConcreteFactoryAwareBuilder(Settings())

        with pytest.raises(ValueError) as exc_info:
            builder._ensure_factory(model="test-model")

        assert "provider and model are required" in str(exc_info.value)

    def test_ensure_factory_raises_error_without_model(self):
        """Test that _ensure_factory raises ValueError when model missing."""
        builder = ConcreteFactoryAwareBuilder(Settings())

        with pytest.raises(ValueError) as exc_info:
            builder._ensure_factory(provider=Mock())

        assert "provider and model are required" in str(exc_info.value)

    def test_ensure_factory_with_all_parameters(self):
        """Test _ensure_factory with all parameters when factory exists."""
        builder = ConcreteFactoryAwareBuilder(Settings())
        mock_provider = Mock()

        # Create a mock factory to inject
        mock_factory = Mock()
        builder._factory = mock_factory

        factory = builder._ensure_factory(
            provider=mock_provider,
            model="test-model",
            temperature=0.5,
            max_tokens=2048,
            provider_name="test-provider",
            profile_name="test-profile",
            tool_selection={"test": "config"},
            thinking=True,
        )

        # Verify it returns the stored factory (doesn't recreate)
        assert factory is mock_factory

    def test_register_components_registers_all_non_none(self):
        """Test that _register_components registers all non-None components."""
        builder = ConcreteFactoryAwareBuilder(Settings())
        components = {
            "component1": Mock(),
            "component2": Mock(),
            "component3": None,
            "component4": Mock(),
        }

        builder._register_components(components)

        # Verify non-None components were registered
        assert builder.has_component("component1")
        assert builder.has_component("component2")
        assert not builder.has_component("component3")
        assert builder.has_component("component4")

    def test_register_components_skips_none_values(self):
        """Test that _register_components skips None values."""
        builder = ConcreteFactoryAwareBuilder(Settings())
        components = {
            "valid": Mock(),
            "none_value": None,
            "another_valid": Mock(),
        }

        builder._register_components(components)

        # Verify None values were skipped
        assert builder.get_component("valid") is not None
        assert builder.get_component("none_value") is None
        assert builder.get_component("another_valid") is not None

    def test_register_components_with_all_none(self):
        """Test _register_components with all None components."""
        builder = ConcreteFactoryAwareBuilder(Settings())
        components = {"component1": None, "component2": None}

        builder._register_components(components)

        # Verify nothing was registered
        assert builder.has_component("component1") is False
        assert builder.has_component("component2") is False

    def test_register_components_with_empty_dict(self):
        """Test _register_components with empty dictionary."""
        builder = ConcreteFactoryAwareBuilder(Settings())
        components = {}

        # Should not raise any errors
        builder._register_components(components)

    def test_factory_property_returns_none_when_not_created(self):
        """Test that factory property returns None when factory not created."""
        builder = ConcreteFactoryAwareBuilder(Settings())
        assert builder.factory is None

    def test_factory_property_returns_factory_when_created(self):
        """Test that factory property returns factory when created."""
        mock_factory = Mock()
        builder = ConcreteFactoryAwareBuilder(Settings(), factory=mock_factory)
        assert builder.factory is mock_factory

    def test_factory_property_does_not_create_factory(self):
        """Test that factory property does not create factory."""
        builder = ConcreteFactoryAwareBuilder(Settings())

        # Accessing factory property should not create factory
        factory = builder.factory
        assert factory is None

        # Verify factory is still None
        assert builder._factory is None

    def test_preserves_component_builder_functionality(self):
        """Test that FactoryAwareBuilder preserves ComponentBuilder functionality."""
        builder = ConcreteFactoryAwareBuilder(Settings())

        # Test component registration
        mock_component = Mock()
        builder.register_component("test", mock_component)
        assert builder.get_component("test") is mock_component
        assert builder.has_component("test")

        # Test cache clearing
        builder.clear_cache()
        assert not builder.has_component("test")

    def test_multiple_calls_to_ensure_factory_reuse_instance(self):
        """Test that multiple calls to _ensure_factory reuse the same instance."""
        builder = ConcreteFactoryAwareBuilder(Settings())
        mock_factory = Mock()
        builder._factory = mock_factory

        # Call _ensure_factory multiple times with different parameters
        factory1 = builder._ensure_factory(provider=Mock(), model="model1")
        factory2 = builder._ensure_factory(provider=Mock(), model="model2")
        factory3 = builder._ensure_factory(provider=Mock(), model="model3")

        # All should return the same instance
        assert factory1 is factory2
        assert factory2 is factory3
        assert factory1 is mock_factory

    def test_settings_attribute_accessible(self):
        """Test that settings attribute is accessible from FactoryAwareBuilder."""
        settings = Settings()
        builder = ConcreteFactoryAwareBuilder(settings)
        assert builder.settings is settings
