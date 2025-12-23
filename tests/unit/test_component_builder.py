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

"""Tests for ComponentBuilder base class.

Part of HIGH-005: Initialization Complexity reduction.
"""

import pytest
from unittest.mock import Mock
from victor.agent.builders.base import ComponentBuilder
from victor.config.settings import Settings


class ConcreteBuilder(ComponentBuilder):
    """Concrete implementation for testing."""

    def build(self, **kwargs):
        """Build a simple test component."""
        component = {"test": "component"}
        self.register_component("test_component", component)
        return component


def test_component_builder_initialization():
    """Test ComponentBuilder initialization."""
    settings = Mock(spec=Settings)
    builder = ConcreteBuilder(settings)

    assert builder.settings is settings
    assert builder._built_components == {}
    assert builder._logger is not None


def test_component_builder_register_component():
    """Test registering a component."""
    settings = Mock(spec=Settings)
    builder = ConcreteBuilder(settings)

    component = {"name": "test"}
    builder.register_component("my_component", component)

    assert builder.has_component("my_component")
    assert builder.get_component("my_component") is component


def test_component_builder_get_nonexistent_component():
    """Test getting a component that doesn't exist."""
    settings = Mock(spec=Settings)
    builder = ConcreteBuilder(settings)

    assert builder.get_component("nonexistent") is None
    assert not builder.has_component("nonexistent")


def test_component_builder_clear_cache():
    """Test clearing the component cache."""
    settings = Mock(spec=Settings)
    builder = ConcreteBuilder(settings)

    builder.register_component("comp1", {"a": 1})
    builder.register_component("comp2", {"b": 2})
    assert len(builder._built_components) == 2

    builder.clear_cache()
    assert len(builder._built_components) == 0
    assert not builder.has_component("comp1")
    assert not builder.has_component("comp2")


def test_component_builder_build_not_implemented():
    """Test that build() must be implemented by subclasses."""

    class AbstractBuilder(ComponentBuilder):
        pass

    settings = Mock(spec=Settings)

    # Python 3 prevents instantiation of abstract classes with unimplemented abstract methods
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        AbstractBuilder(settings)


def test_component_builder_multiple_components():
    """Test registering and retrieving multiple components."""
    settings = Mock(spec=Settings)
    builder = ConcreteBuilder(settings)

    components = {
        "comp1": {"type": "A"},
        "comp2": {"type": "B"},
        "comp3": {"type": "C"},
    }

    for name, comp in components.items():
        builder.register_component(name, comp)

    for name, comp in components.items():
        assert builder.has_component(name)
        assert builder.get_component(name) is comp


def test_component_builder_overwrite_component():
    """Test overwriting a registered component."""
    settings = Mock(spec=Settings)
    builder = ConcreteBuilder(settings)

    builder.register_component("comp", {"version": 1})
    assert builder.get_component("comp") == {"version": 1}

    builder.register_component("comp", {"version": 2})
    assert builder.get_component("comp") == {"version": 2}


def test_component_builder_build_with_registration():
    """Test that build() can register components."""
    settings = Mock(spec=Settings)
    builder = ConcreteBuilder(settings)

    result = builder.build()

    assert result == {"test": "component"}
    assert builder.has_component("test_component")
    assert builder.get_component("test_component") == result
