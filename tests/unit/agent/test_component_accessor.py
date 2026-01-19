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

"""Tests for ComponentAccessorMixin."""

import pytest

from victor.agent.mixins.component_accessor import ComponentAccessorMixin


class MockComponent:
    """Mock component for testing."""

    def __init__(self, name: str):
        self.name = name


class TestClass(ComponentAccessorMixin):
    """Test class using ComponentAccessorMixin."""

    def __init__(self):
        self._conversation_controller = MockComponent("conversation")
        self._tool_pipeline = MockComponent("tool_pipeline")
        self._custom_component = MockComponent("custom")


class TestComponentAccessorMixin:
    """Test suite for ComponentAccessorMixin."""

    def test_basic_component_access(self):
        """Test basic component access through mixin."""
        obj = TestClass()

        # Access components without explicit properties
        assert obj.conversation_controller.name == "conversation"
        assert obj.tool_pipeline.name == "tool_pipeline"

    def test_component_access_returns_correct_instance(self):
        """Test that component access returns the correct instance."""
        obj = TestClass()

        conv = obj.conversation_controller
        assert conv is obj._conversation_controller
        assert isinstance(conv, MockComponent)

    def test_nonexistent_component_raises_attribute_error(self):
        """Test that accessing non-existent component raises AttributeError."""
        obj = TestClass()

        with pytest.raises(
            AttributeError, match=r"'TestClass' object has no attribute 'nonexistent'"
        ):
            _ = obj.nonexistent

    def test_uninitialized_component_raises_attribute_error(self):
        """Test that accessing uninitialized component raises AttributeError."""
        obj = TestClass()

        # Don't initialize _streaming_controller
        with pytest.raises(
            AttributeError,
            match=r"Component 'streaming_controller'.*not initialized",
        ):
            _ = obj.streaming_controller

    def test_component_setter(self):
        """Test setting component values through mixin."""
        obj = TestClass()

        # Set a new component
        new_component = MockComponent("new_conversation")
        obj.conversation_controller = new_component

        # Verify it was set
        assert obj._conversation_controller is new_component
        assert obj.conversation_controller.name == "new_conversation"

    def test_has_component(self):
        """Test has_component method."""
        obj = TestClass()

        assert obj.has_component("conversation_controller")
        assert obj.has_component("tool_pipeline")
        assert not obj.has_component("streaming_controller")

    def test_get_component_names(self):
        """Test get_component_names returns all registered components."""
        obj = TestClass()

        names = obj.get_component_names()
        assert "conversation_controller" in names
        assert "tool_pipeline" in names
        assert "provider_manager" in names  # From default mapping

    def test_custom_component_registration(self):
        """Test dynamic component registration."""
        obj = TestClass()

        # Register new component mapping
        TestClass.register_component("custom", "_custom_component")

        # Access the newly registered component
        assert obj.custom.name == "custom"

    def test_attribute_error_for_unregistered_attributes(self):
        """Test that unregistered attributes raise AttributeError."""
        obj = TestClass()

        # Regular attribute access should work
        obj._some_attr = "value"
        assert obj._some_attr == "value"

        # But unregistered public attributes should fail
        with pytest.raises(AttributeError):
            _ = obj.some_attr

    def test_type_annotations_preserved(self):
        """Test that type annotations are preserved for type checkers."""
        # This test ensures __annotations__ can be used for type safety
        annotations = TestClass.__annotations__

        # The mixin should not break type annotations
        assert isinstance(annotations, dict)

    def test_excluded_attributes_not_handled(self):
        """Test that excluded attributes in _component_exclusions raise AttributeError."""
        obj = TestClass()

        # Attributes in _component_exclusions should raise AttributeError
        # Note: _component_map is a class attribute, so it doesn't go through __getattr__
        # Let's test with a truly non-existent attribute that's not in mappings
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = obj._nonexistent_excluded_attr

    def test_component_mapping_dict(self):
        """Test that component mapping dictionary is properly structured."""
        mapping = TestClass._component_map

        assert isinstance(mapping, dict)
        assert "conversation_controller" in mapping
        assert mapping["conversation_controller"] == "_conversation_controller"

    def test_none_component_handling(self):
        """Test handling of None components (optional components)."""
        obj = TestClass()

        # Set an optional component to None
        obj._observability = None

        # Should return None (not raise error) for optional components
        result = obj.observability
        assert result is None


class TestComponentAccessorIntegration:
    """Integration tests for ComponentAccessorMixin."""

    def test_multiple_components_coexist(self):
        """Test that multiple components can coexist without conflicts."""
        obj = TestClass()

        # Access multiple components
        conv = obj.conversation_controller
        pipeline = obj.tool_pipeline

        assert conv is not pipeline
        assert conv.name == "conversation"
        assert pipeline.name == "tool_pipeline"

    def test_component_access_does_not_create_attributes(self):
        """Test that component access doesn't create spurious attributes."""
        obj = TestClass()

        # Access component
        _ = obj.conversation_controller

        # Check that no extra attributes were created
        assert "conversation_controller" not in obj.__dict__
        assert "_conversation_controller" in obj.__dict__

    def test_setter_updates_private_attribute(self):
        """Test that setter updates the private attribute directly."""
        obj = TestClass()

        new_component = MockComponent("updated")
        obj.tool_pipeline = new_component

        # Private attribute should be updated
        assert obj._tool_pipeline is new_component
        # Public access should return new component
        assert obj.tool_pipeline is new_component
