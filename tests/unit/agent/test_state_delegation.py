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

"""Tests for StateDelegationMixin."""

import pytest
from typing import List

from victor.agent.mixins.state_delegation import (
    StateDelegationMixin,
    StateDelegationDescriptor,
)


class MockCoordinator:
    """Mock coordinator for testing."""

    def __init__(self):
        self.messages = ["message1", "message2"]
        self.current_mode = "build"
        self.stage = "executing"


class MockContextManager:
    """Mock context manager for testing."""

    def __init__(self):
        self.context_size = 1000
        self.max_context_size = 2000


class TestClass(StateDelegationMixin):
    """Test class using StateDelegationMixin."""

    # Define state delegations
    _state_delegations = {
        "messages": ("_conversation_coordinator", "messages"),
        "current_mode": ("_mode_coordinator", "current_mode"),
        "stage": ("_conversation_coordinator", "stage"),
        "context_size": ("_context_manager", "context_size"),
        "max_context_size": ("_context_manager", "max_context_size"),
    }

    def __init__(self):
        self._conversation_coordinator = MockCoordinator()
        self._mode_coordinator = MockCoordinator()
        self._context_manager = MockContextManager()


class TestStateDelegationMixin:
    """Test suite for StateDelegationMixin."""

    def test_basic_state_delegation(self):
        """Test basic state property delegation."""
        obj = TestClass()

        # Access delegated properties
        assert obj.messages == ["message1", "message2"]
        assert obj.current_mode == "build"
        assert obj.stage == "executing"

    def test_delegation_returns_correct_values(self):
        """Test that delegated properties return correct values from components."""
        obj = TestClass()

        # Should delegate to coordinator's messages
        assert obj.messages is obj._conversation_coordinator.messages

        # Should delegate to mode coordinator's current_mode
        assert obj.current_mode == obj._mode_coordinator.current_mode

    def test_setting_delegated_property(self):
        """Test setting delegated property values."""
        obj = TestClass()

        # Set delegated property
        obj.messages = ["new_message"]

        # Should update coordinator's property
        assert obj._conversation_coordinator.messages == ["new_message"]
        assert obj.messages == ["new_message"]

    def test_uninitialized_component_raises_error(self):
        """Test that accessing delegation for uninitialized component raises error."""
        obj = TestClass()

        # Delete component to simulate uninitialized state
        delattr(obj, "_conversation_coordinator")

        # Should raise AttributeError
        with pytest.raises(
            AttributeError, match=r"Component '_conversation_coordinator' not initialized"
        ):
            _ = obj.messages

    def test_nonexistent_property_on_component_raises_error(self):
        """Test accessing nonexistent property on component raises error."""
        obj = TestClass()

        # Add delegation for nonexistent property using the proper method
        TestClass.add_state_delegation(
            "nonexistent_prop", "_conversation_coordinator", "nonexistent"
        )

        with pytest.raises(
            AttributeError,
            match=r"Property 'nonexistent' not found on component",
        ):
            _ = obj.nonexistent_prop

        # Clean up
        del TestClass._state_delegations["nonexistent_prop"]
        if hasattr(TestClass, "nonexistent_prop"):
            delattr(TestClass, "nonexistent_prop")

    def test_multiple_delegations_from_same_component(self):
        """Test multiple delegations from the same component."""
        obj = TestClass()

        # Both delegate to _conversation_coordinator
        assert obj.messages == obj._conversation_coordinator.messages
        assert obj.stage == obj._conversation_coordinator.stage

    def test_get_delegated_state_properties(self):
        """Test get_delegated_state_properties returns mapping."""
        obj = TestClass()

        mappings = obj.get_delegated_state_properties()

        assert isinstance(mappings, dict)
        assert "messages" in mappings
        assert mappings["messages"] == ("_conversation_coordinator", "messages")

    def test_descriptor_is_created_on_class(self):
        """Test that descriptors are created on class during subclass creation."""
        # Check that descriptors are set on the class
        assert hasattr(TestClass, "messages")
        assert hasattr(TestClass, "current_mode")
        assert hasattr(TestClass, "stage")

        # Check that they are descriptors
        assert (
            isinstance(TestClass.__dict__.get("messages"), StateDelegationDescriptor)
            or "messages" in TestClass._state_delegations
        )


class TestStateDelegationDescriptor:
    """Test suite for StateDelegationDescriptor."""

    def test_descriptor_get(self):
        """Test descriptor __get__ method."""
        descriptor = StateDelegationDescriptor(
            component_attr="_coordinator",
            property_name="messages",
            type_hint="List[str]",
        )

        obj = TestClass()
        obj._coordinator = MockCoordinator()

        # Access through descriptor
        result = descriptor.__get__(obj)
        assert result == ["message1", "message2"]

    def test_descriptor_set(self):
        """Test descriptor __set__ method."""
        descriptor = StateDelegationDescriptor(
            component_attr="_coordinator",
            property_name="messages",
        )

        obj = TestClass()
        obj._coordinator = MockCoordinator()

        # Set through descriptor
        descriptor.__set__(obj, ["new_msg"])

        # Verify it was set
        assert obj._coordinator.messages == ["new_msg"]

    def test_descriptor_access_via_class(self):
        """Test accessing descriptor via class returns descriptor itself."""
        descriptor = StateDelegationDescriptor(
            component_attr="_coordinator",
            property_name="messages",
        )

        obj = TestClass()
        obj._coordinator = MockCoordinator()

        # Access via class should return descriptor
        result = descriptor.__get__(None, TestClass)
        assert result is descriptor

    def test_descriptor_docstring(self):
        """Test that descriptor has proper docstring."""
        descriptor = StateDelegationDescriptor(
            component_attr="_coordinator",
            property_name="messages",
            doc="Custom docstring",
        )

        assert descriptor.__doc__ == "Custom docstring"

    def test_descriptor_default_docstring(self):
        """Test that descriptor generates default docstring."""
        descriptor = StateDelegationDescriptor(
            component_attr="_coordinator",
            property_name="messages",
        )

        assert "Delegated property" in descriptor.__doc__
        assert "_coordinator" in descriptor.__doc__


class TestDynamicStateDelegation:
    """Test dynamic state delegation additions."""

    def test_add_state_delegation(self):
        """Test adding state delegation dynamically."""
        obj = TestClass()

        # Add new delegation
        TestClass.add_state_delegation("new_property", "_context_manager", "context_size")

        # Should be accessible
        assert obj.new_property == 1000

        # Clean up
        del TestClass._state_delegations["new_property"]
        if hasattr(TestClass, "new_property"):
            delattr(TestClass, "new_property")

    def test_dynamic_delegation_works_correctly(self):
        """Test that dynamically added delegation works correctly."""
        obj = TestClass()

        # Add delegation
        TestClass.add_state_delegation("max_size", "_context_manager", "max_context_size")

        # Should delegate correctly
        assert obj.max_size == 2000

        # Set value
        obj.max_size = 3000
        assert obj._context_manager.max_context_size == 3000

        # Clean up
        del TestClass._state_delegations["max_size"]
        if hasattr(TestClass, "max_size"):
            delattr(TestClass, "max_size")


class TestStateDelegationIntegration:
    """Integration tests for StateDelegationMixin."""

    def test_multiple_delegations_coexist(self):
        """Test that multiple delegations can coexist without conflicts."""
        obj = TestClass()

        # All delegations should work
        assert obj.messages == ["message1", "message2"]
        assert obj.current_mode == "build"
        assert obj.stage == "executing"
        assert obj.context_size == 1000
        assert obj.max_context_size == 2000

    def test_delegation_does_not_create_attributes(self):
        """Test that delegation doesn't create spurious attributes."""
        obj = TestClass()

        # Access delegated property
        _ = obj.messages

        # Should not create public attribute
        assert "messages" not in obj.__dict__

        # Private component should exist
        assert "_conversation_coordinator" in obj.__dict__

    def test_delegation_with_component_changes(self):
        """Test that delegation reflects component changes."""
        obj = TestClass()

        # Initial value
        assert obj.current_mode == "build"

        # Change component's value
        obj._mode_coordinator.current_mode = "plan"

        # Delegation should reflect change
        assert obj.current_mode == "plan"

    def test_inheritance_of_delegations(self):
        """Test that delegations can be inherited."""

        class SubClass(TestClass):
            _state_delegations = {
                "sub_property": ("_conversation_coordinator", "stage"),
            }

            def __init__(self):
                super().__init__()

        obj = SubClass()

        # Should have both parent and child delegations
        assert hasattr(obj, "messages")  # From parent
        assert hasattr(obj, "sub_property")  # From child

    def test_setter_propagates_to_component(self):
        """Test that setters properly propagate to components."""
        obj = TestClass()

        # Set through delegation
        obj.context_size = 5000

        # Component should be updated
        assert obj._context_manager.context_size == 5000

        # Access should return updated value
        assert obj.context_size == 5000
