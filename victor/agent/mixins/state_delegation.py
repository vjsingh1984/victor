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

"""State delegation mixin for automatic state property delegation.

This mixin provides automatic delegation of state-related properties to
their respective coordinators/components. It uses a descriptor pattern
for type safety and clean separation of concerns.

Benefits:
- Reduces boilerplate: No need for 15+ explicit state property definitions
- Type-safe: Uses descriptors with proper type hints
- Maintainable: Add new state delegations without adding properties
- Clear ownership: Each state property knows its source

Usage:
    class AgentOrchestrator(StateDelegationMixin):
        def __init__(self):
            self._conversation_coordinator = ConversationCoordinator()
            self._mode_coordinator = ModeCoordinator()

        # Define state delegations
        _state_delegations = {
            "messages": ("_conversation_coordinator", "messages"),
            "current_mode": ("_mode_coordinator", "current_mode"),
        }

    # Now access state directly
    orchestrator.messages  # Delegates to conversation_coordinator.messages
    orchestrator.current_mode  # Delegates to mode_coordinator.current_mode
"""

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class StateDelegationDescriptor:
    """Descriptor for delegating property access to a component.

    This descriptor provides automatic delegation of property access to
    a component attribute. It handles both getting and setting values.

    Example:
        class MyClass:
            messages = StateDelegationDescriptor(
                component_attr="_conversation_coordinator",
                property_name="messages",
                type_hint="List[Message]",
            )

            def __init__(self):
                self._conversation_coordinator = ConversationCoordinator()

        obj = MyClass()
        obj.messages  # Delegates to _conversation_coordinator.messages
    """

    def __init__(
        self,
        component_attr: str,
        property_name: str,
        type_hint: Optional[str] = None,
        doc: Optional[str] = None,
    ):
        """Initialize the delegation descriptor.

        Args:
            component_attr: Name of the component attribute (e.g., "_conversation_coordinator")
            property_name: Name of the property on the component (e.g., "messages")
            type_hint: Optional type hint for documentation
            doc: Optional docstring for the property
        """
        self.component_attr = component_attr
        self.property_name = property_name
        self.type_hint = type_hint
        self.__doc__ = doc or f"Delegated property from {component_attr}.{property_name}"

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Any:
        """Get the property value from the component.

        Args:
            obj: The instance object
            objtype: The type of the instance (None if accessed via class)

        Returns:
            The property value from the component

        Raises:
            AttributeError: If component is not initialized or property doesn't exist
        """
        if obj is None:
            # Accessed via class, return the descriptor itself
            return self

        # Get the component
        try:
            component = object.__getattribute__(obj, self.component_attr)
        except AttributeError:
            raise AttributeError(
                f"Component '{self.component_attr}' not initialized. "
                f"Cannot delegate property '{self.property_name}'."
            )

        if component is None:
            raise AttributeError(
                f"Component '{self.component_attr}' is None. "
                f"Cannot delegate property '{self.property_name}'."
            )

        # Get the property from the component
        try:
            return getattr(component, self.property_name)
        except AttributeError:
            raise AttributeError(
                f"Property '{self.property_name}' not found on component "
                f"'{type(component).__name__}' ({self.component_attr})."
            )

    def __set__(self, obj: Any, value: Any) -> None:
        """Set the property value on the component.

        Args:
            obj: The instance object
            value: The value to set

        Raises:
            AttributeError: If component is not initialized or property is read-only
        """
        # Get the component
        try:
            component = object.__getattribute__(obj, self.component_attr)
        except AttributeError:
            raise AttributeError(
                f"Component '{self.component_attr}' not initialized. "
                f"Cannot set property '{self.property_name}'."
            )

        if component is None:
            raise AttributeError(
                f"Component '{self.component_attr}' is None. "
                f"Cannot set property '{self.property_name}'."
            )

        # Set the property on the component
        try:
            setattr(component, self.property_name, value)
        except AttributeError as e:
            raise AttributeError(
                f"Cannot set property '{self.property_name}' on component "
                f"'{type(component).__name__}' ({self.component_attr}): {e}"
            ) from e


class StateDelegationMixin:
    """Mixin for automatic state property delegation.

    This mixin uses descriptors to delegate state-related properties to
    their respective coordinators. This eliminates the need for explicit
    property definitions while maintaining type safety and documentation.

    Common delegations:
    - messages -> _conversation_coordinator.messages
    - current_mode -> _mode_coordinator.current_mode
    - stage -> _conversation_coordinator.stage

    Type Safety:
        The descriptors preserve type information through __doc__ and
        can be properly annotated in __annotations__.

    Example:
        class AgentOrchestrator(StateDelegationMixin):
            # Define state delegations (can also be defined at class level)
            _state_delegations = {
                "messages": ("_conversation_coordinator", "messages"),
                "current_mode": ("_mode_coordinator", "current_mode"),
            }

            def __init__(self):
                self._conversation_coordinator = ConversationCoordinator()
                self._mode_coordinator = ModeCoordinator()

        # Access delegated properties
        orchestrator.messages  # Works!
        orchestrator.current_mode  # Works!
    """

    # Default state delegations (can be overridden by subclasses)
    _state_delegations: Dict[str, Tuple[str, str]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize subclass by creating delegation descriptors.

        This method is called when a subclass is created. It iterates
        through _state_delegations and creates descriptors for each one.
        """
        super().__init_subclass__(**kwargs)

        # Get all delegations from class hierarchy
        delegations: Dict[str, Tuple[str, Optional[str], str]] = {}

        # Collect from all base classes (in reverse MRO order)
        for base in reversed(cls.__mro__):
            if hasattr(base, "_state_delegations"):
                for prop_name, (comp_attr, prop_name_on_comp) in base._state_delegations.items():
                    # Generate docstring
                    doc = (
                        f"Delegated property from {comp_attr}.{prop_name_on_comp}.\n\n"
                        f"Provides access to state managed by {comp_attr}."
                    )

                    # Store with generated docstring
                    delegations[prop_name] = (comp_attr, prop_name_on_comp, doc)

        # Create descriptors for each delegation
        for prop_name, delegation_info in delegations.items():
            # Only create if not already defined on the class
            if prop_name not in cls.__dict__:
                comp_attr, prop_name_on_comp, doc = delegation_info
                # Skip if property name is None (can't delegate to None property)
                if prop_name_on_comp is None:
                    continue
                # At this point, prop_name_on_comp is guaranteed to be str
                assert isinstance(prop_name_on_comp, str), "Property name must be string"
                descriptor = StateDelegationDescriptor(
                    component_attr=comp_attr,
                    property_name=prop_name_on_comp,
                    doc=doc,
                )
                setattr(cls, prop_name, descriptor)

    @classmethod
    def add_state_delegation(
        cls,
        property_name: str,
        component_attr: str,
        component_property: str,
    ) -> None:
        """Add a new state delegation dynamically.

        This allows adding new delegations after class definition.

        Args:
            property_name: Name of the property on the class
            component_attr: Name of the component attribute
            component_property: Name of the property on the component
        """
        # Add to delegations dict
        cls._state_delegations[property_name] = (component_attr, component_property)

        # Create and set descriptor
        doc = (
            f"Delegated property from {component_attr}.{component_property}.\n\n"
            f"Provides access to state managed by {component_attr}."
        )
        descriptor = StateDelegationDescriptor(
            component_attr=component_attr,
            property_name=component_property,
            doc=doc,
        )
        setattr(cls, property_name, descriptor)

    def get_delegated_state_properties(self) -> Dict[str, Tuple[str, str]]:
        """Get all delegated state property mappings.

        Returns:
            Dict mapping property names to (component_attr, component_property) tuples
        """
        return dict(self._state_delegations)
