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

"""Component accessor mixin for automatic component access.

This mixin provides automatic access to private component attributes
without requiring explicit property definitions. It uses __getattr__
to intercept attribute access and delegate to the corresponding private
attribute (e.g., `conversation_controller` -> `_conversation_controller`).

Benefits:
- Reduces boilerplate: No need for 30+ explicit property definitions
- Type-safe: Works with type checkers when __annotations__ are defined
- Maintainable: Add new components without adding properties
- Backward compatible: Existing code continues to work

Usage:
    class AgentOrchestrator(ComponentAccessorMixin):
        def __init__(self):
            self._conversation_controller = ConversationController()
            self._tool_pipeline = ToolPipeline()

        # Define mappings in __annotations__ for type safety
        __annotations__ = {
            "conversation_controller": "ConversationController",
            "tool_pipeline": "ToolPipeline",
        }

    # Now access components directly
    orchestrator.conversation_controller  # Works!
    orchestrator.tool_pipeline  # Works!
"""

import logging
from typing import Any, Dict, Optional, Set, Type

logger = logging.getLogger(__name__)


class ComponentAccessorMixin:
    """Mixin for automatic component access without explicit properties.

    This mixin uses __getattr__ to provide transparent access to private
    component attributes. When an attribute is accessed that doesn't exist
    on the class, __getattr__ is called. We check if there's a corresponding
    private attribute (prefixed with _) and return it.

    This approach:
    1. Eliminates ~30 property definitions (700+ lines)
    2. Maintains type safety via __annotations__
    3. Provides clear error messages for invalid access
    4. Supports lazy initialization patterns

    Type Safety:
        For type checkers to work correctly, subclasses should define
        component types in __annotations__ or use TYPE_CHECKING imports.

    Example:
        class MyOrchestrator(ComponentAccessorMixin):
            from typing import TYPE_CHECKING
            if TYPE_CHECKING:
                from victor.agent.conversation import ConversationController
                from victor.agent.tool_pipeline import ToolPipeline

            __annotations__ = {
                "conversation_controller": "ConversationController",
                "tool_pipeline": "ToolPipeline",
            }

            def __init__(self):
                self._conversation_controller = ConversationController()
                self._tool_pipeline = ToolPipeline()
    """

    # Component name to private attribute mapping
    # This is used by __getattr__ to find the correct private attribute
    _component_map: Dict[str, str] = {
        "conversation_controller": "_conversation_controller",
        "tool_pipeline": "_tool_pipeline",
        "streaming_controller": "_streaming_controller",
        "streaming_handler": "_streaming_handler",
        "task_analyzer": "_task_analyzer",
        "observability": "_observability",
        "memory_manager": "_memory_manager_wrapper",
        "tool_calling_caps": "_tool_calling_caps_internal",
        "provider_manager": "_provider_manager",
        "context_compactor": "_context_compactor",
        "tool_output_formatter": "_tool_output_formatter",
        "usage_analytics": "_usage_analytics",
        "sequence_tracker": "_sequence_tracker",
        "recovery_handler": "_recovery_handler",
        "recovery_integration": "_recovery_integration",
        "recovery_coordinator": "_recovery_coordinator",
        "prompt_coordinator": "_prompt_coordinator",
        "context_manager": "_context_manager",
        "mode_coordinator": "_mode_coordinator",
        "validation_coordinator": "_validation_coordinator",
        "evaluation_coordinator": "_evaluation_coordinator",
        "metrics_coordinator": "_metrics_coordinator",
        "response_coordinator": "_response_coordinator",
        "workflow_coordinator": "_workflow_coordinator",
        "checkpoint_coordinator": "_checkpoint_coordinator",
        "search_coordinator": "_search_coordinator",
        "tool_selector": "_tool_selector",
        "tool_registrar": "_tool_registrar",
        "factory": "_factory",
        "intelligent_integration": "_intelligent_integration",
        "subagent_orchestrator": "_subagent_orchestrator",
        "team_coordinator": "_team_coordinator",
        "mode_workflow_team_coordinator": "_mode_workflow_team_coordinator",
        "conversation_coordinator": "_conversation_coordinator",
        "vertical_integration_adapter": "_vertical_integration_adapter",
        "safety_checker": "_safety_checker",
        "auto_committer": "_auto_committer",
        "lifecycle_manager": "_lifecycle_manager",
        "provider_switch_coordinator": "_provider_switch_coordinator",
        "streaming_coordinator": "_streaming_coordinator",
    }

    # Attributes that should raise AttributeError instead of being handled
    _component_exclusions: Set[str] = {
        # Internal attributes that should not be accessed
        "_component_map",
        "_component_exclusions",
        "__dict__",
        "__class__",
    }

    def __getattr__(self, name: str) -> Any:
        """Get attribute value from private component storage.

        This method is called only when the attribute is not found through
        normal lookup (i.e., not in instance __dict__ or class hierarchy).

        Args:
            name: Attribute name being accessed

        Returns:
            Value from the corresponding private attribute

        Raises:
            AttributeError: If no corresponding private attribute exists
        """
        # Check if this is a component we should handle
        if name in self._component_exclusions:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Get the private attribute name for this component
        private_attr = self._component_map.get(name)

        if private_attr is None:
            # Not a component we know about
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Get the value from the private attribute
        try:
            value = object.__getattribute__(self, private_attr)
        except AttributeError as e:
            # Private attribute doesn't exist
            raise AttributeError(
                f"Component '{name}' (mapped to '{private_attr}') not initialized. "
                f"Ensure the component is properly initialized before access."
            ) from e

        # Handle None values for optional components
        if value is None and name not in (
            "observability",
            "memory_manager",
            "recovery_handler",
            "auto_committer",
            "subagent_orchestrator",
            "checkpoint_manager",
        ):
            logger.debug(
                f"Component '{name}' is None. This may indicate optional "
                f"component not initialized or disabled."
            )

        return value

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute value, handling component setters specially.

        This allows setters like `orchestrator.observability = value` to work
        correctly by writing to the private attribute.

        Args:
            name: Attribute name being set
            value: Value to set
        """
        # Check if this is a component we should handle
        if name in self._component_map:
            private_attr = self._component_map[name]
            object.__setattr__(self, private_attr, value)
        else:
            # Normal attribute setting
            object.__setattr__(self, name, value)

    def has_component(self, name: str) -> bool:
        """Check if a component is initialized (not None).

        Args:
            name: Component name (without underscore prefix)

        Returns:
            True if component exists and is not None
        """
        if name not in self._component_map:
            return False

        private_attr = self._component_map[name]
        return (
            hasattr(self, private_attr) and object.__getattribute__(self, private_attr) is not None
        )

    def get_component_names(self) -> Set[str]:
        """Get all registered component names.

        Returns:
            Set of component names that can be accessed via this mixin
        """
        return set(self._component_map.keys())

    @classmethod
    def register_component(cls, name: str, private_attr: Optional[str] = None) -> None:
        """Register a new component mapping.

        This allows subclasses or external code to add new component
        mappings dynamically.

        Args:
            name: Public component name (without underscore)
            private_attr: Private attribute name (default: f"_{name}")
        """
        if private_attr is None:
            private_attr = f"_{name}"

        cls._component_map[name] = private_attr
