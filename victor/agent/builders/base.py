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

"""Base class for component builders using builder pattern.

This module provides the abstract base class for all builders used in
AgentOrchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from victor.config.settings import Settings

import logging

logger = logging.getLogger(__name__)


class ComponentBuilder(ABC):
    """Base class for component builders using builder pattern.

    This abstract base class provides common functionality for all builders
    that construct components for AgentOrchestrator initialization.

    The builder pattern allows for:
    - Separation of construction logic from business logic
    - Independent testing of component initialization
    - Clearer dependency chains
    - Easier modification and extension

    Attributes:
        settings: Application settings
        _built_components: Cache of previously built components for reuse
    """

    def __init__(self, settings: Settings):
        """Initialize the builder.

        Args:
            settings: Application settings to use for component initialization
        """
        self.settings = settings
        self._built_components: Dict[str, Any] = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def build(self, **kwargs) -> Any:
        """Build and return the component(s).

        This method must be implemented by concrete builder classes to
        construct their specific components.

        Args:
            **kwargs: Dependencies from other builders or external sources

        Returns:
            The built component(s). Can be a single component or a dictionary
            of multiple components.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement build()")

    def get_component(self, name: str) -> Optional[Any]:
        """Get a previously built component from cache.

        Args:
            name: The name of the component to retrieve

        Returns:
            The component if it exists, None otherwise
        """
        return self._built_components.get(name)

    def register_component(self, name: str, component: Any) -> None:
        """Register a built component in the cache for dependency injection.

        Args:
            name: The name to register the component under
            component: The component instance to register
        """
        self._built_components[name] = component
        self._logger.debug(f"Registered component: {name}")

    def has_component(self, name: str) -> bool:
        """Check if a component has been built and registered.

        Args:
            name: The name of the component to check

        Returns:
            True if the component exists, False otherwise
        """
        return name in self._built_components

    def clear_cache(self) -> None:
        """Clear all cached components.

        This is useful for testing or when rebuilding components with
        different configurations.
        """
        self._built_components.clear()
        self._logger.debug("Cleared component cache")
