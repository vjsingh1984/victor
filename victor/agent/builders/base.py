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


class FactoryAwareBuilder(ComponentBuilder):
    """Base class for builders that use OrchestratorFactory.

    This class extends ComponentBuilder to provide factory management
    capabilities, eliminating duplication across ToolBuilder, ProviderBuilder,
    and ServiceBuilder.

    Attributes:
        settings: Application settings
        _factory: Optional OrchestratorFactory instance (created lazily)
    """

    def __init__(self, settings: Settings, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the factory-aware builder.

        Args:
            settings: Application settings to use for component initialization
            factory: Optional OrchestratorFactory instance (created if not provided)
        """
        super().__init__(settings)
        self._factory = factory

    def _ensure_factory(
        self,
        provider: Any = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        provider_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        tool_selection: Optional[Dict[str, Any]] = None,
        thinking: bool = False,
    ) -> "OrchestratorFactory":
        """Ensure factory exists, creating it if necessary.

        This method implements the factory creation logic that was previously
        duplicated across ToolBuilder, ProviderBuilder, and ServiceBuilder.

        Args:
            provider: LLM provider instance
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            provider_name: Optional provider label from profile
            profile_name: Optional profile name for session tracking
            tool_selection: Optional tool selection configuration
            thinking: Enable extended thinking mode

        Returns:
            The existing or newly created OrchestratorFactory instance

        Raises:
            ValueError: If provider and model are not provided when factory doesn't exist
        """
        if self._factory is None:
            if provider is None or model is None:
                raise ValueError(
                    "provider and model are required when factory is not provided"
                )
            from victor.agent.orchestrator_factory import OrchestratorFactory

            self._factory = OrchestratorFactory(
                settings=self.settings,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                provider_name=provider_name,
                profile_name=profile_name,
                tool_selection=tool_selection,
                thinking=thinking,
            )
            self._logger.debug(
                f"Created OrchestratorFactory with provider={provider}, model={model}"
            )

        return self._factory

    def _register_components(self, components: Dict[str, Any]) -> None:
        """Register all non-None components from a dictionary.

        This method implements the component registration pattern that was
        duplicated across all three builders.

        Args:
            components: Dictionary of component name to component instance
        """
        registered_count = 0
        for name, component in components.items():
            if component is not None:
                self.register_component(name, component)
                registered_count += 1

        self._logger.debug(
            f"Registered {registered_count} components: "
            f"{', '.join(name for name, comp in components.items() if comp is not None)}"
        )

    @property
    def factory(self) -> Optional["OrchestratorFactory"]:
        """Get the factory instance without creating it.

        Returns:
            The factory instance if it exists, None otherwise
        """
        return self._factory
