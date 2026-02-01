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

"""Global capability registry for vertical capability management.

This module provides a centralized registry for managing vertical capability providers,
enabling runtime capability discovery and cross-vertical capability queries.

Key Features:
- Singleton registry for all vertical providers
- Thread-safe provider registration and lookup
- Cross-vertical capability queries
- Capability discovery by type
- Provider lifecycle management

Example:
    # Register provider
    registry = CapabilityRegistry.get_instance()
    registry.register_provider("coding", CodingCapabilityProvider())

    # Get provider
    provider = registry.get_provider("coding")
    provider.apply_git_safety(orchestrator)

    # Cross-vertical queries
    tools = registry.list_capabilities_by_type(CapabilityType.TOOL)
    all_tools = registry.list_all_capabilities(CapabilityType.TOOL)

Design Pattern: Registry + Singleton
- Registry: Central repository for capability providers
- Singleton: Single global instance for consistent access
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional, TYPE_CHECKING
from collections.abc import Callable

from victor.framework.capabilities.base_vertical_capability_provider import (
    BaseVerticalCapabilityProvider,
    CapabilityDefinition,
)
from victor.framework.protocols import CapabilityType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CapabilityRegistry:
    """Global registry for vertical capability providers.

    This class provides a centralized repository for managing capability providers
    across all verticals, enabling runtime discovery and cross-vertical queries.

    Thread Safety:
        All public methods are thread-safe through reentrant locks.

    Example:
        # Get singleton instance
        registry = CapabilityRegistry.get_instance()

        # Register providers
        registry.register_provider("coding", CodingCapabilityProvider())
        registry.register_provider("research", ResearchCapabilityProvider())

        # Get specific provider
        coding_provider = registry.get_provider("coding")
        coding_provider.apply_git_safety(orchestrator)

        # List capabilities
        all_tools = registry.list_all_capabilities(CapabilityType.TOOL)
        coding_tools = registry.list_capabilities("coding", CapabilityType.TOOL)

        # Get specific capability
        cap = registry.get_capability("coding", "git_safety")
        if cap:
            print(f"Found: {cap.description}")
    """

    _instance: Optional["CapabilityRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the capability registry.

        Note: This class should not be instantiated directly.
        Use get_instance() to get the singleton.
        """
        if CapabilityRegistry._instance is not None:
            raise RuntimeError("Use get_instance() to get the CapabilityRegistry singleton")

        self._providers: dict[str, BaseVerticalCapabilityProvider] = {}
        self._provider_lock: threading.Lock = threading.Lock()
        logger.debug("CapabilityRegistry initialized")

    @classmethod
    def get_instance(cls) -> "CapabilityRegistry":
        """Get the singleton registry instance.

        Returns:
            CapabilityRegistry singleton instance

        Example:
            registry = CapabilityRegistry.get_instance()
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for testing).

        Warning: This should only be used in tests to reset state.
        """
        with cls._lock:
            cls._instance = None

    def register_provider(self, vertical: str, provider: BaseVerticalCapabilityProvider) -> None:
        """Register a vertical's capability provider.

        Args:
            vertical: Vertical name (e.g., "coding", "research")
            provider: Capability provider instance

        Raises:
            ValueError: If provider is not a BaseVerticalCapabilityProvider

        Example:
            registry = CapabilityRegistry.get_instance()
            provider = CodingCapabilityProvider()
            registry.register_provider("coding", provider)
        """
        if not isinstance(provider, BaseVerticalCapabilityProvider):
            raise ValueError(
                f"Provider must be BaseVerticalCapabilityProvider, got {type(provider)}"
            )

        with self._provider_lock:
            if vertical in self._providers:
                logger.warning(f"Replacing existing provider for vertical '{vertical}'")
            self._providers[vertical] = provider
            logger.info(f"Registered capability provider for vertical '{vertical}'")

    def get_provider(self, vertical: str) -> Optional[BaseVerticalCapabilityProvider]:
        """Get capability provider for a vertical.

        Args:
            vertical: Vertical name

        Returns:
            Capability provider or None if not registered

        Example:
            provider = registry.get_provider("coding")
            if provider:
                provider.apply_git_safety(orchestrator)
        """
        with self._provider_lock:
            return self._providers.get(vertical)

    def has_provider(self, vertical: str) -> bool:
        """Check if a vertical has a registered provider.

        Args:
            vertical: Vertical name

        Returns:
            True if provider is registered
        """
        with self._provider_lock:
            return vertical in self._providers

    def list_providers(self) -> list[str]:
        """List all registered vertical names.

        Returns:
            List of vertical names

        Example:
            verticals = registry.list_providers()
            print(f"Registered verticals: {verticals}")
        """
        with self._provider_lock:
            return list(self._providers.keys())

    def unregister_provider(self, vertical: str) -> bool:
        """Unregister a vertical's capability provider.

        Args:
            vertical: Vertical name

        Returns:
            True if provider was unregistered, False if not found

        Example:
            success = registry.unregister_provider("coding")
        """
        with self._provider_lock:
            if vertical in self._providers:
                del self._providers[vertical]
                logger.info(f"Unregistered capability provider for vertical '{vertical}'")
                return True
            return False

    def get_capability(self, vertical: str, capability_name: str) -> Optional[Callable[..., None]]:
        """Get specific capability function from a vertical.

        Args:
            vertical: Vertical name
            capability_name: Capability name

        Returns:
            Capability function or None if not found

        Example:
            cap = registry.get_capability("coding", "git_safety")
            if cap:
                cap(orchestrator, block_force_push=True)
        """
        provider = self.get_provider(vertical)
        if not provider:
            logger.warning(f"No provider registered for vertical '{vertical}'")
            return None

        return provider.get_capability(capability_name)

    def get_capability_definition(
        self, vertical: str, capability_name: str
    ) -> Optional[CapabilityDefinition]:
        """Get capability definition from a vertical.

        Args:
            vertical: Vertical name
            capability_name: Capability name

        Returns:
            CapabilityDefinition or None if not found

        Example:
            definition = registry.get_capability_definition("coding", "git_safety")
            if definition:
                print(f"Type: {definition.type}")
                print(f"Default config: {definition.default_config}")
        """
        provider = self.get_provider(vertical)
        if not provider:
            return None

        return provider.get_capability_definition(capability_name)

    def list_capabilities(
        self, vertical: str, capability_type: Optional[CapabilityType] = None
    ) -> list[str]:
        """List capabilities for a vertical, optionally filtered by type.

        Args:
            vertical: Vertical name
            capability_type: Filter by type (optional)

        Returns:
            List of capability names

        Example:
            # All coding capabilities
            all_caps = registry.list_capabilities("coding")

            # Only TOOL capabilities
            tools = registry.list_capabilities("coding", CapabilityType.TOOL)
        """
        provider = self.get_provider(vertical)
        if not provider:
            logger.warning(f"No provider registered for vertical '{vertical}'")
            return []

        return provider.list_capabilities(capability_type)

    def list_all_capabilities(
        self, capability_type: Optional[CapabilityType] = None
    ) -> dict[str, list[str]]:
        """List capabilities across all verticals.

        Args:
            capability_type: Filter by type (optional)

        Returns:
            Dictionary mapping vertical names to capability lists

        Example:
            # All capabilities across all verticals
            all_caps = registry.list_all_capabilities()

            # Only TOOL capabilities
            tools = registry.list_all_capabilities(CapabilityType.TOOL)
            for vertical, caps in tools.items():
                print(f"{vertical}: {caps}")
        """
        result: dict[str, list[str]] = {}

        with self._provider_lock:
            verticals = list(self._providers.keys())

        for vertical in verticals:
            caps = self.list_capabilities(vertical, capability_type)
            if caps:
                result[vertical] = caps

        return result

    def apply_capability(
        self, vertical: str, orchestrator: Any, capability_name: str, **kwargs: Any
    ) -> bool:
        """Apply a capability from a vertical to an orchestrator.

        Args:
            vertical: Vertical name
            orchestrator: Target orchestrator
            capability_name: Capability name
            **kwargs: Configuration options

        Returns:
            True if capability was applied successfully

        Example:
            success = registry.apply_capability(
                "coding",
                orchestrator,
                "git_safety",
                block_force_push=True,
                block_main_push=True,
            )
        """
        provider = self.get_provider(vertical)
        if not provider:
            logger.warning(f"No provider registered for vertical '{vertical}'")
            return False

        try:
            provider.apply_capability(orchestrator, capability_name, **kwargs)
            return True
        except Exception as e:
            logger.error(
                f"Failed to apply capability '{capability_name}' from vertical '{vertical}': {e}",
                exc_info=True,
            )
            return False

    def get_capability_config(
        self, vertical: str, orchestrator: Any, capability_name: str
    ) -> Optional[dict[str, Any]]:
        """Get current configuration for a capability.

        Args:
            vertical: Vertical name
            orchestrator: Target orchestrator
            capability_name: Capability name

        Returns:
            Configuration dict or None if not found

        Example:
            config = registry.get_capability_config("coding", orchestrator, "code_style")
            if config:
                print(f"Formatter: {config.get('formatter')}")
        """
        provider = self.get_provider(vertical)
        if not provider:
            return None

        return provider.get_capability_config(orchestrator, capability_name)

    def get_default_config(self, vertical: str, capability_name: str) -> Optional[dict[str, Any]]:
        """Get default configuration for a capability.

        Args:
            vertical: Vertical name
            capability_name: Capability name

        Returns:
            Default configuration dict or None if not found

        Example:
            defaults = registry.get_default_config("coding", "code_style")
            if defaults:
                print(f"Default formatter: {defaults.get('formatter')}")
        """
        provider = self.get_provider(vertical)
        if not provider:
            return None

        try:
            return provider.get_default_config(capability_name)
        except ValueError:
            return None

    def get_all_capability_configs(self, vertical: str) -> dict[str, Any]:
        """Get all default configurations for a vertical.

        Args:
            vertical: Vertical name

        Returns:
            Dict with all capability configurations

        Example:
            configs = registry.get_all_capability_configs("coding")
            for cap_name, config in configs.items():
                print(f"{cap_name}: {config}")
        """
        provider = self.get_provider(vertical)
        if not provider:
            logger.warning(f"No provider registered for vertical '{vertical}'")
            return {}

        return provider.generate_capability_configs()

    def apply_all_capabilities(self, vertical: str, orchestrator: Any, **kwargs: Any) -> bool:
        """Apply all capabilities for a vertical.

        Args:
            vertical: Vertical name
            orchestrator: Target orchestrator
            **kwargs: Shared options passed to all capabilities

        Returns:
            True if all capabilities applied successfully

        Example:
            success = registry.apply_all_capabilities("coding", orchestrator)
        """
        provider = self.get_provider(vertical)
        if not provider:
            logger.warning(f"No provider registered for vertical '{vertical}'")
            return False

        try:
            provider.apply_all(orchestrator, **kwargs)
            return True
        except Exception as e:
            logger.error(
                f"Failed to apply all capabilities for vertical '{vertical}': {e}",
                exc_info=True,
            )
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dict with registry stats

        Example:
            stats = registry.get_stats()
            print(f"Registered verticals: {stats['vertical_count']}")
            print(f"Total capabilities: {stats['total_capabilities']}")
        """
        with self._provider_lock:
            vertical_count = len(self._providers)
            providers = list(self._providers.values())

        total_capabilities = sum(len(p.list_capabilities()) for p in providers)

        capability_counts: dict[str, int] = {}
        for provider in providers:
            vertical_name = provider._vertical_name
            capability_counts[vertical_name] = len(provider.list_capabilities())

        return {
            "vertical_count": vertical_count,
            "total_capabilities": total_capabilities,
            "capability_counts": capability_counts,
            "verticals": self.list_providers(),
        }


__all__ = ["CapabilityRegistry"]
