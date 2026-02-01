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

"""Orchestrator pool for multi-provider workflows.

Manages multiple orchestrators for different provider profiles with
caching to avoid recreating orchestrators for the same profile.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from victor.config.settings import Settings, ProfileConfig
from victor.providers.registry import ProviderRegistry
from victor.agent.orchestrator_factory import create_orchestrator_factory

logger = logging.getLogger(__name__)


class OrchestratorPool:
    """Pool of orchestrators for multi-provider workflows.

    Responsibility (SRP):
    - Manage multiple orchestrators for different providers
    - Create orchestrators on-demand for unique profiles
    - Reuse orchestrators across workflow executions via caching
    - Provide orchestrator lifecycle management

    Non-responsibility:
    - Orchestrator creation (handled by OrchestratorFactory)
    - Profile resolution (handled by Settings.load_profiles)

    Example:
        pool = OrchestratorPool(settings)
        orchestrator = pool.get_orchestrator("my-profile")
        default_orchestrator = pool.get_default_orchestrator()
    """

    def __init__(self, settings: Settings, container: Optional[Any] = None):
        """Initialize the orchestrator pool.

        Args:
            settings: Application settings
            container: Optional DI container (for coordinator-based orchestrator)
        """
        self._settings = settings
        self._container = container
        self._orchestrators: dict[str, Any] = {}
        self._providers: dict[str, Any] = {}
        self._profiles: dict[str, ProfileConfig] = {}

    def get_orchestrator(self, profile: Optional[str] = None) -> Any:
        """Get orchestrator for a profile.

        Creates and caches orchestrators on-demand. If an orchestrator
        already exists for the profile, returns the cached instance.

        Args:
            profile: Provider profile name. If None, uses default profile.

        Returns:
            AgentOrchestrator instance

        Raises:
            ValueError: If profile is not found
            RuntimeError: If orchestrator creation fails
        """
        # Use default profile if none specified
        if profile is None:
            profile = "default"

        # Check cache first
        if profile in self._orchestrators:
            logger.debug(f"OrchestratorPool: Using cached orchestrator for profile '{profile}'")
            return self._orchestrators[profile]

        # Load profile configuration
        profile_config = self._load_profile_config(profile)
        if profile_config is None:
            raise ValueError(f"Profile '{profile}' not found in configuration")

        # Create provider for this profile
        provider = self._get_or_create_provider(profile, profile_config)

        # Create orchestrator factory
        factory = create_orchestrator_factory(
            settings=self._settings,
            provider=provider,
            model=profile_config.model_name,
        )

        # Create orchestrator from factory
        try:
            orchestrator = factory.create_orchestrator()
            self._orchestrators[profile] = orchestrator
            logger.info(
                f"OrchestratorPool: Created new orchestrator for profile '{profile}' "
                f"(provider={profile_config.provider}, model={profile_config.model_name})"
            )
            return orchestrator
        except Exception as e:
            logger.error(
                f"OrchestratorPool: Failed to create orchestrator for profile '{profile}': {e}"
            )
            raise RuntimeError(f"Failed to create orchestrator for profile '{profile}': {e}") from e

    def get_default_orchestrator(self) -> Any:
        """Get orchestrator for the default profile.

        Returns:
            AgentOrchestrator instance for default profile

        Raises:
            ValueError: If default profile is not found
            RuntimeError: If orchestrator creation fails
        """
        return self.get_orchestrator("default")

    def _load_profile_config(self, profile: str) -> Optional[ProfileConfig]:
        """Load profile configuration from settings.

        Args:
            profile: Profile name

        Returns:
            ProfileConfig or None if not found
        """
        # Check cache first
        if profile in self._profiles:
            return self._profiles[profile]

        # Load all profiles from settings
        profiles = self._settings.load_profiles()

        if profile not in profiles:
            logger.warning(f"OrchestratorPool: Profile '{profile}' not found")
            return None

        # Cache profile config
        profile_config = profiles[profile]
        self._profiles[profile] = profile_config
        return profile_config

    def _get_or_create_provider(self, profile: str, profile_config: ProfileConfig) -> Any:
        """Get or create provider for a profile.

        Args:
            profile: Profile name
            profile_config: Profile configuration

        Returns:
            Provider instance

        Raises:
            RuntimeError: If provider creation fails
        """
        # Check cache first
        if profile in self._providers:
            logger.debug(f"OrchestratorPool: Using cached provider for profile '{profile}'")
            return self._providers[profile]

        # Get provider settings
        provider_settings = self._settings.get_provider_settings(profile_config.provider)

        # Create provider
        try:
            provider = ProviderRegistry.create(
                profile_config.provider,
                **provider_settings,
            )
            self._providers[profile] = provider
            logger.info(
                f"OrchestratorPool: Created provider for profile '{profile}' "
                f"(provider={profile_config.provider})"
            )
            return provider
        except Exception as e:
            logger.error(
                f"OrchestratorPool: Failed to create provider for profile '{profile}': {e}"
            )
            raise RuntimeError(f"Failed to create provider for profile '{profile}': {e}") from e

    def clear_cache(self, profile: Optional[str] = None) -> None:
        """Clear cached orchestrators and providers.

        Args:
            profile: Specific profile to clear, or None to clear all
        """
        if profile:
            # Clear specific profile
            if profile in self._orchestrators:
                del self._orchestrators[profile]
            if profile in self._providers:
                del self._providers[profile]
            if profile in self._profiles:
                del self._profiles[profile]
            logger.debug(f"OrchestratorPool: Cleared cache for profile '{profile}'")
        else:
            # Clear all profiles
            self._orchestrators.clear()
            self._providers.clear()
            self._profiles.clear()
            logger.debug("OrchestratorPool: Cleared all caches")

    def get_cached_profiles(self) -> list[str]:
        """Get list of profiles with cached orchestrators.

        Returns:
            List of profile names
        """
        return list(self._orchestrators.keys())

    def shutdown(self) -> None:
        """Shutdown all orchestrators and providers.

        Cleans up resources for all cached instances.
        """
        import asyncio
        import inspect

        logger.info("OrchestratorPool: Shutting down...")

        # Shutdown orchestrators
        for profile, orchestrator in self._orchestrators.items():
            try:
                if hasattr(orchestrator, "close"):
                    close_method = orchestrator.close
                    if asyncio.iscoroutinefunction(close_method) or inspect.iscoroutinefunction(
                        close_method
                    ):
                        asyncio.run(close_method())
                    elif callable(close_method):
                        result = close_method()
                        if asyncio.iscoroutine(result):
                            asyncio.run(result)
            except Exception as e:
                logger.warning(
                    f"OrchestratorPool: Error closing orchestrator for profile '{profile}': {e}"
                )

        # Shutdown providers
        for profile, provider in self._providers.items():
            try:
                if hasattr(provider, "shutdown"):
                    shutdown_method = provider.shutdown
                    if asyncio.iscoroutinefunction(shutdown_method) or inspect.iscoroutinefunction(
                        shutdown_method
                    ):
                        asyncio.run(shutdown_method())
                    elif callable(shutdown_method):
                        result = shutdown_method()
                        if asyncio.iscoroutine(result):
                            asyncio.run(result)
            except Exception as e:
                logger.warning(
                    f"OrchestratorPool: Error shutting down provider for profile '{profile}': {e}"
                )

        # Clear caches
        self.clear_cache()
        logger.info("OrchestratorPool: Shutdown complete")


__all__ = ["OrchestratorPool"]
