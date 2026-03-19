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

"""Provider service implementation.

Extracts provider management from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Provider initialization and configuration
- Provider switching with validation
- Provider health checks
- Provider capability discovery
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ProviderInfoImpl:
    """Implementation of provider information."""

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        api_key_configured: bool,
        base_url: Optional[str],
        supports_streaming: bool,
        supports_tool_calling: bool,
        max_tokens: int,
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        self.api_key_configured = api_key_configured
        self.base_url = base_url
        self.supports_streaming = supports_streaming
        self.supports_tool_calling = supports_tool_calling
        self.max_tokens = max_tokens


class ProviderService:
    """Service for provider management.

    Extracted from AgentOrchestrator to handle:
    - Provider initialization and configuration
    - Provider switching with validation
    - Provider health checks
    - Provider capability discovery

    This service follows SOLID principles:
    - SRP: Only handles provider operations
    - OCP: Extensible through registry
    - LSP: Implements ProviderServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on abstractions

    Example:
        service = ProviderService(registry=registry)
        await service.switch_provider('anthropic', 'claude-sonnet-4-5')
    """

    def __init__(
        self,
        registry: Any,
        health_checker: Optional[Any] = None,
    ):
        """Initialize the provider service.

        Args:
            registry: Provider registry
            health_checker: Optional health checker component
        """
        self._registry = registry
        self._health_checker = health_checker
        self._current_provider: Optional["BaseProvider"] = None
        self._current_info: Optional[ProviderInfoImpl] = None
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

    async def switch_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        validate: bool = True,
    ) -> None:
        """Switch to a different provider.

        Args:
            provider: Provider name
            model: Optional model name
            validate: If True, validate provider before switching
        """
        self._logger.info(f"Switching to provider: {provider}")

        # Get provider from registry
        new_provider = self._registry.get_provider(provider)
        if new_provider is None:
            raise ValueError(f"Provider not found: {provider}")

        # Set model if specified
        if model:
            new_provider.set_model(model)

        # Validate if requested
        if validate:
            if self._health_checker:
                is_healthy = await self._health_checker.check(new_provider)
                if not is_healthy:
                    raise ValueError(f"Provider health check failed: {provider}")

        # Update current provider
        self._current_provider = new_provider
        self._current_info = self._create_provider_info(new_provider)

        self._logger.info(f"Switched to provider: {provider}")

    def get_current_provider_info(self) -> ProviderInfoImpl:
        """Get current provider information.

        Returns:
            ProviderInfo with current provider details
        """
        if self._current_info is None:
            # Create info for current provider
            if self._current_provider:
                self._current_info = self._create_provider_info(self._current_provider)
            else:
                # Return default info
                self._current_info = ProviderInfoImpl(
                    provider_name="none",
                    model_name="none",
                    api_key_configured=False,
                    base_url=None,
                    supports_streaming=False,
                    supports_tool_calling=False,
                    max_tokens=0,
                )

        return self._current_info

    async def check_provider_health(
        self,
        provider: Optional[str] = None,
    ) -> bool:
        """Check if a provider is healthy.

        Args:
            provider: Provider name to check, or None for current

        Returns:
            True if provider is healthy, False otherwise
        """
        if provider:
            target_provider = self._registry.get_provider(provider)
        else:
            target_provider = self._current_provider

        if target_provider is None:
            return False

        if self._health_checker:
            return await self._health_checker.check(target_provider)

        # Default health check: try to get model
        try:
            return target_provider.get_model() is not None
        except Exception:
            return False

    def get_available_providers(self) -> List[str]:
        """Get list of available providers.

        Returns:
            List of provider names
        """
        return self._registry.list_providers()

    async def get_provider_capabilities(
        self,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get provider capabilities.

        Args:
            provider: Provider name, or None for current

        Returns:
            Dictionary with capability information
        """
        if provider:
            target_provider = self._registry.get_provider(provider)
        else:
            target_provider = self._current_provider

        if target_provider is None:
            return {}

        # Get capabilities from provider
        return {
            "streaming": getattr(target_provider, "supports_streaming", True),
            "tools": getattr(target_provider, "supports_tools", True),
            "max_tokens": getattr(target_provider, "max_tokens", 100000),
        }

    def get_current_provider(self) -> "BaseProvider":
        """Get the current provider instance.

        Returns:
            Current provider instance

        Raises:
            ValueError: If no provider is configured
        """
        if self._current_provider is None:
            raise ValueError("No provider is currently configured")

        return self._current_provider

    async def test_provider(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> bool:
        """Test a provider with a simple request.

        Args:
            provider: Provider name to test
            model: Optional model name

        Returns:
            True if test succeeded, False otherwise
        """
        try:
            target_provider = self._registry.get_provider(provider)
            if target_provider is None:
                return False

            if model:
                target_provider.set_model(model)

            # Simple health check
            return await self.check_provider_health(provider)

        except Exception:
            return False

    def is_healthy(self) -> bool:
        """Check if the provider service is healthy.

        Returns:
            True if the service is healthy
        """
        return self._current_provider is not None

    def _create_provider_info(self, provider: "BaseProvider") -> ProviderInfoImpl:
        """Create provider info from provider instance.

        Args:
            provider: Provider instance

        Returns:
            ProviderInfo with provider details
        """
        return ProviderInfoImpl(
            provider_name=getattr(provider, "name", "unknown"),
            model_name=getattr(provider, "model", "unknown"),
            api_key_configured=getattr(provider, "api_key", None) is not None,
            base_url=getattr(provider, "base_url", None),
            supports_streaming=getattr(provider, "supports_streaming", True),
            supports_tool_calling=getattr(provider, "supports_tools", True),
            max_tokens=getattr(provider, "max_tokens", 100000),
        )
