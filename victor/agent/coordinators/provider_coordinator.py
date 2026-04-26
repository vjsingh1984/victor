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

"""Deprecated coordinator-path shim for provider coordination.

The canonical provider coordinator now lives in
``victor.agent.provider.coordinator``. This module preserves the older
coordinator-path constructor and method names for compatibility while routing
behavior through the canonical implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.agent.provider.coordinator import (
    ProviderCoordinator as CanonicalProviderCoordinator,
)
from victor.agent.provider.coordinator import ProviderCoordinatorConfig, RateLimitInfo

if TYPE_CHECKING:
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ProviderCoordinator(CanonicalProviderCoordinator):
    """[DEPRECATED] Coordinator-path compatibility adapter.

    This preserves the older coordinator-path API that accepted ``settings``
    and exposed async ``switch_provider`` / ``switch_model`` methods, while the
    canonical behavior now lives in ``victor.agent.provider.coordinator``.
    """

    def __init__(
        self,
        provider_manager: Any,
        settings: "Settings",
        config: Optional[ProviderCoordinatorConfig] = None,
    ):
        super().__init__(provider_manager=provider_manager, config=config)
        self.settings = settings
        self._capability_cache: Dict[tuple[str, str], Any] = {}

    async def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None,
        **provider_kwargs: Any,
    ) -> bool:
        """Switch to a different provider using the legacy async signature."""
        try:
            if not provider_kwargs:
                provider_kwargs = self.settings.get_provider_settings(provider_name)

            return await super().switch_provider_async(
                provider_name=provider_name,
                model=model or self.model,
                **provider_kwargs,
            )
        except Exception as exc:
            logger.error(f"Failed to switch provider to {provider_name}: {exc}")
            logger.error(
                "Suggestion: Verify the provider is available and configured. "
                "Run 'victor doctor' to check provider status."
            )
            return False

    async def switch_model(self, model: str) -> bool:
        """Switch to a different model using the legacy async signature."""
        try:
            return await super().switch_model_async(model)
        except Exception as exc:
            logger.error(f"Failed to switch model to {model}: {exc}")
            logger.error(
                "Suggestion: Verify the model is supported by the current provider. "
                f"Run 'victor models --provider {self.provider_name}' to list available models."
            )
            return False

    def get_current_provider_info(self) -> Dict[str, Any]:
        """Return current provider information using the legacy method name."""
        return super().get_current_info()

    async def get_provider_health(self) -> Dict[str, Any]:
        """Return current provider health using the legacy method name."""
        return await super().get_health()

    async def get_healthy_providers(self) -> List[str]:
        """Return healthy providers via the underlying manager."""
        return await self._manager.get_healthy_providers()

    def track_rate_limit(self, error: Exception) -> None:
        """Track a rate limit event for compatibility callers."""
        import time

        self._rate_limit_count += 1
        self._last_rate_limit_time = time.time()

    async def discover_capabilities(
        self,
        provider: Optional["BaseProvider"] = None,
        model: Optional[str] = None,
    ) -> Optional[Any]:
        """Discover provider capabilities asynchronously and cache the result."""
        from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities

        provider_name = provider.name if provider else self.provider_name
        model_name = model or self.model
        cache_key = (provider_name, model_name)

        if cache_key in self._capability_cache:
            return self._capability_cache[cache_key]

        provider_to_use = provider or self.provider
        if not provider_to_use:
            return None

        try:
            discovery = await provider_to_use.discover_capabilities(model_name)
        except Exception as exc:
            logger.warning(
                f"Capability discovery failed for {provider_name}:{model_name} ({exc}); "
                "falling back to config."
            )
            from victor.config.config_loaders import get_provider_limits

            limits = get_provider_limits(provider_name, model_name)
            discovery = ProviderRuntimeCapabilities(
                provider=provider_name,
                model=model_name,
                context_window=limits.context_window,
                supports_tools=provider_to_use.supports_tools(),
                supports_streaming=provider_to_use.supports_streaming(),
                source="config",
            )

        self._capability_cache[cache_key] = discovery
        return discovery


def create_provider_coordinator(
    provider_manager: Any,
    settings: "Settings",
    config: Optional[ProviderCoordinatorConfig] = None,
) -> ProviderCoordinator:
    """Create the coordinator-path compatibility adapter."""
    return ProviderCoordinator(
        provider_manager=provider_manager,
        settings=settings,
        config=config,
    )


__all__ = [
    "ProviderCoordinator",
    "ProviderCoordinatorConfig",
    "RateLimitInfo",
    "create_provider_coordinator",
]
