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

"""Provider Coordinator - Coordinates provider initialization, switching, and health monitoring.

This module extracts provider-related coordination logic from AgentOrchestrator,
providing a focused interface for:
- Provider initialization and configuration
- Provider and model switching capabilities
- Provider health monitoring
- Rate limiting coordination

Design Philosophy:
- Single Responsibility: Coordinates all provider-related operations
- Composable: Works with existing ProviderManager, ProviderSwitcher
- Observable: Provides health status and metrics
- Backward Compatible: Maintains API compatibility with orchestrator

Usage:
    coordinator = ProviderCoordinator(
        provider_manager=manager,
        settings=settings,
    )

    # Switch providers
    await coordinator.switch_provider("anthropic", "claude-sonnet-4-20250514")

    # Get health status
    health = await coordinator.get_provider_health()

    # Monitor provider health
    await coordinator.start_health_monitoring()
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

if TYPE_CHECKING:
    from victor.agent.provider_manager import ProviderManager, ProviderState
    from victor.agent.tool_calling import ToolCallingCapabilities
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


@dataclass
class ProviderCoordinatorConfig:
    """Configuration for ProviderCoordinator.

    Attributes:
        max_rate_limit_retries: Maximum number of retries on rate limit errors
        default_rate_limit_wait: Default wait time in seconds when rate limit hit
        max_rate_limit_wait: Maximum wait time in seconds for rate limit backoff
        enable_health_monitoring: Enable background health monitoring
        health_check_interval: Interval between health checks in seconds
    """

    max_rate_limit_retries: int = 3
    default_rate_limit_wait: float = 60.0
    max_rate_limit_wait: float = 300.0
    enable_health_monitoring: bool = True
    health_check_interval: float = 60.0


@dataclass
class RateLimitInfo:
    """Information about a rate limit error.

    Attributes:
        wait_seconds: Suggested wait time in seconds
        retry_after: Retry-After header value if available
        message: Error message
        error_type: Type of rate limit error
    """

    wait_seconds: float
    retry_after: Optional[float] = None
    message: str = ""
    error_type: str = "rate_limit"


class ProviderCoordinator:
    """Coordinates provider initialization, switching, and health monitoring.

    This class consolidates provider-related operations that were spread across
    the orchestrator, providing a unified interface for:

    1. Provider Management: Switch providers and models with post-switch hooks
    2. Health Monitoring: Track provider health and provide status
    3. Rate Limiting: Calculate wait times for rate limit retries
    4. Capability Discovery: Discover and cache provider capabilities

    Example:
        coordinator = ProviderCoordinator(
            provider_manager=manager,
            settings=settings,
            config=ProviderCoordinatorConfig(max_rate_limit_retries=5),
        )

        # Switch providers
        success = await coordinator.switch_provider(
            "anthropic", "claude-sonnet-4-20250514"
        )

        # Get health status
        health = await coordinator.get_provider_health()
    """

    def __init__(
        self,
        provider_manager: "ProviderManager",
        settings: "Settings",
        config: Optional[ProviderCoordinatorConfig] = None,
    ):
        """Initialize the ProviderCoordinator.

        Args:
            provider_manager: The underlying ProviderManager instance
            settings: Application settings
            config: Optional configuration for rate limiting and health
        """
        self._manager = provider_manager
        self.settings = settings
        self.config = config or ProviderCoordinatorConfig()

        # Post-switch hooks
        self._post_switch_hooks: List[Callable[[ProviderState], None]] = []

        # Rate limit tracking
        self._rate_limit_count: int = 0
        self._last_rate_limit_time: Optional[float] = None

        # Runtime capability cache (per provider/model)
        self._capability_cache: Dict[tuple[str, str], Any] = {}

        logger.debug(
            f"ProviderCoordinator initialized: "
            f"provider={self._manager.provider_name}, "
            f"max_retries={self.config.max_rate_limit_retries}"
        )

    # =====================================================================
    # Properties
    # =====================================================================

    @property
    def provider(self) -> Optional["BaseProvider"]:
        """Get current provider instance."""
        return self._manager.provider

    @property
    def model(self) -> str:
        """Get current model name."""
        return self._manager.model

    @property
    def provider_name(self) -> str:
        """Get current provider name."""
        return self._manager.provider_name

    @property
    def tool_adapter(self) -> Optional[Any]:
        """Get current tool calling adapter."""
        return self._manager.tool_adapter

    @property
    def capabilities(self) -> Optional["ToolCallingCapabilities"]:
        """Get current tool calling capabilities."""
        return self._manager.capabilities

    @property
    def switch_count(self) -> int:
        """Get the number of provider/model switches."""
        return self._manager.switch_count

    # =====================================================================
    # Provider/Model Switching
    # =====================================================================

    async def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None,
        **provider_kwargs: Any,
    ) -> bool:
        """Switch to a different provider.

        Delegates to ProviderManager while handling post-switch hooks.

        Args:
            provider_name: Name of the provider
            model: Optional model name (uses current if not provided)
            **provider_kwargs: Additional provider arguments

        Returns:
            True if switch was successful
        """
        from victor.agent.model_switcher import SwitchReason

        try:
            # Get provider settings from settings if not provided
            if not provider_kwargs:
                provider_kwargs = self.settings.get_provider_settings(provider_name)

            # Determine model to use
            new_model = model or self.model

            # Store old state for analytics
            old_provider_name = self.provider_name
            old_model = self.model

            # Delegate to ProviderManager
            result = await self._manager.switch_provider(
                provider_name=provider_name,
                model=new_model,
                reason=SwitchReason.USER_REQUEST,
                **provider_kwargs,
            )

            if result:
                # Notify post-switch hooks
                self._notify_post_switch_hooks()

                logger.info(
                    f"Switched provider: {old_provider_name}:{old_model} -> "
                    f"{self.provider_name}:{new_model}"
                )

            return result

        except Exception as e:
            logger.error(f"Failed to switch provider to {provider_name}: {e}")
            return False

    async def switch_model(self, model: str) -> bool:
        """Switch to a different model on the current provider.

        This is a lighter-weight switch than switch_provider() - it only
        updates the model and reinitializes the tool adapter.

        Args:
            model: New model name

        Returns:
            True if switch was successful
        """
        from victor.agent.model_switcher import SwitchReason

        try:
            old_model = self.model

            # Delegate to ProviderManager
            result = await self._manager.switch_model(
                model=model,
                reason=SwitchReason.USER_REQUEST,
            )

            if result:
                # Notify post-switch hooks
                self._notify_post_switch_hooks()

                logger.info(f"Switched model: {old_model} -> {model}")

            return result

        except Exception as e:
            logger.error(f"Failed to switch model to {model}: {e}")
            return False

    def get_current_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider and model.

        Returns:
            Dictionary with provider/model info and capabilities
        """
        info = self._manager.get_info()

        # Add coordinator-specific info
        info["rate_limit_count"] = self._rate_limit_count
        info["switch_count"] = self.switch_count

        return info

    # =====================================================================
    # Health Monitoring
    # =====================================================================

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        await self._manager.start_health_monitoring()
        logger.info("ProviderCoordinator: Started health monitoring")

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        await self._manager.stop_health_monitoring()
        logger.info("ProviderCoordinator: Stopped health monitoring")

    async def get_provider_health(self) -> Dict[str, Any]:
        """Get health status of current provider.

        Returns:
            Dictionary with health information
        """
        state = self._manager.get_current_state()
        if not state:
            return {
                "healthy": False,
                "provider": None,
                "model": None,
                "error": "No provider configured",
            }

        return {
            "healthy": state.is_healthy,
            "provider": state.provider_name,
            "model": state.model,
            "last_error": state.last_error,
            "switch_count": state.switch_count,
        }

    async def get_healthy_providers(self) -> List[str]:
        """Get list of healthy providers.

        Returns:
            List of healthy provider names
        """
        return await self._manager.get_healthy_providers()

    # =====================================================================
    # Rate Limiting
    # =====================================================================

    def get_rate_limit_wait_time(self, error: Exception) -> float:
        """Calculate wait time from a rate limit error.

        Parses common rate limit error formats to extract wait time.
        Falls back to default wait time if parsing fails.

        Args:
            error: The rate limit exception

        Returns:
            Wait time in seconds
        """
        # Check for retry_after attribute (e.g., ProviderRateLimitError)
        if hasattr(error, "retry_after") and error.retry_after is not None:
            try:
                return min(float(error.retry_after), self.config.max_rate_limit_wait)
            except (ValueError, TypeError):
                pass

        error_str = str(error)

        # Try to extract retry-after from common patterns
        patterns = [
            # "try again in X.XXs" or "try again in X seconds"
            r"try\s+again\s+in\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)?",
            # "retry after X seconds"
            r"retry\s+after\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)?",
            # "wait X seconds"
            r"wait\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)?",
            # "X seconds" at end
            r"(\d+(?:\.\d+)?)\s*(?:seconds?|s)\s*$",
            # Just a number (assume seconds)
            r"(\d+(?:\.\d+)?)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                try:
                    wait_time = float(match.group(1))
                    # Clamp to max wait time
                    return min(wait_time, self.config.max_rate_limit_wait)
                except ValueError:
                    continue

        # Check for Retry-After header in error message
        if hasattr(error, "response"):
            response = getattr(error, "response", None)
            if response and hasattr(response, "headers"):
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        return min(float(retry_after), self.config.max_rate_limit_wait)
                    except ValueError:
                        pass

        return self.config.default_rate_limit_wait

    def track_rate_limit(self, error: Exception) -> None:
        """Track a rate limit error.

        Args:
            error: The rate limit exception
        """
        import time

        self._rate_limit_count += 1
        self._last_rate_limit_time = time.time()

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limit statistics.

        Returns:
            Dict with rate limit count and last occurrence time
        """
        return {
            "rate_limit_count": self._rate_limit_count,
            "last_rate_limit_time": self._last_rate_limit_time,
        }

    # =====================================================================
    # Capability Discovery
    # =====================================================================

    async def discover_capabilities(
        self, provider: Optional["BaseProvider"] = None, model: Optional[str] = None
    ) -> Optional[Any]:
        """Discover provider capabilities asynchronously and cache result.

        Args:
            provider: Provider instance (uses current if not provided)
            model: Model name (uses current if not provided)

        Returns:
            ProviderRuntimeCapabilities or None
        """
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

    # =====================================================================
    # Post-Switch Hooks
    # =====================================================================

    def register_post_switch_hook(
        self,
        callback: Callable[[ProviderState], None],
    ) -> None:
        """Register a callback for after provider/model switches.

        Args:
            callback: Function called with new ProviderState after switch
        """
        self._post_switch_hooks.append(callback)

        # Also register with underlying manager for async switches
        self._manager.add_switch_callback(callback)

    def _notify_post_switch_hooks(self) -> None:
        """Notify all post-switch hooks of a switch."""
        state = self._manager.get_current_state()
        if not state:
            return

        for callback in self._post_switch_hooks:
            try:
                callback(state)
            except Exception as e:
                logger.warning(f"Post-switch hook error: {e}")

    # =====================================================================
    # Lifecycle
    # =====================================================================

    async def close(self) -> None:
        """Close coordinator and cleanup resources."""
        await self.stop_health_monitoring()
        await self._manager.close()
        logger.debug("ProviderCoordinator closed")


def create_provider_coordinator(
    provider_manager: "ProviderManager",
    settings: "Settings",
    config: Optional[ProviderCoordinatorConfig] = None,
) -> ProviderCoordinator:
    """Factory function to create a ProviderCoordinator.

    Args:
        provider_manager: The underlying ProviderManager instance
        settings: Application settings
        config: Optional configuration for rate limiting and health

    Returns:
        Configured ProviderCoordinator instance
    """
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
