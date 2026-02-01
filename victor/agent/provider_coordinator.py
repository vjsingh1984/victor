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

"""Provider coordination with rate limiting and health management.

This module provides a high-level coordinator that wraps ProviderManager,
adding rate limiting, retry logic, and coordination capabilities for
robust LLM interactions.

Design Pattern: Decorator/Wrapper
=================================
ProviderCoordinator wraps ProviderManager, adding:
- Rate limit handling with exponential backoff
- Post-switch hooks for coordination
- Health monitoring coordination
- Retry logic for streaming operations

Usage:
    from victor.agent.provider_coordinator import ProviderCoordinator

    coordinator = ProviderCoordinator(
        provider_manager=manager,
        config=ProviderCoordinatorConfig(max_rate_limit_retries=5),
    )

    # Stream with automatic rate limit retry
    async for chunk in coordinator.stream_with_rate_limit_retry(stream_func):
        process(chunk)
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    TypeVar,
)
from collections.abc import AsyncIterator, Callable

from victor.agent.provider_manager import ProviderManager, ProviderState
from victor.agent.tool_calling import ToolCallingCapabilities
from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    """High-level coordinator for provider management with rate limiting.

    Wraps ProviderManager, adding:
    - Rate limit handling with exponential backoff
    - Post-switch hooks for coordination
    - Health monitoring coordination
    - Retry logic for streaming operations

    Features:
    - Automatic retry on rate limits with backoff
    - Callback hooks for provider/model switches
    - Health status aggregation
    - Thread-safe operations
    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        config: Optional[ProviderCoordinatorConfig] = None,
        tool_adapter: Optional[Any] = None,
        capabilities: Optional[ToolCallingCapabilities] = None,
    ):
        """Initialize the provider coordinator.

        Args:
            provider_manager: The underlying ProviderManager instance
            config: Optional configuration for rate limiting and health
            tool_adapter: Optional tool calling adapter override
            capabilities: Optional tool calling capabilities override
        """
        self._manager = provider_manager
        self.config = config or ProviderCoordinatorConfig()

        # Override tool adapter/capabilities if provided
        self._tool_adapter_override = tool_adapter
        self._capabilities_override = capabilities

        # Post-switch hooks
        self._post_switch_hooks: list[Callable[[ProviderState], None]] = []

        # Health monitoring state
        self._health_task: Optional[asyncio.Task[None]] = None
        self._is_monitoring: bool = False

        # Rate limit tracking
        self._rate_limit_count: int = 0
        self._last_rate_limit_time: Optional[float] = None

        logger.debug(
            f"ProviderCoordinator initialized: "
            f"provider={self._manager.provider_name}, "
            f"max_retries={self.config.max_rate_limit_retries}"
        )

    @property
    def provider(self) -> Optional[BaseProvider]:
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
        """Get current tool calling adapter.

        Returns override if set, otherwise from manager.
        """
        if self._tool_adapter_override is not None:
            return self._tool_adapter_override
        return self._manager.tool_adapter

    @property
    def capabilities(self) -> Optional[ToolCallingCapabilities]:
        """Get current tool calling capabilities.

        Returns override if set, otherwise from manager.
        """
        if self._capabilities_override is not None:
            return self._capabilities_override
        return self._manager.capabilities

    @property
    def switch_count(self) -> int:
        """Get the number of provider/model switches."""
        return self._manager.switch_count

    def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Switch to a different provider synchronously.

        Note: This is a synchronous wrapper. For async operations,
        use the underlying provider_manager directly.

        Args:
            provider_name: Name of the provider
            model: Optional model name (uses current if not provided)
            **kwargs: Additional provider arguments

        Returns:
            True if switch was successful
        """
        # Run async switch in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Cannot run sync in async context - caller should use async API
                logger.warning(
                    "switch_provider called synchronously in async context. "
                    "Consider using async API."
                )
                return False
            else:
                result = loop.run_until_complete(
                    self._manager.switch_provider(provider_name, model, **kwargs)
                )
        except RuntimeError:
            # No event loop, create one
            result = asyncio.run(self._manager.switch_provider(provider_name, model, **kwargs))

        if isinstance(result, bool):
            if result:
                self._notify_post_switch_hooks()
            return result
        else:
            # Handle SwitchResult case
            self._notify_post_switch_hooks()
            return bool(result)

    def switch_model(self, model: str) -> bool:
        """Switch to a different model synchronously.

        Note: This is a synchronous wrapper. For async operations,
        use the underlying provider_manager directly.

        Args:
            model: New model name

        Returns:
            True if switch was successful
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning(
                    "switch_model called synchronously in async context. "
                    "Consider using async API."
                )
                return False
            else:
                result = loop.run_until_complete(self._manager.switch_model(model))
        except RuntimeError:
            result = asyncio.run(self._manager.switch_model(model))

        if result:
            self._notify_post_switch_hooks()

        return result

    def get_current_info(self) -> dict[str, Any]:
        """Get information about current provider and model.

        Returns:
            Dictionary with provider/model info and capabilities
        """
        info = self._manager.get_info()

        # Add coordinator-specific info
        info["rate_limit_count"] = self._rate_limit_count
        info["is_health_monitoring"] = self._is_monitoring
        info["switch_count"] = self.switch_count

        return info

    async def get_health(self) -> dict[str, Any]:
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

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._is_monitoring:
            return

        await self._manager.start_health_monitoring()
        self._is_monitoring = True
        logger.info("ProviderCoordinator: Started health monitoring")

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        if not self._is_monitoring:
            return

        await self._manager.stop_health_monitoring()
        self._is_monitoring = False
        logger.info("ProviderCoordinator: Stopped health monitoring")

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

    async def stream_with_rate_limit_retry(
        self,
        stream_func: Callable[..., AsyncIterator[T]],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[T]:
        """Wrap a streaming function with rate limit retry logic.

        Automatically retries on rate limit errors with exponential backoff.

        Args:
            stream_func: Async generator function to wrap
            *args: Positional arguments for stream_func
            **kwargs: Keyword arguments for stream_func

        Yields:
            Items from the stream function

        Raises:
            Exception: If max retries exceeded or non-rate-limit error
        """
        retries = 0
        backoff_multiplier = 1.0

        while retries <= self.config.max_rate_limit_retries:
            try:
                async for item in stream_func(*args, **kwargs):
                    yield item
                # Success - reset retry count
                return

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "rate limit" in error_str
                    or "rate_limit" in error_str
                    or "429" in error_str
                    or "too many requests" in error_str
                    or "quota exceeded" in error_str
                )

                if not is_rate_limit:
                    # Not a rate limit error, re-raise
                    raise

                retries += 1
                self._rate_limit_count += 1

                if retries > self.config.max_rate_limit_retries:
                    logger.error(
                        f"Max rate limit retries ({self.config.max_rate_limit_retries}) "
                        f"exceeded for {self.provider_name}"
                    )
                    raise

                # Calculate wait time with exponential backoff
                base_wait = self.get_rate_limit_wait_time(e)
                wait_time = min(
                    base_wait * backoff_multiplier,
                    self.config.max_rate_limit_wait,
                )

                logger.warning(
                    f"Rate limit hit for {self.provider_name}:{self.model}. "
                    f"Retry {retries}/{self.config.max_rate_limit_retries} "
                    f"after {wait_time:.1f}s"
                )

                await asyncio.sleep(wait_time)
                backoff_multiplier *= 1.5  # Exponential backoff

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

    async def close(self) -> None:
        """Close coordinator and cleanup resources."""
        await self.stop_health_monitoring()
        await self._manager.close()
        logger.debug("ProviderCoordinator closed")


def create_provider_coordinator(
    provider_manager: ProviderManager,
    config: Optional[ProviderCoordinatorConfig] = None,
    tool_adapter: Optional[Any] = None,
    capabilities: Optional[ToolCallingCapabilities] = None,
) -> ProviderCoordinator:
    """Factory function to create a ProviderCoordinator.

    Args:
        provider_manager: The underlying ProviderManager instance
        config: Optional configuration for rate limiting and health
        tool_adapter: Optional tool calling adapter override
        capabilities: Optional tool calling capabilities override

    Returns:
        Configured ProviderCoordinator instance
    """
    return ProviderCoordinator(
        provider_manager=provider_manager,
        config=config,
        tool_adapter=tool_adapter,
        capabilities=capabilities,
    )
