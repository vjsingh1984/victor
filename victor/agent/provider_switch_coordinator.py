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

"""Provider Switch Coordinator - Coordinates provider/model switching operations.

This module extracts provider switching coordination from AgentOrchestrator:
- Switch validation and checks
- Health verification coordination
- Fallback handling
- Post-switch hook execution
- Error handling and retry logic

Design Principles:
- Single Responsibility: Coordinate switching workflow only
- Delegation: Use ProviderSwitcher for actual switching
- Composable: Works with existing health monitors and event emitters
- Observable: Support for hooks and callbacks

Usage:
    coordinator = ProviderSwitchCoordinator(
        provider_switcher=switcher,
        health_monitor=monitor,
    )

    # Switch provider
    success = await coordinator.switch_provider(
        provider_name="anthropic",
        model="claude-sonnet-4-20250514",
        reason="user_request",
        verify_health=True,
    )

    # Switch model
    success = await coordinator.switch_model(
        model="claude-opus-4-20250514",
        reason="user_request",
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.provider.switcher import ProviderSwitcher
    from victor.agent.provider.health_monitor import ProviderHealthMonitor

logger = logging.getLogger(__name__)


class ProviderSwitchCoordinator:
    """Coordinate provider/model switching operations.

    This coordinator manages the workflow of switching between providers
    and models, handling validation, health checks, fallback logic, and
    post-switch hooks.

    Responsibilities:
    - Validate switch requests
    - Coordinate health checks (if enabled)
    - Execute switches via ProviderSwitcher
    - Handle errors and retry logic
    - Execute post-switch hooks
    - Provide switch history and state

    Example:
        coordinator = ProviderSwitchCoordinator(
            provider_switcher=switcher,
            health_monitor=monitor,
        )

        # Switch with health verification
        success = await coordinator.switch_provider(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            reason="user_request",
            verify_health=True,
        )

        # Get current state
        state = coordinator.get_current_state()
    """

    def __init__(
        self,
        provider_switcher: "ProviderSwitcher",
        health_monitor: Optional["ProviderHealthMonitor"] = None,
    ):
        """Initialize the provider switch coordinator.

        Args:
            provider_switcher: ProviderSwitcher for actual switching logic
            health_monitor: Optional health monitor for pre-switch checks
        """
        self._provider_switcher = provider_switcher
        self._health_monitor = health_monitor
        self._post_switch_hooks: List[Callable[[Any], None]] = []

    async def switch_provider(
        self,
        provider_name: str,
        model: str,
        reason: str = "manual",
        settings: Optional[Any] = None,
        verify_health: bool = False,
        max_retries: int = 0,
        **provider_kwargs: Any,
    ) -> bool:
        """Switch to a different provider.

        This method coordinates the complete switch workflow:
        1. Validate the switch request
        2. Perform health check (if enabled)
        3. Execute the switch via ProviderSwitcher
        4. Handle errors with retry logic
        5. Execute post-switch hooks

        Args:
            provider_name: Name of the provider to switch to
            model: Model identifier
            reason: Reason for the switch (user_request, fallback, etc.)
            settings: Optional settings for provider configuration
            verify_health: If True, perform health check before switch
            max_retries: Number of retries on transient errors
            **provider_kwargs: Additional provider arguments

        Returns:
            True if switch succeeded, False otherwise
        """
        # Validate request
        is_valid, error = self.validate_switch_request(provider_name, model)
        if not is_valid:
            logger.error(f"Invalid switch request: {error}")
            return False

        # Attempt switch with retry logic
        for attempt in range(max_retries + 1):
            try:
                # Perform health check if enabled
                if verify_health and self._health_monitor:
                    try:
                        new_provider = self._create_provider(provider_name, **provider_kwargs)
                        is_healthy = await self._health_monitor.check_health(new_provider)
                        if not is_healthy:
                            logger.warning(
                                f"Provider {provider_name} failed health check (attempt {attempt + 1})"
                            )
                            # Could attempt fallback here
                    except Exception as e:
                        logger.warning(f"Health check failed: {e}")
                        # Continue with switch despite health check failure

                # Execute switch via ProviderSwitcher
                success = await self._provider_switcher.switch_provider(
                    provider_name=provider_name,
                    model=model,
                    reason=reason,
                    settings=settings,
                    **provider_kwargs,
                )

                if success:
                    # Execute post-switch hooks
                    self._execute_post_switch_hooks()

                    logger.info(f"Successfully switched to {provider_name}:{model} ({reason})")
                    return True
                else:
                    # Switch returned False
                    if attempt < max_retries:
                        logger.info(f"Switch failed, retrying... (attempt {attempt + 1})")
                        continue
                    else:
                        logger.error(f"Switch failed after {max_retries + 1} attempts")
                        return False

            except Exception as e:
                logger.warning(f"Switch attempt {attempt + 1} failed: {e}")

                if attempt < max_retries:
                    # Retry on transient errors
                    continue
                else:
                    # Final attempt failed
                    logger.error(f"Switch failed after {max_retries + 1} attempts: {e}")
                    return False

        return False

    async def switch_model(
        self,
        model: str,
        reason: str = "manual",
    ) -> bool:
        """Switch to a different model on current provider.

        Args:
            model: New model name
            reason: Reason for the switch

        Returns:
            True if switch succeeded, False otherwise
        """
        try:
            # Execute switch via ProviderSwitcher
            success = await self._provider_switcher.switch_model(
                model=model,
                reason=reason,
            )

            if success:
                # Execute post-switch hooks
                self._execute_post_switch_hooks()

                logger.info(f"Successfully switched to model {model} ({reason})")
                return True
            else:
                logger.error(f"Failed to switch to model {model}")
                return False

        except Exception as e:
            logger.error(f"Error switching model to {model}: {e}")
            return False

    def validate_switch_request(
        self,
        provider_name: str,
        model: str,
    ) -> tuple[bool, Optional[str]]:
        """Validate a provider/model switch request.

        Args:
            provider_name: Name of the provider
            model: Model name

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not provider_name or not provider_name.strip():
            return False, "Provider name cannot be empty"

        if not model or not model.strip():
            return False, "Model name cannot be empty"

        return True, None

    def get_switch_history(self) -> List[Dict[str, Any]]:
        """Get history of provider/model switches.

        Returns:
            List of switch event dictionaries
        """
        return self._provider_switcher.get_switch_history()

    def get_current_state(self) -> Optional[Any]:
        """Get current provider/model state.

        Returns:
            Current ProviderSwitcherState or None
        """
        return self._provider_switcher.get_current_state()

    def on_switch(self, callback: Callable[[Any], None]) -> None:
        """Register callback for provider/model switches.

        Args:
            callback: Function to call when switch occurs
        """
        self._provider_switcher.on_switch(callback)

    def register_post_switch_hook(self, hook: Callable[[Any], None]) -> None:
        """Register a post-switch hook.

        Post-switch hooks are called after a successful switch.
        They receive the new state as an argument.

        Args:
            hook: Function to call after successful switch
        """
        self._post_switch_hooks.append(hook)

    def _create_provider(
        self,
        provider_name: str,
        **provider_kwargs: Any,
    ) -> Any:
        """Create a provider instance for health checking.

        Args:
            provider_name: Name of the provider
            **provider_kwargs: Provider arguments

        Returns:
            Provider instance
        """
        from victor.providers.registry import ProviderRegistry

        return ProviderRegistry.create(provider_name, **provider_kwargs)

    def _execute_post_switch_hooks(self) -> None:
        """Execute all registered post-switch hooks.

        Gets the current state and passes it to each hook.
        """
        state = self.get_current_state()
        if state is None:
            logger.warning("No state available, skipping post-switch hooks")
            return

        for hook in self._post_switch_hooks:
            try:
                hook(state)
            except Exception as e:
                logger.error(f"Post-switch hook error: {e}")


def create_provider_switch_coordinator(
    provider_switcher: "ProviderSwitcher",
    health_monitor: Optional["ProviderHealthMonitor"] = None,
) -> ProviderSwitchCoordinator:
    """Factory function to create a ProviderSwitchCoordinator.

    Args:
        provider_switcher: ProviderSwitcher for switching logic
        health_monitor: Optional health monitor for pre-switch checks

    Returns:
        Configured ProviderSwitchCoordinator instance
    """
    return ProviderSwitchCoordinator(
        provider_switcher=provider_switcher,
        health_monitor=health_monitor,
    )


__all__ = [
    "ProviderSwitchCoordinator",
    "create_provider_switch_coordinator",
]
