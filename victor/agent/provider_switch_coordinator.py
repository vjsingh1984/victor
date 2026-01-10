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
- Post-switch hook execution with priority ordering
- Error handling and retry logic

Design Principles:
- Single Responsibility: Coordinate switching workflow only
- Delegation: Use ProviderSwitcher for actual switching
- Composable: Works with existing health monitors and event emitters
- Observable: Priority-ordered hooks with async support (SRP compliance)
- Open/Closed: New hooks can be added without modifying coordinator

Usage:
    coordinator = ProviderSwitchCoordinator(
        provider_switcher=switcher,
        health_monitor=monitor,
    )

    # Register priority-ordered hooks
    coordinator.register_hook(ExplorationSettingsHook(tracker), priority=10)
    coordinator.register_hook(PromptBuilderHook(builder), priority=20)
    coordinator.register_hook(SystemPromptHook(prompt_builder), priority=30)
    coordinator.register_hook(ToolBudgetHook(settings), priority=40)

    # Switch provider - hooks execute in priority order
    success = await coordinator.switch_provider(
        provider_name="anthropic",
        model="claude-sonnet-4-20250514",
        reason="user_request",
        verify_health=True,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from victor.agent.provider.switcher import ProviderSwitcher
    from victor.agent.provider.health_monitor import ProviderHealthMonitor

logger = logging.getLogger(__name__)


class HookPriority(IntEnum):
    """Standard hook priorities for predictable execution order.

    Lower values execute first. Use these as guidelines:
    - FIRST (0): Critical setup that other hooks depend on
    - EARLY (10): Exploration/capability settings
    - NORMAL (20): Prompt and builder updates
    - LATE (30): System prompt rebuilding
    - LAST (40): Budget and final adjustments
    """

    FIRST = 0
    EARLY = 10
    NORMAL = 20
    LATE = 30
    LAST = 40


@dataclass(frozen=True)
class SwitchContext:
    """Context passed to post-switch hooks.

    Provides all information hooks need to update state after a switch.
    Immutable to prevent hooks from modifying shared context.

    Attributes:
        old_provider: Previous provider name (None if first switch)
        old_model: Previous model name (None if first switch)
        new_provider: New provider name
        new_model: New model name
        reason: Reason for the switch (user_request, fallback, etc.)
        switch_type: Type of switch ('provider' or 'model')
        respect_sticky_budget: If True, don't reset tool budget on user override
        metadata: Additional context-specific data
    """

    old_provider: Optional[str]
    old_model: Optional[str]
    new_provider: str
    new_model: str
    reason: str
    switch_type: str  # 'provider' or 'model'
    respect_sticky_budget: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PostSwitchHook(Protocol):
    """Protocol for post-switch hooks (ISP compliance).

    Hooks implement a single responsibility and execute after successful switch.
    Supports both sync and async execution via duck typing.

    Example:
        class MyHook:
            name = "my_hook"

            def execute(self, context: SwitchContext) -> None:
                # Update state based on new provider/model
                pass

        # Or async:
        class MyAsyncHook:
            name = "my_async_hook"

            async def execute(self, context: SwitchContext) -> None:
                await do_async_update(context)
    """

    name: str

    def execute(self, context: SwitchContext) -> None:
        """Execute the hook with switch context."""


@dataclass
class RegisteredHook:
    """A hook registered with its priority."""

    hook: PostSwitchHook
    priority: int

    def __lt__(self, other: "RegisteredHook") -> bool:
        """Sort by priority (lower first)."""
        return self.priority < other.priority


class ProviderSwitchCoordinator:
    """Coordinate provider/model switching operations.

    This coordinator manages the workflow of switching between providers
    and models, handling validation, health checks, fallback logic, and
    priority-ordered post-switch hooks.

    Responsibilities:
    - Validate switch requests
    - Coordinate health checks (if enabled)
    - Execute switches via ProviderSwitcher
    - Handle errors and retry logic
    - Execute priority-ordered post-switch hooks
    - Provide switch history and state

    Example:
        coordinator = ProviderSwitchCoordinator(
            provider_switcher=switcher,
            health_monitor=monitor,
        )

        # Register hooks with priorities
        coordinator.register_hook(
            ExplorationSettingsHook(tracker),
            priority=HookPriority.EARLY,
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
        # Legacy callback-based hooks (for backward compatibility)
        self._post_switch_hooks: List[Callable[[Any], None]] = []
        # New priority-ordered protocol-based hooks
        self._registered_hooks: List[RegisteredHook] = []
        # Track current provider/model for context
        self._current_provider: Optional[str] = None
        self._current_model: Optional[str] = None

    async def switch_provider(
        self,
        provider_name: str,
        model: str,
        reason: str = "manual",
        settings: Optional[Any] = None,
        verify_health: bool = False,
        max_retries: int = 0,
        respect_sticky_budget: bool = True,
        **provider_kwargs: Any,
    ) -> bool:
        """Switch to a different provider.

        This method coordinates the complete switch workflow:
        1. Validate the switch request
        2. Perform health check (if enabled)
        3. Execute the switch via ProviderSwitcher
        4. Handle errors with retry logic
        5. Execute priority-ordered post-switch hooks

        Args:
            provider_name: Name of the provider to switch to
            model: Model identifier
            reason: Reason for the switch (user_request, fallback, etc.)
            settings: Optional settings for provider configuration
            verify_health: If True, perform health check before switch
            max_retries: Number of retries on transient errors
            respect_sticky_budget: If True, don't reset tool budget on user override
            **provider_kwargs: Additional provider arguments

        Returns:
            True if switch succeeded, False otherwise
        """
        # Validate request
        is_valid, error = self.validate_switch_request(provider_name, model)
        if not is_valid:
            logger.error(f"Invalid switch request: {error}")
            return False

        # Store old state for context
        old_provider = self._current_provider
        old_model = self._current_model

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
                    # Update current state
                    self._current_provider = provider_name
                    self._current_model = model

                    # Build context for hooks
                    context = SwitchContext(
                        old_provider=old_provider,
                        old_model=old_model,
                        new_provider=provider_name,
                        new_model=model,
                        reason=reason,
                        switch_type="provider",
                        respect_sticky_budget=respect_sticky_budget,
                        metadata=provider_kwargs,
                    )

                    # Execute hooks
                    await self._execute_hooks(context)

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
        respect_sticky_budget: bool = False,
    ) -> bool:
        """Switch to a different model on current provider.

        Args:
            model: New model name
            reason: Reason for the switch
            respect_sticky_budget: If True, don't reset tool budget on user override

        Returns:
            True if switch succeeded, False otherwise
        """
        old_model = self._current_model

        try:
            # Execute switch via ProviderSwitcher
            success = await self._provider_switcher.switch_model(
                model=model,
                reason=reason,
            )

            if success:
                # Update current state
                self._current_model = model

                # Build context for hooks
                context = SwitchContext(
                    old_provider=self._current_provider,
                    old_model=old_model,
                    new_provider=self._current_provider or "",
                    new_model=model,
                    reason=reason,
                    switch_type="model",
                    respect_sticky_budget=respect_sticky_budget,
                )

                # Execute hooks
                await self._execute_hooks(context)

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
        """Register a legacy post-switch hook (backward compatibility).

        Post-switch hooks are called after a successful switch.
        They receive the new state as an argument.

        Note:
            Prefer register_hook() for new code.

        Args:
            hook: Function to call after successful switch
        """
        self._post_switch_hooks.append(hook)

    def register_hook(
        self,
        hook: PostSwitchHook,
        priority: int = HookPriority.NORMAL,
    ) -> None:
        """Register a priority-ordered post-switch hook.

        Hooks are executed in priority order (lower values first).
        Use HookPriority enum for standard priorities.

        Args:
            hook: Hook implementing PostSwitchHook protocol
            priority: Execution priority (lower = earlier)

        Example:
            coordinator.register_hook(
                ExplorationSettingsHook(tracker),
                priority=HookPriority.EARLY,
            )
        """
        registered = RegisteredHook(hook=hook, priority=priority)
        self._registered_hooks.append(registered)
        # Keep sorted by priority
        self._registered_hooks.sort()
        logger.debug(f"Registered hook '{hook.name}' with priority {priority}")

    def unregister_hook(self, name: str) -> bool:
        """Unregister a hook by name.

        Args:
            name: Name of the hook to remove

        Returns:
            True if hook was found and removed
        """
        original_len = len(self._registered_hooks)
        self._registered_hooks = [
            h for h in self._registered_hooks if h.hook.name != name
        ]
        removed = len(self._registered_hooks) < original_len
        if removed:
            logger.debug(f"Unregistered hook '{name}'")
        return removed

    def list_hooks(self) -> List[tuple[str, int]]:
        """List registered hooks with their priorities.

        Returns:
            List of (hook_name, priority) tuples in execution order
        """
        return [(h.hook.name, h.priority) for h in self._registered_hooks]

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

    async def _execute_hooks(self, context: SwitchContext) -> None:
        """Execute all registered hooks in priority order.

        Executes both legacy callback hooks and new protocol-based hooks.
        Errors in individual hooks are logged but don't stop execution.

        Args:
            context: Switch context with old/new provider/model info
        """
        import asyncio
        import inspect

        # Execute legacy hooks first (for backward compatibility)
        state = self.get_current_state()
        if state is not None:
            for hook in self._post_switch_hooks:
                try:
                    hook(state)
                except Exception as e:
                    logger.error(f"Legacy post-switch hook error: {e}")

        # Execute new protocol-based hooks in priority order
        for registered in self._registered_hooks:
            try:
                result = registered.hook.execute(context)
                # Support async hooks via duck typing
                if inspect.isawaitable(result):
                    await result
                logger.debug(f"Executed hook '{registered.hook.name}'")
            except Exception as e:
                logger.error(
                    f"Post-switch hook '{registered.hook.name}' error: {e}",
                    exc_info=True,
                )

    def _execute_post_switch_hooks(self) -> None:
        """Legacy method for backward compatibility.

        Deprecated: Use _execute_hooks() with SwitchContext instead.
        """
        import asyncio

        # Create a minimal context
        context = SwitchContext(
            old_provider=None,
            old_model=None,
            new_provider=self._current_provider or "",
            new_model=self._current_model or "",
            reason="unknown",
            switch_type="unknown",
        )

        # Run async method synchronously for legacy callers
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._execute_hooks(context))
        except RuntimeError:
            # No running loop, run synchronously
            asyncio.run(self._execute_hooks(context))


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
    "PostSwitchHook",
    "SwitchContext",
    "HookPriority",
    "RegisteredHook",
]
