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

"""Provider lifecycle manager for orchestrator.

Phase 1.2: Extract ProviderLifecycleManager.

This manager handles provider/model switch operations, extracting logic
from orchestrator's _apply_post_switch_hooks() method. It manages:
1. Model exploration settings update
2. Prompt contributor retrieval
3. Prompt builder creation
4. Tool budget calculation

Design Pattern:
- Follows SRP by managing only provider lifecycle concerns
- Uses DI container for accessing vertical extensions
- Provides clean interface for orchestrator integration
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.framework.prompts.common_prompts import SystemPromptBuilder

logger = logging.getLogger(__name__)


class ProviderLifecycleManager:
    """Manager for provider lifecycle operations.

    This class implements ProviderLifecycleProtocol and provides the
    concrete implementation for post-switch hook operations.

    The manager is designed to be instantiated once and reused across
    provider/model switches.

    Usage:
        manager = ProviderLifecycleManager(container)

        # After provider switch
        manager.apply_exploration_settings(tracker, capabilities)
        contributors = manager.get_prompt_contributors()
        builder = manager.create_prompt_builder(...)
        budget = manager.calculate_tool_budget(capabilities, settings)

    Attributes:
        _container: DI container for service resolution
    """

    def __init__(self, container: "ServiceContainer") -> None:
        """Initialize the provider lifecycle manager.

        Args:
            container: DI container for resolving vertical extensions
        """
        self._container = container
        logger.debug("ProviderLifecycleManager initialized")

    def apply_exploration_settings(self, tracker: Any, capabilities: Any) -> None:
        """Apply model-specific exploration settings to tracker.

        Updates the unified tracker with exploration multiplier and
        continuation patience from the model's capabilities.

        Args:
            tracker: UnifiedTaskTracker instance with set_model_exploration_settings
            capabilities: ToolCallingCapabilities with exploration_multiplier
                         and continuation_patience attributes
        """
        tracker.set_model_exploration_settings(
            exploration_multiplier=capabilities.exploration_multiplier,
            continuation_patience=capabilities.continuation_patience,
        )
        logger.debug(
            f"Applied exploration settings: multiplier={capabilities.exploration_multiplier}, "
            f"patience={capabilities.continuation_patience}"
        )

    def get_prompt_contributors(self) -> List[Any]:
        """Get prompt contributors from vertical extensions.

        Retrieves registered prompt contributors from the vertical
        extensions in the DI container. Handles errors gracefully.

        Returns:
            List of IPromptContributor instances (empty if none available)
        """
        try:
            from victor.core.verticals.protocols import VerticalExtensions

            extensions = self._container.get_optional(VerticalExtensions)
            if extensions and hasattr(extensions, "prompt_contributors"):
                contributors = extensions.prompt_contributors
                if contributors:
                    logger.debug(f"Retrieved {len(contributors)} prompt contributors")
                    return contributors
        except ImportError:
            logger.debug("VerticalExtensions module not available")
        except AttributeError as e:
            logger.warning(f"VerticalExtensions missing expected attributes: {e}")
        except Exception as e:
            logger.debug(f"Error retrieving prompt contributors: {e}")

        return []

    def create_prompt_builder(
        self,
        provider_name: str,
        model: str,
        tool_adapter: Any,
        capabilities: Any,
        prompt_contributors: List[Any],
    ) -> "SystemPromptBuilder":
        """Create a new SystemPromptBuilder with given parameters.

        Args:
            provider_name: Name of the current provider (e.g., "anthropic")
            model: Name of the current model (e.g., "claude-3-opus")
            tool_adapter: Tool calling adapter for the provider
            capabilities: ToolCallingCapabilities instance
            prompt_contributors: List of prompt contributors from verticals

        Returns:
            Configured SystemPromptBuilder instance
        """
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder(
            provider_name=provider_name,
            model=model,
            tool_adapter=tool_adapter,
            capabilities=capabilities,
            prompt_contributors=prompt_contributors,
        )
        logger.debug(f"Created SystemPromptBuilder for {provider_name}/{model}")
        return builder

    def calculate_tool_budget(self, capabilities: Any, settings: Any) -> int:
        """Calculate the appropriate tool budget.

        Uses the model's recommended budget with a minimum of 50,
        unless settings has a tool_call_budget override.

        Args:
            capabilities: ToolCallingCapabilities with recommended_tool_budget
            settings: Application settings (may have tool_call_budget override)

        Returns:
            Tool budget value (minimum 50)
        """
        # Check for settings override first
        settings_budget = getattr(settings, "tool_call_budget", None)
        if settings_budget is not None:
            logger.debug(f"Using settings override tool budget: {settings_budget}")
            return int(settings_budget)

        # Use recommended budget with minimum of 50
        recommended = getattr(capabilities, "recommended_tool_budget", 50)
        budget = max(recommended, 50)
        logger.debug(f"Calculated tool budget: {budget} (recommended: {recommended})")
        return budget

    def should_respect_sticky_budget(self, tracker: Any) -> bool:
        """Check if sticky budget should be respected.

        When a user manually overrides the tool budget, we set a "sticky"
        flag to prevent automatic resets on provider switch.

        Args:
            tracker: UnifiedTaskTracker that may have _sticky_user_budget flag

        Returns:
            True if sticky budget is set and should be respected
        """
        return getattr(tracker, "_sticky_user_budget", False)


def create_provider_lifecycle_manager(
    container: "ServiceContainer",
) -> ProviderLifecycleManager:
    """Factory function for creating ProviderLifecycleManager.

    This function is used for DI container registration.

    Args:
        container: DI container for service resolution

    Returns:
        New ProviderLifecycleManager instance
    """
    return ProviderLifecycleManager(container)
