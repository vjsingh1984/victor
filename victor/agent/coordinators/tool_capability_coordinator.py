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

"""Tool capability coordinator for provider/model capability checks.

This coordinator manages tool capability checks against the capability matrix,
providing a centralized location for capability-related logic. It handles
provider/model combinations and checks if tools are supported.

Key Features:
- Tool calling capability checks
- Model capability matrix queries
- Supported model listings
- Warning management for capability issues
- Provider capability validation

Design Patterns:
- SRP: Single responsibility for capability checking
- Strategy Pattern: Different capability check strategies
- Dependency Inversion: Depends on protocols, not implementations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from rich.console import Console

logger = logging.getLogger(__name__)


@dataclass
class CapabilityCheckResult:
    """Result of a capability check.

    Attributes:
        supported: Whether the capability is supported
        provider: Provider name
        model: Model name
        capability: Capability that was checked
        alternative_models: List of alternative models that support the capability
        reason: Reason if capability is not supported
    """

    supported: bool
    provider: str
    model: str
    capability: str
    alternative_models: List[str] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class ModelCapabilityInfo:
    """Information about a model's capabilities.

    Attributes:
        provider: Provider name
        model: Model name
        tool_calling_supported: Whether tool calling is supported
        parallel_calls_supported: Whether parallel tool calls are supported
        other_capabilities: Set of other capability names
    """

    provider: str
    model: str
    tool_calling_supported: bool
    parallel_calls_supported: bool
    other_capabilities: Set[str] = field(default_factory=set)


class ToolCapabilityCoordinator:
    """Coordinator for tool capability checks.

    This coordinator provides a unified interface for checking tool calling
    capabilities across different providers and models. It manages capability
    lookups and provides helpful error messages.

    Example:
        ```python
        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=tool_capabilities,
            console=console
        )

        # Check if model supports tool calls
        result = coordinator.check_tool_calling_capability(
            provider_name="anthropic",
            model="claude-sonnet-4-5"
        )

        if not result.supported:
            print(f"Tool calling not supported: {result.reason}")
            print(f"Try these models: {result.alternative_models}")
        ```
    """

    def __init__(
        self,
        tool_capabilities: Any,
        console: Optional["Console"] = None,
        warn_once: bool = True,
    ):
        """Initialize the tool capability coordinator.

        Args:
            tool_capabilities: Tool capabilities matrix/capability checker
            console: Optional console for user-facing messages
            warn_once: Whether to warn only once per provider/model combo
        """
        self._tool_capabilities = tool_capabilities
        self._console = console
        self._warn_once = warn_once
        self._warned_cache: Dict[tuple[str, str], bool] = {}

    def check_tool_calling_capability(
        self,
        provider_name: Optional[str],
        model: str,
    ) -> CapabilityCheckResult:
        """Check if provider/model combo supports tool calling.

        Args:
            provider_name: Name of the provider (e.g., "anthropic", "openai")
            model: Model name (e.g., "claude-sonnet-4-5")

        Returns:
            CapabilityCheckResult with support information and alternatives
        """
        # Default to supported if no provider name
        if not provider_name:
            return CapabilityCheckResult(
                supported=True,
                provider="unknown",
                model=model,
                capability="tool_calling",
            )

        # Check capability
        supported = self._tool_capabilities.is_tool_call_supported(provider_name, model)

        if supported:
            return CapabilityCheckResult(
                supported=True,
                provider=provider_name,
                model=model,
                capability="tool_calling",
            )

        # Not supported - get alternatives and reason
        alternative_models = list(self._tool_capabilities.get_supported_models(provider_name))
        reason = (
            f"Model '{model}' is not marked as tool-call-capable for provider '{provider_name}'"
        )

        return CapabilityCheckResult(
            supported=False,
            provider=provider_name,
            model=model,
            capability="tool_calling",
            alternative_models=alternative_models,
            reason=reason,
        )

    def log_capability_warning(
        self,
        provider_name: str,
        model: str,
        result: CapabilityCheckResult,
    ) -> None:
        """Log capability warning with user-friendly message.

        Args:
            provider_name: Provider name
            model: Model name
            result: Capability check result
        """
        # Check if we should warn (warn once mode)
        if self._warn_once:
            cache_key = (provider_name, model)
            if cache_key in self._warned_cache:
                return
            self._warned_cache[cache_key] = True

        # Format alternative models string
        known = ", ".join(result.alternative_models) or "none"

        # Log warning
        logger.warning(
            f"Model '{model}' is not marked as tool-call-capable for provider '{provider_name}'. "
            f"Known tool-capable models: {known}"
        )

        # Console message if available
        if self._console:
            self._console.print(
                f"[yellow]⚠ Model '{model}' is not marked as tool-call-capable for provider '{provider_name}'. "
                f"Running without tools.[/]"
            )

    def get_supported_models(self, provider_name: str) -> List[str]:
        """Get list of models that support tool calling for a provider.

        Args:
            provider_name: Provider name

        Returns:
            List of model names that support tool calling
        """
        try:
            return list(self._tool_capabilities.get_supported_models(provider_name))
        except Exception as e:
            logger.warning(f"Failed to get supported models for {provider_name}: {e}")
            return []

    def get_capability_info(
        self,
        provider_name: str,
        model: str,
    ) -> Optional[ModelCapabilityInfo]:
        """Get detailed capability information for a model.

        Args:
            provider_name: Provider name
            model: Model name

        Returns:
            ModelCapabilityInfo with detailed capabilities or None if unavailable
        """
        try:
            # Check tool calling support
            tool_calling_supported = self._tool_capabilities.is_tool_call_supported(
                provider_name, model
            )

            # Check parallel calls support (if available)
            parallel_calls_supported = False
            if hasattr(self._tool_capabilities, "supports_parallel_calls"):
                parallel_calls_supported = self._tool_capabilities.supports_parallel_calls(
                    provider_name, model
                )

            # Get other capabilities (if available)
            other_capabilities: Set[str] = set()
            if hasattr(self._tool_capabilities, "get_capabilities"):
                other_capabilities = self._tool_capabilities.get_capabilities(provider_name, model)

            return ModelCapabilityInfo(
                provider=provider_name,
                model=model,
                tool_calling_supported=tool_calling_supported,
                parallel_calls_supported=parallel_calls_supported,
                other_capabilities=other_capabilities,
            )

        except Exception as e:
            logger.warning(f"Failed to get capability info for {provider_name}/{model}: {e}")
            return None

    def should_use_tools(self) -> bool:
        """Check if tools should be used for current configuration.

        This method always returns True as tool selection is handled
        by the tool selector based on query context.

        Returns:
            True (tools are always available for selection)
        """
        return True

    def validate_capability(
        self,
        provider_name: Optional[str],
        model: str,
        capability: str = "tool_calling",
    ) -> bool:
        """Validate that a capability is supported.

        Args:
            provider_name: Provider name
            model: Model name
            capability: Capability name to validate

        Returns:
            True if capability is supported
        """
        if capability == "tool_calling":
            result = self.check_tool_calling_capability(provider_name, model)
            return result.supported

        # For other capabilities, try to get capability info
        info = self.get_capability_info(provider_name or "", model)
        if info is None:
            return True  # Default to supported if unknown

        return capability in info.other_capabilities

    def get_capable_models(
        self,
        provider_name: str,
        capability: str = "tool_calling",
    ) -> List[str]:
        """Get models that support a specific capability.

        Args:
            provider_name: Provider name
            capability: Capability name (default: "tool_calling")

        Returns:
            List of model names supporting the capability
        """
        if capability == "tool_calling":
            return self.get_supported_models(provider_name)

        # For other capabilities, we'd need to query the capability matrix
        # This is a placeholder for future enhancement
        return []


def create_tool_capability_coordinator(
    tool_capabilities: Any,
    console: Optional["Console"] = None,
    warn_once: bool = True,
) -> ToolCapabilityCoordinator:
    """Factory function to create a ToolCapabilityCoordinator.

    Args:
        tool_capabilities: Tool capabilities matrix
        console: Optional console for messages
        warn_once: Whether to warn only once per model

    Returns:
        Configured ToolCapabilityCoordinator instance
    """
    return ToolCapabilityCoordinator(
        tool_capabilities=tool_capabilities,
        console=console,
        warn_once=warn_once,
    )
