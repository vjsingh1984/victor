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

"""Configuration coordinator for loading and validating configuration.

This module implements the ConfigCoordinator which consolidates configuration
loading from multiple sources (IConfigProvider implementations) with validation
and caching.

Design Patterns:
    - Strategy Pattern: Multiple config providers via IConfigProvider
    - Chain of Responsibility: Try providers in priority order
    - Caching: Cache loaded configuration to avoid repeated loads
    - SRP: Focused only on configuration loading and validation

Usage:
    from victor.agent.coordinators.config_coordinator import ConfigCoordinator
    from victor.protocols import IConfigProvider

    # Create coordinator with multiple providers
    settings_provider = SettingsConfigProvider(settings)
    env_provider = EnvironmentConfigProvider()
    coordinator = ConfigCoordinator(providers=[settings_provider, env_provider])

    # Load configuration
    config = await coordinator.load_config(session_id="abc123")

    # Validate configuration
    result = await coordinator.validate_config(config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from victor.agent.protocols import ToolAccessContext

from victor.protocols import IConfigProvider

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from configuration validation.

    Attributes:
        valid: Whether configuration is valid
        errors: List of validation errors
        warnings: List of validation warnings
        metadata: Additional validation metadata
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] | None = None


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration.

    Attributes:
        session_id: Session identifier
        provider: LLM provider name
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tool_selection: Tool selection configuration
        thinking: Enable extended thinking mode
        profile_name: Profile name
        metadata: Additional configuration
    """

    session_id: str
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    tool_selection: dict[str, Any] | None = None
    thinking: bool = False
    profile_name: str | None = None
    metadata: dict[str, Any] | None = None


class ConfigCoordinator:
    """Configuration loading and validation coordination.

    This coordinator manages multiple IConfigProvider implementations,
    loading configuration from various sources in priority order and
    providing validation.

    Responsibilities:
    - Load configuration from multiple providers
    - Merge configuration from multiple sources
    - Validate configuration against schema
    - Cache configuration to avoid repeated loads
    - Handle configuration errors gracefully

    Provider Priority:
    Providers are tried in order of priority (higher first). The first
    provider to return a valid configuration is used for that part
    of the configuration.
    """

    def __init__(
        self,
        providers: Optional[list[IConfigProvider]] = None,
        enable_cache: bool = True,
    ) -> None:
        """Initialize the configuration coordinator.

        Args:
            providers: List of config providers (ordered by priority)
            enable_cache: Enable configuration caching
        """
        # Sort providers by priority (lowest first, so higher priority can override)
        self._providers = sorted(providers or [], key=lambda p: p.priority())
        self._enable_cache = enable_cache
        self._config_cache: dict[str, dict[str, Any]] = {}

    async def load_config(
        self,
        session_id: str,
        config_override: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Load configuration for a session.

        Tries each provider in priority order and merges the results.
        Later providers can override values from earlier providers.

        Args:
            session_id: Session identifier
            config_override: Optional override values

        Returns:
            Merged configuration dictionary

        Raises:
            ConfigurationError: If all providers fail

        Example:
            config = await coordinator.load_config(
                session_id="abc123",
                config_override={"temperature": 0.5}
            )
            # Returns merged config with temperature overridden
        """
        # Check cache first
        if self._enable_cache and session_id in self._config_cache:
            config = self._config_cache[session_id].copy()
        else:
            # Load from all providers
            config = {}
            errors = []

            # Try each provider and merge results
            for provider in self._providers:
                try:
                    provider_config = await provider.get_config(session_id)
                    if provider_config:
                        # Merge with existing config (later providers override)
                        config = self._deep_merge(config, provider_config)
                except Exception as e:
                    # Collect errors for later
                    errors.append((provider.__class__.__name__, str(e)))

            # If no providers returned config and we had errors, raise the first one
            if not config and errors and not self._providers:
                # No providers configured
                raise ValueError("No configuration providers configured")
            elif not config and errors:
                # All providers failed, re-raise the first error
                raise ValueError(f"Configuration loading failed: {errors[0][1]}")

            # Cache the merged config
            if self._enable_cache and config:
                self._config_cache[session_id] = config.copy()

        # Apply overrides
        if config_override:
            config = self._deep_merge(config, config_override)

        return config

    async def load_orchestrator_config(
        self,
        session_id: str,
        provider: str,
        model: str,
        config_override: Optional[dict[str, Any]] = None,
    ) -> OrchestratorConfig:
        """Load orchestrator-specific configuration.

        Loads configuration and converts it to OrchestratorConfig dataclass.

        Args:
            session_id: Session identifier
            provider: LLM provider name
            model: Model identifier
            config_override: Optional override values

        Returns:
            OrchestratorConfig instance

        Example:
            config = await coordinator.load_orchestrator_config(
                session_id="abc123",
                provider="anthropic",
                model="claude-sonnet-4-5"
            )
        """
        base_config = await self.load_config(session_id, config_override)

        return OrchestratorConfig(
            session_id=session_id,
            provider=provider,
            model=model,
            temperature=base_config.get("temperature", 0.7),
            max_tokens=base_config.get("max_tokens", 4096),
            tool_selection=base_config.get("tool_selection"),
            thinking=base_config.get("thinking", False),
            profile_name=base_config.get("profile_name"),
            metadata=base_config.get("metadata"),
        )

    async def validate_config(
        self,
        config: dict[str, Any] | OrchestratorConfig,
    ) -> ValidationResult:
        """Validate configuration.

        Validates configuration against schema and business rules.

        Args:
            config: Configuration to validate

        Returns:
            ValidationResult with validation status

        Example:
            result = await coordinator.validate_config(config)
            if not result.valid:
                print(f"Validation errors: {result.errors}")
        """
        errors = []
        warnings = []

        # Convert to dict if OrchestratorConfig
        if isinstance(config, OrchestratorConfig):
            config_dict = {
                "provider": config.provider,
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "tool_selection": config.tool_selection,
                "thinking": config.thinking,
            }
        else:
            config_dict = config

        # Validate required fields
        if "provider" not in config_dict or not config_dict["provider"]:
            errors.append("Configuration missing required field: provider")

        if "model" not in config_dict or not config_dict["model"]:
            errors.append("Configuration missing required field: model")

        # Validate temperature range
        temperature = config_dict.get("temperature", 0.7)
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            errors.append(f"Invalid temperature: {temperature} (must be between 0 and 2)")

        # Validate max_tokens
        max_tokens = config_dict.get("max_tokens", 4096)
        if not isinstance(max_tokens, int) or max_tokens < 1:
            errors.append(f"Invalid max_tokens: {max_tokens} (must be positive integer)")

        # Validate tool_selection format
        tool_selection = config_dict.get("tool_selection")
        if tool_selection is not None and not isinstance(tool_selection, dict):
            errors.append(f"Invalid tool_selection: {tool_selection} (must be dict or None)")

        # Warnings for non-critical issues (need to re-check types for mypy)
        temp_value: float = temperature if isinstance(temperature, (int, float)) else 0.7
        if temp_value > 1.0:
            warnings.append(f"High temperature ({temp_value}) may produce unpredictable results")

        max_tokens_value: int = max_tokens if isinstance(max_tokens, int) else 4096
        if max_tokens_value > 128000:
            warnings.append(
                f"Very large max_tokens ({max_tokens_value}) may exceed model context limits"
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"config_type": type(config).__name__},
        )

    def invalidate_cache(self, session_id: Optional[str] = None) -> None:
        """Invalidate configuration cache.

        Args:
            session_id: Specific session to invalidate (None = all)

        Example:
            # Invalidate specific session
            coordinator.invalidate_cache(session_id="abc123")

            # Invalidate all sessions
            coordinator.invalidate_cache()
        """
        if session_id:
            self._config_cache.pop(session_id, None)
        else:
            self._config_cache.clear()

    def add_provider(self, provider: IConfigProvider) -> None:
        """Add a configuration provider.

        Args:
            provider: Config provider to add

        Example:
            provider = EnvironmentConfigProvider()
            coordinator.add_provider(provider)
        """
        self._providers.append(provider)
        # Re-sort by priority (lowest first, so higher priority can override)
        self._providers.sort(key=lambda p: p.priority())

    def remove_provider(self, provider: IConfigProvider) -> None:
        """Remove a configuration provider.

        Args:
            provider: Config provider to remove
        """
        if provider in self._providers:
            self._providers.remove(provider)

    def _deep_merge(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    async def create_provider_from_config(
        self,
        config: dict[str, Any],
        settings: Any,
        provider_registry: Any = None,
    ) -> Any:
        """Create a provider instance from configuration.

        Args:
            config: Configuration dictionary with provider/model info
            settings: Settings object for provider-level settings
            provider_registry: Optional provider registry (for testing)

        Returns:
            Provider instance

        Raises:
            ValueError: If provider creation fails
        """
        # Import here to allow mocking in tests
        if provider_registry is None:
            from victor.providers.registry import ProviderRegistry

            provider_registry = ProviderRegistry

        provider_name = config.get("provider")
        if not provider_name:
            raise ValueError("Configuration missing 'provider' field")

        # Get provider-level settings
        provider_settings = settings.get_provider_settings(provider_name)

        # Merge profile-level overrides if present
        profile_overrides = config.get("profile_overrides")
        if profile_overrides:
            provider_settings.update(profile_overrides)
            logger.debug(f"Applied profile overrides: {list(profile_overrides.keys())}")

        # Apply timeout multiplier from model capabilities
        # Slow local models (Ollama, LMStudio) get longer timeouts
        from victor.agent.tool_calling.capabilities import ModelCapabilityLoader

        cap_loader = ModelCapabilityLoader()
        caps = cap_loader.get_capabilities(provider_name, config.get("model", ""))
        if caps and caps.timeout_multiplier > 1.0:
            base_timeout = provider_settings.get("timeout", 300)
            adjusted_timeout = int(base_timeout * caps.timeout_multiplier)
            provider_settings["timeout"] = adjusted_timeout
            logger.info(
                f"Adjusted timeout for {provider_name}/{config.get('model')}: "
                f"{base_timeout}s -> {adjusted_timeout}s (multiplier: {caps.timeout_multiplier}x)"
            )

        # Create provider instance
        provider = provider_registry.create(provider_name, **provider_settings)
        return provider


# Built-in config providers


class SettingsConfigProvider(IConfigProvider):
    """Configuration provider that reads from Settings object.

    This provider extracts configuration from Victor's Settings object,
    which is the primary configuration source for the application.

    Attributes:
        settings: The Settings object
        _priority: Provider priority (higher = tried first)
    """

    def __init__(self, settings: Any, priority: int = 100):
        """Initialize the settings config provider.

        Args:
            settings: Victor Settings object
            priority: Provider priority (default: 100, high priority)
        """
        self._settings = settings
        self._priority = priority

    async def get_config(self, session_id: str) -> dict[str, Any]:
        """Get configuration from Settings.

        Args:
            session_id: Session identifier (not used for Settings)

        Returns:
            Configuration dictionary from Settings
        """
        config = {}

        # Extract common settings
        if hasattr(self._settings, "temperature"):
            config["temperature"] = self._settings.temperature
        if hasattr(self._settings, "max_tokens"):
            config["max_tokens"] = self._settings.max_tokens
        if hasattr(self._settings, "thinking"):
            config["thinking"] = self._settings.thinking
        if hasattr(self._settings, "tool_selection"):
            config["tool_selection"] = self._settings.tool_selection

        return config

    def priority(self) -> int:
        """Get provider priority."""
        return self._priority


class EnvironmentConfigProvider(IConfigProvider):
    """Configuration provider that reads from environment variables.

    This provider extracts configuration from environment variables,
    allowing external configuration without code changes.

    Attributes:
        _prefix: Environment variable prefix
        _priority: Provider priority
    """

    def __init__(self, prefix: str = "VICTOR_", priority: int = 50):
        """Initialize the environment config provider.

        Args:
            prefix: Environment variable prefix (default: "VICTOR_")
            priority: Provider priority (default: 50, medium priority)
        """
        import os

        self._prefix = prefix
        self._env = os.environ
        self._priority = priority

    async def get_config(self, session_id: str) -> dict[str, Any]:
        """Get configuration from environment variables.

        Args:
            session_id: Session identifier

        Returns:
            Configuration dictionary from environment
        """

        config = {}

        # Check for common environment variables
        if f"{self._prefix}TEMPERATURE" in self._env:
            try:
                config["temperature"] = float(self._env[f"{self._prefix}TEMPERATURE"])
            except ValueError:
                pass

        if f"{self._prefix}MAX_TOKENS" in self._env:
            try:
                config["max_tokens"] = int(self._env[f"{self._prefix}MAX_TOKENS"])
            except ValueError:
                pass

        if f"{self._prefix}THINKING" in self._env:
            config["thinking"] = self._env[f"{self._prefix}THINKING"].lower() == "true"

        return config

    def priority(self) -> int:
        """Get provider priority."""
        return self._priority


class ProfileConfigProvider(IConfigProvider):
    """Configuration provider that reads from Victor profiles.

    This provider extracts configuration from Victor's profile system,
    which stores provider, model, and other settings in profiles.yaml.

    Attributes:
        settings: The Settings object
        profile_name: Profile name to load
        _priority: Provider priority (higher = tried first)
    """

    def __init__(self, settings: Any, profile_name: str = "default", priority: int = 200):
        """Initialize the profile config provider.

        Args:
            settings: Victor Settings object
            profile_name: Profile name to load (default: "default")
            priority: Provider priority (default: 200, higher than Settings)
        """
        self._settings = settings
        self._profile_name = profile_name
        self._priority = priority

    async def get_config(self, session_id: str) -> dict[str, Any]:
        """Get configuration from profile.

        Args:
            session_id: Session identifier (not used for profiles)

        Returns:
            Configuration dictionary from profile

        Raises:
            ValueError: If profile not found
        """
        # Load profiles
        profiles = self._settings.load_profiles()
        profile = profiles.get(self._profile_name)

        if not profile:
            available = list(profiles.keys())
            # Use difflib for similar name suggestions
            import difflib

            suggestions = difflib.get_close_matches(self._profile_name, available, n=3, cutoff=0.4)

            error_msg = f"Profile not found: '{self._profile_name}'"
            if suggestions:
                error_msg += f"\n  Did you mean: {', '.join(suggestions)}?"
            if available:
                error_msg += f"\n  Available profiles: {', '.join(sorted(available))}"
            else:
                error_msg += "\n  No profiles configured. Run 'victor init' or create ~/.victor/profiles.yaml'"
            raise ValueError(error_msg)

        # Extract configuration from profile
        config = {
            "provider": profile.provider,
            "model": profile.model,
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "tool_selection": profile.tool_selection,
            "profile_name": self._profile_name,
        }

        # Add profile-level overrides if present
        if hasattr(profile, "__pydantic_extra__") and profile.__pydantic_extra__:
            config["profile_overrides"] = profile.__pydantic_extra__

        return config

    def priority(self) -> int:
        """Get provider priority."""
        return self._priority


__all__ = [
    "ConfigCoordinator",
    "ValidationResult",
    "OrchestratorConfig",
    "SettingsConfigProvider",
    "EnvironmentConfigProvider",
    "ProfileConfigProvider",
    "ToolAccessConfigCoordinator",
]


# =============================================================================
# Tool Access Configuration Coordinator
# =============================================================================


class ToolAccessConfigCoordinator:
    """Coordinator for tool access configuration management.

    This coordinator extracts tool access configuration logic from the
    orchestrator, providing a unified interface for:
    - Building tool access context
    - Querying enabled tools
    - Checking individual tool access
    - Setting enabled tools with propagation

    Design Patterns:
        - Facade Pattern: Simplifies access to tool access configuration
        - Delegation Pattern: Delegates to ToolAccessController and ModeCoordinator
        - SRP: Focused only on tool access configuration

    Usage:
        from victor.agent.coordinators.config_coordinator import (
            ToolAccessConfigCoordinator,
        )

        coordinator = ToolAccessConfigCoordinator(
            tool_access_controller=controller,
            mode_coordinator=mode_coordinator,
            tool_registry=registry,
        )

        # Check if tool is enabled
        if coordinator.is_tool_enabled("bash"):
            # Tool is enabled

        # Get all enabled tools
        enabled = coordinator.get_enabled_tools()
    """

    def __init__(
        self,
        tool_access_controller: Optional[Any] = None,
        mode_coordinator: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
    ) -> None:
        """Initialize the tool access configuration coordinator.

        Args:
            tool_access_controller: ToolAccessController instance
            mode_coordinator: ModeCoordinator instance
            tool_registry: ToolRegistry instance
        """
        self._tool_access_controller = tool_access_controller
        self._mode_coordinator = mode_coordinator
        self._tool_registry = tool_registry

    def build_tool_access_context(
        self,
        session_enabled_tools: Optional[set[str]] = None,
        current_mode: Optional[str] = None,
    ) -> ToolAccessContext:
        """Build ToolAccessContext for unified access control checks.

        Consolidates context construction used by get_enabled_tools() and
        is_tool_enabled() to ensure consistent access control decisions.

        Args:
            session_enabled_tools: Tools enabled for this session
            current_mode: Current agent mode name

        Returns:
            ToolAccessContext with session tools and current mode
        """
        from victor.agent.protocols import ToolAccessContext

        # Get current mode from coordinator if not provided
        if current_mode is None and self._mode_coordinator:
            current_mode = self._mode_coordinator.current_mode_name

        return ToolAccessContext(
            session_enabled_tools=session_enabled_tools,
            current_mode=current_mode,
        )

    def get_enabled_tools(
        self,
        session_enabled_tools: Optional[set[str]] = None,
    ) -> set[str]:
        """Get currently enabled tool names.

        Uses ToolAccessController if available for unified access control.
        In BUILD mode (allow_all_tools=True), expands to all available tools
        regardless of vertical restrictions.

        Args:
            session_enabled_tools: Optional override for session tools

        Returns:
            Set of enabled tool names for this session
        """
        # Use ToolAccessController if available (new unified approach)
        if self._tool_access_controller:
            context = self.build_tool_access_context(session_enabled_tools=session_enabled_tools)
            result: Any = self._tool_access_controller.get_allowed_tools(context)
            return cast(set[str], result)

        # Check mode coordinator for BUILD mode (allows all tools minus disallowed)
        if self._mode_coordinator:
            mode_config = self._mode_coordinator.get_mode_config()
            if mode_config.allow_all_tools:
                all_tools = self.get_available_tools()
                # Remove any explicitly disallowed tools
                disallowed: Any = mode_config.disallowed_tools
                enabled = all_tools - cast(set[str], disallowed)
                return enabled

        # Check for framework-set tools (vertical filtering)
        if session_enabled_tools:
            return session_enabled_tools

        # Fall back to all available tools
        return self.get_available_tools()

    def is_tool_enabled(
        self,
        tool_name: str,
        session_enabled_tools: Optional[set[str]] = None,
    ) -> bool:
        """Check if a specific tool is enabled.

        Uses ToolAccessController for unified layered access control:
        Safety (L0) > Mode (L1) > Session (L2) > Vertical (L3) > Stage (L4) > Intent (L5)

        Falls back to legacy logic if controller not available.

        Args:
            tool_name: Name of tool to check
            session_enabled_tools: Optional override for session tools

        Returns:
            True if tool is enabled
        """
        # Use ToolAccessController if available (new unified approach)
        if self._tool_access_controller:
            context = self.build_tool_access_context(session_enabled_tools=session_enabled_tools)
            decision: Any = self._tool_access_controller.check_access(tool_name, context)
            return cast(bool, getattr(decision, "allowed", True))

        # Check mode coordinator for mode-based restrictions
        if self._mode_coordinator and self._mode_coordinator.is_tool_allowed(tool_name):
            # Tool is allowed by mode, check if it exists in registry
            if self._tool_registry and tool_name in self._tool_registry.list_tools():
                return True

        # Fall back to session/vertical restrictions
        enabled = self.get_enabled_tools(session_enabled_tools)
        return tool_name in enabled

    def set_enabled_tools(
        self,
        tools: set[str],
        session_enabled_tools_attr: Optional[Any] = None,
        tool_selector: Optional[Any] = None,
        vertical_context: Optional[Any] = None,
        tiered_config: Optional[Any] = None,
    ) -> None:
        """Set which tools are enabled for this session.

        This is the single source of truth for enabled tools configuration.
        It updates all relevant components: tool selector, vertical context,
        tool access controller, and tiered configuration.

        Args:
            tools: Set of tool names to enable
            session_enabled_tools_attr: Reference to orchestrator's _enabled_tools attr
            tool_selector: ToolSelector instance for propagation
            vertical_context: Vertical context for propagation
            tiered_config: Optional TieredToolConfig to propagate for stage filtering.
                          If None, will attempt to retrieve from active vertical.
        """
        # Update session enabled tools if provided
        if session_enabled_tools_attr is not None:
            session_enabled_tools_attr.clear()
            session_enabled_tools_attr.update(tools)

        # Apply to vertical context
        if vertical_context and hasattr(vertical_context, "enabled_tools"):
            vertical_context.enabled_tools = tools

        # Propagate to tool_selector for selection-time filtering
        if tool_selector:
            tool_selector.set_enabled_tools(tools)
            logger.info(f"Enabled tools filter propagated to selector: {sorted(tools)}")

            # Also propagate TieredToolConfig for stage-aware filtering
            if tiered_config is None:
                # Try to get tiered config from active vertical
                tiered_config = self._get_vertical_tiered_config()
            if tiered_config is not None:
                tool_selector.set_tiered_config(tiered_config)
                logger.info(
                    f"Tiered config propagated to selector: "
                    f"mandatory={sorted(tiered_config.mandatory)}, "
                    f"vertical_core={sorted(tiered_config.vertical_core)}"
                )

    def get_available_tools(self) -> set[str]:
        """Get all registered tool names.

        Returns:
            Set of tool names available in registry
        """
        if self._tool_registry:
            return set(self._tool_registry.list_tools())
        return set()

    def _get_vertical_tiered_config(self) -> Any:
        """Get TieredToolConfig from active vertical if available.

        Returns:
            TieredToolConfig or None
        """
        try:
            from victor.core.verticals.vertical_loader import get_vertical_loader

            loader = get_vertical_loader()
            if loader.active_vertical:
                return loader.active_vertical.get_tiered_tools()
        except Exception as e:
            logger.debug(f"Could not get tiered config from vertical: {e}")
        return None

    # Configuration change detection

    def detect_config_changes(
        self,
        old_config: dict[str, Any],
        new_config: dict[str, Any],
    ) -> dict[str, tuple[Any, Any]]:
        """Detect changes between two configurations.

        Args:
            old_config: Previous configuration
            new_config: New configuration

        Returns:
            Dictionary of changed keys with (old_value, new_value) tuples
        """
        changes = {}
        all_keys = set(old_config.keys()) | set(new_config.keys())

        for key in all_keys:
            old_val = old_config.get(key)
            new_val = new_config.get(key)

            if old_val != new_val:
                changes[key] = (old_val, new_val)

        return changes

    def validate_mode_transition(
        self,
        from_mode: str,
        to_mode: str,
        current_tools: set[str],
    ) -> ValidationResult:
        """Validate that a mode transition is safe given current tools.

        Args:
            from_mode: Current mode name
            to_mode: Target mode name
            current_tools: Currently enabled tools

        Returns:
            ValidationResult with validation status
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check if moving to more restrictive mode
        restrictive_modes = {"plan", "explore"}
        if to_mode.lower() in restrictive_modes and from_mode not in restrictive_modes:
            # Check for tools that will be disabled
            write_tools = {
                "write_file",
                "write",
                "edit_files",
                "edit",
                "shell",
                "bash",
                "execute_bash",
                "git_commit",
                "git_push",
                "delete_file",
            }

            write_tools_enabled = current_tools & write_tools
            if write_tools_enabled:
                warnings.append(
                    f"Transitioning to {to_mode} mode will disable write tools: "
                    f"{sorted(write_tools_enabled)}"
                )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"transition": f"{from_mode} -> {to_mode}"},
        )
