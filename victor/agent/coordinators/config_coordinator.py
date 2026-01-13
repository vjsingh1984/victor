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

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.protocols import IConfigProvider


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
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] | None = None


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
    tool_selection: Dict[str, Any] | None = None
    thinking: bool = False
    profile_name: str | None = None
    metadata: Dict[str, Any] | None = None


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
        providers: Optional[List[IConfigProvider]] = None,
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
        self._config_cache: Dict[str, Dict[str, Any]] = {}

    async def load_config(
        self,
        session_id: str,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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

            # Try each provider and merge results
            for provider in self._providers:
                try:
                    provider_config = await provider.get_config(session_id)
                    if provider_config:
                        # Merge with existing config (later providers override)
                        config = self._deep_merge(config, provider_config)
                except Exception as e:
                    # Log error but continue to next provider
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Config provider {provider.__class__.__name__} failed: {e}"
                    )

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
        config_override: Optional[Dict[str, Any]] = None,
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
        config: Dict[str, Any] | OrchestratorConfig,
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

        # Warnings for non-critical issues
        if temperature > 1.0:
            warnings.append(f"High temperature ({temperature}) may produce unpredictable results")

        if max_tokens > 128000:
            warnings.append(
                f"Very large max_tokens ({max_tokens}) may exceed model context limits"
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
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
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

    async def get_config(self, session_id: str) -> Dict[str, Any]:
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

    async def get_config(self, session_id: str) -> Dict[str, Any]:
        """Get configuration from environment variables.

        Args:
            session_id: Session identifier

        Returns:
            Configuration dictionary from environment
        """
        import os

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


__all__ = [
    "ConfigCoordinator",
    "ValidationResult",
    "OrchestratorConfig",
    "SettingsConfigProvider",
    "EnvironmentConfigProvider",
]
