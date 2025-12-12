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

"""Configuration loading and validation for the agent."""

import logging
import os
from typing import Any, Dict, List, Optional, Set


from victor.config.settings import Settings
from victor.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


def _get_critical_tools(registry: Optional["ToolRegistry"] = None) -> Set[str]:
    """Get critical tools using dynamic discovery from tool_selection module.

    Critical tools are those with priority=Priority.CRITICAL in their @tool decorator.
    """
    from victor.agent.tool_selection import get_critical_tools

    return get_critical_tools(registry)


class ConfigLoader:
    """Loads and validates agent configuration.

    Responsibilities:
    - Load tool configurations from profiles.yaml
    - Validate tool names against registry
    - Warn about disabled core tools
    - Resolve environment variables in configuration
    """

    def __init__(self, settings: Settings):
        """Initialize config loader.

        Args:
            settings: Application settings
        """
        self.settings = settings

    def load_tool_config(self, tool_registry: ToolRegistry) -> None:
        """Load and apply tool configurations.

        Loads tool enable/disable states from the 'tools' section in profiles.yaml.

        Expected formats:
        ```yaml
        tools:
          enabled:
            - read_file
            - write_file
          disabled:
            - code_review
        ```

        Or:
        ```yaml
        tools:
          code_review:
            enabled: false
        ```

        Args:
            tool_registry: Registry to configure
        """
        try:
            tool_config = self.settings.load_tool_config()
            if not tool_config:
                return

            registered_tools = {tool.name for tool in tool_registry.list_tools(only_enabled=False)}

            # Format 1: Lists of enabled/disabled tools
            self._apply_enabled_list(tool_config, tool_registry, registered_tools)
            self._apply_disabled_list(tool_config, tool_registry, registered_tools)

            # Format 2: Individual tool settings
            self._apply_individual_settings(tool_config, tool_registry, registered_tools)

            # Log final state
            self._log_tool_states(tool_registry)

        except Exception as e:
            logger.error(f"Failed to load tool configuration: {e}")

    def _apply_enabled_list(
        self, config: Dict[str, Any], registry: ToolRegistry, registered: Set[str]
    ) -> None:
        """Apply enabled list from configuration."""
        if "enabled" not in config:
            return

        enabled_tools = config.get("enabled", [])

        # Validate tool names
        invalid = [t for t in enabled_tools if t not in registered]
        if invalid:
            logger.warning(
                f"Invalid tool names in 'enabled' list: {', '.join(invalid)}. "
                f"Available: {', '.join(sorted(registered))}"
            )

        # Warn about missing core tools (dynamically discovered)
        core_tools = _get_critical_tools(registry)
        missing_core = core_tools - set(enabled_tools)
        if missing_core:
            logger.warning(
                f"'enabled' list missing core tools: {', '.join(missing_core)}. "
                f"This may limit agent functionality."
            )

        # Disable all, then enable specified
        for tool in registry.list_tools(only_enabled=False):
            registry.disable_tool(tool.name)

        for tool_name in enabled_tools:
            if tool_name in registered:
                registry.enable_tool(tool_name)

    def _apply_disabled_list(
        self, config: Dict[str, Any], registry: ToolRegistry, registered: Set[str]
    ) -> None:
        """Apply disabled list from configuration."""
        if "disabled" not in config:
            return

        disabled_tools = config.get("disabled", [])

        # Validate tool names
        invalid = [t for t in disabled_tools if t not in registered]
        if invalid:
            logger.warning(
                f"Invalid tool names in 'disabled' list: {', '.join(invalid)}. "
                f"Available: {', '.join(sorted(registered))}"
            )

        # Warn about disabling core tools (dynamically discovered)
        core_tools = _get_critical_tools(registry)
        disabled_core = core_tools & set(disabled_tools)
        if disabled_core:
            logger.warning(
                f"Disabling core tools: {', '.join(disabled_core)}. "
                f"This may limit agent functionality."
            )

        for tool_name in disabled_tools:
            if tool_name in registered:
                registry.disable_tool(tool_name)

    def _apply_individual_settings(
        self, config: Dict[str, Any], registry: ToolRegistry, registered: Set[str]
    ) -> None:
        """Apply individual tool settings from configuration."""
        for tool_name, tool_config in config.items():
            if not isinstance(tool_config, dict) or "enabled" not in tool_config:
                continue

            if tool_name not in registered:
                logger.warning(
                    f"Invalid tool name: '{tool_name}'. "
                    f"Available: {', '.join(sorted(registered))}"
                )
                continue

            if tool_config["enabled"]:
                registry.enable_tool(tool_name)
            else:
                registry.disable_tool(tool_name)
                # Check if this is a core tool (dynamically discovered)
                core_tools = _get_critical_tools(registry)
                if tool_name in core_tools:
                    logger.warning(
                        f"Disabling core tool '{tool_name}'. "
                        f"This may limit agent functionality."
                    )

    def _log_tool_states(self, registry: ToolRegistry) -> None:
        """Log the final tool enabled/disabled states."""
        disabled = [name for name, enabled in registry.get_tool_states().items() if not enabled]
        if disabled:
            logger.info(f"Disabled tools: {', '.join(sorted(disabled))}")

    @staticmethod
    def resolve_env_vars(value: str) -> str:
        """Resolve environment variables in a string value.

        Supports ${VAR} and ${VAR:-default} syntax.

        Args:
            value: String that may contain env var references

        Returns:
            String with env vars resolved
        """
        import re

        def replacer(match: re.Match) -> str:
            var_expr = match.group(1)
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return os.environ.get(var_name, default)
            return os.environ.get(var_expr, "")

        return re.sub(r"\$\{([^}]+)\}", replacer, value)

    @staticmethod
    def resolve_endpoint_list(
        endpoints: List[str], env_var_prefix: Optional[str] = None
    ) -> List[str]:
        """Resolve and filter endpoint list.

        Args:
            endpoints: List of endpoint URLs (may contain env vars)
            env_var_prefix: Optional env var to prepend additional endpoints

        Returns:
            List of resolved, non-empty endpoint URLs
        """
        resolved: List[str] = []

        # Add env var endpoints first if specified
        if env_var_prefix:
            env_endpoints = os.environ.get(env_var_prefix, "")
            if env_endpoints:
                resolved.extend(ep.strip() for ep in env_endpoints.split(",") if ep.strip())

        # Resolve each configured endpoint
        for endpoint in endpoints:
            resolved_ep = ConfigLoader.resolve_env_vars(endpoint)
            if resolved_ep.strip():
                resolved.append(resolved_ep.strip())

        return resolved


# =============================================================================
# PROFILE VALIDATOR
# =============================================================================
# Validates profile configurations at startup to catch common issues early:
# - Model name normalization (qwen25 → qwen2.5)
# - Tool capability detection
# - Ollama model availability (optional)
# =============================================================================


class ProfileValidationResult:
    """Result of profile validation."""

    def __init__(
        self,
        profile_name: str,
        is_valid: bool = True,
        warnings: Optional[List[str]] = None,
        errors: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.profile_name = profile_name
        self.is_valid = is_valid
        self.warnings = warnings or []
        self.errors = errors or []
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        lines = [f"Profile '{self.profile_name}': {'VALID' if self.is_valid else 'INVALID'}"]
        if self.errors:
            lines.append(f"  Errors: {', '.join(self.errors)}")
        if self.warnings:
            lines.append(f"  Warnings: {', '.join(self.warnings)}")
        if self.suggestions:
            lines.append(f"  Suggestions: {', '.join(self.suggestions)}")
        return "\n".join(lines)


class ProfileValidator:
    """Validates profile configurations at startup.

    Performs the following checks:
    1. Model name normalization - applies aliases (qwen25 → qwen2.5)
    2. Tool capability detection - checks if model supports native tool calls
    3. Ollama model availability - verifies model is available (optional async check)

    Usage:
        validator = ProfileValidator(settings)
        results = validator.validate_profiles()
        for result in results:
            if not result.is_valid:
                print(result)
    """

    def __init__(self, settings: Settings):
        """Initialize profile validator.

        Args:
            settings: Application settings with profile configurations
        """
        self.settings = settings
        self._capability_loader = None

    def _get_capability_loader(self):
        """Lazy-load the ModelCapabilityLoader to avoid circular imports."""
        if self._capability_loader is None:
            from victor.agent.tool_calling.capabilities import ModelCapabilityLoader

            self._capability_loader = ModelCapabilityLoader()
        return self._capability_loader

    def validate_profile(
        self,
        profile_name: str,
        provider: str,
        model: str,
        check_ollama_availability: bool = False,
    ) -> ProfileValidationResult:
        """Validate a single profile configuration.

        Args:
            profile_name: Name of the profile
            provider: Provider name (ollama, anthropic, openai, etc.)
            model: Model identifier
            check_ollama_availability: If True, check Ollama model availability

        Returns:
            ProfileValidationResult with validation status and messages
        """
        from victor.agent.tool_calling.capabilities import (
            normalize_model_name,
            get_model_name_variants,
        )

        warnings: List[str] = []
        errors: List[str] = []
        suggestions: List[str] = []

        # Step 1: Check model name normalization
        normalized = normalize_model_name(model)
        if normalized != model.lower():
            suggestions.append(
                f"Model name normalized: '{model}' → '{normalized}'. "
                f"Consider using '{normalized}' in your profile for consistency."
            )

        # Step 2: Check tool capability detection
        loader = self._get_capability_loader()
        variants = get_model_name_variants(model)

        # Try to get capabilities with any variant
        capabilities = None
        matched_variant = None
        for variant in variants:
            caps = loader.get_capabilities(provider, variant)
            if caps is not None:
                capabilities = caps
                matched_variant = variant
                break

        if capabilities is None:
            warnings.append(
                f"Model '{model}' not found in capability patterns for provider '{provider}'. "
                f"Tool calling may be disabled."
            )
            suggestions.append(
                f"Add a pattern for '{normalized}*' to model_capabilities.yaml, or use "
                f"--provider ollama --model with a known tool-capable model."
            )
        elif not capabilities.native_tool_calls:
            warnings.append(
                f"Model '{matched_variant or model}' matched but does not support native tool calls. "
                f"Using fallback text parsing."
            )
        else:
            logger.debug(
                f"Profile '{profile_name}': Model '{matched_variant or model}' supports native tool calls"
            )

        # Step 3: Check Ollama availability (optional, async)
        if check_ollama_availability and provider == "ollama":
            ollama_warning = self._check_ollama_model(model)
            if ollama_warning:
                warnings.append(ollama_warning)

        is_valid = len(errors) == 0
        return ProfileValidationResult(
            profile_name=profile_name,
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            suggestions=suggestions,
        )

    def _check_ollama_model(self, model: str) -> Optional[str]:
        """Check if Ollama model is available (sync check).

        This is a lightweight check using environment-configured OLLAMA_HOST.

        Args:
            model: Model name to check

        Returns:
            Warning message if model not available, None otherwise
        """
        import urllib.request
        import urllib.error
        import json

        ollama_host = os.environ.get("OLLAMA_HOST", self.settings.ollama_base_url)

        try:
            url = f"{ollama_host}/api/tags"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                models = data.get("models", [])
                model_names = [m.get("name", "").lower() for m in models]

                # Check exact match or prefix match
                model_lower = model.lower()
                for name in model_names:
                    if name == model_lower or name.startswith(f"{model_lower}:"):
                        return None

                # Model not found
                return (
                    f"Model '{model}' not found on Ollama server at {ollama_host}. "
                    f"Available models: {', '.join(sorted(set(m.split(':')[0] for m in model_names))[:5])}..."
                )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            logger.debug(f"Could not check Ollama availability: {e}")
            return None  # Don't warn if we can't connect

    def validate_all_ollama_profiles(
        self, check_availability: bool = False
    ) -> List[ProfileValidationResult]:
        """Validate all Ollama profiles in settings.

        Args:
            check_availability: If True, check if models are available on Ollama server

        Returns:
            List of ProfileValidationResult for each Ollama profile
        """
        results: List[ProfileValidationResult] = []

        # Get profiles from settings if available
        profiles_yaml = getattr(self.settings, "_profiles_yaml", None)
        if not profiles_yaml or "profiles" not in profiles_yaml:
            return results

        for profile_name, profile_config in profiles_yaml.get("profiles", {}).items():
            if not isinstance(profile_config, dict):
                continue

            provider = profile_config.get("provider", "").lower()
            model = profile_config.get("model", "")

            if provider == "ollama" and model:
                result = self.validate_profile(
                    profile_name=profile_name,
                    provider=provider,
                    model=model,
                    check_ollama_availability=check_availability,
                )
                results.append(result)

        return results

    def log_validation_summary(self, results: List[ProfileValidationResult]) -> None:
        """Log a summary of validation results.

        Args:
            results: List of validation results to summarize
        """
        if not results:
            return

        warnings_count = sum(len(r.warnings) for r in results)
        errors_count = sum(len(r.errors) for r in results)

        if errors_count > 0:
            logger.error(f"Profile validation found {errors_count} errors, {warnings_count} warnings")
            for result in results:
                if result.errors:
                    logger.error(str(result))
        elif warnings_count > 0:
            logger.warning(f"Profile validation found {warnings_count} warnings")
            for result in results:
                if result.warnings:
                    logger.warning(str(result))


def validate_profiles_on_startup(settings: Settings, check_availability: bool = False) -> None:
    """Convenience function to validate profiles at startup.

    Call this from the CLI entrypoint to catch configuration issues early.

    Args:
        settings: Application settings
        check_availability: If True, also check Ollama model availability
    """
    validator = ProfileValidator(settings)
    results = validator.validate_all_ollama_profiles(check_availability=check_availability)
    validator.log_validation_summary(results)
