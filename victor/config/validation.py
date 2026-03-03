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

"""Configuration validation for early error detection.

This module provides validation functions that check configuration at startup
rather than at runtime, providing clear, actionable error messages to users.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Validation Result
# =============================================================================


class ValidationError:
    """Represents a configuration validation error with helpful context."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        suggestion: str | None = None,
        error_code: str | None = None,
    ):
        self.message = message
        self.field = field
        self.suggestion = suggestion
        self.error_code = error_code
        # Severity is not stored, just used during validation
        self._severity = None  # type: str | None

    def __str__(self) -> str:
        parts = [f"✗ {self.message}"]
        if self.field:
            parts.append(f"\n  Field: {self.field}")
        if self.suggestion:
            parts.append(f"\n  💡 Suggestion: {self.suggestion}")
        if self.error_code:
            parts.append(f"\n  Error Code: {self.error_code}")
        return "\n".join(parts)


class ValidationResult:
    """Result of configuration validation."""

    def __init__(self, is_valid: bool, errors: List[ValidationError] | None = None):
        self.is_valid = is_valid
        self.errors = errors or []

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def add_error(self, error: ValidationError) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False


# =============================================================================
# Validation Checks
# =============================================================================


def validate_environment(settings: "Settings" | None = None) -> ValidationResult:
    """Validate environment variables and system configuration.

    Args:
        settings: Optional Settings instance for provider-aware validation

    Returns:
        ValidationResult with any environment issues
    """
    result = ValidationResult(True)

    # Check Python version
    if sys.version_info < (3, 10):
        result.add_error(ValidationError(
            message=f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported",
            suggestion="Python 3.10 or higher is required. "
            "Try: python3.10 -m pip install victor-ai or upgrade your Python version.",
            error_code="PYTHON_VERSION",
        ))

    # Check for deprecated configuration
    deprecated_vars = [
        "VICTOR_API_KEY",
        "OPENAI_API_BASE",
        "ANTHROPIC_API_BASE",
    ]
    found_deprecated = [var for var in deprecated_vars if os.getenv(var)]
    if found_deprecated:
        result.add_error(ValidationError(
            message=f"Deprecated environment variable(s) found: {', '.join(found_deprecated)}",
            suggestion="Remove deprecated variables. Use provider-specific environment variables "
            "(e.g., ANTHROPIC_API_KEY) instead.",
            error_code="DEPRECATED_ENV_VAR",
        ))

    # Check for API keys (only if using cloud providers)
    if settings and hasattr(settings, 'default_provider'):
        provider = settings.default_provider
        local_providers = ["ollama", "lmstudio", "vllm", "llama.cpp"]

        # Only check for API keys if not using a local provider
        if provider and provider.lower() not in local_providers:
            has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
            has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

            if not has_anthropic_key and not has_openai_key:
                result.add_error(ValidationError(
                    message="No API key found for cloud provider",
                    suggestion="Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable. "
                    "For local models, use --provider ollama (requires Ollama to be running).",
                    error_code="NO_API_KEY",
                ))

    return result


def validate_paths(settings: "Settings") -> ValidationResult:
    """Validate file system paths in configuration.

    Args:
        settings: Settings instance to validate

    Returns:
        ValidationResult with any path issues
    """
    result = ValidationResult(True)

    # Get config directory (victor config directory)
    try:
        config_dir = settings.get_config_dir()
        if config_dir and not config_dir.exists():
            result.add_error(ValidationError(
                message=f"Configuration directory does not exist: {config_dir}",
                suggestion="Run 'victor init' to create configuration or set VICTOR_CONFIG_DIR to a valid location.",
                field="VICTOR_CONFIG_DIR",
                error_code="CONFIG_DIR_NOT_FOUND",
            ))
    except Exception as e:
        logger.debug(f"Could not validate config directory: {e}")

    # Validate codebase persist directory if specified
    if hasattr(settings, 'codebase_persist_directory') and settings.codebase_persist_directory:
        persist_path = Path(settings.codebase_persist_directory)
        if not persist_path.exists():
            try:
                persist_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                result.add_error(ValidationError(
                    message=f"Cannot create codebase persist directory: {persist_path}",
                    suggestion=f"Check permissions for {persist_path.parent} or set codebase_persist_directory "
                    f"to a writable location. Error: {e}",
                    field="codebase_persist_directory",
                    error_code="PERSIST_DIR_PERMISSIONS",
                ))

    return result


def validate_provider_settings(settings: "Settings") -> ValidationResult:
    """Validate provider configuration.

    Args:
        settings: Settings instance to validate

    Returns:
        ValidationResult with any provider issues
    """
    result = ValidationResult(True)

    # Check if default provider is configured
    default_provider = settings.default_provider
    if not default_provider:
        result.add_error(ValidationError(
            message="No default provider configured",
            suggestion="Set the 'default_provider' setting in your profile or use --provider flag. "
            "Available: anthropic, openai, ollama, lmstudio, vllm, and 18 more.",
            field="default_provider",
            error_code="NO_DEFAULT_PROVIDER",
        ))
        return result  # Cannot validate further without provider

    # Check for local providers that require additional setup
    local_providers = ["ollama", "lmstudio", "vllm", "llama.cpp"]
    if default_provider.lower() in local_providers:
        # Check if the provider is running/accessible
        if default_provider.lower() == "ollama":
            try:
                import subprocess

                result_check = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    timeout=2,
                )
                if result_check.returncode != 0:
                    result.add_error(ValidationError(
                        message="Ollama provider selected but Ollama is not running",
                        suggestion="Start Ollama: 'ollama serve' or 'brew services start ollama'. "
                        "Verify with: ollama list",
                        field="default_provider",
                        error_code="OLLAMA_NOT_RUNNING",
                    ))
            except FileNotFoundError:
                result.add_error(ValidationError(
                    message="Ollama provider selected but Ollama is not installed",
                    suggestion="Install Ollama: 'curl -fsSL https://ollama.com/install.sh | sh' "
                    "or use: brew install ollama",
                    field="default_provider",
                    error_code="OLLAMA_NOT_INSTALLED",
                ))
            except Exception as e:
                logger.debug(f"Could not check Ollama status: {e}")

    # Check API keys for cloud providers
    cloud_providers = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "azure": "AZURE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "bedrock": "AWS_ACCESS_KEY_ID",
    }

    if default_provider.lower() in cloud_providers:
        env_var = cloud_providers.get(default_provider.lower())
        if env_var and not os.getenv(env_var):
            result.add_error(ValidationError(
                message=f"{default_provider.title()} provider selected but API key not found",
                suggestion=f"Set {env_var} environment variable or configure in profile YAML. "
                f"Example: export {env_var}=your_key_here",
                field=f"default_provider.{default_provider}",
                error_code="API_KEY_MISSING",
            ))

    return result


def validate_tool_settings(settings: "Settings") -> ValidationResult:
    """Validate tool-related configuration.

    Args:
        settings: Settings Settings instance to validate

    Returns:
        ValidationResult with any tool issues
    """
    result = ValidationResult(True)

    # Validate fallback_max_tools
    if settings.fallback_max_tools < 1:
        result.add_error(ValidationError(
            message=f"fallback_max_tools must be at least 1, got {settings.fallback_max_tools}",
            suggestion="Set fallback_max_tools to 1 or higher in your profile YAML.",
            field="fallback_max_tools",
            error_code="INVALID_TOOL_BUDGET",
        ))

    if settings.fallback_max_tools > 20:
        result.add_error(ValidationError(
            message=f"fallback_max_tools is very high: {settings.fallback_max_tools}",
            suggestion="Consider reducing fallback_max_tools to 10 or fewer for better performance. "
            "Large tool budgets can lead to slower response times.",
            field="fallback_max_tools",
            error_code="HIGH_TOOL_BUDGET",
        ))

    # Validate cache settings
    if settings.tool_selection_cache_ttl < 0:
        result.add_error(ValidationError(
            message=f"tool_selection_cache_ttl cannot be negative: {settings.tool_selection_cache_ttl}",
            suggestion="Set tool_selection_cache_ttl to 0 or higher in your profile YAML.",
            field="tool_selection_cache_ttl",
            error_code="INVALID_CACHE_TTL",
        ))

    return result


def validate_performance_settings(settings: "Settings") -> ValidationResult:
    """Validate performance-related settings.

    Args:
        settings: Settings Settings instance to validate

    Returns:
        ValidationResult with any performance issues
    """
    result = ValidationResult(True)

    # Check if optimizations are enabled (warnings only, not errors)
    try:
        if not settings.framework_preload_enabled:
            result.add_error(ValidationError(
                message="Framework preloading is disabled",
                suggestion="Enable framework_preload in your profile YAML for 50-70% faster first requests. "
                "This adds a small startup cost but dramatically improves first-use experience.",
                field="framework_preload_enabled",
                error_code="PRELOAD_DISABLED",
            ))

        if not settings.http_connection_pool_enabled:
            result.add_error(ValidationError(
                message="HTTP connection pooling is disabled",
                suggestion="Enable http_connection_pool_enabled in your profile YAML for 20-30% "
                "faster HTTP requests to web search and API providers.",
                field="http_connection_pool_enabled",
                error_code="HTTP_POOL_DISABLED",
            ))

        if not settings.tool_selection_cache_enabled:
            result.add_error(ValidationError(
                message="Tool selection cache is disabled",
                suggestion="Enable tool_selection_cache_enabled in your profile YAML for 20-40% faster "
                "conversational responses (semantic search caching).",
                field="tool_selection_cache_enabled",
                error_code="TOOL_CACHE_DISABLED",
            ))
    except Exception as e:
        # Settings might not have these attributes in older versions
        logger.debug(f"Could not validate performance settings: {e}")

    return result


# =============================================================================
# Main Validation Entry Point
# =============================================================================


def validate_configuration(settings: "Settings" | None = None) -> ValidationResult:
    """Perform comprehensive configuration validation.

    This function checks all aspects of the configuration and provides
    clear, actionable error messages for any issues found.

    Args:
        settings: Optional Settings instance. If None, will load default settings.

    Returns:
        ValidationResult indicating whether configuration is valid and any errors found
    """
    if settings is None:
        try:
            from victor.config.settings import Settings

            settings = Settings()
        except Exception as e:
            # Settings creation failed catastrophically
            return ValidationResult(
                is_valid=False,
                errors=[
                    ValidationError(
                        message=f"Failed to load settings: {e}",
                        suggestion="Check your profiles.yaml and environment variables. "
                        "Run 'victor config validate' for detailed diagnostics.",
                        error_code="SETTINGS_LOAD_FAILED",
                    )
                ],
            )

    # Run all validation checks
    result = ValidationResult(True)

    # Environment checks (pass settings for provider-aware validation)
    env_result = validate_environment(settings)
    result.errors.extend(env_result.errors)

    # Path checks
    path_result = validate_paths(settings)
    result.errors.extend(path_result.errors)

    # Provider checks
    provider_result = validate_provider_settings(settings)
    result.errors.extend(provider_result.errors)

    # Tool settings
    tool_result = validate_tool_settings(settings)
    result.errors.extend(tool_result.errors)

    # Performance settings
    perf_result = validate_performance_settings(settings)
    result.errors.extend(perf_result.errors)

    # Separate warnings from errors
    errors = []
    warnings = []

    for error in result.errors:
        # Check if this is a warning (has severity="warning" in suggestion or error_code in ("PRELOAD_DISABLED", "HTTP_POOL_DISABLED", "TOOL_CACHE_DISABLED"))
        if "warning" in str(error.suggestion or "").lower() or error.error_code in ("PRELOAD_DISABLED", "HTTP_POOL_DISABLED", "TOOL_CACHE_DISABLED"):
            warnings.append(error)
        else:
            errors.append(error)

    # Update result with only errors (warnings are informational)
    result.errors = errors
    result.is_valid = len(errors) == 0

    return result


def format_validation_result(result: ValidationResult) -> str:
    """Format validation result for user display.

    Args:
        result: ValidationResult from validate_configuration

    Returns:
        Formatted string suitable for terminal display
    """
    lines = []

    if result.is_valid:
        lines.append("✓ Configuration is valid!")
    else:
        lines.append(f"✗ Configuration validation failed ({result.error_count} error(s):")
        lines.append("")
        for i, error in enumerate(result.errors, 1):
            lines.append(f"\n{i}. {error.message}")
            if error.field:
                lines.append(f"   Field: {error.field}")
            if error.suggestion:
                lines.append(f"   💡 {error.suggestion}")
            if error.error_code:
                lines.append(f"   Code: {error.error_code}")

    return "\n".join(lines)


# =============================================================================
# CLI Command
# =============================================================================


def validate_command() -> int:
    """Run validation and exit with appropriate status code.

    This is the entry point for 'victor config validate' command.

    Returns:
        Exit code (0 for success, 1 for validation failure)
    """
    try:
        from victor.config.settings import Settings, load_settings

        settings = load_settings()
        result = validate_configuration(settings)

        print(format_validation_result(result))

        # Return exit code
        if result.is_valid:
            return 0
        else:
            return 1

    except Exception as e:
        print(f"✗ Validation failed with unexpected error: {e}")
        logger.exception("Configuration validation failed")
        return 1
