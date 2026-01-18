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

"""Shared provider configuration and API key resolution.

This module provides utilities for provider configuration management,
extracting common patterns from BaseProvider._resolve_api_key().
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a provider.

    Attributes:
        api_key: The resolved API key (may be empty string)
        base_url: Optional base URL for API endpoints
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        extra_config: Additional provider-specific configuration
    """

    api_key: str
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            **self.extra_config,
        }


# Common environment variable name patterns for providers
# Maps provider name to possible environment variable names
PROVIDER_ENV_VAR_PATTERNS: Dict[str, List[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "groq": ["GROQ_API_KEY", "GROQCLOUD_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "moonshot": ["MOONSHOT_API_KEY"],
    "cerebras": ["CEREBRAS_API_KEY"],
    "fireworks": ["FIREWORKS_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "replicate": ["REPLICATE_API_TOKEN"],
    "huggingface": ["HUGGINGFACE_API_KEY", "HF_TOKEN"],
    "mistral": ["MISTRAL_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY", "PALM_API_KEY"],
    "xai": ["XAI_API_KEY", "GROK_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "vertex": ["VERTEX_AI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"],
    "azure": ["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"],
    "zai": ["ZAI_API_KEY"],
    "ollama": [],  # Local, no API key
    "lmstudio": [],  # Local, no API key
    "vllm": [],  # Local, no API key
    "llamacpp": [],  # Local, no API key
}


def resolve_api_key(
    api_key: Optional[str],
    provider_name: str,
    *,
    env_var_names: Optional[List[str]] = None,
    use_keyring: bool = True,
    allow_empty: bool = True,
    log_warning: bool = True,
) -> str:
    """Resolve API key from multiple sources with proper fallback order.

    Resolution order:
    1. Provided api_key parameter (highest priority)
    2. Environment variables ({PROVIDER_NAME}_API_KEY and variants)
    3. Keyring (via victor.config.api_keys.get_api_key)
    4. Empty string (with optional warning logged)

    Args:
        api_key: API key provided as parameter (highest priority)
        provider_name: Name of the provider (e.g., "anthropic", "openai")
        env_var_names: Optional list of environment variable names to check.
            If not provided, uses standard patterns from PROVIDER_ENV_VAR_PATTERNS.
        use_keyring: Whether to check keyring as fallback (default: True)
        allow_empty: Whether to allow empty API key (default: True)
        log_warning: Whether to log warning when API key not found (default: True)

    Returns:
        Resolved API key (empty string if not found and allow_empty=True)

    Raises:
        ValueError: If allow_empty=False and no API key is found

    Example:
        # Use default patterns
        api_key = resolve_api_key(None, "openai")

        # Custom environment variable names
        api_key = resolve_api_key(None, "myprovider", env_var_names=["MYPROVIDER_KEY"])

        # Require API key (raise if not found)
        api_key = resolve_api_key(None, "openai", allow_empty=False)
    """
    # 1. Use provided key if available (including empty string for explicit empty)
    # Note: We check `is not None` to allow empty string as a valid value
    if api_key is not None:
        return api_key

    # 2. Try environment variables
    env_var_name: Optional[str] = None
    env_key: Optional[str] = None

    if env_var_names is None:
        # Use standard patterns for this provider
        env_var_names = PROVIDER_ENV_VAR_PATTERNS.get(
            provider_name.lower(),
            [f"{provider_name.upper()}_API_KEY"],
        )

    for var_name in env_var_names:
        env_key = os.environ.get(var_name, "")
        if env_key:
            env_var_name = var_name
            logger.debug(f"Using {env_var_name} for {provider_name}")
            return env_key

    # 3. Try keyring
    if use_keyring:
        try:
            from victor.config.api_keys import get_api_key

            keyring_key = get_api_key(provider_name)
            if keyring_key:
                logger.debug(f"Using keyring for {provider_name}")
                return keyring_key
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Keyring lookup failed for {provider_name}: {e}")

    # 4. Not found
    if log_warning or not allow_empty:
        # Build the primary env var name for the warning
        primary_env_var = env_var_names[0] if env_var_names else f"{provider_name.upper()}_API_KEY"

        if not allow_empty:
            raise ValueError(
                f"{provider_name.capitalize()} API key not found. "
                f"Set {primary_env_var} environment variable, "
                f"use 'victor keys --set {provider_name} --keyring', "
                f"or pass api_key parameter."
            )

        logger.warning(
            "%s API key not provided. Set %s environment variable, "
            "use 'victor keys --set %s --keyring', or pass api_key parameter.",
            provider_name.capitalize(),
            primary_env_var,
            provider_name,
        )

    return ""


def create_provider_config(
    provider_name: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 3,
    env_var_names: Optional[List[str]] = None,
    **extra_config: Any,
) -> ProviderConfig:
    """Create a ProviderConfig with resolved API key.

    This is a convenience function that combines API key resolution
    with configuration creation.

    Args:
        provider_name: Name of the provider
        api_key: Optional API key (will be resolved if not provided)
        base_url: Optional base URL
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        env_var_names: Optional list of env var names for API key lookup
        **extra_config: Additional provider-specific configuration

    Returns:
        ProviderConfig with resolved API key

    Example:
        config = create_provider_config("openai", timeout=120)
        provider = OpenAIProvider(**config.to_dict())
    """
    resolved_key = resolve_api_key(api_key, provider_name, env_var_names=env_var_names)

    return ProviderConfig(
        api_key=resolved_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        extra_config=extra_config,
    )


def get_env_var_names_for_provider(provider_name: str) -> List[str]:
    """Get the standard environment variable names for a provider.

    Args:
        provider_name: Name of the provider

    Returns:
        List of environment variable names to check, in priority order

    Example:
        >>> get_env_var_names_for_provider("openai")
        ['OPENAI_API_KEY']
        >>> get_env_var_names_for_provider("groq")
        ['GROQ_API_KEY', 'GROQCLOUD_API_KEY']
    """
    return PROVIDER_ENV_VAR_PATTERNS.get(
        provider_name.lower(),
        [f"{provider_name.upper()}_API_KEY"],
    )


def register_provider_env_patterns(patterns: Dict[str, List[str]]) -> None:
    """Register custom environment variable patterns for providers.

    This allows external plugins to register their own env var patterns.

    Args:
        patterns: Dictionary mapping provider names to env var name lists

    Example:
        register_provider_env_patterns({
            "myprovider": ["MYPROVIDER_API_KEY", "MYPROVIDER_TOKEN"]
        })
    """
    for provider_name, var_names in patterns.items():
        PROVIDER_ENV_VAR_PATTERNS[provider_name.lower()] = var_names


def is_local_provider(provider_name: str) -> bool:
    """Check if a provider is local (doesn't require API key).

    Args:
        provider_name: Name of the provider

    Returns:
        True if provider is local and doesn't need an API key
    """
    return provider_name.lower() in {
        "ollama",
        "lmstudio",
        "vllm",
        "llamacpp",
    }


def needs_api_key(provider_name: str) -> bool:
    """Check if a provider requires an API key.

    Args:
        provider_name: Name of the provider

    Returns:
        True if provider requires an API key for authentication
    """
    return not is_local_provider(provider_name)
