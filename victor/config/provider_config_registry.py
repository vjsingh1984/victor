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

"""Provider configuration registry - OCP-compliant provider settings.

This module implements the Strategy pattern for provider-specific configuration,
eliminating the OCP violation in Settings.get_provider_settings().

Design Goals:
- Open for extension: Add new providers by registering a strategy
- Closed for modification: No changes to core logic needed
- Single Responsibility: Each strategy handles one provider
- Simplified: DefaultProviderConfig handles most simple API key providers

Migration Notes:
- Simple API key providers now use DefaultProviderConfig
- Complex providers (OAuth, multiple endpoints) use dedicated strategies
- Legacy strategy classes are deprecated but kept for backward compatibility

Usage:
    from victor.config.provider_config_registry import (
        get_provider_config_registry,
        register_provider_config,
    )

    # Get settings for a provider
    registry = get_provider_config_registry()
    settings = registry.get_settings("anthropic", settings_instance)

    # Register a custom provider
    @register_provider_config("my_provider")
    class MyProviderConfig(ProviderConfigStrategy):
        def get_settings(self, settings: "Settings") -> Dict[str, Any]:
            return {"api_key": get_api_key("my_provider"), "base_url": "..."}
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

if TYPE_CHECKING:
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class ProviderConfigStrategy(ABC):
    """Abstract base for provider-specific configuration strategies.

    Each provider implements this interface to define how its
    settings are resolved (API keys, base URLs, etc.).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The provider name this strategy handles."""
        ...

    @property
    def aliases(self) -> List[str]:
        """Alternative names that map to this provider."""
        return []

    @abstractmethod
    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get provider-specific settings.

        Args:
            settings: The Settings instance for accessing config
            base_settings: Settings already loaded from profiles.yaml

        Returns:
            Dictionary of provider settings (api_key, base_url, etc.)
        """
        ...


# =============================================================================
# Built-in Provider Strategies
# =============================================================================

# Default endpoints for simple API key providers
# These providers use API key auth with a single endpoint
DEFAULT_PROVIDER_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "xai": "https://api.x.ai/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "zai": "https://api.z.ai/api/paas/v4/",
    "groqcloud": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "huggingface": "https://api-inference.huggingface.co",
    "replicate": "https://api.replicate.com/v1",
}


class DefaultProviderConfig(ProviderConfigStrategy):
    """Default configuration strategy for simple API key providers.

    This strategy handles providers that:
    - Use API key authentication
    - Have a single endpoint
    - Don't require special configuration

    This eliminates the need for individual strategy classes for each provider.
    """

    def __init__(self, provider_name: str, base_url: Optional[str] = None):
        """Initialize default provider config.

        Args:
            provider_name: The provider name
            base_url: Optional base URL (uses default if not specified)
        """
        self._provider_name = provider_name
        self._base_url = base_url or DEFAULT_PROVIDER_ENDPOINTS.get(provider_name)

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get provider settings using default resolution."""
        from victor.config.api_keys import get_api_key

        result = dict(base_settings)

        # Try to get API key from settings attribute first (for backwards compat)
        settings_attr = f"{self._provider_name}_api_key"
        raw_key = getattr(settings, settings_attr, None)
        if raw_key and hasattr(raw_key, "get_secret_value"):
            api_key = raw_key.get_secret_value()
        else:
            api_key = raw_key or get_api_key(self._provider_name)

        if api_key:
            result["api_key"] = api_key

        # Set base URL if available
        if self._base_url:
            result.setdefault("base_url", self._base_url)

        return result


# =============================================================================
# Complex Provider Strategies (OAuth, Multiple Endpoints, URL Detection)
# =============================================================================

# NOTE: Simple provider strategies below are deprecated.
# Use DefaultProviderConfig instead for new providers.
# Existing strategies kept for backward compatibility.


class AnthropicConfig(ProviderConfigStrategy):
    """Configuration strategy for Anthropic Claude.

    DEPRECATED: Use DefaultProviderConfig instead.
    This strategy is kept for backward compatibility.
    """

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Delegate to default strategy
        return DefaultProviderConfig("anthropic").get_settings(settings, base_settings)


class OpenAIConfig(ProviderConfigStrategy):
    """Configuration strategy for OpenAI."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        from victor.config.api_keys import get_api_key

        result = dict(base_settings)
        auth_mode = base_settings.get("auth_mode", "api_key")

        if auth_mode == "oauth":
            # OAuth mode: use ChatGPT subscription
            result["auth_mode"] = "oauth"
            result.setdefault("base_url", "https://chatgpt.com/backend-api/codex/v1")
        else:
            # API key mode
            raw_key = settings.openai_api_key
            api_key = (raw_key.get_secret_value() if raw_key else None) or get_api_key("openai")
            if api_key:
                result["api_key"] = api_key
            result.setdefault("base_url", "https://api.openai.com/v1")

        return result


class GoogleConfig(ProviderConfigStrategy):
    """Configuration strategy for Google/Gemini.

    DEPRECATED: Use DefaultProviderConfig instead.
    This strategy is kept for backward compatibility.
    """

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def aliases(self) -> List[str]:
        return ["gemini"]

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Delegate to default strategy
        return DefaultProviderConfig("google").get_settings(settings, base_settings)


class XAIConfig(ProviderConfigStrategy):
    """Configuration strategy for xAI/Grok.

    DEPRECATED: Use DefaultProviderConfig instead.
    This strategy is kept for backward compatibility.
    """

    @property
    def provider_name(self) -> str:
        return "xai"

    @property
    def aliases(self) -> List[str]:
        return ["grok"]

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Delegate to default strategy
        return DefaultProviderConfig("xai").get_settings(settings, base_settings)


class OllamaConfig(ProviderConfigStrategy):
    """Configuration strategy for Ollama (local)."""

    @property
    def provider_name(self) -> str:
        return "ollama"

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(base_settings)
        result.setdefault("base_url", settings.ollama_base_url)
        return result


class LMStudioConfig(ProviderConfigStrategy):
    """Configuration strategy for LM Studio (local)."""

    @property
    def provider_name(self) -> str:
        return "lmstudio"

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(base_settings)

        urls = getattr(settings, "lmstudio_base_urls", []) or []
        # If provider config supplied a list, merge/override
        if "base_url" in result:
            cfg_url = result["base_url"]
            if isinstance(cfg_url, list):
                urls = cfg_url
            elif isinstance(cfg_url, str):
                urls = [cfg_url]

        chosen = None
        try:
            import httpx

            for url in urls:
                try:
                    resp = httpx.get(f"{url}/v1/models", timeout=1.5)
                    if resp.status_code == 200:
                        chosen = url
                        break
                except Exception:
                    continue
        except Exception:
            pass

        if urls:
            result["base_url"] = f"{(chosen or urls[0]).rstrip('/')}/v1"
        return result


class VLLMConfig(ProviderConfigStrategy):
    """Configuration strategy for vLLM (local)."""

    @property
    def provider_name(self) -> str:
        return "vllm"

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(base_settings)
        result.setdefault("base_url", settings.vllm_base_url)
        return result


class MoonshotConfig(ProviderConfigStrategy):
    """Configuration strategy for Moonshot/Kimi.

    DEPRECATED: Use DefaultProviderConfig instead.
    This strategy is kept for backward compatibility.
    """

    @property
    def provider_name(self) -> str:
        return "moonshot"

    @property
    def aliases(self) -> List[str]:
        return ["kimi"]

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Delegate to default strategy
        return DefaultProviderConfig("moonshot").get_settings(settings, base_settings)


class DeepSeekConfig(ProviderConfigStrategy):
    """Configuration strategy for DeepSeek.

    DEPRECATED: Use DefaultProviderConfig instead.
    This strategy is kept for backward compatibility.
    """

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Delegate to default strategy
        return DefaultProviderConfig("deepseek").get_settings(settings, base_settings)


class ZAIConfig(ProviderConfigStrategy):
    """Configuration strategy for Z.AI (ZhipuAI).

    DEPRECATED: Use DefaultProviderConfig instead.
    This strategy is kept for backward compatibility.

    Note: For GLM Coding Plan endpoint, use model suffix "glm-4.6:coding".
    The ZAIProvider automatically detects the :coding suffix and uses the coding endpoint.
    """

    @property
    def provider_name(self) -> str:
        return "zai"

    @property
    def aliases(self) -> List[str]:
        return ["zhipuai", "zhipu"]

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Delegate to default strategy
        return DefaultProviderConfig("zai").get_settings(settings, base_settings)


# NOTE: ZAICodingPlanConfig has been removed.
# GLM Coding Plan endpoint should now be accessed via model suffix:
#   Use model="glm-4.6:coding" instead of provider="zai-coding-plan"
# The ZAIProvider automatically detects the :coding suffix and uses the coding endpoint.


class QwenConfig(ProviderConfigStrategy):
    """Configuration strategy for Qwen (Alibaba Cloud)."""

    @property
    def provider_name(self) -> str:
        return "qwen"

    @property
    def aliases(self) -> List[str]:
        return ["alibaba", "dashscope"]

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        from victor.config.api_keys import get_api_key

        result = dict(base_settings)
        auth_mode = base_settings.get("auth_mode", "api_key")
        if auth_mode == "oauth":
            result["auth_mode"] = "oauth"
            result.setdefault("base_url", "https://portal.qwen.ai/v1/")
        else:
            api_key = get_api_key("qwen")
            if api_key:
                result["api_key"] = api_key
            result.setdefault("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1/")
        return result


class GroqCloudConfig(ProviderConfigStrategy):
    """Configuration strategy for Groq Cloud.

    DEPRECATED: Use DefaultProviderConfig instead.
    This strategy is kept for backward compatibility.
    """

    @property
    def provider_name(self) -> str:
        return "groqcloud"

    @property
    def aliases(self) -> List[str]:
        return ["groq"]

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Delegate to default strategy
        return DefaultProviderConfig("groqcloud").get_settings(settings, base_settings)


# =============================================================================
# Registry
# =============================================================================


@dataclass
class ProviderConfigRegistry:
    """Registry for provider configuration strategies.

    Implements the Registry pattern for OCP compliance.
    Thread-safe for concurrent access.
    """

    _strategies: Dict[str, ProviderConfigStrategy] = field(default_factory=dict)
    _aliases: Dict[str, str] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def register(self, strategy: ProviderConfigStrategy) -> None:
        """Register a provider configuration strategy.

        Args:
            strategy: The strategy to register
        """
        with self._lock:
            name = strategy.provider_name
            self._strategies[name] = strategy

            # Register aliases
            for alias in strategy.aliases:
                self._aliases[alias] = name

            logger.debug(f"Registered provider config: {name}")

    def get_settings(
        self,
        provider: str,
        settings: "Settings",
        profile_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get settings for a provider.

        Args:
            provider: Provider name (or alias)
            settings: Settings instance
            profile_overrides: Optional profile-level overrides (e.g., auth_mode from ProfileConfig.__pydantic_extra__)

        Returns:
            Provider settings dictionary
        """
        with self._lock:
            # Resolve alias
            resolved = self._aliases.get(provider, provider)

            # Load base settings from profiles.yaml
            base_settings = {}
            provider_config = settings.load_provider_config(resolved)
            if provider_config:
                base_settings.update(provider_config.model_dump(exclude_none=True))

            # Apply profile overrides BEFORE calling strategy
            # This allows the strategy to see auth_mode and make decisions
            if profile_overrides:
                base_settings.update(profile_overrides)

            # Get strategy
            strategy = self._strategies.get(resolved)
            if strategy:
                return strategy.get_settings(settings, base_settings)

            # Fallback to DefaultProviderConfig for simple API key providers
            if provider in DEFAULT_PROVIDER_ENDPOINTS:
                logger.debug(f"Using default config strategy for provider '{provider}'")
                return DefaultProviderConfig(provider).get_settings(settings, base_settings)

            # Last resort: return base settings if no strategy
            logger.debug(f"No config strategy for provider '{provider}', using base settings")
            return base_settings

    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        with self._lock:
            return list(self._strategies.keys())


# =============================================================================
# Singleton and Decorators
# =============================================================================

_registry_instance: Optional[ProviderConfigRegistry] = None
_registry_lock = threading.Lock()


def get_provider_config_registry() -> ProviderConfigRegistry:
    """Get the global provider config registry.

    Returns:
        Global ProviderConfigRegistry instance
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = ProviderConfigRegistry()
                _register_builtin_providers(_registry_instance)

    return _registry_instance


def _register_builtin_providers(registry: ProviderConfigRegistry) -> None:
    """Register all built-in provider configurations."""
    builtin_strategies = [
        AnthropicConfig(),
        OpenAIConfig(),
        GoogleConfig(),
        XAIConfig(),
        OllamaConfig(),
        LMStudioConfig(),
        VLLMConfig(),
        MoonshotConfig(),
        DeepSeekConfig(),
        GroqCloudConfig(),
        ZAIConfig(),
        # NOTE: ZAICodingPlanConfig removed - use model suffix "glm-4.6:coding" instead
        QwenConfig(),
    ]

    for strategy in builtin_strategies:
        registry.register(strategy)


def register_provider_config(
    provider_name: str,
) -> Callable[[Type[ProviderConfigStrategy]], Type[ProviderConfigStrategy]]:
    """Decorator to register a provider config strategy.

    Usage:
        @register_provider_config("my_provider")
        class MyProviderConfig(ProviderConfigStrategy):
            ...
    """

    def decorator(
        cls: Type[ProviderConfigStrategy],
    ) -> Type[ProviderConfigStrategy]:
        instance = cls()
        get_provider_config_registry().register(instance)
        return cls

    return decorator


__all__ = [
    "ProviderConfigStrategy",
    "ProviderConfigRegistry",
    "get_provider_config_registry",
    "register_provider_config",
    # Default strategy (handles most simple API key providers)
    "DefaultProviderConfig",
    "DEFAULT_PROVIDER_ENDPOINTS",
    # Built-in strategies (for testing/extension)
    # NOTE: Simple provider strategies below are deprecated - use DefaultProviderConfig
    "AnthropicConfig",
    "OpenAIConfig",
    "GoogleConfig",
    "XAIConfig",
    "OllamaConfig",
    "LMStudioConfig",
    "VLLMConfig",
    "MoonshotConfig",
    "DeepSeekConfig",
    "GroqCloudConfig",
    "ZAIConfig",
    # NOTE: ZAICodingPlanConfig removed - use model suffix "glm-4.6:coding" instead
    "QwenConfig",
]
