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

"""Provider Config Factory Registry - OCP-compliant provider configuration.

This module implements the Open/Closed Principle by allowing providers
to register their configuration strategies without modifying core code.

Design Philosophy:
- Open for Extension: New providers register themselves via decorators
- Closed for Modification: Core code doesn't change when adding providers
- Strategy Pattern: Each provider has its own config/listing strategy
- Registry Pattern: Central registry for provider strategies

Usage:
    @register_provider_config("myprovider")
    class MyProviderConfig:
        @staticmethod
        async def list_models(settings):
            # Provider-specific model listing
            return ["model1", "model2"]

    # Use the registry
    models = await ProviderConfigRegistry.list_models("myprovider", settings)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# =============================================================================
# Provider Strategy Protocol
# =============================================================================


class ProviderConfigStrategy(ABC):
    """Abstract base class for provider configuration strategies.

    Each provider should implement this interface and register it
    with the ProviderConfigRegistry.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""

    @abstractmethod
    async def list_models(
        self,
        settings: Any,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List available models for this provider.

        Args:
            settings: Application settings
            endpoint: Optional endpoint override

        Returns:
            List of model dictionaries
        """

    @abstractmethod
    def get_config_form(self) -> Type:
        """Get the config form class for this provider."""


# =============================================================================
# Provider Registry
# =============================================================================


@dataclass
class ProviderRegistration:
    """Registration info for a provider."""

    provider_name: str
    strategy: ProviderConfigStrategy
    priority: int = 0  # Higher priority = preferred


class ProviderConfigRegistry:
    """Registry for provider configuration strategies.

    This registry implements the Open/Closed Principle by allowing
    providers to register their strategies without modifying core code.

    Example:
        @ProviderConfigRegistry.register("myprovider")
        class MyProviderStrategy(ProviderConfigStrategy):
            async def list_models(self, settings, endpoint):
                return [...]
    """

    _providers: Dict[str, ProviderRegistration] = {}
    _lock = asyncio.Lock()

    @classmethod
    def register(
        cls,
        provider_name: str,
        priority: int = 0,
    ) -> Callable[[Type[ProviderConfigStrategy]], Type[ProviderConfigStrategy]]:
        """Decorator to register a provider strategy.

        Args:
            provider_name: Name of the provider
            priority: Priority for provider selection (higher = preferred)

        Returns:
            Decorator function

        Example:
            @ProviderConfigRegistry.register("myprovider", priority=10)
            class MyProviderStrategy(ProviderConfigStrategy):
                ...
        """

        def decorator(strategy_class: Type[ProviderConfigStrategy]) -> Type[ProviderConfigStrategy]:
            # Create instance of the strategy
            instance = strategy_class()

            registration = ProviderRegistration(
                provider_name=provider_name,
                strategy=instance,
                priority=priority,
            )

            cls._providers[provider_name] = registration
            logger.debug(f"Registered provider strategy: {provider_name}")
            return strategy_class

        return decorator

    @classmethod
    def get_strategy(cls, provider_name: str) -> Optional[ProviderConfigStrategy]:
        """Get the strategy for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider strategy or None if not registered
        """
        registration = cls._providers.get(provider_name)
        return registration.strategy if registration else None

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, provider_name: str) -> bool:
        """Check if a provider is registered.

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider is registered
        """
        return provider_name in cls._providers

    @classmethod
    async def list_models(
        cls,
        provider_name: str,
        settings: Any,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List models for a provider using its registered strategy.

        Args:
            provider_name: Name of the provider
            settings: Application settings
            endpoint: Optional endpoint override

        Returns:
            List of model dictionaries

        Raises:
            ValueError: If provider is not registered
        """
        strategy = cls.get_strategy(provider_name)
        if not strategy:
            raise ValueError(
                f"Provider '{provider_name}' not registered. "
                f"Available providers: {', '.join(cls.list_providers())}"
            )

        return await strategy.list_models(settings, endpoint)

    @classmethod
    def get_config_form(cls, provider_name: str) -> Optional[Type]:
        """Get the config form class for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Config form class or None if not registered
        """
        strategy = cls.get_strategy(provider_name)
        if not strategy:
            return None
        return strategy.get_config_form()

    @classmethod
    def get_prioritized_providers(cls) -> List[str]:
        """Get providers sorted by priority (highest first).

        Returns:
            List of provider names sorted by priority
        """
        providers = list(cls._providers.items())
        providers.sort(key=lambda x: x[1].priority, reverse=True)
        return [name for name, _ in providers]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (for testing)."""
        cls._providers.clear()


# =============================================================================
# Built-in Provider Strategies
# =============================================================================


class OllamaProviderStrategy(ProviderConfigStrategy):
    """Strategy for Ollama provider."""

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def list_models(
        self,
        settings: Any,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List Ollama models."""
        from victor.providers.ollama_provider import OllamaProvider

        provider_settings = settings.get_provider_settings("ollama")
        provider = OllamaProvider(**provider_settings)

        try:
            models_list = await provider.list_models()
            return models_list or []
        finally:
            await provider.close()

    def get_config_form(self) -> Type:
        """Get Ollama config form."""
        try:
            from victor.ui.commands.models import OllamaConfigForm

            return OllamaConfigForm
        except ImportError:
            return None


class LMStudioProviderStrategy(ProviderConfigStrategy):
    """Strategy for LMStudio provider."""

    @property
    def provider_name(self) -> str:
        return "lmstudio"

    async def list_models(
        self,
        settings: Any,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List LMStudio models."""
        from victor.providers.lmstudio_provider import LMStudioProvider

        provider_settings = settings.get_provider_settings("lmstudio")

        if endpoint:
            provider_settings["base_url"] = endpoint

        provider = await LMStudioProvider.create(**provider_settings)
        models_list = await provider.list_models()
        return models_list or []

    def get_config_form(self) -> Type:
        """Get LMStudio config form."""
        try:
            from victor.ui.commands.models import LMStudioConfigForm

            return LMStudioConfigForm
        except ImportError:
            return None


class AnthropicProviderStrategy(ProviderConfigStrategy):
    """Strategy for Anthropic provider."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def list_models(
        self,
        settings: Any,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List Anthropic models (static list)."""
        return [
            {"id": "claude-sonnet-4-5-20250514", "name": "Claude Sonnet 4.5"},
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
            {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku"},
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
        ]

    def get_config_form(self) -> Type:
        """Get Anthropic config form."""
        try:
            from victor.ui.commands.models import AnthropicConfigForm

            return AnthropicConfigForm
        except ImportError:
            return None


class OpenAIProviderStrategy(ProviderConfigStrategy):
    """Strategy for OpenAI provider."""

    @property
    def provider_name(self) -> str:
        return "openai"

    async def list_models(
        self,
        settings: Any,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List OpenAI models (static list)."""
        return [
            {"id": "gpt-4o", "name": "GPT-4o"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
        ]

    def get_config_form(self) -> Type:
        """Get OpenAI config form."""
        try:
            from victor.ui.commands.models import OpenAIConfigForm

            return OpenAIConfigForm
        except ImportError:
            return None


class GoogleProviderStrategy(ProviderConfigStrategy):
    """Strategy for Google provider."""

    @property
    def provider_name(self) -> str:
        return "google"

    async def list_models(
        self,
        settings: Any,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List Google models (static list)."""
        return [
            {"id": "gemini-2.5-pro-exp-03-25", "name": "Gemini 2.5 Pro"},
            {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
        ]

    def get_config_form(self) -> Type:
        """Get Google config form."""
        try:
            from victor.ui.commands.models import GoogleConfigForm

            return GoogleConfigForm
        except ImportError:
            return None


class GroqProviderStrategy(ProviderConfigStrategy):
    """Strategy for Groq provider."""

    @property
    def provider_name(self) -> str:
        return "groqcloud"

    async def list_models(
        self,
        settings: Any,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List Groq models (static list)."""
        return [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile"},
            {"id": "llama-3.1-70b-versatile", "name": "Llama 3.1 70B Versatile"},
        ]

    def get_config_form(self) -> Type:
        """Get Groq config form."""
        try:
            from victor.ui.commands.models import GroqConfigForm

            return GroqConfigForm
        except ImportError:
            return None


class CerebrasProviderStrategy(ProviderConfigStrategy):
    """Strategy for Cerebras provider."""

    @property
    def provider_name(self) -> str:
        return "cerebras"

    async def list_models(
        self,
        settings: Any,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List Cerebras models (static list)."""
        return [
            {"id": "llama3.1-70b", "name": "Llama 3.1 70B"},
        ]

    def get_config_form(self) -> Type:
        """Get Cerebras config form."""
        try:
            from victor.ui.commands.models import CerebrasConfigForm

            return CerebrasConfigForm
        except ImportError:
            return None


class LlamaCppProviderStrategy(ProviderConfigStrategy):
    """Strategy for llama.cpp provider."""

    @property
    def provider_name(self) -> str:
        return "llamacpp"

    async def list_models(
        self,
        settings: Any,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List llama.cpp models (static - needs local file)."""
        return []  # llama.cpp needs a model file path

    def get_config_form(self) -> Type:
        """Get llama.cpp config form."""
        try:
            from victor.ui.commands.models import LlamaCppConfigForm

            return LlamaCppConfigForm
        except ImportError:
            return None


class VLLMProviderStrategy(ProviderConfigStrategy):
    """Strategy for vLLM provider."""

    @property
    def provider_name(self) -> str:
        return "vllm"

    async def list_models(
        self,
        settings: Any,
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List vLLM models."""
        from victor.providers.vllm_provider import VLLMProvider

        provider_settings = settings.get_provider_settings("vllm")

        if endpoint:
            provider_settings["base_url"] = endpoint

        provider = VLLMProvider(**provider_settings)
        models_list = await provider.list_models()
        return models_list or []

    def get_config_form(self) -> Type:
        """Get vLLM config form."""
        try:
            from victor.ui.commands.models import VLLMConfigForm

            return VLLMConfigForm
        except ImportError:
            return None


# Register built-in providers
ProviderConfigRegistry.register("ollama", priority=10)(OllamaProviderStrategy)
ProviderConfigRegistry.register("lmstudio", priority=10)(LMStudioProviderStrategy)
ProviderConfigRegistry.register("llamacpp", priority=5)(LlamaCppProviderStrategy)
ProviderConfigRegistry.register("vllm", priority=10)(VLLMProviderStrategy)
ProviderConfigRegistry.register("anthropic", priority=20)(AnthropicProviderStrategy)
ProviderConfigRegistry.register("openai", priority=20)(OpenAIProviderStrategy)
ProviderConfigRegistry.register("google", priority=15)(GoogleProviderStrategy)
ProviderConfigRegistry.register("cerebras", priority=10)(CerebrasProviderStrategy)
ProviderConfigRegistry.register("groqcloud", priority=10)(GroqProviderStrategy)


# =============================================================================
# Public API
# =============================================================================


def register_provider_config(
    provider_name: str,
    priority: int = 0,
) -> Callable[[Type[ProviderConfigStrategy]], Type[ProviderConfigStrategy]]:
    """Register a provider configuration strategy.

    This is a convenience function that delegates to the registry.

    Args:
        provider_name: Name of the provider
        priority: Priority for provider selection (higher = preferred)

    Returns:
        Decorator function

    Example:
        @register_provider_config("myprovider", priority=10)
        class MyProviderStrategy(ProviderConfigStrategy):
            ...
    """
    return ProviderConfigRegistry.register(provider_name, priority)


__all__ = [
    "ProviderConfigStrategy",
    "ProviderConfigRegistry",
    "ProviderRegistration",
    "register_provider_config",
    # Built-in strategies
    "OllamaProviderStrategy",
    "LMStudioProviderStrategy",
    "AnthropicProviderStrategy",
    "OpenAIProviderStrategy",
    "GoogleProviderStrategy",
    "GroqProviderStrategy",
    "CerebrasProviderStrategy",
    "LlamaCppProviderStrategy",
    "VLLMProviderStrategy",
]
