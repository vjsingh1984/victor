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

"""
Registry for tool calling adapters.

Provides factory methods for creating the appropriate adapter based on
provider name and model.
"""

import logging
from typing import Any, Dict, Optional, Type

from victor.agent.tool_calling.base import BaseToolCallingAdapter

logger = logging.getLogger(__name__)


class ToolCallingAdapterRegistry:
    """Registry for tool calling adapters.

    Maintains a mapping of provider names to adapter classes and provides
    factory methods for creating adapters.

    Usage:
        # Register a custom adapter
        ToolCallingAdapterRegistry.register("custom", CustomAdapter)

        # Get an adapter for a provider
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", model="llama3.1")
    """

    _adapters: Dict[str, Type[BaseToolCallingAdapter]] = {}

    @classmethod
    def register(cls, provider_name: str, adapter_class: Type[BaseToolCallingAdapter]) -> None:
        """Register an adapter class for a provider.

        Args:
            provider_name: Name of the provider (lowercase)
            adapter_class: Adapter class to register
        """
        cls._adapters[provider_name.lower()] = adapter_class
        logger.debug(
            f"Registered tool calling adapter: {provider_name} -> {adapter_class.__name__}"
        )

    @classmethod
    def get_adapter(
        cls,
        provider_name: str,
        model: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseToolCallingAdapter:
        """Get an adapter for a provider.

        Args:
            provider_name: Name of the provider
            model: Model name/identifier
            config: Optional configuration

        Returns:
            Appropriate adapter instance

        Raises:
            ValueError: If no adapter found for provider
        """
        provider_key = provider_name.lower()

        # Import adapters here to avoid circular imports
        from victor.agent.tool_calling.adapters import (
            AnthropicToolCallingAdapter,
            GoogleToolCallingAdapter,
            LMStudioToolCallingAdapter,
            OllamaToolCallingAdapter,
            OpenAICompatToolCallingAdapter,
            OpenAIToolCallingAdapter,
        )

        # Register default adapters if not already registered
        if not cls._adapters:
            cls._adapters = {
                "anthropic": AnthropicToolCallingAdapter,
                "openai": OpenAIToolCallingAdapter,
                "google": GoogleToolCallingAdapter,
                "ollama": OllamaToolCallingAdapter,
                # LMStudio uses dedicated adapter (like Ollama) with FallbackParsingMixin
                "lmstudio": LMStudioToolCallingAdapter,
                # vLLM uses OpenAI-compatible adapter
                "vllm": OpenAICompatToolCallingAdapter,
            }

        # Check for exact match
        if provider_key in cls._adapters:
            adapter_class = cls._adapters[provider_key]

            # Handle OpenAI-compatible providers (vLLM) that need variant info
            if adapter_class == OpenAICompatToolCallingAdapter:
                return adapter_class(  # type: ignore[call-arg]
                    model=model,
                    config=config,
                    provider_variant=provider_key,
                )
            return adapter_class(model=model, config=config)

        # Check for cloud providers
        cloud_providers = {"anthropic", "openai", "google", "xai"}
        if provider_key in cloud_providers:
            # xAI uses OpenAI-compatible format
            if provider_key == "xai":
                return OpenAIToolCallingAdapter(model=model, config=config)
            raise ValueError(f"No adapter registered for cloud provider: {provider_name}")

        # Default to OpenAI-compatible for unknown providers
        logger.warning(
            f"No specific adapter for provider '{provider_name}', "
            "using OpenAI-compatible adapter"
        )
        return OpenAICompatToolCallingAdapter(
            model=model,
            config=config,
            provider_variant=provider_key,
        )

    @classmethod
    def list_providers(cls) -> list:
        """List registered provider names."""
        # Ensure defaults are loaded
        cls.get_adapter("ollama")  # Triggers default registration
        return list(cls._adapters.keys())

    @classmethod
    def is_registered(cls, provider_name: str) -> bool:
        """Check if a provider has a registered adapter."""
        # Ensure defaults are loaded
        if not cls._adapters:
            cls.get_adapter("ollama")  # Triggers default registration
        return provider_name.lower() in cls._adapters
