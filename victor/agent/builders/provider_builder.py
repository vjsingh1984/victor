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

"""ProviderBuilder for building provider components.

This module provides a builder for provider-related components used by
AgentOrchestrator, including:
- Provider instances
- ProviderManager
- Tool adapters
- Tool calling capabilities
- System prompt builder

Part of HIGH-005: Initialization Complexity reduction.
"""

from typing import Any, Dict, Optional
from victor.agent.builders.base import ComponentBuilder
from victor.agent.orchestrator_factory import OrchestratorFactory


class ProviderBuilder(ComponentBuilder):
    """Build provider and provider-related components.

    This builder creates all provider-related components that the orchestrator
    needs for LLM interaction. It delegates to OrchestratorFactory for the
    actual component creation while providing a cleaner, more focused API.

    Components built:
        - provider: BaseProvider instance
        - provider_manager: ProviderManager for switching/health
        - provider_name: Provider label (e.g., "anthropic", "openai")
        - tool_adapter: BaseToolCallingAdapter for provider-specific behavior
        - tool_calling_caps: ToolCallingCapabilities
        - prompt_builder: SystemPromptBuilder with vertical extensions
        - tool_calling_models: Dict of models supporting tool calling
        - tool_capabilities: Dict of model-specific capabilities
    """

    def __init__(self, settings, factory: Optional[OrchestratorFactory] = None):
        """Initialize the ProviderBuilder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance (created if not provided)
        """
        super().__init__(settings)
        self._factory = factory

    def build(
        self,
        provider: Any = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        provider_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        tool_selection: Optional[Dict[str, Any]] = None,
        thinking: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build all provider components.

        Args:
            provider: Optional provider instance (built if not provided)
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            provider_name: Optional provider label from profile
            profile_name: Optional profile name for session tracking
            tool_selection: Optional tool selection configuration
            thinking: Enable extended thinking mode
            **kwargs: Additional dependencies

        Returns:
            Dictionary of built provider components with keys:
            - provider: BaseProvider instance
            - provider_manager: ProviderManager
            - provider_name: Provider label string
            - tool_adapter: BaseToolCallingAdapter
            - tool_calling_caps: ToolCallingCapabilities
            - prompt_builder: SystemPromptBuilder
            - tool_calling_models: Dict of models with tool support
            - tool_capabilities: Dict of model-specific capabilities

        Raises:
            ValueError: If provider is not provided and cannot be built
        """
        provider_components = {}

        # Create or reuse factory
        if self._factory is None:
            if provider is None or model is None:
                raise ValueError("provider and model are required when factory is not provided")
            self._factory = OrchestratorFactory(
                settings=self.settings,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                provider_name=provider_name,
                profile_name=profile_name,
                tool_selection=tool_selection,
                thinking=thinking,
            )

        # Build tool calling matrix
        tool_calling_models, tool_capabilities = self._factory.create_tool_calling_matrix()
        provider_components["tool_calling_models"] = tool_calling_models
        provider_components["tool_capabilities"] = tool_capabilities

        # Build provider manager with adapter
        (
            provider_manager,
            provider_instance,
            model_id,
            provider_label,
            tool_adapter,
            tool_calling_caps,
        ) = self._factory.create_provider_manager_with_adapter(
            provider, model or "unknown", provider_name
        )

        provider_components["provider_manager"] = provider_manager
        provider_components["provider"] = provider_instance
        provider_components["model"] = model_id
        provider_components["provider_name"] = provider_label
        provider_components["tool_adapter"] = tool_adapter
        provider_components["tool_calling_caps"] = tool_calling_caps

        # Build system prompt builder
        prompt_builder = self._factory.create_system_prompt_builder(
            provider_name=provider_label,
            model=model_id,
            tool_adapter=tool_adapter,
            tool_calling_caps=tool_calling_caps,
        )
        provider_components["prompt_builder"] = prompt_builder

        # Register all built components
        for name, component in provider_components.items():
            if component is not None:
                self.register_component(name, component)

        self._logger.info(
            f"ProviderBuilder built {len(provider_components)} provider components: "
            f"{', '.join(provider_components.keys())}"
        )

        return provider_components

    def get_tool_adapter(self) -> Optional[Any]:
        """Get the tool adapter from built components.

        Returns:
            The tool adapter if built, None otherwise
        """
        return self.get_component("tool_adapter")

    def get_tool_calling_caps(self) -> Optional[Any]:
        """Get the tool calling capabilities from built components.

        Returns:
            The tool calling capabilities if built, None otherwise
        """
        return self.get_component("tool_calling_caps")

    def get_provider(self) -> Optional[Any]:
        """Get the provider from built components.

        Returns:
            The provider instance if built, None otherwise
        """
        return self.get_component("provider")

    def get_provider_manager(self) -> Optional[Any]:
        """Get the provider manager from built components.

        Returns:
            The provider manager if built, None otherwise
        """
        return self.get_component("provider_manager")
