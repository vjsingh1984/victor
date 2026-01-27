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

"""Test factories for creating test objects and fixtures.

This module provides factory classes for creating test objects:
- MockProviderFactory: Create mock providers with various configurations
- TestSettingsBuilder: Build test settings with different configurations
- TestFixtureFactory: Create complex test fixtures for integration tests

Usage:
    from tests.factories import MockProviderFactory, TestSettingsBuilder

    # Create a mock provider
    provider = MockProviderFactory.create_anthropic()

    # Build test settings
    settings = TestSettingsBuilder().with_tool_budget(100).build()
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from victor.agent.protocols import (
    ToolExecutorProtocol,
    ToolRegistryProtocol,
)
from victor.config.settings import Settings
from victor.core.container import ServiceContainer, ServiceLifetime
from victor.providers.base import BaseProvider, Message, StreamChunk


class MockProviderFactory:
    """Factory for creating mock providers with various configurations.

    Provides methods to create mock providers for different LLM providers
    with configurable behavior and responses.

    Examples:
        # Create a mock Anthropic provider
        provider = MockProviderFactory.create_anthropic()

        # Create a provider with custom response
        provider = MockProviderFactory.create_with_response(
            "custom response",
            supports_tools=True
        )

        # Create a failing provider
        provider = MockProviderFactory.create_failing_provider()
    """

    @staticmethod
    def _create_base_provider(
        name: str,
        model: str = "test-model",
        response_content: str = "Test response",
        supports_tools: bool = True,
        supports_streaming: bool = True,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Mock:
        """Create a base mock provider with common configuration.

        Args:
            name: Provider name
            model: Model identifier
            response_content: Default chat response content
            supports_tools: Whether provider supports tool calling
            supports_streaming: Whether provider supports streaming
            tool_calls: Optional tool calls to return

        Returns:
            Mock provider with async methods
        """
        provider = Mock(spec=BaseProvider)
        provider.name = name
        provider.model = model

        # Mock chat method
        response = Mock()
        response.content = response_content
        response.role = "assistant"
        response.model = model
        response.stop_reason = "stop"
        response.tool_calls = tool_calls
        response.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        response.raw_response = None
        response.metadata = None

        provider.chat = AsyncMock(return_value=response)

        # Mock stream method
        async def mock_stream(*args, **kwargs):
            yield StreamChunk(content=response_content, is_final=False)
            yield StreamChunk(content="", is_final=True, stop_reason="stop")

        provider.stream = mock_stream

        # Mock capability checks
        provider.supports_tools = Mock(return_value=supports_tools)
        provider.supports_streaming = Mock(return_value=supports_streaming)

        return provider

    @classmethod
    def create_anthropic(
        cls,
        model: str = "claude-sonnet-4-5",
        response_content: str = "Anthropic response",
    ) -> Mock:
        """Create a mock Anthropic provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Anthropic provider
        """
        return cls._create_base_provider(
            name="anthropic",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_openai(
        cls,
        model: str = "gpt-4",
        response_content: str = "OpenAI response",
    ) -> Mock:
        """Create a mock OpenAI provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock OpenAI provider
        """
        return cls._create_base_provider(
            name="openai",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_google(
        cls,
        model: str = "gemini-pro",
        response_content: str = "Google response",
    ) -> Mock:
        """Create a mock Google provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Google provider
        """
        return cls._create_base_provider(
            name="google",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_ollama(
        cls,
        model: str = "llama2",
        response_content: str = "Ollama response",
    ) -> Mock:
        """Create a mock Ollama provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Ollama provider
        """
        return cls._create_base_provider(
            name="ollama",
            model=model,
            response_content=response_content,
            supports_tools=False,
            supports_streaming=True,
        )

    @classmethod
    def create_with_response(
        cls,
        response_content: str,
        name: str = "mock",
        model: str = "test-model",
        supports_tools: bool = True,
        supports_streaming: bool = True,
    ) -> Mock:
        """Create a mock provider with custom response.

        Args:
            response_content: Response content to return
            name: Provider name
            model: Model identifier
            supports_tools: Whether provider supports tools
            supports_streaming: Whether provider supports streaming

        Returns:
            Mock provider with custom response
        """
        return cls._create_base_provider(
            name=name,
            model=model,
            response_content=response_content,
            supports_tools=supports_tools,
            supports_streaming=supports_streaming,
        )

    @classmethod
    def create_with_tool_calls(
        cls,
        tool_calls: List[Dict[str, Any]],
        name: str = "mock",
        model: str = "test-model",
    ) -> Mock:
        """Create a mock provider that returns tool calls.

        Args:
            tool_calls: Tool calls to return
            name: Provider name
            model: Model identifier

        Returns:
            Mock provider with tool calls
        """
        return cls._create_base_provider(
            name=name,
            model=model,
            response_content="",  # Empty content for tool calls
            supports_tools=True,
            supports_streaming=True,
            tool_calls=tool_calls,
        )

    @classmethod
    def create_failing_provider(
        cls,
        name: str = "failing",
        error_message: str = "Provider error",
    ) -> Mock:
        """Create a mock provider that always fails.

        Args:
            name: Provider name
            error_message: Error message to raise

        Returns:
            Mock provider that raises exceptions
        """
        provider = Mock(spec=BaseProvider)
        provider.name = name
        provider.model = "failing-model"

        # Chat method raises exception
        async def failing_chat(*args, **kwargs):
            raise Exception(error_message)

        provider.chat = failing_chat

        # Stream method raises exception
        async def failing_stream(*args, **kwargs):
            raise Exception(error_message)
            yield  # Never reached, but needed for generator

        provider.stream = failing_stream

        provider.supports_tools = Mock(return_value=False)
        provider.supports_streaming = Mock(return_value=False)

        return provider

    @classmethod
    def create_rate_limited_provider(cls, name: str = "rate_limited") -> Mock:
        """Create a mock provider that simulates rate limiting.

        Args:
            name: Provider name

        Returns:
            Mock provider that simulates rate limits
        """
        from victor.providers.errors import RateLimitError

        provider = Mock(spec=BaseProvider)
        provider.name = name
        provider.model = "rate-limited-model"

        # First call succeeds, subsequent calls fail
        call_count = 0

        async def rate_limited_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RateLimitError("Rate limit exceeded", retry_after=60)
            response = Mock()
            response.content = "First call succeeds"
            response.role = "assistant"
            response.model = "rate-limited-model"
            response.stop_reason = "stop"
            response.tool_calls = None
            response.usage = {"prompt_tokens": 10, "completion_tokens": 20}
            return response

        provider.chat = rate_limited_chat
        provider.supports_tools = Mock(return_value=True)
        provider.supports_streaming = Mock(return_value=False)

        return provider

    # ==========================================================================
    # Additional Provider Factory Methods (22 total providers)
    # ==========================================================================

    @classmethod
    def create_azure_openai(
        cls,
        model: str = "gpt-4",
        response_content: str = "Azure OpenAI response",
    ) -> Mock:
        """Create a mock Azure OpenAI provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Azure OpenAI provider
        """
        return cls._create_base_provider(
            name="azure_openai",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_cerebras(
        cls,
        model: str = "llama3.1-70b",
        response_content: str = "Cerebras response",
    ) -> Mock:
        """Create a mock Cerebras provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Cerebras provider
        """
        return cls._create_base_provider(
            name="cerebras",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_deepseek(
        cls,
        model: str = "deepseek-chat",
        response_content: str = "DeepSeek response",
    ) -> Mock:
        """Create a mock DeepSeek provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock DeepSeek provider
        """
        return cls._create_base_provider(
            name="deepseek",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_fireworks(
        cls,
        model: str = "llama-v2-7b",
        response_content: str = "Fireworks response",
    ) -> Mock:
        """Create a mock Fireworks provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Fireworks provider
        """
        return cls._create_base_provider(
            name="fireworks",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_groq(
        cls,
        model: str = "llama3-70b-8192",
        response_content: str = "Groq response",
    ) -> Mock:
        """Create a mock Groq provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Groq provider
        """
        return cls._create_base_provider(
            name="groq",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_huggingface(
        cls,
        model: str = "gpt2",
        response_content: str = "HuggingFace response",
    ) -> Mock:
        """Create a mock HuggingFace provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock HuggingFace provider
        """
        return cls._create_base_provider(
            name="huggingface",
            model=model,
            response_content=response_content,
            supports_tools=False,
            supports_streaming=False,
        )

    @classmethod
    def create_lmstudio(
        cls,
        model: str = "local-model",
        response_content: str = "LMStudio response",
    ) -> Mock:
        """Create a mock LMStudio provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock LMStudio provider
        """
        return cls._create_base_provider(
            name="lmstudio",
            model=model,
            response_content=response_content,
            supports_tools=False,
            supports_streaming=True,
        )

    @classmethod
    def create_llamacpp(
        cls,
        model: str = "llama2",
        response_content: str = "Llama.cpp response",
    ) -> Mock:
        """Create a mock Llama.cpp provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Llama.cpp provider
        """
        return cls._create_base_provider(
            name="llamacpp",
            model=model,
            response_content=response_content,
            supports_tools=False,
            supports_streaming=True,
        )

    @classmethod
    def create_mistral(
        cls,
        model: str = "mistral-large",
        response_content: str = "Mistral response",
    ) -> Mock:
        """Create a mock Mistral provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Mistral provider
        """
        return cls._create_base_provider(
            name="mistral",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_moonshot(
        cls,
        model: str = "moonshot-v1-8k",
        response_content: str = "Moonshot response",
    ) -> Mock:
        """Create a mock Moonshot provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Moonshot provider
        """
        return cls._create_base_provider(
            name="moonshot",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_openrouter(
        cls,
        model: str = "anthropic/claude-2",
        response_content: str = "OpenRouter response",
    ) -> Mock:
        """Create a mock OpenRouter provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock OpenRouter provider
        """
        return cls._create_base_provider(
            name="openrouter",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_replicate(
        cls,
        model: str = "meta/llama-2-70b",
        response_content: str = "Replicate response",
    ) -> Mock:
        """Create a mock Replicate provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Replicate provider
        """
        return cls._create_base_provider(
            name="replicate",
            model=model,
            response_content=response_content,
            supports_tools=False,
            supports_streaming=False,
        )

    @classmethod
    def create_together(
        cls,
        model: str = "togethercomputer/llama-2-70b",
        response_content: str = "Together response",
    ) -> Mock:
        """Create a mock Together provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Together provider
        """
        return cls._create_base_provider(
            name="together",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_vertex(
        cls,
        model: str = "gemini-pro",
        response_content: str = "Vertex response",
    ) -> Mock:
        """Create a mock Vertex provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Vertex provider
        """
        return cls._create_base_provider(
            name="vertex",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_vllm(
        cls,
        model: str = "facebook/llama-2-70b",
        response_content: str = "vLLM response",
    ) -> Mock:
        """Create a mock vLLM provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock vLLM provider
        """
        return cls._create_base_provider(
            name="vllm",
            model=model,
            response_content=response_content,
            supports_tools=False,
            supports_streaming=True,
        )

    @classmethod
    def create_xai(
        cls,
        model: str = "grok-beta",
        response_content: str = "xAI response",
    ) -> Mock:
        """Create a mock xAI provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock xAI provider
        """
        return cls._create_base_provider(
            name="xai",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_zai(
        cls,
        model: str = "zai-model",
        response_content: str = "ZAI response",
    ) -> Mock:
        """Create a mock ZAI provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock ZAI provider
        """
        return cls._create_base_provider(
            name="zai",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )

    @classmethod
    def create_bedrock(
        cls,
        model: str = "anthropic.claude-3",
        response_content: str = "Bedrock response",
    ) -> Mock:
        """Create a mock Bedrock provider.

        Args:
            model: Model identifier
            response_content: Default response content

        Returns:
            Mock Bedrock provider
        """
        return cls._create_base_provider(
            name="bedrock",
            model=model,
            response_content=response_content,
            supports_tools=True,
            supports_streaming=True,
        )


class TestSettingsBuilder:
    """Builder for creating test settings with various configurations.

    Provides a fluent interface for building test settings with
    different configurations for testing.

    Examples:
        # Build with tool budget
        settings = TestSettingsBuilder().with_tool_budget(100).build()

        # Build with multiple settings
        settings = (
            TestSettingsBuilder()
            .with_tool_budget(50)
            .with_max_iterations(25)
            .with_provider("openai")
            .build()
        )

        # Build with airgapped mode
        settings = TestSettingsBuilder().with_airgapped_mode().build()
    """

    def __init__(self) -> None:
        """Initialize builder with default settings."""
        self._settings = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "temperature": 0.7,
            "max_tokens": 4096,
            "tool_budget": 30,
            "max_iterations": 50,
            "airgapped_mode": False,
            "tool_cache_enabled": True,
            "conversation_memory_enabled": False,
            "analytics_enabled": False,
            "debug_logging": False,
            "enable_observability": False,
            "tool_selection_strategy": "hybrid",
            "parallel_tool_execution": True,
            "max_concurrent_tools": 5,
            "enable_tool_deduplication": False,
            "enable_continuation_rl_learning": False,
            "plugin_enabled": False,
            "use_mcp_tools": False,
        }

    def with_provider(self, provider: str) -> "TestSettingsBuilder":
        """Set the provider.

        Args:
            provider: Provider name (e.g., "anthropic", "openai")

        Returns:
            Self for method chaining
        """
        self._settings["provider"] = provider
        return self

    def with_model(self, model: str) -> "TestSettingsBuilder":
        """Set the model.

        Args:
            model: Model identifier

        Returns:
            Self for method chaining
        """
        self._settings["model"] = model
        return self

    def with_temperature(self, temperature: float) -> "TestSettingsBuilder":
        """Set the temperature.

        Args:
            temperature: Sampling temperature

        Returns:
            Self for method chaining
        """
        self._settings["temperature"] = temperature
        return self

    def with_max_tokens(self, max_tokens: int) -> "TestSettingsBuilder":
        """Set the maximum tokens.

        Args:
            max_tokens: Maximum tokens to generate

        Returns:
            Self for method chaining
        """
        self._settings["max_tokens"] = max_tokens
        return self

    def with_tool_budget(self, budget: int) -> "TestSettingsBuilder":
        """Set the tool call budget.

        Args:
            budget: Maximum tool calls allowed

        Returns:
            Self for method chaining
        """
        self._settings["tool_budget"] = budget
        return self

    def with_max_iterations(self, iterations: int) -> "TestSettingsBuilder":
        """Set the maximum iterations.

        Args:
            iterations: Maximum iterations allowed

        Returns:
            Self for method chaining
        """
        self._settings["max_iterations"] = iterations
        return self

    def with_airgapped_mode(self, enabled: bool = True) -> "TestSettingsBuilder":
        """Enable or disable airgapped mode.

        Args:
            enabled: Whether to enable airgapped mode

        Returns:
            Self for method chaining
        """
        self._settings["airgapped_mode"] = enabled
        return self

    def with_tool_cache(self, enabled: bool = True) -> "TestSettingsBuilder":
        """Enable or disable tool cache.

        Args:
            enabled: Whether to enable tool cache

        Returns:
            Self for method chaining
        """
        self._settings["tool_cache_enabled"] = enabled
        return self

    def with_conversation_memory(self, enabled: bool = True) -> "TestSettingsBuilder":
        """Enable or disable conversation memory.

        Args:
            enabled: Whether to enable conversation memory

        Returns:
            Self for method chaining
        """
        self._settings["conversation_memory_enabled"] = enabled
        return self

    def with_analytics(self, enabled: bool = True) -> "TestSettingsBuilder":
        """Enable or disable analytics.

        Args:
            enabled: Whether to enable analytics

        Returns:
            Self for method chaining
        """
        self._settings["analytics_enabled"] = enabled
        return self

    def with_debug_logging(self, enabled: bool = True) -> "TestSettingsBuilder":
        """Enable or disable debug logging.

        Args:
            enabled: Whether to enable debug logging

        Returns:
            Self for method chaining
        """
        self._settings["debug_logging"] = enabled
        return self

    def with_observability(self, enabled: bool = True) -> "TestSettingsBuilder":
        """Enable or disable observability.

        Args:
            enabled: Whether to enable observability

        Returns:
            Self for method chaining
        """
        self._settings["enable_observability"] = enabled
        return self

    def with_tool_selection_strategy(self, strategy: str) -> "TestSettingsBuilder":
        """Set the tool selection strategy.

        Args:
            strategy: Strategy name (auto, keyword, semantic, hybrid)

        Returns:
            Self for method chaining
        """
        self._settings["tool_selection_strategy"] = strategy
        return self

    def with_parallel_tools(
        self, enabled: bool = True, max_concurrent: int = 5
    ) -> "TestSettingsBuilder":
        """Configure parallel tool execution.

        Args:
            enabled: Whether to enable parallel execution
            max_concurrent: Maximum concurrent tools

        Returns:
            Self for method chaining
        """
        self._settings["parallel_tool_execution"] = enabled
        self._settings["max_concurrent_tools"] = max_concurrent
        return self

    def with_tool_deduplication(
        self, enabled: bool = True, window_size: int = 10
    ) -> "TestSettingsBuilder":
        """Configure tool deduplication.

        Args:
            enabled: Whether to enable deduplication
            window_size: Deduplication window size

        Returns:
            Self for method chaining
        """
        self._settings["enable_tool_deduplication"] = enabled
        self._settings["tool_deduplication_window_size"] = window_size
        return self

    def with_rl_learning(self, enabled: bool = True) -> "TestSettingsBuilder":
        """Enable or disable RL learning.

        Args:
            enabled: Whether to enable RL learning

        Returns:
            Self for method chaining
        """
        self._settings["enable_continuation_rl_learning"] = enabled
        return self

    def with_plugins(self, enabled: bool = True) -> "TestSettingsBuilder":
        """Enable or disable plugins.

        Args:
            enabled: Whether to enable plugins

        Returns:
            Self for method chaining
        """
        self._settings["plugin_enabled"] = enabled
        return self

    def with_mcp_tools(self, enabled: bool = True) -> "TestSettingsBuilder":
        """Enable or disable MCP tools.

        Args:
            enabled: Whether to enable MCP tools

        Returns:
            Self for method chaining
        """
        self._settings["use_mcp_tools"] = enabled
        return self

    def with_custom_setting(self, key: str, value: Any) -> "TestSettingsBuilder":
        """Set a custom setting.

        Args:
            key: Setting key
            value: Setting value

        Returns:
            Self for method chaining
        """
        self._settings[key] = value
        return self

    def build(self) -> Mock:
        """Build the settings object.

        Returns:
            Mock settings object with configured values
        """
        settings = Mock(spec=Settings)
        for key, value in self._settings.items():
            setattr(settings, key, value)

        # Add default attributes that tests might expect
        settings.api_key = None
        settings.timeout = 30
        settings.project_root = Path.cwd()

        return settings


class TestFixtureFactory:
    """Factory for creating complex test fixtures for integration tests.

    Provides methods to create complex test fixtures including:
    - Orchestrator components
    - DI containers with services
    - Tool registries
    - Message histories

    Examples:
        # Create a DI container with basic services
        container = TestFixtureFactory.create_container()

        # Create a tool registry with test tools
        registry = TestFixtureFactory.create_tool_registry()

        # Create a complete orchestrator fixture
        fixture = TestFixtureFactory.create_orchestrator_fixture()
    """

    @staticmethod
    def create_container() -> ServiceContainer:
        """Create a DI container with basic services registered.

        Returns:
            ServiceContainer with mock services
        """
        from victor.agent.protocols import (
            ResponseSanitizerProtocol,
            ToolExecutorProtocol,
        )
        from victor.agent.response_sanitizer import ResponseSanitizer
        from victor.analytics.logger import UsageLogger
        from victor.analytics.debug_logger import DebugLogger

        container = ServiceContainer()

        # Register response sanitizer
        container.register(
            ResponseSanitizerProtocol,
            lambda c: ResponseSanitizer(),
            ServiceLifetime.SINGLETON,
        )

        # Register usage logger
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            log_file = Path(f.name)

        container.register(
            type(Mock(spec=UsageLogger)),
            lambda c: Mock(spec=UsageLogger),
            ServiceLifetime.SINGLETON,
        )

        # Register debug logger
        container.register(
            type(Mock(spec=DebugLogger)),
            lambda c: Mock(spec=DebugLogger, enabled=False),
            ServiceLifetime.SINGLETON,
        )

        return container

    @staticmethod
    def create_tool_registry(tools: Optional[List[Any]] = None) -> Mock:
        """Create a mock tool registry.

        Args:
            tools: Optional list of tools to register

        Returns:
            Mock tool registry
        """
        from victor.tools.registry import ToolRegistry

        # Create real registry
        real_registry = ToolRegistry()

        # Add tools if provided
        if tools:
            for tool in tools:
                real_registry.register_tool(tool)

        # Wrap in mock for testability
        registry = Mock(spec=ToolRegistryProtocol)
        registry._real_registry = real_registry

        # Delegate methods to real registry
        registry.get_tool = lambda name: real_registry.get_tool(name)
        registry.has_tool = lambda name: real_registry.has_tool(name)
        registry.list_tools = Mock(return_value=real_registry.list_tools())

        return registry

    @staticmethod
    def create_tool_executor(
        tool_registry: Optional[Any] = None,
    ) -> Mock:
        """Create a mock tool executor.

        Args:
            tool_registry: Optional tool registry

        Returns:
            Mock tool executor
        """
        executor = Mock(spec=ToolExecutorProtocol)

        if tool_registry is None:
            tool_registry = TestFixtureFactory.create_tool_registry()

        executor.tool_registry = tool_registry

        # Mock execute_tool to return success
        async def mock_execute(tool_name: str, **kwargs):
            return {
                "tool": tool_name,
                "result": f"Executed {tool_name}",
                "success": True,
                "error": None,
            }

        executor.execute_tool = mock_execute

        return executor

    @staticmethod
    def create_message_history(
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Message]:
        """Create a message history for testing.

        Args:
            system_prompt: Optional system prompt
            messages: Optional list of messages

        Returns:
            List of Message objects
        """
        history = []

        if system_prompt:
            history.append(Message(role="system", content=system_prompt))

        if messages:
            for msg in messages:
                history.append(Message(role=msg["role"], content=msg["content"]))
        else:
            # Add default conversation
            history.append(Message(role="user", content="Hello, can you help me?"))
            history.append(
                Message(
                    role="assistant",
                    content="Of course! What would you like help with?",
                )
            )

        return history

    @staticmethod
    def create_conversation_store(
        db_path: Optional[Path] = None,
    ) -> Any:
        """Create a conversation store for testing.

        Args:
            db_path: Optional database path (uses temp file if not provided)

        Returns:
            ConversationStore instance
        """
        from victor.agent.conversation_memory import ConversationStore

        if db_path is None:
            # Create temp database
            fd, db_path = tempfile.mkstemp(suffix=".db")
            import os

            os.close(fd)

        store = ConversationStore(
            db_path=db_path,
            max_context_tokens=100000,
            response_reserve=4096,
        )

        return store

    @staticmethod
    def create_orchestrator_fixture(
        provider: Optional[Mock] = None,
        settings: Optional[Mock] = None,
        tool_registry: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Create a complete orchestrator fixture for testing.

        This fixture includes:
        - Mock provider
        - Mock settings
        - Tool registry
        - Tool executor
        - Message history
        - Conversation store

        Args:
            provider: Optional mock provider
            settings: Optional mock settings
            tool_registry: Optional tool registry

        Returns:
            Dictionary with all orchestrator components
        """
        from victor.config.settings import get_project_paths

        # Create components
        if provider is None:
            provider = MockProviderFactory.create_anthropic()

        if settings is None:
            settings = TestSettingsBuilder().build()

        if tool_registry is None:
            tool_registry = TestFixtureFactory.create_tool_registry()

        tool_executor = TestFixtureFactory.create_tool_executor(tool_registry)

        message_history = TestFixtureFactory.create_message_history(
            system_prompt="You are a helpful assistant.",
        )

        # Create conversation store
        fd, db_path = tempfile.mkstemp(suffix=".db")
        import os

        os.close(fd)
        conversation_store = TestFixtureFactory.create_conversation_store(Path(db_path))

        return {
            "provider": provider,
            "settings": settings,
            "tool_registry": tool_registry,
            "tool_executor": tool_executor,
            "message_history": message_history,
            "conversation_store": conversation_store,
            "db_path": db_path,
        }

    @staticmethod
    def create_stream_chunk(
        content: str = "chunk",
        is_final: bool = False,
        stop_reason: Optional[str] = None,
    ) -> StreamChunk:
        """Create a StreamChunk for testing streaming.

        Args:
            content: Chunk content
            is_final: Whether this is the final chunk
            stop_reason: Reason for stopping

        Returns:
            StreamChunk object
        """
        return StreamChunk(
            content=content,
            is_final=is_final,
            stop_reason=stop_reason,
        )

    @staticmethod
    def create_completion_response(
        content: str = "Response",
        role: str = "assistant",
        model: str = "test-model",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Mock:
        """Create a mock completion response.

        Args:
            content: Response content
            role: Message role
            model: Model name
            tool_calls: Optional tool calls

        Returns:
            Mock completion response
        """
        response = Mock()
        response.content = content
        response.role = role
        response.model = model
        response.stop_reason = "stop"
        response.tool_calls = tool_calls
        response.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        response.raw_response = None
        response.metadata = None
        return response


# Convenience functions for common fixtures


def create_mock_settings(**kwargs) -> Mock:
    """Create a mock settings object with default values.

    Args:
        **kwargs: Settings to override

    Returns:
        Mock settings object
    """
    builder = TestSettingsBuilder()
    for key, value in kwargs.items():
        builder.with_custom_setting(key, value)
    return builder.build()


def create_mock_provider(**kwargs) -> Mock:
    """Create a mock provider with default values.

    Args:
        **kwargs: Provider attributes to override

    Returns:
        Mock provider
    """
    return MockProviderFactory.create_anthropic(**kwargs)


def create_test_container() -> ServiceContainer:
    """Create a test DI container with basic services.

    Returns:
        ServiceContainer with mock services
    """
    return TestFixtureFactory.create_container()


def create_test_tool_registry(tools: Optional[List[Any]] = None) -> Mock:
    """Create a test tool registry.

    Args:
        tools: Optional list of tools

    Returns:
        Mock tool registry
    """
    return TestFixtureFactory.create_tool_registry(tools)
