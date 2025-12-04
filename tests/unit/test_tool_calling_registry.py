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

"""Tests for tool calling adapter registry."""


from victor.agent.tool_calling.registry import ToolCallingAdapterRegistry
from victor.agent.tool_calling.base import BaseToolCallingAdapter
from victor.agent.tool_calling.adapters import (
    AnthropicToolCallingAdapter,
    GoogleToolCallingAdapter,
    LMStudioToolCallingAdapter,
    OllamaToolCallingAdapter,
    OpenAICompatToolCallingAdapter,
    OpenAIToolCallingAdapter,
)


class TestToolCallingAdapterRegistry:
    """Tests for ToolCallingAdapterRegistry class."""

    def setup_method(self):
        """Reset registry before each test."""
        # Clear the registry to ensure clean state
        ToolCallingAdapterRegistry._adapters = {}

    def test_register_adapter(self):
        """Test registering a custom adapter (covers lines 54-55)."""

        # Create a mock adapter class
        class CustomAdapter(BaseToolCallingAdapter):
            @property
            def provider_name(self) -> str:
                return "custom"

        # Register it
        ToolCallingAdapterRegistry.register("custom_provider", CustomAdapter)

        # Verify it was registered
        assert "custom_provider" in ToolCallingAdapterRegistry._adapters
        assert ToolCallingAdapterRegistry._adapters["custom_provider"] == CustomAdapter

    def test_register_adapter_case_insensitive(self):
        """Test that registration is case-insensitive."""

        class CustomAdapter(BaseToolCallingAdapter):
            @property
            def provider_name(self) -> str:
                return "custom"

        ToolCallingAdapterRegistry.register("MyProvider", CustomAdapter)
        assert "myprovider" in ToolCallingAdapterRegistry._adapters

    def test_get_adapter_anthropic(self):
        """Test getting Anthropic adapter."""
        adapter = ToolCallingAdapterRegistry.get_adapter("anthropic", model="claude-3")
        assert isinstance(adapter, AnthropicToolCallingAdapter)

    def test_get_adapter_openai(self):
        """Test getting OpenAI adapter."""
        adapter = ToolCallingAdapterRegistry.get_adapter("openai", model="gpt-4")
        assert isinstance(adapter, OpenAIToolCallingAdapter)

    def test_get_adapter_ollama(self):
        """Test getting Ollama adapter."""
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", model="llama3.1:8b")
        assert isinstance(adapter, OllamaToolCallingAdapter)

    def test_get_adapter_lmstudio(self):
        """Test getting LMStudio adapter (dedicated adapter)."""
        adapter = ToolCallingAdapterRegistry.get_adapter("lmstudio", model="local-model")
        assert isinstance(adapter, LMStudioToolCallingAdapter)
        assert adapter.provider_name == "lmstudio"

    def test_get_adapter_vllm(self):
        """Test getting vLLM adapter (OpenAI-compatible)."""
        adapter = ToolCallingAdapterRegistry.get_adapter("vllm", model="local-model")
        assert isinstance(adapter, OpenAICompatToolCallingAdapter)
        assert adapter.provider_name == "vllm"

    def test_get_adapter_google(self):
        """Test getting Google adapter uses dedicated GoogleToolCallingAdapter."""
        adapter = ToolCallingAdapterRegistry.get_adapter("google", model="gemini-pro")
        assert isinstance(adapter, GoogleToolCallingAdapter)

    def test_get_adapter_xai(self):
        """Test getting xAI adapter uses OpenAI format (covers lines 116-117)."""
        adapter = ToolCallingAdapterRegistry.get_adapter("xai", model="grok-1")
        assert isinstance(adapter, OpenAIToolCallingAdapter)

    def test_get_adapter_unknown_provider(self):
        """Test unknown provider falls back to OpenAI-compatible."""
        adapter = ToolCallingAdapterRegistry.get_adapter("unknown_provider", model="some-model")
        assert isinstance(adapter, OpenAICompatToolCallingAdapter)
        assert adapter.provider_name == "unknown_provider"

    def test_get_adapter_case_insensitive(self):
        """Test get_adapter is case-insensitive."""
        adapter = ToolCallingAdapterRegistry.get_adapter("OPENAI", model="gpt-4")
        assert isinstance(adapter, OpenAIToolCallingAdapter)

    def test_get_adapter_with_config(self):
        """Test get_adapter passes config."""
        config = {"timeout": 30}
        adapter = ToolCallingAdapterRegistry.get_adapter(
            "ollama", model="llama3.1:8b", config=config
        )
        assert isinstance(adapter, OllamaToolCallingAdapter)
        # Config is passed during construction

    def test_list_providers(self):
        """Test listing registered providers."""
        providers = ToolCallingAdapterRegistry.list_providers()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "ollama" in providers
        assert "lmstudio" in providers
        assert "vllm" in providers

    def test_is_registered_true(self):
        """Test is_registered returns True for registered providers."""
        assert ToolCallingAdapterRegistry.is_registered("anthropic") is True
        assert ToolCallingAdapterRegistry.is_registered("openai") is True
        assert ToolCallingAdapterRegistry.is_registered("ollama") is True

    def test_is_registered_false(self):
        """Test is_registered returns False for unknown providers."""
        assert ToolCallingAdapterRegistry.is_registered("nonexistent_provider") is False

    def test_is_registered_case_insensitive(self):
        """Test is_registered is case-insensitive."""
        assert ToolCallingAdapterRegistry.is_registered("ANTHROPIC") is True
        assert ToolCallingAdapterRegistry.is_registered("Ollama") is True

    def test_is_registered_lazy_init(self):
        """Test is_registered triggers lazy initialization (covers line 143)."""
        # Clear the registry completely
        ToolCallingAdapterRegistry._adapters = {}

        # Call is_registered - should trigger lazy init
        result = ToolCallingAdapterRegistry.is_registered("ollama")

        # Should return True after lazy init
        assert result is True
        # Adapters should now be populated
        assert len(ToolCallingAdapterRegistry._adapters) > 0

    def test_default_adapters_populated(self):
        """Test default adapters are populated on first get_adapter call."""
        # Clear registry
        ToolCallingAdapterRegistry._adapters = {}

        # Make a call
        ToolCallingAdapterRegistry.get_adapter("ollama")

        # Check defaults are populated
        assert "anthropic" in ToolCallingAdapterRegistry._adapters
        assert "openai" in ToolCallingAdapterRegistry._adapters
        assert "ollama" in ToolCallingAdapterRegistry._adapters
        assert "lmstudio" in ToolCallingAdapterRegistry._adapters
        assert "vllm" in ToolCallingAdapterRegistry._adapters


class TestToolCallingAdapterRegistryEdgeCases:
    """Edge case tests for ToolCallingAdapterRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        ToolCallingAdapterRegistry._adapters = {}

    def test_register_overwrites_existing(self):
        """Test registering same provider twice overwrites."""

        class AdapterA(BaseToolCallingAdapter):
            @property
            def provider_name(self) -> str:
                return "test"

        class AdapterB(BaseToolCallingAdapter):
            @property
            def provider_name(self) -> str:
                return "test"

        ToolCallingAdapterRegistry.register("test", AdapterA)
        ToolCallingAdapterRegistry.register("test", AdapterB)

        assert ToolCallingAdapterRegistry._adapters["test"] == AdapterB

    def test_get_adapter_empty_model(self):
        """Test get_adapter with empty model string."""
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", model="")
        assert isinstance(adapter, OllamaToolCallingAdapter)

    def test_get_adapter_none_config(self):
        """Test get_adapter with None config."""
        adapter = ToolCallingAdapterRegistry.get_adapter("anthropic", model="claude-3", config=None)
        assert isinstance(adapter, AnthropicToolCallingAdapter)

    def test_multiple_unknown_providers(self):
        """Test multiple unknown providers get unique adapters."""
        adapter1 = ToolCallingAdapterRegistry.get_adapter("provider_a", model="model1")
        adapter2 = ToolCallingAdapterRegistry.get_adapter("provider_b", model="model2")

        assert adapter1.provider_name == "provider_a"
        assert adapter2.provider_name == "provider_b"
        assert adapter1 is not adapter2
