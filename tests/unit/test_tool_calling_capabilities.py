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

"""Tests for tool calling capability loader and adapters."""


from victor.agent.tool_calling import (
    ModelCapabilityLoader,
    ToolCallingAdapterRegistry,
    get_model_capabilities,
    ToolCallFormat,
)


class TestModelCapabilityLoader:
    """Tests for ModelCapabilityLoader."""

    def test_loader_is_singleton(self):
        """Loader should be a singleton."""
        loader1 = ModelCapabilityLoader()
        loader2 = ModelCapabilityLoader()
        assert loader1 is loader2

    def test_get_provider_names(self):
        """Should return list of configured providers."""
        loader = ModelCapabilityLoader()
        providers = loader.get_provider_names()
        assert "ollama" in providers
        assert "anthropic" in providers
        assert "openai" in providers

    def test_get_model_patterns(self):
        """Should return list of configured model patterns."""
        loader = ModelCapabilityLoader()
        patterns = loader.get_model_patterns()
        assert any("llama" in p.lower() for p in patterns)
        assert any("qwen" in p.lower() for p in patterns)

    def test_ollama_provider_defaults(self):
        """Ollama provider should have expected defaults."""
        caps = get_model_capabilities("ollama")
        # Ollama default without specific model - uses provider defaults
        assert caps.json_fallback_parsing is True
        assert caps.xml_fallback_parsing is True

    def test_anthropic_provider_defaults(self):
        """Anthropic provider should have native tool support."""
        caps = get_model_capabilities("anthropic")
        assert caps.native_tool_calls is True
        assert caps.streaming_tool_calls is True
        assert caps.parallel_tool_calls is True

    def test_llama31_model_override(self):
        """Llama 3.1 should have native tool support via model pattern."""
        caps = get_model_capabilities("ollama", "llama3.1:8b")
        assert caps.native_tool_calls is True
        assert caps.requires_strict_prompting is False
        assert caps.recommended_tool_budget == 15

    def test_qwen25_model_override(self):
        """Qwen 2.5 should have native tool support."""
        caps = get_model_capabilities("ollama", "qwen2.5:7b")
        assert caps.native_tool_calls is True
        assert caps.requires_strict_prompting is False

    def test_qwen3_thinking_mode(self):
        """Qwen 3 should have thinking mode enabled."""
        caps = get_model_capabilities("ollama", "qwen3:4b")
        assert caps.thinking_mode is True

    def test_codellama_limited_support(self):
        """CodeLlama should have limited tool support."""
        caps = get_model_capabilities("ollama", "codellama:7b")
        assert caps.native_tool_calls is False
        assert caps.requires_strict_prompting is True
        assert caps.recommended_tool_budget == 8

    def test_vllm_provider_defaults(self):
        """vLLM should have robust tool calling."""
        caps = get_model_capabilities("vllm", "meta-llama/Llama-3.1-8B")
        assert caps.native_tool_calls is True
        assert caps.streaming_tool_calls is True
        assert caps.tool_choice_param is True


class TestToolCallingAdapterRegistry:
    """Tests for ToolCallingAdapterRegistry."""

    def test_get_ollama_adapter(self):
        """Should return OllamaToolCallingAdapter for ollama."""
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", "llama3.1:8b")
        assert adapter.provider_name == "ollama"
        caps = adapter.get_capabilities()
        assert caps.native_tool_calls is True

    def test_get_anthropic_adapter(self):
        """Should return AnthropicToolCallingAdapter for anthropic."""
        adapter = ToolCallingAdapterRegistry.get_adapter("anthropic", "claude-3-opus")
        assert adapter.provider_name == "anthropic"
        caps = adapter.get_capabilities()
        assert caps.tool_call_format == ToolCallFormat.ANTHROPIC

    def test_get_openai_adapter(self):
        """Should return OpenAIToolCallingAdapter for openai."""
        adapter = ToolCallingAdapterRegistry.get_adapter("openai", "gpt-4")
        assert adapter.provider_name == "openai"
        caps = adapter.get_capabilities()
        assert caps.tool_call_format == ToolCallFormat.OPENAI

    def test_get_lmstudio_adapter(self):
        """Should return OpenAICompatToolCallingAdapter for lmstudio."""
        adapter = ToolCallingAdapterRegistry.get_adapter("lmstudio", "qwen2.5-coder")
        assert adapter.provider_name == "lmstudio"

    def test_get_vllm_adapter(self):
        """Should return OpenAICompatToolCallingAdapter for vllm."""
        adapter = ToolCallingAdapterRegistry.get_adapter("vllm", "mistral-7b")
        assert adapter.provider_name == "vllm"
        caps = adapter.get_capabilities()
        assert caps.tool_call_format == ToolCallFormat.VLLM

    def test_unknown_provider_fallback(self):
        """Unknown provider should fall back to OpenAI-compatible."""
        adapter = ToolCallingAdapterRegistry.get_adapter("custom_provider", "some-model")
        # Should not raise, returns OpenAI-compatible adapter
        assert adapter is not None

    def test_list_providers(self):
        """Should list registered providers."""
        providers = ToolCallingAdapterRegistry.list_providers()
        assert "ollama" in providers
        assert "anthropic" in providers

    def test_is_registered(self):
        """Should check if provider is registered."""
        assert ToolCallingAdapterRegistry.is_registered("ollama") is True
        assert ToolCallingAdapterRegistry.is_registered("anthropic") is True


class TestAdapterSystemPromptHints:
    """Tests for system prompt hints generation."""

    def test_ollama_native_hints(self):
        """Ollama with native support should have tool usage hints."""
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", "llama3.1:8b")
        hints = adapter.get_system_prompt_hints()
        assert "TOOL USAGE" in hints
        assert "one at a time" in hints.lower()

    def test_ollama_fallback_hints(self):
        """Ollama without native support should have strict hints."""
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", "codellama:7b")
        hints = adapter.get_system_prompt_hints()
        assert "CRITICAL" in hints
        assert "ONE AT A TIME" in hints

    def test_qwen3_thinking_hints(self):
        """Qwen3 should include thinking mode hints."""
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", "qwen3:4b")
        hints = adapter.get_system_prompt_hints()
        assert "QWEN3" in hints or "/no_think" in hints

    def test_anthropic_no_hints(self):
        """Anthropic should have no hints (native support)."""
        adapter = ToolCallingAdapterRegistry.get_adapter("anthropic", "claude-3-opus")
        hints = adapter.get_system_prompt_hints()
        assert hints == ""

    def test_openai_no_hints(self):
        """OpenAI should have no hints (native support)."""
        adapter = ToolCallingAdapterRegistry.get_adapter("openai", "gpt-4")
        hints = adapter.get_system_prompt_hints()
        assert hints == ""

    def test_vllm_has_hints(self):
        """vLLM should have tool calling hints."""
        adapter = ToolCallingAdapterRegistry.get_adapter("vllm", "llama-3.1-8b")
        hints = adapter.get_system_prompt_hints()
        assert "TOOL" in hints
