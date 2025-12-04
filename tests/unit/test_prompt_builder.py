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

"""Tests for SystemPromptBuilder module."""

from unittest.mock import MagicMock

from victor.agent.prompt_builder import (
    SystemPromptBuilder,
    build_system_prompt,
    CLOUD_PROVIDERS,
    LOCAL_PROVIDERS,
)


class TestSystemPromptBuilder:
    """Tests for SystemPromptBuilder class."""

    def test_is_cloud_provider_returns_true_for_cloud(self):
        """Test cloud provider detection."""
        for provider in ["anthropic", "openai", "google", "xai"]:
            builder = SystemPromptBuilder(provider_name=provider, model="test")
            assert builder.is_cloud_provider()
            assert not builder.is_local_provider()

    def test_is_local_provider_returns_true_for_local(self):
        """Test local provider detection."""
        for provider in ["ollama", "lmstudio", "vllm"]:
            builder = SystemPromptBuilder(provider_name=provider, model="test")
            assert builder.is_local_provider()
            assert not builder.is_cloud_provider()

    def test_is_cloud_provider_case_insensitive(self):
        """Test provider detection is case insensitive."""
        builder = SystemPromptBuilder(provider_name="ANTHROPIC", model="test")
        assert builder.is_cloud_provider()

    def test_has_native_tool_support_for_qwen(self):
        """Test native tool support detection for Qwen models."""
        builder = SystemPromptBuilder(provider_name="ollama", model="qwen2.5-coder:7b")
        assert builder.has_native_tool_support()

        builder = SystemPromptBuilder(provider_name="ollama", model="qwen3:30b")
        assert builder.has_native_tool_support()

    def test_has_native_tool_support_for_llama(self):
        """Test native tool support detection for Llama models."""
        builder = SystemPromptBuilder(provider_name="ollama", model="llama3.1:8b")
        assert builder.has_native_tool_support()

        builder = SystemPromptBuilder(provider_name="ollama", model="llama-3.2:3b")
        assert builder.has_native_tool_support()

    def test_has_native_tool_support_for_mistral(self):
        """Test native tool support detection for Mistral models."""
        builder = SystemPromptBuilder(provider_name="ollama", model="mistral:7b")
        assert builder.has_native_tool_support()

        builder = SystemPromptBuilder(provider_name="ollama", model="mixtral:8x7b")
        assert builder.has_native_tool_support()

    def test_no_native_tool_support_for_other_models(self):
        """Test that other models are not marked as having native support."""
        builder = SystemPromptBuilder(provider_name="ollama", model="codellama:7b")
        assert not builder.has_native_tool_support()

        builder = SystemPromptBuilder(provider_name="ollama", model="phi:2b")
        assert not builder.has_native_tool_support()

    def test_build_returns_string(self):
        """Test that build() returns a string."""
        builder = SystemPromptBuilder(provider_name="anthropic", model="claude-3")
        result = builder.build()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_cloud_prompt_mentions_tools(self):
        """Test that cloud prompt mentions tool usage."""
        builder = SystemPromptBuilder(provider_name="anthropic", model="claude-3")
        result = builder._build_cloud_prompt()
        assert "tool" in result.lower()
        assert "read_file" in result or "list_directory" in result

    def test_build_with_adapter_uses_hints(self):
        """Test that adapter hints are used when available."""
        mock_adapter = MagicMock()
        mock_adapter.get_system_prompt_hints.return_value = "Custom adapter hints"
        mock_adapter.get_capabilities.return_value = None

        builder = SystemPromptBuilder(
            provider_name="test",
            model="test",
            tool_adapter=mock_adapter,
        )
        result = builder._build_with_adapter()
        assert "Custom adapter hints" in result

    def test_build_ollama_prompt_native_support(self):
        """Test Ollama prompt for models with native support."""
        builder = SystemPromptBuilder(provider_name="ollama", model="qwen2.5:7b")
        result = builder._build_ollama_prompt()
        assert "tool calling capability" in result.lower()
        assert "JSON" in result or "json" in result

    def test_build_ollama_prompt_no_native_support(self):
        """Test Ollama prompt for models without native support."""
        builder = SystemPromptBuilder(provider_name="ollama", model="codellama:7b")
        result = builder._build_ollama_prompt()
        assert "EXACTLY" in result or "ONE AT A TIME" in result

    def test_build_ollama_prompt_qwen3_includes_thinking(self):
        """Test Qwen3 prompt includes thinking mode guidance."""
        builder = SystemPromptBuilder(provider_name="ollama", model="qwen3:30b")
        result = builder._build_ollama_prompt()
        assert "QWEN3" in result or "/no_think" in result

    def test_build_lmstudio_prompt_native_support(self):
        """Test LMStudio prompt for models with native support."""
        builder = SystemPromptBuilder(provider_name="lmstudio", model="qwen2.5-coder:7b")
        result = builder._build_lmstudio_prompt()
        assert "native" in result.lower()

    def test_build_vllm_prompt(self):
        """Test vLLM prompt."""
        builder = SystemPromptBuilder(provider_name="vllm", model="mistral")
        result = builder._build_vllm_prompt()
        assert "OpenAI-compatible" in result

    def test_build_default_prompt(self):
        """Test default prompt for unknown providers."""
        builder = SystemPromptBuilder(provider_name="unknown", model="test")
        result = builder._build_default_prompt()
        assert "code analyst" in result.lower()
        assert "tool" in result.lower()


class TestBuildSystemPromptFunction:
    """Tests for the build_system_prompt convenience function."""

    def test_convenience_function_returns_string(self):
        """Test that the convenience function works."""
        result = build_system_prompt(
            provider_name="anthropic",
            model="claude-3",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_convenience_function_with_adapter(self):
        """Test convenience function with adapter."""
        mock_adapter = MagicMock()
        mock_adapter.get_system_prompt_hints.return_value = "Hints from adapter"
        mock_adapter.get_capabilities.return_value = None

        result = build_system_prompt(
            provider_name="test",
            model="test",
            tool_adapter=mock_adapter,
        )
        assert "Hints from adapter" in result


class TestProviderConstants:
    """Tests for provider constants."""

    def test_cloud_providers_set(self):
        """Test CLOUD_PROVIDERS constant."""
        assert "anthropic" in CLOUD_PROVIDERS
        assert "openai" in CLOUD_PROVIDERS
        assert "google" in CLOUD_PROVIDERS
        assert "xai" in CLOUD_PROVIDERS

    def test_local_providers_set(self):
        """Test LOCAL_PROVIDERS constant."""
        assert "ollama" in LOCAL_PROVIDERS
        assert "lmstudio" in LOCAL_PROVIDERS
        assert "vllm" in LOCAL_PROVIDERS


class TestSystemPromptBuilderEdgeCases:
    """Edge case tests for SystemPromptBuilder."""

    def test_build_for_vllm_provider(self):
        """Test _build_for_provider for vllm (covers line 157-158)."""
        builder = SystemPromptBuilder(provider_name="vllm", model="test")
        result = builder._build_for_provider()
        assert "OpenAI-compatible" in result

    def test_build_for_lmstudio_provider(self):
        """Test _build_for_provider for lmstudio (covers lines 161-162)."""
        builder = SystemPromptBuilder(provider_name="lmstudio", model="qwen2.5:7b")
        result = builder._build_for_provider()
        assert isinstance(result, str)

    def test_build_for_ollama_provider(self):
        """Test _build_for_provider for ollama (covers lines 165-166)."""
        builder = SystemPromptBuilder(provider_name="ollama", model="llama3.1:8b")
        result = builder._build_for_provider()
        assert isinstance(result, str)

    def test_build_for_unknown_provider(self):
        """Test _build_for_provider for unknown provider (covers lines 168-169)."""
        builder = SystemPromptBuilder(provider_name="totally_unknown", model="test")
        result = builder._build_for_provider()
        assert "code analyst" in result.lower()

    def test_build_lmstudio_prompt_no_native_support(self):
        """Test LMStudio prompt for models without native support (covers line 224)."""
        builder = SystemPromptBuilder(provider_name="lmstudio", model="codellama:7b")
        result = builder._build_lmstudio_prompt()
        assert "ONE AT A TIME" in result
        assert "CRITICAL RULES" in result

    def test_build_with_adapter_no_hints_no_caps(self):
        """Test _build_with_adapter includes base_prompt and grounding rules when no hints/caps."""
        mock_adapter = MagicMock()
        mock_adapter.get_system_prompt_hints.return_value = None
        mock_adapter.get_capabilities.return_value = None

        builder = SystemPromptBuilder(
            provider_name="test",
            model="test",
            tool_adapter=mock_adapter,
        )
        result = builder._build_with_adapter()
        # Should include base prompt and grounding rules
        assert "You are a code analyst for this repository." in result
        assert "CRITICAL - TOOL OUTPUT GROUNDING:" in result
