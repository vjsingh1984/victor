"""Tests for provider-aware tiered tool selection strategy.

Verifies that:
- BaseProvider.supports_prompt_caching() defaults to False
- AnthropicProvider overrides to True
- _cache_optimization_enabled derives from provider capability
- Q&A detection heuristic is accurate
- Streaming pipeline skips tools for Q&A tasks
"""

from unittest.mock import MagicMock, patch

import pytest

from victor.agent.coordinators.execution_coordinator import ExecutionCoordinator


# =====================================================================
# Step 1-2: Provider capability detection
# =====================================================================


class TestSupportsPromptCaching:
    """Test supports_prompt_caching() on BaseProvider and subclasses."""

    def test_base_provider_defaults_to_false(self):
        from victor.providers.base import BaseProvider

        # BaseProvider is abstract; test via a mock that inherits the default method
        mock_provider = MagicMock(spec=BaseProvider)
        # Call the real method via unbound reference
        assert BaseProvider.supports_prompt_caching(mock_provider) is False

    def test_anthropic_provider_returns_true(self):
        from victor.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        assert provider.supports_prompt_caching() is True

    @pytest.mark.parametrize(
        "provider_module,provider_class",
        [
            ("victor.providers.openai_provider", "OpenAIProvider"),
            ("victor.providers.google_provider", "GoogleProvider"),
            ("victor.providers.deepseek_provider", "DeepSeekProvider"),
            ("victor.providers.xai_provider", "XAIProvider"),
            ("victor.providers.groq_provider", "GroqProvider"),
            ("victor.providers.bedrock_provider", "BedrockProvider"),
            ("victor.providers.vertex_provider", "VertexAIProvider"),
            ("victor.providers.azure_openai_provider", "AzureOpenAIProvider"),
            ("victor.providers.openrouter_provider", "OpenRouterProvider"),
            # Local inference engines (KV prefix caching for latency savings)
            ("victor.providers.ollama_provider", "OllamaProvider"),
            ("victor.providers.vllm_provider", "VLLMProvider"),
            ("victor.providers.lmstudio_provider", "LMStudioProvider"),
            ("victor.providers.llamacpp_provider", "LlamaCppProvider"),
            ("victor.providers.mlx_provider", "MLXProvider"),
            # Cloud inference with auto-caching
            ("victor.providers.fireworks_provider", "FireworksProvider"),
            ("victor.providers.together_provider", "TogetherProvider"),
            ("victor.providers.cerebras_provider", "CerebrasProvider"),
        ],
    )
    def test_caching_providers_return_true(self, provider_module, provider_class):
        """All cloud providers with cached token discounts should return True."""
        import importlib

        mod = importlib.import_module(provider_module)
        cls = getattr(mod, provider_class)
        # Call unbound method to avoid __init__ side effects
        assert cls.supports_prompt_caching(MagicMock()) is True

    def test_is_caching_provider_convenience(self):
        from victor.providers.base import is_caching_provider

        mock_caching = MagicMock()
        mock_caching.supports_prompt_caching.return_value = True
        assert is_caching_provider(mock_caching) is True

        mock_no_cache = MagicMock()
        mock_no_cache.supports_prompt_caching.return_value = False
        assert is_caching_provider(mock_no_cache) is False

    def test_is_caching_provider_no_method(self):
        from victor.providers.base import is_caching_provider

        plain_obj = object()
        assert is_caching_provider(plain_obj) is False


# =====================================================================
# Step 3: _cache_optimization_enabled wiring
# =====================================================================


class TestCacheOptimizationWiring:
    """Test that _cache_optimization_enabled derives from provider capability."""

    def _make_orch_mock(self, provider_caches: bool, setting_enabled: bool = True):
        """Create a mock orchestrator with the real _cache_optimization_enabled property."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        # Set up provider
        orch.provider = MagicMock()
        orch.provider.supports_prompt_caching.return_value = provider_caches
        # Set up settings
        orch.settings = MagicMock()
        orch.settings.context = MagicMock()
        orch.settings.context.cache_optimization_enabled = setting_enabled
        # Use the real property
        type(orch)._cache_optimization_enabled = AgentOrchestrator._cache_optimization_enabled
        return orch

    def test_disabled_for_non_caching_provider(self):
        orch = self._make_orch_mock(provider_caches=False)
        assert orch._cache_optimization_enabled is False

    def test_enabled_for_caching_provider(self):
        orch = self._make_orch_mock(provider_caches=True)
        assert orch._cache_optimization_enabled is True

    def test_setting_override_disables(self):
        orch = self._make_orch_mock(provider_caches=True, setting_enabled=False)
        assert orch._cache_optimization_enabled is False

    def test_no_provider_defaults_false(self):
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch.provider = None
        orch.settings = None
        type(orch)._cache_optimization_enabled = AgentOrchestrator._cache_optimization_enabled
        assert orch._cache_optimization_enabled is False


# =====================================================================
# Step 7: Q&A detection accuracy
# =====================================================================


class TestQADetection:
    """Test _is_question_only heuristic accuracy."""

    @pytest.mark.parametrize(
        "msg",
        [
            "What is 2 + 2?",
            "How does Python handle memory?",
            "Explain decorators in Python",
            "What is the difference between list and tuple?",
            "Tell me about async/await",
            "Reply with just the number 4.",
            "What are the benefits of Rust?",
            "How can I learn machine learning?",
            "Why is the sky blue?",
            "Describe the observer pattern",
            "Who is Guido van Rossum?",
            "Define polymorphism",
            "Say hello",
        ],
    )
    def test_qa_detected(self, msg):
        assert ExecutionCoordinator._is_question_only(msg) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "Fix the bug in main.py",
            "Create a new test file",
            "Refactor the database module",
            "Add error handling to the login function",
            "Write a function to sort an array",
            "Run the tests and fix failures",
            "Update the README",
            "Debug the authentication issue",
            "Delete the old migration files",
            "Install the missing dependency",
            "Build a REST API endpoint",
            "Deploy to staging",
            "Configure the CI pipeline",
            "Look at the code in utils.py and tell me whats wrong",
            "Implement a binary search tree",
        ],
    )
    def test_action_not_qa(self, msg):
        assert ExecutionCoordinator._is_question_only(msg) is False

    def test_empty_string_not_qa(self):
        assert ExecutionCoordinator._is_question_only("") is False

    def test_long_question_with_action_not_qa(self):
        msg = "Can you fix the broken authentication in the login page?"
        assert ExecutionCoordinator._is_question_only(msg) is False


# =====================================================================
# Streaming pipeline Q&A bypass
# =====================================================================


class TestStreamingPipelineQABypass:
    """Test that streaming pipeline skips tools for Q&A tasks."""

    def test_qa_context_flag_default_false(self):
        from victor.agent.streaming.context import StreamingChatContext

        ctx = StreamingChatContext(user_message="test")
        assert ctx.is_qa_task is False

    def test_qa_context_flag_can_be_set(self):
        from victor.agent.streaming.context import StreamingChatContext

        ctx = StreamingChatContext(user_message="What is 2+2?", is_qa_task=True)
        assert ctx.is_qa_task is True
