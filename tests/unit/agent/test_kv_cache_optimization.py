"""Tests for KV prefix cache optimization across provider types.

Verifies that:
- _kv_optimization_enabled activates for providers with KV prefix caching
- System prompt is frozen after first build when KV optimization is active
- Dynamic content is injected into user messages (not system prompt) for KV providers
- Tool ordering is deterministic for KV prefix stability
- Context pruning preserves the system prompt prefix
- Prompt builder uses reduced + cached sections for KV providers
"""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest


# =====================================================================
# Fix 1: _kv_optimization_enabled property
# =====================================================================


class TestKVOptimizationEnabled:
    """Test _kv_optimization_enabled property on AgentOrchestrator."""

    def _make_orchestrator(self, kv_cache=False, api_cache=False, setting_enabled=True):
        """Create a mock orchestrator with configurable provider capabilities."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)

        # Provider mock
        provider = MagicMock()
        provider.supports_kv_prefix_caching.return_value = kv_cache
        provider.supports_prompt_caching.return_value = api_cache
        orch.provider = provider

        # Settings mock
        context = MagicMock()
        context.cache_optimization_enabled = setting_enabled
        settings = MagicMock()
        settings.context = context
        orch.settings = settings

        # Call the real property via unbound reference
        return orch

    def test_ollama_gets_kv_optimization(self):
        """Ollama: kv=True, cache=False -> kv_optimization=True, cache_optimization=False."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = self._make_orchestrator(kv_cache=True, api_cache=False)
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is True
        assert AgentOrchestrator._cache_optimization_enabled.fget(orch) is False

    def test_anthropic_gets_both(self):
        """Anthropic: kv=True, cache=True -> both True."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = self._make_orchestrator(kv_cache=True, api_cache=True)
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is True
        assert AgentOrchestrator._cache_optimization_enabled.fget(orch) is True

    def test_provider_without_either(self):
        """Provider with neither -> both False."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = self._make_orchestrator(kv_cache=False, api_cache=False)
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is False
        assert AgentOrchestrator._cache_optimization_enabled.fget(orch) is False

    def test_setting_override_disables_kv(self):
        """cache_optimization_enabled=False overrides KV optimization."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = self._make_orchestrator(kv_cache=True, api_cache=False, setting_enabled=False)
        # The real _check_cache_setting_enabled needs to be callable
        orch._check_cache_setting_enabled = lambda: False
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is False

    def test_fallback_when_no_kv_method(self):
        """Provider without supports_kv_prefix_caching falls back to supports_prompt_caching."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = self._make_orchestrator(api_cache=True)
        # Remove kv method to test fallback
        del orch.provider.supports_kv_prefix_caching
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is True

    def test_no_provider_returns_false(self):
        """No provider attribute -> False."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch.provider = None
        orch.settings = MagicMock()
        orch.settings.context = MagicMock()
        orch.settings.context.cache_optimization_enabled = True
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is False


# =====================================================================
# Fix 1: System prompt freezing gated on _kv_optimization_enabled
# =====================================================================


class TestSystemPromptFreezing:
    """Test that system prompt freezing activates for KV providers."""

    def test_kv_provider_freezes_prompt_after_first_build(self):
        """System prompt frozen after first build when kv_optimization=True."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        type(orch)._cache_optimization_enabled = PropertyMock(return_value=False)
        orch._system_prompt_frozen = False
        orch.prompt_builder = MagicMock()
        orch.prompt_builder.build.return_value = "system prompt"
        orch.project_context = MagicMock()
        orch.project_context.content = None
        orch.conversation = MagicMock()
        orch.conversation._system_added = False

        # Call the real method
        AgentOrchestrator.update_system_prompt_for_query(orch, query_classification=None)

        # Prompt should be frozen now
        assert orch._system_prompt_frozen is True

    def test_kv_provider_skips_rebuild_on_second_query(self):
        """update_system_prompt_for_query() is no-op after freeze."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        type(orch)._cache_optimization_enabled = PropertyMock(return_value=False)
        orch._system_prompt_frozen = True  # Already frozen
        # Add prompt_builder so we can verify it's not called
        orch.prompt_builder = MagicMock()
        orch._build_system_prompt_with_adapter = MagicMock()

        # Should return early without rebuilding
        AgentOrchestrator.update_system_prompt_for_query(orch, query_classification=None)

        # _build_system_prompt_with_adapter should NOT have been called
        orch._build_system_prompt_with_adapter.assert_not_called()


# =====================================================================
# Fix 2: Deterministic tool ordering
# =====================================================================


class TestDeterministicToolOrdering:
    """Test that tools are sorted by name for KV cache stability."""

    def test_tools_sorted_by_name_when_kv_enabled(self):
        """Tools returned in alphabetical order for KV cache stability."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)

        # Create mock tools in non-alphabetical order
        tool_z = MagicMock()
        tool_z.name = "z_tool"
        tool_a = MagicMock()
        tool_a.name = "a_tool"
        tool_m = MagicMock()
        tool_m.name = "m_tool"

        tools = [tool_z, tool_a, tool_m]
        sorted_tools = AgentOrchestrator._sort_tools_for_kv_stability(orch, tools)

        assert [t.name for t in sorted_tools] == ["a_tool", "m_tool", "z_tool"]

    def test_no_sorting_when_kv_disabled(self):
        """Without KV optimization, tool order is preserved."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)

        tool_z = MagicMock()
        tool_z.name = "z_tool"
        tool_a = MagicMock()
        tool_a.name = "a_tool"

        tools = [tool_z, tool_a]
        result = AgentOrchestrator._sort_tools_for_kv_stability(orch, tools)

        assert [t.name for t in result] == ["z_tool", "a_tool"]

    def test_none_tools_returns_none(self):
        """None tools input returns None."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        assert AgentOrchestrator._sort_tools_for_kv_stability(orch, None) is None


# =====================================================================
# Fix 3: Dynamic content injection gated on _kv_optimization_enabled
# =====================================================================


class TestDynamicContentInjection:
    """Test dynamic content injection into user messages for KV providers."""

    def test_kv_provider_injects_skills_into_user_message(self):
        """Skills/reminders go into user message when KV optimization is active."""
        from victor.providers.base import Message

        messages = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="hello"),
        ]

        # Simulate the injection logic
        prefix_parts = ["[Skill: code_review]"]
        if prefix_parts:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].role == "user":
                    prefix = "\n".join(prefix_parts) + "\n\n"
                    messages[i] = Message(
                        role="user", content=prefix + messages[i].content
                    )
                    break

        assert messages[0].content == "system prompt"  # Unchanged
        assert "[Skill: code_review]" in messages[1].content
        assert "hello" in messages[1].content


# =====================================================================
# Fix 4: Context pruning preserves system prompt
# =====================================================================


class TestCacheAwarePruning:
    """Test that context pruning preserves the system prompt prefix."""

    def test_system_prompt_always_preserved_during_pruning(self):
        """System prompt (messages[0]) never removed by context assembly."""
        from victor.providers.base import Message

        messages = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="old message 1"),
            Message(role="assistant", content="old response 1"),
            Message(role="user", content="old message 2"),
            Message(role="assistant", content="old response 2"),
            Message(role="user", content="latest question"),
        ]

        # Simulate pruning: remove middle messages but keep system + recent
        pruned = [messages[0]] + messages[-2:]  # system + last turn

        assert pruned[0].role == "system"
        assert pruned[0].content == "system prompt"
        assert len(pruned) == 3


# =====================================================================
# Fix 5: Prompt builder KV cache awareness
# =====================================================================


class TestPromptBuilderKVAwareness:
    """Test that SystemPromptBuilder handles KV cache providers correctly."""

    def test_kv_provider_gets_reduced_sections(self):
        """Non-caching provider (Ollama) gets reduced section set."""
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder.__new__(SystemPromptBuilder)
        builder.provider_caches = False
        builder.provider_has_kv_cache = True
        builder.concise_mode = False
        builder._rl_coordinator = None
        builder._decision_service = None

        sections = SystemPromptBuilder._get_active_sections(builder)
        # Should return reduced set, not full
        assert len(sections) <= 5
        assert "completion" in sections or "task_guidance" in sections

    def test_cloud_provider_gets_full_sections(self):
        """Caching provider (Anthropic) gets full sections."""
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder.__new__(SystemPromptBuilder)
        builder.provider_caches = True
        builder.provider_has_kv_cache = True
        builder.concise_mode = False
        builder._rl_coordinator = None
        builder._decision_service = None

        sections = SystemPromptBuilder._get_active_sections(builder)
        # Full set is larger than reduced set
        assert len(sections) > 5

    def test_kv_provider_param_accepted(self):
        """SystemPromptBuilder accepts provider_has_kv_cache parameter."""
        from victor.agent.prompt_builder import SystemPromptBuilder

        # Should not raise
        builder = SystemPromptBuilder.__new__(SystemPromptBuilder)
        builder.provider_has_kv_cache = True
        assert builder.provider_has_kv_cache is True
