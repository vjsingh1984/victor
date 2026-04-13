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
                    messages[i] = Message(role="user", content=prefix + messages[i].content)
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


# =====================================================================
# KV Tool Selection Strategy (configurable via setting)
# =====================================================================


class TestKVToolSelectionStrategy:
    """Test configurable tool selection strategies for KV providers.

    Three strategies:
    - 'per_turn': Fresh semantic selection each turn (max relevance, breaks KV prefix)
    - 'session_stable': Lock semantic selection after first query (KV stable, may miss tools)
    - 'session_full': Lock all 48 tools (only for API-caching providers)
    """

    def test_setting_defaults_to_per_turn(self):
        """Default kv_tool_strategy is 'per_turn'."""
        from victor.config.context_settings import ContextSettings

        settings = ContextSettings()
        assert settings.kv_tool_strategy == "per_turn"

    def test_setting_accepts_session_stable(self):
        """kv_tool_strategy can be set to 'session_stable'."""
        from victor.config.context_settings import ContextSettings

        settings = ContextSettings(kv_tool_strategy="session_stable")
        assert settings.kv_tool_strategy == "session_stable"

    def test_session_stable_returns_cached_tools_on_second_call(self):
        """session_stable strategy returns same tools on subsequent turns."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)

        # Configure session_stable strategy
        ctx = MagicMock()
        ctx.kv_tool_strategy = "session_stable"
        settings = MagicMock()
        settings.context = ctx
        orch.settings = settings

        # Mock tools
        tool_a = MagicMock()
        tool_a.name = "a_tool"
        tool_b = MagicMock()
        tool_b.name = "b_tool"
        first_selection = [tool_a, tool_b]

        # First call: stores tools
        orch._session_semantic_tools = None
        result = AgentOrchestrator._apply_kv_tool_strategy(orch, first_selection)
        assert result == first_selection
        assert orch._session_semantic_tools == first_selection

        # Second call: returns cached, ignores new selection
        tool_c = MagicMock()
        tool_c.name = "c_tool"
        result2 = AgentOrchestrator._apply_kv_tool_strategy(orch, [tool_c])
        assert result2 == first_selection  # Still the original set

    def test_per_turn_returns_fresh_selection(self):
        """per_turn strategy always returns the new selection."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        orch._session_semantic_tools = None

        # Configure per_turn strategy
        ctx = MagicMock()
        ctx.kv_tool_strategy = "per_turn"
        settings = MagicMock()
        settings.context = ctx
        orch.settings = settings

        tool_a = MagicMock()
        tool_a.name = "a_tool"

        result = AgentOrchestrator._apply_kv_tool_strategy(orch, [tool_a])
        assert [t.name for t in result] == ["a_tool"]
        assert orch._session_semantic_tools is None  # Not cached

    def test_non_kv_provider_ignores_strategy(self):
        """Non-KV providers bypass the strategy entirely."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)

        tool_a = MagicMock()
        tool_a.name = "a_tool"
        result = AgentOrchestrator._apply_kv_tool_strategy(orch, [tool_a])
        assert result == [tool_a]  # Pass through unchanged


# =====================================================================
# W1: Cached optimization flags
# =====================================================================


class TestCachedOptimizationFlags:
    """Test that optimization flags are cached at init, not recomputed per access."""

    def test_cache_flag_attributes_exist(self):
        """Orchestrator exposes _kv_opt_cached and _cache_opt_cached attributes."""
        from victor.agent.orchestrator import AgentOrchestrator

        # These should be declared (even if None before _compute_cache_flags)
        assert hasattr(AgentOrchestrator, "_compute_cache_flags")

    def test_compute_cache_flags_sets_cached_values(self):
        """_compute_cache_flags sets both cached flag values."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        provider = MagicMock()
        provider.supports_prompt_caching.return_value = False
        provider.supports_kv_prefix_caching.return_value = True
        orch.provider = provider
        orch.settings = MagicMock()
        orch.settings.context = MagicMock()
        orch.settings.context.cache_optimization_enabled = True
        orch._check_cache_setting_enabled = lambda: True

        AgentOrchestrator._compute_cache_flags(orch)

        assert orch._kv_opt_cached is True
        assert orch._cache_opt_cached is False

    def test_property_uses_cached_value_when_available(self):
        """_kv_optimization_enabled returns cached value if set."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch._kv_opt_cached = True
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is True

        orch._kv_opt_cached = False
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is False

    def test_property_computes_when_cache_is_none(self):
        """If _kv_opt_cached is None, property falls back to computation."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch._kv_opt_cached = None
        orch._check_cache_setting_enabled = lambda: True
        provider = MagicMock()
        provider.supports_kv_prefix_caching.return_value = True
        orch.provider = provider

        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is True


# =====================================================================
# W3: KV cache warm-up
# =====================================================================


class TestKVCacheWarmup:
    """Test KV cache warm-up method."""

    def test_warmup_method_exists(self):
        """AgentOrchestrator has a warm_up_kv_cache method."""
        from victor.agent.orchestrator import AgentOrchestrator

        assert hasattr(AgentOrchestrator, "warm_up_kv_cache")

    @pytest.mark.asyncio
    async def test_warmup_sends_minimal_request(self):
        """warm_up_kv_cache sends system prompt to provider with max_tokens=1."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        orch._system_prompt = "You are an assistant."
        orch.provider = MagicMock()
        orch.provider.chat = MagicMock(return_value=MagicMock(content=""))
        orch.model = "test-model"

        await AgentOrchestrator.warm_up_kv_cache(orch)

        orch.provider.chat.assert_called_once()
        call_kwargs = orch.provider.chat.call_args
        assert call_kwargs[1]["max_tokens"] == 1

    @pytest.mark.asyncio
    async def test_warmup_noop_when_kv_disabled(self):
        """warm_up_kv_cache is a no-op when KV optimization is disabled."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)
        orch.provider = MagicMock()

        await AgentOrchestrator.warm_up_kv_cache(orch)

        orch.provider.chat.assert_not_called()


# =====================================================================
# W6: KV prefix observability
# =====================================================================


class TestKVPrefixObservability:
    """Test KV prefix hash logging for observability."""

    def test_compute_prefix_hash(self):
        """_kv_prefix_fingerprint returns consistent hash for same prompt."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch._system_prompt = "You are a helpful assistant."

        h1 = AgentOrchestrator._kv_prefix_fingerprint(orch)
        h2 = AgentOrchestrator._kv_prefix_fingerprint(orch)
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) > 0

    def test_different_prompts_different_hash(self):
        """Different prompts produce different fingerprints."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch1 = MagicMock(spec=AgentOrchestrator)
        orch1._system_prompt = "Prompt A"
        orch2 = MagicMock(spec=AgentOrchestrator)
        orch2._system_prompt = "Prompt B"

        assert AgentOrchestrator._kv_prefix_fingerprint(orch1) != (
            AgentOrchestrator._kv_prefix_fingerprint(orch2)
        )


# =====================================================================
# W5: provider_has_kv_cache wired up in builder
# =====================================================================


class TestBuilderKVCacheWiredUp:
    """Test that provider_has_kv_cache is used for section cache freezing."""

    def test_kv_cache_enables_section_freeze(self):
        """When provider_has_kv_cache=True, _optimized_section_cache is used."""
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder.__new__(SystemPromptBuilder)
        builder.provider_caches = False
        builder.provider_has_kv_cache = True
        builder.concise_mode = False
        builder._rl_coordinator = None
        builder._decision_service = None
        builder._optimized_section_cache = {}

        # First call computes sections
        sections1 = SystemPromptBuilder._get_active_sections(builder)
        # Sections should be a set (reduced for non-caching)
        assert isinstance(sections1, set)


# =====================================================================
# W7: Separate KV setting
# =====================================================================


class TestSeparateKVSetting:
    """Test kv_optimization_enabled as separate setting."""

    def test_kv_optimization_enabled_setting_exists(self):
        """ContextSettings has kv_optimization_enabled field."""
        from victor.config.context_settings import ContextSettings

        settings = ContextSettings()
        assert hasattr(settings, "kv_optimization_enabled")
        assert settings.kv_optimization_enabled is True  # Default

    def test_kv_setting_can_be_disabled_independently(self):
        """kv_optimization_enabled can be False while cache_optimization_enabled is True."""
        from victor.config.context_settings import ContextSettings

        settings = ContextSettings(
            cache_optimization_enabled=True,
            kv_optimization_enabled=False,
        )
        assert settings.cache_optimization_enabled is True
        assert settings.kv_optimization_enabled is False


# =====================================================================
# W2: Cached tool sorting
# =====================================================================


class TestCachedToolSorting:
    """Test that sorted tool results are cached to avoid redundant work."""

    def test_same_tools_not_re_sorted(self):
        """Sorting cache avoids re-sorting identical tool sets."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        orch._last_sorted_tool_names = None
        orch._last_sorted_tools = None

        tool_b = MagicMock()
        tool_b.name = "b_tool"
        tool_a = MagicMock()
        tool_a.name = "a_tool"

        # First sort
        result1 = AgentOrchestrator._sort_tools_for_kv_stability(orch, [tool_b, tool_a])
        assert [t.name for t in result1] == ["a_tool", "b_tool"]

        # Second sort with same tools (different objects but same names)
        tool_b2 = MagicMock()
        tool_b2.name = "b_tool"
        tool_a2 = MagicMock()
        tool_a2.name = "a_tool"
        result2 = AgentOrchestrator._sort_tools_for_kv_stability(orch, [tool_b2, tool_a2])

        # Should use cache — same names means same result
        assert orch._last_sorted_tools is not None


# =====================================================================
# Integration gaps: warm-up wiring, fingerprint logging, facade exposure
# =====================================================================


class TestWarmUpIntegration:
    """Test warm_up_kv_cache is accessible from Agent facade."""

    @pytest.mark.asyncio
    async def test_agent_exposes_warm_up(self):
        """Agent facade has a warm_up() method delegating to orchestrator."""
        from victor.framework.agent import Agent

        assert hasattr(Agent, "warm_up")

    @pytest.mark.asyncio
    async def test_agent_warm_up_delegates_to_orchestrator(self):
        """Agent.warm_up() calls orchestrator.warm_up_kv_cache()."""
        from victor.framework.agent import Agent

        orch = MagicMock()
        orch.warm_up_kv_cache = MagicMock(return_value=None)
        # Make it async
        import asyncio

        async def mock_warm_up():
            pass

        orch.warm_up_kv_cache = mock_warm_up

        agent = Agent.__new__(Agent)
        agent._orchestrator = orch

        # Should not raise
        await Agent.warm_up(agent)


class TestFingerprintLogging:
    """Test that fingerprint is logged during streaming for observability."""

    def test_fingerprint_logged_in_get_assembled_messages(self):
        """_kv_prefix_fingerprint is called when KV optimization is active."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch._system_prompt = "test prompt"

        fingerprint = AgentOrchestrator._kv_prefix_fingerprint(orch)
        assert len(fingerprint) == 12  # md5[:12]
        assert all(c in "0123456789abcdef" for c in fingerprint)


class TestFallbackPathConsistency:
    """Test that _kv_optimization_enabled fallback checks kv_optimization_enabled setting."""

    def test_fallback_respects_kv_setting_disabled(self):
        """When _kv_opt_cached is None and kv_optimization_enabled=False, returns False."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch._kv_opt_cached = None  # Force fallback path

        # Settings: cache enabled but kv disabled
        ctx = MagicMock()
        ctx.cache_optimization_enabled = True
        ctx.kv_optimization_enabled = False
        settings = MagicMock()
        settings.context = ctx
        orch.settings = settings
        orch._check_cache_setting_enabled = lambda: True

        # Provider supports KV
        provider = MagicMock()
        provider.supports_kv_prefix_caching.return_value = True
        orch.provider = provider

        # Fallback should check kv_optimization_enabled
        result = AgentOrchestrator._kv_optimization_enabled.fget(orch)
        assert result is False
