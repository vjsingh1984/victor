"""Unit tests for ToolService KV-prefix strategy methods (Item 3)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.services.tool_service import ToolService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, *, schema_level: str | None = "full", priority=None) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.schema_level = schema_level
    tool.to_schema = MagicMock(return_value={"name": name, "description": "x" * 100})
    if priority is not None:
        tool.priority = priority
    else:
        del tool.priority  # ensure hasattr returns False
    return tool


def _make_service() -> ToolService:
    """Construct a minimal ToolService with required dependencies mocked."""
    settings = MagicMock()
    settings.tools.budget = 50
    registry = MagicMock()
    registry.get_all_tools.return_value = []
    service = ToolService.__new__(ToolService)
    service._settings = settings
    service._registry = registry
    service._logger = MagicMock()
    service._usage_stats = {}
    return service


# ---------------------------------------------------------------------------
# sort_tools_for_kv_stability
# ---------------------------------------------------------------------------


class TestSortToolsForKvStability:
    def test_sorts_full_before_compact_before_stub(self):
        svc = _make_service()
        tools = [
            _make_tool("z_stub", schema_level="stub"),
            _make_tool("a_full", schema_level="full"),
            _make_tool("b_compact", schema_level="compact"),
        ]
        result = svc.sort_tools_for_kv_stability(tools, kv_optimization_enabled=True)
        assert [t.name for t in result] == ["a_full", "b_compact", "z_stub"]

    def test_alphabetical_within_same_level(self):
        svc = _make_service()
        tools = [
            _make_tool("c_full", schema_level="full"),
            _make_tool("a_full", schema_level="full"),
        ]
        result = svc.sort_tools_for_kv_stability(tools, kv_optimization_enabled=True)
        assert result[0].name == "a_full"

    def test_returns_none_when_tools_is_none(self):
        svc = _make_service()
        assert svc.sort_tools_for_kv_stability(None) is None

    def test_passthrough_when_kv_disabled(self):
        svc = _make_service()
        tools = [_make_tool("z"), _make_tool("a")]
        result = svc.sort_tools_for_kv_stability(tools, kv_optimization_enabled=False)
        assert result is tools  # same object, no copy

    def test_unknown_schema_level_treated_as_stub(self):
        svc = _make_service()
        tools = [
            _make_tool("b_full", schema_level="full"),
            _make_tool("a_unknown", schema_level=None),
        ]
        result = svc.sort_tools_for_kv_stability(tools)
        assert result[0].name == "b_full"
        assert result[1].name == "a_unknown"


# ---------------------------------------------------------------------------
# estimate_tool_tokens
# ---------------------------------------------------------------------------


class TestEstimateToolTokens:
    def test_returns_positive_integer(self):
        svc = _make_service()
        tool = _make_tool("my_tool")
        with (patch("victor.config.tool_tiers.get_tool_tier", return_value="full"),):
            result = svc.estimate_tool_tokens(tool)
        assert isinstance(result, int)
        assert result > 0

    def test_fallback_on_schema_error(self):
        svc = _make_service()
        tool = MagicMock()
        tool.name = "bad_tool"
        tool.to_schema = MagicMock(side_effect=RuntimeError("boom"))
        with patch("victor.config.tool_tiers.get_tool_tier", return_value="full"):
            result = svc.estimate_tool_tokens(tool)
        # fallback: len(name) + 50
        assert result == len("bad_tool") + 50

    def test_uses_provider_category_when_given(self):
        svc = _make_service()
        tool = _make_tool("my_tool")
        with (
            patch(
                "victor.config.tool_tiers.get_provider_tool_tier",
                return_value="compact",
            ) as mock_pt,
            patch("victor.config.tool_tiers.get_tool_tier") as mock_gt,
        ):
            svc.estimate_tool_tokens(tool, provider_category="small")
        mock_pt.assert_called_once_with("my_tool", "small")
        mock_gt.assert_not_called()


# ---------------------------------------------------------------------------
# apply_kv_tool_strategy — session_stable
# ---------------------------------------------------------------------------


class TestApplyKvToolStrategySessionStable:
    def test_additive_appends_new_tools(self):
        # P4: session_stable is grow-only/additive — a newly-selected tool joins the
        # cached set instead of being locked out (the mid-session intent-shift fix).
        svc = _make_service()
        cached = [_make_tool("cached")]
        fresh = [_make_tool("fresh")]
        provider = MagicMock(supports_prompt_caching=MagicMock(return_value=False))
        provider.context_window = MagicMock(return_value=8192)
        result = svc.apply_kv_tool_strategy(
            fresh,
            kv_optimization_enabled=True,
            provider=provider,
            model="m",
            session_semantic_tools=cached,
            kv_tool_strategy="session_stable",
        )
        assert [t.name for t in result] == ["cached", "fresh"]

    def test_returns_cached_object_when_no_new_tools(self):
        # A turn that selects nothing new returns the cached set unchanged -> KV cache hit.
        svc = _make_service()
        cached = [_make_tool("a"), _make_tool("b")]
        fresh = [_make_tool("a")]  # subset of cached
        provider = MagicMock(supports_prompt_caching=MagicMock(return_value=False))
        provider.context_window = MagicMock(return_value=8192)
        result = svc.apply_kv_tool_strategy(
            fresh,
            kv_optimization_enabled=True,
            provider=provider,
            model="m",
            session_semantic_tools=cached,
            kv_tool_strategy="session_stable",
        )
        assert result is cached

    def test_additive_alias_behaves_like_session_stable(self):
        svc = _make_service()
        cached = [_make_tool("a")]
        provider = MagicMock(supports_prompt_caching=MagicMock(return_value=False))
        provider.context_window = MagicMock(return_value=8192)
        result = svc.apply_kv_tool_strategy(
            [_make_tool("b")],
            kv_optimization_enabled=True,
            provider=provider,
            model="m",
            session_semantic_tools=cached,
            kv_tool_strategy="additive",
        )
        assert [t.name for t in result] == ["a", "b"]

    def test_returns_tools_on_first_call_no_cache(self):
        svc = _make_service()
        tools = [_make_tool("t")]
        provider = MagicMock(supports_prompt_caching=MagicMock(return_value=False))
        provider.context_window = MagicMock(return_value=8192)
        result = svc.apply_kv_tool_strategy(
            tools,
            kv_optimization_enabled=True,
            provider=provider,
            model="m",
            session_semantic_tools=None,
            kv_tool_strategy="session_stable",
        )
        assert result is tools

    def test_passthrough_when_kv_disabled(self):
        svc = _make_service()
        tools = [_make_tool("t")]
        provider = MagicMock()
        result = svc.apply_kv_tool_strategy(
            tools,
            kv_optimization_enabled=False,
            provider=provider,
            model="m",
            kv_tool_strategy="session_stable",
        )
        assert result is tools


# ---------------------------------------------------------------------------
# apply_context_aware_strategy — compact providers
# ---------------------------------------------------------------------------


class TestApplyContextAwareStrategy:
    def _provider(self, *, supports_caching: bool = False, context_window: int = 8192):
        p = MagicMock()
        p.supports_prompt_caching = MagicMock(return_value=supports_caching)
        p.context_window = MagicMock(return_value=context_window)
        p.name = "test_provider"
        return p

    def test_session_lock_for_caching_provider(self):
        svc = _make_service()
        tools = [_make_tool(f"t{i}") for i in range(5)]
        provider = self._provider(supports_caching=True, context_window=100000)
        with (
            patch("victor.config.tool_tiers.get_provider_category", return_value="cloud"),
            patch.object(svc, "estimate_tool_tokens", return_value=50),
        ):
            result = svc.apply_context_aware_strategy(tools, provider=provider, model="gpt")
        assert result == tools  # session-lock: returns all tools unchanged

    def test_semantic_selection_for_small_context(self):
        svc = _make_service()
        tools = [_make_tool(f"t{i}") for i in range(10)]
        provider = self._provider(supports_caching=False, context_window=4096)
        with (
            patch("victor.config.tool_tiers.get_provider_category", return_value="small"),
            patch.object(svc, "estimate_tool_tokens", return_value=100),
            patch.object(svc, "semantic_select_tools", return_value=tools[:3]) as mock_ss,
        ):
            result = svc.apply_context_aware_strategy(tools, provider=provider, model="small-m")
        mock_ss.assert_called_once()
        assert result == tools[:3]


# ---------------------------------------------------------------------------
# semantic_select_tools
# ---------------------------------------------------------------------------


class TestSemanticSelectTools:
    def test_respects_token_budget(self):
        svc = _make_service()
        tools = [_make_tool(f"t{i}") for i in range(5)]
        with patch.object(svc, "estimate_tool_tokens", return_value=200):
            result = svc.semantic_select_tools(tools, max_tokens=500)
        # 200*2 = 400 ≤ 500, but 200*3 = 600 > 500 → first 2
        assert len(result) == 2

    def test_empty_tools_returns_empty(self):
        svc = _make_service()
        assert svc.semantic_select_tools([], max_tokens=1000) == []
