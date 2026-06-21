from victor.providers.base import ToolDefinition


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(name=name, description=f"{name} description", parameters={})


def test_postprocessor_applies_edge_filter_then_caps_tools():
    from victor.agent.tool_selection_postprocessor import (
        ToolSelectionPostProcessContext,
        ToolSelectionPostProcessor,
    )

    calls: list[str] = []
    processor = ToolSelectionPostProcessor()
    tools = [_tool(f"tool_{index}") for index in range(9)]
    context = ToolSelectionPostProcessContext(
        user_message="fix the failing tests",
        stage=None,
        fallback_max_tools=4,
        max_mcp_tools=7,
        schema_promotion_threshold=0.0,
        max_schema_tokens=0,
    )

    result = processor.apply(
        tools,
        context=context,
        should_use_edge_filter=True,
        apply_edge_filter=lambda current, user_message, stage: (
            calls.append("edge") or current[:6]
        ),
        cap_mcp_tools=lambda current, max_mcp: calls.append("mcp") or current[:5],
    )

    assert calls == ["edge", "mcp"]
    assert [tool.name for tool in result] == ["tool_0", "tool_1", "tool_2", "tool_3"]


def test_postprocessor_applies_promotion_before_budget():
    from victor.agent.tool_selection_postprocessor import (
        ToolSelectionPostProcessContext,
        ToolSelectionPostProcessor,
    )

    calls: list[str] = []
    processor = ToolSelectionPostProcessor()
    tools = [_tool("read"), _tool("search"), _tool("grep")]
    context = ToolSelectionPostProcessContext(
        user_message="search the codebase",
        stage=None,
        fallback_max_tools=8,
        max_mcp_tools=12,
        schema_promotion_threshold=0.8,
        max_schema_tokens=100,
    )

    result = processor.apply(
        tools,
        context=context,
        should_use_edge_filter=False,
        cap_mcp_tools=lambda current, _max_mcp: current,
        selection_scores={"search": 0.92},
        promote_schema_stubs=lambda current, scores, threshold: (
            calls.append("promote") or current
        ),
        enforce_token_budget=lambda current, max_tokens: calls.append("budget") or current[:2],
    )

    assert calls == ["promote", "budget"]
    assert [tool.name for tool in result] == ["read", "search"]


def test_truncation_preserves_selected_web_tool():
    """A selected web tool ranked below the cap must survive the order-blind truncation."""
    from victor.agent.tool_selection_postprocessor import (
        ToolSelectionPostProcessContext,
        ToolSelectionPostProcessor,
    )

    processor = ToolSelectionPostProcessor()
    # web_search sits at index 8 (just past fallback_max_tools=8) — the exact failing shape.
    tools = [_tool(f"tool_{i}") for i in range(8)] + [_tool("web_search"), _tool("tool_9")]
    context = ToolSelectionPostProcessContext(
        user_message="research the SDLC 3.0 white paper and cite sources",
        stage=None,
        fallback_max_tools=8,
        max_mcp_tools=12,
        schema_promotion_threshold=0.0,
        max_schema_tokens=0,
        web_tools=frozenset({"web_search"}),
    )

    result = processor.apply(
        tools,
        context=context,
        should_use_edge_filter=False,
        cap_mcp_tools=lambda current, _max_mcp: current,
    )

    names = [t.name for t in result]
    assert "web_search" in names  # survived
    assert len(result) == 8  # budget respected: 7 top non-web + web_search
    assert names == [
        "tool_0",
        "tool_1",
        "tool_2",
        "tool_3",
        "tool_4",
        "tool_5",
        "tool_6",
        "web_search",
    ]


def test_truncation_without_web_protection_unchanged():
    """Without web_tools, behavior is the plain order-blind cap (no regression)."""
    from victor.agent.tool_selection_postprocessor import (
        ToolSelectionPostProcessContext,
        ToolSelectionPostProcessor,
    )

    processor = ToolSelectionPostProcessor()
    tools = [_tool(f"tool_{i}") for i in range(8)] + [_tool("web_search")]
    context = ToolSelectionPostProcessContext(
        user_message="fix the bug",
        stage=None,
        fallback_max_tools=8,
        max_mcp_tools=12,
        schema_promotion_threshold=0.0,
        max_schema_tokens=0,
    )

    result = processor.apply(
        tools,
        context=context,
        should_use_edge_filter=False,
        cap_mcp_tools=lambda current, _max_mcp: current,
    )

    assert "web_search" not in [t.name for t in result]  # plain cap drops index 8
    assert len(result) == 8


def test_edge_filter_reattaches_dropped_web_tool():
    """A web tool dropped by the relevance edge filter is re-attached and survives."""
    from victor.agent.tool_selection_postprocessor import (
        ToolSelectionPostProcessContext,
        ToolSelectionPostProcessor,
    )

    processor = ToolSelectionPostProcessor()
    tools = [_tool(f"tool_{i}") for i in range(9)] + [_tool("web_search")]
    context = ToolSelectionPostProcessContext(
        user_message="research external sources",
        stage=None,
        fallback_max_tools=8,
        max_mcp_tools=12,
        schema_promotion_threshold=0.0,
        max_schema_tokens=0,
        web_tools=frozenset({"web_search"}),
    )

    result = processor.apply(
        tools,
        context=context,
        should_use_edge_filter=True,
        # Edge filter keeps only the first 5 (drops web_search by relevance).
        apply_edge_filter=lambda current, _msg, _stage: current[:5],
        cap_mcp_tools=lambda current, _max_mcp: current,
    )

    assert "web_search" in [t.name for t in result]  # re-attached after the edge filter


def test_postprocessor_skips_optional_transforms_when_disabled():
    from victor.agent.tool_selection_postprocessor import (
        ToolSelectionPostProcessContext,
        ToolSelectionPostProcessor,
    )

    processor = ToolSelectionPostProcessor()
    tools = [_tool("read"), _tool("search")]
    context = ToolSelectionPostProcessContext(
        user_message="read the file",
        stage=None,
        fallback_max_tools=8,
        max_mcp_tools=12,
        schema_promotion_threshold=0.0,
        max_schema_tokens=0,
    )

    result = processor.apply(
        tools,
        context=context,
        should_use_edge_filter=False,
        cap_mcp_tools=lambda current, _max_mcp: current,
    )

    assert result == tools
