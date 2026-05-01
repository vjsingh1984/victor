from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.providers.base import ToolDefinition


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(name=name, description=f"{name} description", parameters={})


def _make_selector():
    from victor.agent.tool_selection import ToolSelector

    registry = MagicMock()
    registry.list_tools.return_value = []
    return ToolSelector(tools=registry)


def test_load_semantic_cached_selection_returns_cached_tools_and_context():
    selector = _make_selector()
    selector._tool_selection_cache_enabled = True
    cache = object()
    selector._init_tool_selection_cache = MagicMock(return_value=cache)
    selector._semantic_cache_key_builder = MagicMock()
    selector._semantic_cache_key_builder.build.return_value = "cache-key"
    cached_tools = [_tool("read")]
    selector._semantic_cache_adapter = MagicMock()
    selector._semantic_cache_adapter.load_cached_tools.return_value = cached_tools

    tools, cache_key, resolved_cache = selector._load_semantic_cached_selection(
        user_message="read file",
        conversation_history=None,
        conversation_depth=0,
        stage=SimpleNamespace(value="analysis"),
    )

    assert tools == cached_tools
    assert cache_key == "cache-key"
    assert resolved_cache is cache


def test_finalize_semantic_selection_records_fallback_and_persists_cache():
    selector = _make_selector()
    selector._get_fallback_tools = MagicMock(return_value=[_tool("read")])
    selector._post_processor = MagicMock()
    selector._post_processor.apply.return_value = [_tool("read")]
    selector._selection_recorder = MagicMock()
    selector._apply_vertical_strategy = MagicMock(side_effect=lambda tools, _msg: tools)
    selector._filter_by_enabled = MagicMock(side_effect=lambda tools: tools)
    selector._semantic_cache_adapter = MagicMock()
    selector._semantic_cache_adapter.store_cached_tools.return_value = True
    selector._tool_selection_cache_enabled = True
    selector._tool_selection_cache_ttl = 300

    result = selector._finalize_semantic_selection(
        user_message="read file",
        tools=[],
        stage=None,
        cache=object(),
        cache_key="cache-key",
    )

    assert [tool.name for tool in result] == ["read"]
    selector._selection_recorder.record_result.assert_called_once_with(
        is_fallback=True,
        num_tools=1,
    )
    selector._semantic_cache_adapter.store_cached_tools.assert_called_once()
