from victor.providers.base import ToolDefinition


def _tool(name: str, description: str = "") -> ToolDefinition:
    return ToolDefinition(name=name, description=description or f"{name} description", parameters={})


def test_cache_adapter_restores_tools_from_cached_payload():
    from victor.agent.tool_selection_cache import SemanticToolSelectionCacheAdapter

    adapter = SemanticToolSelectionCacheAdapter()
    payload = {
        "tool_names": ["read", "search"],
        "tool_descriptions": {"read": "Read files", "search": "Search files"},
        "tool_parameters": {"read": {"type": "object"}},
    }

    restored = adapter.restore_tools(payload)

    assert [tool.name for tool in restored] == ["read", "search"]
    assert restored[0].description == "Read files"
    assert restored[0].parameters == {"type": "object"}
    assert restored[1].parameters == {}


def test_cache_adapter_returns_none_for_incomplete_payload():
    from victor.agent.tool_selection_cache import SemanticToolSelectionCacheAdapter

    adapter = SemanticToolSelectionCacheAdapter()

    restored = adapter.restore_tools(
        {
            "tool_names": ["read"],
            "tool_descriptions": {},
            "tool_parameters": {},
        }
    )

    assert restored is None


def test_cache_adapter_serializes_tools_for_storage():
    from victor.agent.tool_selection_cache import SemanticToolSelectionCacheAdapter

    adapter = SemanticToolSelectionCacheAdapter()
    payload = adapter.serialize_tools([_tool("read"), _tool("search", "Search repo")])

    assert payload == {
        "tool_names": ["read", "search"],
        "tool_descriptions": {
            "read": "read description",
            "search": "Search repo",
        },
        "tool_parameters": {
            "read": {},
            "search": {},
        },
    }
