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


def test_cache_adapter_loads_restored_tools_from_cache():
    from victor.agent.tool_selection_cache import SemanticToolSelectionCacheAdapter

    class _FakeCache:
        def get(self, result_type, cache_key):
            assert cache_key == "abc123"
            assert result_type.value == "tool_selection"
            return {
                "tool_names": ["read"],
                "tool_descriptions": {"read": "Read files"},
                "tool_parameters": {"read": {}},
            }

    adapter = SemanticToolSelectionCacheAdapter()
    restored = adapter.load_cached_tools(_FakeCache(), "abc123")

    assert [tool.name for tool in restored] == ["read"]


def test_cache_adapter_stores_serialized_tools_with_ttl():
    from victor.agent.tool_selection_cache import SemanticToolSelectionCacheAdapter

    seen: dict[str, object] = {}

    class _FakeCache:
        def set(self, result_type, cache_key, payload, ttl):
            seen["result_type"] = result_type.value
            seen["cache_key"] = cache_key
            seen["payload"] = payload
            seen["ttl"] = ttl

    adapter = SemanticToolSelectionCacheAdapter()
    stored = adapter.store_cached_tools(_FakeCache(), "key-1", [_tool("read")], ttl=60)

    assert stored is True
    assert seen == {
        "result_type": "tool_selection",
        "cache_key": "key-1",
        "payload": {
            "tool_names": ["read"],
            "tool_descriptions": {"read": "read description"},
            "tool_parameters": {"read": {}},
        },
        "ttl": 60,
    }
