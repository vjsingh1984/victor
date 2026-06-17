"""Regression tests for graph tool output bounding.

A single ``graph()`` call could previously materialize the entire node/edge set
(e.g. ``SELECT * FROM graph_edge``, a deep neighbor traversal, or "map all
components recursively"), producing multi-megabyte payloads that blew the context
window and triggered aggressive compaction (an observed result drove
``estimated_output_tokens=646224``). ``_bound_graph_result`` caps every list-valued
field and enforces a total serialized-size ceiling. These tests lock that in.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from victor.tools.graph_tool import (
    GRAPH_MAX_LIST_ITEMS,
    GRAPH_MAX_QUERY_ROWS,
    GRAPH_MAX_RESULT_CHARS,
    GraphMode,
    _bound_graph_result,
    _ensure_sql_row_limit,
    graph,
)


def test_caps_top_level_list_to_max_items():
    payload = {
        "mode": "query",
        "result": {"results": [{"id": i} for i in range(5000)]},
    }
    bounded = _bound_graph_result(payload)

    assert len(bounded["result"]["results"]) == GRAPH_MAX_LIST_ITEMS
    assert bounded["truncated"] is True
    assert "truncation_note" in bounded


def test_caps_nested_dict_of_lists_per_entry():
    # neighbors mode returns {"neighbors_by_depth": {depth: [nodes...]}}
    payload = {
        "mode": "neighbors",
        "result": {
            "neighbors_by_depth": {
                "1": [{"n": i} for i in range(900)],
                "2": [{"n": i} for i in range(900)],
            }
        },
    }
    bounded = _bound_graph_result(payload)

    depths = bounded["result"]["neighbors_by_depth"]
    assert len(depths["1"]) == GRAPH_MAX_LIST_ITEMS
    assert len(depths["2"]) == GRAPH_MAX_LIST_ITEMS
    assert bounded["truncated"] is True


def test_char_backstop_shrinks_below_item_cap():
    # Even GRAPH_MAX_LIST_ITEMS rows blow the char budget when each item is huge,
    # so the per-list cap must shrink further to fit GRAPH_MAX_RESULT_CHARS.
    payload = {
        "mode": "query",
        "result": {"results": [{"blob": "y" * 2000} for _ in range(GRAPH_MAX_LIST_ITEMS)]},
    }
    bounded = _bound_graph_result(payload)

    assert len(bounded["result"]["results"]) < GRAPH_MAX_LIST_ITEMS
    assert len(json.dumps(bounded)) <= GRAPH_MAX_RESULT_CHARS
    assert bounded["truncated"] is True


def test_small_result_is_untouched():
    payload = {"mode": "stats", "result": {"nodes": 10, "edges": 20}}
    bounded = _bound_graph_result(payload)

    assert bounded == payload
    assert "truncated" not in bounded


def test_non_dict_payload_passthrough():
    assert _bound_graph_result("just a string") == "just a string"
    assert _bound_graph_result(42) == 42


def test_sql_row_limit_appended_when_absent():
    sql, applied = _ensure_sql_row_limit("SELECT * FROM graph_edge", 1000)
    assert applied is True
    assert sql == "SELECT * FROM graph_edge LIMIT 1000"


def test_sql_row_limit_appended_before_trailing_semicolon():
    sql, applied = _ensure_sql_row_limit("SELECT * FROM graph_node ;", 1000)
    assert applied is True
    assert sql == "SELECT * FROM graph_node LIMIT 1000"


def test_sql_row_limit_honors_user_limit():
    original = "SELECT * FROM graph_node LIMIT 50"
    sql, applied = _ensure_sql_row_limit(original, 1000)
    assert applied is False
    assert sql == original


def test_sql_row_limit_case_insensitive_detection():
    original = "select name from graph_node limit 5"
    sql, applied = _ensure_sql_row_limit(original, 1000)
    assert applied is False
    assert sql == original


@pytest.mark.asyncio
async def test_graph_query_injects_limit_at_db_source():
    """An unbounded SELECT gets a defensive LIMIT before hitting the database."""
    mock_db = MagicMock()
    mock_db.query.return_value = [{"node_id": 1}]

    mock_loaded = MagicMock()
    mock_loaded.root_path = Path(".")
    mock_loaded.rebuilt = False

    with (
        patch("victor.core.database.get_project_database", return_value=mock_db),
        patch("victor.tools.graph_tool._load_graph", return_value=mock_loaded),
    ):
        result = await graph(mode=GraphMode.QUERY, query="SELECT * FROM graph_edge")

    executed_sql = mock_db.query.call_args[0][0]
    assert f"LIMIT {GRAPH_MAX_QUERY_ROWS}" in executed_sql
    assert result["result"]["row_limit_applied"] == GRAPH_MAX_QUERY_ROWS


@pytest.mark.asyncio
async def test_graph_query_tool_bounds_large_result_end_to_end():
    """The public graph() wrapper bounds an oversized query result."""
    mock_db = MagicMock()
    mock_db.query.return_value = [{"node_id": i, "name": f"sym_{i}"} for i in range(4000)]

    mock_loaded = MagicMock()
    mock_loaded.root_path = Path(".")
    mock_loaded.rebuilt = False

    with (
        patch("victor.core.database.get_project_database", return_value=mock_db),
        patch("victor.tools.graph_tool._load_graph", return_value=mock_loaded),
    ):
        result = await graph(mode=GraphMode.QUERY, query="SELECT * FROM graph_node")

    assert result["success"] is True
    assert len(result["result"]["results"]) <= GRAPH_MAX_LIST_ITEMS
    assert result["truncated"] is True
