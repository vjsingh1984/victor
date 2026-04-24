import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from victor.tools.graph_tool import graph, GraphMode


@pytest.mark.asyncio
async def test_graph_query_success():
    # Setup mocks
    mock_db = MagicMock()
    mock_db.query.return_value = [{"type": "class", "count": 10}, {"type": "function", "count": 50}]

    mock_loaded = MagicMock()
    mock_loaded.root_path = Path(".")
    mock_loaded.rebuilt = False

    with (
        patch("victor.core.database.get_project_database", return_value=mock_db),
        patch("victor.tools.graph_tool._load_graph", return_value=mock_loaded),
    ):

        sql = "SELECT type, count(*) as count FROM graph_node GROUP BY type"
        result = await graph(mode=GraphMode.QUERY, query=sql)

        assert result["success"] is True
        assert "results" in result["result"]
        assert len(result["result"]["results"]) == 2
        assert result["result"]["results"][0]["type"] == "class"
        assert result["result"]["results"][0]["count"] == 10


@pytest.mark.asyncio
async def test_graph_query_security_block():
    # Setup mocks
    mock_loaded = MagicMock()
    mock_loaded.root_path = Path(".")

    with patch("victor.tools.graph_tool._load_graph", return_value=mock_loaded):
        # Forbidden UPDATE
        sql = "UPDATE graph_node SET name = 'hacker'"
        result = await graph(mode=GraphMode.QUERY, query=sql)

        assert result["success"] is True  # graph() itself succeeded
        assert result["result"]["success"] is False  # but the inner query failed security
        assert "Only SELECT queries are allowed" in result["result"]["error"]


@pytest.mark.asyncio
async def test_graph_query_security_forbidden_keyword():
    # Setup mocks
    mock_loaded = MagicMock()
    mock_loaded.root_path = Path(".")

    with patch("victor.tools.graph_tool._load_graph", return_value=mock_loaded):
        # Forbidden DROP inside a SELECT (unlikely but check anyway)
        sql = "SELECT * FROM graph_node; DROP TABLE graph_node"
        result = await graph(mode=GraphMode.QUERY, query=sql)

        assert result["success"] is True
        assert result["result"]["success"] is False
        assert "Forbidden keyword detected" in result["result"]["error"]
