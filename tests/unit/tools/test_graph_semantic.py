import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from victor.tools.graph_tool import graph, GraphMode
from victor.storage.graph.protocol import GraphNode


@pytest.fixture
def mock_node():
    return GraphNode(
        node_id="node_1",
        name="ToolRegistry",
        type="class",
        file="victor/tools/registry.py",
        docstring="Registry for managing tools.",
        signature="class ToolRegistry:",
        lang="python",
    )


@pytest.fixture
def mock_similar_node():
    return GraphNode(
        node_id="node_2",
        name="SearchTool",
        type="class",
        file="victor/tools/search.py",
        docstring="A tool for searching.",
        signature="class SearchTool:",
        lang="python",
    )


@pytest.mark.asyncio
async def test_graph_semantic_discovery(mock_node, mock_similar_node):
    # Setup mocks
    mock_analyzer = MagicMock()
    mock_analyzer.nodes = {"node_1": mock_node, "node_2": mock_similar_node}

    def mock_resolve(ref, **kwargs):
        if ref == "node_1" or ref == "ToolRegistry":
            return "node_1"
        if ref == "node_2" or ref == "SearchTool":
            return "node_2"
        return None

    mock_analyzer.resolve_node_id.side_effect = mock_resolve
    mock_analyzer.get_neighbors.return_value = {"neighbors_by_depth": {}}

    mock_index = AsyncMock()
    mock_index.semantic_search.return_value = [
        {
            "file_path": "victor/tools/search.py",
            "symbol_name": "SearchTool",
            "score": 0.85,
            "symbol_type": "class",
        }
    ]

    mock_loaded = MagicMock()
    mock_loaded.analyzer = mock_analyzer
    mock_loaded.index = mock_index
    mock_loaded.root_path = Path(".")
    mock_loaded.rebuilt = False

    with patch("victor.tools.graph_tool._load_graph", return_value=mock_loaded):
        result = await graph(mode=GraphMode.SEMANTIC, node="node_1", threshold=0.5)

        assert result["success"] is True
        assert "potential_relationships" in result["result"]
        relationships = result["result"]["potential_relationships"]
        assert len(relationships) == 1
        assert relationships[0]["name"] == "SearchTool"
        assert relationships[0]["similarity"] == 0.85


@pytest.mark.asyncio
async def test_graph_semantic_filters_existing_neighbors(mock_node, mock_similar_node):
    # Setup mocks
    mock_analyzer = MagicMock()
    mock_analyzer.nodes = {"node_1": mock_node, "node_2": mock_similar_node}

    def mock_resolve(ref, **kwargs):
        if ref == "node_1" or ref == "ToolRegistry":
            return "node_1"
        if ref == "node_2" or ref == "SearchTool":
            return "node_2"
        return None

    mock_analyzer.resolve_node_id.side_effect = mock_resolve

    # Mock existing structural relationship
    mock_analyzer.get_neighbors.return_value = {
        "neighbors_by_depth": {1: [{"node_id": "node_2", "name": "SearchTool"}]}
    }

    mock_index = AsyncMock()
    mock_index.semantic_search.return_value = [
        {
            "file_path": "victor/tools/search.py",
            "symbol_name": "SearchTool",
            "score": 0.85,
            "symbol_type": "class",
        }
    ]

    mock_loaded = MagicMock()
    mock_loaded.analyzer = mock_analyzer
    mock_loaded.index = mock_index
    mock_loaded.root_path = Path(".")
    mock_loaded.rebuilt = False

    with patch("victor.tools.graph_tool._load_graph", return_value=mock_loaded):
        result = await graph(mode=GraphMode.SEMANTIC, node="node_1", threshold=0.5)

        assert result["success"] is True
        # node_2 is already a neighbor, so it should be filtered out
        assert len(result["result"]["potential_relationships"]) == 0
