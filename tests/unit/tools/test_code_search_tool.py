# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for code_search_tool."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from victor.tools.code_search_tool import code_search


def _settings(**overrides):
    defaults = {
        "semantic_similarity_threshold": 0.25,
        "semantic_query_expansion_enabled": True,
        "enable_hybrid_search": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_code_search_bug_mode_uses_provider_capability(tmp_path) -> None:
    """Bug mode should delegate to the provider-backed similar bug search."""
    mock_index = SimpleNamespace(
        find_similar_bugs=AsyncMock(
            return_value=[
                {
                    "file_path": "src/parser.py",
                    "content": "def parse_json(data): return json.loads(data)",
                    "score": 0.91,
                    "graph_context": {"callers": [{"name": "main"}]},
                }
            ]
        )
    )
    exec_ctx = {"settings": _settings()}

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, False)),
    ):
        result = await code_search(
            query="json parsing crash on empty payload",
            path=str(tmp_path),
            k=4,
            mode="bugs",
            lang="python",
            _exec_ctx=exec_ctx,
        )

    mock_index.find_similar_bugs.assert_awaited_once_with(
        bug_description="json parsing crash on empty payload",
        language="python",
        top_k=4,
        include_graph_context=True,
        context_limit=3,
    )
    assert result["success"] is True
    assert result["mode"] == "bugs"
    assert result["count"] == 1
    assert result["results"][0]["search_mode"] == "bug_similarity"
    assert result["results"][0]["graph_context"]["callers"][0]["name"] == "main"
    assert result["metadata"]["provider_capability"] == "find_similar_bugs"


@pytest.mark.asyncio
async def test_code_search_bug_mode_falls_back_to_semantic_when_unsupported(tmp_path) -> None:
    """Bug mode should degrade to semantic search when provider support is absent."""
    mock_index = SimpleNamespace(
        find_similar_bugs=AsyncMock(
            side_effect=NotImplementedError("find_similar_bugs unsupported")
        ),
        semantic_search=AsyncMock(
            return_value=[
                {
                    "file_path": "src/parser.py",
                    "content": "def parse_json(data): return json.loads(data)",
                    "score": 0.74,
                }
            ]
        ),
    )
    exec_ctx = {"settings": _settings()}

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, True)),
    ):
        result = await code_search(
            query="json parsing crash on empty payload",
            path=str(tmp_path),
            k=2,
            mode="bugs",
            lang="python",
            _exec_ctx=exec_ctx,
        )

    mock_index.semantic_search.assert_awaited_once_with(
        query="json parsing crash on empty payload",
        max_results=2,
        filter_metadata={"language": "python"},
        similarity_threshold=0.25,
        expand_query=True,
    )
    assert result["success"] is True
    assert result["mode"] == "semantic"
    assert result["metadata"]["requested_mode"] == "bugs"
    assert result["metadata"]["fallback_mode"] == "semantic"
    assert "mode_fallback=semantic" in result["metadata"]["filters_applied"]
