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
async def test_code_search_localize_mode_uses_provider_capability(tmp_path) -> None:
    """Localize mode should delegate to provider-backed issue localization when available."""
    mock_index = SimpleNamespace(
        localize_issue=AsyncMock(
            return_value=[
                {
                    "file_path": "src/repository.py",
                    "content": "class BaseRepository:\n    def save(self): ...",
                    "score": 0.93,
                    "symbol_name": "BaseRepository.save",
                    "metadata": {
                        "localization": {
                            "seed_score": 0.76,
                            "graph_score": 0.17,
                            "matched_hints": ["BaseRepository"],
                        }
                    },
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
            query="which files should I edit to add a logger parameter to BaseRepository",
            path=str(tmp_path),
            k=5,
            mode="localize",
            lang="python",
            _exec_ctx=exec_ctx,
        )

    mock_index.localize_issue.assert_awaited_once_with(
        issue_description="which files should I edit to add a logger parameter to BaseRepository",
        language="python",
        top_k=5,
        include_graph_context=True,
        context_limit=3,
    )
    assert result["success"] is True
    assert result["mode"] == "localize"
    assert result["count"] == 1
    assert result["results"][0]["search_mode"] == "issue_localization"
    assert result["metadata"]["provider_capability"] == "localize_issue"
    assert result["results"][0]["metadata"]["localization"]["matched_hints"] == ["BaseRepository"]


@pytest.mark.asyncio
async def test_code_search_localize_mode_falls_back_to_semantic_when_unsupported(
    tmp_path,
) -> None:
    """Localize mode should degrade to semantic search when provider support is absent."""
    mock_index = SimpleNamespace(
        localize_issue=AsyncMock(side_effect=NotImplementedError("localize_issue unsupported")),
        semantic_search=AsyncMock(
            return_value=[
                {
                    "file_path": "src/repository.py",
                    "content": "class BaseRepository:\n    def save(self): ...",
                    "score": 0.69,
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
            query="which files should I edit to add a logger parameter to BaseRepository",
            path=str(tmp_path),
            k=3,
            mode="localize",
            lang="python",
            _exec_ctx=exec_ctx,
        )

    mock_index.semantic_search.assert_awaited_once_with(
        query="which files should I edit to add a logger parameter to BaseRepository",
        max_results=3,
        filter_metadata=None,
        similarity_threshold=0.25,
        expand_query=True,
    )
    assert result["success"] is True
    assert result["mode"] == "semantic"
    assert result["metadata"]["requested_mode"] == "localize"
    assert result["metadata"]["fallback_mode"] == "semantic"
    assert "mode_fallback=semantic" in result["metadata"]["filters_applied"]


@pytest.mark.asyncio
async def test_code_search_impact_mode_uses_provider_capability(tmp_path) -> None:
    """Impact mode should delegate to provider-backed blast-radius analysis when available."""
    mock_index = SimpleNamespace(
        analyze_change_impact=AsyncMock(
            return_value=[
                {
                    "file_path": "src/service.py",
                    "content": "def create_user(...): repo.save(user)",
                    "score": 0.95,
                    "symbol_name": "UserService.create_user",
                    "metadata": {
                        "impact": {
                            "seed_score": 0.72,
                            "graph_score": 0.23,
                            "matched_hints": ["BaseRepository.save"],
                        }
                    },
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
            query="what breaks if I change BaseRepository.save",
            path=str(tmp_path),
            k=4,
            mode="impact",
            lang="python",
            _exec_ctx=exec_ctx,
        )

    mock_index.analyze_change_impact.assert_awaited_once_with(
        change_description="what breaks if I change BaseRepository.save",
        language="python",
        top_k=4,
        include_graph_context=True,
        context_limit=3,
    )
    assert result["success"] is True
    assert result["mode"] == "impact"
    assert result["count"] == 1
    assert result["results"][0]["search_mode"] == "change_impact"
    assert result["metadata"]["provider_capability"] == "analyze_change_impact"


@pytest.mark.asyncio
async def test_code_search_impact_mode_falls_back_to_semantic_when_unsupported(
    tmp_path,
) -> None:
    """Impact mode should degrade to semantic search when provider support is absent."""
    mock_index = SimpleNamespace(
        analyze_change_impact=AsyncMock(
            side_effect=NotImplementedError("analyze_change_impact unsupported")
        ),
        semantic_search=AsyncMock(
            return_value=[
                {
                    "file_path": "src/repository.py",
                    "content": "class BaseRepository:\n    def save(self): ...",
                    "score": 0.71,
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
            query="what breaks if I change BaseRepository.save",
            path=str(tmp_path),
            k=3,
            mode="impact",
            lang="python",
            _exec_ctx=exec_ctx,
        )

    mock_index.semantic_search.assert_awaited_once_with(
        query="what breaks if I change BaseRepository.save",
        max_results=3,
        filter_metadata=None,
        similarity_threshold=0.25,
        expand_query=True,
    )
    assert result["success"] is True
    assert result["mode"] == "semantic"
    assert result["metadata"]["requested_mode"] == "impact"
    assert result["metadata"]["fallback_mode"] == "semantic"
    assert "mode_fallback=semantic" in result["metadata"]["filters_applied"]


@pytest.mark.asyncio
async def test_code_search_bug_mode_falls_back_to_semantic_when_unsupported(
    tmp_path,
) -> None:
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
        filter_metadata=None,  # "language" stripped (not in index schema)
        similarity_threshold=0.25,
        expand_query=True,
    )
    assert result["success"] is True
    assert result["mode"] == "semantic"
    assert result["metadata"]["requested_mode"] == "bugs"
    assert result["metadata"]["fallback_mode"] == "semantic"
    assert "mode_fallback=semantic" in result["metadata"]["filters_applied"]


@pytest.mark.asyncio
async def test_code_search_semantic_mode_adds_graph_follow_up_for_entrypoint(
    tmp_path,
) -> None:
    """Semantic results that identify an entrypoint should suggest graph follow-ups."""
    mock_index = SimpleNamespace(
        semantic_search=AsyncMock(
            return_value=[
                {
                    "file_path": "src/main.py",
                    "content": "def main():\n    parse_json(data)\n",
                    "score": 0.82,
                    "name": "main",
                    "symbol_type": "function",
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
            query="entry point for request processing",
            path=str(tmp_path),
            k=3,
            _exec_ctx=exec_ctx,
        )

    follow_ups = result["metadata"]["follow_up_suggestions"]
    assert result["success"] is True
    assert result["mode"] == "semantic"
    assert result["hint"].endswith('graph(mode="trace", node="main", depth=3)')
    assert follow_ups[0]["tool"] == "graph"
    assert follow_ups[0]["arguments"] == {"mode": "trace", "node": "main", "depth": 3}
    assert any(item["arguments"]["mode"] == "callers" for item in follow_ups)
    assert any(item["arguments"]["mode"] == "callees" for item in follow_ups)
