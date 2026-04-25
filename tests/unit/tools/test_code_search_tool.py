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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.tools.code_search_tool import (
    SearchFilters,
    _build_codebase_embedding_config,
    _get_index_build_failure_cache,
    _get_or_build_index,
    code_search,
)
from victor.tools.context import ToolExecutionContext


def _settings(**overrides):
    defaults = {
        "semantic_similarity_threshold": 0.25,
        "semantic_query_expansion_enabled": True,
        "enable_hybrid_search": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_codebase_embedding_config_forwards_existing_search_settings(tmp_path) -> None:
    """Embedding config should propagate dimension, batch size, and extra options."""
    settings = _settings(
        codebase_vector_store="lancedb",
        codebase_embedding_provider="sentence-transformers",
        codebase_embedding_model="BAAI/bge-small-en-v1.5",
        codebase_dimension=768,
        codebase_batch_size=12,
        codebase_embedding_extra_config={"table_name": "custom_embeddings"},
    )

    config = _build_codebase_embedding_config(settings, tmp_path)

    assert config["vector_store"] in {"lancedb", "victor_structural_bridge"}
    assert config["embedding_model_type"] == "sentence-transformers"
    assert config["embedding_model_name"] == "BAAI/bge-small-en-v1.5"
    assert config["extra_config"]["dimension"] == 768
    assert config["extra_config"]["batch_size"] == 12
    assert config["extra_config"]["structural_indexing_enabled"] is True
    assert config["extra_config"]["code_chunking_strategy"] == "tree_sitter_structural"
    assert config["extra_config"]["chunk_size"] == 500
    assert config["extra_config"]["chunk_overlap"] == 50
    assert config["extra_config"]["workspace_root"] == str(tmp_path)
    assert config["extra_config"]["table_name"] == "custom_embeddings"
    if config["vector_store"] == "victor_structural_bridge":
        assert config["extra_config"]["upstream_vector_store"] == "lancedb"


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
    filters = SearchFilters(language="python")

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, False)),
    ):
        result = await code_search(
            query="json parsing crash on empty payload",
            path=str(tmp_path),
            k=4,
            mode="bugs",
            filters=filters,
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
    filters = SearchFilters(language="python")

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, False)),
    ):
        result = await code_search(
            query="which files should I edit to add a logger parameter to BaseRepository",
            path=str(tmp_path),
            k=5,
            mode="localize",
            filters=filters,
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
    filters = SearchFilters(language="python")

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, True)),
    ):
        result = await code_search(
            query="which files should I edit to add a logger parameter to BaseRepository",
            path=str(tmp_path),
            k=3,
            mode="localize",
            filters=filters,
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
    assert "lang=python" in result["metadata"]["filters_applied"]
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
    filters = SearchFilters(language="python")

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, False)),
    ):
        result = await code_search(
            query="what breaks if I change BaseRepository.save",
            path=str(tmp_path),
            k=4,
            mode="impact",
            filters=filters,
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
    filters = SearchFilters(language="python")

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, True)),
    ):
        result = await code_search(
            query="what breaks if I change BaseRepository.save",
            path=str(tmp_path),
            k=3,
            mode="impact",
            filters=filters,
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
    assert "lang=python" in result["metadata"]["filters_applied"]
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
    filters = SearchFilters(language="python")

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, True)),
    ):
        result = await code_search(
            query="json parsing crash on empty payload",
            path=str(tmp_path),
            k=2,
            mode="bugs",
            filters=filters,
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
    assert "lang=python" in result["metadata"]["filters_applied"]
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
    assert any(item["arguments"]["mode"] == "callers|callees" for item in follow_ups)


@pytest.mark.asyncio
async def test_code_search_literal_mode_escalation_preserves_requested_mode_context(
    tmp_path,
) -> None:
    """Literal auto-escalation should keep the original requested mode in metadata."""
    mock_index = SimpleNamespace(
        semantic_search=AsyncMock(
            return_value=[
                {
                    "file_path": "src/main.py",
                    "content": "def main():\n    return run_app()\n",
                    "score": 0.82,
                    "line_number": 3,
                    "metadata": {},
                }
            ]
        )
    )
    exec_ctx = {"settings": _settings()}
    literal_search = AsyncMock(return_value={"success": True, "results": [], "count": 0, "mode": "literal"})

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, False)),
    ), patch(
        "victor.tools.code_search_tool._literal_search",
        new=literal_search,
    ):
        result = await code_search(
            query="main entrypoint",
            path=str(tmp_path),
            k=3,
            mode="literal",
            _exec_ctx=exec_ctx,
        )

    literal_search.assert_awaited_once_with("main entrypoint", str(tmp_path), 3, None)
    assert result["success"] is True
    assert result["mode"] == "semantic"
    assert result["metadata"]["requested_mode"] == "literal"
    assert result["metadata"]["fallback_mode"] == "semantic"
    assert result["metadata"]["fallback_reason"] == "literal_no_results"
    assert "mode_fallback=semantic" in result["metadata"]["filters_applied"]


@pytest.mark.asyncio
async def test_code_search_strips_vectors_and_console_only_fields(tmp_path) -> None:
    """Semantic search responses should omit vectors while keeping rich preview fields."""
    mock_index = SimpleNamespace(
        semantic_search=AsyncMock(
            return_value=[
                {
                    "file_path": "src/main.py",
                    "content": "def main():\n    return run_app()\n",
                    "score": 0.82,
                    "line_number": 3,
                    "metadata": {
                        "id": "symbol:src/main.py:main",
                        "vector": [0.1, 0.2, 0.3],
                        "file_path": "src/main.py",
                        "content": "def main():\n    return run_app()\n",
                        "symbol_type": "function",
                        "end_line": 12,
                        "language": "python",
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
            query="main entry point",
            path=str(tmp_path),
            k=3,
            _exec_ctx=exec_ctx,
        )

    hit = result["results"][0]
    assert result["success"] is True
    assert result["contains_markup"] is True
    assert "src/main.py" in result["formatted_results"]
    assert "vector" not in hit["metadata"]
    assert "content" not in hit["metadata"]
    assert "file_path" not in hit["metadata"]
    assert hit["metadata"]["symbol_type"] == "function"
    assert hit["metadata"]["end_line"] == 12
    assert hit["metadata"]["language"] == "python"


@pytest.mark.asyncio
async def test_code_search_semantic_mode_resolves_file_backed_snippet_for_opaque_payload(
    tmp_path,
) -> None:
    """Semantic hits should expose grounded code snippets even when backend content is opaque."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    target = source_dir / "parser.py"
    target.write_text(
        "import json\n\n"
        "def parse_json(data):\n"
        "    if not data:\n"
        "        return {}\n"
        "    return json.loads(data)\n",
        encoding="utf-8",
    )

    mock_index = SimpleNamespace(
        semantic_search=AsyncMock(
            return_value=[
                {
                    "file_path": "src/parser.py",
                    "content": "symbol:src/parser.py:parse_json",
                    "score": 0.77,
                    "line_number": 3,
                    "metadata": {
                        "unified_id": "symbol:src/parser.py:parse_json",
                        "end_line": 6,
                        "symbol_name": "parse_json",
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
            query="where is parse_json defined",
            path=str(tmp_path),
            k=3,
            _exec_ctx=exec_ctx,
        )

    hit = result["results"][0]
    assert "def parse_json(data):" in hit["snippet"]
    assert "return json.loads(data)" in hit["content"]
    assert result["metadata"]["chunking_strategy"] == "tree_sitter_structural"
    assert result["metadata"]["snippet_strategy"] == "line_window"
    assert result["metadata"]["vector_store"] == "lancedb"
    assert result["metadata"]["embedding_dimension"] == 384
    assert result["metadata"]["retrieval_utility"]["strategy"] == "bounded_code_utility"
    assert "src/parser.py" in result["formatted_results"]


@pytest.mark.asyncio
async def test_code_search_hybrid_mode_preserves_extension_filter_for_keyword_side(
    tmp_path,
) -> None:
    """Hybrid keyword retrieval should honor the caller's extension filter."""
    mock_index = SimpleNamespace(
        semantic_search=AsyncMock(
            return_value=[
                {
                    "file_path": "src/main.py",
                    "content": "def main():\n    return run_app()\n",
                    "score": 0.82,
                    "line_number": 3,
                    "metadata": {},
                }
            ]
        )
    )
    exec_ctx = {"settings": _settings(enable_hybrid_search=True)}
    filters = SearchFilters(extensions=["py"])

    literal_search = AsyncMock(
        return_value={
            "success": True,
            "results": [
                {
                    "file_path": "src/main.py",
                    "content": "def main(): return run_app()",
                    "score": 1.0,
                    "line_number": 3,
                    "metadata": {},
                }
            ],
        }
    )

    class _HybridEngine:
        def combine_results(self, semantic_results, keyword_results, max_results):
            return [
                SimpleNamespace(
                    file_path="src/main.py",
                    content="def main(): return run_app()",
                    combined_score=0.91,
                    semantic_score=0.82,
                    keyword_score=1.0,
                    line_number=3,
                    metadata={},
                )
            ]

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, False)),
    ), patch(
        "victor.tools.code_search_tool._literal_search",
        new=literal_search,
    ), patch(
        "victor.framework.search.create_hybrid_search_engine",
        new=lambda semantic_weight, keyword_weight: _HybridEngine(),
    ):
        result = await code_search(
            query="main entrypoint",
            path=str(tmp_path),
            k=3,
            filters=filters,
            _exec_ctx=exec_ctx,
        )

    literal_search.assert_awaited_once_with("main entrypoint", str(tmp_path), 6, exts=["py"])
    assert result["success"] is True
    assert result["mode"] == "hybrid"


@pytest.mark.asyncio
async def test_code_search_hybrid_mode_uses_keyword_results_when_semantic_is_empty(
    tmp_path,
) -> None:
    """Hybrid mode should still return keyword hits when semantic retrieval is empty."""
    mock_index = SimpleNamespace(semantic_search=AsyncMock(return_value=[]))
    exec_ctx = {"settings": _settings(enable_hybrid_search=True)}

    literal_search = AsyncMock(
        return_value={
            "success": True,
            "results": [
                {
                    "file_path": "src/main.py",
                    "content": "def main(): return run_app()",
                    "score": 1.0,
                    "line_number": 3,
                    "metadata": {},
                }
            ],
        }
    )

    class _HybridEngine:
        def combine_results(self, semantic_results, keyword_results, max_results):
            assert semantic_results == []
            assert len(keyword_results) == 1
            return [
                SimpleNamespace(
                    file_path="src/main.py",
                    content="def main(): return run_app()",
                    combined_score=0.4,
                    semantic_score=0.0,
                    keyword_score=1.0,
                    line_number=3,
                    metadata={},
                )
            ]

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, False)),
    ), patch(
        "victor.tools.code_search_tool._literal_search",
        new=literal_search,
    ), patch(
        "victor.framework.search.create_hybrid_search_engine",
        new=lambda semantic_weight, keyword_weight: _HybridEngine(),
    ):
        result = await code_search(
            query="main entrypoint",
            path=str(tmp_path),
            k=3,
            _exec_ctx=exec_ctx,
        )

    literal_search.assert_awaited_once_with("main entrypoint", str(tmp_path), 6, exts=None)
    assert result["success"] is True
    assert result["mode"] == "hybrid"
    assert result["count"] == 1
    assert result["results"][0]["file_path"] == "src/main.py"
    assert result["results"][0]["search_mode"] == "hybrid"


@pytest.mark.asyncio
async def test_code_search_semantic_mode_applies_bounded_utility_reranking(tmp_path) -> None:
    """Semantic results should lift implementation code ahead of weaker duplicate/test hits."""
    source_dir = tmp_path / "victor"
    tests_dir = tmp_path / "tests"
    source_dir.mkdir()
    tests_dir.mkdir()

    (source_dir / "parser.py").write_text(
        "def parse_json(data):\n"
        "    return data\n\n"
        "def parse_json_or_none(data):\n"
        "    return data or None\n",
        encoding="utf-8",
    )
    (tests_dir / "test_parser.py").write_text(
        "def test_parse_json():\n"
        "    assert parse_json('{}') == '{}'\n",
        encoding="utf-8",
    )

    mock_index = SimpleNamespace(
        semantic_search=AsyncMock(
            return_value=[
                {
                    "file_path": "tests/test_parser.py",
                    "content": "def test_parse_json(): assert parse_json('{}') == '{}'",
                    "score": 0.92,
                    "line_number": 1,
                    "symbol_name": "test_parse_json",
                },
                {
                    "file_path": "victor/parser.py",
                    "content": "def parse_json(data): return data",
                    "score": 0.89,
                    "line_number": 1,
                    "symbol_name": "parse_json",
                },
                {
                    "file_path": "victor/parser.py",
                    "content": "def parse_json(data): return data",
                    "score": 0.88,
                    "line_number": 4,
                    "symbol_name": "parse_json_or_none",
                },
            ]
        )
    )
    exec_ctx = {"settings": _settings()}

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, False)),
    ):
        result = await code_search(
            query="where is parse_json defined",
            path=str(tmp_path),
            k=3,
            _exec_ctx=exec_ctx,
        )

    ordered_paths = [item["file_path"] for item in result["results"]]
    assert ordered_paths[0] == "victor/parser.py"
    assert ordered_paths.count("victor/parser.py") == 2
    assert "tests/test_parser.py" in ordered_paths[1:]
    assert result["metadata"]["retrieval_utility"]["repeated_file_hits"] == 1
    assert result["metadata"]["retrieval_utility"]["file_diversity"] == 2


@pytest.mark.asyncio
async def test_code_search_reports_non_timeout_index_build_fallback_reason(
    tmp_path, recwarn
) -> None:
    """Index build exceptions should not be mislabeled as timeouts."""
    exec_ctx = {"settings": _settings()}
    literal_result = {"success": True, "results": [], "count": 0, "mode": "literal"}

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(side_effect=RuntimeError("index build failed")),
    ), patch(
        "victor.tools.code_search_tool._literal_search",
        new=AsyncMock(return_value=dict(literal_result)),
    ) as literal_search:
        result = await code_search(
            query="parse json entrypoint",
            path=str(tmp_path),
            k=3,
            _exec_ctx=exec_ctx,
        )

    literal_search.assert_awaited_once()
    assert result["fallback"] == "semantic_index_error"
    assert not recwarn


@pytest.mark.asyncio
async def test_code_search_caches_index_build_failure_in_plain_dict_fallback(
    tmp_path, monkeypatch
) -> None:
    """The default dict-backed failure cache should record build failures."""
    exec_ctx = {"settings": _settings()}
    literal_result = {"success": True, "results": [], "count": 0, "mode": "literal"}
    failure_cache: dict[str, object] = {}
    failing_build = AsyncMock(side_effect=RuntimeError("index build failed"))
    failing_build._failure_cache = failure_cache

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=failing_build,
    ), patch(
        "victor.tools.code_search_tool._literal_search",
        new=AsyncMock(return_value=dict(literal_result)),
    ):
        result = await code_search(
            query="parse json entrypoint",
            path=str(tmp_path),
            k=3,
            _exec_ctx=exec_ctx,
        )

    assert result["fallback"] == "semantic_index_error"
    assert len(failure_cache) == 1
    cached_entry = next(iter(failure_cache.values()))
    assert cached_entry.value["error"] == "index build failed"


@pytest.mark.asyncio
async def test_code_search_reports_non_timeout_semantic_search_fallback_reason(tmp_path) -> None:
    """Semantic search exceptions should not be mislabeled as timeouts."""
    mock_index = SimpleNamespace(
        semantic_search=AsyncMock(side_effect=RuntimeError("semantic search failed"))
    )
    exec_ctx = {"settings": _settings()}
    literal_result = {"success": True, "results": [], "count": 0, "mode": "literal"}

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, False)),
    ), patch(
        "victor.tools.code_search_tool._literal_search",
        new=AsyncMock(return_value=dict(literal_result)),
    ) as literal_search:
        result = await code_search(
            query="parse json entrypoint",
            path=str(tmp_path),
            k=3,
            _exec_ctx=exec_ctx,
        )

    literal_search.assert_awaited_once()
    assert result["fallback"] == "semantic_search_error"


@pytest.mark.asyncio
async def test_code_search_literal_fallback_preserves_mode_context_after_semantic_failure(
    tmp_path,
) -> None:
    """Literal fallback should keep requested-mode context after mode->semantic downgrade."""
    mock_index = SimpleNamespace(
        localize_issue=AsyncMock(side_effect=NotImplementedError("localize_issue unsupported")),
        semantic_search=AsyncMock(side_effect=RuntimeError("semantic search failed")),
    )
    exec_ctx = {"settings": _settings()}
    filters = SearchFilters(language="python")
    literal_result = {"success": True, "results": [], "count": 0, "mode": "literal"}

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, False)),
    ), patch(
        "victor.tools.code_search_tool._literal_search",
        new=AsyncMock(return_value=dict(literal_result)),
    ) as literal_search:
        result = await code_search(
            query="which files should I edit to add a logger parameter to BaseRepository",
            path=str(tmp_path),
            k=3,
            mode="localize",
            filters=filters,
            _exec_ctx=exec_ctx,
        )

    literal_search.assert_awaited_once()
    assert result["fallback"] == "semantic_search_error"
    assert result["metadata"]["requested_mode"] == "localize"
    assert result["metadata"]["fallback_mode"] == "semantic"
    assert result["metadata"]["vector_store"] == "lancedb"
    assert "mode_fallback=semantic" in result["metadata"]["filters_applied"]


@pytest.mark.asyncio
async def test_code_search_literal_fallback_preserves_requested_mode_on_index_build_failure(
    tmp_path,
) -> None:
    """Early literal fallback should keep requested-mode context when index build fails."""
    exec_ctx = {"settings": _settings()}
    filters = SearchFilters(language="python")
    literal_result = {"success": True, "results": [], "count": 0, "mode": "literal"}

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(side_effect=RuntimeError("index build failed")),
    ), patch(
        "victor.tools.code_search_tool._literal_search",
        new=AsyncMock(return_value=dict(literal_result)),
    ) as literal_search:
        result = await code_search(
            query="which files should I edit to add a logger parameter to BaseRepository",
            path=str(tmp_path),
            k=3,
            mode="localize",
            filters=filters,
            _exec_ctx=exec_ctx,
        )

    literal_search.assert_awaited_once()
    assert result["fallback"] == "semantic_index_error"
    assert result["metadata"]["requested_mode"] == "localize"
    assert result["metadata"]["fallback_mode"] == "literal"


@pytest.mark.asyncio
async def test_code_search_skips_semantic_search_when_cached_index_is_stale(tmp_path) -> None:
    """Known-stale semantic indexes should fall back immediately to literal search."""
    mock_index = SimpleNamespace(semantic_search=AsyncMock())
    exec_ctx = {"settings": _settings()}
    literal_result = {"success": True, "results": [], "count": 0, "mode": "literal"}
    fake_cache = {str(tmp_path): {"stale": True}}

    with patch(
        "victor.tools.code_search_tool._get_or_build_index",
        new=AsyncMock(return_value=(mock_index, False)),
    ), patch(
        "victor.tools.code_search_tool._get_index_cache",
        new=lambda exec_ctx=None: fake_cache,
    ), patch(
        "victor.tools.code_search_tool._literal_search",
        new=AsyncMock(return_value=dict(literal_result)),
    ) as literal_search:
        result = await code_search(
            query="parse json entrypoint",
            path=str(tmp_path),
            k=3,
            _exec_ctx=exec_ctx,
        )

    mock_index.semantic_search.assert_not_awaited()
    literal_search.assert_awaited_once()
    assert result["fallback"] == "semantic_index_stale"


def test_get_index_build_failure_cache_ignores_mock_cache_manager_fallback(monkeypatch) -> None:
    """Bare mocks should not fabricate a cache manager for failure-cache resolution."""
    sentinel_cache = {}
    monkeypatch.setattr(_get_or_build_index, "_failure_cache", sentinel_cache, raising=False)

    result = _get_index_build_failure_cache(MagicMock())

    assert result is sentinel_cache


def test_get_index_build_failure_cache_uses_tool_execution_context_namespace() -> None:
    """Typed execution contexts should still use their injected cache manager."""

    class _CacheManager:
        def __init__(self) -> None:
            self.seen: list[str] = []
            self.namespace = {}

        def get_namespace(self, name: str) -> dict[str, object]:
            self.seen.append(name)
            return self.namespace

    cache_manager = _CacheManager()
    exec_ctx = ToolExecutionContext(
        session_id="session-1",
        workspace_root=Path("/tmp"),
        cache_manager=cache_manager,
    )

    result = _get_index_build_failure_cache(exec_ctx)

    assert result is cache_manager.namespace
    assert cache_manager.seen == ["index_build_failures"]
