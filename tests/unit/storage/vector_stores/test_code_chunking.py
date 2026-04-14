from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import patch

import pytest

from victor.storage.vector_stores.base import EmbeddingConfig
from victor.storage.vector_stores.code_chunking import (
    CodeChunkingContext,
    SymbolSpanCodeChunker,
    TreeSitterParseContext,
    TreeSitterStructuralCodeChunker,
    create_code_chunker,
)
from victor.storage.vector_stores.proximadb_multi import ProximaDBMultiModelProvider


@dataclass
class FakeSymbol:
    name: str
    symbol_type: str
    line_start: int
    line_end: int
    parent_symbol: Optional[str] = None

    @property
    def qualified_name(self) -> str:
        if self.parent_symbol:
            return f"{self.parent_symbol}.{self.name}"
        return self.name


@dataclass
class FakeNode:
    type: str
    start_byte: int
    end_byte: int
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    named_children: list["FakeNode"] = field(default_factory=list)


class StubEmbeddingModel:
    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def embed_text(self, text: str) -> list[float]:
        del text
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def get_dimension(self) -> int:
        return 4


class FakeClient:
    def create_collection(self, name: str, config: Any = None, **kwargs: Any) -> dict[str, Any]:
        del config, kwargs
        return {"name": name}

    def create_graph(
        self,
        graph_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        del name, description, schema
        return {"graph_id": graph_id}


def _line_col_for_offset(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset)
    line_start = text.rfind("\n", 0, offset)
    if line_start < 0:
        col = offset
    else:
        col = offset - (line_start + 1)
    return line, col


def _make_node(
    text: str,
    snippet: str,
    node_type: str,
    *,
    start_search: int = 0,
    children: Optional[list[FakeNode]] = None,
) -> FakeNode:
    start = text.index(snippet, start_search)
    end = start + len(snippet)
    return FakeNode(
        type=node_type,
        start_byte=start,
        end_byte=end,
        start_point=_line_col_for_offset(text, start),
        end_point=_line_col_for_offset(text, end),
        named_children=list(children or []),
    )


def _make_root(text: str, children: list[FakeNode]) -> FakeNode:
    return FakeNode(
        type="module",
        start_byte=0,
        end_byte=len(text),
        start_point=(0, 0),
        end_point=_line_col_for_offset(text, len(text)),
        named_children=children,
    )


def test_symbol_span_chunker_splits_large_symbols_with_metadata() -> None:
    content = "\n".join(
        [
            "def parse_json(data):",
            "    first = normalize(data)",
            "    second = validate(first)",
            "    return serialize(second)",
        ]
    )
    symbols = [FakeSymbol("parse_json", "function", 1, 4)]

    chunker = SymbolSpanCodeChunker(chunk_size=4, chunk_overlap=1)
    chunks = chunker.chunk(
        "src/example.py",
        content,
        CodeChunkingContext(symbols=symbols),
    )

    assert len(chunks) >= 2
    assert all(chunk.symbol_name == "parse_json" for chunk in chunks)
    assert all(chunk.chunk_type == "function" for chunk in chunks)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 4


def test_tree_sitter_structural_chunker_preserves_module_level_context() -> None:
    content = "\n".join(
        [
            "import json",
            "",
            "CONSTANT = 1",
            "",
            "def parse_json(data):",
            "    value = json.loads(data)",
            "    return value",
            "",
            'result = handle(parse_json("{}"))',
        ]
    )
    symbols = [FakeSymbol("parse_json", "function", 5, 7)]

    import_node = _make_node(content, "import json", "import_statement")
    constant_node = _make_node(content, "CONSTANT = 1", "assignment")
    function_text = "\n".join(
        [
            "def parse_json(data):",
            "    value = json.loads(data)",
            "    return value",
        ]
    )
    function_node = _make_node(content, function_text, "function_definition")
    result_node = _make_node(
        content,
        'result = handle(parse_json("{}"))',
        "expression_statement",
    )
    root = _make_root(content, [import_node, constant_node, function_node, result_node])
    parse_context = TreeSitterParseContext.from_content(content, root)

    chunker = TreeSitterStructuralCodeChunker(chunk_size=8, chunk_overlap=0)
    chunks = chunker.chunk(
        "src/example.py",
        content,
        CodeChunkingContext(symbols=symbols, parse_context=parse_context),
    )

    assert len(chunks) >= 3
    assert any(chunk.symbol_name == "parse_json" for chunk in chunks)
    module_chunks = [chunk for chunk in chunks if chunk.symbol_name is None]
    assert module_chunks
    all_chunk_text = "\n".join(chunk.content for chunk in chunks)
    assert "import json" in all_chunk_text
    assert "CONSTANT = 1" in all_chunk_text
    assert 'result = handle(parse_json("{}"))' in all_chunk_text


def test_tree_sitter_structural_chunker_falls_back_without_parse_context() -> None:
    content = "\n".join(
        [
            "def parse_json(data):",
            "    value = normalize(data)",
            "    return value",
        ]
    )
    symbols = [FakeSymbol("parse_json", "function", 1, 3)]
    context = CodeChunkingContext(symbols=symbols)

    structural = TreeSitterStructuralCodeChunker(chunk_size=4, chunk_overlap=0)
    fallback = SymbolSpanCodeChunker(chunk_size=4, chunk_overlap=0)

    assert structural.chunk("src/example.py", content, context) == fallback.chunk(
        "src/example.py",
        content,
        context,
    )


def test_create_code_chunker_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown code chunking strategy"):
        create_code_chunker("does_not_exist", chunk_size=10, chunk_overlap=0)


def test_provider_uses_configured_structural_chunker() -> None:
    config = EmbeddingConfig(
        vector_store="proximadb_multi",
        embedding_model_type="sentence-transformers",
        embedding_model_name="all-MiniLM-L12-v2",
        distance_metric="cosine",
        extra_config={
            "workspace": "victor_test_repo",
            "dimension": 4,
            "chunk_size": 8,
            "chunk_overlap": 0,
            "code_chunking_strategy": "tree_sitter_structural",
        },
    )

    with patch(
        "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
        return_value=StubEmbeddingModel(),
    ):
        provider = ProximaDBMultiModelProvider(config, client=FakeClient())

    assert provider._code_chunking_strategy == "tree_sitter_structural"
    assert isinstance(provider._code_chunker, TreeSitterStructuralCodeChunker)
