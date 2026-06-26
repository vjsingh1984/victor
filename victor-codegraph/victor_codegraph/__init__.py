"""victor-codegraph — shared code->CPG chunker.

One tree-sitter symbol+relation chunker, three consumers (Victor, ProximaDB SDK,
AnvaiOps). See ProximaDB ADR-029 / Victor ADR-014.

    from victor_codegraph import chunk, parse, to_proxima_records, ChunkConfig

    chunks = chunk(source, file_path="foo.py")          # size-capped, embeddable
    parsed = parse(source, file_path="foo.py")           # symbols + relations
    records = to_proxima_records(parsed, repo_graph_id="myrepo")
"""

from __future__ import annotations

from .adapter import relation_to_record, symbol_to_record, to_proxima_records
from .config import ChunkConfig
from .languages import detect_language
from .model import (
    CodeChunk,
    CodeRelation,
    CodeRelationType,
    CodeSymbol,
    CodeSymbolType,
    ParsedCode,
    SourceLocation,
)
from .parser import chunk, parse

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "chunk",
    "parse",
    "ChunkConfig",
    "detect_language",
    "to_proxima_records",
    "symbol_to_record",
    "relation_to_record",
    "CodeChunk",
    "CodeSymbol",
    "CodeRelation",
    "CodeSymbolType",
    "CodeRelationType",
    "ParsedCode",
    "SourceLocation",
]
