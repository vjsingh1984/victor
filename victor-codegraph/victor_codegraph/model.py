"""Canonical, neutral data model for code symbols, relations, and chunks.

This is the *union* of the two donor implementations (ProximaDB SDK ``code.py`` and
Victor ``victor-coding``): ProximaDB contributed the richer symbol/relation taxonomy;
Victor contributed the size-aware ``CodeChunk`` with hierarchical IDs. The model carries
no SaaS/DB/framework concept, so every consumer can depend on it freely.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class CodeSymbolType(IntEnum):
    """Kinds of code symbol that can be extracted (superset across languages)."""

    FILE = 1
    MODULE = 2
    PACKAGE = 3
    CLASS = 4
    INTERFACE = 5
    TRAIT = 6
    STRUCT = 7
    ENUM = 8
    FUNCTION = 9
    METHOD = 10
    CONSTRUCTOR = 11
    PROPERTY = 12
    FIELD = 13
    CONSTANT = 14
    VARIABLE = 15
    PARAMETER = 16
    TYPE_ALIAS = 17
    MACRO = 18


class CodeRelationType(IntEnum):
    """Directed relationships between code symbols."""

    CALLS = 1
    CALLED_BY = 2
    EXTENDS = 3
    IMPLEMENTS = 4
    USES_TYPE = 5
    RETURNS_TYPE = 6
    IMPORTS = 7
    IMPORTED_BY = 8
    DEPENDS_ON = 9
    CONTAINS = 10
    CONTAINED_BY = 11
    DEFINES = 12
    REFERENCES = 13
    REFERENCED_BY = 14
    OVERRIDES = 15
    OVERRIDDEN_BY = 16
    TESTS = 17
    TESTED_BY = 18


@dataclass
class SourceLocation:
    """Where a symbol lives in source. Lines are 1-based; bytes are 0-based."""

    file_path: str
    start_line: int = 0
    start_column: int = 0
    end_line: int = 0
    end_column: int = 0
    byte_offset: int = 0
    byte_length: int = 0


@dataclass
class CodeSymbol:
    """A semantic code unit (function/class/method/struct/...)."""

    id: str
    symbol_type: CodeSymbolType
    fully_qualified_name: str
    simple_name: str
    location: SourceLocation
    source_code: str
    language: str
    documentation: str | None = None
    signature: str | None = None
    modifiers: list[str] = field(default_factory=list)
    scope_chain: list[str] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    return_type: str | None = None
    complexity: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeRelation:
    """A directed edge between two symbols (by id)."""

    from_symbol_id: str
    to_symbol_id: str
    relation_type: CodeRelationType
    call_site: SourceLocation | None = None
    context: str | None = None
    confidence: float = 1.0


@dataclass
class ParsedCode:
    """Result of parsing one source file."""

    file_path: str
    language: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    relations: list[CodeRelation] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    content_hash: str = ""


@dataclass
class CodeChunk:
    """An embeddable, size-capped chunk projected from a symbol.

    A symbol within the size budget yields exactly one chunk; an oversized symbol is
    body-split into several chunks sharing ``symbol_id`` (see ``sizing``). ``chunk_id``
    is hierarchical and deterministic so incremental re-index is an idempotent upsert.
    """

    chunk_id: str
    text: str
    symbol_id: str
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] = field(default_factory=dict)


def deterministic_symbol_id(file_path: str, name: str, line: int, column: int = 0) -> str:
    """Stable 16-hex id keyed on (file, name, line, col) — same input, same id."""

    key = f"{file_path}:{name}:{line}:{column}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def content_hash(content: str) -> str:
    """SHA-256 of file content, for change detection on the re-index hot path."""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()
