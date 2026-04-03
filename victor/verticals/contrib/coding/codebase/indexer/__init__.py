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

"""Codebase indexing for intelligent code awareness.

This is the HIGHEST PRIORITY feature to match Claude Code capabilities.

Supports both keyword search and semantic search (with embeddings).

Features:
- AST-based symbol extraction
- Keyword and semantic search
- File watching for automatic staleness detection
- Lazy reindexing when stale
- Incremental updates for changed files

This package was decomposed from a single indexer.py file into:
- models.py: Pydantic models (IndexedSymbol, FileMetadata, CodebaseFileHandler)
- queries.py: Tree-sitter query dictionaries for multi-language support
- parallel.py: Module-level functions for parallel file indexing
- visitor.py: AST visitor for Python symbol extraction
- background.py: Background indexer service
- core.py: Main CodebaseIndex class
"""

from victor.verticals.contrib.coding.codebase.indexer.models import (  # noqa: F401
    CodebaseFileHandler,
    FileMetadata,
    IndexedSymbol,
    Symbol,
    WATCHDOG_AVAILABLE,
)
from victor.verticals.contrib.coding.codebase.indexer.core import CodebaseIndex  # noqa: F401
from victor.verticals.contrib.coding.codebase.indexer.visitor import SymbolVisitor  # noqa: F401
from victor.verticals.contrib.coding.codebase.indexer.background import (  # noqa: F401
    BackgroundIndexerService,
    start_background_indexer,
)
from victor.verticals.contrib.coding.codebase.indexer.queries import (  # noqa: F401
    EXTENSION_TO_LANGUAGE,
    SYMBOL_QUERIES,
    CALL_QUERIES,
    REFERENCE_QUERIES,
    INHERITS_QUERIES,
    IMPLEMENTS_QUERIES,
    COMPOSITION_QUERIES,
    ENCLOSING_NAME_FIELDS,
)

__all__ = [
    "CodebaseIndex",
    "CodebaseFileHandler",
    "FileMetadata",
    "IndexedSymbol",
    "Symbol",
    "SymbolVisitor",
    "BackgroundIndexerService",
    "start_background_indexer",
    "WATCHDOG_AVAILABLE",
    "EXTENSION_TO_LANGUAGE",
    "SYMBOL_QUERIES",
    "CALL_QUERIES",
    "REFERENCE_QUERIES",
    "INHERITS_QUERIES",
    "IMPLEMENTS_QUERIES",
    "COMPOSITION_QUERIES",
    "ENCLOSING_NAME_FIELDS",
]
