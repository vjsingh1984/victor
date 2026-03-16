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

"""CodebaseIndex: main indexer class for intelligent code understanding.

Supports AST-based symbol extraction, keyword and semantic search,
dependency graph analysis, and file watching for auto-staleness detection.
"""

import ast
import asyncio
import json
import logging
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from tree_sitter import Query

from victor.verticals.contrib.coding.codebase.graph.protocol import GraphEdge, GraphNode
from victor.verticals.contrib.coding.codebase.tree_sitter_extractor import TreeSitterExtractor
from victor.verticals.contrib.coding.codebase.unified_extractor import (
    UnifiedSymbolExtractor,
    EnrichedSymbol,
)
from victor.verticals.contrib.coding.languages.registry import get_language_registry
from victor.verticals.contrib.coding.languages.tiers import get_tier, LanguageTier
from victor.verticals.contrib.coding.codebase.graph.registry import create_graph_store
from victor.verticals.contrib.coding.codebase.symbol_resolver import SymbolResolver
from victor.core.utils.ast_helpers import (
    extract_base_classes,
    is_stdlib_module as _is_stdlib_module,
)

from victor.verticals.contrib.coding.codebase.indexer.models import (
    CodebaseFileHandler,
    FileMetadata,
    Symbol,
    WATCHDOG_AVAILABLE,
    Observer,
)
from victor.verticals.contrib.coding.codebase.indexer.queries import (
    _PRIMITIVE_TYPES,
    CALL_QUERIES,
    COMPOSITION_QUERIES,
    ENCLOSING_NAME_FIELDS,
    EXTENSION_TO_LANGUAGE,
    IMPLEMENTS_QUERIES,
    INHERITS_QUERIES,
    REFERENCE_QUERIES,
    SYMBOL_QUERIES,
)
from victor.verticals.contrib.coding.codebase.indexer.parallel import (
    _process_file_parallel,
)

if TYPE_CHECKING:
    from victor.verticals.contrib.coding.codebase.embeddings.base import BaseEmbeddingProvider
    from victor.verticals.contrib.coding.codebase.graph.protocol import GraphStoreProtocol


logger = logging.getLogger(__name__)


class CodebaseIndex:
    """Indexes codebase for intelligent code understanding.

    This is the foundation for matching Claude Code's codebase awareness.

    Supports:
    - AST-based symbol extraction
    - Keyword search
    - Semantic search (with embeddings)
    - Dependency graph analysis
    """

    # All source file patterns to watch (multi-language support)
    WATCHED_PATTERNS = [
        "*.py",
        "*.pyw",  # Python
        "*.js",
        "*.jsx",
        "*.mjs",  # JavaScript
        "*.ts",
        "*.tsx",  # TypeScript
        "*.go",  # Go
        "*.rs",  # Rust
        "*.java",
        "*.kt",
        "*.scala",  # JVM
        "*.rb",  # Ruby
        "*.php",  # PHP
        "*.cs",  # C#
        "*.cpp",
        "*.cc",
        "*.c",
        "*.h",
        "*.hpp",  # C/C++
        "*.swift",  # Swift
        "*.dart",  # Dart
        "*.json",
        "*.yaml",
        "*.yml",
        "*.toml",
        "*.ini",
        "*.properties",
        "*.conf",
        "*.hocon",
    ]

    # Unified ID generation for graph-embedding correlation
    @staticmethod
    def make_symbol_id(file_path: str, symbol_name: str) -> str:
        """Generate unified symbol ID for graph and embedding correlation."""
        return f"symbol:{file_path}:{symbol_name}"

    @staticmethod
    def make_file_id(file_path: str) -> str:
        """Generate unified file ID for graph nodes."""
        return f"file:{file_path}"

    def __init__(
        self,
        root_path: str,
        ignore_patterns: Optional[List[str]] = None,
        use_embeddings: bool = True,
        embedding_config: Optional[Dict[str, Any]] = None,
        enable_watcher: bool = True,
        graph_store: Optional["GraphStoreProtocol"] = None,
        graph_store_name: Optional[str] = None,
        graph_path: Optional[Path] = None,
        parallel_workers: int = 0,
    ):
        """Initialize codebase indexer.

        Args:
            root_path: Root directory of the codebase
            ignore_patterns: Patterns to ignore (e.g., ["venv/", "node_modules/"])
            use_embeddings: Enable semantic search with embeddings (default: True).
            embedding_config: Configuration for embedding provider (optional)
            enable_watcher: Whether to enable file watching for auto-staleness detection
            graph_store: Optional graph store for symbol relationships.
            graph_store_name: Optional graph backend name (currently only "sqlite")
            graph_path: Optional explicit graph store path
            parallel_workers: Number of parallel workers for file indexing.
                0 = auto-detect (min(cpu_count, 8)), 1 = sequential (default: 0)
        """
        self.root = Path(root_path).resolve()

        # Parallel indexing configuration
        if parallel_workers == 0:
            import multiprocessing

            self._parallel_workers = min(multiprocessing.cpu_count(), 4)
        else:
            self._parallel_workers = parallel_workers

        self.ignore_patterns = ignore_patterns or [
            "venv/",
            "env/",
            "node_modules/",
            "__pycache__/",
            "*.pyc",
            "dist/",
            "build/",
            "out/",
            "htmlcov/",
            "coverage/",
            "vendor/",
            "third_party/",
            "archive/",
        ]

        # Indexed data
        self.files: Dict[str, FileMetadata] = {}
        self.symbols: Dict[str, Symbol] = {}
        self.symbol_index: Dict[str, List[str]] = {}

        # Staleness tracking
        self._is_indexed = False
        self._is_stale = False
        self._changed_files: Set[str] = set()
        self._last_indexed: Optional[float] = None
        self._staleness_lock = threading.Lock()

        # File watcher
        self._watcher_enabled = enable_watcher and WATCHDOG_AVAILABLE
        self._observer: Optional[Observer] = None
        self._file_handler: Optional[CodebaseFileHandler] = None

        # Callbacks for change notifications
        self._change_callbacks: List[Callable[[str], None]] = []

        # Graph store
        if graph_store is None:
            self.graph_store: Optional["GraphStoreProtocol"] = create_graph_store(
                project_path=self.root
            )
        else:
            self.graph_store = graph_store
        self._graph_nodes: List[GraphNode] = []
        self._graph_edges: List[GraphEdge] = []
        self._pending_call_edges: List[tuple[str, str, str]] = []
        self._pending_inherit_edges: List[tuple[str, str, str]] = []
        self._pending_implements_edges: List[tuple[str, str, str]] = []
        self._pending_compose_edges: List[tuple[str, str, str]] = []
        self._symbol_resolver = SymbolResolver()

        # Embedding support
        self.use_embeddings = use_embeddings
        self.embedding_provider: Optional["BaseEmbeddingProvider"] = None
        if use_embeddings:
            self._initialize_embeddings(embedding_config)

        # Unified tree-sitter extractor
        self._language_registry = get_language_registry()
        self._language_registry.discover_plugins()
        self._tree_sitter_extractor = TreeSitterExtractor(self._language_registry)

        # Tier-aware unified symbol extractor
        self._unified_extractor = UnifiedSymbolExtractor(
            tree_sitter=self._tree_sitter_extractor,
            lsp_service=None,
            enable_lsp_enrichment=True,
        )

    def _reset_graph_buffers(self) -> None:
        self._graph_nodes = []
        self._graph_edges = []
        self._pending_call_edges = []
        self._pending_inherit_edges = []
        self._pending_implements_edges = []
        self._pending_compose_edges = []
        self._symbol_resolver = SymbolResolver()

    def _enriched_to_symbol(self, enriched: EnrichedSymbol, relative_path: str) -> Symbol:
        """Convert an EnrichedSymbol to the legacy Symbol format."""
        signature = enriched.signature
        if not signature and (enriched.parameters or enriched.return_type):
            if enriched.symbol_type in ("function", "method"):
                params = ", ".join(enriched.parameters) if enriched.parameters else ""
                ret = f" -> {enriched.return_type}" if enriched.return_type else ""
                prefix = "async " if enriched.is_async else ""
                signature = f"{prefix}def {enriched.name}({params}){ret}"

        return Symbol(
            name=enriched.name,
            type=enriched.symbol_type,
            file_path=relative_path,
            line_number=enriched.line_number,
            end_line=enriched.end_line,
            docstring=enriched.docstring,
            signature=signature,
            parent_symbol=enriched.parent_symbol,
        )

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on ignore patterns."""
        try:
            rel_path = path.relative_to(self.root)
            path_str = str(rel_path)
        except ValueError:
            path_str = str(path)
            rel_path = path

        for part in rel_path.parts:
            if part.startswith(".") and part not in (".", ".."):
                return True

        for pattern in self.ignore_patterns:
            if pattern.endswith("/"):
                if pattern[:-1] in path_str or f"/{pattern[:-1]}/" in path_str:
                    return True
            elif "*" in pattern:
                import fnmatch

                if fnmatch.fnmatch(path.name, pattern):
                    return True
            else:
                if pattern in path_str:
                    return True
        return False

    async def index_codebase(self) -> None:
        """Index the entire codebase."""
        self._reset_graph_buffers()
        self.files.clear()
        self.symbols.clear()
        self.symbol_index.clear()

        if self.graph_store:
            try:
                await self.graph_store.delete_by_repo()
                logger.debug("Cleared graph store for full rebuild")
            except Exception as e:
                logger.warning(f"Failed to clear graph store: {e}")

        files_to_index: List[Tuple[Path, str]] = []
        for pattern in self.WATCHED_PATTERNS:
            for file_path in self.root.rglob(pattern):
                if file_path.is_file() and not self._should_ignore(file_path):
                    language = self._detect_language(file_path)
                    files_to_index.append((file_path, language))

        if self._parallel_workers > 1 and len(files_to_index) > 50:
            await self._index_files_parallel(files_to_index)
        else:
            for file_path, language in files_to_index:
                try:
                    await self._index_tree_sitter_file(file_path, language)
                except Exception as exc:
                    logger.debug(f"Failed to index {file_path}: {exc}")

        self._resolve_cross_file_calls()
        self._build_dependency_graph()

        if self.graph_store and self._graph_nodes:
            await self.graph_store.upsert_nodes(self._graph_nodes)
        if self.graph_store and self._graph_edges:
            await self.graph_store.upsert_edges(self._graph_edges)

        if self.use_embeddings and self.embedding_provider:
            try:
                await self.embedding_provider.clear_index()
                logger.debug("Cleared embedding index for full rebuild")
            except Exception as e:
                logger.warning(f"Failed to clear embedding index: {e}")

            documents = []
            for rel_path, file_meta in self.files.items():
                for symbol in file_meta.symbols:
                    text_for_embedding = self._get_symbol_embedding_text(symbol)
                    if text_for_embedding:
                        unified_id = self.make_symbol_id(rel_path, symbol.name)
                        documents.append(
                            {
                                "id": unified_id,
                                "content": text_for_embedding,
                                "metadata": {
                                    "file_path": rel_path,
                                    "symbol_name": symbol.name,
                                    "symbol_type": symbol.type,
                                    "line_number": symbol.line_number,
                                    "end_line": symbol.end_line,
                                },
                            }
                        )

            batch_size = 500
            embedding_count = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                try:
                    await self.embedding_provider.index_documents(batch)
                    embedding_count += len(batch)
                    if i > 0 and i % 5000 == 0:
                        logger.info(f"Embedded {embedding_count}/{len(documents)} symbols...")
                except Exception as e:
                    logger.warning(f"Failed to embed batch at {i}: {e}")

            logger.info(f"Created {embedding_count} embeddings for semantic search")

        self._is_indexed = True
        self._is_stale = False
        self._last_indexed = time.time()
        logger.info(f"Indexed {len(self.files)} files with {len(self.symbols)} symbols")

    async def _index_files_parallel(self, files_to_index: List[Tuple[Path, str]]) -> None:
        """Index files using parallel processing with ProcessPoolExecutor."""
        start_time = time.time()
        root_str = str(self.root)
        total_files = len(files_to_index)
        processed = 0
        errors = 0

        logger.info(
            f"Starting parallel indexing: {total_files} files, " f"{self._parallel_workers} workers"
        )

        tasks = [(str(file_path), root_str, language) for file_path, language in files_to_index]

        with ProcessPoolExecutor(max_workers=self._parallel_workers) as executor:
            futures = {executor.submit(_process_file_parallel, *task): task for task in tasks}

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        self._merge_parallel_result(result)
                        processed += 1
                    else:
                        errors += 1
                except Exception as exc:
                    logger.debug(f"Parallel index failed for {task[0]}: {exc}")
                    errors += 1

                if (processed + errors) % 500 == 0:
                    logger.debug(
                        f"Progress: {processed + errors}/{total_files} files "
                        f"({processed} ok, {errors} failed)"
                    )

        elapsed = time.time() - start_time
        files_per_sec = total_files / elapsed if elapsed > 0 else 0
        logger.info(
            f"Parallel indexing complete: {processed}/{total_files} files in {elapsed:.2f}s "
            f"({files_per_sec:.1f} files/sec, {errors} errors)"
        )

    def _merge_parallel_result(self, result: Dict[str, Any]) -> None:
        """Merge a parallel processing result into the index."""
        symbols = [
            Symbol(
                name=s["name"],
                type=s["type"],
                file_path=s["file_path"],
                line_number=s["line_number"],
                end_line=s.get("end_line"),
            )
            for s in result.get("symbols", [])
        ]

        metadata = FileMetadata(
            path=result["path"],
            language=result["language"],
            symbols=symbols,
            imports=result.get("imports", []),
            call_edges=result.get("call_edges", []),
            inherit_edges=result.get("inherit_edges", []),
            implements_edges=result.get("implements_edges", []),
            compose_edges=result.get("compose_edges", []),
            references=result.get("references", []),
            last_modified=result["last_modified"],
            indexed_at=time.time(),
            size=result["size"],
            lines=result["lines"],
        )

        self.files[metadata.path] = metadata
        self._record_symbols(metadata)

    async def ensure_indexed(self, auto_reindex: bool = True) -> None:
        """Ensure the index is ready for querying."""
        if not self._is_indexed:
            logger.debug("Index not built, building initial index")
            await self.index_codebase()
        elif auto_reindex and self._is_stale:
            logger.debug("Index is stale, rebuilding")
            await self.index_codebase()

    async def incremental_reindex(self, files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform incremental reindexing of changed files only."""
        stats = {
            "updated": [],
            "added": [],
            "removed": [],
            "unchanged": 0,
            "errors": [],
        }

        if not self._is_indexed:
            logger.info("No existing index, performing full index")
            await self.index_codebase()
            stats["added"] = list(self.files.keys())
            return stats

        if files:
            files_to_check = [Path(f) if not isinstance(f, Path) else f for f in files]
        else:
            files_to_check = []
            for pattern in self.WATCHED_PATTERNS:
                for file_path in self.root.rglob(pattern):
                    if file_path.is_file() and not self._should_ignore(file_path):
                        files_to_check.append(file_path)

        current_files: Set[str] = set()

        for file_path in files_to_check:
            if not file_path.exists():
                continue

            rel_path = str(file_path.relative_to(self.root))
            current_files.add(rel_path)

            try:
                current_mtime = file_path.stat().st_mtime

                if rel_path in self.files:
                    existing_mtime = self.files[rel_path].last_modified
                    if current_mtime <= existing_mtime:
                        stats["unchanged"] += 1
                        continue

                    language = self._detect_language(file_path)
                    await self._index_single_file(file_path, language)
                    stats["updated"].append(rel_path)
                    logger.debug(f"Updated index for: {rel_path}")
                else:
                    language = self._detect_language(file_path)
                    await self._index_single_file(file_path, language)
                    stats["added"].append(rel_path)
                    logger.debug(f"Added to index: {rel_path}")

            except Exception as e:
                logger.warning(f"Failed to reindex {rel_path}: {e}")
                stats["errors"].append({"file": rel_path, "error": str(e)})

        if not files:
            for indexed_path in list(self.files.keys()):
                if indexed_path not in current_files:
                    await self._remove_file_from_index(indexed_path)
                    stats["removed"].append(indexed_path)
                    logger.debug(f"Removed from index: {indexed_path}")

        with self._staleness_lock:
            self._is_stale = False
            self._changed_files.clear()
            self._last_indexed = time.time()

        total_changes = len(stats["updated"]) + len(stats["added"]) + len(stats["removed"])
        if total_changes > 0:
            logger.info(
                f"Incremental reindex: {len(stats['updated'])} updated, "
                f"{len(stats['added'])} added, {len(stats['removed'])} removed, "
                f"{stats['unchanged']} unchanged"
            )
        else:
            logger.debug(f"Incremental reindex: no changes detected ({stats['unchanged']} files)")

        return stats

    async def _index_single_file(self, file_path: Path, language: str) -> None:
        """Index a single file and update the index structures."""
        rel_path = str(file_path.relative_to(self.root))

        if rel_path in self.symbol_index:
            for symbol_key in self.symbol_index[rel_path]:
                self.symbols.pop(symbol_key, None)
            self.symbol_index[rel_path] = []

        await self._index_tree_sitter_file(file_path, language)

        if self.use_embeddings and self.embedding_provider and rel_path in self.files:
            file_meta = self.files[rel_path]
            for symbol in file_meta.symbols:
                text_for_embedding = self._get_symbol_embedding_text(symbol)
                if text_for_embedding:
                    try:
                        unified_id = self.make_symbol_id(rel_path, symbol.name)
                        await self.embedding_provider.index_document(
                            doc_id=unified_id,
                            content=text_for_embedding,
                            metadata={
                                "file_path": rel_path,
                                "symbol_name": symbol.name,
                                "symbol_type": symbol.type,
                                "line_number": symbol.line_number,
                            },
                        )
                    except Exception as e:
                        logger.debug(f"Failed to embed symbol {symbol.name}: {e}")

    async def _remove_file_from_index(self, rel_path: str) -> None:
        """Remove a file and its symbols from the index."""
        if rel_path in self.symbol_index:
            for symbol_key in self.symbol_index[rel_path]:
                self.symbols.pop(symbol_key, None)

                if self.use_embeddings and self.embedding_provider:
                    try:
                        await self.embedding_provider.delete_document(symbol_key)
                    except Exception:
                        pass

            del self.symbol_index[rel_path]

        self.files.pop(rel_path, None)

        if self.graph_store:
            try:
                await self.graph_store.delete_by_file(rel_path)
            except Exception as e:
                logger.debug(f"Failed to remove graph nodes for {rel_path}: {e}")

    def _get_symbol_embedding_text(self, symbol: Symbol) -> str:
        """Generate text representation of a symbol for embedding."""
        parts = []
        parts.append(f"{symbol.type} {symbol.name}")
        if symbol.signature:
            parts.append(symbol.signature)
        if symbol.docstring:
            parts.append(symbol.docstring[:500])
        return " ".join(parts)

    def _detect_language(self, file_path: Path, default: str = "python") -> str:
        """Detect language from extension for tree-sitter queries."""
        detected = self._language_registry.detect_language(file_path)
        if detected:
            return detected
        return EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), default)

    def _is_config_language(self, language: str) -> bool:
        """Return True if the language is a config/metadata file."""
        return language.startswith("config")

    def _extract_references(
        self, file_path: Path, language: str, fallback_calls: List[str], imports: List[str]
    ) -> List[str]:
        """Extract identifier references using tree-sitter when available."""
        refs: Set[str] = set(fallback_calls) | set(imports)

        query_src = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.references:
                query_src = plugin.tree_sitter_queries.references
        except Exception:
            pass

        if not query_src:
            query_src = REFERENCE_QUERIES.get(language)

        if not query_src:
            return list(refs)
        try:
            from victor.verticals.contrib.coding.codebase.tree_sitter_manager import get_parser
        except Exception:
            return list(refs)

        try:
            parser = get_parser(language)
        except Exception:
            return list(refs)

        if parser is None:
            return list(refs)

        try:
            from tree_sitter import QueryCursor

            content = file_path.read_bytes()
            tree = parser.parse(content)
            query = Query(parser.language, query_src)
            cursor = QueryCursor(query)
            captures_dict = cursor.captures(tree.root_node)
            for _capture_name, nodes in captures_dict.items():
                for node in nodes:
                    text = node.text.decode("utf-8", errors="ignore")
                    if text:
                        refs.add(text)
        except Exception:
            pass

        if not refs:
            try:
                text = file_path.read_text(encoding="utf-8")
                for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text):
                    refs.add(match.group(0))
            except Exception:
                return list(refs)

        return list(refs)

    def _extract_config_keys(self, content: str, language: str) -> List[tuple[str, int]]:
        """Extract top-level-ish config keys for JSON/YAML/INI/property files."""
        keys: dict[str, int] = {}

        def _walk(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    dotted = f"{prefix}.{k}" if prefix else str(k)
                    keys.setdefault(dotted, 1)
                    _walk(v, dotted)
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    dotted = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                    keys.setdefault(dotted, 1)
                    _walk(item, dotted)

        try:
            if language == "config-json":
                data = json.loads(content)
                _walk(data)
            elif language == "config-yaml":
                try:
                    import yaml

                    data = yaml.safe_load(content)
                    _walk(data)
                except Exception:
                    pass
        except Exception:
            pass

        if not keys:
            for match in re.finditer(
                r'^[\s"\']*([A-Za-z0-9_.\-]+)\s*[:=]', content, flags=re.MULTILINE
            ):
                key = match.group(1)
                line_no = content.count("\n", 0, match.start()) + 1
                keys.setdefault(key, line_no)

        return list(keys.items())

    def _extract_symbols_with_tree_sitter(self, file_path: Path, language: str) -> List[Symbol]:
        """Extract lightweight symbol declarations for non-Python languages via tree-sitter."""
        query_defs = SYMBOL_QUERIES.get(language)
        if not query_defs:
            return []
        symbols: List[Symbol] = []
        parser = None
        try:
            from victor.verticals.contrib.coding.codebase.tree_sitter_manager import get_parser

            try:
                parser = get_parser(language)
            except Exception:
                parser = None
        except Exception:
            parser = None

        if parser is not None:
            try:
                content = file_path.read_bytes()
                tree = parser.parse(content)
                for sym_type, query_src in query_defs:
                    try:
                        from tree_sitter import QueryCursor

                        query = Query(parser.language, query_src)
                        cursor = QueryCursor(query)
                        captures_dict = cursor.captures(tree.root_node)
                        for _capture_name, nodes in captures_dict.items():
                            for node in nodes:
                                text = node.text.decode("utf-8", errors="ignore")
                                if not text:
                                    continue
                                symbols.append(
                                    Symbol(
                                        name=text,
                                        type=sym_type,
                                        file_path=str(file_path.relative_to(self.root)),
                                        line_number=node.start_point[0] + 1,
                                    )
                                )
                    except Exception:
                        continue
            except Exception:
                symbols = []
        if symbols:
            return symbols

        # Regex fallback
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            return []

        regex_fallbacks: Dict[str, list[tuple[str, str]]] = {
            "javascript": [
                ("class", r"class\s+(\w+)"),
                ("function", r"function\s+(\w+)"),
            ],
            "typescript": [
                ("class", r"class\s+(\w+)"),
                ("function", r"function\s+(\w+)"),
            ],
            "go": [
                ("function", r"func\s+(?:\([\w\*\s,]+\)\s*)?(\w+)"),
                ("class", r"type\s+(\w+)\s+struct"),
            ],
            "java": [
                ("class", r"(?:class|interface)\s+(\w+)"),
                ("function", r"(?:public|private|protected|\s)+\s*\w+\s+(\w+)\s*\("),
            ],
        }
        for sym_type, pattern in regex_fallbacks.get(language, []):
            for match in re.finditer(pattern, text):
                name = match.group(1)
                symbols.append(
                    Symbol(
                        name=name,
                        type=sym_type,
                        file_path=str(file_path.relative_to(self.root)),
                        line_number=text.count("\n", 0, match.start()) + 1,
                    )
                )
        return symbols

    def _extract_inheritance(
        self, file_path: Path, language: str, symbols: List[Symbol]
    ) -> List[tuple[str, str]]:
        """Extract child->base inheritance edges."""
        edges: List[tuple[str, str]] = []
        if language == "python":
            for sym in symbols:
                if sym.type == "class" and sym.base_classes:
                    for base in sym.base_classes:
                        edges.append((sym.name, base))
            if edges:
                return edges
            try:
                import ast as py_ast

                tree = py_ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
                for node in py_ast.walk(tree):
                    if isinstance(node, py_ast.ClassDef):
                        for base_name in extract_base_classes(node):
                            edges.append((node.name, base_name))
            except Exception:
                pass
            if edges:
                return edges

        query_src = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.inheritance:
                query_src = plugin.tree_sitter_queries.inheritance
        except Exception:
            pass
        if not query_src:
            query_src = INHERITS_QUERIES.get(language)

        parser = None
        try:
            from victor.verticals.contrib.coding.codebase.tree_sitter_manager import get_parser

            parser = get_parser(language)
        except Exception:
            parser = None

        if parser is not None and query_src:
            try:
                from tree_sitter import QueryCursor

                content = file_path.read_bytes()
                tree = parser.parse(content)
                query = Query(parser.language, query_src)
                cursor = QueryCursor(query)
                for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                    child_nodes = cap_dict.get("child", [])
                    base_nodes = cap_dict.get("base", [])
                    if child_nodes and base_nodes:
                        child_text = child_nodes[0].text.decode("utf-8", errors="ignore")
                        base_text = base_nodes[0].text.decode("utf-8", errors="ignore")
                        if child_text and base_text:
                            edges.append((child_text, base_text))
            except Exception:
                pass

        if not edges:
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                return edges
            for match in re.finditer(r"class\s+(\w+)\s+extends\s+(\w+)", text):
                edges.append((match.group(1), match.group(2)))
        return edges

    def _extract_implements(
        self, file_path: Path, language: str, symbols: List[Symbol]
    ) -> List[tuple[str, str]]:
        """Extract child->interface implements edges for typed languages."""
        edges: List[tuple[str, str]] = []
        query_src = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.implements:
                query_src = plugin.tree_sitter_queries.implements
        except Exception:
            pass
        if not query_src:
            query_src = IMPLEMENTS_QUERIES.get(language)
        parser = None
        try:
            from victor.verticals.contrib.coding.codebase.tree_sitter_manager import get_parser

            parser = get_parser(language)
        except Exception:
            parser = None

        if parser is not None and query_src:
            try:
                from tree_sitter import QueryCursor

                content = file_path.read_bytes()
                tree = parser.parse(content)
                query = Query(parser.language, query_src)
                cursor = QueryCursor(query)
                for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                    child_nodes = cap_dict.get("child", [])
                    iface_nodes = cap_dict.get("interface", []) or cap_dict.get("base", [])
                    if child_nodes and iface_nodes:
                        child_text = child_nodes[0].text.decode("utf-8", errors="ignore")
                        iface_text = iface_nodes[0].text.decode("utf-8", errors="ignore")
                        if child_text and iface_text:
                            edges.append((child_text, iface_text))
            except Exception:
                pass

        if not edges:
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                return edges
            for match in re.finditer(r"class\s+(\w+)\s+implements\s+([\w, ]+)", text):
                child = match.group(1)
                bases = [b.strip() for b in match.group(2).split(",") if b.strip()]
                for base in bases:
                    edges.append((child, base))
        return edges

    def _extract_composition(
        self, file_path: Path, language: str, symbols: List[Symbol]
    ) -> List[tuple[str, str]]:
        """Extract has-a/composition edges (owner -> member type)."""
        edges: List[tuple[str, str]] = []

        if language == "python":
            for sym in symbols:
                if sym.type == "class" and hasattr(sym, "composition"):
                    edges.extend(sym.composition)
            return edges

        query_src = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.composition:
                query_src = plugin.tree_sitter_queries.composition
        except Exception:
            pass
        if not query_src:
            query_src = COMPOSITION_QUERIES.get(language)

        parser = None
        try:
            from victor.verticals.contrib.coding.codebase.tree_sitter_manager import get_parser

            parser = get_parser(language)
        except Exception:
            parser = None

        if parser is not None and query_src:
            try:
                from tree_sitter import QueryCursor

                content = file_path.read_bytes()
                tree = parser.parse(content)
                query = Query(parser.language, query_src)
                cursor = QueryCursor(query)
                for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                    owner_nodes = cap_dict.get("owner", [])
                    type_nodes = cap_dict.get("type", [])
                    if owner_nodes and type_nodes:
                        owner_text = owner_nodes[0].text.decode("utf-8", errors="ignore")
                        type_text = type_nodes[0].text.decode("utf-8", errors="ignore")
                        if owner_text and type_text:
                            edges.append((owner_text, type_text))
            except Exception:
                pass

        if edges:
            return edges

        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            return edges

        owner: Optional[str] = None
        for line in text.splitlines():
            class_match = re.search(r"class\s+(\w+)", line)
            if class_match:
                owner = class_match.group(1)
                continue
            if line.strip().startswith("}"):
                owner = owner if "class" in line else None
            field_match = re.search(r"(\w+)\s*[:]\s*(\w+)", line)
            java_field = re.search(r"(\w+)\s+(\w+)\s*;", line)
            new_expr = re.search(r"new\s+(\w+)\s*\(", line)
            target_type = None
            if field_match:
                target_type = field_match.group(2)
            elif java_field and owner:
                target_type = java_field.group(1)
            elif new_expr:
                target_type = new_expr.group(1)
            if owner and target_type:
                edges.append((owner, target_type))
        return edges

    def _find_enclosing_symbol_name(self, node, language: str) -> Optional[str]:
        """Best-effort caller lookup by walking ancestors."""
        fields = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.enclosing_scopes:
                fields = plugin.tree_sitter_queries.enclosing_scopes
        except Exception:
            pass
        if not fields:
            fields = ENCLOSING_NAME_FIELDS.get(language, [])
        current = node.parent
        method_name: Optional[str] = None
        class_name: Optional[str] = None
        while current is not None:
            for node_type, field_name in fields:
                if current.type == node_type:
                    field = current.child_by_field_name(field_name)
                    if not field:
                        continue
                    if field.type == "function_declarator":
                        inner = field.child_by_field_name("declarator")
                        if inner:
                            field = inner
                    text = field.text.decode("utf-8", errors="ignore")
                    if node_type in (
                        "class_declaration",
                        "interface_declaration",
                        "class_specifier",
                    ):
                        class_name = class_name or text
                    else:
                        method_name = method_name or text
            current = current.parent
        if method_name:
            if class_name:
                return f"{class_name}.{method_name}"
            return method_name
        return class_name

    def _extract_calls_with_tree_sitter(
        self, file_path: Path, language: str
    ) -> List[tuple[str, str]]:
        """Extract caller->callee pairs using tree-sitter (non-Python)."""
        query_src = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.calls:
                query_src = plugin.tree_sitter_queries.calls
        except Exception:
            pass
        if not query_src:
            query_src = CALL_QUERIES.get(language)
        if not query_src:
            return []
        try:
            from victor.verticals.contrib.coding.codebase.tree_sitter_manager import get_parser
        except Exception:
            return []
        try:
            parser = get_parser(language)
        except Exception:
            return []
        if parser is None:
            return []

        content = file_path.read_bytes()
        tree = parser.parse(content)
        try:
            query = Query(parser.language, query_src)
        except Exception:
            return []

        call_edges: List[tuple[str, str]] = []
        try:
            from tree_sitter import QueryCursor

            cursor = QueryCursor(query)
            captures_dict = cursor.captures(tree.root_node)
            for _capture_name, nodes in captures_dict.items():
                for node in nodes:
                    callee = node.text.decode("utf-8", errors="ignore")
                    caller = self._find_enclosing_symbol_name(node, language)
                    if caller and callee:
                        call_edges.append((caller, callee))
        except Exception:
            call_edges = []

        if call_edges:
            return call_edges

        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            return []
        pattern = re.compile(r"(\w+)\s*\(")
        caller = None
        for line in text.splitlines():
            func_decl = re.search(r"function\s+(\w+)", line)
            if func_decl:
                caller = func_decl.group(1)
            method_decl = re.search(r"(\w+)\s*\([^)]*\)\s*\{", line)
            if method_decl:
                caller = method_decl.group(1)
            for callee in pattern.findall(line):
                if caller and callee and callee not in {"function", caller}:
                    call_edges.append((caller, callee))
        return call_edges

    _IMPORT_QUERIES: Dict[str, str] = {
        "javascript": """
            (import_statement source: (string) @source)
            (call_expression
                function: (identifier) @_fn
                arguments: (arguments (string) @source)
                (#eq? @_fn "require"))
        """,
        "typescript": """
            (import_statement source: (string) @source)
            (call_expression
                function: (identifier) @_fn
                arguments: (arguments (string) @source)
                (#eq? @_fn "require"))
        """,
        "rust": """
            (use_declaration argument: (_) @source)
        """,
        "go": """
            (import_spec path: (interpreted_string_literal) @source)
        """,
        "java": """
            (import_declaration (scoped_identifier) @source)
        """,
    }

    def _extract_imports_with_tree_sitter(self, file_path: Path, language: str) -> List[str]:
        """Extract import/require/use statements using tree-sitter (non-Python)."""
        query_src = self._IMPORT_QUERIES.get(language)
        if not query_src:
            return []
        try:
            from victor.verticals.contrib.coding.codebase.tree_sitter_manager import get_parser
        except Exception:
            return []
        try:
            parser = get_parser(language)
        except Exception:
            return []
        if parser is None:
            return []

        imports: List[str] = []
        try:
            from tree_sitter import QueryCursor

            content = file_path.read_bytes()
            tree = parser.parse(content)
            query = Query(parser.language, query_src)
            cursor = QueryCursor(query)
            captures_dict = cursor.captures(tree.root_node)
            for node in captures_dict.get("source", []):
                text = node.text.decode("utf-8", errors="ignore")
                cleaned = text.strip("'\"")
                if cleaned:
                    imports.append(cleaned)
        except Exception:
            pass
        return imports

    async def _index_tree_sitter_file(self, file_path: Path, language: str) -> None:
        """Index a file using tier-aware symbol extraction."""
        try:
            stat = file_path.stat()
            content = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.debug(f"Skipping {file_path} due to read error: {exc}")
            return

        relative_path = str(file_path.relative_to(self.root))

        tier_config = get_tier(language)
        symbols: List[Symbol] = []

        if tier_config.tier in (LanguageTier.TIER_1, LanguageTier.TIER_2):
            try:
                enriched = await self._unified_extractor.extract_symbols(
                    file_path, language, content
                )
                if enriched:
                    symbols = [self._enriched_to_symbol(s, relative_path) for s in enriched]
                    logger.debug(
                        f"Unified extractor: {len(symbols)} symbols from {file_path.name} "
                        f"(tier={tier_config.tier.name})"
                    )
            except Exception as e:
                logger.debug(f"Unified extraction failed for {file_path}: {e}")

        if not symbols:
            symbols = self._extract_symbols_with_tree_sitter(file_path, language)

        call_edges = self._extract_calls_with_tree_sitter(file_path, language)
        imports: List[str] = []

        if language == "python":
            try:
                tree = ast.parse(content, filename=str(file_path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
            except Exception as exc:
                logger.debug(f"Failed to parse imports for {file_path}: {exc}")
        else:
            imports = self._extract_imports_with_tree_sitter(file_path, language)

        metadata = FileMetadata(
            path=str(file_path.relative_to(self.root)),
            language=language,
            symbols=symbols,
            imports=imports,
            last_modified=stat.st_mtime,
            indexed_at=time.time(),
            size=stat.st_size,
            lines=content.count("\n") + 1,
            call_edges=call_edges,
        )

        metadata.inherit_edges = self._extract_inheritance(file_path, language, symbols)
        metadata.implements_edges = self._extract_implements(file_path, language, symbols)
        metadata.compose_edges = self._extract_composition(file_path, language, symbols)

        metadata.references = self._extract_references(
            file_path,
            language,
            [callee for _, callee in call_edges],
            metadata.imports,
        )

        self.files[metadata.path] = metadata
        self._record_symbols(metadata)

    def _record_symbols(self, metadata: FileMetadata) -> None:
        """Record symbol metadata and populate graph buffers."""
        symbol_names = {s.name for s in metadata.symbols}

        file_node_id: Optional[str] = None
        if self.graph_store:
            file_node_id = self.make_file_id(metadata.path)
            self._graph_nodes.append(
                GraphNode(
                    node_id=file_node_id,
                    type="file",
                    name=Path(metadata.path).name,
                    file=metadata.path,
                    line=None,
                    lang=metadata.language,
                    metadata={"lines": metadata.lines, "size": metadata.size},
                )
            )

        for symbol in metadata.symbols:
            unified_id = self.make_symbol_id(metadata.path, symbol.name)
            self.symbols[unified_id] = symbol
            if metadata.path not in self.symbol_index:
                self.symbol_index[metadata.path] = []
            self.symbol_index[metadata.path].append(symbol.name)

            if self.graph_store:
                parent_id = None
                if symbol.parent_symbol:
                    parent_id = self.make_symbol_id(metadata.path, symbol.parent_symbol)

                self._graph_nodes.append(
                    GraphNode(
                        node_id=unified_id,
                        type=symbol.type,
                        name=symbol.name,
                        file=metadata.path,
                        line=symbol.line_number,
                        end_line=symbol.end_line,
                        lang=metadata.language,
                        signature=symbol.signature,
                        docstring=symbol.docstring,
                        parent_id=parent_id,
                        embedding_ref=unified_id,
                        metadata={},
                    )
                )
                self._graph_edges.append(
                    GraphEdge(
                        src=file_node_id or self.make_file_id(metadata.path),
                        dst=unified_id,
                        type="CONTAINS",
                        metadata={"path": metadata.path},
                    )
                )

        if self.graph_store and metadata.call_edges:
            for caller, callee in metadata.call_edges:
                caller_id = self.make_symbol_id(metadata.path, caller)
                if caller not in symbol_names or callee not in symbol_names:
                    self._pending_call_edges.append((caller_id, callee, metadata.path))
                    continue
                callee_id = self.make_symbol_id(metadata.path, callee)
                self._graph_edges.append(
                    GraphEdge(
                        src=caller_id,
                        dst=callee_id,
                        type="CALLS",
                        metadata={"path": metadata.path},
                    )
                )

        if self.graph_store and metadata.inherit_edges:
            for child, base in metadata.inherit_edges:
                child_id = self.make_symbol_id(metadata.path, child)
                if child not in symbol_names:
                    continue
                if base in symbol_names:
                    base_id = self.make_symbol_id(metadata.path, base)
                    self._graph_edges.append(
                        GraphEdge(
                            src=child_id,
                            dst=base_id,
                            type="INHERITS",
                            metadata={"path": metadata.path},
                        )
                    )
                else:
                    self._pending_inherit_edges.append((child_id, base, metadata.path))

        if self.graph_store and metadata.implements_edges:
            for child, base in metadata.implements_edges:
                child_id = self.make_symbol_id(metadata.path, child)
                if child not in symbol_names:
                    continue
                if base in symbol_names:
                    base_id = self.make_symbol_id(metadata.path, base)
                    self._graph_edges.append(
                        GraphEdge(
                            src=child_id,
                            dst=base_id,
                            type="IMPLEMENTS",
                            metadata={"path": metadata.path},
                        )
                    )
                else:
                    self._pending_implements_edges.append((child_id, base, metadata.path))

        if self.graph_store and metadata.compose_edges:
            for owner, member in metadata.compose_edges:
                owner_id = self.make_symbol_id(metadata.path, owner)
                if owner not in symbol_names:
                    continue
                if member in symbol_names:
                    member_id = self.make_symbol_id(metadata.path, member)
                    self._graph_edges.append(
                        GraphEdge(
                            src=owner_id,
                            dst=member_id,
                            type="COMPOSED_OF",
                            metadata={"path": metadata.path},
                        )
                    )
                else:
                    self._pending_compose_edges.append((owner_id, member, metadata.path))

        if self.graph_store and metadata.imports:
            for imp in metadata.imports:
                is_stdlib = _is_stdlib_module(imp)
                module_node_id = f"module:{imp}"
                self._graph_nodes.append(
                    GraphNode(
                        node_id=module_node_id,
                        type="module" if not is_stdlib else "stdlib_module",
                        name=imp,
                        file=metadata.path,
                        lang=metadata.language,
                    )
                )
                self._graph_edges.append(
                    GraphEdge(
                        src=self.make_file_id(metadata.path),
                        dst=module_node_id,
                        type="IMPORTS",
                        metadata={"path": metadata.path, "is_stdlib": is_stdlib},
                    )
                )

    def _resolve_cross_file_calls(self) -> None:
        """Resolve pending cross-file edges (CALLS, INHERITS, IMPLEMENTS, COMPOSED_OF)."""
        has_pending = (
            self._pending_call_edges
            or self._pending_inherit_edges
            or self._pending_implements_edges
            or self._pending_compose_edges
        )
        if not has_pending:
            return

        node_ids = list(self.symbols.keys())
        self._symbol_resolver.ingest(node_ids)

        _seen_external: set[str] = set()

        for caller_id, callee_name, file_path in self._pending_call_edges:
            target_id = self._symbol_resolver.resolve(callee_name, preferred_file=file_path)
            if not target_id:
                target_id = self._symbol_resolver.resolve(
                    callee_name.split(".")[-1], preferred_file=file_path
                )
            if not target_id:
                continue
            self._graph_edges.append(
                GraphEdge(
                    src=caller_id,
                    dst=target_id,
                    type="CALLS",
                    metadata={"path": file_path, "resolved": True},
                )
            )

        for child_id, base_name, file_path in self._pending_inherit_edges:
            target_id = self._symbol_resolver.resolve(base_name, preferred_file=file_path)
            if not target_id:
                target_id = self._symbol_resolver.resolve(
                    base_name.split(".")[-1], preferred_file=file_path
                )
            if not target_id:
                target_id = f"external_type:{base_name}"
                if target_id not in _seen_external:
                    _seen_external.add(target_id)
                    self._graph_nodes.append(
                        GraphNode(
                            node_id=target_id,
                            type="external_type",
                            name=base_name,
                            file=file_path,
                            metadata={"external": True},
                        )
                    )
            self._graph_edges.append(
                GraphEdge(
                    src=child_id,
                    dst=target_id,
                    type="INHERITS",
                    metadata={
                        "path": file_path,
                        "resolved": target_id.startswith("symbol:"),
                    },
                )
            )

        for child_id, base_name, file_path in self._pending_implements_edges:
            target_id = self._symbol_resolver.resolve(base_name, preferred_file=file_path)
            if not target_id:
                target_id = self._symbol_resolver.resolve(
                    base_name.split(".")[-1], preferred_file=file_path
                )
            if not target_id:
                target_id = f"external_type:{base_name}"
                if target_id not in _seen_external:
                    _seen_external.add(target_id)
                    self._graph_nodes.append(
                        GraphNode(
                            node_id=target_id,
                            type="external_type",
                            name=base_name,
                            file=file_path,
                            metadata={"external": True},
                        )
                    )
            self._graph_edges.append(
                GraphEdge(
                    src=child_id,
                    dst=target_id,
                    type="IMPLEMENTS",
                    metadata={
                        "path": file_path,
                        "resolved": target_id.startswith("symbol:"),
                    },
                )
            )

        for owner_id, member_name, file_path in self._pending_compose_edges:
            target_id = self._symbol_resolver.resolve(member_name, preferred_file=file_path)
            if not target_id:
                target_id = self._symbol_resolver.resolve(
                    member_name.split(".")[-1], preferred_file=file_path
                )
            if not target_id:
                if member_name in _PRIMITIVE_TYPES:
                    continue
                target_id = f"external_type:{member_name}"
                if target_id not in _seen_external:
                    _seen_external.add(target_id)
                    self._graph_nodes.append(
                        GraphNode(
                            node_id=target_id,
                            type="external_type",
                            name=member_name,
                            file=file_path,
                            metadata={"external": True},
                        )
                    )
            self._graph_edges.append(
                GraphEdge(
                    src=owner_id,
                    dst=target_id,
                    type="COMPOSED_OF",
                    metadata={
                        "path": file_path,
                        "resolved": target_id.startswith("symbol:"),
                    },
                )
            )

        for metadata in self.files.values():
            if not metadata.references:
                continue
            file_node = f"file:{metadata.path}"
            for ref in metadata.references:
                target_id = self._symbol_resolver.resolve(ref, preferred_file=metadata.path)
                if not target_id:
                    target_id = self._symbol_resolver.resolve(
                        ref.split(".")[-1], preferred_file=metadata.path
                    )
                if not target_id:
                    continue
                self._graph_edges.append(
                    GraphEdge(
                        src=file_node,
                        dst=target_id,
                        type="REFERENCES",
                        metadata={"path": metadata.path, "resolved": True},
                    )
                )

    def _build_dependency_graph(self) -> None:
        """Build dependency graph between files."""
        for _file_path, metadata in self.files.items():
            for imp in metadata.imports:
                possible_paths = [
                    f"{imp.replace('.', '/')}.py",
                    f"{imp.replace('.', '/')}/__init__.py",
                ]

                for possible_path in possible_paths:
                    if possible_path in self.files:
                        metadata.dependencies.append(possible_path)
                        break

    async def find_relevant_files(
        self,
        query: str,
        max_files: int = 10,
        auto_reindex: bool = True,
    ) -> List[FileMetadata]:
        """Find files relevant to a query."""
        await self.ensure_indexed(auto_reindex=auto_reindex)

        results = []
        query_lower = query.lower()

        for file_path, metadata in self.files.items():
            relevance_score = 0

            if query_lower in file_path.lower():
                relevance_score += 10

            for symbol in metadata.symbols:
                if query_lower in symbol.name.lower():
                    relevance_score += 5
                if symbol.docstring and query_lower in symbol.docstring.lower():
                    relevance_score += 3

            for imp in metadata.imports:
                if query_lower in imp.lower():
                    relevance_score += 2

            if relevance_score > 0:
                results.append((relevance_score, metadata))

        results.sort(key=lambda x: x[0], reverse=True)
        return [metadata for _, metadata in results[:max_files]]

    def find_symbol(self, symbol_name: str) -> Optional[Symbol]:
        """Find a symbol by name."""
        for _key, symbol in self.symbols.items():
            if symbol.name == symbol_name:
                return symbol
        return None

    def get_file_context(self, file_path: str) -> Dict[str, Any]:
        """Get full context for a file including dependencies."""
        if file_path not in self.files:
            return {}

        metadata = self.files[file_path]

        return {
            "file": metadata,
            "symbols": metadata.symbols,
            "imports": metadata.imports,
            "dependencies": [self.files[dep] for dep in metadata.dependencies if dep in self.files],
            "dependents": self._find_dependents(file_path),
        }

    def _find_dependents(self, file_path: str) -> List[FileMetadata]:
        """Find files that depend on this file."""
        dependents = []
        for metadata in self.files.values():
            if file_path in metadata.dependencies:
                dependents.append(metadata)
        return dependents

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics including staleness information."""
        with self._staleness_lock:
            is_stale = self._is_stale
            changed_count = len(self._changed_files)
            last_indexed = self._last_indexed

        stats = {
            "total_files": len(self.files),
            "total_symbols": len(self.symbols),
            "total_lines": sum(f.lines for f in self.files.values()),
            "languages": {"python": len(self.files)},
            "embeddings_enabled": self.use_embeddings,
            "is_indexed": self._is_indexed,
            "is_stale": is_stale,
            "changed_files_count": changed_count,
            "last_indexed": last_indexed,
            "watcher_enabled": self._watcher_enabled,
            "watcher_running": self._observer is not None,
        }
        if self.use_embeddings and self.embedding_provider:
            stats["embedding_stats"] = asyncio.run(self.embedding_provider.get_stats())
        return stats

    def _initialize_embeddings(self, config: Optional[Dict[str, Any]]) -> None:
        """Initialize embedding provider."""
        try:
            from victor.verticals.contrib.coding.codebase.embeddings import (
                EmbeddingConfig,
                EmbeddingRegistry,
            )

            if not config:
                config = {}

            from victor.config.settings import get_project_paths, load_settings

            settings = load_settings()
            default_persist_dir = get_project_paths(self.root).embeddings_dir

            embedding_config = EmbeddingConfig(
                vector_store=config.get(
                    "vector_store",
                    getattr(settings, "codebase_vector_store", "lancedb"),
                ),
                embedding_model_type=config.get(
                    "embedding_model_type",
                    getattr(settings, "codebase_embedding_provider", "sentence-transformers"),
                ),
                embedding_model_name=config.get(
                    "embedding_model_name",
                    getattr(settings, "codebase_embedding_model", "BAAI/bge-small-en-v1.5"),
                ),
                persist_directory=config.get("persist_directory", str(default_persist_dir)),
                extra_config=config.get("extra_config", {}),
            )

            self.embedding_provider = EmbeddingRegistry.create(embedding_config)
            print(
                f"\u2713 Embeddings enabled: {embedding_config.embedding_model_name} + "
                f"{embedding_config.vector_store}"
            )
            print(f"  Storage: {embedding_config.persist_directory}")

        except ImportError as e:
            print(f"\u26a0\ufe0f  Warning: Embeddings not available: {e}")
            print("   Install with: pip install chromadb sentence-transformers")
            self.use_embeddings = False
            self.embedding_provider = None

    async def semantic_search(
        self,
        query: str,
        max_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        auto_reindex: bool = True,
        similarity_threshold: Optional[float] = None,
        expand_query: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        if not self.use_embeddings or not self.embedding_provider:
            raise ValueError("Embeddings not enabled. Initialize with use_embeddings=True")

        await self.ensure_indexed(auto_reindex=auto_reindex)

        if not self.embedding_provider._initialized:
            await self.embedding_provider.initialize()

        queries_to_search = [query]
        if expand_query:
            from victor.verticals.contrib.coding.codebase.query_expander import (
                expand_query as expand_fn,
            )

            queries_to_search = expand_fn(query, max_expansions=5)
            if len(queries_to_search) > 1:
                logger.debug(
                    f"Semantic search: expanded '{query}' to {len(queries_to_search)} queries"
                )

        all_results = []
        seen_docs = set()

        for search_query in queries_to_search:
            results = await self.embedding_provider.search_similar(
                query=search_query,
                limit=max_results * 2,
                filter_metadata=filter_metadata,
            )

            for result in results:
                doc_key = (result.file_path, result.line_number or 0)
                if doc_key not in seen_docs:
                    all_results.append(result)
                    seen_docs.add(doc_key)

        if similarity_threshold is not None:
            all_results = [r for r in all_results if r.score >= similarity_threshold]
            if not all_results:
                logger.debug(
                    f"Semantic search: threshold {similarity_threshold:.2f} filtered all results. "
                    "Consider lowering threshold or checking query."
                )

        all_results.sort(key=lambda r: r.score, reverse=True)
        all_results = all_results[:max_results]

        return [
            {
                "file_path": result.file_path,
                "symbol_name": result.symbol_name,
                "content": result.content,
                "score": result.score,
                "line_number": result.line_number,
                "end_line": result.metadata.get("end_line"),
                "metadata": result.metadata,
            }
            for result in all_results
        ]

    async def hybrid_search(
        self,
        query: str,
        graph_query: Optional[str] = None,
        document_filter: Optional[Dict[str, Any]] = None,
        time_range: Optional[tuple[datetime, datetime]] = None,
        top_k: int = 10,
        auto_reindex: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run provider-backed hybrid search when supported by the embedding store."""
        provider = await self._get_embedding_capability_provider(auto_reindex=auto_reindex)
        if not hasattr(provider, "hybrid_search"):
            raise NotImplementedError(
                f"Embedding provider {type(provider).__name__} does not support hybrid_search()"
            )
        return await provider.hybrid_search(
            query=query,
            graph_query=graph_query,
            document_filter=document_filter,
            time_range=time_range,
            top_k=top_k,
        )

    async def get_code_metrics(
        self,
        file_path: str,
        days: int = 30,
        auto_reindex: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return time-series style code metrics when supported by the provider."""
        provider = await self._get_embedding_capability_provider(auto_reindex=auto_reindex)
        if not hasattr(provider, "get_code_metrics"):
            raise NotImplementedError(
                f"Embedding provider {type(provider).__name__} does not support get_code_metrics()"
            )
        return await provider.get_code_metrics(file_path=file_path, days=days)

    async def find_callers(
        self,
        function_name: str,
        file_path: Optional[str] = None,
        edge_type: str = "CALLS",
        max_depth: int = 1,
        auto_reindex: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return reverse call graph matches when supported by the provider."""
        provider = await self._get_embedding_capability_provider(auto_reindex=auto_reindex)
        if not hasattr(provider, "find_callers"):
            raise NotImplementedError(
                f"Embedding provider {type(provider).__name__} does not support find_callers()"
            )
        return await provider.find_callers(
            function_name=function_name,
            file_path=file_path,
            edge_type=edge_type,
            max_depth=max_depth,
        )

    async def find_callees(
        self,
        function_name: str,
        file_path: Optional[str] = None,
        edge_type: str = "CALLS",
        max_depth: int = 1,
        auto_reindex: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return forward call graph matches when supported by the provider."""
        provider = await self._get_embedding_capability_provider(auto_reindex=auto_reindex)
        if not hasattr(provider, "find_callees"):
            raise NotImplementedError(
                f"Embedding provider {type(provider).__name__} does not support find_callees()"
            )
        return await provider.find_callees(
            function_name=function_name,
            file_path=file_path,
            edge_type=edge_type,
            max_depth=max_depth,
        )

    async def trace_execution_path(
        self,
        entry_function: str,
        file_path: Optional[str] = None,
        max_depth: int = 3,
        edge_type: str = "CALLS",
        auto_reindex: bool = True,
    ) -> Dict[str, Any]:
        """Trace a bounded execution path when supported by the provider."""
        provider = await self._get_embedding_capability_provider(auto_reindex=auto_reindex)
        if not hasattr(provider, "trace_execution_path"):
            raise NotImplementedError(
                f"Embedding provider {type(provider).__name__} does not support "
                "trace_execution_path()"
            )
        return await provider.trace_execution_path(
            entry_function=entry_function,
            file_path=file_path,
            max_depth=max_depth,
            edge_type=edge_type,
        )

    async def find_similar_bugs(
        self,
        bug_description: str,
        language: Optional[str] = None,
        top_k: int = 10,
        include_graph_context: bool = True,
        context_limit: int = 3,
        auto_reindex: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return provider-backed bug similarity matches when supported."""
        provider = await self._get_embedding_capability_provider(auto_reindex=auto_reindex)
        if not hasattr(provider, "find_similar_bugs"):
            raise NotImplementedError(
                f"Embedding provider {type(provider).__name__} does not support "
                "find_similar_bugs()"
            )
        return await provider.find_similar_bugs(
            bug_description=bug_description,
            language=language,
            top_k=top_k,
            include_graph_context=include_graph_context,
            context_limit=context_limit,
        )

    async def _get_embedding_capability_provider(
        self, auto_reindex: bool
    ) -> "BaseEmbeddingProvider":
        """Return an initialized embedding provider for advanced provider capabilities."""
        if not self.use_embeddings or not self.embedding_provider:
            raise ValueError("Embeddings not enabled. Initialize with use_embeddings=True")

        await self.ensure_indexed(auto_reindex=auto_reindex)
        if not self.embedding_provider._initialized:
            await self.embedding_provider.initialize()
        return self.embedding_provider

    def _build_symbol_context(self, symbol: Symbol) -> str:
        """Build context string for a symbol (for embedding)."""
        parts = [
            f"Symbol: {symbol.name}",
            f"Type: {symbol.type}",
            f"File: {symbol.file_path}",
        ]

        if symbol.signature:
            parts.append(f"Signature: {symbol.signature}")

        if symbol.docstring:
            parts.append(f"Documentation: {symbol.docstring}")

        return "\n".join(parts)
