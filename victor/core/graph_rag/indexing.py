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

"""Graph indexing pipeline - G-Indexing stage of Graph RAG.

This module implements the first stage of the Graph RAG pipeline:
building a rich code graph with embeddings for semantic retrieval.

Based on GraphRAG methodology:
1. Build symbol graph from code structure
2. Build CCG (optional) for statement-level granularity
3. Generate embeddings for nodes
4. Cache subgraphs for efficient retrieval

ARCHITECTURAL SEAM (2026-04-29):
Language-specific edge detection should be handled by per-language
handlers implementing the LanguageEdgeHandler protocol. See
language_handlers.py for the protocol and handlers/ directory
for implementations.

MIGRATION STATUS:
- Core indexing pipeline: Uses language handlers via get_edge_handler()
- Fallback: Legacy _build_calls_edges() for unsupported languages
- Target: All edge detection moved to victor-coding language plugins

Future sessions: Add handlers for new languages in handlers/ directory,
then move them to victor-coding package as part of language plugins.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.storage.graph.protocol import GraphStoreProtocol

logger = logging.getLogger(__name__)


# Import at runtime for use in non-type-checked contexts
def _get_graph_types() -> tuple[type, type]:
    """Lazy import of graph types."""
    from victor.storage.graph.protocol import GraphEdge, GraphNode

    return GraphNode, GraphEdge


@dataclass
class GraphIndexStats:
    """Statistics from graph indexing operation.

    Attributes:
        files_processed: Number of files successfully processed
        files_skipped: Number of files skipped (too large, excluded, etc.)
        nodes_created: Number of graph nodes created
        edges_created: Number of graph edges created
        ccg_nodes_created: Number of CCG (statement-level) nodes
        ccg_edges_created: Number of CCG edges
        embeddings_generated: Number of embeddings generated
        subgraphs_cached: Number of subgraphs cached
        processing_time_seconds: Total processing time
        error_count: Number of errors encountered
        errors: List of error messages
    """

    files_processed: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    ccg_nodes_created: int = 0
    ccg_edges_created: int = 0
    embeddings_generated: int = 0
    subgraphs_cached: int = 0
    processing_time_seconds: float = 0.0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary.

        Returns:
            Dictionary representation of stats
        """
        return {
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "files_unchanged": self.files_unchanged,
            "files_deleted": self.files_deleted,
            "nodes_created": self.nodes_created,
            "edges_created": self.edges_created,
            "ccg_nodes_created": self.ccg_nodes_created,
            "ccg_edges_created": self.ccg_edges_created,
            "embeddings_generated": self.embeddings_generated,
            "subgraphs_cached": self.subgraphs_cached,
            "processing_time_seconds": self.processing_time_seconds,
            "error_count": self.error_count,
            "errors": self.errors[:10],  # Limit errors in output
        }


class GraphIndexingPipeline:
    """Pipeline for building and indexing code graphs.

    This class orchestrates the graph indexing process:
    1. Discover source files
    2. Extract symbols and build graph
    3. Build CCG if enabled
    4. Generate embeddings if enabled
    5. Cache subgraphs for fast retrieval

    Attributes:
        graph_store: Graph store for persisting nodes and edges
        config: Indexing configuration
    """

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
        config: Any,  # GraphIndexConfig
    ) -> None:
        """Initialize the indexing pipeline.

        Args:
            graph_store: Graph store for persistence
            config: GraphIndexConfig instance
        """
        self.graph_store = graph_store
        self.config = config
        self._ccg_builder = None
        self._files_to_process: Set[str] = set()

        if config.enable_ccg:
            try:
                from victor.core.indexing import CodeContextGraphBuilder

                self._ccg_builder = CodeContextGraphBuilder(graph_store)
            except ImportError:
                logger.warning("CCG builder not available, skipping CCG construction")

    async def index_repository(
        self,
        root_path: Optional[Path] = None,
    ) -> GraphIndexStats:
        """Index a repository into the graph store.

        Args:
            root_path: Optional root path override (uses config.root_path if None)

        Returns:
            GraphIndexStats with indexing results
        """
        root = (root_path or self.config.root_path).resolve()
        stats = GraphIndexStats()
        start_time = time.time()

        logger.info(f"Starting graph indexing for {root}")

        # Initialize graph store
        await self.graph_store.initialize()

        # Discover source files
        files = await self._discover_files(root)
        logger.info("Discovered %d indexable source files", len(files))

        planning_stats = await self._prepare_incremental_work(files)
        self._merge_stats(stats, planning_stats)

        if self.config.incremental:
            files = [file_path for file_path in files if str(file_path) in self._files_to_process]
            logger.info(
                "Incremental graph indexing plan for %s: %d changed, %d unchanged, %d deleted",
                root,
                len(files),
                planning_stats.files_unchanged,
                planning_stats.files_deleted,
            )

        # Process files in batches
        for i in range(0, len(files), self.config.chunk_size):
            batch = files[i : i + self.config.chunk_size]
            batch_stats = await self._process_batch(batch)
            self._merge_stats(stats, batch_stats)

        graph_changed = bool(
            stats.files_processed
            or stats.files_deleted
            or stats.nodes_created
            or stats.edges_created
            or stats.ccg_nodes_created
            or stats.ccg_edges_created
        )

        # Generate embeddings if enabled
        if self.config.enable_embeddings and graph_changed:
            embedding_stats = await self._generate_embeddings()
            self._merge_stats(stats, embedding_stats)

        # Cache subgraphs if enabled
        if self.config.enable_subgraph_cache and graph_changed:
            subgraph_stats = await self._cache_subgraphs()
            self._merge_stats(stats, subgraph_stats)

        stats.processing_time_seconds = time.time() - start_time

        logger.info(
            f"Indexing complete: {stats.files_processed} files, "
            f"{stats.nodes_created} nodes, {stats.edges_created} edges"
        )

        return stats

    async def _discover_files(self, root_path: Path) -> List[Path]:
        """Discover source files to index.

        Args:
            root_path: Root directory to search

        Returns:
            List of file paths to index
        """
        import fnmatch

        files: List[Path] = []
        exclude_patterns = self.config.exclude_patterns
        include_patterns = self.config.include_patterns

        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Check file size
            try:
                if file_path.stat().st_size > self.config.max_file_size_bytes:
                    continue
            except OSError:
                continue

            rel_path = file_path.relative_to(root_path)
            path_str = str(rel_path)

            # Check exclude patterns
            excluded = any(
                fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path_str, f"**/{pattern}")
                for pattern in exclude_patterns
            )
            if excluded:
                continue

            # Check include patterns (if specified)
            if include_patterns:
                included = any(
                    fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path_str, f"**/{pattern}")
                    for pattern in include_patterns
                )
                if not included:
                    continue

            language = self._detect_language(file_path)
            if not self._is_indexable_language(language):
                continue

            files.append(file_path)

        return sorted(files)

    async def _prepare_incremental_work(self, files: List[Path]) -> GraphIndexStats:
        """Prepare an incremental indexing plan based on file mtimes and deletions."""
        self._files_to_process = {str(file_path) for file_path in files}
        if not self.config.incremental:
            return GraphIndexStats()

        stats = GraphIndexStats()
        current_files = {str(file_path): file_path for file_path in files}
        file_mtimes = {
            path_str: file_path.stat().st_mtime for path_str, file_path in current_files.items()
        }

        stale_files = set(await self.graph_store.get_stale_files(file_mtimes))
        indexed_files = await self._get_indexed_files()
        deleted_files = sorted(indexed_files - set(current_files))

        for file_path in deleted_files:
            await self.graph_store.delete_by_file(file_path)
            stats.files_deleted += 1

        files_to_process: Set[str] = set()
        for file_path in sorted(stale_files):
            await self.graph_store.delete_by_file(file_path)
            files_to_process.add(file_path)

        stats.files_unchanged = max(0, len(current_files) - len(files_to_process))
        self._files_to_process = files_to_process
        return stats

    async def _get_indexed_files(self) -> Set[str]:
        """Return currently indexed files, preferring store-native mtime metadata."""
        get_indexed_files = getattr(self.graph_store, "get_indexed_files", None)
        if callable(get_indexed_files):
            try:
                indexed_files = await get_indexed_files()
                return {str(file_path) for file_path in indexed_files}
            except Exception as exc:
                logger.debug("Graph store get_indexed_files fallback: %s", exc)

        indexed_nodes = await self.graph_store.get_all_nodes()
        return {node.file for node in indexed_nodes if node.file}

    def _is_indexable_language(self, language: str) -> bool:
        """Return whether the built-in indexer can extract useful graph data for a language."""
        symbol_languages = {
            "python",
            "javascript",
            "typescript",
            "go",
            "rust",
            "java",
            "c",
            "cpp",
        }
        if language in symbol_languages:
            return True

        if not self.config.enable_ccg:
            return False

        try:
            from victor.core.indexing.ccg_builder import SUPPORTED_CCG_LANGUAGES

            return language in SUPPORTED_CCG_LANGUAGES
        except Exception:
            return False

    async def _process_batch(self, files: List[Path]) -> GraphIndexStats:
        """Process a batch of files.

        Args:
            files: List of file paths to process

        Returns:
            Batch processing stats
        """
        stats = GraphIndexStats()

        for file_path in files:
            try:
                file_stats = await self._process_file(file_path)
                self._merge_stats(stats, file_stats)
            except Exception as e:
                stats.error_count += 1
                stats.errors.append(f"{file_path}: {e}")
                logger.warning(f"Error processing {file_path}: {e}")

        return stats

    async def _process_file(self, file_path: Path) -> GraphIndexStats:
        """Process a single file.

        Args:
            file_path: Path to file

        Returns:
            File processing stats
        """
        GraphNode, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        stats = GraphIndexStats()
        stats.files_processed += 1

        # Detect language
        language = self._detect_language(file_path)
        if language == "unknown":
            stats.files_skipped += 1
            return stats

        # Extract symbols using tree-sitter
        symbol_nodes = await self._extract_symbols(file_path, language)
        stats.nodes_created += len(symbol_nodes)

        # Build symbol edges (CALLS, REFERENCES, CONTAINS)
        symbol_edges = await self._build_symbol_edges(symbol_nodes, file_path)
        stats.edges_created += len(symbol_edges)

        # Store symbols
        await self.graph_store.upsert_nodes(symbol_nodes)
        await self.graph_store.upsert_edges(symbol_edges)

        # Build CCG if enabled
        if self.config.enable_ccg and self._ccg_builder:
            ccg_nodes, ccg_edges = await self._ccg_builder.build_ccg_for_file(file_path, language)
            stats.ccg_nodes_created += len(ccg_nodes)
            stats.ccg_edges_created += len(ccg_edges)

            if ccg_nodes:
                await self.graph_store.upsert_nodes(ccg_nodes)
            if ccg_edges:
                await self.graph_store.upsert_edges(ccg_edges)

        # Update file mtime for staleness tracking
        mtime = file_path.stat().st_mtime
        await self.graph_store.update_file_mtime(str(file_path), mtime)

        return stats

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension.

        Args:
            file_path: Path to file

        Returns:
            Language identifier
        """
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "c_sharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".md": "markdown",
        }
        return ext_map.get(file_path.suffix.lower(), "unknown")

    async def _extract_symbols(
        self,
        file_path: Path,
        language: str,
    ) -> List[Any]:
        """Extract symbols from a source file.

        Args:
            file_path: Path to source file
            language: Programming language

        Returns:
            List of symbol nodes
        """
        GraphNode, _ = _get_graph_types()

        nodes: List[Any] = []

        try:
            source_code = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return nodes

        # Use tree-sitter to extract symbols
        try:
            # Try new API (tree-sitter 0.25+ with language packages)
            import tree_sitter as ts

            # Language module mapping (matches tree_sitter_manager)
            lang_modules = {
                "python": ("tree_sitter_python", "language"),
                "javascript": ("tree_sitter_javascript", "language"),
                "typescript": ("tree_sitter_typescript", "language_typescript"),
                "go": ("tree_sitter_go", "language"),
                "rust": ("tree_sitter_rust", "language"),
                "java": ("tree_sitter_java", "language"),
                "c": ("tree_sitter_c", "language"),
                "cpp": ("tree_sitter_cpp", "language"),
            }

            if language not in lang_modules:
                raise ValueError(f"Unsupported language: {language}")

            module_name, func_name = lang_modules[language]
            lang_module = __import__(module_name)
            lang_func = getattr(lang_module, func_name)
            lang_obj = lang_func()
            ts_language = (
                ts.Language(lang_obj) if not isinstance(lang_obj, ts.Language) else lang_obj
            )

            parser = ts.Parser(ts_language)
            tree = parser.parse(bytes(source_code, "utf-8"))

            # Extract function/class definitions
            nodes.extend(
                self._extract_definitions(tree.root_node, file_path, language, source_code)
            )

        except (ImportError, AttributeError, ValueError, Exception) as e:
            # Fall back to simple regex-based extraction
            logger.debug(f"Tree-sitter extraction failed for {language}: {e}, using fallback")
            nodes.extend(self._extract_symbols_fallback(source_code, file_path, language))

        return nodes

    def _extract_definitions(
        self,
        root_node: Any,
        file_path: Path,
        language: str,
        source_code: str,
    ) -> List[Any]:
        """Extract function/class definitions using tree-sitter.

        Args:
            root_node: Tree-sitter root node
            file_path: Path to source file
            language: Programming language
            source_code: Source code text

        Returns:
            List of symbol nodes
        """
        GraphNode, _ = _get_graph_types()

        nodes: List[Any] = []
        source_lines = source_code.splitlines()

        def extract(node: Any, parent_id: str | None = None, depth: int = 0) -> None:
            if not hasattr(node, "type"):
                return

            node_type = node.type

            # Check if this is a definition we care about
            if self._is_definition_type(node_type, language):
                # Get line numbers
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Extract name
                name = self._extract_name(node)

                # Get signature
                signature = self._extract_signature(node, source_lines, start_line, end_line)

                # Get docstring
                docstring = self._extract_docstring(
                    node, source_lines, start_line, end_line, language
                )

                # Generate node ID
                import hashlib

                node_id = hashlib.sha256(f"{file_path}:{name}:{start_line}".encode()).hexdigest()[
                    :16
                ]

                # Determine visibility
                visibility = self._determine_visibility(node, name, language)

                # Create node
                graph_node = GraphNode(
                    node_id=node_id,
                    type=self._map_node_type(node_type),
                    name=name,
                    file=str(file_path),
                    line=start_line,
                    end_line=end_line,
                    lang=language,
                    signature=signature,
                    docstring=docstring,
                    parent_id=parent_id,
                    visibility=visibility,
                    ast_kind=node_type,
                )
                nodes.append(graph_node)

                # DEBUG: Log node creation
                logger.debug(
                    f"Created node: {name} (type={node_type}, parent_id={parent_id[:8] + '...' if parent_id else None})"
                )

                # Recurse into children (for nested classes/functions)
                for child in node.children:
                    extract(child, graph_node.node_id, depth + 1)
            else:
                # Continue traversing
                for child in node.children:
                    extract(child, parent_id, depth + 1)

        extract(root_node)
        return nodes

    def _is_definition_type(self, node_type: str, language: str) -> bool:
        """Check if node type is a definition we should extract.

        Args:
            node_type: Tree-sitter node type
            language: Programming language

        Returns:
            True if this is a definition type
        """
        definition_types = {
            "python": {
                "function_definition",
                "class_definition",
                "async_function_definition",
                "decorated_definition",
            },
            "javascript": {
                "function_declaration",
                "function_expression",
                "class_declaration",
                "class_expression",
                "method_definition",
            },
            "typescript": {
                "function_declaration",
                "function_expression",
                "class_declaration",
                "class_expression",
                "method_definition",
                "interface_declaration",
                "type_alias_declaration",
            },
            "go": {
                "function_declaration",
                "type_declaration",
                "method_declaration",
            },
            "rust": {
                "function_item",
                "struct_item",
                "impl_item",
                "trait_item",
            },
            "java": {
                "method_declaration",
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
            },
        }

        return node_type in definition_types.get(language, set())

    def _map_node_type(self, tree_sitter_type: str) -> str:
        """Map tree-sitter node type to our graph node type.

        Args:
            tree_sitter_type: Tree-sitter node type

        Returns:
            Graph node type string
        """
        type_map = {
            # Python
            "function_definition": "function",
            "async_function_definition": "function",
            "class_definition": "class",
            "decorated_definition": "function",
            # JavaScript/TypeScript
            "function_declaration": "function",
            "function_expression": "function",
            "arrow_function": "function",
            "class_declaration": "class",
            "class_expression": "class",
            "method_definition": "method",
            "interface_declaration": "interface",
            "type_alias_declaration": "type_alias",
            # Go
            "go_function_declaration": "function",
            "go_method_declaration": "method",
            "go_type_declaration": "type",
            # Rust
            "function_item": "function",
            "struct_item": "struct",
            "impl_item": "impl",
            "trait_item": "trait",
            # Java
            "java_method_declaration": "method",
            "java_class_declaration": "class",
            "java_interface_declaration": "interface",
            "enum_declaration": "enum",
        }
        return type_map.get(tree_sitter_type, "unknown")

    def _extract_name(self, node: Any) -> str:
        """Extract name from a definition node.

        Args:
            node: Tree-sitter node

        Returns:
            Name string
        """

        # Find the first identifier child
        def find_identifier(n: Any) -> str | None:
            if hasattr(n, "type") and n.type == "identifier":
                text = getattr(n, "text", b"")
                return text.decode() if isinstance(text, bytes) else text
            for child in getattr(n, "children", []):
                result = find_identifier(child)
                if result:
                    return result
            return None

        name = find_identifier(node)
        return name or "<unnamed>"

    def _extract_signature(
        self,
        node: Any,
        source_lines: List[str],
        start_line: int,
        end_line: int,
    ) -> str | None:
        """Extract function/method signature.

        Args:
            node: Tree-sitter node
            source_lines: Source code lines
            start_line: Start line number
            end_line: End line number

        Returns:
            Signature string or None
        """
        if 0 < start_line <= len(source_lines):
            # Get the first line as signature preview
            return source_lines[start_line - 1].strip()[:200]
        return None

    def _extract_docstring(
        self,
        node: Any,
        source_lines: List[str],
        start_line: int,
        end_line: int,
        language: str,
    ) -> str | None:
        """Extract docstring from a definition node.

        Args:
            node: Tree-sitter node
            source_lines: Source code lines
            start_line: Start line number
            end_line: End line number
            language: Programming language

        Returns:
            Docstring or None
        """
        # Look for string literals in the function body
        # This is a simplified implementation
        return None

    def _determine_visibility(
        self,
        node: Any,
        name: str,
        language: str,
    ) -> str | None:
        """Determine visibility of a definition.

        Args:
            node: Tree-sitter node
            name: Definition name
            language: Programming language

        Returns:
            Visibility string or None
        """
        # Python: underscore prefix = private
        if language == "python":
            if name.startswith("_"):
                return "private"
            return "public"

        # JavaScript/TypeScript: check modifiers
        if language in {"javascript", "typescript"}:
            for child in getattr(node, "children", []):
                if hasattr(child, "type") and child.type == "property_identifier":
                    text = getattr(child, "text", b"")
                    if isinstance(text, bytes):
                        text = text.decode()
                    if text == "private":
                        return "private"
                    if text == "protected":
                        return "protected"
            return "public"

        # Java: check modifiers
        if language == "java":
            for child in getattr(node, "children", []):
                if hasattr(child, "type") and child.type in {"modifiers", "modifier", "annotation"}:
                    text = getattr(child, "text", b"")
                    if isinstance(text, bytes):
                        text = text.decode()
                    if "private" in text:
                        return "private"
                    if "protected" in text:
                        return "protected"
            return "public"

        return None

    def _extract_symbols_fallback(
        self,
        source_code: str,
        file_path: Path,
        language: str,
    ) -> List[Any]:
        """Fallback symbol extraction using simple regex patterns.

        Args:
            source_code: Source code text
            file_path: Path to source file
            language: Programming language

        Returns:
            List of symbol nodes
        """
        GraphNode, _ = _get_graph_types()
        import re
        import hashlib

        nodes: List[Any] = []

        if language == "python":
            # Simple regex for Python functions and classes
            func_pattern = r"^(def|async def)\s+(\w+)"
            class_pattern = r"^class\s+(\w+)"

            for i, line in enumerate(source_code.splitlines(), 1):
                func_match = re.match(func_pattern, line.strip())
                if func_match:
                    name = func_match.group(2)
                    node_id = hashlib.sha256(f"{file_path}:{name}:{i}".encode()).hexdigest()[:16]
                    nodes.append(
                        GraphNode(
                            node_id=node_id,
                            type="function",
                            name=name,
                            file=str(file_path),
                            line=i,
                            lang=language,
                            ast_kind="function_definition",
                        )
                    )

                class_match = re.match(class_pattern, line.strip())
                if class_match:
                    name = class_match.group(1)
                    node_id = hashlib.sha256(f"{file_path}:{name}:{i}".encode()).hexdigest()[:16]
                    nodes.append(
                        GraphNode(
                            node_id=node_id,
                            type="class",
                            name=name,
                            file=str(file_path),
                            line=i,
                            lang=language,
                            ast_kind="class_definition",
                        )
                    )

        return nodes

    async def _build_symbol_edges(
        self,
        nodes: List[Any],
        file_path: Path,
    ) -> List[Any]:
        """Build edges between symbol nodes.

        Args:
            nodes: Symbol nodes from the file
            file_path: Path to source file

        Returns:
            List of graph edges
        """
        _, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        edges: List[Any] = []

        # Build CONTAINS edges for parent-child relationships
        for node in nodes:
            if node.parent_id:
                edges.append(
                    GraphEdge(
                        src=node.parent_id,
                        dst=node.node_id,
                        type=EdgeType.CONTAINS,
                    )
                )

        # Build CALLS edges by analyzing source code for cross-references
        calls_edges = await self._build_calls_edges(nodes, file_path)
        edges.extend(calls_edges)

        return edges

    async def _build_calls_edges(
        self,
        nodes: List[Any],
        file_path: Path,
    ) -> List[Any]:
        """Build CALLS edges by analyzing function calls in source code.

        ARCHITECTURAL SEAM (2026-04-29):
        This method now uses the LanguageEdgeHandler protocol for
        per-language edge detection. Handlers are registered in
        handlers/ directory and discovered via get_edge_handler().

        MIGRATION PATH:
        1. New languages: Add handler in handlers/{lang}_edges.py
        2. Register handler in handlers/__init__.py
        3. Handler is auto-discovered here via get_edge_handler()
        4. Eventually move handlers to victor-coding language plugins

        Fallback: Legacy _build_calls_edges_legacy() used when no
        handler is registered for the language. To be removed once
        all supported languages have handlers.

        Args:
            nodes: Symbol nodes from the file
            file_path: Path to source file

        Returns:
            List of CALLS edges
        """
        _, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        edges: List[Any] = []

        # Build name -> node_id mapping for lookups
        name_to_ids: Dict[str, List[str]] = {}
        for node in nodes:
            if node.name:
                name_to_ids.setdefault(node.name, []).append(node.node_id)

        # Detect language
        language = self._detect_language(file_path)
        if language == "unknown":
            return edges

        # NEW PATH: Use language-specific edge handler (2026-04-29)
        try:
            from victor.core.graph_rag.language_handlers import get_edge_handler

            handler = get_edge_handler(language)
            if handler is not None:
                logger.debug(f"Using edge handler for language: {language}")
                edges.extend(
                    await self._build_edges_with_handler(handler, nodes, file_path, name_to_ids)
                )
                return edges
        except ImportError:
            logger.debug("Language handler system not available, using legacy")

        # FALLBACK: Legacy implementation for unsupported languages
        # TODO: Remove once all languages have handlers
        logger.debug(f"Using legacy CALLS edge detection for: {language}")
        return await self._build_calls_edges_legacy(nodes, file_path, name_to_ids)

    async def _build_edges_with_handler(
        self,
        handler: Any,
        nodes: List[Any],
        file_path: Path,
        name_to_ids: Dict[str, List[str]],
    ) -> List[Any]:
        """Build edges using a language-specific edge handler.

        Args:
            handler: LanguageEdgeHandler instance
            nodes: Symbol nodes from the file
            file_path: Path to source file
            name_to_ids: Name to node_id mapping

        Returns:
            List of CALLS edges
        """
        _, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        edges: List[Any] = []

        try:
            import tree_sitter as ts

            # Get language module
            language = self._detect_language(file_path)
            lang_modules = {
                "python": ("tree_sitter_python", "language"),
                "javascript": ("tree_sitter_javascript", "language"),
                "typescript": ("tree_sitter_typescript", "language_typescript"),
                "go": ("tree_sitter_go", "language"),
                "rust": ("tree_sitter_rust", "language"),
                "java": ("tree_sitter_java", "language"),
            }

            if language not in lang_modules:
                return edges

            module_name, func_name = lang_modules[language]
            lang_module = __import__(module_name)
            lang_func = getattr(lang_module, func_name)
            lang_obj = lang_func()
            ts_language = (
                ts.Language(lang_obj) if not isinstance(lang_obj, ts.Language) else lang_obj
            )

            # Parse source
            source_code = file_path.read_text(encoding="utf-8")
            parser = ts.Parser(ts_language)
            tree = parser.parse(bytes(source_code, "utf-8"))

            # Use handler to detect calls
            from victor.core.graph_rag.language_handlers import CallEdge

            result = await handler.detect_calls_edges(tree, source_code, file_path)

            # Create CALLS edges from detected calls
            for call in result.calls:
                caller_ids = name_to_ids.get(call.caller_name, [])
                callee_ids = name_to_ids.get(call.callee_name, [])

                for caller_id in caller_ids:
                    for callee_id in callee_ids:
                        if caller_id != callee_id:  # No self-loops
                            edges.append(
                                GraphEdge(
                                    src=caller_id,
                                    dst=callee_id,
                                    type=EdgeType.CALLS,
                                )
                            )

            logger.debug(f"Handler detected {len(result.calls)} calls, created {len(edges)} edges")

        except (ImportError, Exception) as e:
            logger.warning(f"Handler-based edge detection failed: {e}")

        return edges

    async def _build_calls_edges_legacy(
        self,
        nodes: List[Any],
        file_path: Path,
        name_to_ids: Dict[str, List[str]],
    ) -> List[Any]:
        """Legacy CALLS edge detection (to be removed).

        DEPRECATED: This method exists as a fallback for languages
        without dedicated edge handlers. Once all languages have
        handlers, this method should be removed.

        TODO: Remove after all languages migrate to handler pattern (target: 2026-Q2)

        Args:
            nodes: Symbol nodes from the file
            file_path: Path to source file
            name_to_ids: Name to node_id mapping

        Returns:
            List of CALLS edges
        """
        _, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        edges: List[Any] = []

        # Try tree-sitter for call analysis
        try:
            import tree_sitter as ts

            # Detect language
            language = self._detect_language(file_path)
            if language == "unknown":
                return edges

            # Get language module
            lang_modules = {
                "python": ("tree_sitter_python", "language"),
                "javascript": ("tree_sitter_javascript", "language"),
                "typescript": ("tree_sitter_typescript", "language_typescript"),
                "go": ("tree_sitter_go", "language"),
                "rust": ("tree_sitter_rust", "language"),
                "java": ("tree_sitter_java", "language"),
            }

            if language not in lang_modules:
                return edges

            module_name, func_name = lang_modules[language]
            lang_module = __import__(module_name)
            lang_func = getattr(lang_module, func_name)
            lang_obj = lang_func()
            ts_language = (
                ts.Language(lang_obj) if not isinstance(lang_obj, ts.Language) else lang_obj
            )

            # Parse source
            source_code = file_path.read_text(encoding="utf-8")
            parser = ts.Parser(ts_language)
            tree = parser.parse(bytes(source_code, "utf-8"))

            # Find function calls
            calls = self._extract_function_calls(tree.root_node, language)
            logger.debug(f"Legacy: Found {len(calls)} function calls in {file_path.name}")

            # Create CALLS edges
            for caller_name, callee_name in calls:
                # Find matching nodes
                caller_ids = name_to_ids.get(caller_name, [])
                callee_ids = name_to_ids.get(callee_name, [])

                # Create edges (one-to-many: a function might call multiple overloads)
                for caller_id in caller_ids:
                    for callee_id in callee_ids:
                        if caller_id != callee_id:  # No self-loops
                            edges.append(
                                GraphEdge(
                                    src=caller_id,
                                    dst=callee_id,
                                    type=EdgeType.CALLS,
                                )
                            )

        except (ImportError, Exception) as e:
            logger.debug(f"Legacy CALLS edge extraction failed: {e}")

        return edges

    def _extract_function_calls(
        self,
        root_node: Any,
        language: str,
    ) -> List[tuple[str, str]]:
        """Extract function calls from tree-sitter AST.

        Args:
            root_node: Tree-sitter root node
            language: Programming language

        Returns:
            List of (caller_name, callee_name) tuples
        """
        calls: List[tuple[str, str]] = []
        current_function: str | None = None

        def extract(node: Any, parent_function: str | None = None) -> None:
            nonlocal current_function

            if not hasattr(node, "type"):
                return

            node_type = node.type

            # Track current function scope
            if node_type in ("function_definition", "class_definition"):
                # Extract function/class name
                name = self._extract_name_from_node(node)
                if name:
                    # For methods inside classes, track parent
                    current_function = name
                    parent_function = name

            # Find call nodes (language-specific)
            if language == "python" and node_type == "call":
                callee_name = self._extract_callee_name(node, language)
                if callee_name and current_function:
                    calls.append((current_function, callee_name))

            elif language in ("javascript", "typescript") and node_type == "call_expression":
                callee_name = self._extract_callee_name(node, language)
                if callee_name and current_function:
                    calls.append((current_function, callee_name))

            # Recurse
            for child in node.children:
                extract(child, parent_function)

        extract(root_node)
        return calls

    def _extract_name_from_node(self, node: Any) -> str | None:
        """Extract name from a tree-sitter definition node.

        Args:
            node: Tree-sitter node

        Returns:
            Name string or None
        """
        # Look for "name" child
        for child in node.children:
            if hasattr(child, "type") and child.type == "identifier":
                if hasattr(child, "text"):
                    return child.text.decode("utf-8", errors="ignore")

            # Also check for named children
            if hasattr(child, "children"):
                for subchild in child.children:
                    if hasattr(subchild, "type") and subchild.type == "identifier":
                        if hasattr(subchild, "text"):
                            return subchild.text.decode("utf-8", errors="ignore")
        return None

    def _extract_callee_name(self, call_node: Any, language: str) -> str | None:
        """Extract callee name from a call node.

        Args:
            call_node: Tree-sitter call node
            language: Programming language

        Returns:
            Callee name or None
        """
        # For Python: call -> function (identifier)
        # For JS/TS: call_expression -> member_expression or identifier

        def find_identifier(node: Any) -> str | None:
            if hasattr(node, "type") and node.type == "identifier":
                if hasattr(node, "text"):
                    return node.text.decode("utf-8", errors="ignore")
            for child in node.children:
                result = find_identifier(child)
                if result:
                    return result
            return None

        # For Python, look at first child of call
        if language == "python" and call_node.children:
            func_node = call_node.children[0]
            if func_node.type == "identifier":
                return func_node.text.decode("utf-8", errors="ignore")
            elif func_node.type == "attribute":
                # For method calls like obj.method(), get the method name
                for child in func_node.children:
                    if child.type == "identifier":
                        return child.text.decode("utf-8", errors="ignore")

        # For JS/TS, find first identifier
        elif language in ("javascript", "typescript"):
            return find_identifier(call_node)

        return None

    async def _generate_embeddings(self) -> GraphIndexStats:
        """Generate embeddings for all nodes in the graph.

        Uses GraphAwareEmbedder to generate structure-aware embeddings
        that capture both semantic content and graph neighborhood.

        Returns:
            Stats for embedding generation
        """
        stats = GraphIndexStats()
        start_time = time.time()

        try:
            from victor.processing.graph_embeddings import GraphAwareEmbedder, GraphEmbeddingConfig
            from victor.storage.graph.protocol import GraphNode
            from victor.storage.embeddings.service import get_embedding_service

            # Initialize embedder
            config = GraphEmbeddingConfig(
                neighborhood_radius=self.config.embedding_neighborhood_radius or 2,
                include_edge_types=True,
                structural_weight=0.3,
                semantic_weight=0.7,
                max_neighbors=self.config.embedding_max_neighbors or 50,
            )
            embedder = GraphAwareEmbedder(config=config)

            # Get embedding service for direct text embeddings
            embedding_service = get_embedding_service()
            if embedding_service is None:
                logger.warning("EmbeddingService not available, skipping embedding generation")
                return stats

            # Get all nodes from graph store
            nodes = await self.graph_store.get_all_nodes()
            logger.info(f"Generating embeddings for {len(nodes)} nodes")

            # Process nodes in batches for efficiency
            batch_size = self.config.embedding_batch_size or 100
            all_embeddings: Dict[str, List[float]] = {}

            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]

                # Generate embeddings with graph context
                embeddings = await embedder.embed_batch(batch, self.graph_store)
                all_embeddings.update(embeddings)

                logger.debug(f"Generated embeddings for batch {i // batch_size + 1}")

            # Store embeddings and update nodes
            for node_id, embedding in all_embeddings.items():
                try:
                    # Store embedding in vector store (if available)
                    await self.graph_store.set_node_embedding(node_id, embedding)

                    # Update node metadata
                    await self.graph_store.update_node_metadata(
                        node_id,
                        {"embedding_ref": f"emb:{node_id}", "has_embedding": True},
                    )
                except Exception as e:
                    logger.warning(f"Failed to store embedding for {node_id}: {e}")
                    stats.error_count += 1
                    stats.errors.append(f"Embedding storage failed for {node_id}: {e}")

            stats.embeddings_generated = len(all_embeddings)
            logger.info(
                f"Generated {len(all_embeddings)} embeddings in {time.time() - start_time:.2f}s"
            )

        except ImportError as e:
            logger.warning(f"Graph embedding components not available: {e}")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            stats.error_count += 1
            stats.errors.append(f"Embedding generation: {e}")

        return stats

    async def _cache_subgraphs(self) -> GraphIndexStats:
        """Cache subgraphs for important nodes.

        Returns:
            Stats for subgraph caching
        """
        stats = GraphIndexStats()

        # TODO: Implement subgraph caching
        # This would:
        # 1. Identify important nodes (high centrality, recent changes, etc.)
        # 2. Compute subgraphs around these nodes
        # 3. Cache them in the graph store

        logger.info("Subgraph caching not yet implemented")
        return stats

    def _merge_stats(self, target: GraphIndexStats, source: GraphIndexStats) -> None:
        """Merge source stats into target stats.

        Args:
            target: Target stats to merge into
            source: Source stats to merge from
        """
        target.files_processed += source.files_processed
        target.files_skipped += source.files_skipped
        target.files_unchanged += source.files_unchanged
        target.files_deleted += source.files_deleted
        target.nodes_created += source.nodes_created
        target.edges_created += source.edges_created
        target.ccg_nodes_created += source.ccg_nodes_created
        target.ccg_edges_created += source.ccg_edges_created
        target.embeddings_generated += source.embeddings_generated
        target.subgraphs_cached += source.subgraphs_cached
        target.error_count += source.error_count
        target.errors.extend(source.errors)


__all__ = ["GraphIndexingPipeline", "GraphIndexStats"]
