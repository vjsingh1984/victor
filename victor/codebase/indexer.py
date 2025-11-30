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
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import ast
import asyncio

from pydantic import BaseModel

if TYPE_CHECKING:
    from victor.codebase.embeddings.base import BaseEmbeddingProvider


class Symbol(BaseModel):
    """Represents a code symbol (function, class, variable)."""

    name: str
    type: str  # function, class, variable, import
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    references: List[str] = []  # Files that reference this symbol


class FileMetadata(BaseModel):
    """Metadata about a source file."""

    path: str
    language: str
    symbols: List[Symbol] = []
    imports: List[str] = []
    dependencies: List[str] = []  # Files this file depends on
    last_modified: float
    size: int
    lines: int


class CodebaseIndex:
    """Indexes codebase for intelligent code understanding.

    This is the foundation for matching Claude Code's codebase awareness.

    Supports:
    - AST-based symbol extraction
    - Keyword search
    - Semantic search (with embeddings)
    - Dependency graph analysis
    """

    def __init__(
        self,
        root_path: str,
        ignore_patterns: Optional[List[str]] = None,
        use_embeddings: bool = False,
        embedding_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize codebase indexer.

        Args:
            root_path: Root directory of the codebase
            ignore_patterns: Patterns to ignore (e.g., ["venv/", "node_modules/"])
            use_embeddings: Whether to use semantic search with embeddings
            embedding_config: Configuration for embedding provider (optional)
        """
        self.root = Path(root_path).resolve()
        self.ignore_patterns = ignore_patterns or [
            "venv/",
            ".venv/",
            "env/",
            "node_modules/",
            ".git/",
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            ".mypy_cache/",
            "dist/",
            "build/",
        ]

        # Indexed data
        self.files: Dict[str, FileMetadata] = {}
        self.symbols: Dict[str, Symbol] = {}  # symbol_name -> Symbol
        self.symbol_index: Dict[str, List[str]] = {}  # file -> symbol names

        # Embedding support (optional)
        self.use_embeddings = use_embeddings
        self.embedding_provider: Optional["BaseEmbeddingProvider"] = None
        if use_embeddings:
            self._initialize_embeddings(embedding_config)

    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        rel_path = str(path.relative_to(self.root))
        return any(pattern in rel_path for pattern in self.ignore_patterns)

    async def index_codebase(self) -> None:
        """Index the entire codebase.

        This is the main entry point for building the index.
        Includes both AST indexing and optional semantic indexing with embeddings.
        """
        print(f"🔍 Indexing codebase at {self.root}")

        # Find all Python files
        python_files = [
            f for f in self.root.rglob("*.py") if f.is_file() and not self.should_ignore(f)
        ]

        print(f"Found {len(python_files)} Python files")

        # Index files in parallel (AST parsing)
        tasks = [self.index_file(file) for file in python_files]
        await asyncio.gather(*tasks)

        # Build dependency graph
        self._build_dependency_graph()

        print(f"✅ Indexed {len(self.files)} files, {len(self.symbols)} symbols")

        # Index with embeddings if enabled
        if self.use_embeddings and self.embedding_provider:
            await self._index_with_embeddings()

    async def _index_with_embeddings(self) -> None:
        """Index symbols with embeddings for semantic search."""
        if not self.embedding_provider:
            return

        print("\n🤖 Generating embeddings for semantic search...")

        # Initialize provider if needed
        if not self.embedding_provider._initialized:
            await self.embedding_provider.initialize()

        # Build documents for each symbol
        documents = []
        for file_path, metadata in self.files.items():
            for symbol in metadata.symbols:
                doc = {
                    "id": f"{file_path}:{symbol.name}",
                    "content": self._build_symbol_context(symbol),
                    "metadata": {
                        "file_path": file_path,
                        "symbol_name": symbol.name,
                        "symbol_type": symbol.type,
                        "line_number": symbol.line_number,
                    },
                }
                documents.append(doc)

        if documents:
            # Index with embedding provider
            await self.embedding_provider.index_documents(documents)
            print(f"✅ Generated embeddings for {len(documents)} symbols")
        else:
            print("⚠️  No symbols to index with embeddings")

    async def index_file(self, file_path: Path) -> None:
        """Index a single file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))

            # Extract metadata
            metadata = FileMetadata(
                path=str(file_path.relative_to(self.root)),
                language="python",
                last_modified=file_path.stat().st_mtime,
                size=file_path.stat().st_size,
                lines=content.count("\n") + 1,
            )

            # Extract symbols and imports
            visitor = SymbolVisitor(metadata)
            visitor.visit(tree)

            self.files[metadata.path] = metadata

            # Index symbols
            for symbol in metadata.symbols:
                self.symbols[f"{metadata.path}:{symbol.name}"] = symbol
                if metadata.path not in self.symbol_index:
                    self.symbol_index[metadata.path] = []
                self.symbol_index[metadata.path].append(symbol.name)

        except Exception as e:
            print(f"Error indexing {file_path}: {e}")

    def _build_dependency_graph(self) -> None:
        """Build dependency graph between files."""
        for _file_path, metadata in self.files.items():
            for imp in metadata.imports:
                # Try to resolve import to file path
                # This is a simplified version - full implementation would be more robust
                possible_paths = [
                    f"{imp.replace('.', '/')}.py",
                    f"{imp.replace('.', '/')}/__init__.py",
                ]

                for possible_path in possible_paths:
                    if possible_path in self.files:
                        metadata.dependencies.append(possible_path)
                        break

    async def find_relevant_files(self, query: str, max_files: int = 10) -> List[FileMetadata]:
        """Find files relevant to a query.

        This is a simplified version. Full implementation would use:
        - Semantic search with embeddings
        - BM25 ranking
        - Graph-based relevance

        Args:
            query: Search query
            max_files: Maximum number of files to return

        Returns:
            List of relevant file metadata
        """
        results = []

        # Simple keyword search for now
        query_lower = query.lower()

        for file_path, metadata in self.files.items():
            # Check if query matches:
            # 1. File name
            # 2. Symbol names
            # 3. Imports
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

        # Sort by relevance and return top N
        results.sort(key=lambda x: x[0], reverse=True)
        return [metadata for _, metadata in results[:max_files]]

    def find_symbol(self, symbol_name: str) -> Optional[Symbol]:
        """Find a symbol by name.

        Args:
            symbol_name: Name of symbol to find

        Returns:
            Symbol if found, None otherwise
        """
        # Search all files
        for _key, symbol in self.symbols.items():
            if symbol.name == symbol_name:
                return symbol
        return None

    def get_file_context(self, file_path: str) -> Dict[str, Any]:
        """Get full context for a file including dependencies.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file context
        """
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
        """Get index statistics."""
        stats = {
            "total_files": len(self.files),
            "total_symbols": len(self.symbols),
            "total_lines": sum(f.lines for f in self.files.values()),
            "languages": {"python": len(self.files)},
            "embeddings_enabled": self.use_embeddings,
        }
        if self.use_embeddings and self.embedding_provider:
            stats["embedding_stats"] = asyncio.run(self.embedding_provider.get_stats())
        return stats

    def _initialize_embeddings(self, config: Optional[Dict[str, Any]]) -> None:
        """Initialize embedding provider.

        Args:
            config: Embedding configuration dict
        """
        try:
            from victor.codebase.embeddings import EmbeddingConfig, EmbeddingRegistry

            # Create config with defaults
            if not config:
                config = {}

            embedding_config = EmbeddingConfig(
                vector_store=config.get("vector_store", "chromadb"),
                embedding_model_type=config.get("embedding_model_type", "sentence-transformers"),
                embedding_model_name=config.get("embedding_model_name", "all-mpnet-base-v2"),
                persist_directory=config.get(
                    "persist_directory", str(Path.home() / ".victor/embeddings")
                ),
                extra_config=config.get("extra_config", {}),
            )

            # Create embedding provider
            self.embedding_provider = EmbeddingRegistry.create(embedding_config)
            print(
                f"✓ Embeddings enabled: {embedding_config.embedding_model_name} + {embedding_config.vector_store}"
            )

        except ImportError as e:
            print(f"⚠️  Warning: Embeddings not available: {e}")
            print("   Install with: pip install chromadb sentence-transformers")
            self.use_embeddings = False
            self.embedding_provider = None

    async def semantic_search(
        self, query: str, max_results: int = 10, filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings.

        Args:
            query: Search query (natural language)
            max_results: Maximum number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of search results with file paths, symbols, and relevance scores
        """
        if not self.use_embeddings or not self.embedding_provider:
            raise ValueError("Embeddings not enabled. Initialize with use_embeddings=True")

        # Ensure provider is initialized
        if not self.embedding_provider._initialized:
            await self.embedding_provider.initialize()

        # Search using embedding provider
        results = await self.embedding_provider.search_similar(
            query=query, limit=max_results, filter_metadata=filter_metadata
        )

        # Convert to dict format
        return [
            {
                "file_path": result.file_path,
                "symbol_name": result.symbol_name,
                "content": result.content,
                "score": result.score,
                "line_number": result.line_number,
                "metadata": result.metadata,
            }
            for result in results
        ]

    def _build_symbol_context(self, symbol: Symbol) -> str:
        """Build context string for a symbol (for embedding).

        Args:
            symbol: Symbol to build context for

        Returns:
            Context string combining symbol information
        """
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


class SymbolVisitor(ast.NodeVisitor):
    """AST visitor to extract symbols from Python code."""

    def __init__(self, metadata: FileMetadata):
        self.metadata = metadata
        self.current_class: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        symbol = Symbol(
            name=node.name,
            type="class",
            file_path=self.metadata.path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
        )
        self.metadata.symbols.append(symbol)

        # Visit class methods
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        name = node.name
        if self.current_class:
            name = f"{self.current_class}.{name}"

        # Build signature
        args = [arg.arg for arg in node.args.args]
        signature = f"{node.name}({', '.join(args)})"

        symbol = Symbol(
            name=name,
            type="function",
            file_path=self.metadata.path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            signature=signature,
        )
        self.metadata.symbols.append(symbol)

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            self.metadata.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statement."""
        if node.module:
            self.metadata.imports.append(node.module)


# TODO: Future enhancements
# 1. Add semantic search with embeddings (ChromaDB, FAISS)
# 2. Add support for more languages (JavaScript, TypeScript, Go, etc.)
# 3. Add incremental indexing (only reindex changed files)
# 4. Add symbol reference tracking (who calls what)
# 5. Add type information extraction
# 6. Add test coverage mapping
# 7. Add documentation extraction
# 8. Add complexity metrics
