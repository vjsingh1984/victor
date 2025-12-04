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
"""

import ast
import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

if TYPE_CHECKING:
    from victor.codebase.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


# Try to import watchdog for file watching
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object


# Module-level function for ProcessPoolExecutor (must be picklable)
def _parse_file_worker(args: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    """Parse a single Python file and extract metadata.

    This is a module-level function for use with ProcessPoolExecutor.
    Returns a dict with file metadata that can be converted to FileMetadata.

    Args:
        args: Tuple of (file_path_str, root_path_str)

    Returns:
        Dict with file metadata or None if parsing failed
    """
    file_path_str, root_path_str = args
    file_path = Path(file_path_str)
    root_path = Path(root_path_str)

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        # Extract metadata
        stat = file_path.stat()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        rel_path = str(file_path.relative_to(root_path))

        # Extract symbols and imports using a simple visitor
        symbols = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(
                    {
                        "name": node.name,
                        "type": "class",
                        "file_path": rel_path,
                        "line_number": node.lineno,
                        "docstring": ast.get_docstring(node),
                        "signature": None,
                    }
                )
            elif isinstance(node, ast.FunctionDef):
                # Check if it's a method (inside a class)
                name = node.name
                # Build signature
                args = [arg.arg for arg in node.args.args]
                signature = f"{node.name}({', '.join(args)})"
                symbols.append(
                    {
                        "name": name,
                        "type": "function",
                        "file_path": rel_path,
                        "line_number": node.lineno,
                        "docstring": ast.get_docstring(node),
                        "signature": signature,
                    }
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return {
            "path": rel_path,
            "language": "python",
            "last_modified": stat.st_mtime,
            "indexed_at": time.time(),
            "size": stat.st_size,
            "lines": content.count("\n") + 1,
            "content_hash": content_hash,
            "symbols": symbols,
            "imports": imports,
        }

    except Exception as e:
        logger.debug(f"Failed to parse {file_path}: {e}")
        return None


class CodebaseFileHandler(FileSystemEventHandler):
    """File system event handler for tracking codebase changes.

    Tracks file modifications, creations, and deletions to mark
    the index as stale when relevant files change.
    """

    def __init__(
        self,
        on_change: Callable[[str], None],
        file_patterns: List[str] = None,
        ignore_patterns: List[str] = None,
    ):
        """Initialize file handler.

        Args:
            on_change: Callback when a file changes (receives file path)
            file_patterns: File patterns to watch (e.g., ["*.py"])
            ignore_patterns: Patterns to ignore
        """
        super().__init__()
        self.on_change = on_change
        self.file_patterns = file_patterns or ["*.py"]
        self.ignore_patterns = ignore_patterns or [
            "__pycache__",
            ".git",
            "node_modules",
            ".pytest_cache",
            "venv",
            ".venv",
        ]
        self._debounce_lock = threading.Lock()
        self._pending_changes: Set[str] = set()
        self._debounce_timer: Optional[threading.Timer] = None
        self._debounce_delay = 0.5  # 500ms debounce

    def _should_process(self, path: str) -> bool:
        """Check if path should be processed."""
        path_obj = Path(path)

        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in str(path_obj):
                return False

        # Check file patterns
        for pattern in self.file_patterns:
            if path_obj.match(pattern):
                return True

        return False

    def _debounced_notify(self) -> None:
        """Notify of changes after debounce period."""
        with self._debounce_lock:
            changes = list(self._pending_changes)
            self._pending_changes.clear()
            self._debounce_timer = None

        for path in changes:
            try:
                self.on_change(path)
            except Exception as e:
                logger.warning(f"Error in file change callback: {e}")

    def _schedule_notification(self, path: str) -> None:
        """Schedule a debounced notification."""
        with self._debounce_lock:
            self._pending_changes.add(path)

            # Cancel existing timer
            if self._debounce_timer:
                self._debounce_timer.cancel()

            # Schedule new timer
            self._debounce_timer = threading.Timer(self._debounce_delay, self._debounced_notify)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def on_modified(self, event) -> None:
        """Handle file modification."""
        if not event.is_directory and self._should_process(event.src_path):
            self._schedule_notification(event.src_path)

    def on_created(self, event) -> None:
        """Handle file creation."""
        if not event.is_directory and self._should_process(event.src_path):
            self._schedule_notification(event.src_path)

    def on_deleted(self, event) -> None:
        """Handle file deletion."""
        if not event.is_directory and self._should_process(event.src_path):
            self._schedule_notification(event.src_path)


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
    last_modified: float  # File mtime when indexed
    indexed_at: float = 0.0  # When this file was indexed
    size: int
    lines: int
    content_hash: Optional[str] = None  # SHA256 hash for change detection


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
        enable_watcher: bool = True,
    ):
        """Initialize codebase indexer.

        Args:
            root_path: Root directory of the codebase
            ignore_patterns: Patterns to ignore (e.g., ["venv/", "node_modules/"])
            use_embeddings: Whether to use semantic search with embeddings
            embedding_config: Configuration for embedding provider (optional)
            enable_watcher: Whether to enable file watching for auto-staleness detection
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

        # Embedding support (optional)
        self.use_embeddings = use_embeddings
        self.embedding_provider: Optional["BaseEmbeddingProvider"] = None
        if use_embeddings:
            self._initialize_embeddings(embedding_config)

    @property
    def is_stale(self) -> bool:
        """Check if index is stale and needs refresh.

        Returns:
            True if files have changed since last indexing
        """
        with self._staleness_lock:
            return self._is_stale or not self._is_indexed

    @property
    def changed_files_count(self) -> int:
        """Get count of changed files since last index."""
        with self._staleness_lock:
            return len(self._changed_files)

    @property
    def _metadata_file(self) -> Path:
        """Path to persistent metadata file."""
        from victor.config.settings import get_project_paths

        return get_project_paths(self.root).index_metadata

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file contents.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash as hex string
        """
        try:
            content = file_path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]  # First 16 chars
        except Exception:
            return ""

    def _save_metadata(self) -> None:
        """Persist file metadata to disk for restart recovery."""
        try:
            self._metadata_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            metadata = {
                "last_indexed": self._last_indexed,
                "root_path": str(self.root),
                "files": {
                    path: {
                        "last_modified": meta.last_modified,
                        "indexed_at": meta.indexed_at,
                        "size": meta.size,
                        "lines": meta.lines,
                        "content_hash": meta.content_hash,
                        "symbol_count": len(meta.symbols),
                    }
                    for path, meta in self.files.items()
                },
            }

            self._metadata_file.write_text(json.dumps(metadata, indent=2))
            logger.debug(f"Saved index metadata to {self._metadata_file}")

        except Exception as e:
            logger.warning(f"Failed to save index metadata: {e}")

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load persisted metadata from disk.

        Returns:
            Metadata dict if available, None otherwise
        """
        try:
            if self._metadata_file.exists():
                content = self._metadata_file.read_text()
                return json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to load index metadata: {e}")
        return None

    def check_staleness_by_mtime(self) -> Tuple[bool, List[str], List[str]]:
        """Check if files have changed by comparing mtimes.

        This is the reliable method for startup - compares current file
        mtimes with stored mtimes from last indexing.

        Returns:
            Tuple of (is_stale, modified_files, deleted_files)
        """
        saved = self._load_metadata()
        if not saved:
            # No saved metadata, need full index
            return True, [], []

        stored_files = saved.get("files", {})
        modified_files = []
        deleted_files = []

        # Check each stored file
        for rel_path, file_info in stored_files.items():
            file_path = self.root / rel_path
            stored_mtime = file_info.get("last_modified", 0)

            if not file_path.exists():
                # File was deleted
                deleted_files.append(rel_path)
            else:
                current_mtime = file_path.stat().st_mtime
                if current_mtime > stored_mtime:
                    # File was modified
                    modified_files.append(rel_path)

        # Check for new files (files that exist but weren't indexed)
        for py_file in self.root.rglob("*.py"):
            if self.should_ignore(py_file):
                continue
            try:
                rel_path = str(py_file.relative_to(self.root))
                if rel_path not in stored_files:
                    # New file
                    modified_files.append(rel_path)
            except ValueError:
                pass

        is_stale = len(modified_files) > 0 or len(deleted_files) > 0
        return is_stale, modified_files, deleted_files

    async def startup_check(self, auto_reindex: bool = True) -> Dict[str, Any]:
        """Check index status at startup and reindex if needed.

        This should be called when Victor starts to ensure the index
        is up-to-date. It compares file mtimes with stored mtimes.

        Args:
            auto_reindex: If True, automatically reindex stale files

        Returns:
            Dictionary with status information
        """
        logger.info("Checking codebase index status at startup...")

        # Check if we have any saved metadata
        saved = self._load_metadata()
        if not saved:
            logger.info("No existing index found. Full indexing required.")
            if auto_reindex:
                await self.index_codebase(force=True)
            return {
                "status": "indexed" if auto_reindex else "needs_index",
                "action": "full_index" if auto_reindex else "none",
                "files_indexed": len(self.files) if auto_reindex else 0,
            }

        # Check for mtime-based staleness
        is_stale, modified, deleted = self.check_staleness_by_mtime()

        if not is_stale:
            logger.info("Index is up to date based on file mtimes.")
            # Restore in-memory state from saved metadata
            self._is_indexed = True
            self._last_indexed = saved.get("last_indexed")
            return {
                "status": "up_to_date",
                "action": "none",
                "files_in_index": len(saved.get("files", {})),
            }

        logger.info(f"Index is stale: {len(modified)} modified, {len(deleted)} deleted files")

        if auto_reindex:
            # Mark these files for reindexing
            with self._staleness_lock:
                self._changed_files.update(modified)
                self._is_stale = True

            # Use incremental or full reindex based on change count
            total_changes = len(modified) + len(deleted)
            if total_changes <= 10:
                result = await self.incremental_reindex()
                return {
                    "status": "reindexed",
                    "action": "incremental",
                    "files_modified": len(modified),
                    "files_deleted": len(deleted),
                    **result,
                }
            else:
                result = await self.reindex()
                return {
                    "status": "reindexed",
                    "action": "full",
                    "files_modified": len(modified),
                    "files_deleted": len(deleted),
                    **result,
                }

        return {
            "status": "stale",
            "action": "none",
            "files_modified": modified,
            "files_deleted": deleted,
        }

    def _on_file_changed(self, file_path: str) -> None:
        """Callback when a file changes.

        Args:
            file_path: Path to the changed file
        """
        with self._staleness_lock:
            self._is_stale = True
            try:
                rel_path = str(Path(file_path).relative_to(self.root))
                self._changed_files.add(rel_path)
                logger.debug(f"File changed, index marked stale: {rel_path}")
            except ValueError:
                # File outside root, ignore
                pass

    def start_watcher(self) -> bool:
        """Start file watcher for automatic staleness detection.

        Returns:
            True if watcher started successfully, False otherwise
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not installed. Install with: pip install watchdog")
            return False

        if self._observer is not None:
            logger.debug("File watcher already running")
            return True

        try:
            self._file_handler = CodebaseFileHandler(
                on_change=self._on_file_changed,
                file_patterns=["*.py"],
                ignore_patterns=self.ignore_patterns,
            )

            self._observer = Observer()
            self._observer.schedule(self._file_handler, str(self.root), recursive=True)
            self._observer.start()

            logger.info(f"Started file watcher for {self.root}")
            return True

        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            self._observer = None
            return False

    def stop_watcher(self) -> None:
        """Stop file watcher."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None
            self._file_handler = None
            logger.info("Stopped file watcher")

    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        rel_path = str(path.relative_to(self.root))
        return any(pattern in rel_path for pattern in self.ignore_patterns)

    async def index_codebase(self, force: bool = False) -> None:
        """Index the entire codebase.

        This is the main entry point for building the index.
        Includes both AST indexing and optional semantic indexing with embeddings.

        Uses ProcessPoolExecutor for parallel AST parsing on multi-core systems.

        Args:
            force: Force full reindex even if not stale
        """
        if not force and self._is_indexed and not self.is_stale:
            logger.debug("Index is up to date, skipping reindex")
            return

        print(f"🔍 Indexing codebase at {self.root}")

        # Find all Python files
        python_files = [
            f for f in self.root.rglob("*.py") if f.is_file() and not self.should_ignore(f)
        ]

        print(f"Found {len(python_files)} Python files")

        # Index files using parallel processing for CPU-bound AST parsing
        start_time = time.perf_counter()
        await self._parallel_index_files(python_files)
        parse_time = time.perf_counter() - start_time

        # Build dependency graph
        self._build_dependency_graph()

        # Update staleness tracking
        with self._staleness_lock:
            self._is_indexed = True
            self._is_stale = False
            self._changed_files.clear()
            self._last_indexed = time.time()

        print(
            f"✅ Indexed {len(self.files)} files, {len(self.symbols)} symbols "
            f"in {parse_time:.2f}s"
        )

        # Index with embeddings if enabled
        if self.use_embeddings and self.embedding_provider:
            await self._index_with_embeddings()

        # Start watcher if enabled
        if self._watcher_enabled and self._observer is None:
            self.start_watcher()

    async def _parallel_index_files(self, python_files: List[Path]) -> None:
        """Index files using parallel processing.

        Uses ProcessPoolExecutor for CPU-bound AST parsing on systems with
        multiple cores. Falls back to sequential processing for small file
        counts or single-core systems.

        Args:
            python_files: List of Python files to index
        """
        # Determine optimal worker count
        cpu_count = os.cpu_count() or 1
        num_workers = min(cpu_count, len(python_files), 8)  # Cap at 8 workers

        # Use parallel processing for larger codebases
        if len(python_files) >= 10 and num_workers > 1:
            logger.info(
                f"[Indexer] Using parallel AST parsing: "
                f"{len(python_files)} files, {num_workers} workers"
            )

            # Prepare arguments for workers
            root_str = str(self.root)
            work_items = [(str(f), root_str) for f in python_files]

            # Run in executor to not block the event loop
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._run_parallel_parsing(work_items, num_workers),
            )

            # Process results
            for result in results:
                if result is not None:
                    self._store_parsed_result(result)
        else:
            # Sequential processing for small codebases
            logger.debug(f"[Indexer] Using sequential parsing: {len(python_files)} files")
            for file_path in python_files:
                await self.index_file(file_path)

    def _run_parallel_parsing(
        self,
        work_items: List[Tuple[str, str]],
        num_workers: int,
    ) -> List[Optional[Dict[str, Any]]]:
        """Run parallel file parsing using ProcessPoolExecutor.

        Args:
            work_items: List of (file_path, root_path) tuples
            num_workers: Number of worker processes

        Returns:
            List of parsed results (some may be None for failed parses)
        """
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_parse_file_worker, item): item for item in work_items}
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)  # 30s timeout per file
                    results.append(result)
                except Exception as e:
                    item = futures[future]
                    logger.warning(f"Failed to parse {item[0]}: {e}")
                    results.append(None)
        return results

    def _store_parsed_result(self, result: Dict[str, Any]) -> None:
        """Store a parsed file result into the index.

        Args:
            result: Dict with file metadata from _parse_file_worker
        """
        # Convert symbol dicts to Symbol objects
        symbols = [
            Symbol(
                name=s["name"],
                type=s["type"],
                file_path=s["file_path"],
                line_number=s["line_number"],
                docstring=s.get("docstring"),
                signature=s.get("signature"),
            )
            for s in result.get("symbols", [])
        ]

        # Create FileMetadata
        metadata = FileMetadata(
            path=result["path"],
            language=result["language"],
            symbols=symbols,
            imports=result.get("imports", []),
            last_modified=result["last_modified"],
            indexed_at=result["indexed_at"],
            size=result["size"],
            lines=result["lines"],
            content_hash=result.get("content_hash"),
        )

        # Store in index
        self.files[metadata.path] = metadata

        # Index symbols
        for symbol in metadata.symbols:
            self.symbols[f"{metadata.path}:{symbol.name}"] = symbol
            if metadata.path not in self.symbol_index:
                self.symbol_index[metadata.path] = []
            self.symbol_index[metadata.path].append(symbol.name)

    async def reindex(self) -> Dict[str, Any]:
        """Force a full reindex of the codebase.

        This is the method to call from slash commands like /reindex.

        Returns:
            Dictionary with reindex statistics
        """
        start_time = time.time()

        # Clear existing data
        self.files.clear()
        self.symbols.clear()
        self.symbol_index.clear()

        # Force reindex
        await self.index_codebase(force=True)

        elapsed = time.time() - start_time

        # Persist metadata for startup recovery
        self._save_metadata()

        return {
            "success": True,
            "files_indexed": len(self.files),
            "symbols_indexed": len(self.symbols),
            "elapsed_seconds": round(elapsed, 2),
            "embeddings_enabled": self.use_embeddings,
        }

    async def incremental_reindex(self) -> Dict[str, Any]:
        """Incrementally reindex only changed files.

        More efficient than full reindex when few files have changed.
        Uses incremental embedding updates to only re-embed changed files.

        Returns:
            Dictionary with reindex statistics
        """
        with self._staleness_lock:
            changed = list(self._changed_files)
            changed_set = set(changed)

        if not changed:
            return {
                "success": True,
                "files_reindexed": 0,
                "message": "No files changed since last index",
            }

        start_time = time.time()
        reindexed_count = 0
        deleted_count = 0

        for rel_path in changed:
            file_path = self.root / rel_path

            if file_path.exists():
                # Remove old data for this file
                if rel_path in self.files:
                    del self.files[rel_path]
                    # Remove old symbols
                    for key in list(self.symbols.keys()):
                        if key.startswith(f"{rel_path}:"):
                            del self.symbols[key]
                    if rel_path in self.symbol_index:
                        del self.symbol_index[rel_path]

                # Reindex the file
                await self.index_file(file_path)
                reindexed_count += 1
            else:
                # File was deleted, just remove it
                if rel_path in self.files:
                    del self.files[rel_path]
                    for key in list(self.symbols.keys()):
                        if key.startswith(f"{rel_path}:"):
                            del self.symbols[key]
                    if rel_path in self.symbol_index:
                        del self.symbol_index[rel_path]
                deleted_count += 1

        # Rebuild dependency graph
        self._build_dependency_graph()

        # Update embeddings incrementally if enabled
        if self.use_embeddings and self.embedding_provider:
            await self._index_with_embeddings(
                incremental=True,
                changed_files=changed_set,
            )

        # Update staleness tracking
        with self._staleness_lock:
            self._is_stale = False
            self._changed_files.clear()
            self._last_indexed = time.time()

        elapsed = time.time() - start_time

        # Persist metadata for startup recovery
        self._save_metadata()

        return {
            "success": True,
            "files_reindexed": reindexed_count,
            "files_deleted": deleted_count,
            "elapsed_seconds": round(elapsed, 2),
        }

    async def ensure_indexed(self, auto_reindex: bool = True) -> None:
        """Ensure the index is up to date before searching.

        This implements lazy reindexing - reindex only when needed.

        Args:
            auto_reindex: If True, automatically reindex when stale
        """
        if not self._is_indexed:
            # Never indexed, do full index
            await self.index_codebase()
        elif self.is_stale and auto_reindex:
            # Index is stale, do incremental reindex if few files changed
            if self.changed_files_count <= 10:
                logger.info(
                    f"Index stale ({self.changed_files_count} files changed), "
                    "doing incremental reindex"
                )
                await self.incremental_reindex()
            else:
                logger.info(
                    f"Index stale ({self.changed_files_count} files changed), " "doing full reindex"
                )
                await self.reindex()

    async def _index_with_embeddings(
        self,
        use_advanced_chunking: bool = True,
        incremental: bool = False,
        changed_files: Optional[Set[str]] = None,
    ) -> None:
        """Index codebase with embeddings for semantic search.

        Uses the robust CodeChunker for AST-aware, hierarchical chunking.
        Supports incremental updates and content-based deduplication.

        Args:
            use_advanced_chunking: If True, use CodeChunker with body-aware chunking.
                                   If False, use simple symbol-based chunking (legacy).
            incremental: If True, only process changed files (requires changed_files).
            changed_files: Set of file paths that changed (for incremental updates).
        """
        if not self.embedding_provider:
            return

        print("\n🤖 Generating embeddings for semantic search...")

        # Initialize provider if needed
        if not self.embedding_provider._initialized:
            await self.embedding_provider.initialize()

        documents = []
        content_hashes: Set[str] = set()  # For deduplication
        duplicate_count = 0

        # Determine which files to process
        if incremental and changed_files:
            files_to_process = {fp: self.files[fp] for fp in changed_files if fp in self.files}
            print(f"🔄 Incremental mode: processing {len(files_to_process)} changed files")
        else:
            files_to_process = self.files

        if use_advanced_chunking:
            # Use robust CodeChunker for hierarchical, body-aware chunking
            try:
                from victor.codebase.chunker import (
                    CodeChunker,
                    ChunkConfig,
                    ChunkingStrategy,
                )

                config = ChunkConfig(
                    strategy=ChunkingStrategy.BODY_AWARE,
                    max_chunk_tokens=512,  # ~2048 chars
                    overlap_tokens=64,  # ~256 chars overlap
                    large_symbol_threshold=30,  # Chunk functions >30 lines
                    include_file_summary=True,
                    include_class_summary=True,
                )

                chunker = CodeChunker(config)
                total_chunks = 0

                for file_path in files_to_process.keys():
                    abs_path = self.root / file_path
                    if abs_path.exists():
                        chunks = chunker.chunk_file(abs_path, file_path)
                        for chunk in chunks:
                            doc = chunk.to_document()
                            # Content-based deduplication
                            content_hash = hashlib.sha256(doc["content"].encode()).hexdigest()[:16]
                            if content_hash not in content_hashes:
                                content_hashes.add(content_hash)
                                # Add hash to metadata for future incremental updates
                                doc["metadata"]["content_hash"] = content_hash
                                documents.append(doc)
                            else:
                                duplicate_count += 1
                        total_chunks += len(chunks)

                print("📊 Chunking strategy: BODY_AWARE (hierarchical)")
                print(f"📁 Files processed: {len(files_to_process)}")
                print(f"🧩 Chunks created: {total_chunks}")
                if duplicate_count > 0:
                    print(f"🔁 Duplicates removed: {duplicate_count}")

            except ImportError as e:
                logger.warning(f"CodeChunker not available, falling back to simple chunking: {e}")
                use_advanced_chunking = False

        if not use_advanced_chunking:
            # Fallback: Simple symbol-based chunking (legacy)
            print("📊 Chunking strategy: SYMBOL_ONLY (legacy)")
            for file_path, metadata in files_to_process.items():
                for symbol in metadata.symbols:
                    content = self._build_symbol_context(symbol)
                    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

                    # Content-based deduplication
                    if content_hash in content_hashes:
                        duplicate_count += 1
                        continue

                    content_hashes.add(content_hash)
                    doc = {
                        "id": f"{file_path}:{symbol.name}",
                        "content": content,
                        "metadata": {
                            "file_path": file_path,
                            "symbol_name": symbol.name,
                            "symbol_type": symbol.type,
                            "line_number": symbol.line_number,
                            "content_hash": content_hash,
                        },
                    }
                    documents.append(doc)

            if duplicate_count > 0:
                print(f"🔁 Duplicates removed: {duplicate_count}")

        if documents:
            if incremental and changed_files:
                # For incremental updates, delete old documents first then add new
                for file_path in changed_files:
                    await self.embedding_provider.delete_by_file(file_path)
                await self.embedding_provider.index_documents(documents)
                print(f"✅ Updated embeddings for {len(documents)} chunks")
            else:
                # Full rebuild: clear and add all
                await self.embedding_provider.clear_index()
                await self.embedding_provider.index_documents(documents)
                print(f"✅ Generated embeddings for {len(documents)} chunks")
        else:
            print("⚠️  No content to index with embeddings")

    async def index_file(self, file_path: Path) -> None:
        """Index a single file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))

            # Extract metadata with timestamps and hash
            stat = file_path.stat()
            content_hash = self._compute_file_hash(file_path)

            metadata = FileMetadata(
                path=str(file_path.relative_to(self.root)),
                language="python",
                last_modified=stat.st_mtime,
                indexed_at=time.time(),  # When we indexed it
                size=stat.st_size,
                lines=content.count("\n") + 1,
                content_hash=content_hash,
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

    async def find_relevant_files(
        self,
        query: str,
        max_files: int = 10,
        auto_reindex: bool = True,
    ) -> List[FileMetadata]:
        """Find files relevant to a query.

        Automatically reindexes if the index is stale (lazy reindexing).

        Args:
            query: Search query
            max_files: Maximum number of files to return
            auto_reindex: If True, automatically reindex when stale

        Returns:
            List of relevant file metadata
        """
        # Lazy reindexing - ensure index is up to date
        await self.ensure_indexed(auto_reindex=auto_reindex)

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
        """Initialize embedding provider.

        Embeddings are stored in {rootrepo}/.embeddings/ directory by default.
        This keeps all index data co-located with the repository.

        Args:
            config: Embedding configuration dict
        """
        try:
            from victor.codebase.embeddings import EmbeddingConfig, EmbeddingRegistry

            # Create config with defaults
            if not config:
                config = {}

            # Default persist directory is {rootrepo}/.victor/embeddings/
            # This keeps embeddings with the project for isolation
            from victor.config.settings import get_project_paths

            default_persist_dir = get_project_paths(self.root).embeddings_dir

            embedding_config = EmbeddingConfig(
                vector_store=config.get("vector_store", "chromadb"),
                embedding_model_type=config.get("embedding_model_type", "sentence-transformers"),
                embedding_model_name=config.get("embedding_model_name", "all-mpnet-base-v2"),
                persist_directory=config.get("persist_directory", str(default_persist_dir)),
                extra_config=config.get("extra_config", {}),
            )

            # Create embedding provider
            self.embedding_provider = EmbeddingRegistry.create(embedding_config)
            print(
                f"✓ Embeddings enabled: {embedding_config.embedding_model_name} + "
                f"{embedding_config.vector_store}"
            )
            print(f"  Storage: {embedding_config.persist_directory}")

        except ImportError as e:
            print(f"⚠️  Warning: Embeddings not available: {e}")
            print("   Install with: pip install chromadb sentence-transformers")
            self.use_embeddings = False
            self.embedding_provider = None

    async def semantic_search(
        self,
        query: str,
        max_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        auto_reindex: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings.

        Automatically reindexes if the index is stale (lazy reindexing).

        Args:
            query: Search query (natural language)
            max_results: Maximum number of results
            filter_metadata: Optional metadata filters
            auto_reindex: If True, automatically reindex when stale

        Returns:
            List of search results with file paths, symbols, and relevance scores
        """
        if not self.use_embeddings or not self.embedding_provider:
            raise ValueError("Embeddings not enabled. Initialize with use_embeddings=True")

        # Lazy reindexing - ensure index is up to date
        await self.ensure_indexed(auto_reindex=auto_reindex)

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
# [DONE] 1. Add semantic search with embeddings (ChromaDB, LanceDB)
# 2. Add support for more languages (JavaScript, TypeScript, Go, etc.)
# [DONE] 3. Add incremental indexing (only reindex changed files)
# [DONE] 4. Add file watching for automatic staleness detection
# 5. Add symbol reference tracking (who calls what)
# 6. Add type information extraction
# 7. Add test coverage mapping
# 8. Add documentation extraction
# 9. Add complexity metrics
