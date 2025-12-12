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
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from tree_sitter import Query

from victor.codebase.graph.protocol import GraphEdge, GraphNode
from victor.codebase.graph.registry import create_graph_store
from victor.codebase.graph.sqlite_store import SqliteGraphStore
from victor.codebase.symbol_resolver import SymbolResolver

if TYPE_CHECKING:
    from victor.codebase.embeddings.base import BaseEmbeddingProvider
    from victor.codebase.graph.protocol import GraphStoreProtocol

logger = logging.getLogger(__name__)

REFERENCE_QUERIES: Dict[str, str] = {
    "python": """
        (call function: (identifier) @name)
        (call function: (attribute attribute: (identifier) @name))
        (attribute object: (_) attribute: (identifier) @name)
        (identifier) @name
    """,
    "javascript": """
        (call_expression function: (identifier) @name)
        (call_expression function: (member_expression property: (property_identifier) @name))
        (member_expression property: (property_identifier) @name)
        (new_expression constructor: (identifier) @name)
        (identifier) @name
    """,
    "typescript": """
        (call_expression function: (identifier) @name)
        (call_expression function: (member_expression property: (property_identifier) @name))
        (member_expression property: (property_identifier) @name)
        (new_expression constructor: (identifier) @name)
        (identifier) @name
    """,
    "java": """
        (method_invocation name: (identifier) @name)
        (method_invocation object: (identifier) @name)
        (field_access field: (identifier) @name)
    """,
    "go": """
        (call_expression function: (identifier) @name)
        (call_expression function: (selector_expression field: (field_identifier) @name))
        (selector_expression field: (field_identifier) @name)
        (identifier) @name
    """,
}

# Map file extensions to tree-sitter language ids
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".json": "config-json",
    ".yaml": "config-yaml",
    ".yml": "config-yaml",
    ".toml": "config-toml",
    ".ini": "config-ini",
    ".properties": "config-properties",
    ".conf": "config-hocon",
    ".hocon": "config-hocon",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
}

# Tree-sitter symbol queries per language for lightweight multi-language graph capture.
SYMBOL_QUERIES: Dict[str, List[tuple[str, str]]] = {
    "javascript": [
        ("class", "(class_declaration name: (identifier) @name)"),
        ("function", "(function_declaration name: (identifier) @name)"),
        ("function", "(method_definition name: (property_identifier) @name)"),
        ("function", "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function)))"),
        ("function", "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression)))"),
        ("function", "(assignment_expression left: (identifier) @name right: (arrow_function))"),
    ],
    "typescript": [
        ("class", "(class_declaration name: (identifier) @name)"),
        ("function", "(function_declaration name: (identifier) @name)"),
        ("function", "(method_signature name: (property_identifier) @name)"),
        ("function", "(method_definition name: (property_identifier) @name)"),
        ("function", "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function)))"),
        ("function", "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression)))"),
        ("function", "(assignment_expression left: (identifier) @name right: (arrow_function))"),
    ],
    "go": [
        ("function", "(function_declaration name: (identifier) @name)"),
        ("function", "(method_declaration name: (field_identifier) @name)"),
        ("class", "(type_declaration (type_spec name: (type_identifier) @name))"),
    ],
    "java": [
        ("class", "(class_declaration name: (identifier) @name)"),
        ("class", "(interface_declaration name: (identifier) @name)"),
        ("function", "(method_declaration name: (identifier) @name)"),
    ],
    "cpp": [
        ("class", "(class_specifier name: (type_identifier) @name)"),
        ("function", "(function_definition declarator: (function_declarator declarator: (identifier) @name))"),
        ("function", "(function_definition declarator: (function_declarator declarator: (field_identifier) @name))"),
    ],
}

INHERITS_QUERIES: Dict[str, str] = {
    "javascript": """
        (class_declaration
            name: (identifier) @child
            heritage: (class_heritage (identifier) @base))
    """,
    "typescript": """
        (class_declaration
            name: (identifier) @child
            heritage: (class_heritage (identifier) @base))
    """,
    "java": """
        (class_declaration
            name: (identifier) @child
            super_classes: (superclass (type_identifier) @base))
    """,
    "cpp": """
        (class_specifier
            name: (type_identifier) @child
            (base_class_clause (base_class (type_identifier) @base))
        )
    """,
}

IMPLEMENTS_QUERIES: Dict[str, str] = {
    "typescript": """
        (class_declaration
            name: (identifier) @child
            (heritage_clause (identifier) @interface))
    """,
    "java": """
        (class_declaration
            name: (identifier) @child
            interfaces: (super_interfaces (type_list (type_identifier) @interface)))
        (interface_declaration
            name: (identifier) @child
            interfaces: (super_interfaces (type_list (type_identifier) @interface)))
    """,
    "cpp": """
        (class_specifier
            name: (type_identifier) @child
            (base_class_clause (base_class (type_identifier) @base))
        )
    """,
}

COMPOSITION_QUERIES: Dict[str, str] = {
    "javascript": """
        (class_declaration
            name: (identifier) @owner
            body: (class_body
                (method_definition
                    body: (statement_block
                        (expression_statement
                            (assignment_expression
                                left: (member_expression object: (this) property: (property_identifier))
                                right: (new_expression constructor: (identifier) @type)))))))
    """,
    "typescript": """
        (class_declaration
            name: (identifier) @owner
            body: (class_body
                (field_definition
                    type: (type_annotation (type_identifier) @type))
                (public_field_definition
                    type: (type_annotation (type_identifier) @type))
                (method_definition
                    body: (statement_block
                        (expression_statement
                            (assignment_expression
                                left: (member_expression object: (this) property: (property_identifier))
                                right: (new_expression constructor: (identifier) @type)))))))
    """,
    "go": """
        (type_declaration
            (type_spec
                name: (type_identifier) @owner
                type: (struct_type
                    (field_declaration
                        type: (type_identifier) @type))))
    """,
    "java": """
        (class_declaration
            name: (identifier) @owner
            body: (class_body
                (field_declaration
                    type: (type_identifier) @type)))
    """,
    "cpp": """
        (class_specifier
            name: (type_identifier) @owner
            body: (field_declaration_list
                (field_declaration
                    type: (type_identifier) @type)))
    """,
}

# Tree-sitter call queries (callee only) for multi-language call/reference edges.
CALL_QUERIES: Dict[str, str] = {
    "javascript": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (member_expression property: (property_identifier) @callee))
        (call_expression function: (subscript_expression index: (property_identifier) @callee))
        (new_expression constructor: (identifier) @callee)
    """,
    "typescript": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (member_expression property: (property_identifier) @callee))
        (call_expression function: (subscript_expression index: (property_identifier) @callee))
        (new_expression constructor: (identifier) @callee)
    """,
    "go": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (selector_expression field: (field_identifier) @callee))
        (type_conversion_expression type: (type_identifier) @callee)
    """,
    "java": """
        (method_invocation name: (identifier) @callee)
        (object_creation_expression type: (type_identifier) @callee)
        (super_method_invocation name: (identifier) @callee)
    """,
    "cpp": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (field_expression field: (field_identifier) @callee))
        (new_expression type: (type_identifier) @callee)
    """,
}

# Mapping of function/method node types to name field for caller resolution.
ENCLOSING_NAME_FIELDS: Dict[str, List[tuple[str, str]]] = {
    "javascript": [
        ("function_declaration", "name"),
        ("method_definition", "name"),
        ("class_declaration", "name"),  # used for Class.method combination
    ],
    "typescript": [
        ("function_declaration", "name"),
        ("method_definition", "name"),
        ("method_signature", "name"),
        ("class_declaration", "name"),
    ],
    "go": [
        ("function_declaration", "name"),
        ("method_declaration", "name"),
    ],
    "java": [
        ("method_declaration", "name"),
        ("class_declaration", "name"),
        ("interface_declaration", "name"),
    ],
    "cpp": [
        ("function_definition", "declarator"),
        ("class_specifier", "name"),
    ],
}


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


from pydantic import BaseModel, Field


class Symbol(BaseModel):
    """Represents a code symbol (function, class, variable)."""

    name: str
    type: str  # function, class, variable, import
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    references: List[str] = Field(default_factory=list)  # Files that reference this symbol
    base_classes: List[str] = Field(default_factory=list)  # inheritance targets
    composition: List[tuple[str, str]] = Field(default_factory=list)  # (owner, member) for has-a


class FileMetadata(BaseModel):
    """Metadata about a source file."""

    path: str
    language: str
    symbols: List[Symbol] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)  # Files this file depends on
    call_edges: List[tuple[str, str]] = Field(default_factory=list)  # (caller, callee) pairs
    inherit_edges: List[tuple[str, str]] = Field(default_factory=list)  # (child, base)
    implements_edges: List[tuple[str, str]] = Field(default_factory=list)  # (child, interface)
    compose_edges: List[tuple[str, str]] = Field(default_factory=list)  # (owner, member)
    references: List[str] = Field(default_factory=list)  # Identifier references (tree-sitter/AST)
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

    def __init__(
        self,
        root_path: str,
        ignore_patterns: Optional[List[str]] = None,
        use_embeddings: bool = False,
        embedding_config: Optional[Dict[str, Any]] = None,
        enable_watcher: bool = True,
        graph_store: Optional["GraphStoreProtocol"] = None,
        graph_store_name: Optional[str] = None,
        graph_path: Optional[Path] = None,
    ):
        """Initialize codebase indexer.

        Args:
            root_path: Root directory of the codebase
            ignore_patterns: Patterns to ignore (e.g., ["venv/", "node_modules/"])
            use_embeddings: Whether to use semantic search with embeddings
            embedding_config: Configuration for embedding provider (optional)
            enable_watcher: Whether to enable file watching for auto-staleness detection
            graph_store: Optional graph store for symbol relationships. If None,
                a per-repo store is created under .victor/graph/graph.db.
            graph_store_name: Optional graph backend name (currently only "sqlite")
            graph_path: Optional explicit graph store path
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

        # Callbacks for change notifications (e.g., SymbolStore)
        self._change_callbacks: List[Callable[[str], None]] = []

        # Graph store (per-repo, embedded)
        if graph_store is None:
            backend = graph_store_name or os.getenv("VICTOR_GRAPH_STORE", "sqlite")
            if graph_path is None:
                from victor.config.settings import get_project_paths

                graph_path = get_project_paths(self.root).project_victor_dir / "graph" / "graph.db"
            self.graph_store: Optional["GraphStoreProtocol"] = create_graph_store(
                backend, Path(graph_path)
            )
        else:
            self.graph_store = graph_store
        self._graph_nodes: List[GraphNode] = []
        self._graph_edges: List[GraphEdge] = []
        self._pending_call_edges: List[tuple[str, str, str]] = []  # caller_id, callee_name, file
        self._pending_inherit_edges: List[tuple[str, str, str]] = []  # child_id, base_name, file
        self._pending_implements_edges: List[tuple[str, str, str]] = []  # child_id, interface, file
        self._pending_compose_edges: List[tuple[str, str, str]] = []  # owner_id, member_type, file
        self._symbol_resolver = SymbolResolver()

        # Embedding support (optional)
        self.use_embeddings = use_embeddings
        self.embedding_provider: Optional["BaseEmbeddingProvider"] = None
        if use_embeddings:
            self._initialize_embeddings(embedding_config)

    def _reset_graph_buffers(self) -> None:
        self._graph_nodes = []
        self._graph_edges = []
        self._pending_call_edges = []
        self._pending_inherit_edges = []
        self._pending_implements_edges = []
        self._pending_compose_edges = []
        self._symbol_resolver = SymbolResolver()

    def _detect_language(self, file_path: Path, default: str = "python") -> str:
        """Detect language from extension for tree-sitter queries."""
        return EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), default)

    def _is_config_language(self, language: str) -> bool:
        """Return True if the language is a config/metadata file."""
        return language.startswith("config")

    def _extract_references(
        self, file_path: Path, language: str, fallback_calls: List[str], imports: List[str]
    ) -> List[str]:
        """Extract identifier references using tree-sitter when available."""
        refs: Set[str] = set(fallback_calls) | set(imports)
        query_src = REFERENCE_QUERIES.get(language)
        if not query_src:
            return list(refs)
        try:
            from victor.codebase.tree_sitter_manager import get_parser
        except Exception:
            return list(refs)

        try:
            parser = get_parser(language)
        except Exception:
            return list(refs)

        if parser is None:
            return list(refs)

        try:
            content = file_path.read_bytes()
            tree = parser.parse(content)
            query = Query(parser.language, query_src)
            captures = query.captures(tree.root_node)
            for node, _capture_name in captures:
                text = node.text.decode("utf-8", errors="ignore")
                if text:
                    refs.add(text)
        except Exception:
            # Graceful degradation; fall back to existing refs
            pass

        # Regex fallback to catch simple identifier usage when tree-sitter misses
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
                    import yaml  # type: ignore

                    data = yaml.safe_load(content)
                    _walk(data)
                except Exception:
                    pass
        except Exception:
            # Fall back to regex below
            pass

        if not keys:
            # Regex fallback for generic key/value formats
            for match in re.finditer(
                r'^[\s"\']*([A-Za-z0-9_.\-]+)\s*[:=]', content, flags=re.MULTILINE
            ):
                key = match.group(1)
                line_no = content.count("\n", 0, match.start()) + 1
                keys.setdefault(key, line_no)

        return [(k, v) for k, v in keys.items()]

    def _extract_symbols_with_tree_sitter(self, file_path: Path, language: str) -> List[Symbol]:
        """Extract lightweight symbol declarations for non-Python languages via tree-sitter."""
        query_defs = SYMBOL_QUERIES.get(language)
        print(f"Extracting symbols for {language} from {file_path}")
        print(f"Query defs: {query_defs}")
        if not query_defs:
            return []
        symbols: List[Symbol] = []
        parser = None
        try:
            from victor.codebase.tree_sitter_manager import get_parser

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
                        query = Query(parser.language, query_src)
                        captures = query.captures(tree.root_node)
                        print(f"Captures for {query_src}: {captures}")
                        for node, _ in captures:
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

        # Regex fallback when grammar support is unavailable
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
        # Python handled separately via AST (base_classes on symbols)
        if language == "python":
            for sym in symbols:
                if sym.type == "class" and sym.base_classes:
                    for base in sym.base_classes:
                        edges.append((sym.name, base))
            return edges

        query_src = INHERITS_QUERIES.get(language)
        parser = None
        try:
            from victor.codebase.tree_sitter_manager import get_parser

            parser = get_parser(language)
        except Exception:
            parser = None

        if parser is not None and query_src:
            try:
                content = file_path.read_bytes()
                tree = parser.parse(content)
                query = Query(parser.language, query_src)
                captures = query.captures(tree.root_node)
                child = None
                for node, capture_name in captures:
                    text = node.text.decode("utf-8", errors="ignore")
                    if capture_name == "child":
                        child = text
                    elif capture_name == "base" and child:
                        edges.append((child, text))
                        child = None
            except Exception:
                pass

        if not edges:
            # Regex fallback
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
        query_src = IMPLEMENTS_QUERIES.get(language)
        parser = None
        try:
            from victor.codebase.tree_sitter_manager import get_parser

            parser = get_parser(language)
        except Exception:
            parser = None

        if parser is not None and query_src:
            try:
                content = file_path.read_bytes()
                tree = parser.parse(content)
                query = Query(parser.language, query_src)
                captures = query.captures(tree.root_node)
                child = None
                for node, capture_name in captures:
                    text = node.text.decode("utf-8", errors="ignore")
                    if capture_name == "child":
                        child = text
                    elif capture_name in {"interface", "base"} and child:
                        edges.append((child, text))
                        child = None
            except Exception:
                pass

        if not edges:
            # Regex fallback for implements
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

        # Python handled via AST visitor (class attributes)
        if language == "python":
            for sym in symbols:
                if sym.type == "class" and hasattr(sym, "composition"):
                    edges.extend(getattr(sym, "composition"))
            return edges

        query_src = COMPOSITION_QUERIES.get(language)
        parser = None
        try:
            from victor.codebase.tree_sitter_manager import get_parser

            parser = get_parser(language)
        except Exception:
            parser = None

        if parser is not None and query_src:
            try:
                content = file_path.read_bytes()
                tree = parser.parse(content)
                query = Query(parser.language, query_src)
                captures = query.captures(tree.root_node)
                owner = None
                for node, capture_name in captures:
                    text = node.text.decode("utf-8", errors="ignore")
                    if capture_name == "owner":
                        owner = text
                    elif capture_name == "type" and owner:
                        edges.append((owner, text))
                # Do not clear owner on missing type; next owner will overwrite.
            except Exception:
                pass

        if edges:
            return edges

        # Regex fallback for typed declarations and new expressions
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
            # TypeScript/Java style property: field: Type or Type field;
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
                    text = field.text.decode("utf-8", errors="ignore")
                    if node_type in ("class_declaration", "interface_declaration"):
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
        query_src = CALL_QUERIES.get(language)
        if not query_src:
            return []
        try:
            from victor.codebase.tree_sitter_manager import get_parser
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
            captures = query.captures(tree.root_node)
            for node, _ in captures:
                callee = node.text.decode("utf-8", errors="ignore")
                caller = self._find_enclosing_symbol_name(node, language)
                if caller and callee:
                    call_edges.append((caller, callee))
        except Exception:
            call_edges = []

        if call_edges:
            return call_edges

        # Regex fallback when tree-sitter capture fails
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

    def _check_embeddings_integrity(self) -> bool:
        """Check if embeddings storage exists and has data.

        Supports multiple vector stores configured via settings:
        - LanceDB (default): Creates .lance directories for tables
        - ChromaDB: Creates chroma.sqlite3 file

        Returns:
            True if embeddings are intact, False if missing or empty
        """
        if not self.use_embeddings:
            return True  # Not using embeddings, so they're "intact"

        try:
            from victor.config.settings import get_project_paths, load_settings

            settings = load_settings()
            vector_store = getattr(settings, "codebase_vector_store", "lancedb")
            embeddings_dir = get_project_paths(self.root).embeddings_dir

            # Check if directory exists
            if not embeddings_dir.exists():
                logger.warning(f"Embeddings directory missing: {embeddings_dir}")
                return False

            # Check if directory has content
            contents = list(embeddings_dir.iterdir())
            if not contents:
                logger.warning(f"Embeddings directory is empty: {embeddings_dir}")
                return False

            # Vector store specific validation
            if vector_store == "lancedb":
                # LanceDB creates .lance directories for each table
                lance_tables = [d for d in contents if d.is_dir() and d.suffix == ".lance"]
                if not lance_tables:
                    # Also check for any subdirectories with data (table directories)
                    has_data = any(d.is_dir() and list(d.iterdir()) for d in contents if d.is_dir())
                    if not has_data:
                        logger.warning(f"LanceDB has no tables: {embeddings_dir}")
                        return False
            elif vector_store == "chromadb":
                # ChromaDB creates chroma.sqlite3 file
                chroma_db = embeddings_dir / "chroma.sqlite3"
                if not chroma_db.exists():
                    logger.warning(f"ChromaDB database missing: {chroma_db}")
                    return False
            else:
                # Generic check: any files or non-empty subdirectories
                has_data = any(f.is_file() for f in contents) or any(
                    d.is_dir() and list(d.iterdir()) for d in contents if d.is_dir()
                )
                if not has_data:
                    logger.warning(f"Embeddings directory has no valid data: {embeddings_dir}")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Failed to check embeddings integrity: {e}")
            return False

    async def startup_check(self, auto_reindex: bool = True) -> Dict[str, Any]:
        """Check index status at startup and reindex if needed.

        This should be called when Victor starts to ensure the index
        is up-to-date. It compares file mtimes with stored mtimes and
        verifies embeddings storage exists if embeddings are enabled.

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

        # Check if embeddings are enabled but storage is missing
        embeddings_intact = self._check_embeddings_integrity()
        if not embeddings_intact:
            logger.info("Embeddings storage missing or empty. Full indexing required.")
            print("⚠️  Embeddings storage missing or corrupted. Triggering full reindex...")
            if auto_reindex:
                await self.index_codebase(force=True)
            return {
                "status": "indexed" if auto_reindex else "needs_index",
                "action": "full_index" if auto_reindex else "none",
                "reason": "embeddings_missing",
                "files_indexed": len(self.files) if auto_reindex else 0,
            }

        # Check for mtime-based staleness
        is_stale, modified, deleted = self.check_staleness_by_mtime()

        if not is_stale:
            logger.info("Index is up to date based on file mtimes.")
            # Restore in-memory state from saved metadata
            self._is_indexed = True
            self._last_indexed = saved.get("last_indexed")
            file_count = len(saved.get("files", {}))
            print(f"✅ Index up to date ({file_count} files)")
            return {
                "status": "up_to_date",
                "action": "none",
                "files_in_index": file_count,
            }

        print(f"Index stale: {len(modified)} modified, {len(deleted)} deleted files")

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

    def register_change_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback to be notified when files change.

        This allows other systems (like SymbolStore) to stay synchronized.

        Args:
            callback: Function to call with the changed file path
        """
        self._change_callbacks.append(callback)

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

                # Notify registered callbacks
                for callback in self._change_callbacks:
                    try:
                        callback(file_path)
                    except Exception as e:
                        logger.warning(f"Change callback failed: {e}")

            except ValueError:
                # File outside root, ignore
                pass

    def start_watcher(self) -> bool:
        """Start file watcher for automatic staleness detection.

        Watches all supported source file types (Python, JS, TS, Go, Rust, etc.)

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
                file_patterns=self.WATCHED_PATTERNS,
                ignore_patterns=self.ignore_patterns,
            )

            self._observer = Observer()
            self._observer.schedule(self._file_handler, str(self.root), recursive=True)
            self._observer.start()

            logger.info(f"Started file watcher for {self.root} (all languages)")
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

        # Reset graph buffers and clear existing graph store for this repo
        self._reset_graph_buffers()
        self._pending_call_edges = []
        if self.graph_store:
            await self.graph_store.delete_by_repo()

        # Collect files by language (extension driven)
        language_files: Dict[str, List[Path]] = {}
        for ext, lang in EXTENSION_TO_LANGUAGE.items():
            for file_path in self.root.rglob(f"*{ext}"):
                if file_path.is_file() and not self.should_ignore(file_path):
                    language_files.setdefault(lang, []).append(file_path)

        python_files = language_files.pop("python", [])

        print(
            f"Found {len(python_files)} Python files and "
            f"{sum(len(v) for v in language_files.values())} non-Python files"
        )

        # Index files using parallel processing for CPU-bound AST parsing
        start_time = time.perf_counter()
        await self._parallel_index_files(python_files)

        # Index non-Python files sequentially (tree-sitter powered, lightweight)
        for language, files in language_files.items():
            for file_path in files:
                if self._is_config_language(language):
                    await self._index_config_file(file_path, language)
                else:
                    await self._index_tree_sitter_file(file_path, language)

        parse_time = time.perf_counter() - start_time

        # Persist graph nodes/edges if configured
        if self.graph_store:
            self._resolve_cross_file_calls()
        await self._persist_graph_store()

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

        self._record_symbols(metadata)

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
        self._reset_graph_buffers()
        self._pending_call_edges = []

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

    async def _persist_graph_store(self) -> None:
        """Flush buffered nodes/edges to the graph store."""
        if not self.graph_store:
            self._reset_graph_buffers()
            return

        if self._graph_nodes:
            await self.graph_store.upsert_nodes(self._graph_nodes)
        if self._graph_edges:
            await self.graph_store.upsert_edges(self._graph_edges)
        self._reset_graph_buffers()

    async def incremental_reindex(self) -> Dict[str, Any]:
        """Incrementally reindex only changed files.

        More efficient than full reindex when few files have changed.
        Uses incremental embedding updates to only re-embed changed files.

        Returns:
            Dictionary with reindex statistics
        """
        # Graph store currently rebuilt via full reindex to keep consistency
        if self.graph_store:
            return await self.reindex()

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
            metadata.call_edges = visitor.call_edges
            metadata.compose_edges = visitor.composition_edges
            # Map base classes into inheritance edges
            metadata.inherit_edges = []
            for sym in metadata.symbols:
                if sym.type == "class" and sym.base_classes:
                    for base in sym.base_classes:
                        metadata.inherit_edges.append((sym.name, base))
                if sym.type == "class":
                    sym.composition = [
                        edge for edge in visitor.composition_edges if edge[0] == sym.name
                    ]
            metadata.references = self._extract_references(
                file_path,
                self._detect_language(file_path, metadata.language),
                [callee for _, callee in metadata.call_edges],
                metadata.imports,
            )

            self.files[metadata.path] = metadata
            self._record_symbols(metadata)

            # Graph capture for sequential indexing
            self._record_symbols(metadata)

        except Exception as e:
            print(f"Error indexing {file_path}: {e}")

    async def _index_config_file(self, file_path: Path, language: str) -> None:
        """Lightweight indexing for config/metadata files."""
        try:
            stat = file_path.stat()
            content = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.debug(f"Skipping config file {file_path}: {exc}")
            return

        keys_with_line = self._extract_config_keys(content, language)
        symbols: List[Symbol] = []
        for key, line_no in keys_with_line:
            symbols.append(
                Symbol(
                    name=key,
                    type="config_key",
                    file_path=str(file_path.relative_to(self.root)),
                    line_number=line_no,
                )
            )

        metadata = FileMetadata(
            path=str(file_path.relative_to(self.root)),
            language=language,
            symbols=symbols,
            imports=[],
            last_modified=stat.st_mtime,
            indexed_at=time.time(),
            size=stat.st_size,
            lines=content.count("\n") + 1,
            content_hash=self._compute_file_hash(file_path),
        )
        metadata.references = [name for name, _ in keys_with_line]
        self.files[metadata.path] = metadata
        self._record_symbols(metadata)

    async def _index_tree_sitter_file(self, file_path: Path, language: str) -> None:
        """Index a non-Python file using tree-sitter for symbols/references."""
        try:
            stat = file_path.stat()
            content = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.debug(f"Skipping {file_path} due to read error: {exc}")
            return

        symbols = self._extract_symbols_with_tree_sitter(file_path, language)
        call_edges = self._extract_calls_with_tree_sitter(file_path, language)

        metadata = FileMetadata(
            path=str(file_path.relative_to(self.root)),
            language=language,
            symbols=symbols,
            imports=[],  # TODO: add richer import extraction per language
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

        # Always create a file node so config/docs files without symbols still appear in the graph.
        file_node_id: Optional[str] = None
        if self.graph_store:
            file_node_id = f"file:{metadata.path}"
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
            # Symbol registry
            self.symbols[f"{metadata.path}:{symbol.name}"] = symbol
            if metadata.path not in self.symbol_index:
                self.symbol_index[metadata.path] = []
            self.symbol_index[metadata.path].append(symbol.name)

            if self.graph_store:
                symbol_id = f"symbol:{metadata.path}:{symbol.name}"

                self._graph_nodes.append(
                    GraphNode(
                        node_id=symbol_id,
                        type=symbol.type,
                        name=symbol.name,
                        file=metadata.path,
                        line=symbol.line_number,
                        lang=metadata.language,
                        metadata={
                            "signature": symbol.signature,
                            "docstring": symbol.docstring,
                        },
                    )
                )
                self._graph_edges.append(
                    GraphEdge(
                        src=file_node_id or f"file:{metadata.path}",
                        dst=symbol_id,
                        type="CONTAINS",
                        metadata={"path": metadata.path},
                    )
                )

        # Add simple intra-file CALLS edges when both endpoints are known symbols
        if self.graph_store and metadata.call_edges:
            for caller, callee in metadata.call_edges:
                if caller not in symbol_names or callee not in symbol_names:
                    # Track for potential cross-file resolution
                    caller_id = f"symbol:{metadata.path}:{caller}"
                    self._pending_call_edges.append((caller_id, callee, metadata.path))
                    continue
                caller_id = f"symbol:{metadata.path}:{caller}"
                callee_id = f"symbol:{metadata.path}:{callee}"
                self._graph_edges.append(
                    GraphEdge(
                        src=caller_id,
                        dst=callee_id,
                        type="CALLS",
                        metadata={"path": metadata.path},
                    )
                )

        # Inheritance edges (child -> base)
        if self.graph_store and metadata.inherit_edges:
            for child, base in metadata.inherit_edges:
                child_id = f"symbol:{metadata.path}:{child}"
                if child not in symbol_names:
                    continue
                # if base is in current file, link directly, else resolve later
                if base in symbol_names:
                    base_id = f"symbol:{metadata.path}:{base}"
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

        # Implements edges (child -> interface/abstract)
        if self.graph_store and metadata.implements_edges:
            for child, base in metadata.implements_edges:
                child_id = f"symbol:{metadata.path}:{child}"
                if child not in symbol_names:
                    continue
                if base in symbol_names:
                    base_id = f"symbol:{metadata.path}:{base}"
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

        # Composition edges (owner -> member type)
        if self.graph_store and metadata.compose_edges:
            for owner, member in metadata.compose_edges:
                owner_id = f"symbol:{metadata.path}:{owner}"
                if owner not in symbol_names:
                    continue
                if member in symbol_names:
                    member_id = f"symbol:{metadata.path}:{member}"
                    self._graph_edges.append(
                        GraphEdge(
                            src=owner_id,
                            dst=member_id,
                            type="COMPOSES",
                            metadata={"path": metadata.path},
                        )
                    )
                else:
                    self._pending_compose_edges.append((owner_id, member, metadata.path))

        # Add IMPORTS edges from file to imported module names (cross-file reference scaffold)
        if self.graph_store and metadata.imports:
            for imp in metadata.imports:
                module_node_id = f"module:{imp}"
                self._graph_nodes.append(
                    GraphNode(
                        node_id=module_node_id,
                        type="module",
                        name=imp,
                        file=metadata.path,
                        lang=metadata.language,
                    )
                )
                self._graph_edges.append(
                    GraphEdge(
                        src=f"file:{metadata.path}",
                        dst=module_node_id,
                        type="IMPORTS",
                        metadata={"path": metadata.path},
                    )
                )

    def _resolve_cross_file_calls(self) -> None:
        """Resolve pending CALLS edges across files by matching symbol names globally."""
        if not self._pending_call_edges:
            return

        # Build resolver index from graph nodes
        node_ids = []
        for sym_key in self.symbols.keys():
            node_ids.append(f"symbol:{sym_key}")
        self._symbol_resolver.ingest(node_ids)

        # Cross-file CALLS resolution
        for caller_id, callee_name, file_path in self._pending_call_edges:
            target_id = self._symbol_resolver.resolve(callee_name, preferred_file=file_path)
            if not target_id:
                # Try short name heuristic (after ingest it exists already)
                target_id = self._symbol_resolver.resolve(callee_name.split(".")[-1], preferred_file=file_path)
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

        # INHERITS resolution
        for child_id, base_name, file_path in self._pending_inherit_edges:
            target_id = self._symbol_resolver.resolve(base_name, preferred_file=file_path)
            if not target_id:
                target_id = self._symbol_resolver.resolve(base_name.split(".")[-1], preferred_file=file_path)
            if not target_id:
                continue
            self._graph_edges.append(
                GraphEdge(
                    src=child_id,
                    dst=target_id,
                    type="INHERITS",
                    metadata={"path": file_path, "resolved": True},
                )
            )

        # IMPLEMENTS resolution (interfaces/abstract types)
        for child_id, base_name, file_path in self._pending_implements_edges:
            target_id = self._symbol_resolver.resolve(base_name, preferred_file=file_path)
            if not target_id:
                target_id = self._symbol_resolver.resolve(base_name.split(".")[-1], preferred_file=file_path)
            if not target_id:
                continue
            self._graph_edges.append(
                GraphEdge(
                    src=child_id,
                    dst=target_id,
                    type="IMPLEMENTS",
                    metadata={"path": file_path, "resolved": True},
                )
            )

        # COMPOSES/has-a relationships resolution
        for owner_id, member_name, file_path in self._pending_compose_edges:
            target_id = self._symbol_resolver.resolve(member_name, preferred_file=file_path)
            if not target_id:
                target_id = self._symbol_resolver.resolve(member_name.split(".")[-1], preferred_file=file_path)
            if not target_id:
                continue
            self._graph_edges.append(
                GraphEdge(
                    src=owner_id,
                    dst=target_id,
                    type="COMPOSES",
                    metadata={"path": file_path, "resolved": True},
                )
            )

        # REFERENCES edges (file -> symbol) for any referenced identifier
        for metadata in self.files.values():
            if not metadata.references:
                continue
            file_node = f"file:{metadata.path}"
            for ref in metadata.references:
                target_id = self._symbol_resolver.resolve(ref, preferred_file=metadata.path)
                if not target_id:
                    target_id = self._symbol_resolver.resolve(ref.split(".")[-1], preferred_file=metadata.path)
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

        Embeddings are stored in {rootrepo}/.victor/embeddings/ directory by default.
        This keeps all index data co-located with the repository.

        Configuration is read from settings.py with sensible defaults:
        - vector_store: lancedb (disk-based ANN, lower memory)
        - embedding_model: BAAI/bge-small-en-v1.5 (384-dim, excellent for code)

        Args:
            config: Embedding configuration dict (overrides settings if provided)
        """
        try:
            from victor.codebase.embeddings import EmbeddingConfig, EmbeddingRegistry

            # Create config with defaults from settings
            if not config:
                config = {}

            # Load settings for defaults
            from victor.config.settings import get_project_paths, load_settings

            settings = load_settings()
            default_persist_dir = get_project_paths(self.root).embeddings_dir

            # Use settings as defaults, allow config to override
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
        self.current_function: Optional[str] = None
        self.call_edges: List[tuple[str, str]] = []
        self.composition_edges: List[tuple[str, str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        bases: List[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
            elif isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
                bases.append(base.value.id)
        symbol = Symbol(
            name=node.name,
            type="class",
            file_path=self.metadata.path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            base_classes=bases,
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
        old_function = self.current_function
        self.current_function = name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            self.metadata.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statement."""
        if node.module:
            self.metadata.imports.append(node.module)

    def visit_Call(self, node: ast.Call) -> None:
        """Capture simple call relationships for intra-file graph edges."""
        if self.current_function:
            callee = None
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr

            if callee:
                self.call_edges.append((self.current_function, callee))

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Detect has-a relationships for class attributes."""
        if self.current_class:
            target_type: Optional[str] = None
            if isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name):
                    target_type = func.id
                elif isinstance(func, ast.Attribute):
                    target_type = func.attr
            elif isinstance(node.value, ast.Name):
                target_type = node.value.id
            if target_type:
                self.composition_edges.append((self.current_class, target_type))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Capture annotated attributes inside classes for composition edges."""
        if self.current_class:
            target_type: Optional[str] = None
            if isinstance(node.annotation, ast.Name):
                target_type = node.annotation.id
            elif isinstance(node.annotation, ast.Attribute):
                target_type = node.annotation.attr
            if target_type:
                self.composition_edges.append((self.current_class, target_type))
        self.generic_visit(node)


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
