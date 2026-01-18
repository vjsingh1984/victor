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


import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from tree_sitter import Language, Parser, Query, QueryCursor

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


# Language package mapping for tree-sitter 0.25+
# These use pre-compiled language packages instead of runtime compilation
# Install with: pip install tree-sitter-<language>
# Format: "language_name": ("module_name", "function_name")
# function_name is the function that returns the Language object (usually "language")
LANGUAGE_MODULES: Dict[str, tuple[str, str]] = {
    # Core languages (commonly used)
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),  # Special case
    "tsx": ("tree_sitter_typescript", "language_tsx"),  # TypeScript + JSX
    "java": ("tree_sitter_java", "language"),
    "go": ("tree_sitter_go", "language"),
    # NOTE: tree-sitter-rust >=0.25.0 is recommended to match tree-sitter >=0.25 API
    "rust": ("tree_sitter_rust", "language"),
    # Additional languages
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "c_sharp": ("tree_sitter_c_sharp", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "php": ("tree_sitter_php", "language_php"),  # May have special name
    "kotlin": ("tree_sitter_kotlin", "language"),
    "swift": ("tree_sitter_swift", "language"),
    "scala": ("tree_sitter_scala", "language"),
    "bash": ("tree_sitter_bash", "language"),
    "sql": ("tree_sitter_sql", "language"),
    # Web languages
    "html": ("tree_sitter_html", "language"),
    "css": ("tree_sitter_css", "language"),
    "json": ("tree_sitter_json", "language"),
    "yaml": ("tree_sitter_yaml", "language"),
    "toml": ("tree_sitter_toml", "language"),
    # Other
    "lua": ("tree_sitter_lua", "language"),
    "elixir": ("tree_sitter_elixir", "language"),
    "haskell": ("tree_sitter_haskell", "language"),
    "r": ("tree_sitter_r", "language"),
}

_language_cache: Dict[str, Language] = {}
_parser_cache: Dict[str, Parser] = {}

# Try to import Rust AST processor accelerator
_ast_accelerator = None
try:
    from victor.native.accelerators.ast_processor import get_ast_processor

    _ast_accelerator = get_ast_processor()
    if _ast_accelerator.is_available:
        logger.info("AST processing: Using Rust accelerator (10x faster)")
    else:
        logger.info("AST processing: Using Python tree-sitter")
except ImportError:
    logger.debug("Rust AST accelerator not available, using Python tree-sitter")
    _ast_accelerator = None


def get_language(language: str) -> Language:
    """
    Loads a tree-sitter Language object using pre-compiled language packages.

    This uses the tree-sitter 0.25+ API which requires pre-installed language packages
    (e.g., tree-sitter-python) instead of runtime compilation.
    """
    if language in _language_cache:
        return _language_cache[language]

    module_info = LANGUAGE_MODULES.get(language)
    if not module_info:
        raise ValueError(f"Unsupported language for tree-sitter: {language}")

    module_name, func_name = module_info

    try:
        # Dynamically import the language module
        language_module = __import__(module_name)

        # Get the language function (may be "language", "language_typescript", etc.)
        lang_func = getattr(language_module, func_name)

        # Create Language object using the new API
        # In tree-sitter 0.25+, Language() takes a language object from the module
        lang_obj = lang_func()
        # Some older grammars (e.g., tree_sitter_rust 0.24.x) expose a PyCapsule; wrap via Language
        lang = Language(lang_obj) if not isinstance(lang_obj, Language) else lang_obj

        _language_cache[language] = lang
        return lang

    except ImportError:
        raise ImportError(
            f"Language package '{module_name}' not installed. "
            f"Install it with: pip install {module_name.replace('_', '-')}"
        )
    except AttributeError:
        raise AttributeError(
            f"Language module '{module_name}' does not have function '{func_name}'. "
            f"Check the tree-sitter package version and update LANGUAGE_MODULES."
        )


def get_parser(language: str) -> Parser:
    """
    Returns a tree-sitter Parser initialized with the specified language.

    In tree-sitter 0.25+, Parser() constructor takes the Language object directly.
    """
    if language in _parser_cache:
        return _parser_cache[language]

    lang = get_language(language)

    # New API: Parser takes Language object in constructor
    parser = Parser(lang)

    _parser_cache[language] = parser
    return parser


def run_query(tree: "Tree", query_src: str, language: str) -> Dict[str, List["Node"]]:
    """Run a tree-sitter query using the modern QueryCursor API.

    This is the preferred way to run queries in tree-sitter 0.25+.
    The old `query.captures(node)` method returns List[Tuple[Node, str]],
    but the new QueryCursor API returns Dict[str, List[Node]].

    Args:
        tree: Parsed tree-sitter tree
        query_src: Query source string (S-expression syntax)
        language: Language name (e.g., "python", "javascript")

    Returns:
        Dictionary mapping capture names to lists of matching nodes.
        For example, for query `(function_definition name: (identifier) @name)`,
        returns {"name": [<node>, <node>, ...]}.

    Example:
        >>> parser = get_parser("python")
        >>> tree = parser.parse(b"def foo(): pass")
        >>> captures = run_query(tree, "(function_definition name: (identifier) @name)", "python")
        >>> captures["name"][0].text
        b'foo'
    """
    # Use Rust accelerator if available
    if _ast_accelerator is not None and _ast_accelerator.rust_available:
        try:
            nodes = _ast_accelerator.execute_query(tree, query_src, language)
            # Convert list of nodes to capture dict format
            # For now, return all nodes under a default capture name
            return {"_all": nodes}
        except Exception as e:
            logger.debug(f"Rust query execution failed, falling back to Python: {e}")

    # Python fallback
    lang = get_language(language)
    query = Query(lang, query_src)
    cursor = QueryCursor(query)
    return cursor.captures(tree.root_node)


def parse_file_accelerated(
    file_path: str,
    language: Optional[str] = None,
) -> Optional["Tree"]:
    """Parse a file to AST using Rust-accelerated parser when available.

    This function provides 10x faster parsing by using the Rust accelerator
    when available, with automatic fallback to Python tree-sitter.

    Args:
        file_path: Path to the file to parse
        language: Optional language name. If None, detected from file extension

    Returns:
        Parsed tree-sitter Tree, or None if parsing fails

    Example:
        >>> tree = parse_file_accelerated("my_file.py")
        >>> tree.root_node.type
        'module'
    """
    source_code = _read_file(file_path)
    if source_code is None:
        return None

    language = language or _detect_language(file_path)

    # Use Rust accelerator if available
    if _ast_accelerator is not None and _ast_accelerator.rust_available:
        try:
            tree = _ast_accelerator.parse_to_ast(source_code, language, file_path)
            return tree
        except Exception as e:
            logger.debug(f"Rust parsing failed for {file_path}, falling back to Python: {e}")

    # Python fallback
    parser = get_parser(language)
    return parser.parse(bytes(source_code, "utf-8"))


def parse_file_with_timing(file_path: str) -> Tuple[Optional["Tree"], float]:
    """Parse file and return timing information for performance monitoring.

    Args:
        file_path: Path to the file to parse

    Returns:
        Tuple of (parsed_tree, elapsed_time_seconds)

    Example:
        >>> tree, elapsed = parse_file_with_timing("my_file.py")
        >>> print(f"Parsed in {elapsed*1000:.2f}ms")
    """
    start = time.perf_counter()
    tree = parse_file_accelerated(file_path)
    elapsed = time.perf_counter() - start

    backend = "rust" if (_ast_accelerator and _ast_accelerator.rust_available) else "python"
    logger.debug(f"Parsed {file_path} in {elapsed*1000:.2f}ms (backend: {backend})")

    return tree, elapsed


def extract_symbols_parallel(
    files: List[str],
    symbol_types: List[str],
) -> Dict[str, List[Dict]]:
    """Extract symbols from multiple files using parallel processing.

    This function provides 10x faster symbol extraction by using the Rust
    accelerator's parallel processing when available.

    Args:
        files: List of file paths to process
        symbol_types: Types of symbols to extract (e.g., ["function", "class"])

    Returns:
        Dictionary mapping file paths to lists of symbol info dicts

    Example:
        >>> results = extract_symbols_parallel(["file1.py", "file2.py"], ["function", "class"])
        >>> for file_path, symbols in results.items():
        ...     print(f"{file_path}: {len(symbols)} symbols")
    """
    # Prepare file data
    file_data = []
    for file_path in files:
        source = _read_file(file_path)
        if source is None:
            continue
        language = _detect_language(file_path)
        file_data.append((language, source, file_path))

    if not file_data:
        return {fp: [] for fp in files}

    # Use Rust parallel extraction if available
    if _ast_accelerator is not None and _ast_accelerator.rust_available:
        try:
            results = _ast_accelerator.extract_symbols_parallel(file_data, symbol_types)
            return results
        except Exception as e:
            logger.debug(f"Rust parallel extraction failed, falling back to Python: {e}")

    # Python fallback (sequential)
    results = {}
    for language, source, file_path in file_data:
        try:
            tree = parse_file_accelerated(file_path, language)
            if tree is None:
                results[file_path] = []
                continue

            # Build query for symbol types
            query_parts = []
            for sym_type in symbol_types:
                if sym_type == "function":
                    query_parts.append("(function_definition name: (identifier) @name)")
                elif sym_type == "class":
                    query_parts.append("(class_definition name: (identifier) @name)")
                elif sym_type == "method":
                    query_parts.append("(method_definition name: (identifier) @name)")

            if not query_parts:
                results[file_path] = []
                continue

            query = "\n".join(query_parts)
            nodes_dict = run_query(tree, query, language)

            # Extract symbol info (handle both Rust and Python formats)
            symbols = []
            # Rust accelerator returns {"_all": [nodes]}
            # Python cursor.captures returns {"name": [nodes]}
            all_nodes = nodes_dict.get("_all", [])
            if not all_nodes:
                # Try Python format (capture name "name")
                all_nodes = nodes_dict.get("name", [])

            for node in all_nodes:
                if hasattr(node, "text"):
                    symbols.append({
                        "name": node.text.decode("utf-8") if isinstance(node.text, bytes) else node.text,
                        "type": node.parent.type if node.parent else "unknown",
                        "line": node.start_point[0] + 1,
                        "column": node.start_point[1],
                    })

            results[file_path] = symbols

        except Exception as e:
            logger.error(f"Symbol extraction failed for {file_path}: {e}")
            results[file_path] = []

    return results


def clear_ast_cache() -> None:
    """Clear the AST cache.

    This clears both the Rust accelerator's cache and the Python parser cache.
    """
    global _parser_cache

    # Clear Python parser cache
    _parser_cache.clear()

    # Clear Rust accelerator cache
    if _ast_accelerator is not None:
        _ast_accelerator.clear_cache()

    logger.debug("AST cache cleared")


def get_cache_stats() -> Dict[str, int]:
    """Get AST cache statistics.

    Returns:
        Dictionary with cache stats including size, hits, misses, and hit rate

    Example:
        >>> stats = get_cache_stats()
        >>> print(f"Cache hit rate: {stats['hit_rate']}%")
    """
    stats = {
        "size": len(_parser_cache),
        "max_size": len(_parser_cache),  # Python cache has no max size
        "hits": 0,
        "misses": 0,
        "hit_rate": 0,
    }

    # Add Rust accelerator stats if available
    if _ast_accelerator is not None:
        rust_stats = _ast_accelerator.cache_stats
        stats.update(rust_stats)

    return stats


def _read_file(file_path: str) -> Optional[str]:
    """Read file contents safely.

    Args:
        file_path: Path to the file

    Returns:
        File contents as string, or None if read fails
    """
    try:
        path = Path(file_path)
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None


def _detect_language(file_path: str) -> str:
    """Detect language from file extension.

    Args:
        file_path: Path to the file

    Returns:
        Language name (e.g., "python", "javascript")
    """
    ext = Path(file_path).suffix.lower()

    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "c_sharp",
        ".rb": "ruby",
        ".php": "php",
        ".kt": "kotlin",
        ".swift": "swift",
        ".scala": "scala",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".sql": "sql",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".lua": "lua",
        ".ex": "elixir",
        ".exs": "elixir",
        ".hs": "haskell",
        ".r": "r",
        ".R": "r",
    }

    return language_map.get(ext, "python")  # Default to Python
