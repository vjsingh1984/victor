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


from typing import TYPE_CHECKING, Dict, List

from tree_sitter import Language, Parser, Query, QueryCursor

if TYPE_CHECKING:
    from tree_sitter import Node, Tree


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
    lang = get_language(language)
    query = Query(lang, query_src)
    cursor = QueryCursor(query)
    return cursor.captures(tree.root_node)
