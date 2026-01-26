"""
Tree-sitter based symbol extractor.

Provides universal symbol extraction for any language with tree-sitter support.
This is the fallback extractor when native AST is not available.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..types import ExtractedSymbol, TreeSitterCapability

logger = logging.getLogger(__name__)

# Language module mappings for tree-sitter
# Maps language name to (module_name, function_name)
LANGUAGE_MODULES: Dict[str, Tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "csharp": ("tree_sitter_c_sharp", "language"),
    "scala": ("tree_sitter_scala", "language"),
    "kotlin": ("tree_sitter_kotlin", "language"),
    "swift": ("tree_sitter_swift", "language"),
    "lua": ("tree_sitter_lua", "language"),
    "bash": ("tree_sitter_bash", "language"),
    "html": ("tree_sitter_html", "language"),
    "css": ("tree_sitter_css", "language"),
    "json": ("tree_sitter_json", "language"),
    "yaml": ("tree_sitter_yaml", "language"),
    "toml": ("tree_sitter_toml", "language"),
    "markdown": ("tree_sitter_markdown", "language"),
    # Additional configuration and documentation formats
    "xml": ("tree_sitter_xml", "language"),
    "hocon": ("tree_sitter_hocon", "language"),
    "ini": ("tree_sitter_ini", "language"),
    "properties": ("tree_sitter_properties", "language"),
    "graphql": ("tree_sitter_graphql", "language"),
    "proto": ("tree_sitter_proto", "language"),
    "jsonnet": ("tree_sitter_jsonnet", "language"),
    "rst": ("tree_sitter_rst", "language"),
    "latex": ("tree_sitter_latex", "language"),
    "gitignore": ("tree_sitter_gitignore", "language"),
    "dotenv": ("tree_sitter_dotenv", "language"),
    "csv": ("tree_sitter_csv", "language"),
}

# Symbol extraction queries for common languages
# These S-expression queries extract class, function, and method definitions
SYMBOL_QUERIES: Dict[str, str] = {
    "python": """
        (class_definition
            name: (identifier) @name) @definition.class
        (function_definition
            name: (identifier) @name) @definition.function
    """,
    "javascript": """
        (class_declaration
            name: (identifier) @name) @definition.class
        (function_declaration
            name: (identifier) @name) @definition.function
        (method_definition
            name: (property_identifier) @name) @definition.method
        (arrow_function
            (identifier) @name) @definition.function
    """,
    "typescript": """
        (class_declaration
            name: (type_identifier) @name) @definition.class
        (function_declaration
            name: (identifier) @name) @definition.function
        (method_definition
            name: (property_identifier) @name) @definition.method
        (interface_declaration
            name: (type_identifier) @name) @definition.interface
    """,
    "go": """
        (function_declaration
            name: (identifier) @name) @definition.function
        (method_declaration
            name: (field_identifier) @name) @definition.method
        (type_declaration
            (type_spec
                name: (type_identifier) @name)) @definition.type
    """,
    "rust": """
        (function_item
            name: (identifier) @name) @definition.function
        (struct_item
            name: (type_identifier) @name) @definition.struct
        (enum_item
            name: (type_identifier) @name) @definition.enum
        (impl_item
            type: (type_identifier) @name) @definition.impl
        (trait_item
            name: (type_identifier) @name) @definition.trait
    """,
    "java": """
        (class_declaration
            name: (identifier) @name) @definition.class
        (method_declaration
            name: (identifier) @name) @definition.method
        (interface_declaration
            name: (identifier) @name) @definition.interface
    """,
    "c": """
        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @name)) @definition.function
        (struct_specifier
            name: (type_identifier) @name) @definition.struct
    """,
    "cpp": """
        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @name)) @definition.function
        (class_specifier
            name: (type_identifier) @name) @definition.class
        (struct_specifier
            name: (type_identifier) @name) @definition.struct
    """,
}


class TreeSitterExtractor:
    """
    Universal tree-sitter based symbol extractor.

    Uses tree-sitter grammars to extract symbols from any supported language.
    Falls back to generic extraction for languages without specific queries.
    """

    def __init__(self) -> None:
        self._parser_cache: Dict[str, Any] = {}
        self._language_cache: Dict[str, Any] = {}
        self._query_cache: Dict[Tuple[str, str], Any] = {}
        self._tree_sitter_available = self._check_tree_sitter()

    def _check_tree_sitter(self) -> bool:
        """Check if tree-sitter is available."""
        try:
            import tree_sitter
            return True
        except ImportError:
            logger.warning("tree-sitter not available, falling back to no-op extraction")
            return False

    def is_available(self) -> bool:
        """Check if tree-sitter extraction is available."""
        return self._tree_sitter_available

    def extract(
        self,
        code: str,
        file_path: Path,
        language: str,
    ) -> List[ExtractedSymbol]:
        """
        Extract symbols from code using tree-sitter.

        Args:
            code: Source code to parse
            file_path: Path to the source file
            language: Language identifier

        Returns:
            List of extracted symbols
        """
        if not self._tree_sitter_available:
            return []

        try:
            parser = self._get_parser(language)
            if not parser:
                logger.debug(f"No tree-sitter parser available for {language}")
                return []

            tree = parser.parse(bytes(code, "utf8"))
            return self._extract_symbols(tree, str(file_path), language)

        except Exception as e:
            logger.warning(f"Tree-sitter extraction failed for {file_path}: {e}")
            return []

    def _get_parser(self, language: str) -> Optional[Any]:
        """Get or create a parser for the language."""
        if language in self._parser_cache:
            return self._parser_cache[language]

        try:
            import tree_sitter

            lang = self._get_language(language)
            if not lang:
                return None

            parser = tree_sitter.Parser(lang)
            self._parser_cache[language] = parser
            return parser

        except Exception as e:
            logger.debug(f"Failed to create parser for {language}: {e}")
            return None

    def _get_language(self, language: str) -> Optional[Any]:
        """Get the tree-sitter Language object for a language."""
        if language in self._language_cache:
            return self._language_cache[language]

        module_info = LANGUAGE_MODULES.get(language)
        if not module_info:
            logger.debug(f"No tree-sitter module mapping for {language}")
            return None

        module_name, func_name = module_info

        try:
            import importlib
            from tree_sitter import Language

            module = importlib.import_module(module_name)
            lang_func = getattr(module, func_name)
            lang_obj = lang_func()

            # Wrap PyCapsule with Language if needed (tree-sitter 0.25+ API)
            if not isinstance(lang_obj, Language):
                lang = Language(lang_obj)
            else:
                lang = lang_obj

            self._language_cache[language] = lang
            return lang

        except ImportError as e:
            logger.debug(f"tree-sitter grammar not installed for {language}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to load tree-sitter language for {language}: {e}")
            return None

    def _extract_symbols(
        self,
        tree: Any,
        file_path: str,
        language: str,
    ) -> List[ExtractedSymbol]:
        """Extract symbols from a parsed tree."""
        symbols: List[ExtractedSymbol] = []

        # Get language-specific query
        query_src = SYMBOL_QUERIES.get(language)
        if query_src:
            symbols.extend(
                self._extract_with_query(tree, file_path, language, query_src)
            )
        else:
            # Fall back to generic extraction
            symbols.extend(
                self._extract_generic(tree.root_node, file_path)
            )

        return symbols

    def _extract_with_query(
        self,
        tree: Any,
        file_path: str,
        language: str,
        query_src: str,
    ) -> List[ExtractedSymbol]:
        """Extract symbols using a tree-sitter query."""
        symbols: List[ExtractedSymbol] = []

        try:
            from tree_sitter import Query, QueryCursor

            lang = self._get_language(language)
            if not lang:
                return []

            # Cache queries
            cache_key = (language, query_src)
            if cache_key in self._query_cache:
                query = self._query_cache[cache_key]
            else:
                query = Query(lang, query_src)
                self._query_cache[cache_key] = query

            # Use QueryCursor for captures (tree-sitter 0.25+ API)
            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)

            # captures is a dict: {"name": [nodes], "definition.class": [nodes], ...}
            # We need to match name nodes with their parent definitions
            name_nodes = captures.get("name", [])

            # Extract from definition captures
            for capture_name, nodes in captures.items():
                if capture_name.startswith("definition."):
                    symbol_type = capture_name.replace("definition.", "")
                    for node in nodes:
                        # Find matching name node within this definition
                        name = self._find_name_in_node(node)
                        if name:
                            symbols.append(
                                ExtractedSymbol(
                                    name=name,
                                    symbol_type=symbol_type,
                                    file_path=file_path,
                                    line_number=node.start_point[0] + 1,
                                    end_line=node.end_point[0] + 1,
                                )
                            )

        except Exception as e:
            logger.warning(f"Query-based extraction failed: {e}")

        return symbols

    def _extract_generic(
        self,
        node: Any,
        file_path: str,
        parent: Optional[str] = None,
    ) -> List[ExtractedSymbol]:
        """Generic symbol extraction for languages without queries."""
        symbols: List[ExtractedSymbol] = []

        # Common node types that indicate symbol definitions
        symbol_types = {
            "class_definition": "class",
            "class_declaration": "class",
            "function_definition": "function",
            "function_declaration": "function",
            "method_definition": "method",
            "method_declaration": "method",
            "interface_declaration": "interface",
            "struct_specifier": "struct",
            "enum_specifier": "enum",
        }

        if node.type in symbol_types:
            name = self._find_name_in_node(node)
            if name:
                symbols.append(
                    ExtractedSymbol(
                        name=name,
                        symbol_type=symbol_types[node.type],
                        file_path=file_path,
                        line_number=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent_symbol=parent,
                    )
                )
                parent = name

        # Recurse into children
        for child in node.children:
            symbols.extend(self._extract_generic(child, file_path, parent))

        return symbols

    def _find_name_in_node(self, node: Any) -> Optional[str]:
        """Find the name identifier in a node."""
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "property_identifier"):
                return child.text.decode("utf8")
            if child.type == "name":
                return child.text.decode("utf8")
        return None

    def has_syntax_errors(self, code: str, language: str) -> bool:
        """Check if code has syntax errors."""
        if not self._tree_sitter_available:
            return False

        parser = self._get_parser(language)
        if not parser:
            return False

        try:
            tree = parser.parse(bytes(code, "utf8"))
            return self._has_error_nodes(tree.root_node)
        except Exception:
            return True

    def _has_error_nodes(self, node: Any) -> bool:
        """Check if node tree contains ERROR or MISSING nodes."""
        if node.type == "ERROR" or node.is_missing:
            return True
        for child in node.children:
            if self._has_error_nodes(child):
                return True
        return False

    def get_error_locations(
        self,
        code: str,
        language: str,
    ) -> List[Dict[str, Any]]:
        """Get locations of syntax errors in code."""
        if not self._tree_sitter_available:
            return []

        parser = self._get_parser(language)
        if not parser:
            return []

        try:
            tree = parser.parse(bytes(code, "utf8"))
            errors: List[Dict[str, Any]] = []
            self._collect_errors(tree.root_node, errors)
            return errors
        except Exception as e:
            return [{
                "line": 1,
                "column": 0,
                "message": str(e),
                "type": "parse_error",
            }]

    def _collect_errors(
        self,
        node: Any,
        errors: List[Dict[str, Any]],
    ) -> None:
        """Collect error nodes from tree."""
        if node.type == "ERROR":
            errors.append({
                "line": node.start_point[0] + 1,
                "column": node.start_point[1],
                "end_line": node.end_point[0] + 1,
                "end_column": node.end_point[1],
                "message": "Syntax error",
                "type": "error",
            })
        elif node.is_missing:
            errors.append({
                "line": node.start_point[0] + 1,
                "column": node.start_point[1],
                "message": f"Missing: {node.type}",
                "type": "missing",
            })

        for child in node.children:
            self._collect_errors(child, errors)

    def get_supported_languages(self) -> List[str]:
        """Get list of languages with known tree-sitter support."""
        return list(LANGUAGE_MODULES.keys())

    def is_language_available(self, language: str) -> bool:
        """Check if tree-sitter grammar is available for a language."""
        if not self._tree_sitter_available:
            return False
        return self._get_language(language) is not None
