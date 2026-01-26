"""
Tree-sitter based code validator.

Uses tree-sitter's error recovery to detect syntax errors in any
supported language. Tree-sitter parsers can produce partial ASTs
even when code has errors, making them useful for validation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..types import (
    CodeValidationResult,
    LanguageTier,
    ValidationConfig,
    ValidationIssue,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)

# Language module mappings (shared with extractor)
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


class TreeSitterValidator:
    """
    Tree-sitter based syntax validator.

    Uses tree-sitter's error recovery to detect syntax errors.
    Tree-sitter parsers mark error nodes with type "ERROR" and
    missing nodes with is_missing=True.
    """

    def __init__(self) -> None:
        self._parser_cache: Dict[str, Any] = {}
        self._language_cache: Dict[str, Any] = {}
        self._tree_sitter_available = self._check_tree_sitter()

    def _check_tree_sitter(self) -> bool:
        """Check if tree-sitter is available."""
        try:
            import tree_sitter
            return True
        except ImportError:
            logger.warning("tree-sitter not available for validation")
            return False

    def is_available(self) -> bool:
        """Check if tree-sitter validation is available."""
        return self._tree_sitter_available

    def validate(
        self,
        code: str,
        file_path: Path,
        language: str,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """
        Validate code syntax using tree-sitter.

        Args:
            code: Source code to validate
            file_path: Path to the source file
            language: Language identifier
            config: Optional validation configuration

        Returns:
            CodeValidationResult with validation status and errors
        """
        config = config or ValidationConfig()

        # Determine tier based on language
        tier = LanguageTier.TIER_3  # Default for tree-sitter only languages
        if language in ("python", "typescript", "javascript", "tsx", "jsx"):
            tier = LanguageTier.TIER_1
        elif language in ("go", "rust", "java", "c", "cpp"):
            tier = LanguageTier.TIER_2

        result = CodeValidationResult(
            is_valid=True,
            language=language,
            tier=tier,
            validators_used=["tree_sitter"],
        )

        if not self._tree_sitter_available:
            result.warnings.append(
                ValidationIssue(
                    line=1,
                    column=0,
                    message="tree-sitter not available, validation skipped",
                    severity=ValidationSeverity.WARNING,
                    source="tree_sitter",
                )
            )
            return result

        if not config.check_syntax:
            return result

        try:
            parser = self._get_parser(language)
            if not parser:
                result.warnings.append(
                    ValidationIssue(
                        line=1,
                        column=0,
                        message=f"No tree-sitter grammar for {language}",
                        severity=ValidationSeverity.WARNING,
                        source="tree_sitter",
                    )
                )
                return result

            # Parse the code
            tree = parser.parse(bytes(code, "utf8"))

            # Collect errors
            errors = self._collect_errors(tree.root_node, config.max_errors)

            for error in errors:
                result.add_issue(
                    ValidationIssue(
                        line=error["line"],
                        column=error["column"],
                        end_line=error.get("end_line"),
                        end_column=error.get("end_column"),
                        message=error["message"],
                        severity=ValidationSeverity.ERROR,
                        source="tree_sitter",
                        code=error.get("code"),
                    )
                )

            if errors:
                result.is_valid = False
                logger.debug(
                    f"Tree-sitter found {len(errors)} errors in {file_path}"
                )
            else:
                logger.debug(f"Tree-sitter validation passed for {file_path}")

        except Exception as e:
            result.is_valid = False
            result.add_issue(
                ValidationIssue(
                    line=1,
                    column=0,
                    message=f"Parse error: {e}",
                    severity=ValidationSeverity.ERROR,
                    source="tree_sitter",
                )
            )
            logger.warning(f"Tree-sitter validation failed for {file_path}: {e}")

        return result

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

        except ImportError:
            logger.debug(f"tree-sitter grammar not installed for {language}")
            return None
        except Exception as e:
            logger.debug(f"Failed to load tree-sitter language: {e}")
            return None

    def _collect_errors(
        self,
        node: Any,
        max_errors: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Collect error nodes from parse tree.

        Args:
            node: Root node of parse tree
            max_errors: Maximum errors to collect (0 = unlimited)

        Returns:
            List of error dictionaries
        """
        errors: List[Dict[str, Any]] = []
        self._visit_errors(node, errors, max_errors)
        return errors

    def _visit_errors(
        self,
        node: Any,
        errors: List[Dict[str, Any]],
        max_errors: int,
    ) -> bool:
        """
        Visit nodes and collect errors.

        Returns:
            True if should continue collecting, False if max reached
        """
        if max_errors > 0 and len(errors) >= max_errors:
            return False

        if node.type == "ERROR":
            # Get the text around the error for context
            error_text = node.text.decode("utf8") if node.text else ""
            if len(error_text) > 50:
                error_text = error_text[:47] + "..."

            errors.append({
                "line": node.start_point[0] + 1,
                "column": node.start_point[1],
                "end_line": node.end_point[0] + 1,
                "end_column": node.end_point[1],
                "message": f"Syntax error: {error_text}" if error_text else "Syntax error",
                "code": "TS0001",
            })

        elif node.is_missing:
            errors.append({
                "line": node.start_point[0] + 1,
                "column": node.start_point[1],
                "message": f"Missing: {node.type}",
                "code": "TS0002",
            })

        # Visit children
        for child in node.children:
            if not self._visit_errors(child, errors, max_errors):
                return False

        return True

    def has_errors(self, code: str, language: str) -> bool:
        """
        Quick check if code has syntax errors.

        Args:
            code: Source code to check
            language: Language identifier

        Returns:
            True if code has syntax errors
        """
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

    def get_supported_languages(self) -> List[str]:
        """Get list of languages with known tree-sitter support."""
        return list(LANGUAGE_MODULES.keys())

    def is_language_available(self, language: str) -> bool:
        """Check if tree-sitter grammar is available for a language."""
        if not self._tree_sitter_available:
            return False
        return self._get_language(language) is not None
