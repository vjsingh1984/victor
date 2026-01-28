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

"""C/C++ language extractor using libclang.

This extractor uses libclang (Python bindings for Clang) when available,
falling back to tree-sitter when libclang is not installed.

libclang provides:
- Full type information
- Error recovery
- Semantic analysis
- Cross-file references

Install libclang with: pip install libclang
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from ..types import ExtractedSymbol
from .base import BaseLanguageProcessor
from .tree_sitter_extractor import TreeSitterExtractor

logger = logging.getLogger(__name__)

# Check if libclang is available
try:
    import clang.cindex as cindex  # type: ignore[import-not-found]

    LIBCLANG_AVAILABLE = True
except ImportError:
    LIBCLANG_AVAILABLE = False
    cindex = None
    logger.debug("libclang not available, C/C++ extraction will use tree-sitter")


class CppExtractor(BaseLanguageProcessor):
    """C/C++ language extractor using libclang.

    Falls back to tree-sitter when libclang is not available.

    libclang provides excellent support for C/C++ including:
    - Full type information and inference
    - Error recovery (partial AST on invalid code)
    - Semantic analysis
    - Cross-translation-unit analysis
    """

    # Cursor kinds that represent symbol definitions
    SYMBOL_KINDS = None

    def __init__(
        self,
        tree_sitter_extractor: Optional[TreeSitterExtractor] = None,
        compilation_database: Optional[str] = None,
    ):
        """Initialize the C/C++ extractor.

        Args:
            tree_sitter_extractor: Optional tree-sitter extractor for fallback
            compilation_database: Optional path to compile_commands.json
        """
        super().__init__()
        self._ts_extractor = tree_sitter_extractor or TreeSitterExtractor()
        self._compilation_database = compilation_database
        self._index = None

        if LIBCLANG_AVAILABLE:
            self._init_libclang()

    def _init_libclang(self) -> None:
        """Initialize libclang index."""
        if not LIBCLANG_AVAILABLE:
            return

        try:
            self._index = cindex.Index.create()

            # Define symbol kinds mapping
            self.__class__.SYMBOL_KINDS = {
                cindex.CursorKind.FUNCTION_DECL: "function",
                cindex.CursorKind.CXX_METHOD: "method",
                cindex.CursorKind.CONSTRUCTOR: "constructor",
                cindex.CursorKind.DESTRUCTOR: "destructor",
                cindex.CursorKind.CLASS_DECL: "class",
                cindex.CursorKind.STRUCT_DECL: "struct",
                cindex.CursorKind.ENUM_DECL: "enum",
                cindex.CursorKind.UNION_DECL: "union",
                cindex.CursorKind.TYPEDEF_DECL: "typedef",
                cindex.CursorKind.NAMESPACE: "namespace",
                cindex.CursorKind.VAR_DECL: "variable",
                cindex.CursorKind.FIELD_DECL: "field",
                cindex.CursorKind.ENUM_CONSTANT_DECL: "enum_constant",
                cindex.CursorKind.FUNCTION_TEMPLATE: "function_template",
                cindex.CursorKind.CLASS_TEMPLATE: "class_template",
            }
        except Exception as e:
            logger.warning(f"Failed to initialize libclang: {e}")
            self._index = None

    def is_available(self) -> bool:
        """Check if native C/C++ extraction is available."""
        return LIBCLANG_AVAILABLE and self._index is not None

    def extract(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
    ) -> List[ExtractedSymbol]:
        """Extract symbols from C/C++ code.

        Args:
            code: Source code to parse
            file_path: Path to the source file
            language: Language identifier ('c' or 'cpp')

        Returns:
            List of extracted symbols
        """
        # Determine language from extension if not provided
        if language is None:
            ext = file_path.suffix.lower()
            language = "cpp" if ext in [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"] else "c"

        if not self.is_available():
            # Fallback to tree-sitter
            return self._ts_extractor.extract(code, file_path, language)

        try:
            return self._extract_with_libclang(code, file_path, language)
        except Exception as e:
            logger.warning(f"libclang extraction failed for {file_path}: {e}")
            # Fallback to tree-sitter on error
            return self._ts_extractor.extract(code, file_path, language)

    def _extract_with_libclang(
        self,
        code: str,
        file_path: Path,
        language: str,
    ) -> List[ExtractedSymbol]:
        """Extract symbols using libclang.

        Args:
            code: C/C++ source code
            file_path: Path to the source file
            language: 'c' or 'cpp'

        Returns:
            List of extracted symbols
        """
        if not self.is_available():
            return []

        # is_available() ensures self._index is not None, use cast for mypy
        index = cast(Any, self._index)

        symbols: List[ExtractedSymbol] = []

        # Parse options
        args = ["-x", "c++" if language == "cpp" else "c"]

        # Parse the code
        tu = index.parse(
            str(file_path),
            args=args,
            unsaved_files=[(str(file_path), code)],
            options=(
                cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                | cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES
            ),
        )

        # Extract symbols from AST
        self._extract_from_cursor(tu.cursor, file_path, symbols, str(file_path))

        return symbols

    def _extract_from_cursor(
        self,
        cursor: Any,
        file_path: Path,
        symbols: List[ExtractedSymbol],
        target_file: str,
        parent: Optional[str] = None,
    ) -> None:
        """Recursively extract symbols from AST cursor."""
        if not LIBCLANG_AVAILABLE or self.SYMBOL_KINDS is None:
            return

        # Only process symbols from the target file
        if cursor.location.file and str(cursor.location.file) != target_file:
            return

        # Check if this cursor represents a symbol definition
        if cursor.kind in self.SYMBOL_KINDS and cursor.is_definition():
            symbol_type = self.SYMBOL_KINDS[cursor.kind]
            name = cursor.spelling

            if name:  # Skip anonymous types
                symbols.append(
                    ExtractedSymbol(
                        name=name,
                        symbol_type=symbol_type,
                        file_path=str(file_path),
                        line_number=cursor.location.line,
                        end_line=cursor.extent.end.line if cursor.extent else None,
                        parent_symbol=parent,
                    )
                )

                # Update parent for nested symbols
                if symbol_type in ("class", "struct", "namespace"):
                    parent = name

        # Recurse into children
        for child in cursor.get_children():
            self._extract_from_cursor(child, file_path, symbols, target_file, parent)

    def has_syntax_errors(self, code: str, language: str = "cpp") -> bool:
        """Check if C/C++ code has syntax errors.

        Args:
            code: C/C++ source code
            language: 'c' or 'cpp'

        Returns:
            True if code has syntax errors
        """
        if not self.is_available():
            return self._ts_extractor.has_syntax_errors(code, language)

        # is_available() ensures self._index is not None, use cast for mypy
        index = cast(Any, self._index)

        args = ["-x", "c++" if language == "cpp" else "c"]

        tu = index.parse(
            "temp.cpp" if language == "cpp" else "temp.c",
            args=args,
            unsaved_files=[("temp.cpp" if language == "cpp" else "temp.c", code)],
        )

        # Check for errors in diagnostics
        for diag in tu.diagnostics:
            if diag.severity >= cindex.Diagnostic.Error:
                return True

        return False

    def get_diagnostics(
        self,
        code: str,
        file_path: Path,
        language: str = "cpp",
    ) -> List[Dict[str, Any]]:
        """Get diagnostics (errors/warnings) from C/C++ code.

        Args:
            code: C/C++ source code
            file_path: Path to the source file
            language: 'c' or 'cpp'

        Returns:
            List of diagnostic dicts with line, column, message, severity
        """
        if not self.is_available():
            return self._ts_extractor.get_error_locations(code, language)

        # is_available() ensures self._index is not None, use cast for mypy
        index = cast(Any, self._index)

        diagnostics: List[Dict[str, Any]] = []

        try:
            args = ["-x", "c++" if language == "cpp" else "c"]

            tu = index.parse(
                str(file_path),
                args=args,
                unsaved_files=[(str(file_path), code)],
            )

            severity_map = {
                cindex.Diagnostic.Ignored: "ignored",
                cindex.Diagnostic.Note: "note",
                cindex.Diagnostic.Warning: "warning",
                cindex.Diagnostic.Error: "error",
                cindex.Diagnostic.Fatal: "fatal",
            }

            for diag in tu.diagnostics:
                diagnostics.append(
                    {
                        "line": diag.location.line,
                        "column": diag.location.column,
                        "message": diag.spelling,
                        "severity": severity_map.get(diag.severity, "unknown"),
                    }
                )

        except Exception as e:
            diagnostics.append(
                {
                    "line": 1,
                    "column": 0,
                    "message": str(e),
                    "severity": "error",
                }
            )

        return diagnostics

    def process(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
    ) -> List[ExtractedSymbol]:
        """Process code and return results (BaseLanguageProcessor interface)."""
        return self.extract(code, file_path, language)
