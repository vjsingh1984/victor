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

"""Go language extractor using gopygo library.

This extractor uses gopygo (Pure Python Go parser) when available,
falling back to tree-sitter when gopygo is not installed.

Install gopygo with: pip install gopygo
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

from ..types import ExtractedSymbol
from .base import BaseLanguageProcessor
from .tree_sitter_extractor import TreeSitterExtractor

logger = logging.getLogger(__name__)

# Check if gopygo is available
try:
    import gopygo  # type: ignore[import-not-found]

    GOPYGO_AVAILABLE = True
except ImportError:
    GOPYGO_AVAILABLE = False
    logger.debug("gopygo not available, Go extraction will use tree-sitter")


class GoExtractor(BaseLanguageProcessor):
    """Go language extractor using gopygo.

    Falls back to tree-sitter when gopygo is not available.
    """

    def __init__(
        self,
        tree_sitter_extractor: Optional[TreeSitterExtractor] = None,
    ):
        """Initialize the Go extractor.

        Args:
            tree_sitter_extractor: Optional tree-sitter extractor for fallback
        """
        super().__init__()
        self._ts_extractor = tree_sitter_extractor or TreeSitterExtractor()

    def is_available(self) -> bool:
        """Check if native Go extraction is available."""
        return GOPYGO_AVAILABLE

    def extract(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
    ) -> List[ExtractedSymbol]:
        """Extract symbols from Go code.

        Args:
            code: Source code to parse
            file_path: Path to the source file
            language: Language identifier (ignored, always 'go')

        Returns:
            List of extracted symbols
        """
        if not GOPYGO_AVAILABLE:
            # Fallback to tree-sitter
            return self._ts_extractor.extract(code, file_path, "go")

        try:
            return self._extract_with_gopygo(code, file_path)
        except Exception as e:
            logger.warning(f"gopygo extraction failed for {file_path}: {e}")
            # Fallback to tree-sitter on error
            return self._ts_extractor.extract(code, file_path, "go")

    def _extract_with_gopygo(
        self,
        code: str,
        file_path: Path,
    ) -> List[ExtractedSymbol]:
        """Extract symbols using gopygo.

        Args:
            code: Go source code
            file_path: Path to the source file

        Returns:
            List of extracted symbols
        """
        if not GOPYGO_AVAILABLE:
            return []

        symbols: List[ExtractedSymbol] = []

        try:
            # Parse Go code
            tree = gopygo.parse(code)

            # Extract package declaration
            if hasattr(tree, "package") and tree.package:
                symbols.append(
                    ExtractedSymbol(
                        name=tree.package.name,
                        symbol_type="package",
                        file_path=str(file_path),
                        line_number=1,
                    )
                )

            # Extract imports
            if hasattr(tree, "imports"):
                for imp in tree.imports:
                    name = getattr(imp, "path", str(imp))
                    line = getattr(imp, "line", 1)
                    symbols.append(
                        ExtractedSymbol(
                            name=name,
                            symbol_type="import",
                            file_path=str(file_path),
                            line_number=line,
                        )
                    )

            # Extract type declarations (struct, interface)
            if hasattr(tree, "decls"):
                for decl in tree.decls:
                    self._extract_decl(decl, file_path, symbols)

        except Exception as e:
            logger.debug(f"gopygo parsing error: {e}")

        return symbols

    def _extract_decl(
        self,
        decl: Any,
        file_path: Path,
        symbols: List[ExtractedSymbol],
    ) -> None:
        """Extract symbols from a declaration."""
        decl_type = type(decl).__name__

        # Handle function declarations
        if decl_type == "FuncDecl":
            name = getattr(decl, "name", None)
            if name:
                line = getattr(decl, "line", 1)
                # Check if it's a method (has receiver)
                recv = getattr(decl, "recv", None)
                symbol_type = "method" if recv else "function"

                symbols.append(
                    ExtractedSymbol(
                        name=name,
                        symbol_type=symbol_type,
                        file_path=str(file_path),
                        line_number=line,
                    )
                )

        # Handle type declarations
        elif decl_type == "GenDecl":
            specs = getattr(decl, "specs", [])
            for spec in specs:
                spec_type = type(spec).__name__

                if spec_type == "TypeSpec":
                    name = getattr(spec, "name", None)
                    if name:
                        line = getattr(spec, "line", 1)
                        type_def = getattr(spec, "type", None)
                        type_def_name = type(type_def).__name__ if type_def else "type"

                        # Determine symbol type
                        if type_def_name == "StructType":
                            symbol_type = "struct"
                        elif type_def_name == "InterfaceType":
                            symbol_type = "interface"
                        else:
                            symbol_type = "type"

                        symbols.append(
                            ExtractedSymbol(
                                name=name,
                                symbol_type=symbol_type,
                                file_path=str(file_path),
                                line_number=line,
                            )
                        )

                elif spec_type == "ValueSpec":
                    # Variable or constant declarations
                    names = getattr(spec, "names", [])
                    line = getattr(spec, "line", 1)
                    for name in names:
                        if hasattr(name, "name"):
                            name = name.name
                        symbols.append(
                            ExtractedSymbol(
                                name=name,
                                symbol_type="variable",
                                file_path=str(file_path),
                                line_number=line,
                            )
                        )

    def has_syntax_errors(self, code: str) -> bool:
        """Check if Go code has syntax errors.

        Args:
            code: Go source code

        Returns:
            True if code has syntax errors
        """
        if not GOPYGO_AVAILABLE:
            return self._ts_extractor.has_syntax_errors(code, "go")

        try:
            gopygo.parse(code)
            return False
        except Exception:
            return True

    def process(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
    ) -> List[ExtractedSymbol]:
        """Process code and return results (BaseLanguageProcessor interface)."""
        return self.extract(code, file_path, language)
