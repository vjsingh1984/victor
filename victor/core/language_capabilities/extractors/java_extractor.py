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

"""Java language extractor using javalang library.

This extractor uses javalang (Pure Python Java parser) when available,
falling back to tree-sitter when javalang is not installed.

Note: javalang only supports Java 8 syntax. For Java 9+ features,
tree-sitter is used automatically.

Install javalang with: pip install javalang
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

from ..types import ExtractedSymbol
from .base import BaseLanguageProcessor
from .tree_sitter_extractor import TreeSitterExtractor

logger = logging.getLogger(__name__)

# Check if javalang is available
try:
    import javalang

    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False
    logger.debug("javalang not available, Java extraction will use tree-sitter")


class JavaExtractor(BaseLanguageProcessor):
    """Java language extractor using javalang.

    Falls back to tree-sitter when javalang is not available.

    Note: javalang only supports Java 8 syntax. Code using Java 9+
    features (modules, var keyword, records, etc.) will fall back
    to tree-sitter extraction.
    """

    def __init__(
        self,
        tree_sitter_extractor: Optional[TreeSitterExtractor] = None,
    ):
        """Initialize the Java extractor.

        Args:
            tree_sitter_extractor: Optional tree-sitter extractor for fallback
        """
        super().__init__()
        self._ts_extractor = tree_sitter_extractor or TreeSitterExtractor()

    def is_available(self) -> bool:
        """Check if native Java extraction is available."""
        return JAVALANG_AVAILABLE

    def extract(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
    ) -> List[ExtractedSymbol]:
        """Extract symbols from Java code.

        Args:
            code: Source code to parse
            file_path: Path to the source file
            language: Language identifier (ignored, always 'java')

        Returns:
            List of extracted symbols
        """
        if not JAVALANG_AVAILABLE:
            # Fallback to tree-sitter
            return self._ts_extractor.extract(code, file_path, "java")

        try:
            return self._extract_with_javalang(code, file_path)
        except Exception as e:
            logger.warning(f"javalang extraction failed for {file_path}: {e}")
            # Fallback to tree-sitter on error (e.g., Java 9+ features)
            return self._ts_extractor.extract(code, file_path, "java")

    def _extract_with_javalang(
        self,
        code: str,
        file_path: Path,
    ) -> List[ExtractedSymbol]:
        """Extract symbols using javalang.

        Args:
            code: Java source code
            file_path: Path to the source file

        Returns:
            List of extracted symbols
        """
        if not JAVALANG_AVAILABLE:
            return []

        symbols: List[ExtractedSymbol] = []

        try:
            # Parse Java code
            tree = javalang.parse.parse(code)

            # Extract package declaration
            if tree.package:
                symbols.append(
                    ExtractedSymbol(
                        name=tree.package.name,
                        symbol_type="package",
                        file_path=str(file_path),
                        line_number=1,
                    )
                )

            # Extract imports
            for imp in tree.imports:
                symbols.append(
                    ExtractedSymbol(
                        name=imp.path,
                        symbol_type="import",
                        file_path=str(file_path),
                        line_number=(
                            getattr(imp, "position", (1, 1))[0] if hasattr(imp, "position") else 1
                        ),
                    )
                )

            # Extract types (classes, interfaces, enums)
            for path, node in tree.filter(javalang.tree.TypeDeclaration):
                self._extract_type(node, file_path, symbols)

        except javalang.parser.JavaSyntaxError as e:
            logger.debug(f"javalang syntax error (may be Java 9+ code): {e}")
            # Fallback to tree-sitter for syntax errors (might be Java 9+ features)
            return self._ts_extractor.extract(code, file_path, "java")
        except Exception as e:
            logger.debug(f"javalang parsing error: {e}")

        return symbols

    def _extract_type(
        self,
        node: Any,
        file_path: Path,
        symbols: List[ExtractedSymbol],
        parent: Optional[str] = None,
    ) -> None:
        """Extract symbols from a type declaration."""
        node_type = type(node).__name__

        # Determine symbol type
        if node_type == "ClassDeclaration":
            symbol_type = "class"
        elif node_type == "InterfaceDeclaration":
            symbol_type = "interface"
        elif node_type == "EnumDeclaration":
            symbol_type = "enum"
        elif node_type == "AnnotationDeclaration":
            symbol_type = "annotation"
        else:
            symbol_type = "type"

        # Get position
        line = 1
        if hasattr(node, "position") and node.position:
            line = node.position[0]

        # Extract visibility
        if hasattr(node, "modifiers") and node.modifiers:
            if "public" in node.modifiers:
                pass
            elif "protected" in node.modifiers:
                pass
            elif "private" in node.modifiers:
                pass
            else:
                pass

        symbols.append(
            ExtractedSymbol(
                name=node.name,
                symbol_type=symbol_type,
                file_path=str(file_path),
                line_number=line,
                parent_symbol=parent,
            )
        )

        # Extract members (fields, methods, constructors)
        if hasattr(node, "body") and node.body:
            for member in node.body:
                self._extract_member(member, file_path, symbols, node.name)

        # Extract nested types
        if hasattr(node, "body") and node.body:
            for member in node.body:
                if isinstance(member, javalang.tree.TypeDeclaration):
                    self._extract_type(member, file_path, symbols, node.name)

    def _extract_member(
        self,
        member: Any,
        file_path: Path,
        symbols: List[ExtractedSymbol],
        parent: str,
    ) -> None:
        """Extract a class/interface member."""
        member_type = type(member).__name__

        if member_type == "MethodDeclaration":
            line = 1
            if hasattr(member, "position") and member.position:
                line = member.position[0]

            symbols.append(
                ExtractedSymbol(
                    name=member.name,
                    symbol_type="method",
                    file_path=str(file_path),
                    line_number=line,
                    parent_symbol=parent,
                )
            )

        elif member_type == "ConstructorDeclaration":
            line = 1
            if hasattr(member, "position") and member.position:
                line = member.position[0]

            symbols.append(
                ExtractedSymbol(
                    name=member.name,
                    symbol_type="constructor",
                    file_path=str(file_path),
                    line_number=line,
                    parent_symbol=parent,
                )
            )

        elif member_type == "FieldDeclaration":
            line = 1
            if hasattr(member, "position") and member.position:
                line = member.position[0]

            # Extract all declared fields
            for declarator in member.declarators:
                symbols.append(
                    ExtractedSymbol(
                        name=declarator.name,
                        symbol_type="field",
                        file_path=str(file_path),
                        line_number=line,
                        parent_symbol=parent,
                    )
                )

        elif member_type == "EnumConstantDeclaration":
            line = 1
            if hasattr(member, "position") and member.position:
                line = member.position[0]

            symbols.append(
                ExtractedSymbol(
                    name=member.name,
                    symbol_type="enum_constant",
                    file_path=str(file_path),
                    line_number=line,
                    parent_symbol=parent,
                )
            )

    def has_syntax_errors(self, code: str) -> bool:
        """Check if Java code has syntax errors.

        Args:
            code: Java source code

        Returns:
            True if code has syntax errors
        """
        if not JAVALANG_AVAILABLE:
            return self._ts_extractor.has_syntax_errors(code, "java")

        try:
            javalang.parse.parse(code)
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
