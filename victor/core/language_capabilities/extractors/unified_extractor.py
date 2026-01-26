"""
Unified language extractor for code indexing.

Uses the capability registry to select the best extraction method
for each language, falling back through the strategy chain.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from ..types import ASTAccessMethod, ExtractedSymbol
from .base import BaseLanguageProcessor
from .python_extractor import PythonASTExtractor
from .tree_sitter_extractor import TreeSitterExtractor

if TYPE_CHECKING:
    from ..registry import LanguageCapabilityRegistry

logger = logging.getLogger(__name__)


class UnifiedLanguageExtractor(BaseLanguageProcessor):
    """
    Unified extractor for code indexing.

    Uses the capability registry to select the best extraction method
    for each language. Falls back through the strategy chain if the
    preferred method is unavailable.

    Extraction Strategy:
    1. Try native AST (Python, Go via gopygo, etc.)
    2. Fall back to tree-sitter
    3. Optionally enrich with LSP information
    """

    def __init__(
        self,
        registry: Optional["LanguageCapabilityRegistry"] = None,
        python_extractor: Optional[PythonASTExtractor] = None,
        tree_sitter_extractor: Optional[TreeSitterExtractor] = None,
    ) -> None:
        """
        Initialize the unified extractor.

        Args:
            registry: Language capability registry (uses singleton if None)
            python_extractor: Python AST extractor (creates one if None)
            tree_sitter_extractor: Tree-sitter extractor (creates one if None)
        """
        super().__init__(registry)
        self._python_extractor = python_extractor or PythonASTExtractor()
        self._tree_sitter = tree_sitter_extractor or TreeSitterExtractor()

        # Native extractors by language
        self._native_extractors: Dict[str, object] = {
            "python": self._python_extractor,
        }

    def process(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
    ) -> List[ExtractedSymbol]:
        """
        Process code and extract symbols.

        Args:
            code: Source code to process
            file_path: Path to the source file
            language: Optional language override

        Returns:
            List of extracted symbols
        """
        return self.extract_symbols(code, file_path, language)

    def extract_symbols(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
    ) -> List[ExtractedSymbol]:
        """
        Extract symbols using best available method.

        Args:
            code: Source code to parse
            file_path: Path to the source file
            language: Optional language override

        Returns:
            List of extracted symbols
        """
        cap = self._get_capability(file_path, language)
        if not cap:
            logger.debug(f"No capability found for {file_path}")
            return []

        # Check if indexing is enabled
        if not cap.indexing_enabled:
            logger.debug(f"Indexing disabled for {cap.name}")
            return []

        # Try methods in strategy order
        for method in cap.indexing_strategy:
            if not cap._method_available(method):
                continue

            symbols = self._try_extraction(method, code, file_path, cap.name)
            if symbols:
                logger.debug(
                    f"Extracted {len(symbols)} symbols from {file_path} " f"using {method.value}"
                )
                return symbols

        logger.debug(f"No extraction method succeeded for {file_path}")
        return []

    def _try_extraction(
        self,
        method: ASTAccessMethod,
        code: str,
        file_path: Path,
        language: str,
    ) -> List[ExtractedSymbol]:
        """
        Try extraction with a specific method.

        Args:
            method: Extraction method to try
            code: Source code
            file_path: File path
            language: Language name

        Returns:
            List of symbols, or empty list if extraction failed
        """
        try:
            if method == ASTAccessMethod.NATIVE:
                return self._extract_native(code, file_path, language)
            elif method == ASTAccessMethod.PYTHON_LIB:
                return self._extract_python_lib(code, file_path, language)
            elif method == ASTAccessMethod.TREE_SITTER:
                return self._extract_tree_sitter(code, file_path, language)
            # LSP extraction would go here
            else:
                logger.debug(f"Extraction method {method.value} not implemented")
                return []

        except Exception as e:
            logger.warning(f"Extraction failed with {method.value} for {language}: {e}")
            return []

    def _extract_native(
        self,
        code: str,
        file_path: Path,
        language: str,
    ) -> List[ExtractedSymbol]:
        """Extract using native AST (Python only currently)."""
        if language == "python":
            return self._python_extractor.extract(code, file_path)
        return []

    def _extract_python_lib(
        self,
        code: str,
        file_path: Path,
        language: str,
    ) -> List[ExtractedSymbol]:
        """Extract using pure Python library (gopygo, javalang, etc.)."""
        # TODO: Implement for Go, Java when libraries are available
        logger.debug(f"Python lib extraction not implemented for {language}")
        return []

    def _extract_tree_sitter(
        self,
        code: str,
        file_path: Path,
        language: str,
    ) -> List[ExtractedSymbol]:
        """Extract using tree-sitter."""
        if not self._tree_sitter.is_available():
            return []
        return self._tree_sitter.extract(code, file_path, language)

    def can_extract(self, file_path: Path, language: Optional[str] = None) -> bool:
        """Check if extraction is supported for a file."""
        cap = self._get_capability(file_path, language)
        if not cap or not cap.indexing_enabled:
            return False
        return cap.get_best_indexing_method() is not None

    def get_extraction_method(
        self,
        file_path: Path,
        language: Optional[str] = None,
    ) -> Optional[ASTAccessMethod]:
        """Get the extraction method that would be used for a file."""
        cap = self._get_capability(file_path, language)
        if not cap or not cap.indexing_enabled:
            return None
        return cap.get_best_indexing_method()
