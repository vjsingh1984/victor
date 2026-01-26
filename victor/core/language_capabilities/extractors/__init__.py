"""
Extractors for code indexing.

This module provides symbol extraction capabilities for different languages.

Native Extractors (with fallback to tree-sitter):
- PythonASTExtractor: Uses Python's built-in ast module
- GoExtractor: Uses gopygo (install: pip install gopygo)
- JavaExtractor: Uses javalang (install: pip install javalang)

Universal Fallback:
- TreeSitterExtractor: Supports 20+ languages via tree-sitter
"""

from .base import BaseLanguageProcessor
from .unified_extractor import UnifiedLanguageExtractor
from .python_extractor import PythonASTExtractor
from .tree_sitter_extractor import TreeSitterExtractor
from .go_extractor import GoExtractor
from .java_extractor import JavaExtractor
from .cpp_extractor import CppExtractor

__all__ = [
    "BaseLanguageProcessor",
    "UnifiedLanguageExtractor",
    "PythonASTExtractor",
    "TreeSitterExtractor",
    "GoExtractor",
    "JavaExtractor",
    "CppExtractor",
]
