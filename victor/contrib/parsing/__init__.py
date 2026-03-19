"""Stub tree-sitter parsing providers for graceful degradation."""

from victor.contrib.parsing.parser import NullTreeSitterParser
from victor.contrib.parsing.extractor import NullTreeSitterExtractor

__all__ = ["NullTreeSitterParser", "NullTreeSitterExtractor"]
