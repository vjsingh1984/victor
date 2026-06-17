"""Stub tree-sitter parsing providers for graceful degradation."""

from victor.contrib.parsing.parser import NullTreeSitterParser
from victor.contrib.parsing.extractor import NullTreeSitterExtractor
from victor.contrib.parsing.analysis import NullTreeSitterAnalysis

__all__ = ["NullTreeSitterParser", "NullTreeSitterExtractor", "NullTreeSitterAnalysis"]
