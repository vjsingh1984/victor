"""Tests for the analysis-level tree-sitter capability contract."""

from __future__ import annotations

from victor.core.capability_registry import CapabilityRegistry
from victor.core.bootstrap import bootstrap_capabilities
from victor.contrib.parsing.analysis import NullTreeSitterAnalysis
from victor.framework.vertical_protocols import TreeSitterAnalysisProtocol


def test_tree_sitter_analysis_protocol_shape() -> None:
    """The framework protocol exposes analysis-level operations."""
    assert hasattr(TreeSitterAnalysisProtocol, "supports_language")
    assert hasattr(TreeSitterAnalysisProtocol, "parse")
    assert hasattr(TreeSitterAnalysisProtocol, "extract_symbols")
    assert hasattr(TreeSitterAnalysisProtocol, "extract_edges")
    assert hasattr(TreeSitterAnalysisProtocol, "extract_imports")
    assert hasattr(TreeSitterAnalysisProtocol, "build_chunk_context")


def test_null_tree_sitter_analysis_degrades_gracefully() -> None:
    """The stub returns empty/None results without raising."""
    provider = NullTreeSitterAnalysis()

    assert provider.supports_language("python") is False
    assert provider.parse(b"def foo(): pass\n", "python", file_path="a.py") is None
    assert provider.extract_symbols(b"def foo(): pass\n", "python", file_path="a.py") == []
    assert provider.extract_edges(b"def foo(): pass\n", "python", file_path="a.py") == []
    assert provider.extract_imports(b"import os\n", "python", file_path="a.py") == []
    assert provider.build_chunk_context("def foo(): pass\n", "python", file_path="a.py") is None


def test_bootstrap_registers_tree_sitter_analysis_stub() -> None:
    """Capability bootstrap registers the analysis-level stub."""
    CapabilityRegistry.reset()
    try:
        bootstrap_capabilities()
        registry = CapabilityRegistry.get_instance()
        provider = registry.get(TreeSitterAnalysisProtocol)

        assert provider is not None
        assert isinstance(provider, NullTreeSitterAnalysis)
        assert not registry.is_enhanced(TreeSitterAnalysisProtocol)
    finally:
        CapabilityRegistry.reset()
