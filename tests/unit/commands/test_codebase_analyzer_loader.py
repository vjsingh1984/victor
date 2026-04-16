"""Tests for codebase analyzer discovery chain in coding_support.

Tests the public API: load_codebase_analyzer_module(),
load_codebase_analyzer_attr(), load_tree_sitter_get_parser().
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest

import victor.core.utils.capability_loader as coding_support


def test_analyzer_loader_prefers_extracted_victor_coding() -> None:
    """load_codebase_analyzer_module() uses importlib discovery."""
    mock_module = types.SimpleNamespace(generate_smart_victor_md=lambda: "content")

    with patch("importlib.import_module", return_value=mock_module):
        result = coding_support.load_codebase_analyzer_module()
    assert result is mock_module


def test_analyzer_loader_falls_back_to_legacy_module() -> None:
    """Falls back through discovery chain when primary import fails."""
    legacy_module = types.SimpleNamespace(generate_smart_victor_md=lambda: "legacy")
    call_count = 0

    def mock_import(name, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if "victor_coding.codebase_analyzer" in name:
            return legacy_module
        raise ImportError(f"No module named '{name}'")

    with patch("importlib.import_module", side_effect=mock_import):
        result = coding_support.load_codebase_analyzer_module()
    assert result is legacy_module


def test_analyzer_loader_raises_clear_error_when_unavailable() -> None:
    """Raises ImportError with helpful message when all paths fail."""
    with patch("importlib.import_module", side_effect=ImportError("not found")):
        with pytest.raises(ImportError, match="victor-coding"):
            coding_support.load_codebase_analyzer_module()


def test_analyzer_attr_loader_raises_clear_error_for_missing_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_codebase_analyzer_attr() raises ImportError for missing symbol."""
    module = types.SimpleNamespace(generate_smart_victor_md=lambda: "content")
    monkeypatch.setattr(coding_support, "load_codebase_analyzer_module", lambda: module)

    with pytest.raises(ImportError, match="required symbol 'generate_enhanced_init_md'"):
        coding_support.load_codebase_analyzer_attr("generate_enhanced_init_md")


def test_tree_sitter_loader_prefers_extracted_victor_coding() -> None:
    """load_tree_sitter_get_parser() returns get_parser callable."""
    mock_module = types.SimpleNamespace(get_parser=lambda lang: f"parser:{lang}")

    with patch("importlib.import_module", return_value=mock_module):
        get_parser = coding_support.load_tree_sitter_get_parser()
    assert get_parser("python") == "parser:python"


def test_tree_sitter_loader_falls_back_to_legacy_module() -> None:
    """Falls back through discovery chain for tree-sitter manager."""
    legacy_module = types.SimpleNamespace(get_parser=lambda lang: f"legacy:{lang}")
    call_count = 0

    def mock_import(name, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if "tree_sitter" in name:
            return legacy_module
        raise ImportError(f"No module named '{name}'")

    with patch("importlib.import_module", side_effect=mock_import):
        get_parser = coding_support.load_tree_sitter_get_parser()
    assert get_parser("python") == "legacy:python"


def test_tree_sitter_loader_raises_clear_error_when_unavailable() -> None:
    """Raises ImportError when tree-sitter manager not found."""
    with patch("importlib.import_module", side_effect=ImportError("not found")):
        with pytest.raises(ImportError, match="tree_sitter_manager requires"):
            coding_support.load_tree_sitter_get_parser()


def test_analyze_command_loader_prefers_extracted_victor_coding() -> None:
    """load_coding_analyze_app() returns analyze_app from module."""
    mock_app = object()
    mock_module = types.SimpleNamespace(app=mock_app)

    with patch.object(coding_support, "_try_entry_point", return_value=None):
        with patch.object(coding_support, "_try_import", return_value=mock_module):
            result = coding_support.load_coding_analyze_app()
    assert result is mock_app


def test_analyze_command_loader_falls_back_to_legacy_module() -> None:
    """Falls back through discovery chain for analyze command."""
    legacy_app = object()
    legacy_module = types.SimpleNamespace(app=legacy_app)

    call_count = 0

    def mock_try_import(name):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ImportError("primary not found")
        return legacy_module

    with patch.object(coding_support, "_try_entry_point", return_value=None):
        with patch.object(coding_support, "_try_import", side_effect=mock_try_import):
            result = coding_support.load_coding_analyze_app()
    assert result is legacy_app
