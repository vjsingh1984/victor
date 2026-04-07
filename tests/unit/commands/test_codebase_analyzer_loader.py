from __future__ import annotations

import types

import pytest

import victor.core.utils.coding_support as coding_support


def test_analyzer_loader_prefers_extracted_victor_coding(monkeypatch: pytest.MonkeyPatch) -> None:
    preferred_module = types.SimpleNamespace(generate_smart_victor_md=lambda: "content")

    monkeypatch.setattr(
        coding_support,
        "_load_extracted_codebase_analyzer",
        lambda: preferred_module,
    )

    def fail_if_legacy_called() -> object:
        raise AssertionError("legacy analyzer path should not be loaded")

    monkeypatch.setattr(
        coding_support,
        "_load_legacy_codebase_analyzer",
        fail_if_legacy_called,
    )
    monkeypatch.setattr(
        coding_support,
        "_CODEBASE_ANALYZER_LOADERS",
        (
            coding_support._load_extracted_codebase_analyzer,
            coding_support._load_legacy_codebase_analyzer,
        ),
    )

    assert coding_support.load_codebase_analyzer_module() is preferred_module


def test_analyzer_loader_falls_back_to_legacy_module(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy_module = types.SimpleNamespace(generate_smart_victor_md=lambda: "content")

    monkeypatch.setattr(
        coding_support,
        "_load_extracted_codebase_analyzer",
        lambda: (_ for _ in ()).throw(ImportError("victor_coding.codebase_analyzer")),
    )
    monkeypatch.setattr(
        coding_support,
        "_load_legacy_codebase_analyzer",
        lambda: legacy_module,
    )
    monkeypatch.setattr(
        coding_support,
        "_CODEBASE_ANALYZER_LOADERS",
        (
            coding_support._load_extracted_codebase_analyzer,
            coding_support._load_legacy_codebase_analyzer,
        ),
    )

    assert coding_support.load_codebase_analyzer_module() is legacy_module


def test_analyzer_loader_raises_clear_error_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coding_support,
        "_load_extracted_codebase_analyzer",
        lambda: (_ for _ in ()).throw(ImportError("victor_coding.codebase_analyzer")),
    )
    monkeypatch.setattr(
        coding_support,
        "_load_legacy_codebase_analyzer",
        lambda: (_ for _ in ()).throw(
            ImportError("victor.verticals.contrib.coding.codebase_analyzer")
        ),
    )
    monkeypatch.setattr(
        coding_support,
        "_CODEBASE_ANALYZER_LOADERS",
        (
            coding_support._load_extracted_codebase_analyzer,
            coding_support._load_legacy_codebase_analyzer,
        ),
    )

    with pytest.raises(ImportError, match="victor-coding package"):
        coding_support.load_codebase_analyzer_module()


def test_analyzer_attr_loader_raises_clear_error_for_missing_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.SimpleNamespace(generate_smart_victor_md=lambda: "content")
    monkeypatch.setattr(coding_support, "load_codebase_analyzer_module", lambda: module)

    with pytest.raises(ImportError, match="required symbol 'generate_enhanced_init_md'"):
        coding_support.load_codebase_analyzer_attr("generate_enhanced_init_md")


def test_tree_sitter_loader_prefers_extracted_victor_coding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferred_module = types.SimpleNamespace(get_parser=lambda language: f"parser:{language}")

    monkeypatch.setattr(
        coding_support,
        "_load_extracted_tree_sitter_manager",
        lambda: preferred_module,
    )

    def fail_if_legacy_called() -> object:
        raise AssertionError("legacy tree-sitter path should not be loaded")

    monkeypatch.setattr(
        coding_support,
        "_load_legacy_tree_sitter_manager",
        fail_if_legacy_called,
    )
    monkeypatch.setattr(
        coding_support,
        "_TREE_SITTER_MANAGER_LOADERS",
        (
            coding_support._load_extracted_tree_sitter_manager,
            coding_support._load_legacy_tree_sitter_manager,
        ),
    )

    get_parser = coding_support.load_tree_sitter_get_parser()

    assert get_parser("python") == "parser:python"


def test_tree_sitter_loader_falls_back_to_legacy_module(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy_module = types.SimpleNamespace(get_parser=lambda language: f"legacy:{language}")

    monkeypatch.setattr(
        coding_support,
        "_load_extracted_tree_sitter_manager",
        lambda: (_ for _ in ()).throw(ImportError("victor_coding.codebase.tree_sitter_manager")),
    )
    monkeypatch.setattr(
        coding_support,
        "_load_legacy_tree_sitter_manager",
        lambda: legacy_module,
    )
    monkeypatch.setattr(
        coding_support,
        "_TREE_SITTER_MANAGER_LOADERS",
        (
            coding_support._load_extracted_tree_sitter_manager,
            coding_support._load_legacy_tree_sitter_manager,
        ),
    )

    get_parser = coding_support.load_tree_sitter_get_parser()

    assert get_parser("python") == "legacy:python"


def test_tree_sitter_loader_raises_clear_error_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coding_support,
        "_load_extracted_tree_sitter_manager",
        lambda: (_ for _ in ()).throw(ImportError("victor_coding.codebase.tree_sitter_manager")),
    )
    monkeypatch.setattr(
        coding_support,
        "_load_legacy_tree_sitter_manager",
        lambda: (_ for _ in ()).throw(
            ImportError("victor.verticals.contrib.coding.codebase.tree_sitter_manager")
        ),
    )
    monkeypatch.setattr(
        coding_support,
        "_TREE_SITTER_MANAGER_LOADERS",
        (
            coding_support._load_extracted_tree_sitter_manager,
            coding_support._load_legacy_tree_sitter_manager,
        ),
    )

    with pytest.raises(ImportError, match="tree_sitter_manager requires the victor-coding"):
        coding_support.load_tree_sitter_get_parser()


def test_analyze_command_loader_prefers_extracted_victor_coding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferred_app = object()

    monkeypatch.setattr(
        coding_support, "_load_extracted_analyze_command_app", lambda: preferred_app
    )

    def fail_if_legacy_called() -> object:
        raise AssertionError("legacy analyze command path should not be loaded")

    monkeypatch.setattr(coding_support, "_load_legacy_analyze_command_app", fail_if_legacy_called)
    monkeypatch.setattr(
        coding_support,
        "_ANALYZE_COMMAND_APP_LOADERS",
        (
            coding_support._load_extracted_analyze_command_app,
            coding_support._load_legacy_analyze_command_app,
        ),
    )

    assert coding_support.load_coding_analyze_app() is preferred_app


def test_analyze_command_loader_falls_back_to_legacy_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_app = object()

    monkeypatch.setattr(
        coding_support,
        "_load_extracted_analyze_command_app",
        lambda: (_ for _ in ()).throw(ImportError("victor_coding.commands.analyze")),
    )
    monkeypatch.setattr(coding_support, "_load_legacy_analyze_command_app", lambda: legacy_app)
    monkeypatch.setattr(
        coding_support,
        "_ANALYZE_COMMAND_APP_LOADERS",
        (
            coding_support._load_extracted_analyze_command_app,
            coding_support._load_legacy_analyze_command_app,
        ),
    )

    assert coding_support.load_coding_analyze_app() is legacy_app
