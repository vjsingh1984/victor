from __future__ import annotations

import types

import pytest

import victor.ui.commands.codebase_support as codebase_support


def test_loader_prefers_extracted_victor_coding(monkeypatch: pytest.MonkeyPatch) -> None:
    preferred_module = types.SimpleNamespace(generate_smart_victor_md=lambda: "content")

    monkeypatch.setattr(
        codebase_support,
        "_load_extracted_codebase_analyzer",
        lambda: preferred_module,
    )

    def fail_if_legacy_called() -> object:
        raise AssertionError("legacy analyzer path should not be loaded")

    monkeypatch.setattr(
        codebase_support,
        "_load_legacy_codebase_analyzer",
        fail_if_legacy_called,
    )
    monkeypatch.setattr(
        codebase_support,
        "_CODEBASE_ANALYZER_LOADERS",
        (
            codebase_support._load_extracted_codebase_analyzer,
            codebase_support._load_legacy_codebase_analyzer,
        ),
    )

    assert codebase_support.load_codebase_analyzer_module() is preferred_module


def test_loader_falls_back_to_legacy_module(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy_module = types.SimpleNamespace(generate_smart_victor_md=lambda: "content")

    monkeypatch.setattr(
        codebase_support,
        "_load_extracted_codebase_analyzer",
        lambda: (_ for _ in ()).throw(ImportError("victor_coding.codebase_analyzer")),
    )
    monkeypatch.setattr(
        codebase_support,
        "_load_legacy_codebase_analyzer",
        lambda: legacy_module,
    )
    monkeypatch.setattr(
        codebase_support,
        "_CODEBASE_ANALYZER_LOADERS",
        (
            codebase_support._load_extracted_codebase_analyzer,
            codebase_support._load_legacy_codebase_analyzer,
        ),
    )

    assert codebase_support.load_codebase_analyzer_module() is legacy_module


def test_loader_raises_clear_error_when_analyzer_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        codebase_support,
        "_load_extracted_codebase_analyzer",
        lambda: (_ for _ in ()).throw(ImportError("victor_coding.codebase_analyzer")),
    )
    monkeypatch.setattr(
        codebase_support,
        "_load_legacy_codebase_analyzer",
        lambda: (_ for _ in ()).throw(
            ImportError("victor.verticals.contrib.coding.codebase_analyzer")
        ),
    )
    monkeypatch.setattr(
        codebase_support,
        "_CODEBASE_ANALYZER_LOADERS",
        (
            codebase_support._load_extracted_codebase_analyzer,
            codebase_support._load_legacy_codebase_analyzer,
        ),
    )

    with pytest.raises(ImportError, match="victor-coding package"):
        codebase_support.load_codebase_analyzer_module()
