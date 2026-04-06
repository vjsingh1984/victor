from __future__ import annotations

import types

import pytest

import victor.ui.commands.codebase_support as codebase_support


def test_loader_prefers_extracted_victor_coding(monkeypatch: pytest.MonkeyPatch) -> None:
    preferred_module = types.SimpleNamespace(generate_smart_victor_md=lambda: "content")

    def fake_import_module(name: str) -> object:
        if name == "victor_coding.codebase_analyzer":
            return preferred_module
        if name == "victor.verticals.contrib.coding.codebase_analyzer":
            raise AssertionError("legacy analyzer path should not be loaded")
        raise ImportError(name)

    monkeypatch.setattr(codebase_support.importlib, "import_module", fake_import_module)

    assert codebase_support.load_codebase_analyzer_module() is preferred_module


def test_loader_falls_back_to_legacy_module(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy_module = types.SimpleNamespace(generate_smart_victor_md=lambda: "content")

    def fake_import_module(name: str) -> object:
        if name == "victor_coding.codebase_analyzer":
            raise ImportError(name)
        if name == "victor.verticals.contrib.coding.codebase_analyzer":
            return legacy_module
        raise ImportError(name)

    monkeypatch.setattr(codebase_support.importlib, "import_module", fake_import_module)

    assert codebase_support.load_codebase_analyzer_module() is legacy_module


def test_loader_raises_clear_error_when_analyzer_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        codebase_support.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )

    with pytest.raises(ImportError, match="victor-coding package"):
        codebase_support.load_codebase_analyzer_module()
