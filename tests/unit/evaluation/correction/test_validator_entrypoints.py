# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Tests for entry-point validator discovery (FEP-0024 Phase 1).

External packages register language validators via the ``victor.code_validators``
Python entry-point group; ``CodeValidatorRegistry.discover_validators`` loads them
(preferred over the built-in path-scanned validators) with no edits to ``victor/``.
"""

import importlib.metadata
from types import SimpleNamespace

import pytest

from victor.evaluation.correction import CodeValidationResult, Language
from victor.evaluation.correction.base import BaseCodeValidator
from victor.evaluation.correction.orchestrator import SelfCorrector
from victor.evaluation.correction.registry import CodeValidatorRegistry


class _FakePythonValidator(BaseCodeValidator):
    """Test-only validator that reports PYTHON code invalid with a sentinel error.

    The built-in Python validator accepts ``x = 1`` as valid, so observing the
    sentinel proves the entry-point validator was used instead.
    """

    sentinel = "FEP0024_FAKE_PYTHON_ERROR"

    @property
    def supported_languages(self):
        return {Language.PYTHON}

    def validate(self, code: str) -> CodeValidationResult:
        return CodeValidationResult.failure([self.sentinel], language=Language.PYTHON)

    def fix(self, code: str, validation: CodeValidationResult) -> str:
        return code  # never fixes


def _entry_point(name: str, load):
    """Build a fake ``importlib.metadata.EntryPoint``-like object."""
    return SimpleNamespace(name=name, load=load)


@pytest.fixture
def patched_entry_points(monkeypatch):
    """Patch ``importlib.metadata.entry_points`` and reset the registry singleton.

    Each ``load`` argument is a zero-arg callable standing in for an entry point's
    ``.load()`` (the entry-point target: a ``BaseCodeValidator`` subclass per the
    FEP-0024 contract, or a callable that raises to simulate a broken entry point).
    """

    def _install(*loads) -> CodeValidatorRegistry:
        def _fake_entry_points(group=None):
            if group == "victor.code_validators":
                return [_entry_point(f"ep_{i}", load) for i, load in enumerate(loads)]
            return []

        monkeypatch.setattr(importlib.metadata, "entry_points", _fake_entry_points)
        CodeValidatorRegistry.reset_singleton()
        return CodeValidatorRegistry()

    return _install


def test_entry_point_validator_is_discovered_and_preferred(patched_entry_points):
    """An entry-point validator is discovered and overrides the built-in for its language."""
    registry = patched_entry_points(lambda: _FakePythonValidator)
    registry.discover_validators()

    validator = registry.get_validator(Language.PYTHON)
    assert isinstance(validator, _FakePythonValidator)


def test_broken_entry_point_is_skipped_not_raised(patched_entry_points):
    """A failing entry point is logged and skipped; other entry points still load."""

    def _boom():
        raise RuntimeError("broken entry point")

    registry = patched_entry_points(_boom, lambda: _FakePythonValidator)

    # Must not raise despite the broken entry point.
    registry.discover_validators()

    assert isinstance(registry.get_validator(Language.PYTHON), _FakePythonValidator)


def test_builtin_validators_still_discovered_without_entry_points(patched_entry_points):
    """With no entry points, the built-in path-scanned validators still register."""
    registry = patched_entry_points()  # no entry points
    registry.discover_validators()

    assert registry.has_validator(Language.PYTHON)


def test_self_corrector_routes_to_entry_point_validator(patched_entry_points):
    """Integration: SelfCorrector uses the entry-point validator end-to-end.

    Discovery runs at SelfCorrector init; the entry-point validator wins for PYTHON,
    so a snippet the built-in accepts is reported invalid with the sentinel error.
    """
    patched_entry_points(lambda: _FakePythonValidator)
    corrector = SelfCorrector()

    _fixed, validation = corrector.validate_and_fix("x = 1", Language.PYTHON)

    assert validation.valid is False
    assert _FakePythonValidator.sentinel in validation.errors
