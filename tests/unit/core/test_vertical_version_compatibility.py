"""Tests for shared Victor version compatibility helpers."""

from __future__ import annotations

from victor.core.verticals.package_schema import (
    is_victor_version_compatible,
    normalize_victor_requirement,
)
from victor.core.verticals.registry_manager import VerticalRegistryManager


def test_normalize_victor_requirement_adds_package_prefix() -> None:
    """Bare specifiers should be normalized with the victor-ai package name."""
    assert normalize_victor_requirement(">=0.5.0") == "victor-ai>=0.5.0"


def test_normalize_victor_requirement_preserves_full_requirement() -> None:
    """Already-qualified requirement strings should be preserved."""
    assert normalize_victor_requirement("victor-ai>=0.5.0") == "victor-ai>=0.5.0"


def test_is_victor_version_compatible_accepts_matching_version() -> None:
    """Compatibility should be true when current version satisfies requirement."""
    assert is_victor_version_compatible("0.6.0", ">=0.5.0") is True
    assert is_victor_version_compatible("0.6.0", "victor-ai>=0.5.0") is True


def test_is_victor_version_compatible_rejects_non_matching_version() -> None:
    """Compatibility should be false when current version is below minimum."""
    assert is_victor_version_compatible("0.4.9", ">=0.5.0") is False
    assert is_victor_version_compatible("0.4.9", "victor-ai>=0.5.0") is False


def test_registry_manager_uses_shared_version_compatibility_helper() -> None:
    """RegistryManager compatibility checks should align with shared helper behavior."""
    manager = VerticalRegistryManager()

    assert manager._check_version_compatibility("0.6.0", ">=0.5.0") is True
    assert manager._check_version_compatibility("0.4.9", ">=0.5.0") is False
