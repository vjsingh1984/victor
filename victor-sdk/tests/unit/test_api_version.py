"""Tests for API version utilities."""

from victor_sdk.core.api_version import (
    CURRENT_API_VERSION,
    MIN_SUPPORTED_API_VERSION,
    is_compatible,
)


def test_current_version_is_2():
    assert CURRENT_API_VERSION == 2


def test_min_version_is_1():
    assert MIN_SUPPORTED_API_VERSION == 1


def test_is_compatible_current():
    assert is_compatible(CURRENT_API_VERSION) is True


def test_is_compatible_min():
    assert is_compatible(MIN_SUPPORTED_API_VERSION) is True


def test_is_compatible_below_min():
    assert is_compatible(0) is False


def test_is_compatible_above_current():
    assert is_compatible(CURRENT_API_VERSION + 1) is False


def test_sdk_exports():
    """Verify manifest types are importable from top-level SDK."""
    from victor_sdk import (
        ExtensionManifest,
        ExtensionType,
        CURRENT_API_VERSION as exported_ver,
        is_compatible as exported_fn,
    )

    assert exported_ver == 2
    assert exported_fn(1) is True
    assert ExtensionManifest is not None
    assert ExtensionType is not None
