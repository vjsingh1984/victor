"""Tests for contrib vertical deprecation warnings.

After P1-3, deprecation warnings are deferred from module-level to
register() to avoid firing during entry-point discovery scans.
These tests verify the warnings fire during plugin registration.
"""

import warnings

import pytest

CONTRIB_VERTICALS = [
    "victor.verticals.contrib.coding",
    "victor.verticals.contrib.rag",
    "victor.verticals.contrib.devops",
    "victor.verticals.contrib.dataanalysis",
    "victor.verticals.contrib.research",
]


@pytest.mark.parametrize("module_path", CONTRIB_VERTICALS)
def test_contrib_register_emits_deprecation_warning(module_path):
    """Calling register() on a contrib plugin should emit a DeprecationWarning.

    When no external package is installed, register() should fire the warning.
    We mock external_package_installed to simulate this scenario.
    """
    import importlib
    from unittest.mock import MagicMock, patch

    mod = importlib.import_module(module_path)
    plugin = getattr(mod, "plugin", None)
    assert plugin is not None, f"{module_path} should export a 'plugin' instance"

    ctx = MagicMock()

    # Simulate no external package installed so register() proceeds to the warning
    with patch("victor.verticals.contrib._compat.external_package_installed", return_value=False):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plugin.register(ctx)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert (
                len(deprecation_warnings) >= 1
            ), f"Expected DeprecationWarning from {module_path}.register(), got: {[x.message for x in w]}"

            # Verify the message mentions the package name
            vertical_name = module_path.split(".")[-1]
            msg_text = str(deprecation_warnings[0].message)
            assert "deprecated" in msg_text.lower()
            assert vertical_name in msg_text


@pytest.mark.parametrize("module_path", CONTRIB_VERTICALS)
def test_contrib_register_skips_when_external_installed(module_path):
    """When external package is installed, register() should skip without warning."""
    import importlib
    from unittest.mock import MagicMock, patch

    mod = importlib.import_module(module_path)
    plugin = getattr(mod, "plugin", None)
    assert plugin is not None

    ctx = MagicMock()

    with patch("victor.verticals.contrib._compat.external_package_installed", return_value=True):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plugin.register(ctx)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0, (
                f"Expected no DeprecationWarning when external package is installed, "
                f"got: {[str(x.message) for x in deprecation_warnings]}"
            )
            # Should not have registered anything
            ctx.register_vertical.assert_not_called()


@pytest.mark.parametrize("module_path", CONTRIB_VERTICALS)
def test_contrib_module_importable(module_path):
    """Contrib vertical modules should be importable without error."""
    import importlib

    mod = importlib.import_module(module_path)
    assert hasattr(mod, "plugin"), f"{module_path} should export a 'plugin' instance"
