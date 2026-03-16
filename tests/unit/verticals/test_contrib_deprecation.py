"""Tests for contrib vertical deprecation warnings."""

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
def test_contrib_import_emits_deprecation_warning(module_path):
    """Importing a contrib vertical should emit a DeprecationWarning."""
    import importlib
    import sys

    # Remove from sys.modules to force re-import
    # (modules may already be cached from other tests)
    modules_to_clear = [k for k in sys.modules if k.startswith(module_path)]
    saved = {}
    for key in modules_to_clear:
        saved[key] = sys.modules.pop(key)

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            importlib.import_module(module_path)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1, (
                f"Expected DeprecationWarning from {module_path}, got: {[x.message for x in w]}"
            )

            # Verify the message mentions the package name
            vertical_name = module_path.split(".")[-1]
            msg_text = str(deprecation_warnings[0].message)
            assert "deprecated" in msg_text.lower()
            assert vertical_name in msg_text
    finally:
        # Restore modules
        for key, mod in saved.items():
            sys.modules[key] = mod
