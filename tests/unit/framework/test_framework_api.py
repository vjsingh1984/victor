"""Tests for framework __init__.py and _api.py public API surface."""

import pytest

from victor.framework._api import PUBLIC_API_NAMES


class TestFrameworkAPI:
    """Verify the framework public API surface."""

    def test_all_public_names_importable(self):
        """All PUBLIC_API_NAMES should be importable from victor.framework."""
        import victor.framework as fw

        missing = []
        for name in PUBLIC_API_NAMES:
            if not hasattr(fw, name):
                missing.append(name)
        assert not missing, f"Missing from victor.framework: {missing}"

    def test_lazy_imports_still_work(self):
        """Lazy imports should still be accessible via __getattr__."""
        import victor.framework as fw

        # These are lazy-loaded, not in PUBLIC_API_NAMES
        assert hasattr(fw, "StateGraph")
        assert hasattr(fw, "CircuitBreaker")

    def test_all_is_superset_of_public_names(self):
        """__all__ should be a superset of PUBLIC_API_NAMES."""
        import victor.framework as fw

        all_set = set(fw.__all__)
        public_set = set(PUBLIC_API_NAMES)
        missing = public_set - all_set
        assert not missing, f"PUBLIC_API_NAMES not in __all__: {missing}"

    def test_public_api_names_subset_of_core_names(self):
        """PUBLIC_API_NAMES should be a subset of _CORE_NAMES in framework."""
        import victor.framework as fw

        assert set(PUBLIC_API_NAMES).issubset(set(fw._CORE_NAMES))

    def test_discover_function_available(self):
        """The discover() function should be importable."""
        from victor.framework import discover

        assert callable(discover)

    def test_frameworkshim_is_not_part_of_eager_public_api_names(self):
        """FrameworkShim should no longer be an eager core export."""
        assert "FrameworkShim" not in PUBLIC_API_NAMES

    def test_frameworkshim_top_level_import_remains_warning_backed_compat(self):
        """Top-level FrameworkShim access should still work with a warning."""
        import victor.framework as fw
        from victor.framework.shim import FrameworkShim as ShimFrameworkShim

        fw.__dict__.pop("FrameworkShim", None)

        with pytest.warns(DeprecationWarning, match="victor\\.framework\\.FrameworkShim") as caught:
            alias = fw.FrameworkShim

        assert alias is ShimFrameworkShim
        message = str(caught[0].message)
        assert "v1.0.0" in message
        assert "2027-06-30" in message
        assert "docs/architecture/migration.md" in message
