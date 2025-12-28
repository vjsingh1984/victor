"""Basic package tests for victor-ai meta-package."""

import pytest


class TestVictorAIPackage:
    """Test victor-ai meta-package basics."""

    def test_compat_import(self):
        """Test that the compatibility module can be imported."""
        import victor_compat

        assert hasattr(victor_compat, "__version__")
        assert victor_compat.__version__ == "0.3.0"

    def test_deprecation_helper(self):
        """Test deprecation warning helper exists."""
        from victor_compat import _deprecation_warning

        # Should not raise
        assert callable(_deprecation_warning)
