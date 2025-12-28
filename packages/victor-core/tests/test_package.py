"""Basic package tests for victor-core."""

import pytest


class TestVictorCorePackage:
    """Test victor-core package basics."""

    def test_import(self):
        """Test that the package can be imported."""
        import victor

        assert hasattr(victor, "__version__")
        assert victor.__version__ == "0.3.0"

    def test_version_format(self):
        """Test version follows semver format."""
        import victor
        import re

        version = victor.__version__
        # Basic semver pattern: X.Y.Z
        assert re.match(r"^\d+\.\d+\.\d+", version)
