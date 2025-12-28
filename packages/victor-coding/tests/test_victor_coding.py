"""Basic package tests for victor-coding."""

import pytest


class TestVictorCodingPackage:
    """Test victor-coding package basics."""

    def test_import(self):
        """Test that the package can be imported."""
        import victor_coding

        assert hasattr(victor_coding, "__version__")
        assert victor_coding.__version__ == "0.1.0"

    def test_version_format(self):
        """Test version follows semver format."""
        import victor_coding
        import re

        version = victor_coding.__version__
        # Basic semver pattern: X.Y.Z
        assert re.match(r"^\d+\.\d+\.\d+", version)

    def test_vertical_lazy_import(self):
        """Test that CodingVertical can be accessed via lazy import."""
        import victor_coding

        # This should work via __getattr__
        vertical_cls = victor_coding.CodingVertical
        assert vertical_cls is not None
        assert vertical_cls.name == "coding"

    def test_vertical_instantiation(self):
        """Test that CodingVertical can be instantiated."""
        from victor_coding import CodingVertical

        vertical = CodingVertical()
        assert vertical.get_name() == "coding"
        assert len(vertical.get_tools()) == 25

    def test_tools_module(self):
        """Test that tools module exists."""
        from victor_coding import tools

        assert hasattr(tools, "__version__")
        assert hasattr(tools, "get_all_coding_tools")
