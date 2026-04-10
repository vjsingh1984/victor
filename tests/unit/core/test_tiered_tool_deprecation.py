"""Tests for TieredToolConfig deprecation warnings."""

import warnings

from victor.core.vertical_types import TieredToolConfig


class TestTieredToolConfigDeprecation:
    """Verify deprecation warnings for deprecated TieredToolConfig fields."""

    def setup_method(self):
        TieredToolConfig.reset_deprecation_warnings()

    def teardown_method(self):
        TieredToolConfig.reset_deprecation_warnings()

    def test_semantic_pool_warns(self):
        """Setting semantic_pool should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TieredToolConfig(semantic_pool={"tool_a", "tool_b"})

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "semantic_pool" in str(deprecation_warnings[0].message)

    def test_stage_tools_warns(self):
        """Setting stage_tools should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TieredToolConfig(stage_tools={"INITIAL": {"read"}})

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "stage_tools" in str(deprecation_warnings[0].message)

    def test_no_warning_when_empty(self):
        """No warning when both fields are empty (default)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TieredToolConfig(mandatory={"read"})

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0

    def test_warn_once_behavior(self):
        """Warning should only be emitted once per field."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TieredToolConfig(semantic_pool={"a"})
            TieredToolConfig(semantic_pool={"b"})

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1

    def test_reset_allows_rewarn(self):
        """reset_deprecation_warnings() should allow re-emission."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TieredToolConfig(semantic_pool={"a"})

        TieredToolConfig.reset_deprecation_warnings()

        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            TieredToolConfig(semantic_pool={"b"})

        deprecation_warnings = [x for x in w2 if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
