"""Tests for StandardGroundingRules registry (core purge — no hardcoded addendums)."""

from victor.core.vertical_types import StandardGroundingRules


class TestGroundingRulesRegistry:
    """Tests for grounding addendum registration and lookup."""

    def setup_method(self):
        """Reset registry before each test."""
        StandardGroundingRules._grounding_addendums.clear()

    # --- empty by default after core purge ---

    def test_no_hardcoded_addendums_in_core(self):
        """After core purge, no vertical-specific addendums in core."""
        assert len(StandardGroundingRules._grounding_addendums) == 0

    def test_for_vertical_unknown_returns_base(self):
        """Unknown vertical returns base rules only."""
        result = StandardGroundingRules.for_vertical("anything")
        assert result == StandardGroundingRules.BASE

    # --- register / unregister ---

    def test_register_addendum(self):
        """Verticals register their own addendums at runtime."""
        StandardGroundingRules.register_addendum(
            "research", "Cite URLs for claims."
        )
        result = StandardGroundingRules.for_vertical("research")
        assert StandardGroundingRules.BASE in result
        assert "Cite URLs" in result

    def test_register_multiple_verticals(self):
        """Multiple verticals can register independently."""
        StandardGroundingRules.register_addendum("a", "Rule A")
        StandardGroundingRules.register_addendum("b", "Rule B")
        assert "Rule A" in StandardGroundingRules.for_vertical("a")
        assert "Rule B" in StandardGroundingRules.for_vertical("b")

    def test_unregister_addendum(self):
        """Unregistering removes the addendum."""
        StandardGroundingRules.register_addendum("sec", "Check perms.")
        StandardGroundingRules.unregister_addendum("sec")
        result = StandardGroundingRules.for_vertical("sec")
        assert result == StandardGroundingRules.BASE

    def test_register_overrides_existing(self):
        """Re-registering replaces the old addendum."""
        StandardGroundingRules.register_addendum("x", "Old rule")
        StandardGroundingRules.register_addendum("x", "New rule")
        result = StandardGroundingRules.for_vertical("x")
        assert "New rule" in result
        assert "Old rule" not in result

    def test_unregister_nonexistent_is_noop(self):
        """Unregistering a never-registered vertical does not raise."""
        StandardGroundingRules.unregister_addendum("ghost")

    def test_for_vertical_extended_mode(self):
        """Extended mode uses EXTENDED base instead of BASE."""
        StandardGroundingRules.register_addendum("v", "Extra rule")
        result = StandardGroundingRules.for_vertical("v", extended=True)
        assert StandardGroundingRules.EXTENDED in result
        assert "Extra rule" in result

    def test_get_base_default(self):
        """get_base() returns BASE by default."""
        assert StandardGroundingRules.get_base() == StandardGroundingRules.BASE

    def test_get_base_extended(self):
        """get_base(extended=True) returns EXTENDED."""
        assert (
            StandardGroundingRules.get_base(extended=True)
            == StandardGroundingRules.EXTENDED
        )
