"""Tests for StandardGroundingRules decoupling (registry-based addendums)."""

from victor.core.vertical_types import StandardGroundingRules


class TestGroundingRulesRegistry:
    """Tests for grounding addendum registration and lookup."""

    def setup_method(self):
        """Reset the registry to defaults before each test."""
        StandardGroundingRules._grounding_addendums.clear()
        StandardGroundingRules._register_defaults()

    # --- backward compatibility ---

    def test_defaults_registered_at_import(self):
        """Built-in addendums should be present after _register_defaults."""
        assert "research" in StandardGroundingRules._grounding_addendums
        assert "data_analysis" in StandardGroundingRules._grounding_addendums
        assert "devops" in StandardGroundingRules._grounding_addendums

    def test_for_vertical_research_backward_compat(self):
        """for_vertical('research') returns base + research addendum."""
        result = StandardGroundingRules.for_vertical("research")
        assert StandardGroundingRules.BASE in result
        assert StandardGroundingRules.RESEARCH_ADDENDUM in result

    def test_for_vertical_data_analysis_backward_compat(self):
        """for_vertical('data_analysis') returns base + data addendum."""
        result = StandardGroundingRules.for_vertical("data_analysis")
        assert StandardGroundingRules.DATA_ADDENDUM in result

    def test_for_vertical_devops_backward_compat(self):
        """for_vertical('devops') returns base + devops addendum."""
        result = StandardGroundingRules.for_vertical("devops")
        assert StandardGroundingRules.DEVOPS_ADDENDUM in result

    def test_for_vertical_unknown_returns_base_only(self):
        """Unknown vertical returns base rules without addendum."""
        result = StandardGroundingRules.for_vertical("unknown_vertical")
        assert result == StandardGroundingRules.BASE

    # --- register / unregister ---

    def test_register_custom_addendum(self):
        """External verticals can register their own addendum."""
        custom_text = "Always validate security constraints."
        StandardGroundingRules.register_addendum("security", custom_text)
        result = StandardGroundingRules.for_vertical("security")
        assert custom_text in result
        assert StandardGroundingRules.BASE in result

    def test_unregister_addendum(self):
        """Unregistering removes the addendum so for_vertical falls back to base."""
        StandardGroundingRules.unregister_addendum("research")
        result = StandardGroundingRules.for_vertical("research")
        assert result == StandardGroundingRules.BASE
        assert StandardGroundingRules.RESEARCH_ADDENDUM not in result

    def test_register_overrides_existing(self):
        """Re-registering the same vertical replaces the old addendum."""
        new_text = "New research rule."
        StandardGroundingRules.register_addendum("research", new_text)
        result = StandardGroundingRules.for_vertical("research")
        assert new_text in result
        assert StandardGroundingRules.RESEARCH_ADDENDUM not in result

    def test_unregister_nonexistent_is_noop(self):
        """Unregistering a vertical that was never registered does not raise."""
        StandardGroundingRules.unregister_addendum("nonexistent")
        # Should not raise
