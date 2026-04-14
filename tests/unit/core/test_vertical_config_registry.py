"""TDD tests for dynamic vertical config registration.

Verifies that verticals can register tool configs and provider hints
dynamically instead of relying on hardcoded dicts.
"""

from victor.core.vertical_types import TieredToolTemplate
from victor.core.verticals.config_registry import VerticalConfigRegistry


class TestDynamicToolConfigRegistration:
    """Verify verticals can register tool configs dynamically."""

    def test_register_and_retrieve_custom_vertical(self):
        """A new vertical can register tool config without core changes."""
        TieredToolTemplate.register_vertical_tools(
            "medical",
            core_tools={"read", "web_search", "web_fetch"},
            readonly_for_analysis=True,
        )
        config = TieredToolTemplate.for_vertical("medical")
        assert config is not None
        assert "read" in config.mandatory
        assert "web_search" in config.vertical_core

    def test_builtin_verticals_require_registration(self):
        """Verticals must register via register_vertical_tools() first."""
        # Without registration, for_vertical returns None
        TieredToolTemplate._registered_verticals.pop("coding", None)
        assert TieredToolTemplate.for_vertical("coding") is None

        # After registration, it resolves
        TieredToolTemplate.register_vertical_tools(
            "coding",
            core_tools={"edit", "write", "shell", "git", "search", "overview"},
            readonly_for_analysis=False,
        )
        coding = TieredToolTemplate.for_vertical("coding")
        assert coding is not None
        assert "edit" in coding.vertical_core
        TieredToolTemplate._registered_verticals.pop("coding", None)

    def test_unknown_vertical_returns_none(self):
        """Unknown vertical without registration returns None."""
        config = TieredToolTemplate.for_vertical("nonexistent_xyz_12345")
        assert config is None

    def test_registered_vertical_can_be_updated(self):
        """Dynamic registration can be updated by re-registering."""
        TieredToolTemplate.register_vertical_tools(
            "research",
            core_tools={"web_search", "web_fetch", "overview"},
            readonly_for_analysis=True,
        )
        original = TieredToolTemplate.for_vertical("research")
        assert original is not None
        assert "shell" not in original.vertical_core

        TieredToolTemplate.register_vertical_tools(
            "research",
            core_tools={"web_search", "web_fetch", "overview", "shell"},
            readonly_for_analysis=False,
        )
        updated = TieredToolTemplate.for_vertical("research")
        assert updated is not None
        assert "shell" in updated.vertical_core

        TieredToolTemplate._registered_verticals.pop("research", None)


class TestDynamicProviderHintsRegistration:
    """Verify verticals can register provider hints and eval criteria."""

    def test_register_provider_hints(self):
        VerticalConfigRegistry.register_vertical_config(
            "medical",
            provider_hints={
                "preferred_providers": ["anthropic"],
                "min_context_window": 50000,
            },
        )
        hints = VerticalConfigRegistry.get_provider_hints("medical")
        assert hints["preferred_providers"] == ["anthropic"]
        assert hints["min_context_window"] == 50000
        VerticalConfigRegistry._registered_provider_hints.pop("medical", None)

    def test_register_evaluation_criteria(self):
        VerticalConfigRegistry.register_vertical_config(
            "medical",
            evaluation_criteria=["diagnosis_accuracy", "source_quality"],
        )
        criteria = VerticalConfigRegistry.get_evaluation_criteria("medical")
        assert "diagnosis_accuracy" in criteria
        VerticalConfigRegistry._registered_eval_criteria.pop("medical", None)

    def test_builtin_hints_still_work(self):
        hints = VerticalConfigRegistry.get_provider_hints("coding")
        assert "anthropic" in hints["preferred_providers"]

    def test_unknown_falls_back_to_default(self):
        hints = VerticalConfigRegistry.get_provider_hints("nonexistent_xyz")
        assert hints == VerticalConfigRegistry._provider_hints["default"]
