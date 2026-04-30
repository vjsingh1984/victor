"""Tests for EvolvedContentResolver."""

import pytest

from victor.agent.evolved_content_resolver import (
    EvolvedContentResolver,
    ResolvedContent,
)


class TestResolvedContent:
    """Test suite for ResolvedContent dataclass."""

    def test_is_evolved(self):
        """Test is_evolved() method."""
        content = ResolvedContent(
            section_name="TEST_SECTION", text="Evolved content", source="evolved", metadata={}
        )
        assert content.is_evolved() is True
        assert content.is_static() is False

    def test_is_static(self):
        """Test is_static() method."""
        content = ResolvedContent(
            section_name="TEST_SECTION", text="Static content", source="static", metadata={}
        )
        assert content.is_static() is True
        assert content.is_evolved() is False


class TestEvolvedContentResolver:
    """Test suite for EvolvedContentResolver."""

    def test_resolve_section_static_fallback(self):
        """Test resolution falls back to static when no evolved content."""
        resolver = EvolvedContentResolver(optimization_injector=None)

        result = resolver.resolve_section(
            section_name="ASI_TOOL_EFFECTIVENESS_GUIDANCE",
            fallback_text="Static fallback text",
        )

        assert result.section_name == "ASI_TOOL_EFFECTIVENESS_GUIDANCE"
        assert result.text == "Static fallback text"
        assert result.source == "static"
        assert result.metadata == {}

    def test_resolve_section_caches_results(self):
        """Test that resolver caches results."""
        resolver = EvolvedContentResolver(optimization_injector=None)

        # Resolve same section twice
        result1 = resolver.resolve_section(
            "TEST_SECTION",
            fallback_text="Test text",
        )
        result2 = resolver.resolve_section(
            "TEST_SECTION",
            fallback_text="Different text",  # Should be ignored due to cache
        )

        assert result1 is result2  # Same cached instance
        assert result1.text == "Test text"

    def test_resolve_multiple_sections(self):
        """Test resolving multiple sections efficiently."""
        resolver = EvolvedContentResolver(optimization_injector=None)

        results = resolver.resolve_multiple(
            section_names=[
                "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
                "GROUNDING_RULES",
                "COMPLETION_GUIDANCE",
            ],
            fallback_map={
                "ASI_TOOL_EFFECTIVENESS_GUIDANCE": "Tool guidance",
                "GROUNDING_RULES": "Grounding text",
                "COMPLETION_GUIDANCE": "Completion text",
            },
        )

        assert len(results) == 3
        assert all(r.source == "static" for r in results)
        assert results[0].text == "Tool guidance"
        assert results[1].text == "Grounding text"
        assert results[2].text == "Completion text"

    def test_resolve_multiple_uses_fallback_map(self):
        """Test that fallback_map is used correctly."""
        resolver = EvolvedContentResolver(optimization_injector=None)

        results = resolver.resolve_multiple(
            section_names=["SECTION_A", "SECTION_B"],
            fallback_map={
                "SECTION_A": "Fallback A",
                "SECTION_B": "Fallback B",
            },
        )

        assert results[0].text == "Fallback A"
        assert results[1].text == "Fallback B"

    def test_resolve_multiple_missing_fallback(self):
        """Test resolve_multiple with missing fallbacks uses empty string."""
        resolver = EvolvedContentResolver(optimization_injector=None)

        results = resolver.resolve_multiple(
            section_names=["SECTION_A", "SECTION_B"],
            fallback_map={},  # No fallbacks provided
        )

        assert results[0].text == ""
        assert results[1].text == ""

    def test_clear_cache(self):
        """Test that clear_cache() removes cached entries."""
        resolver = EvolvedContentResolver(optimization_injector=None)

        # Cache some entries
        resolver.resolve_section("TEST_1", fallback_text="Text 1")
        resolver.resolve_section("TEST_2", fallback_text="Text 2")

        # Clear cache
        resolver.clear_cache()

        # Resolve again with different fallback
        result = resolver.resolve_section(
            "TEST_1",
            fallback_text="New text",
        )

        assert result.text == "New text"

    def test_resolve_with_provider_and_model(self):
        """Test resolution with provider and model context."""
        resolver = EvolvedContentResolver(optimization_injector=None)

        result = resolver.resolve_section(
            section_name="TEST_SECTION",
            provider="anthropic",
            model="claude-sonnet-4-6",
            task_type="edit",
            fallback_text="Fallback",
        )

        # Without injector, should still fall back to static
        assert result.source == "static"
        assert result.text == "Fallback"

    def test_resolved_content_immutability(self):
        """Test that ResolvedContent is immutable (frozen dataclass)."""
        content = ResolvedContent(
            section_name="TEST", text="Content", source="static", metadata={"key": "value"}
        )

        # Attempting to modify should raise an error
        with pytest.raises(Exception):  # FrozenInstanceError
            content.text = "Modified"
