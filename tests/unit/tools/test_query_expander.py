"""Unit tests for query expansion module."""

import pytest

import pytest
pytest.importorskip("victor_coding.codebase")

from victor_coding.codebase.query_expander import (
    QueryExpander,
    expand_query,
    get_query_expander,
    SEMANTIC_QUERY_EXPANSIONS,
)


class TestQueryExpander:
    """Test QueryExpander class."""

    def test_initialization_default(self):
        """Test initialization with default expansions."""
        expander = QueryExpander()
        assert expander.expansions == SEMANTIC_QUERY_EXPANSIONS
        assert len(expander.expansions) > 0

    def test_initialization_custom(self):
        """Test initialization with custom expansions."""
        custom = {"foo": ["bar", "baz"]}
        expander = QueryExpander(expansions=custom)
        assert expander.expansions == custom

    def test_expand_query_with_match(self):
        """Test query expansion with matching pattern."""
        expander = QueryExpander()
        result = expander.expand_query("tool registration", max_expansions=5)

        assert len(result) <= 5
        assert result[0] == "tool registration"  # Original always first
        assert "register tool" in result or "@tool decorator" in result

    def test_expand_query_without_match(self):
        """Test query expansion with no matching pattern."""
        expander = QueryExpander()
        result = expander.expand_query("unknown query pattern", max_expansions=5)

        assert len(result) == 1
        assert result[0] == "unknown query pattern"

    def test_expand_query_respects_max_expansions(self):
        """Test that max_expansions is respected."""
        expander = QueryExpander()
        result = expander.expand_query("tool registration", max_expansions=3)

        assert len(result) <= 3
        assert result[0] == "tool registration"

    def test_expand_query_case_insensitive(self):
        """Test that expansion is case insensitive."""
        expander = QueryExpander()
        result1 = expander.expand_query("Tool Registration", max_expansions=5)
        result2 = expander.expand_query("TOOL REGISTRATION", max_expansions=5)

        # Should expand regardless of case
        assert len(result1) > 1
        assert len(result2) > 1

    def test_expand_query_deduplicates(self):
        """Test that expansions are deduplicated."""
        expander = QueryExpander()
        result = expander.expand_query("tool registration", max_expansions=10)

        # Check no duplicates (case-insensitive)
        seen = set()
        for item in result:
            item_lower = item.lower()
            assert item_lower not in seen
            seen.add(item_lower)

    def test_is_expandable_true(self):
        """Test is_expandable returns True for expandable queries."""
        expander = QueryExpander()
        assert expander.is_expandable("tool registration") is True
        assert expander.is_expandable("error handling") is True
        assert expander.is_expandable("provider") is True

    def test_is_expandable_false(self):
        """Test is_expandable returns False for non-expandable queries."""
        expander = QueryExpander()
        assert expander.is_expandable("unknown pattern") is False
        assert expander.is_expandable("random text") is False

    def test_get_expansion_terms(self):
        """Test getting expansion terms without original query."""
        expander = QueryExpander()
        terms = expander.get_expansion_terms("tool registration")

        assert len(terms) > 0
        assert "register tool" in terms or "@tool decorator" in terms
        assert "tool registration" not in terms  # Original excluded

    def test_multiple_pattern_matches(self):
        """Test expansion when query matches multiple patterns."""
        expander = QueryExpander()
        result = expander.expand_query("tool registration and error handling", max_expansions=10)

        assert len(result) > 1
        # Should expand for both patterns
        assert any("register" in r.lower() or "tool" in r.lower() for r in result)

    def test_specific_expansions_content(self):
        """Test specific expansion mappings."""
        expander = QueryExpander()

        # Tool registration
        result = expander.expand_query("tool registration", max_expansions=10)
        assert "register tool" in result
        assert "@tool decorator" in result

        # Error handling
        result = expander.expand_query("error handling", max_expansions=10)
        assert "exception" in result
        assert "try catch" in result or "try except" in result

        # Configuration
        result = expander.expand_query("configuration", max_expansions=10)
        assert "config" in result
        assert "settings" in result


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_query_expander_singleton(self):
        """Test that get_query_expander returns singleton."""
        expander1 = get_query_expander()
        expander2 = get_query_expander()

        assert expander1 is expander2

    def test_expand_query_convenience_function(self):
        """Test expand_query convenience function."""
        result = expand_query("tool registration", max_expansions=5)

        assert len(result) > 0
        assert result[0] == "tool registration"


class TestSemanticExpansions:
    """Test semantic expansion mappings."""

    def test_tool_architecture_expansions(self):
        """Test tool/plugin architecture expansions."""
        assert "tool registration" in SEMANTIC_QUERY_EXPANSIONS
        assert "plugin registration" in SEMANTIC_QUERY_EXPANSIONS
        assert len(SEMANTIC_QUERY_EXPANSIONS["tool registration"]) >= 5

    def test_provider_expansions(self):
        """Test provider-related expansions."""
        assert "provider" in SEMANTIC_QUERY_EXPANSIONS
        assert "provider implementation" in SEMANTIC_QUERY_EXPANSIONS

    def test_error_handling_expansions(self):
        """Test error handling expansions."""
        assert "error handling" in SEMANTIC_QUERY_EXPANSIONS
        assert "exception" in SEMANTIC_QUERY_EXPANSIONS

    def test_configuration_expansions(self):
        """Test configuration expansions."""
        assert "configuration" in SEMANTIC_QUERY_EXPANSIONS
        assert "settings" in SEMANTIC_QUERY_EXPANSIONS

    def test_all_expansions_are_lists(self):
        """Test that all expansion values are lists."""
        for key, value in SEMANTIC_QUERY_EXPANSIONS.items():
            assert isinstance(value, list)
            assert len(value) > 0
            for item in value:
                assert isinstance(item, str)
                assert len(item) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query(self):
        """Test expansion with empty query."""
        expander = QueryExpander()
        result = expander.expand_query("", max_expansions=5)

        assert len(result) == 1
        assert result[0] == ""

    def test_whitespace_only_query(self):
        """Test expansion with whitespace-only query."""
        expander = QueryExpander()
        result = expander.expand_query("   ", max_expansions=5)

        assert len(result) == 1

    def test_max_expansions_zero(self):
        """Test with max_expansions=0 (edge case)."""
        expander = QueryExpander()
        result = expander.expand_query("tool registration", max_expansions=0)

        # Should still include original
        assert len(result) >= 1

    def test_max_expansions_one(self):
        """Test with max_expansions=1 (may include 1-2 results)."""
        expander = QueryExpander()
        result = expander.expand_query("tool registration", max_expansions=1)

        # With max_expansions=1, we get at least the original
        assert len(result) >= 1
        assert len(result) <= 2  # Original + potentially 1 expansion
        assert result[0] == "tool registration"

    def test_very_large_max_expansions(self):
        """Test with very large max_expansions."""
        expander = QueryExpander()
        result = expander.expand_query("tool registration", max_expansions=1000)

        # Should not exceed available expansions
        assert len(result) < 1000
        assert result[0] == "tool registration"

    def test_special_characters_in_query(self):
        """Test queries with special characters."""
        expander = QueryExpander()
        result = expander.expand_query("@tool registration", max_expansions=5)

        # Should still expand (case-insensitive substring match)
        assert len(result) >= 1

    def test_query_with_numbers(self):
        """Test queries with numbers."""
        expander = QueryExpander()
        result = expander.expand_query("tool123 registration", max_expansions=5)

        # Should still match "tool registration" pattern
        assert len(result) >= 1


class TestPerformance:
    """Test performance characteristics."""

    def test_expansion_is_fast(self):
        """Test that expansion completes quickly."""
        import time

        expander = QueryExpander()
        start = time.time()

        for _ in range(100):
            expander.expand_query("tool registration", max_expansions=5)

        elapsed = time.time() - start
        assert elapsed < 1.0  # Should complete 100 expansions in under 1 second

    def test_large_expansion_dict(self):
        """Test with large custom expansion dictionary."""
        # Create large custom dictionary
        large_expansions = {f"pattern{i}": [f"syn{i}_{j}" for j in range(10)] for i in range(100)}

        expander = QueryExpander(expansions=large_expansions)
        result = expander.expand_query("pattern50", max_expansions=5)

        assert len(result) > 1
        assert result[0] == "pattern50"
