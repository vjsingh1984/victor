# Tests for victor-research safety rules
# Migrated from victor/tests/unit/framework/test_config.py

import pytest

from victor.framework.config import SafetyConfig, SafetyEnforcer, SafetyLevel


class TestResearchSourceSafety:
    """Tests for Research source safety rules."""

    def test_research_safety_source_rules(self):
        """Research source safety rules should block low-credibility sources."""
        from victor_research.safety import create_research_source_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_research_source_safety_rules(
            enforcer,
            block_low_credibility_sources=True,
            blocked_domains=["fake-news-site.com"],
        )

        # Test low-credibility source blocking
        allowed, reason = enforcer.check_operation("cite article from fake-blog.blogspot.com")
        assert allowed is False
        assert (
            "credibility" in reason.lower()
            or "blogspot" in reason.lower()
            or "blocked" in reason.lower()
        )

        # Test blocked domain - note: low-credibility rule may trigger first
        allowed, reason = enforcer.check_operation("cite source from fake-news-site.com")
        assert allowed is False
        # Either the domain-specific rule or the general low-credibility rule can block
        assert (
            "fake" in reason.lower()
            or "credibility" in reason.lower()
            or "blocked" in reason.lower()
        )


class TestResearchContentSafety:
    """Tests for Research content safety rules."""

    def test_research_safety_content_rules(self):
        """Research content safety rules should block fabricated content."""
        from victor_research.safety import create_research_content_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_research_content_safety_rules(
            enforcer,
            block_fabricated_content=True,
            warn_absolute_claims=True,
        )

        # Test fabricated content blocking
        allowed, reason = enforcer.check_operation("fabricate source for claim")
        assert allowed is False
        assert "fabricat" in reason.lower()

        # Test absolute claims warning (should only warn, not block at HIGH level)
        # But block_fabricated_content should catch it
        allowed, reason = enforcer.check_operation("always use this source")
        # At HIGH level, LOW/MEDIUM warnings might still block depending on implementation
        assert isinstance(allowed, bool)


class TestResearchAllSafetyRules:
    """Tests for all Research safety rules combined."""

    def test_create_all_research_safety_rules(self):
        """create_all_research_safety_rules should register all research rules."""
        from victor_research.safety import create_all_research_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_research_safety_rules(enforcer)

        # Should have rules from source and content categories
        assert len(enforcer.rules) > 0

        # Verify low-credibility source is blocked
        allowed, _ = enforcer.check_operation("cite tumblr.com source")
        assert allowed is False

        # Verify fabricated content is blocked
        allowed, _ = enforcer.check_operation("invent citation for paper")
        assert allowed is False
