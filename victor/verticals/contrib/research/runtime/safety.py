"""Research Safety Extension - Patterns for safe research practices.

This module delegates to the framework's safety patterns (victor.security.safety)
for source credibility and content warnings, extending with research-specific patterns.
"""

from typing import Dict, List, Tuple

from victor.core.verticals.protocols import SafetyExtensionProtocol, SafetyPattern

# Import framework safety patterns (DRY principle)
from victor.security.safety.source_credibility import (
    SOURCE_CREDIBILITY_PATTERNS as _FRAMEWORK_CREDIBILITY_PATTERNS,
    validate_source_credibility as _framework_validate_credibility,
    get_source_safety_reminders as _framework_source_reminders,
)
from victor.security.safety.content_patterns import (
    CONTENT_WARNING_PATTERNS as _FRAMEWORK_CONTENT_PATTERNS,
    scan_content_warnings as _framework_scan_warnings,
)

# Risk levels
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

# Research-specific safety patterns as tuples (pattern, description, risk_level)
# These are RESEARCH-SPECIFIC, not generic patterns
_RESEARCH_SAFETY_TUPLES: List[Tuple[str, str, str]] = [
    # High-risk patterns - could spread misinformation (research-specific)
    (r"(?i)fake\s+news|fabricat.*source|invent.*citation", "Fabricating sources", HIGH),
    (r"(?i)plagiari|copy\s+without\s+attribution", "Plagiarism risk", HIGH),
    # Medium-risk patterns - need verification (research-specific)
    (r"(?i)always|never|guaranteed|proven\s+fact", "Absolute claims", MEDIUM),
    (r"(?i)everyone\s+knows|obviously|clearly", "Unsupported generalizations", MEDIUM),
    (r"(?i)studies\s+show|research\s+proves", "Unattributed research claims", MEDIUM),
    (r"(?i)secret|they\s+don't\s+want\s+you\s+to\s+know", "Conspiracy language", MEDIUM),
    # Low-risk patterns - style improvements (research-specific)
    (r"(?i)probably|maybe|might\s+be", "Hedging language", LOW),
    (r"(?i)in\s+my\s+opinion|I\s+think|I\s+believe", "Opinion markers", LOW),
]

# DEPRECATED: Use victor.security.safety.source_credibility instead
# Kept for backward compatibility
SOURCE_CREDIBILITY_PATTERNS: Dict[str, str] = {
    "high_credibility": r"\.gov|\.edu|arxiv\.org|pubmed|doi\.org|nature\.com|science\.org",
    "medium_credibility": r"wikipedia\.org|medium\.com|substack\.com|news\.",
    "low_credibility": r"\.blogspot\.|wordpress\.com/(?!.*official)|tumblr\.com",
}

# DEPRECATED: Use victor.security.safety.content_patterns instead
# Kept for backward compatibility
CONTENT_WARNING_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)graphic\s+content|violence|disturbing", "Potentially disturbing content"),
    (r"(?i)trigger\s+warning|sensitive\s+topic", "Sensitive topic"),
    (r"(?i)controversial|disputed|debated", "Controversial topic"),
]


class ResearchSafetyExtension(SafetyExtensionProtocol):
    """Safety extension for research tasks.

    Delegates to framework safety patterns (victor.security.safety) for
    source credibility and content warnings, extending with research-specific patterns.
    """

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Return research-specific bash patterns.

        Returns:
            List of SafetyPattern for dangerous bash commands.
        """
        return [
            SafetyPattern(
                pattern=p,
                description=d,
                risk_level=r,
                category="research",
            )
            for p, d, r in _RESEARCH_SAFETY_TUPLES
        ]

    def get_danger_patterns(self) -> List[Tuple[str, str, str]]:
        """Return research-specific danger patterns (legacy format).

        Returns:
            List of (regex_pattern, description, risk_level) tuples.
        """
        return _RESEARCH_SAFETY_TUPLES

    def get_blocked_operations(self) -> List[str]:
        """Return operations that should be blocked in research context."""
        return [
            "execute_code",  # Research shouldn't execute arbitrary code
            "modify_system_files",  # Research is read-focused
            "send_email",  # Research shouldn't send communications
            "make_purchase",  # Research shouldn't make transactions
        ]

    def get_content_warnings(self) -> List[Tuple[str, str]]:
        """Return patterns that should trigger content warnings.

        Delegates to framework safety patterns for consistency.
        """
        # Return legacy format for backward compatibility
        return CONTENT_WARNING_PATTERNS

    def validate_source_credibility(self, url: str) -> str:
        """Classify source credibility based on URL patterns.

        Delegates to framework safety patterns for consistency and performance.

        Returns:
            Credibility level: 'high', 'medium', 'low', or 'unknown'
        """
        # Use framework implementation (with native Rust acceleration)
        result = _framework_validate_credibility(url)
        return result.level.value

    def get_safety_reminders(self) -> List[str]:
        """Return safety reminders for research output.

        Extends framework reminders with research-specific guidance.
        """
        # Combine framework reminders with research-specific ones
        framework_reminders = _framework_source_reminders()
        research_reminders = [
            "Cite all sources with accessible URLs",
            "Distinguish between facts, analysis, and opinions",
            "Note limitations and potential biases in sources",
        ]
        # Deduplicate while preserving order
        all_reminders = list(dict.fromkeys(framework_reminders + research_reminders))
        return all_reminders


__all__ = [
    "ResearchSafetyExtension",
    "HIGH",
    "MEDIUM",
    "LOW",
    # New framework-based safety rules
    "create_research_source_safety_rules",
    "create_research_content_safety_rules",
    "create_all_research_safety_rules",
]


# =============================================================================
# Framework-Based Safety Rules (New)
# =============================================================================

"""Framework-based safety rules for research operations.

This section provides factory functions that register safety rules
with the framework-level SafetyEnforcer. This is the new recommended
approach for safety enforcement in research workflows.

Example:
    from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
    from victor.verticals.contrib.research.safety import create_all_research_safety_rules

    enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
    create_all_research_safety_rules(enforcer)

    # Check operations
    allowed, reason = enforcer.check_operation("cite unreliable source")
    if not allowed:
        print(f"Blocked: {reason}")
"""

from victor.framework.config import SafetyEnforcer, SafetyRule, SafetyLevel


def create_research_source_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_low_credibility_sources: bool = True,
    require_source_verification: bool = False,
    blocked_domains: list[str] | None = None,
) -> None:
    """Register research source safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_low_credibility_sources: Block sources from low-credibility domains
        require_source_verification: Require verification for non-.edu/.gov sources
        blocked_domains: List of specific domains to block

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.MEDIUM))
        create_research_source_safety_rules(
            enforcer,
            block_low_credibility_sources=True,
            blocked_domains=["fake-news-site.com"]
        )
    """
    if block_low_credibility_sources:
        enforcer.add_rule(
            SafetyRule(
                name="research_block_low_credibility",
                description="Block low-credibility sources (.blogspot, tumblr, etc.)",
                check_fn=lambda op: any(
                    domain in op.lower()
                    for domain in [
                        ".blogspot.",
                        "wordpress.com/",
                        "tumblr.com",
                        "fake-news",
                        "conspiracy",
                    ]
                )
                and ("cite" in op.lower() or "source" in op.lower()),
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )

    if require_source_verification:
        enforcer.add_rule(
            SafetyRule(
                name="research_require_source_verification",
                description="Require verification for non-.edu/.gov sources",
                check_fn=lambda op: ("cite" in op.lower() or "source" in op.lower())
                and not any(
                    domain in op.lower() for domain in [".edu", ".gov", "arxiv.org", "doi.org"]
                ),
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )

    if blocked_domains:
        for domain in blocked_domains:
            enforcer.add_rule(
                SafetyRule(
                    name=f"research_block_{domain.replace('.', '_')}",
                    description=f"Block sources from {domain}",
                    check_fn=lambda op, d=domain: d in op.lower(),
                    level=SafetyLevel.HIGH,
                    allow_override=False,
                )
            )


def create_research_content_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_fabricated_content: bool = True,
    warn_absolute_claims: bool = True,
    block_plagiarism_risk: bool = True,
) -> None:
    """Register research content safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_fabricated_content: Block fabricating sources/citations
        warn_absolute_claims: Warn about absolute claims (always, never, proven)
        block_plagiarism_risk: Block plagiarism risks

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_research_content_safety_rules(
            enforcer,
            block_fabricated_content=True,
            warn_absolute_claims=True
        )
    """
    if block_fabricated_content:
        enforcer.add_rule(
            SafetyRule(
                name="research_block_fabricated",
                description="Block fabricating sources or citations",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "fake source",
                        "fabricat",
                        "invent citation",
                        "fake citation",
                    ]
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

    if warn_absolute_claims:
        enforcer.add_rule(
            SafetyRule(
                name="research_warn_absolute_claims",
                description="Warn about absolute claims (always, never, proven)",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "always ",
                        "never ",
                        "guaranteed",
                        "proven fact",
                        "everyone knows",
                    ]
                ),
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )

    if block_plagiarism_risk:
        enforcer.add_rule(
            SafetyRule(
                name="research_warn_plagiarism",
                description="Warn about plagiarism risk",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "plagiar",
                        "copy without attribution",
                        "unattributed",
                    ]
                ),
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )


def create_all_research_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    blocked_domains: list[str] | None = None,
) -> None:
    """Register all research safety rules at once.

    This is a convenience function that registers all research-specific
    safety rules with appropriate defaults.

    Args:
        enforcer: SafetyEnforcer to register rules with
        blocked_domains: Specific domains to block

    Example:
        from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.MEDIUM))
        create_all_research_safety_rules(enforcer)

        # Now all operations are checked
        allowed, reason = enforcer.check_operation("cite unreliable source")
    """
    create_research_source_safety_rules(enforcer, blocked_domains=blocked_domains)
    create_research_content_safety_rules(enforcer)
