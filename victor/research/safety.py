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
