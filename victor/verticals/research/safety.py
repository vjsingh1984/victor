"""Research Safety Extension - Patterns for safe research practices."""

from typing import Dict, List, Tuple

from victor.verticals.protocols import SafetyExtensionProtocol, SafetyPattern


# Risk levels
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

# Research-specific safety patterns as tuples (pattern, description, risk_level)
_RESEARCH_SAFETY_TUPLES: List[Tuple[str, str, str]] = [
    # High-risk patterns - could spread misinformation
    (r"(?i)fake\s+news|fabricat.*source|invent.*citation", "Fabricating sources", HIGH),
    (r"(?i)plagiari|copy\s+without\s+attribution", "Plagiarism risk", HIGH),
    (r"(?i)medical\s+advice|diagnos|prescri", "Medical advice risk", HIGH),
    (r"(?i)legal\s+advice|sue|lawsuit\s+against", "Legal advice risk", HIGH),
    (r"(?i)financial\s+advice|invest\s+in|buy\s+stock", "Financial advice risk", HIGH),
    # Medium-risk patterns - need verification
    (r"(?i)always|never|guaranteed|proven\s+fact", "Absolute claims", MEDIUM),
    (r"(?i)everyone\s+knows|obviously|clearly", "Unsupported generalizations", MEDIUM),
    (r"(?i)studies\s+show|research\s+proves", "Unattributed research claims", MEDIUM),
    (r"(?i)secret|they\s+don't\s+want\s+you\s+to\s+know", "Conspiracy language", MEDIUM),
    # Low-risk patterns - style improvements
    (r"(?i)probably|maybe|might\s+be", "Hedging language", LOW),
    (r"(?i)in\s+my\s+opinion|I\s+think|I\s+believe", "Opinion markers", LOW),
]

# Source credibility patterns
SOURCE_CREDIBILITY_PATTERNS: Dict[str, str] = {
    "high_credibility": r"\.gov|\.edu|arxiv\.org|pubmed|doi\.org|nature\.com|science\.org",
    "medium_credibility": r"wikipedia\.org|medium\.com|substack\.com|news\.",
    "low_credibility": r"\.blogspot\.|wordpress\.com/(?!.*official)|tumblr\.com",
}

# Content warning patterns
CONTENT_WARNING_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)graphic\s+content|violence|disturbing", "Potentially disturbing content"),
    (r"(?i)trigger\s+warning|sensitive\s+topic", "Sensitive topic"),
    (r"(?i)controversial|disputed|debated", "Controversial topic"),
]


class ResearchSafetyExtension(SafetyExtensionProtocol):
    """Safety extension for research tasks."""

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
        """Return patterns that should trigger content warnings."""
        return CONTENT_WARNING_PATTERNS

    def validate_source_credibility(self, url: str) -> str:
        """Classify source credibility based on URL patterns.

        Returns:
            Credibility level: 'high', 'medium', 'low', or 'unknown'
        """
        import re

        for level, pattern in SOURCE_CREDIBILITY_PATTERNS.items():
            if re.search(pattern, url, re.IGNORECASE):
                return level.replace("_credibility", "")
        return "unknown"

    def get_safety_reminders(self) -> List[str]:
        """Return safety reminders for research output."""
        return [
            "Verify claims with multiple independent sources",
            "Cite all sources with accessible URLs",
            "Distinguish between facts, analysis, and opinions",
            "Note limitations and potential biases in sources",
            "Avoid providing medical, legal, or financial advice",
        ]
