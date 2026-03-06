"""Base safety patterns shared across all verticals.

Provides common patterns for destructive file operations, privilege
escalation, and secret detection. Verticals extend via get_additional_patterns().
"""

from __future__ import annotations

import logging

from victor.security.safety.types import SafetyPattern

logger = logging.getLogger(__name__)


# Common patterns shared across verticals
DESTRUCTIVE_FILE_PATTERNS = [
    SafetyPattern(
        pattern=r"rm\s+-r[f]?\s+/",
        risk_level="HIGH",
        description="Recursive delete from root",
        category="file_operations",
    ),
    SafetyPattern(
        pattern=r"chmod\s+777",
        risk_level="MEDIUM",
        description="World-writable permissions",
        category="file_operations",
    ),
    SafetyPattern(
        pattern=r">\s*/",
        risk_level="HIGH",
        description="Truncate file from root",
        category="file_operations",
    ),
]

PRIVILEGE_ESCALATION_PATTERNS = [
    SafetyPattern(
        pattern=r"NOPASSWD",
        risk_level="HIGH",
        description="Passwordless sudo configuration",
        category="privilege",
    ),
    SafetyPattern(
        pattern=r"chmod\s+[ugo]*s",
        risk_level="HIGH",
        description="Set SUID/SGID bit",
        category="privilege",
    ),
]

SECRET_DETECTION_PATTERNS = [
    SafetyPattern(
        pattern=r'(?:password|passwd|secret|token)\s*=\s*["\'][^"\']{8,}',
        risk_level="HIGH",
        description="Potential hardcoded secret",
        category="secrets",
    ),
    SafetyPattern(
        pattern=r'(?:api[_-]?key|apikey)\s*=\s*["\'][^"\']{16,}',
        risk_level="HIGH",
        description="Potential API key",
        category="secrets",
    ),
]

GIT_SAFETY_PATTERNS = [
    SafetyPattern(
        pattern=r"git\s+push\s+.*--force",
        risk_level="HIGH",
        description="Force push can overwrite remote history",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+reset\s+--hard",
        risk_level="MEDIUM",
        description="Hard reset discards uncommitted changes",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+branch\s+-[dD]",
        risk_level="MEDIUM",
        description="Branch deletion",
        category="git",
    ),
]

COMMON_PATTERNS = (
    DESTRUCTIVE_FILE_PATTERNS
    + PRIVILEGE_ESCALATION_PATTERNS
    + SECRET_DETECTION_PATTERNS
    + GIT_SAFETY_PATTERNS
)


class BaseSafetyExtension:
    """Base safety extension with common patterns.

    Verticals should subclass and override get_additional_patterns()
    to add domain-specific patterns.
    """

    def get_bash_patterns(self) -> list[SafetyPattern]:
        """Get all patterns: common + vertical-specific."""
        return COMMON_PATTERNS + self.get_additional_patterns()

    # Keep get_patterns as alias for backward compat
    def get_patterns(self) -> list[SafetyPattern]:
        """Alias for get_bash_patterns()."""
        return self.get_bash_patterns()

    def get_additional_patterns(self) -> list[SafetyPattern]:
        """Override in vertical subclass to add domain-specific patterns."""
        return []

    def get_category(self) -> str:
        """Override in vertical subclass."""
        return "general"
