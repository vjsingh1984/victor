"""SDK-owned safety contracts and pattern declarations."""

from victor_contracts.safety.patterns import (
    SafetyPatternDeclaration,
    SafetyPatternType,
    SafetySeverity,
)
from victor_contracts.safety.pii import (
    ANONYMIZATION_SUGGESTIONS,
    PII_COLUMN_PATTERNS,
    PII_CONTENT_PATTERNS,
    PII_SEVERITY,
    PIIMatch,
    PIIScanner,
    PIISeverity,
    PIIType,
    detect_pii_columns,
    detect_pii_in_content,
    get_anonymization_suggestion,
    get_pii_severity,
    get_pii_types,
    get_safety_reminders,
    has_pii,
)
from victor_contracts.safety.runtime import (
    SafetyAction,
    SafetyCategory,
    SafetyCheckResult,
    SafetyCoordinator,
    SafetyRule,
    SafetyStats,
)

from enum import Enum


class SafetyLevel(str, Enum):
    """Safety enforcement level.

    Promoted from victor.framework.config for SDK-only vertical development.
    """

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


__all__ = [
    "ANONYMIZATION_SUGGESTIONS",
    "PII_COLUMN_PATTERNS",
    "PII_CONTENT_PATTERNS",
    "PII_SEVERITY",
    "PIIMatch",
    "PIIScanner",
    "PIISeverity",
    "PIIType",
    "SafetyAction",
    "SafetyCategory",
    "SafetyCheckResult",
    "SafetyCoordinator",
    "SafetyLevel",
    "SafetyPatternDeclaration",
    "SafetyPatternType",
    "SafetyRule",
    "SafetySeverity",
    "SafetyStats",
    "detect_pii_columns",
    "detect_pii_in_content",
    "get_anonymization_suggestion",
    "get_pii_severity",
    "get_pii_types",
    "get_safety_reminders",
    "has_pii",
]
