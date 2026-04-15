"""SDK-owned safety contracts and pattern declarations."""

from victor_sdk.safety.patterns import (
    SafetyPatternDeclaration,
    SafetyPatternType,
    SafetySeverity,
)
from victor_sdk.safety.runtime import (
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
]
