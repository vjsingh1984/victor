"""SDK safety pattern declarations.

Provides data-only safety pattern types that verticals can use to
declare their safety requirements without importing from
victor.security.safety (framework internal).

Verticals declare WHAT patterns to apply. The framework's SafetyCoordinator
decides HOW to enforce them at runtime.
"""

from victor_sdk.safety.patterns import (
    SafetyPatternDeclaration,
    SafetyPatternType,
    SafetySeverity,
)

__all__ = [
    "SafetyPatternDeclaration",
    "SafetyPatternType",
    "SafetySeverity",
]
