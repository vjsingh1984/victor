"""Data-only safety pattern declarations for external verticals.

These are declarative types — verticals specify WHAT safety patterns
they need, not HOW they're enforced. The framework interprets these
declarations at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class SafetyPatternType(str, Enum):
    """Types of safety patterns a vertical can declare."""

    CODE_EXECUTION = "code_execution"
    FILE_DELETION = "file_deletion"
    SECRETS_DETECTION = "secrets_detection"
    PII_DETECTION = "pii_detection"
    INFRASTRUCTURE_CHANGE = "infrastructure_change"
    NETWORK_ACCESS = "network_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CUSTOM = "custom"


class SafetySeverity(str, Enum):
    """Severity level for safety pattern enforcement."""

    BLOCK = "block"
    WARN = "warn"
    LOG = "log"


@dataclass(frozen=True)
class SafetyPatternDeclaration:
    """Declarative safety pattern that a vertical requires.

    Verticals create these to declare what safety patterns they need.
    The framework's SafetyCoordinator resolves them to concrete
    implementations at runtime.

    Example:
        class CodingAssistant(VerticalBase):
            @classmethod
            def get_safety_declarations(cls):
                return [
                    SafetyPatternDeclaration(
                        pattern_type=SafetyPatternType.FILE_DELETION,
                        severity=SafetySeverity.BLOCK,
                        description="Block recursive file deletion",
                    ),
                    SafetyPatternDeclaration(
                        pattern_type=SafetyPatternType.SECRETS_DETECTION,
                        severity=SafetySeverity.WARN,
                    ),
                ]
    """

    pattern_type: SafetyPatternType
    severity: SafetySeverity = SafetySeverity.WARN
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
