# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type definitions for safety pattern registry (Phase 6.1).

This module provides core types for the safety pattern system:
- Severity enum for risk classification
- Action enum for response actions
- SafetyPattern dataclass for pattern definitions
- SafetyViolation dataclass for scan results

Design Principles:
- YAML-first: All definitions serializable to/from YAML
- Domain filtering: Patterns can be scoped to specific verticals
- Backward compatible: Works with existing SafetyRegistry
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Severity levels for safety violations.

    Values ordered from most to least severe:
        CRITICAL: Immediate block, potential system damage
        HIGH: Block by default, security risk
        MEDIUM: Warning, potential issue
        LOW: Informational, minor concern
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_string(cls, value: str) -> "Severity":
        """Create Severity from string value (case-insensitive).

        Args:
            value: String representation

        Returns:
            Severity enum value

        Raises:
            ValueError: If value is not valid
        """
        normalized = value.lower().strip()
        for member in cls:
            if member.value == normalized:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Invalid severity: '{value}'. Valid values: {valid}")

    def __lt__(self, other: "Severity") -> bool:  # type: ignore[override]
        """Compare severity levels (CRITICAL > HIGH > MEDIUM > LOW)."""
        order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
        }
        return order[self] > order[other]

    def __le__(self, other: "Severity") -> bool:  # type: ignore[override]
        """Less than or equal comparison."""
        return self < other or self == other


class Action(str, Enum):
    """Action to take when a safety pattern is matched.

    Values:
        BLOCK: Block the operation, require user override
        WARN: Show warning, allow to proceed
        LOG: Log the occurrence, no user notification
    """

    BLOCK = "block"
    WARN = "warn"
    LOG = "log"

    @classmethod
    def from_string(cls, value: str) -> "Action":
        """Create Action from string value (case-insensitive).

        Args:
            value: String representation

        Returns:
            Action enum value

        Raises:
            ValueError: If value is not valid
        """
        normalized = value.lower().strip()
        for member in cls:
            if member.value == normalized:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Invalid action: '{value}'. Valid values: {valid}")


@dataclass
class SafetyPattern:
    """Safety pattern definition for content scanning.

    Provides a declarative pattern definition that can be:
    - Loaded from YAML files
    - Filtered by domain (vertical)
    - Used for regex-based content matching

    Attributes:
        name: Unique pattern identifier
        pattern: Regex pattern string
        severity: Risk severity level
        message: Human-readable description of the violation
        domains: List of domains this pattern applies to (["all"] for all)
        action: Action to take when matched

    Example:
        pattern = SafetyPattern(
            name="api_key",
            pattern=r"AKIA[0-9A-Z]{16}",
            severity=Severity.HIGH,
            message="AWS API key detected",
            domains=["coding", "devops"],
            action=Action.BLOCK,
        )

        if pattern.matches(content):
            print(f"Found: {pattern.message}")
    """

    name: str
    pattern: str
    severity: Severity
    message: str
    domains: List[str] = field(default_factory=lambda: ["all"])
    action: Action = field(default=Action.BLOCK)
    _compiled_pattern: Optional[re.Pattern[str]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Compile the regex pattern after initialization."""
        try:
            self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{self.pattern}' for {self.name}: {e}")
            self._compiled_pattern = None

    def matches(self, content: str) -> bool:
        """Check if content matches this pattern.

        Args:
            content: Text content to check

        Returns:
            True if pattern matches content
        """
        if self._compiled_pattern is None:
            return False
        return bool(self._compiled_pattern.search(content))

    def find_matches(self, content: str) -> List[str]:
        """Find all matches in content.

        Args:
            content: Text content to search

        Returns:
            List of matched strings
        """
        if self._compiled_pattern is None:
            return []
        return self._compiled_pattern.findall(content)

    def applies_to_domain(self, domain: str) -> bool:
        """Check if this pattern applies to a domain.

        Args:
            domain: Domain (vertical) name

        Returns:
            True if pattern applies to this domain
        """
        if "all" in self.domains:
            return True
        return domain.lower() in [d.lower() for d in self.domains]

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to YAML-serializable dictionary.

        Returns:
            Dictionary suitable for YAML serialization
        """
        result: Dict[str, Any] = {
            "name": self.name,
            "pattern": self.pattern,
            "severity": self.severity.value,
            "message": self.message,
        }

        # Only include non-default values
        if self.domains != ["all"]:
            result["domains"] = self.domains
        if self.action != Action.BLOCK:
            result["action"] = self.action.value

        return result

    @classmethod
    def from_yaml_dict(cls, data: Dict[str, Any]) -> "SafetyPattern":
        """Create SafetyPattern from YAML dictionary.

        Args:
            data: Dictionary loaded from YAML

        Returns:
            SafetyPattern instance

        Raises:
            ValueError: If required fields are missing
        """
        # Required fields
        if "name" not in data:
            raise ValueError("Missing required field: 'name'")
        if "pattern" not in data:
            raise ValueError("Missing required field: 'pattern'")
        if "severity" not in data:
            raise ValueError("Missing required field: 'severity'")
        if "message" not in data:
            raise ValueError("Missing required field: 'message'")

        # Parse enums
        severity = data["severity"]
        if isinstance(severity, str):
            severity = Severity.from_string(severity)

        action = data.get("action", Action.BLOCK)
        if isinstance(action, str):
            action = Action.from_string(action)

        return cls(
            name=data["name"],
            pattern=data["pattern"],
            severity=severity,
            message=data["message"],
            domains=data.get("domains", ["all"]),
            action=action,
        )


@dataclass
class SafetyViolation:
    """Result from a safety pattern match.

    Represents a single violation found during content scanning.

    Attributes:
        pattern_name: Name of the pattern that matched
        severity: Severity level of the violation
        message: Human-readable description
        matched_text: The actual text that matched
        action: Recommended action to take
        line_number: Optional line number in content
        context: Optional surrounding context
    """

    pattern_name: str
    severity: Severity
    message: str
    matched_text: str
    action: Action = Action.BLOCK
    line_number: Optional[int] = None
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with violation details
        """
        result: Dict[str, Any] = {
            "pattern_name": self.pattern_name,
            "severity": self.severity.value,
            "message": self.message,
            "matched_text": self.matched_text,
            "action": self.action.value,
        }
        if self.line_number is not None:
            result["line_number"] = self.line_number
        if self.context is not None:
            result["context"] = self.context
        return result


__all__ = [
    "Severity",
    "Action",
    "SafetyPattern",
    "SafetyViolation",
]
