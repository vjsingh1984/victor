"""Framework-style safety rule helpers for external vertical compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple


class SafetyLevel(Enum):
    """Safety enforcement levels for operation checks."""

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SafetyConfig:
    """Configuration for framework-style safety enforcement."""

    level: SafetyLevel = SafetyLevel.MEDIUM
    require_confirmation: bool = False
    blocked_operations: list[str] = field(default_factory=list)
    audit_log: bool = True
    dry_run: bool = False

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SafetyConfig":
        """Create a safety config from a dictionary."""
        return cls(
            level=SafetyLevel(config.get("level", "medium")),
            require_confirmation=config.get("require_confirmation", False),
            blocked_operations=config.get("blocked_operations", []),
            audit_log=config.get("audit_log", True),
            dry_run=config.get("dry_run", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary representation."""
        return {
            "level": self.level.value,
            "require_confirmation": self.require_confirmation,
            "blocked_operations": list(self.blocked_operations),
            "audit_log": self.audit_log,
            "dry_run": self.dry_run,
        }


@dataclass
class SafetyRule:
    """Operation-level safety rule with a predicate callback."""

    name: str
    description: str
    check_fn: Callable[[str], bool]
    level: SafetyLevel = SafetyLevel.MEDIUM
    allow_override: bool = False


class SafetyEnforcer:
    """Framework-style operation safety enforcer."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.rules: list[SafetyRule] = []

    def add_rule(self, rule: SafetyRule) -> None:
        """Register a safety rule."""
        if any(existing.name == rule.name for existing in self.rules):
            raise ValueError(f"Rule '{rule.name}' already registered")
        self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for index, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(index)
                return True
        return False

    def check_operation(
        self,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check whether an operation is allowed."""
        del context
        if self.config.dry_run:
            return True, f"[DRY RUN] Would execute: {operation}"

        for blocked in self.config.blocked_operations:
            if blocked.lower() in operation.lower():
                return False, f"Blocked by configuration: '{blocked}'"

        for rule in self.rules:
            try:
                if not rule.check_fn(operation):
                    continue
            except Exception:
                continue

            if rule.level == SafetyLevel.HIGH:
                if rule.allow_override and self.config.level == SafetyLevel.LOW:
                    continue
                return False, f"Blocked by safety rule: {rule.name} - {rule.description}"

            if rule.level == SafetyLevel.MEDIUM:
                if self.config.level == SafetyLevel.LOW:
                    continue
                return False, f"Blocked by safety rule: {rule.name} - {rule.description}"

            if rule.level == SafetyLevel.LOW and self.config.level == SafetyLevel.HIGH:
                return False, f"Blocked by safety rule: {rule.name} - {rule.description}"

        return True, None

    def get_rules_by_level(self, level: SafetyLevel) -> list[SafetyRule]:
        """Return all rules at the requested level."""
        return [rule for rule in self.rules if rule.level == level]


__all__ = [
    "SafetyConfig",
    "SafetyEnforcer",
    "SafetyLevel",
    "SafetyRule",
]
