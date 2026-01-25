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

"""Safety pattern registry (Phase 6.1).

This module provides a centralized registry for safety patterns:
- Register patterns from Python code
- Load patterns from YAML files
- Scan content with domain filtering
- Backward compatible with existing SafetyRegistry

Design Principles:
- Thread-safe: All operations protected by locks
- Singleton: Single global registry
- Domain-aware: Patterns can be scoped to verticals
- Scanner-compatible: Works with ISafetyScanner protocol

Example:
    from victor.framework.safety.registry import SafetyPatternRegistry
    from victor.framework.safety.types import SafetyPattern, Severity

    # Get singleton instance
    registry = SafetyPatternRegistry.get_instance()

    # Register pattern
    pattern = SafetyPattern(
        name="api_key",
        pattern=r"AKIA[0-9A-Z]{16}",
        severity=Severity.HIGH,
        message="AWS API key detected",
    )
    registry.register_pattern(pattern)

    # Scan content
    violations = registry.scan(content, domain="coding")
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

import yaml

from victor.framework.safety.types import (
    SafetyPattern,
    SafetyViolation,
    Severity,
    Action,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ScannerProtocol(Protocol):
    """Protocol for safety scanners (ISafetyScanner compatible)."""

    def scan(self, content: str) -> List[SafetyViolation]:
        """Scan content for safety violations."""
        ...


def validate_pattern_yaml(data: Dict[str, Any]) -> List[str]:
    """Validate YAML pattern data against schema.

    Args:
        data: YAML data dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: List[str] = []

    if "patterns" not in data:
        # Empty file is valid
        return errors

    patterns = data.get("patterns", [])
    if not isinstance(patterns, list):
        errors.append("'patterns' must be a list")
        return errors

    valid_severities = {s.value for s in Severity}
    valid_actions = {a.value for a in Action}

    for i, pattern in enumerate(patterns):
        prefix = f"patterns[{i}]"

        if not isinstance(pattern, dict):
            errors.append(f"{prefix}: must be a dictionary")
            continue

        # Required fields
        if "name" not in pattern:
            errors.append(f"{prefix}: missing required field 'name'")
        if "pattern" not in pattern:
            errors.append(f"{prefix}: missing required field 'pattern'")
        if "severity" not in pattern:
            errors.append(f"{prefix}: missing required field 'severity'")
        elif pattern.get("severity", "").lower() not in valid_severities:
            errors.append(
                f"{prefix}: invalid severity '{pattern.get('severity')}'. "
                f"Valid values: {sorted(valid_severities)}"
            )
        if "message" not in pattern:
            errors.append(f"{prefix}: missing required field 'message'")

        # Optional field validation
        if "action" in pattern:
            if pattern["action"].lower() not in valid_actions:
                errors.append(
                    f"{prefix}: invalid action '{pattern['action']}'. "
                    f"Valid values: {sorted(valid_actions)}"
                )

        if "domains" in pattern and not isinstance(pattern["domains"], list):
            errors.append(f"{prefix}: 'domains' must be a list")

    return errors


class SafetyPatternRegistry:
    """Thread-safe singleton registry for safety patterns.

    Provides centralized registration, YAML loading, and domain-aware
    scanning for safety patterns across all verticals.

    Thread Safety:
        All public methods are protected by a reentrant lock.

    Backward Compatibility:
        Supports registering ISafetyScanner instances for integration
        with existing safety infrastructure.
    """

    _instance: Optional["SafetyPatternRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize safety pattern registry.

        Note: Use get_instance() instead of direct instantiation.
        """
        self._patterns: Dict[str, SafetyPattern] = {}
        self._scanners: Dict[str, ScannerProtocol] = {}
        self._op_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "SafetyPatternRegistry":
        """Get singleton registry instance.

        Returns:
            SafetyPatternRegistry singleton
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing).

        Warning: Only use in tests to reset state.
        """
        with cls._lock:
            cls._instance = None

    # =========================================================================
    # Pattern Management
    # =========================================================================

    def register_pattern(
        self,
        pattern: SafetyPattern,
        replace: bool = False,
    ) -> None:
        """Register a safety pattern.

        Args:
            pattern: SafetyPattern to register
            replace: If True, replace existing pattern

        Raises:
            ValueError: If pattern already registered and replace=False
        """
        with self._op_lock:
            if pattern.name in self._patterns and not replace:
                raise ValueError(
                    f"Pattern '{pattern.name}' is already registered. "
                    "Use replace=True to overwrite."
                )

            self._patterns[pattern.name] = pattern
            logger.debug(f"Registered safety pattern '{pattern.name}'")

    def unregister(self, name: str) -> bool:
        """Unregister a pattern.

        Args:
            name: Pattern name

        Returns:
            True if pattern was unregistered, False if not found
        """
        with self._op_lock:
            if name in self._patterns:
                del self._patterns[name]
                logger.debug(f"Unregistered safety pattern '{name}'")
                return True
            return False

    def get_pattern(self, name: str) -> Optional[SafetyPattern]:
        """Get pattern by name.

        Args:
            name: Pattern name

        Returns:
            SafetyPattern or None if not found
        """
        with self._op_lock:
            return self._patterns.get(name)

    def list_patterns(self) -> List[str]:
        """List all registered pattern names.

        Returns:
            List of pattern names
        """
        with self._op_lock:
            return list(self._patterns.keys())

    def list_by_severity(self, severity: Severity) -> List[str]:
        """List patterns by severity level.

        Args:
            severity: Severity to filter by

        Returns:
            List of matching pattern names
        """
        with self._op_lock:
            return [
                name
                for name, pattern in self._patterns.items()
                if pattern.severity == severity
            ]

    def list_by_domain(self, domain: str) -> List[str]:
        """List patterns that apply to a domain.

        Args:
            domain: Domain (vertical) name

        Returns:
            List of matching pattern names
        """
        with self._op_lock:
            return [
                name
                for name, pattern in self._patterns.items()
                if pattern.applies_to_domain(domain)
            ]

    # =========================================================================
    # Scanning
    # =========================================================================

    def scan(
        self,
        content: str,
        domain: str = "all",
    ) -> List[SafetyViolation]:
        """Scan content for safety violations.

        Args:
            content: Text content to scan
            domain: Domain to filter patterns (default: "all")

        Returns:
            List of SafetyViolation instances
        """
        violations: List[SafetyViolation] = []

        with self._op_lock:
            patterns = list(self._patterns.values())

        for pattern in patterns:
            if not pattern.applies_to_domain(domain):
                continue

            if pattern.matches(content):
                matches = pattern.find_matches(content)
                for matched_text in matches:
                    violations.append(
                        SafetyViolation(
                            pattern_name=pattern.name,
                            severity=pattern.severity,
                            message=pattern.message,
                            matched_text=matched_text,
                            action=pattern.action,
                        )
                    )

        return violations

    def scan_with_scanner(
        self,
        scanner_name: str,
        content: str,
    ) -> List[SafetyViolation]:
        """Scan content with a specific registered scanner.

        Args:
            scanner_name: Name of registered scanner
            content: Text content to scan

        Returns:
            List of SafetyViolation instances

        Raises:
            KeyError: If scanner not found
        """
        with self._op_lock:
            scanner = self._scanners.get(scanner_name)

        if scanner is None:
            raise KeyError(f"Scanner '{scanner_name}' not registered")

        return scanner.scan(content)

    # =========================================================================
    # Scanner Management (Backward Compatibility)
    # =========================================================================

    def register_scanner(
        self,
        name: str,
        scanner: ScannerProtocol,
    ) -> None:
        """Register a scanner for backward compatibility.

        This allows integration with existing ISafetyScanner implementations.

        Args:
            name: Scanner identifier
            scanner: Scanner instance implementing scan() method
        """
        with self._op_lock:
            self._scanners[name] = scanner
            logger.debug(f"Registered scanner '{name}'")

    def unregister_scanner(self, name: str) -> bool:
        """Unregister a scanner.

        Args:
            name: Scanner name

        Returns:
            True if scanner was unregistered, False if not found
        """
        with self._op_lock:
            if name in self._scanners:
                del self._scanners[name]
                return True
            return False

    def get_scanner(self, name: str) -> Optional[ScannerProtocol]:
        """Get registered scanner by name.

        Args:
            name: Scanner name

        Returns:
            Scanner instance or None if not found
        """
        with self._op_lock:
            return self._scanners.get(name)

    def list_scanners(self) -> List[str]:
        """List all registered scanner names.

        Returns:
            List of scanner names
        """
        with self._op_lock:
            return list(self._scanners.keys())

    # =========================================================================
    # YAML Loading
    # =========================================================================

    def load_from_yaml(self, path: Path) -> int:
        """Load patterns from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Number of patterns loaded

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If YAML is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Pattern file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Validate YAML schema
        errors = validate_pattern_yaml(data)
        if errors:
            raise ValueError(f"Invalid pattern YAML: {'; '.join(errors)}")

        count = 0
        for pattern_data in data.get("patterns", []):
            try:
                pattern = SafetyPattern.from_yaml_dict(pattern_data)
                self.register_pattern(pattern, replace=True)
                count += 1
            except Exception as e:
                logger.warning(
                    f"Failed to load pattern from YAML: {e}",
                    exc_info=True,
                )

        logger.info(f"Loaded {count} patterns from {path}")
        return count

    def load_from_yaml_string(self, yaml_content: str) -> int:
        """Load patterns from YAML string.

        Args:
            yaml_content: YAML content string

        Returns:
            Number of patterns loaded
        """
        data = yaml.safe_load(yaml_content) or {}

        errors = validate_pattern_yaml(data)
        if errors:
            raise ValueError(f"Invalid pattern YAML: {'; '.join(errors)}")

        count = 0
        for pattern_data in data.get("patterns", []):
            try:
                pattern = SafetyPattern.from_yaml_dict(pattern_data)
                self.register_pattern(pattern, replace=True)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load pattern: {e}")

        return count

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry stats
        """
        with self._op_lock:
            severity_counts: Dict[str, int] = {}
            for pattern in self._patterns.values():
                sev = pattern.severity.value
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            return {
                "pattern_count": len(self._patterns),
                "scanner_count": len(self._scanners),
                "severity_counts": severity_counts,
                "patterns": self.list_patterns(),
                "scanners": self.list_scanners(),
            }


__all__ = [
    "SafetyPatternRegistry",
    "validate_pattern_yaml",
    "ScannerProtocol",
]
