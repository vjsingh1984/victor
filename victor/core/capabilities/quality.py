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

"""Quality capability abstraction (Phase 8.1).

This module provides a generic quality threshold capability that can be used
across different verticals with consistent enforcement patterns:

- Coding: Test coverage thresholds
- RAG: Retrieval precision/recall thresholds
- DataAnalysis: Data quality/completeness thresholds

Example:
    from victor.core.capabilities.quality import QualityCapability, Enforcement

    # Create quality capability for test coverage
    coverage_cap = QualityCapability(
        name="test_coverage",
        metric_name="coverage",
        threshold=0.8,
        enforcement=Enforcement.WARN,
    )

    # Check a value
    result = coverage_cap.check(0.75)
    if not result.passed:
        print(f"Coverage {result.value} below threshold {result.threshold}")
        if result.enforcement == Enforcement.BLOCK:
            raise Exception("Operation blocked due to low coverage")

    # Convert to CapabilityDefinition for registry
    definition = coverage_cap.to_capability_definition()
    registry.register(definition)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from victor.core.capabilities.types import CapabilityDefinition, CapabilityType

logger = logging.getLogger(__name__)


class Enforcement(str, Enum):
    """Enforcement action when quality threshold is not met.

    Values:
        BLOCK: Block the operation, require user override
        WARN: Show warning, allow to proceed
        LOG: Log the occurrence, no user notification
    """

    BLOCK = "block"
    WARN = "warn"
    LOG = "log"

    @classmethod
    def from_string(cls, value: str) -> "Enforcement":
        """Create Enforcement from string value (case-insensitive).

        Args:
            value: String representation

        Returns:
            Enforcement enum value

        Raises:
            ValueError: If value is not valid
        """
        normalized = value.lower().strip()
        for member in cls:
            if member.value == normalized:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Invalid enforcement: '{value}'. Valid values: {valid}")


@dataclass
class QualityResult:
    """Result from a quality threshold check.

    Attributes:
        passed: Whether the value met the threshold
        value: The actual value checked
        threshold: The threshold that was checked against
        metric_name: Name of the metric
        enforcement: Enforcement action if threshold not met (None if passed)
    """

    passed: bool
    value: float
    threshold: float
    metric_name: str = ""
    enforcement: Optional[Enforcement] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with result details
        """
        result: Dict[str, Any] = {
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "metric_name": self.metric_name,
        }
        if self.enforcement is not None:
            result["enforcement"] = self.enforcement.value
        return result

    def get_message(self) -> str:
        """Generate human-readable message for this result.

        Returns:
            Status message describing the result
        """
        if self.passed:
            return (
                f"Quality check passed: {self.metric_name} {self.value:.2%} "
                f"meets threshold {self.threshold:.2%}"
            )
        else:
            action = f" [{self.enforcement.value}]" if self.enforcement else ""
            return (
                f"Quality check failed{action}: {self.metric_name} {self.value:.2%} "
                f"below threshold {self.threshold:.2%}"
            )


@dataclass
class QualityCapability:
    """Generic quality threshold capability.

    Provides consistent quality enforcement across verticals:
    - Coding: test coverage thresholds
    - RAG: retrieval precision thresholds
    - DataAnalysis: data quality thresholds

    Attributes:
        name: Unique capability identifier
        metric_name: Name of the metric being checked (e.g., "coverage", "precision")
        threshold: Quality threshold (0.0 - 1.0)
        enforcement: Action when threshold not met (default: WARN)
        description: Human-readable description
        tags: Discovery and filtering tags

    Example:
        cap = QualityCapability(
            name="test_coverage",
            metric_name="coverage",
            threshold=0.8,
            enforcement=Enforcement.WARN,
        )

        result = cap.check(0.75)
        if not result.passed:
            print(f"Coverage below threshold: {result.get_message()}")
    """

    name: str
    metric_name: str
    threshold: float  # 0.0 - 1.0
    enforcement: Enforcement = field(default=Enforcement.WARN)
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def check(self, value: float) -> QualityResult:
        """Check if a value meets the quality threshold.

        Args:
            value: The value to check (0.0 - 1.0)

        Returns:
            QualityResult with check outcome
        """
        passed = value >= self.threshold

        return QualityResult(
            passed=passed,
            value=value,
            threshold=self.threshold,
            metric_name=self.metric_name,
            enforcement=None if passed else self.enforcement,
        )

    def to_capability_definition(self) -> CapabilityDefinition:
        """Convert to CapabilityDefinition for registry.

        This allows QualityCapability instances to be registered
        with the CapabilityRegistry using a consistent format.

        Returns:
            CapabilityDefinition instance
        """
        # Build default config
        default_config = {
            "threshold": self.threshold,
            "metric_name": self.metric_name,
            "enforcement": self.enforcement.value,
        }

        # Build config schema
        config_schema = {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": f"Quality threshold for {self.metric_name}",
                },
                "metric_name": {
                    "type": "string",
                    "description": "Name of the metric being checked",
                },
                "enforcement": {
                    "type": "string",
                    "enum": ["block", "warn", "log"],
                    "description": "Action when threshold not met",
                },
            },
            "required": ["threshold"],
        }

        # Build tags
        tags = ["quality", "threshold", self.metric_name]
        tags.extend(self.tags)

        return CapabilityDefinition(
            name=self.name,
            capability_type=CapabilityType.MODE,
            description=self.description or f"Quality threshold for {self.metric_name}",
            config_schema=config_schema,
            default_config=default_config,
            tags=list(set(tags)),  # Deduplicate
        )

    @classmethod
    def from_capability_definition(cls, definition: CapabilityDefinition) -> "QualityCapability":
        """Create QualityCapability from a CapabilityDefinition.

        Args:
            definition: CapabilityDefinition to convert

        Returns:
            QualityCapability instance

        Raises:
            ValueError: If definition lacks required config
        """
        config = definition.default_config

        if "threshold" not in config:
            raise ValueError("CapabilityDefinition must have 'threshold' in default_config")

        enforcement_str = config.get("enforcement", "warn")
        enforcement = Enforcement.from_string(enforcement_str)

        return cls(
            name=definition.name,
            metric_name=config.get("metric_name", definition.name),
            threshold=config["threshold"],
            enforcement=enforcement,
            description=definition.description,
            tags=definition.tags.copy(),
        )


__all__ = [
    "QualityCapability",
    "QualityResult",
    "Enforcement",
]
