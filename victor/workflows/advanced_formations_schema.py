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

"""YAML schema and validation for advanced team formations.

This module provides schema definitions and validation functions for
advanced team formations in YAML workflow configurations.

Advanced Formations Supported:
    - dynamic: Automatically switches formation based on progress
    - adaptive: AI-powered formation selection
    - hybrid: Multi-phase execution combining multiple formations

Usage:
    from victor.workflows.advanced_formations_schema import (
        validate_advanced_formation_config,
        AdvancedFormationConfig,
        DynamicFormationConfig,
        AdaptiveFormationConfig,
        HybridFormationConfig,
    )

    # Validate YAML config
    config = validate_advanced_formation_config(yaml_data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models
# =============================================================================


class DynamicTriggerType(str, Enum):
    """Types of triggers for dynamic formation switching."""

    DEPENDENCIES_EMERGE = "dependencies_emerge"
    SEQUENTIAL_DEPENDENCY = "sequential_dependency"
    CONFLICT_DETECTED = "conflict_detected"
    CONSENSUS_NEEDED = "consensus_needed"
    DISAGREEMENT = "disagreement"
    TIME_PRESSURE = "time_pressure"
    SLOW_PROGRESS = "slow_progress"
    QUALITY_CONCERNS = "quality_concerns"
    VALIDATION_NEEDED = "validation_needed"


class FormationTarget(str, Enum):
    """Target formations for dynamic switching."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PIPELINE = "pipeline"
    CONSENSUS = "consensus"


class AdaptiveCriterion(str, Enum):
    """Criteria for adaptive formation selection."""

    COMPLEXITY = "complexity"
    DEADLINE = "deadline"
    RESOURCE_AVAILABILITY = "resource_availability"
    DEPENDENCY_LEVEL = "dependency_level"
    COLLABORATION_NEEDED = "collaboration_needed"
    UNCERTAINTY = "uncertainty"


@dataclass
class DynamicSwitchingRule:
    """A single switching rule for dynamic formation."""

    trigger: str
    target_formation: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"trigger": self.trigger, "target_formation": self.target_formation}


@dataclass
class DynamicFormationConfig:
    """Configuration for dynamic formation.

    Attributes:
        initial_formation: Starting formation (default: "parallel")
        switching_rules: List of switching rules
        max_switches: Maximum number of formation switches
        enable_auto_detection: Enable automatic dependency/conflict detection
    """

    initial_formation: str = "parallel"
    switching_rules: List[Dict[str, str]] = field(default_factory=list)
    max_switches: int = 5
    enable_auto_detection: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_formation": self.initial_formation,
            "switching_rules": self.switching_rules,
            "max_switches": self.max_switches,
            "enable_auto_detection": self.enable_auto_detection,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicFormationConfig":
        """Create from dictionary."""
        return cls(
            initial_formation=data.get("initial_formation", "parallel"),
            switching_rules=data.get("switching_rules", []),
            max_switches=data.get("max_switches", 5),
            enable_auto_detection=data.get("enable_auto_detection", True),
        )


@dataclass
class AdaptiveFormationConfig:
    """Configuration for adaptive formation.

    Attributes:
        criteria: List of criteria to consider
        default_formation: Default formation
        fallback_formation: Fallback formation
        scoring_weights: Custom scoring weights (optional)
        use_ml: Use ML model if available
    """

    criteria: List[str] = field(
        default_factory=lambda: ["complexity", "deadline", "resource_availability"]
    )
    default_formation: str = "parallel"
    fallback_formation: str = "sequential"
    scoring_weights: Optional[Dict[str, Dict[str, float]]] = None
    use_ml: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "criteria": self.criteria,
            "default_formation": self.default_formation,
            "fallback_formation": self.fallback_formation,
            "scoring_weights": self.scoring_weights,
            "use_ml": self.use_ml,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveFormationConfig":
        """Create from dictionary."""
        return cls(
            criteria=data.get("criteria", ["complexity", "deadline", "resource_availability"]),
            default_formation=data.get("default_formation", "parallel"),
            fallback_formation=data.get("fallback_formation", "sequential"),
            scoring_weights=data.get("scoring_weights"),
            use_ml=data.get("use_ml", False),
        )


@dataclass
class HybridPhaseConfig:
    """Configuration for a single hybrid phase.

    Attributes:
        formation: Formation to use
        goal: Goal of this phase
        duration_budget: Time budget in seconds (None = no limit)
        iteration_limit: Max iterations (None = no limit)
        completion_criteria: Completion criteria description
    """

    formation: str
    goal: str
    duration_budget: Optional[float] = None
    iteration_limit: Optional[int] = None
    completion_criteria: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "formation": self.formation,
            "goal": self.goal,
            "duration_budget": self.duration_budget,
            "iteration_limit": self.iteration_limit,
            "completion_criteria": self.completion_criteria,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridPhaseConfig":
        """Create from dictionary."""
        return cls(
            formation=data["formation"],
            goal=data["goal"],
            duration_budget=data.get("duration_budget"),
            iteration_limit=data.get("iteration_limit"),
            completion_criteria=data.get("completion_criteria"),
        )


@dataclass
class HybridFormationConfig:
    """Configuration for hybrid formation.

    Attributes:
        phases: List of phases to execute
        enable_phase_logging: Log phase transitions
        stop_on_first_failure: Stop if any phase fails
    """

    phases: List[HybridPhaseConfig]
    enable_phase_logging: bool = True
    stop_on_first_failure: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phases": [phase.to_dict() for phase in self.phases],
            "enable_phase_logging": self.enable_phase_logging,
            "stop_on_first_failure": self.stop_on_first_failure,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridFormationConfig":
        """Create from dictionary."""
        phases = [HybridPhaseConfig.from_dict(phase_data) for phase_data in data.get("phases", [])]
        return cls(
            phases=phases,
            enable_phase_logging=data.get("enable_phase_logging", True),
            stop_on_first_failure=data.get("stop_on_first_failure", False),
        )


@dataclass
class AdvancedFormationConfig:
    """Unified configuration for any advanced formation.

    Attributes:
        type: Formation type (dynamic, adaptive, hybrid)
        dynamic_config: Dynamic formation config (if type is "dynamic")
        adaptive_config: Adaptive formation config (if type is "adaptive")
        hybrid_config: Hybrid formation config (if type is "hybrid")
    """

    type: str
    dynamic_config: Optional[DynamicFormationConfig] = None
    adaptive_config: Optional[AdaptiveFormationConfig] = None
    hybrid_config: Optional[HybridFormationConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"type": self.type}

        if self.dynamic_config:
            result["dynamic_config"] = self.dynamic_config.to_dict()
        if self.adaptive_config:
            result["adaptive_config"] = self.adaptive_config.to_dict()
        if self.hybrid_config:
            result["hybrid_config"] = self.hybrid_config.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvancedFormationConfig":
        """Create from dictionary."""
        formation_type = data.get("type", "")

        config = cls(type=formation_type)

        if formation_type == "dynamic":
            dynamic_data = data.get("dynamic_config", {})
            config.dynamic_config = DynamicFormationConfig.from_dict(dynamic_data)
        elif formation_type == "adaptive":
            adaptive_data = data.get("adaptive_config", {})
            config.adaptive_config = AdaptiveFormationConfig.from_dict(adaptive_data)
        elif formation_type == "hybrid":
            hybrid_data = data.get("hybrid_config", {})
            config.hybrid_config = HybridFormationConfig.from_dict(hybrid_data)

        return config


# =============================================================================
# Validation Functions
# =============================================================================


class ValidationError(Exception):
    """Validation error for advanced formation config."""

    pass


def validate_dynamic_config(config: DynamicFormationConfig) -> List[str]:
    """Validate dynamic formation configuration.

    Args:
        config: Dynamic formation config

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Validate initial formation
    valid_formations = ["sequential", "parallel", "hierarchical", "pipeline", "consensus"]
    if config.initial_formation not in valid_formations:
        errors.append(
            f"Invalid initial_formation '{config.initial_formation}'. "
            f"Must be one of: {valid_formations}"
        )

    # Validate max switches
    if config.max_switches < 0:
        errors.append("max_switches must be non-negative")

    if config.max_switches > 20:
        errors.append("max_switches should not exceed 20")

    # Validate switching rules
    for i, rule in enumerate(config.switching_rules):
        if not isinstance(rule, dict):
            errors.append(f"switching_rules[{i}] must be a dictionary")
            continue

        if "trigger" not in rule:
            errors.append(f"switching_rules[{i}] missing 'trigger' key")

        if "target_formation" not in rule:
            errors.append(f"switching_rules[{i}] missing 'target_formation' key")
        elif rule["target_formation"] not in valid_formations:
            errors.append(
                f"switching_rules[{i}] has invalid target_formation "
                f"'{rule['target_formation']}'. Must be one of: {valid_formations}"
            )

    return errors


def validate_adaptive_config(config: AdaptiveFormationConfig) -> List[str]:
    """Validate adaptive formation configuration.

    Args:
        config: Adaptive formation config

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Validate criteria
    valid_criteria = [
        "complexity",
        "deadline",
        "resource_availability",
        "dependency_level",
        "collaboration_needed",
        "uncertainty",
    ]

    for criterion in config.criteria:
        if criterion not in valid_criteria:
            errors.append(f"Invalid criterion '{criterion}'. Must be one of: {valid_criteria}")

    # Validate formations
    valid_formations = ["sequential", "parallel", "hierarchical", "pipeline", "consensus"]

    if config.default_formation not in valid_formations:
        errors.append(
            f"Invalid default_formation '{config.default_formation}'. "
            f"Must be one of: {valid_formations}"
        )

    if config.fallback_formation not in valid_formations:
        errors.append(
            f"Invalid fallback_formation '{config.fallback_formation}'. "
            f"Must be one of: {valid_formations}"
        )

    # Validate scoring weights if provided
    if config.scoring_weights:
        for formation, weights in config.scoring_weights.items():
            if formation not in valid_formations:
                errors.append(
                    f"Invalid formation in scoring_weights: '{formation}'. "
                    f"Must be one of: {valid_formations}"
                )

            if not isinstance(weights, dict):
                errors.append(f"scoring_weights['{formation}'] must be a dictionary")
                continue

            for criterion, weight in weights.items():
                if not isinstance(weight, (int, float)):
                    errors.append(
                        f"scoring_weights['{formation}']['{criterion}'] " f"must be a number"
                    )

    return errors


def validate_hybrid_config(config: HybridFormationConfig) -> List[str]:
    """Validate hybrid formation configuration.

    Args:
        config: Hybrid formation config

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Validate phases
    if not config.phases:
        errors.append("hybrid formation must have at least one phase")
        return errors

    valid_formations = ["sequential", "parallel", "hierarchical", "pipeline", "consensus"]

    for i, phase in enumerate(config.phases):
        if phase.formation not in valid_formations:
            errors.append(
                f"phases[{i}] has invalid formation '{phase.formation}'. "
                f"Must be one of: {valid_formations}"
            )

        if not phase.goal:
            errors.append(f"phases[{i}] must have a goal")

        if phase.duration_budget is not None and phase.duration_budget <= 0:
            errors.append(f"phases[{i}] duration_budget must be positive")

        if phase.iteration_limit is not None and phase.iteration_limit <= 0:
            errors.append(f"phases[{i}] iteration_limit must be positive")

    return errors


def validate_advanced_formation_config(config: AdvancedFormationConfig) -> List[str]:
    """Validate advanced formation configuration.

    Args:
        config: Advanced formation config

    Returns:
        List of validation errors (empty if valid)

    Raises:
        ValidationError: If validation fails
    """
    errors = []

    # Validate type
    valid_types = ["dynamic", "adaptive", "hybrid"]
    if config.type not in valid_types:
        errors.append(f"Invalid formation type '{config.type}'. Must be one of: {valid_types}")
        return errors

    # Type-specific validation
    if config.type == "dynamic":
        if config.dynamic_config is None:
            errors.append("dynamic formation requires dynamic_config")
        else:
            errors.extend(validate_dynamic_config(config.dynamic_config))

    elif config.type == "adaptive":
        if config.adaptive_config is None:
            errors.append("adaptive formation requires adaptive_config")
        else:
            errors.extend(validate_adaptive_config(config.adaptive_config))

    elif config.type == "hybrid":
        if config.hybrid_config is None:
            errors.append("hybrid formation requires hybrid_config")
        else:
            errors.extend(validate_hybrid_config(config.hybrid_config))

    return errors


def parse_advanced_formation_from_yaml(yaml_data: Dict[str, Any]) -> AdvancedFormationConfig:
    """Parse advanced formation configuration from YAML data.

    Args:
        yaml_data: Dictionary from YAML parsing

    Returns:
        AdvancedFormationConfig instance

    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        config = AdvancedFormationConfig.from_dict(yaml_data)

        # Validate
        errors = validate_advanced_formation_config(config)
        if errors:
            raise ValidationError(
                "Invalid advanced formation configuration:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        return config

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to parse advanced formation config: {e}")


# =============================================================================
# Example YAML Schema
# =============================================================================

ADVANCED_FORMATION_YAML_SCHEMA = """
# Advanced Team Formation YAML Schema

# Dynamic Formation - switches based on progress
type: dynamic
dynamic_config:
  initial_formation: parallel  # sequential, parallel, hierarchical, pipeline, consensus
  max_switches: 5
  enable_auto_detection: true
  switching_rules:
    - trigger: dependencies_emerge
      target_formation: sequential
    - trigger: conflict_detected
      target_formation: consensus
    - trigger: consensus_needed
      target_formation: consensus
    - trigger: time_pressure
      target_formation: parallel

---

# Adaptive Formation - AI-powered selection
type: adaptive
adaptive_config:
  criteria:
    - complexity
    - deadline
    - resource_availability
    - dependency_level
    - collaboration_needed
  default_formation: parallel
  fallback_formation: sequential
  use_ml: false  # Set to true to use ML model if available
  # Optional: custom scoring weights
  scoring_weights:
    parallel:
      complexity: 0.8
      deadline: 0.7
      resource_availability: 0.9
      dependency_level: -0.5
      uncertainty: 0.3

---

# Hybrid Formation - multi-phase execution
type: hybrid
hybrid_config:
  enable_phase_logging: true
  stop_on_first_failure: false
  phases:
    - formation: parallel
      goal: Explore the problem space rapidly
      duration_budget: 30.0  # seconds
    - formation: sequential
      goal: Synthesize findings from exploration
      iteration_limit: 3
    - formation: consensus
      goal: Validate and agree on final solution
      completion_criteria: All team members satisfied
"""


__all__ = [
    # Configuration classes
    "DynamicFormationConfig",
    "AdaptiveFormationConfig",
    "HybridFormationConfig",
    "HybridPhaseConfig",
    "AdvancedFormationConfig",
    # Enums
    "DynamicTriggerType",
    "FormationTarget",
    "AdaptiveCriterion",
    # Validation
    "validate_dynamic_config",
    "validate_adaptive_config",
    "validate_hybrid_config",
    "validate_advanced_formation_config",
    "parse_advanced_formation_from_yaml",
    "ValidationError",
    # Schema
    "ADVANCED_FORMATION_YAML_SCHEMA",
]
