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

"""YAML integration for the validation framework.

This module provides:
- YAML-based validator configuration
- Workflow YAML integration for validation nodes
- Handler definitions in YAML

Example workflow YAML:

    workflows:
      my_workflow:
        nodes:
          - id: validate_input
            type: validate_pipeline
            validators:
              - type: threshold
                field: score
                min: 0
                max: 100
              - type: pattern
                field: email
                pattern_type: email
            on_failure: halt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Dict, List, Optional, Union

from victor.framework.validation.pipeline import (
    ChainHandler,
    ConditionalHandler,
    HaltHandler,
    RetryHandler,
    SkipHandler,
    ValidationAction,
    ValidationConfig,
    ValidationHandler,
    ValidationPipeline,
)
from victor.framework.validation.validators import (
    CompositeLogic,
    CompositeValidator,
    ConditionalValidator,
    LengthValidator,
    PatternValidator,
    PresenceValidator,
    RangeValidator,
    ThresholdValidator,
    TransformingValidator,
    TypeValidator,
    ValidatorProtocol,
)

logger = logging.getLogger(__name__)


# =============================================================================
# YAML Configuration Schema
# =============================================================================


@dataclass
class ValidatorConfig:
    """Configuration for a single validator from YAML.

    Attributes:
        type: Validator type (threshold, range, presence, pattern, type, length, composite)
        field_name: Field to validate (renamed from 'field' to avoid shadowing with dataclass.field)
        config: Validator-specific configuration
    """

    type: str
    field_name: Optional[str] = None
    config: Dict[str, Any] = dataclass_field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidatorConfig:
        """Create from YAML dict."""
        return cls(
            type=data.get("type", ""),
            field_name=data.get("field"),  # Map 'field' in YAML to 'field_name'
            config=data.get("config", data),
        )


@dataclass
class HandlerConfig:
    """Configuration for a validation handler from YAML.

    Attributes:
        type: Handler type (halt, skip, retry, conditional, chain)
        config: Handler-specific configuration
    """

    type: str
    config: Dict[str, Any] = dataclass_field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HandlerConfig:
        """Create from YAML dict."""
        # Support both 'type' field and direct handler specification
        if isinstance(data, str):
            return cls(type=data)
        return cls(
            type=data.get("type", "halt"),
            config={k: v for k, v in data.items() if k != "type"},
        )


@dataclass
class ValidationNodeConfig:
    """Configuration for a validation node in workflow YAML.

    Attributes:
        validators: List of validator configurations
        handler: Handler configuration
        halt_on_error: Whether to halt on first error
        collect_all_errors: Whether to collect all errors
        max_retries: Maximum retry attempts
        timeout_seconds: Optional timeout
        enable_logging: Whether to log validation steps
    """

    validators: List[ValidatorConfig] = dataclass_field(default_factory=list)
    handler: Optional[HandlerConfig] = None
    halt_on_error: bool = True
    collect_all_errors: bool = True
    max_retries: int = 3
    timeout_seconds: Optional[float] = None
    enable_logging: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationNodeConfig":
        """Create from YAML dict."""
        validators = [
            ValidatorConfig.from_dict(v)
            for v in data.get("validators", [])
        ]

        handler_config = data.get("handler")
        handler = HandlerConfig.from_dict(handler_config) if handler_config else None

        return cls(
            validators=validators,
            handler=handler,
            halt_on_error=data.get("halt_on_error", True),
            collect_all_errors=data.get("collect_all_errors", True),
            max_retries=data.get("max_retries", 3),
            timeout_seconds=data.get("timeout_seconds"),
            enable_logging=data.get("enable_logging", True),
        )


# =============================================================================
# Validator Factory
# =============================================================================


class ValidatorFactory:
    """Factory for creating validators from YAML configuration.

    Supports all built-in validators and can be extended with custom ones.
    """

    # Built-in validator types
    VALIDATOR_TYPES = {
        "threshold": ThresholdValidator,
        "range": RangeValidator,
        "presence": PresenceValidator,
        "pattern": PatternValidator,
        "type": TypeValidator,
        "length": LengthValidator,
        "composite": CompositeValidator,
        "conditional": ConditionalValidator,
        "transforming": TransformingValidator,
    }

    def __init__(self):
        """Initialize the factory."""
        self._custom_validators: Dict[str, type[ValidatorProtocol]] = {}

    def register_validator(
        self,
        name: str,
        validator_class: type[ValidatorProtocol],
    ) -> None:
        """Register a custom validator type.

        Args:
            name: Validator type name
            validator_class: Validator class
        """
        self._custom_validators[name] = validator_class

    def create_validator(
        self,
        config: ValidatorConfig,
    ) -> ValidatorProtocol:
        """Create a validator from configuration.

        Args:
            config: Validator configuration

        Returns:
            Validator instance

        Raises:
            ValueError: If validator type is unknown
        """
        validator_type = config.type.lower()

        # Check custom validators first
        if validator_type in self._custom_validators:
            return self._create_custom_validator(config)

        # Check built-in validators
        if validator_type == "threshold":
            return self._create_threshold_validator(config)
        elif validator_type == "range":
            return self._create_range_validator(config)
        elif validator_type == "presence":
            return self._create_presence_validator(config)
        elif validator_type == "pattern":
            return self._create_pattern_validator(config)
        elif validator_type == "type":
            return self._create_type_validator(config)
        elif validator_type == "length":
            return self._create_length_validator(config)
        elif validator_type == "composite":
            return self._create_composite_validator(config)
        elif validator_type == "conditional":
            return self._create_conditional_validator(config)
        elif validator_type == "transforming":
            return self._create_transforming_validator(config)
        else:
            raise ValueError(f"Unknown validator type: {validator_type}")

    def _create_threshold_validator(
        self,
        config: ValidatorConfig,
    ) -> ThresholdValidator:
        """Create a threshold validator."""
        cfg = config.config
        return ThresholdValidator(
            field=config.field_name,
            min_value=cfg.get("min"),
            max_value=cfg.get("max"),
            min_inclusive=cfg.get("min_inclusive", True),
            max_inclusive=cfg.get("max_inclusive", True),
            error_code=cfg.get("error_code"),
        )

    def _create_range_validator(
        self,
        config: ValidatorConfig,
    ) -> RangeValidator:
        """Create a range validator."""
        cfg = config.config
        return RangeValidator(
            field=config.field_name,
            min_value=cfg.get("min"),
            max_value=cfg.get("max"),
            exclusive_min=cfg.get("exclusive_min", False),
            exclusive_max=cfg.get("exclusive_max", False),
            error_code=cfg.get("error_code"),
        )

    def _create_presence_validator(
        self,
        config: ValidatorConfig,
    ) -> PresenceValidator:
        """Create a presence validator."""
        cfg = config.config
        return PresenceValidator(
            field=config.field_name,
            required=cfg.get("required", True),
            allow_empty=cfg.get("allow_empty", True),
            check_truthiness=cfg.get("check_truthiness", False),
            error_code=cfg.get("error_code"),
        )

    def _create_pattern_validator(
        self,
        config: ValidatorConfig,
    ) -> PatternValidator:
        """Create a pattern validator."""
        cfg = config.config
        return PatternValidator(
            field=config.field_name,
            pattern=cfg.get("pattern"),
            pattern_type=cfg.get("pattern_type"),
            flags=cfg.get("flags", 0),
            error_code=cfg.get("error_code"),
        )

    def _create_type_validator(
        self,
        config: ValidatorConfig,
    ) -> TypeValidator:
        """Create a type validator."""
        cfg = config.config
        expected_type = cfg.get("expected_type") or cfg.get("type")
        return TypeValidator(
            field=config.field_name,
            expected_type=expected_type,
            coerce=cfg.get("coerce", False),
            error_code=cfg.get("error_code"),
        )

    def _create_length_validator(
        self,
        config: ValidatorConfig,
    ) -> LengthValidator:
        """Create a length validator."""
        cfg = config.config
        return LengthValidator(
            field=config.field_name,
            min_length=cfg.get("min_length"),
            max_length=cfg.get("max_length"),
            exact_length=cfg.get("exact_length"),
            error_code=cfg.get("error_code"),
        )

    def _create_composite_validator(
        self,
        config: ValidatorConfig,
    ) -> CompositeValidator:
        """Create a composite validator."""
        cfg = config.config

        # Create child validators
        validators = []
        for validator_dict in cfg.get("validators", []):
            validator_config = ValidatorConfig.from_dict(validator_dict)
            validators.append(self.create_validator(validator_config))

        logic = cfg.get("logic", "all")
        return CompositeValidator(
            validators=validators,
            logic=logic,
            field=config.field_name,
            error_code=cfg.get("error_code"),
        )

    def _create_conditional_validator(
        self,
        config: ValidatorConfig,
    ) -> ConditionalValidator:
        """Create a conditional validator."""
        cfg = config.config

        # Create the inner validator
        inner_config = ValidatorConfig.from_dict(cfg["validator"])
        inner = self.create_validator(inner_config)

        # Create the condition function
        condition_path = cfg.get("condition")
        if condition_path:
            def condition(data: Dict[str, Any]) -> bool:
                parts = condition_path.split(".")
                value = data
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        return False
                return bool(value)
        else:
            def condition(data: Dict[str, Any]) -> bool:
                return data.get("_validate", True)

        return ConditionalValidator(
            validator=inner,
            condition=condition,
            field=config.field_name,
            error_code=cfg.get("error_code"),
        )

    def _create_transforming_validator(
        self,
        config: ValidatorConfig,
    ) -> TransformingValidator:
        """Create a transforming validator."""
        cfg = config.config

        # Create the inner validator
        inner_config = ValidatorConfig.from_dict(cfg["validator"])
        inner = self.create_validator(inner_config)

        # Get the transform type
        transform_type = cfg.get("transform", "strip")

        if transform_type == "strip":
            transform = str.strip
        elif transform_type == "lower":
            transform = str.lower
        elif transform_type == "upper":
            transform = str.upper
        elif transform_type == "title":
            transform = str.title
        elif transform_type == "trim":
            transform = lambda x: x.strip() if isinstance(x, str) else x
        else:
            transform = lambda x: x

        return TransformingValidator(
            validator=inner,
            transform=transform,
            field=config.field_name,
            error_code=cfg.get("error_code"),
        )

    def _create_custom_validator(
        self,
        config: ValidatorConfig,
    ) -> ValidatorProtocol:
        """Create a custom validator."""
        validator_class = self._custom_validators[config.type]
        return validator_class.from_config(config)


# =============================================================================
# Handler Factory
# =============================================================================


class HandlerFactory:
    """Factory for creating validation handlers from YAML configuration."""

    def create_handler(
        self,
        config: Optional[HandlerConfig] = None,
    ) -> ValidationHandler:
        """Create a handler from configuration.

        Args:
            config: Handler configuration (None = default HaltHandler)

        Returns:
            Handler instance
        """
        if config is None:
            return HaltHandler()

        handler_type = config.type.lower()

        if handler_type == "halt":
            return self._create_halt_handler(config)
        elif handler_type == "skip":
            return self._create_skip_handler(config)
        elif handler_type == "retry":
            return self._create_retry_handler(config)
        elif handler_type == "conditional":
            return self._create_conditional_handler(config)
        elif handler_type == "chain":
            return self._create_chain_handler(config)
        else:
            logger.warning(f"Unknown handler type: {handler_type}, using halt")
            return HaltHandler()

    def _create_halt_handler(self, config: HandlerConfig) -> HaltHandler:
        """Create a halt handler."""
        return HaltHandler()

    def _create_skip_handler(self, config: HandlerConfig) -> SkipHandler:
        """Create a skip handler."""
        return SkipHandler(
            log_warnings=config.config.get("log_warnings", True),
        )

    def _create_retry_handler(self, config: HandlerConfig) -> RetryHandler:
        """Create a retry handler."""
        return RetryHandler(
            max_retries=config.config.get("max_retries", 3),
            backoff_factor=config.config.get("backoff_factor", 1.0),
            retry_on_warnings=config.config.get("retry_on_warnings", False),
        )

    def _create_conditional_handler(
        self,
        config: HandlerConfig,
    ) -> ConditionalHandler:
        """Create a conditional handler."""
        cfg = config.config

        # Parse the action
        action_str = cfg.get("action", "halt")
        action = ValidationAction(action_str)

        # Create condition function
        def condition(result) -> bool:
            # Check severity level
            min_severity = cfg.get("min_severity", "error")
            if min_severity == "error":
                return len(result.errors) > 0
            elif min_severity == "warning":
                return len(result.errors) > 0 or len(result.warnings) > 0
            elif min_severity == "critical":
                return any(
                    i.severity.value == "critical" for i in result.errors
                )
            return not result.is_valid

        # Create fallback handler
        fallback_config = cfg.get("fallback", {"type": "halt"})
        fallback = self.create_handler(HandlerConfig.from_dict(fallback_config))

        return ConditionalHandler(
            condition=condition,
            action=action,
            fallback_handler=fallback,
        )

    def _create_chain_handler(self, config: HandlerConfig) -> ChainHandler:
        """Create a chain handler."""
        cfg = config.config
        handlers = []

        for handler_dict in cfg.get("handlers", []):
            handler_config = HandlerConfig.from_dict(handler_dict)
            handlers.append(self.create_handler(handler_config))

        return ChainHandler(handlers)


# =============================================================================
# Pipeline Builder from YAML
# =============================================================================


class ValidationPipelineBuilder:
    """Build validation pipelines from YAML configuration.

    Example:
        builder = ValidationPipelineBuilder()

        yaml_config = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ],
            "handler": {"type": "halt"},
        }

        pipeline = builder.from_dict(yaml_config)
    """

    def __init__(
        self,
        validator_factory: Optional[ValidatorFactory] = None,
        handler_factory: Optional[HandlerFactory] = None,
    ):
        """Initialize the builder.

        Args:
            validator_factory: Factory for creating validators
            handler_factory: Factory for creating handlers
        """
        self._validator_factory = validator_factory or ValidatorFactory()
        self._handler_factory = handler_factory or HandlerFactory()

    def from_dict(
        self,
        config: Dict[str, Any],
    ) -> ValidationPipeline:
        """Create a pipeline from dictionary configuration.

        Args:
            config: Pipeline configuration from YAML

        Returns:
            ValidationPipeline instance
        """
        node_config = ValidationNodeConfig.from_dict(config)

        # Create validators
        validators = []
        for validator_config in node_config.validators:
            try:
                validator = self._validator_factory.create_validator(validator_config)
                validators.append(validator)
            except Exception as e:
                logger.error(f"Failed to create validator: {e}")

        # Create handler
        handler = self._handler_factory.create_handler(node_config.handler)

        # Create pipeline
        return ValidationPipeline(
            validators=validators,
            handler=handler,
            halt_on_error=node_config.halt_on_error,
            collect_all_errors=node_config.collect_all_errors,
            max_retries=node_config.max_retries,
            timeout_seconds=node_config.timeout_seconds,
            enable_logging=node_config.enable_logging,
        )

    def from_yaml_node(
        self,
        node_data: Dict[str, Any],
    ) -> ValidationPipeline:
        """Create a pipeline from a workflow YAML node.

        Args:
            node_data: Node data from workflow YAML

        Returns:
            ValidationPipeline instance
        """
        return self.from_dict(node_data)

    def register_validator(
        self,
        name: str,
        validator_class: type[ValidatorProtocol],
    ) -> None:
        """Register a custom validator type.

        Args:
            name: Validator type name
            validator_class: Validator class
        """
        self._validator_factory.register_validator(name, validator_class)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_pipeline_from_yaml(
    yaml_config: Dict[str, Any],
) -> ValidationPipeline:
    """Create a validation pipeline from YAML configuration.

    Args:
        yaml_config: YAML configuration dictionary

    Returns:
        ValidationPipeline instance

    Example:
        yaml_config = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ],
        }
        pipeline = create_pipeline_from_yaml(yaml_config)
    """
    builder = ValidationPipelineBuilder()
    return builder.from_dict(yaml_config)


def validate_from_yaml(
    data: Dict[str, Any],
    yaml_config: Dict[str, Any],
) -> Any:
    """Validate data using YAML configuration.

    Args:
        data: Data to validate
        yaml_config: YAML configuration

    Returns:
        ValidationResult from running the pipeline
    """
    pipeline = create_pipeline_from_yaml(yaml_config)
    return pipeline.validate(data)


# Workflow node handler function
def validate_pipeline_handler(
    node_config: Dict[str, Any],
    graph_state: Dict[str, Any],
    orchestrator: Any = None,
) -> Dict[str, Any]:
    """Handler for validate_pipeline workflow nodes.

    This function can be registered as a node handler in the workflow system.

    Args:
        node_config: Node configuration from workflow YAML
        graph_state: Current workflow state
        orchestrator: Optional orchestrator instance

    Returns:
        Updated graph state
    """
    # Create pipeline from node config
    builder = ValidationPipelineBuilder()
    pipeline = builder.from_yaml_node(node_config)

    # Validate the state
    result = pipeline.validate(graph_state)

    # Add validation result to state
    state = dict(graph_state)
    state["_validation_result"] = result.to_dict()

    if not result.is_valid:
        state["_validation_errors"] = [str(e) for e in result.errors]

    return state


__all__ = [
    # Configuration classes
    "ValidatorConfig",
    "HandlerConfig",
    "ValidationNodeConfig",
    # Factories
    "ValidatorFactory",
    "HandlerFactory",
    "ValidationPipelineBuilder",
    # Convenience functions
    "create_pipeline_from_yaml",
    "validate_from_yaml",
    "validate_pipeline_handler",
]
