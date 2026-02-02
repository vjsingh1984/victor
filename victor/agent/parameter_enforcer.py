"""
Parameter Enforcement Decorator for Tool Execution.

This module implements a Decorator pattern to validate and enforce tool parameters,
providing automatic inference for missing required parameters when possible.

SOLID Principles Applied:
- Single Responsibility: Each class has one purpose (validation, inference, enforcement)
- Open/Closed: New inference strategies can be added without modifying existing code
- Liskov Substitution: All inference strategies are interchangeable
- Interface Segregation: ParameterSpec defines minimal interface
- Dependency Inversion: Enforcer depends on abstractions (specs), not concrete tools

Addresses GAP-9 from Grok/DeepSeek provider testing.
"""

from typing import Optional, Any
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import logging
import uuid

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Supported parameter types for validation and coercion."""

    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    FLOAT = "float"
    ARRAY = "array"
    OBJECT = "object"


class InferenceStrategy(Enum):
    """Strategies for inferring missing parameter values."""

    NONE = "none"  # No inference, must be provided
    FROM_CONTEXT = "from_context"  # Infer from context dict
    FROM_PREVIOUS_ARGS = "from_previous_args"  # Infer from previous tool args
    FROM_DEFAULT = "from_default"  # Use default value
    FROM_WORKING_DIR = "from_working_dir"  # Use current working directory


class ParameterValidationError(Exception):
    """Raised when parameter validation fails."""

    def __init__(
        self,
        param_name: str,
        message: str,
        provided_value: Any = None,
        expected_type: Optional[str] = None,
        examples: Optional[list[str]] = None,
        valid_range: Optional[dict[str, Any]] = None,
    ):
        self.param_name = param_name
        self.provided_value = provided_value
        self.expected_type = expected_type
        self.examples = examples or []
        self.valid_range = valid_range
        self.correlation_id = str(uuid.uuid4())[:8]

        # Build detailed error message
        error_parts = [f"Validation error for parameter '{param_name}': {message}"]

        if provided_value is not None:
            provided_type = type(provided_value).__name__
            error_parts.append(f"  Provided: '{provided_value}' (type: {provided_type})")

        if expected_type:
            error_parts.append(f"  Expected type: {expected_type}")

        if examples:
            error_parts.append(f"  Valid examples: {', '.join(examples[:3])}")

        if valid_range:
            range_parts = []
            if "min" in valid_range:
                range_parts.append(f"min={valid_range['min']}")
            if "max" in valid_range:
                range_parts.append(f"max={valid_range['max']}")
            if range_parts:
                error_parts.append(f"  Valid range: {', '.join(range_parts)}")

        # Add recovery hint
        error_parts.append(f"\n  [Correlation ID: {self.correlation_id}]")
        error_parts.append(f"  Recovery: Provide a valid value for '{param_name}'")

        full_message = "\n".join(error_parts)
        super().__init__(full_message)


class ParameterInferenceError(Exception):
    """Raised when a required parameter cannot be inferred."""

    def __init__(
        self,
        param_name: str,
        tool_name: str,
        inference_strategy: Optional[str] = None,
        available_context_keys: Optional[list[str]] = None,
    ):
        self.param_name = param_name
        self.tool_name = tool_name
        self.inference_strategy = inference_strategy
        self.available_context_keys = available_context_keys or []
        self.correlation_id = str(uuid.uuid4())[:8]

        # Build detailed error message
        error_parts = [f"Cannot infer required parameter '{param_name}' for tool '{tool_name}'"]

        if inference_strategy:
            error_parts.append(f"  Inference strategy: {inference_strategy}")

        if available_context_keys:
            error_parts.append(f"  Available context keys: {', '.join(available_context_keys[:5])}")

        # Add recovery hint
        error_parts.append(f"\n  [Correlation ID: {self.correlation_id}]")
        error_parts.append(
            f"  Recovery: Provide value for '{param_name}' explicitly, "
            f"or ensure it's available in context"
        )

        full_message = "\n".join(error_parts)
        super().__init__(full_message)


@dataclass
class ParameterSpec:
    """Specification for a tool parameter."""

    name: str
    param_type: ParameterType
    required: bool = True
    default: Any = None
    description: str = ""
    inference_strategy: InferenceStrategy = InferenceStrategy.NONE
    inference_key: Optional[str] = None  # Key to look up in context/previous args
    inference_keys: Optional[list[str]] = None  # Multiple keys to try


@dataclass
class ParameterValidationResult:
    """Result of parameter enforcement validation.

    Renamed from ValidationResult to be semantically distinct:
    - ToolValidationResult (victor.tools.base): Tool parameter validation
    - ConfigValidationResult (victor.core.validation): Configuration validation
    - ContentValidationResult (victor.framework.middleware): Content validation
    - ParameterValidationResult (here): Parameter enforcement with missing_required
    - CodeValidationResult (victor.evaluation.correction.types): Code validation
    """

    is_valid: bool
    missing_required: list[str] = field(default_factory=list)
    type_errors: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ParameterEnforcer:
    """
    Enforces parameter requirements for tool execution.

    Validates that required parameters are present, provides defaults for
    optional parameters, and attempts to infer missing required parameters
    from context when possible.
    """

    def __init__(
        self,
        tool_name: str,
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self.tool_name = tool_name
        self.specs = {spec.name: spec for spec in parameter_specs}
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate(self, args: dict[str, Any]) -> ParameterValidationResult:
        """
        Validate arguments against parameter specifications.

        Args:
            args: The arguments to validate

        Returns:
            ParameterValidationResult with validation status and any errors
        """
        missing_required: list[str] = []
        type_errors: dict[str, str] = {}

        for name, spec in self.specs.items():
            if spec.required:
                value = args.get(name)
                if value is None:
                    missing_required.append(name)

        is_valid = len(missing_required) == 0 and len(type_errors) == 0

        return ParameterValidationResult(
            is_valid=is_valid,
            missing_required=missing_required,
            type_errors=type_errors,
        )

    def enforce(
        self,
        args: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Enforce parameter requirements, filling in missing values where possible.

        Args:
            args: The arguments to enforce
            context: Optional context for inference

        Returns:
            Enforced arguments with missing values filled in

        Raises:
            ParameterInferenceError: If a required parameter cannot be inferred
            ParameterValidationError: If a parameter has an invalid value
        """
        context = context or {}
        enforced = dict(args)

        for name, spec in self.specs.items():
            current_value = enforced.get(name)

            # Handle None values
            if current_value is None:
                # Try to infer the value regardless of required status
                if spec.inference_strategy != InferenceStrategy.NONE:
                    inferred = self._infer_value(spec, context)
                    if inferred is not None:
                        enforced[name] = inferred
                        self._logger.debug(f"Inferred {name}={inferred} for {self.tool_name}")
                        continue  # Move to next parameter

                # If we couldn't infer, try defaults or raise error
                if spec.required:
                    if spec.default is not None:
                        enforced[name] = spec.default
                    else:
                        raise ParameterInferenceError(name, self.tool_name)
                elif spec.default is not None:
                    enforced[name] = spec.default
            else:
                # Coerce type if needed
                try:
                    enforced[name] = self._coerce_type(current_value, spec)
                except (ValueError, TypeError) as e:
                    raise ParameterValidationError(name, str(e))

        return enforced

    def _infer_value(
        self,
        spec: ParameterSpec,
        context: dict[str, Any],
    ) -> Optional[Any]:
        """
        Attempt to infer a parameter value based on its inference strategy.

        Args:
            spec: The parameter specification
            context: Context for inference

        Returns:
            Inferred value or None if inference failed
        """
        if spec.inference_strategy == InferenceStrategy.NONE:
            return None

        if spec.inference_strategy == InferenceStrategy.FROM_WORKING_DIR:
            return context.get("working_directory")

        if spec.inference_strategy == InferenceStrategy.FROM_CONTEXT:
            key = spec.inference_key or spec.name
            return context.get(key)

        if spec.inference_strategy == InferenceStrategy.FROM_PREVIOUS_ARGS:
            return self._infer_from_previous_args(spec, context)

        if spec.inference_strategy == InferenceStrategy.FROM_DEFAULT:
            return spec.default

        return None

    def _infer_from_previous_args(
        self,
        spec: ParameterSpec,
        context: dict[str, Any],
    ) -> Optional[Any]:
        """
        Infer parameter value from previous tool arguments.

        Args:
            spec: The parameter specification
            context: Context containing previous_tool_args

        Returns:
            Inferred value or None
        """
        previous_args = context.get("previous_tool_args", {})

        # Get keys to search for
        keys_to_try = []
        if spec.inference_keys:
            keys_to_try = spec.inference_keys
        elif spec.inference_key:
            keys_to_try = [spec.inference_key]
        else:
            keys_to_try = [spec.name]

        # Search through all previous tool args
        for tool_args in previous_args.values():
            if isinstance(tool_args, dict):
                for key in keys_to_try:
                    if key in tool_args:
                        return tool_args[key]

        return None

    def _coerce_type(self, value: Any, spec: ParameterSpec) -> Any:
        """
        Coerce a value to the expected type.

        Args:
            value: The value to coerce
            spec: The parameter specification

        Returns:
            Coerced value

        Raises:
            ParameterValidationError: If coercion fails
        """
        if spec.param_type == ParameterType.STRING:
            return str(value) if value is not None else value

        if spec.param_type == ParameterType.INTEGER:
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    raise ParameterValidationError(
                        spec.name,
                        f"Cannot convert string '{value}' to integer",
                        provided_value=value,
                        expected_type="integer",
                        examples=["42", "100", "0"],
                    )
            if isinstance(value, float):
                return int(value)
            raise ParameterValidationError(
                spec.name,
                f"Cannot convert {type(value).__name__} to integer",
                provided_value=value,
                expected_type="integer",
                examples=["42", "100", "0"],
            )

        if spec.param_type == ParameterType.FLOAT:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    raise ParameterValidationError(
                        spec.name,
                        f"Cannot convert string '{value}' to float",
                        provided_value=value,
                        expected_type="float",
                        examples=["3.14", "2.5", "0.0"],
                    )
            raise ParameterValidationError(
                spec.name,
                f"Cannot convert {type(value).__name__} to float",
                provided_value=value,
                expected_type="float",
                examples=["3.14", "2.5", "0.0"],
            )

        if spec.param_type == ParameterType.BOOLEAN:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lower = value.lower()
                if lower in ("true", "1", "yes", "on"):
                    return True
                if lower in ("false", "0", "no", "off"):
                    return False
                raise ParameterValidationError(
                    spec.name,
                    f"Cannot convert string '{value}' to boolean",
                    provided_value=value,
                    expected_type="boolean",
                    examples=["true", "false", "yes", "no", "1", "0"],
                )
            if isinstance(value, (int, float)):
                return bool(value)
            raise ParameterValidationError(
                spec.name,
                f"Cannot convert {type(value).__name__} to boolean",
                provided_value=value,
                expected_type="boolean",
                examples=["true", "false", "1", "0"],
            )

        if spec.param_type == ParameterType.ARRAY:
            if isinstance(value, list):
                return value
            if isinstance(value, (tuple, set)):
                return list(value)
            raise ParameterValidationError(
                spec.name,
                f"Cannot convert {type(value).__name__} to array",
                provided_value=value,
                expected_type="array",
                examples=['["item1", "item2"]', '["a", "b", "c"]'],
            )

        if spec.param_type == ParameterType.OBJECT:
            if isinstance(value, dict):
                return value
            raise ParameterValidationError(
                spec.name,
                f"Cannot convert {type(value).__name__} to object",
                provided_value=value,
                expected_type="object",
                examples=['{"key": "value"}', '{"name": "test", "count": 5}'],
            )

        return value


def enforce_parameters(
    tool_name: str,
    specs: list[ParameterSpec],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to enforce parameters on a tool execution function.

    Usage:
        @enforce_parameters(
            tool_name="read",
            specs=[
                ParameterSpec(name="path", param_type=ParameterType.STRING, required=True),
                ParameterSpec(name="offset", param_type=ParameterType.INTEGER, default=0),
            ]
        )
        def execute(**kwargs):
            ...

    Args:
        tool_name: Name of the tool
        specs: List of parameter specifications

    Returns:
        Decorator function
    """
    enforcer = ParameterEnforcer(tool_name=tool_name, parameter_specs=specs)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, context: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
            # Enforce parameters
            enforced_kwargs = enforcer.enforce(kwargs, context=context)
            return func(*args, **enforced_kwargs)

        # Attach enforcer for introspection
        wrapper._parameter_enforcer = enforcer
        return wrapper

    return decorator


def create_enforcer_for_tool(
    tool_name: str,
    schema: dict[str, Any],
) -> ParameterEnforcer:
    """
    Factory function to create a ParameterEnforcer from a JSON schema.

    Args:
        tool_name: Name of the tool
        schema: JSON schema for the tool parameters

    Returns:
        Configured ParameterEnforcer
    """
    specs = []
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Type mapping from JSON schema to ParameterType
    type_mapping = {
        "string": ParameterType.STRING,
        "integer": ParameterType.INTEGER,
        "number": ParameterType.FLOAT,
        "boolean": ParameterType.BOOLEAN,
        "array": ParameterType.ARRAY,
        "object": ParameterType.OBJECT,
    }

    for name, prop in properties.items():
        json_type = prop.get("type", "string")
        param_type = type_mapping.get(json_type, ParameterType.STRING)

        spec = ParameterSpec(
            name=name,
            param_type=param_type,
            required=name in required,
            default=prop.get("default"),
            description=prop.get("description", ""),
        )
        specs.append(spec)

    return ParameterEnforcer(tool_name=tool_name, parameter_specs=specs)


# Pre-configured enforcers for common tools
TOOL_ENFORCERS: dict[str, ParameterEnforcer] = {}


def get_enforcer_for_tool(tool_name: str) -> Optional[ParameterEnforcer]:
    """Get the pre-configured enforcer for a tool, if available."""
    return TOOL_ENFORCERS.get(tool_name)


def register_tool_enforcer(
    tool_name: str,
    specs: list[ParameterSpec],
) -> None:
    """Register a parameter enforcer for a tool."""
    TOOL_ENFORCERS[tool_name] = ParameterEnforcer(
        tool_name=tool_name,
        parameter_specs=specs,
    )


# Register enforcers for tools that commonly have missing parameters
register_tool_enforcer(
    "symbol",
    [
        ParameterSpec(
            name="file_path",
            param_type=ParameterType.STRING,
            required=True,
            description="Path to the file to analyze",
            inference_strategy=InferenceStrategy.FROM_PREVIOUS_ARGS,
            inference_keys=["path", "file_path", "file"],
        ),
        ParameterSpec(
            name="symbol_name",
            param_type=ParameterType.STRING,
            required=True,
            description="Name of the symbol to find",
        ),
    ],
)

register_tool_enforcer(
    "read",
    [
        ParameterSpec(
            name="path",
            param_type=ParameterType.STRING,
            required=True,
            description="Path to the file to read",
        ),
        ParameterSpec(
            name="offset",
            param_type=ParameterType.INTEGER,
            required=False,
            default=0,
            description="Line offset to start reading from",
        ),
        ParameterSpec(
            name="limit",
            param_type=ParameterType.INTEGER,
            required=False,
            default=2000,
            description="Maximum number of lines to read",
        ),
    ],
)

register_tool_enforcer(
    "grep",
    [
        ParameterSpec(
            name="pattern",
            param_type=ParameterType.STRING,
            required=True,
            description="Pattern to search for",
        ),
        ParameterSpec(
            name="path",
            param_type=ParameterType.STRING,
            required=False,
            default=".",
            description="Path to search in",
            inference_strategy=InferenceStrategy.FROM_WORKING_DIR,
        ),
    ],
)

register_tool_enforcer(
    "ls",
    [
        ParameterSpec(
            name="path",
            param_type=ParameterType.STRING,
            required=False,
            default=".",
            description="Path to list",
            inference_strategy=InferenceStrategy.FROM_WORKING_DIR,
        ),
    ],
)

register_tool_enforcer(
    "glob",
    [
        ParameterSpec(
            name="pattern",
            param_type=ParameterType.STRING,
            required=True,
            description="Glob pattern to match",
        ),
        ParameterSpec(
            name="path",
            param_type=ParameterType.STRING,
            required=False,
            default=".",
            description="Base path for glob",
            inference_strategy=InferenceStrategy.FROM_WORKING_DIR,
        ),
    ],
)
