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

"""Validation Framework for workflows and agent operations.

This framework provides protocol-based validation interfaces and common validators
that eliminate duplicate validation patterns across verticals.

Phase 3 Task 3: Create Validation Framework

Main Components:
- ValidationPipeline: Generic validate -> check -> handle -> retry flow
- Validators: ThresholdValidator, RangeValidator, PresenceValidator, PatternValidator
- CompositeValidator: Combine multiple validators with configurable logic
- YAML integration: Validation in workflow YAML definitions

Example:
    from victor.framework.validation import (
        ValidationPipeline,
        ThresholdValidator,
        RangeValidator,
        CompositeValidator,
    )

    # Create a validation pipeline
    pipeline = ValidationPipeline(
        validators=[
            ThresholdValidator(min_value=0, max_value=100),
            RangeValidator(min_value=10, max_value=90),
        ],
        halt_on_error=True,
    )

    # Validate data
    result = pipeline.validate({"score": 85})
    if result.is_valid:
        print("Validation passed!")
    else:
        for error in result.errors:
            print(f"Error: {error}")

    # Use in workflow YAML
    # workflows:
    #   my_workflow:
    #     nodes:
    #       - id: validate_data
    #         type: validate_pipeline
    #         validators:
    #           - type: threshold
    #             field: score
    #             min: 0
    #             max: 100
    #           - type: range
    #             field: score
    #             min: 10
    #             max: 90
"""

from victor.framework.validation.pipeline import (
    ValidationAction,
    ValidationConfig,
    ValidationContext,
    ValidationHandler,
    ValidationPipeline,
    ValidationResult,
    ValidationStage,
    create_validation_pipeline,
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

# YAML integration (lazy import to avoid circular dependencies)
try:
    from victor.framework.validation.yaml import (
        HandlerConfig,
        HandlerFactory,
        ValidationNodeConfig,
        ValidationPipelineBuilder,
        ValidatorConfig,
        ValidatorFactory,
        create_pipeline_from_yaml,
        validate_from_yaml,
        validate_pipeline_handler,
    )

    _YAML_EXPORTS = [
        "ValidatorConfig",
        "HandlerConfig",
        "ValidationNodeConfig",
        "ValidatorFactory",
        "HandlerFactory",
        "ValidationPipelineBuilder",
        "create_pipeline_from_yaml",
        "validate_from_yaml",
        "validate_pipeline_handler",
    ]
except ImportError:
    _YAML_EXPORTS = []

__all__ = [
    # Pipeline
    "ValidationPipeline",
    "ValidationConfig",
    "ValidationContext",
    "ValidationResult",
    "ValidationStage",
    "ValidationAction",
    "ValidationHandler",
    "create_validation_pipeline",
    # Validators
    "ValidatorProtocol",
    "ThresholdValidator",
    "RangeValidator",
    "PresenceValidator",
    "PatternValidator",
    "TypeValidator",
    "LengthValidator",
    "CompositeValidator",
    "CompositeLogic",
    "ConditionalValidator",
    "TransformingValidator",
] + _YAML_EXPORTS
