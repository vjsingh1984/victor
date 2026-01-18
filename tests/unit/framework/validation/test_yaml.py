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

"""Unit tests for YAML integration in validation framework."""

import pytest

from victor.framework.validation.yaml import (
    HandlerConfig,
    HandlerFactory,
    ValidationNodeConfig,
    ValidationPipelineBuilder,
    ValidatorConfig,
    ValidatorFactory,
    create_pipeline_from_yaml,
    validate_from_yaml,
)
from victor.framework.validation.pipeline import ValidationAction, ValidationPipeline


# =============================================================================
# ValidatorConfig Tests
# =============================================================================


class TestValidatorConfig:
    """Tests for ValidatorConfig."""

    def test_from_dict_minimal(self):
        """Test creating config from minimal dict."""
        data = {"type": "threshold", "field": "score"}
        config = ValidatorConfig.from_dict(data)
        assert config.type == "threshold"
        assert config.field_name == "score"
        # When no explicit 'config' key, the whole data dict is used as config
        assert config.config == data

    def test_from_dict_with_config(self):
        """Test creating config with config field."""
        data = {
            "type": "threshold",
            "field": "score",
            "config": {"min": 0, "max": 100},
        }
        config = ValidatorConfig.from_dict(data)
        assert config.config == {"min": 0, "max": 100}

    def test_from_dict_flat_config(self):
        """Test creating config with flat config."""
        data = {
            "type": "threshold",
            "field": "score",
            "min": 0,
            "max": 100,
        }
        config = ValidatorConfig.from_dict(data)
        assert config.config == {"type": "threshold", "field": "score", "min": 0, "max": 100}


# =============================================================================
# HandlerConfig Tests
# =============================================================================


class TestHandlerConfig:
    """Tests for HandlerConfig."""

    def test_from_dict_with_type(self):
        """Test creating config with type."""
        data = {"type": "halt"}
        config = HandlerConfig.from_dict(data)
        assert config.type == "halt"

    def test_from_dict_string(self):
        """Test creating config from string."""
        config = HandlerConfig.from_dict("skip")
        assert config.type == "skip"

    def test_from_dict_with_config(self):
        """Test creating config with additional config."""
        data = {"type": "retry", "max_retries": 5}
        config = HandlerConfig.from_dict(data)
        assert config.type == "retry"
        assert config.config == {"max_retries": 5}


# =============================================================================
# ValidationNodeConfig Tests
# =============================================================================


class TestValidationNodeConfig:
    """Tests for ValidationNodeConfig."""

    def test_from_dict_minimal(self):
        """Test creating node config from minimal dict."""
        data = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ]
        }
        config = ValidationNodeConfig.from_dict(data)
        assert len(config.validators) == 1
        assert config.validators[0].type == "threshold"
        assert config.halt_on_error is True
        assert config.max_retries == 3

    def test_from_dict_with_all_options(self):
        """Test creating node config with all options."""
        data = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ],
            "handler": {"type": "skip"},
            "halt_on_error": False,
            "collect_all_errors": False,
            "max_retries": 5,
            "timeout_seconds": 30.0,
            "enable_logging": False,
        }
        config = ValidationNodeConfig.from_dict(data)
        assert config.halt_on_error is False
        assert config.collect_all_errors is False
        assert config.max_retries == 5
        assert config.timeout_seconds == 30.0
        assert config.enable_logging is False


# =============================================================================
# ValidatorFactory Tests
# =============================================================================


class TestValidatorFactory:
    """Tests for ValidatorFactory."""

    @pytest.fixture
    def factory(self):
        """Create a validator factory."""
        return ValidatorFactory()

    def test_create_threshold_validator(self, factory):
        """Test creating a threshold validator."""
        config = ValidatorConfig(
            type="threshold",
            field_name="score",
            config={"min": 0, "max": 100},
        )
        validator = factory.create_validator(config)
        assert validator.__class__.__name__ == "ThresholdValidator"
        assert validator._field == "score"

    def test_create_range_validator(self, factory):
        """Test creating a range validator."""
        config = ValidatorConfig(
            type="range",
            field_name="age",
            config={"min": 18, "max": 65},
        )
        validator = factory.create_validator(config)
        assert validator.__class__.__name__ == "RangeValidator"

    def test_create_presence_validator(self, factory):
        """Test creating a presence validator."""
        config = ValidatorConfig(
            type="presence",
            field_name="email",
            config={"required": True, "allow_empty": False},
        )
        validator = factory.create_validator(config)
        assert validator.__class__.__name__ == "PresenceValidator"

    def test_create_pattern_validator(self, factory):
        """Test creating a pattern validator."""
        config = ValidatorConfig(
            type="pattern",
            field_name="email",
            config={"pattern_type": "email"},
        )
        validator = factory.create_validator(config)
        assert validator.__class__.__name__ == "PatternValidator"

    def test_create_type_validator(self, factory):
        """Test creating a type validator."""
        config = ValidatorConfig(
            type="type",
            field_name="count",
            config={"expected_type": "int"},
        )
        validator = factory.create_validator(config)
        assert validator.__class__.__name__ == "TypeValidator"

    def test_create_length_validator(self, factory):
        """Test creating a length validator."""
        config = ValidatorConfig(
            type="length",
            field_name="name",
            config={"min_length": 3, "max_length": 20},
        )
        validator = factory.create_validator(config)
        assert validator.__class__.__name__ == "LengthValidator"

    def test_create_composite_validator(self, factory):
        """Test creating a composite validator."""
        config = ValidatorConfig(
            type="composite",
            field_name="value",
            config={
                "logic": "all",
                "validators": [
                    {"type": "type", "field": "value", "expected_type": "int"},
                    {"type": "threshold", "field": "value", "min": 0, "max": 100},
                ],
            },
        )
        validator = factory.create_validator(config)
        assert validator.__class__.__name__ == "CompositeValidator"
        assert len(validator._validators) == 2

    def test_unknown_validator_type(self, factory):
        """Test creating a validator with unknown type."""
        config = ValidatorConfig(
            type="unknown_type",
            field_name="value",
            config={},
        )
        with pytest.raises(ValueError, match="Unknown validator type"):
            factory.create_validator(config)

    def test_register_custom_validator(self, factory):
        """Test registering a custom validator."""

        class CustomValidator:
            def __init__(self, field: str):
                self.field = field

            @classmethod
            def from_config(cls, config):
                # Map field_name to field parameter
                return cls(field=config.field_name)

            @property
            def name(self) -> str:
                return "CustomValidator"

            def validate(self, data: dict, context) -> "ValidationResult":
                from victor.framework.validation.pipeline import ValidationResult

                return ValidationResult(is_valid=True)

        factory.register_validator("custom", CustomValidator)

        config = ValidatorConfig(
            type="custom",
            field_name="value",
            config={},
        )
        validator = factory.create_validator(config)
        assert isinstance(validator, CustomValidator)


# =============================================================================
# HandlerFactory Tests
# =============================================================================


class TestHandlerFactory:
    """Tests for HandlerFactory."""

    @pytest.fixture
    def factory(self):
        """Create a handler factory."""
        return HandlerFactory()

    def test_create_halt_handler(self, factory):
        """Test creating a halt handler."""
        config = HandlerConfig(type="halt")
        handler = factory.create_handler(config)
        assert handler.__class__.__name__ == "HaltHandler"

    def test_create_skip_handler(self, factory):
        """Test creating a skip handler."""
        config = HandlerConfig(type="skip")
        handler = factory.create_handler(config)
        assert handler.__class__.__name__ == "SkipHandler"

    def test_create_retry_handler(self, factory):
        """Test creating a retry handler."""
        config = HandlerConfig(
            type="retry",
            config={"max_retries": 5, "backoff_factor": 2.0},
        )
        handler = factory.create_handler(config)
        assert handler.__class__.__name__ == "RetryHandler"

    def test_create_conditional_handler(self, factory):
        """Test creating a conditional handler."""
        config = HandlerConfig(
            type="conditional",
            config={"action": "retry", "min_severity": "error"},
        )
        handler = factory.create_handler(config)
        assert handler.__class__.__name__ == "ConditionalHandler"

    def test_create_chain_handler(self, factory):
        """Test creating a chain handler."""
        config = HandlerConfig(
            type="chain",
            config={
                "handlers": [
                    {"type": "retry"},
                    {"type": "halt"},
                ]
            },
        )
        handler = factory.create_handler(config)
        assert handler.__class__.__name__ == "ChainHandler"

    def test_none_returns_default(self, factory):
        """Test that None config returns default halt handler."""
        handler = factory.create_handler(None)
        assert handler.__class__.__name__ == "HaltHandler"

    def test_unknown_type_falls_back(self, factory):
        """Test that unknown type falls back to halt handler."""
        config = HandlerConfig(type="unknown")
        handler = factory.create_handler(config)
        assert handler.__class__.__name__ == "HaltHandler"


# =============================================================================
# ValidationPipelineBuilder Tests
# =============================================================================


class TestValidationPipelineBuilder:
    """Tests for ValidationPipelineBuilder."""

    @pytest.fixture
    def builder(self):
        """Create a pipeline builder."""
        return ValidationPipelineBuilder()

    def test_from_dict_simple(self, builder):
        """Test building a pipeline from simple config."""
        config = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ],
        }
        pipeline = builder.from_dict(config)
        assert isinstance(pipeline, ValidationPipeline)
        assert len(pipeline._validators) == 1

    def test_from_dict_with_handler(self, builder):
        """Test building a pipeline with handler."""
        config = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ],
            "handler": {"type": "skip"},
        }
        pipeline = builder.from_dict(config)
        assert isinstance(pipeline, ValidationPipeline)
        assert pipeline._handler.__class__.__name__ == "SkipHandler"

    def test_from_dict_with_all_options(self, builder):
        """Test building a pipeline with all options."""
        config = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ],
            "handler": {"type": "halt"},
            "halt_on_error": False,
            "collect_all_errors": False,
            "max_retries": 5,
            "timeout_seconds": 30.0,
            "enable_logging": False,
        }
        pipeline = builder.from_dict(config)
        assert pipeline._halt_on_error is False
        assert pipeline._collect_all_errors is False
        assert pipeline._max_retries == 5
        assert pipeline._timeout_seconds == 30.0
        assert pipeline._enable_logging is False

    def test_from_yaml_node(self, builder):
        """Test building a pipeline from workflow YAML node."""
        node_data = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ],
        }
        pipeline = builder.from_yaml_node(node_data)
        assert isinstance(pipeline, ValidationPipeline)

    def test_register_custom_validator(self, builder):
        """Test registering a custom validator."""

        class CustomValidator:
            def __init__(self, field: str):
                self.field = field

            @classmethod
            def from_config(cls, config):
                # Map field_name to field parameter
                return cls(field=config.field_name)

            @property
            def name(self) -> str:
                return "CustomValidator"

            def validate(self, data: dict, context):
                from victor.framework.validation.pipeline import ValidationResult

                return ValidationResult(is_valid=True)

        builder.register_validator("custom", CustomValidator)

        config = {
            "validators": [
                {"type": "custom", "field": "value"},
            ],
        }
        pipeline = builder.from_dict(config)
        assert len(pipeline._validators) == 1


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_pipeline_from_yaml(self):
        """Test creating a pipeline from YAML config."""
        yaml_config = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ],
        }
        pipeline = create_pipeline_from_yaml(yaml_config)
        assert isinstance(pipeline, ValidationPipeline)

    def test_validate_from_yaml_valid(self):
        """Test validating valid data from YAML config."""
        yaml_config = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ],
        }
        data = {"score": 85}
        result = validate_from_yaml(data, yaml_config)
        assert result.is_valid

    def test_validate_from_yaml_invalid(self):
        """Test validating invalid data from YAML config."""
        yaml_config = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ],
        }
        data = {"score": 150}
        result = validate_from_yaml(data, yaml_config)
        assert not result.is_valid


# =============================================================================
# Integration Tests
# =============================================================================


class TestYAMLIntegration:
    """Integration tests for YAML integration."""

    def test_complex_workflow_yaml(self):
        """Test a complex workflow YAML configuration."""
        yaml_config = {
            "validators": [
                {
                    "type": "composite",
                    "field": "user",
                    "logic": "all",
                    "validators": [
                        {"type": "presence", "field": "user.email"},
                        {
                            "type": "pattern",
                            "field": "user.email",
                            "pattern_type": "email",
                        },
                        {"type": "presence", "field": "user.age"},
                        {"type": "type", "field": "user.age", "expected_type": "int"},
                        {
                            "type": "threshold",
                            "field": "user.age",
                            "min": 18,
                            "max": 120,
                        },
                    ],
                },
            ],
            "handler": {"type": "halt"},
            "halt_on_error": True,
        }

        pipeline = create_pipeline_from_yaml(yaml_config)

        # Valid data
        data = {"user": {"email": "test@example.com", "age": 25}}
        result = pipeline.validate(data)
        assert result.is_valid

        # Invalid email
        data = {"user": {"email": "not-an-email", "age": 25}}
        result = pipeline.validate(data)
        assert not result.is_valid

    def test_validation_node_handler(self):
        """Test the validate_pipeline_handler function."""
        from victor.framework.validation.yaml import validate_pipeline_handler

        node_config = {
            "validators": [
                {"type": "threshold", "field": "score", "min": 0, "max": 100},
            ],
        }
        graph_state = {"score": 85}

        new_state = validate_pipeline_handler(node_config, graph_state)

        assert "_validation_result" in new_state
        assert new_state["_validation_result"]["is_valid"] is True

        # Test with invalid data
        graph_state = {"score": 150}
        new_state = validate_pipeline_handler(node_config, graph_state)

        assert "_validation_result" in new_state
        assert new_state["_validation_result"]["is_valid"] is False
        assert "_validation_errors" in new_state
