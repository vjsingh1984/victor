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

"""Unit tests for validation pipeline."""

from unittest.mock import Mock

import pytest

from victor.framework.validation.pipeline import (
    ChainHandler,
    ConditionalHandler,
    HaltHandler,
    RetryHandler,
    SkipHandler,
    ValidationAction,
    ValidationContext,
    ValidationHandler,
    ValidationIssue,
    ValidationPipeline,
    ValidationResult,
    ValidationStage,
    ValidationSeverity,
    create_validation_pipeline,
    is_valid,
    validate_and_get_errors,
)
from victor.framework.validation.validators import (
    ThresholdValidator,
    TypeValidator,
)


@pytest.fixture
def context() -> ValidationContext:
    """Create a validation context for tests."""
    return ValidationContext(data={})


@pytest.fixture
def simple_validators():
    """Create simple validators for testing."""
    return [
        ThresholdValidator(field="score", min_value=0, max_value=100),
        TypeValidator(field="score", expected_type=int),
    ]


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_default_is_valid(self):
        """Test that default result is valid."""
        result = ValidationResult()
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_makes_invalid(self):
        """Test that adding error makes result invalid."""
        result = ValidationResult()
        result.add_error("field", "Error message")
        assert not result.is_valid
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        """Test that adding warning keeps result valid."""
        result = ValidationResult()
        result.add_warning("field", "Warning message")
        assert result.is_valid
        assert len(result.warnings) == 1

    def test_add_info_keeps_valid(self):
        """Test that adding info keeps result valid."""
        result = ValidationResult()
        result.add_info("field", "Info message")
        assert result.is_valid
        assert len(result.info) == 1

    def test_merge_results(self):
        """Test merging two results."""
        result1 = ValidationResult()
        result1.add_error("field1", "Error 1")

        result2 = ValidationResult()
        result2.add_error("field2", "Error 2")
        result2.add_warning("field", "Warning")

        result1.merge(result2)

        assert not result1.is_valid
        assert len(result1.errors) == 2
        assert len(result1.warnings) == 1

    def test_summary(self):
        """Test result summary."""
        result = ValidationResult()
        assert result.summary() == "Validation passed"

        result.add_warning("field", "Warning")
        assert "1 warning" in result.summary()

        result.add_error("field", "Error")
        assert "failed" in result.summary()

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = ValidationResult()
        result.add_error("field", "Error", code="test_code")
        result.add_warning("field2", "Warning")

        d = result.to_dict()
        assert d["is_valid"] is False
        assert d["error_count"] == 1
        assert d["warning_count"] == 1
        assert len(d["issues"]) == 2


# =============================================================================
# ValidationContext Tests
# =============================================================================


class TestValidationContext:
    """Tests for ValidationContext."""

    def test_get_and_set(self):
        """Test getting and setting values."""
        context = ValidationContext(data={})
        context.set("key", "value")
        assert context.get("key") == "value"
        assert context.get("missing", "default") == "default"

    def test_get_nested(self):
        """Test getting nested values."""
        context = ValidationContext(data={"user": {"profile": {"name": "John"}}})
        assert context.get_nested("user.profile.name") == "John"
        assert context.get_nested("user.missing") is None

    def test_can_retry(self):
        """Test retry counter."""
        context = ValidationContext(data={}, max_retries=3)
        assert context.can_retry()
        context.increment_retry()
        assert context.can_retry()
        context.retry_count = 3
        assert not context.can_retry()

    def test_copy(self):
        """Test copying context."""
        context = ValidationContext(
            data={"key": "value"},
            retry_count=1,
        )
        copy = context.copy()
        assert copy.data == context.data
        assert copy.retry_count == context.retry_count
        copy.set("key", "new_value")
        assert context.get("key") == "value"


# =============================================================================
# ValidationPipeline Tests
# =============================================================================


class TestValidationPipeline:
    """Tests for ValidationPipeline."""

    def test_validate_with_passing_validators(self, context):
        """Test validation with all validators passing."""
        validators = [
            ThresholdValidator(field="score", min_value=0, max_value=100),
            TypeValidator(field="score", expected_type=int),
        ]
        pipeline = ValidationPipeline(validators=validators)

        data = {"score": 85}
        result = pipeline.validate(data)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_with_failing_validator(self, context):
        """Test validation with failing validator."""
        validators = [
            ThresholdValidator(field="score", min_value=0, max_value=100),
        ]
        pipeline = ValidationPipeline(validators=validators)

        data = {"score": 150}
        result = pipeline.validate(data)

        assert not result.is_valid
        assert len(result.errors) == 1

    def test_validate_with_halt_on_error(self, context):
        """Test halt_on_error behavior."""
        validators = [
            ThresholdValidator(field="score", min_value=0, max_value=100),
            TypeValidator(field="score", expected_type=str),  # Won't run
        ]
        pipeline = ValidationPipeline(
            validators=validators,
            halt_on_error=True,
            collect_all_errors=False,  # Need this for halt_on_error to work
        )

        data = {"score": 150}
        result = pipeline.validate(data)

        assert not result.is_valid
        assert len(result.errors) == 1  # Only first error

    def test_validate_collect_all_errors(self, context):
        """Test collecting all errors."""
        validators = [
            ThresholdValidator(field="score", min_value=0, max_value=100),
            TypeValidator(field="score", expected_type=str),
        ]
        pipeline = ValidationPipeline(
            validators=validators,
            halt_on_error=False,
            collect_all_errors=True,
        )

        data = {"score": 150}
        result = pipeline.validate(data)

        assert not result.is_valid
        assert len(result.errors) == 2  # Both errors

    def test_validate_batch(self, context):
        """Test batch validation."""
        validators = [
            ThresholdValidator(field="score", min_value=0, max_value=100),
        ]
        pipeline = ValidationPipeline(validators=validators)

        batch = [
            {"score": 50},
            {"score": 150},
            {"score": 85},
        ]
        results = pipeline.validate_batch(batch)

        assert len(results) == 3
        assert results[0].is_valid
        assert not results[1].is_valid
        assert results[2].is_valid

    def test_validate_stream(self, context):
        """Test stream validation."""
        validators = [
            ThresholdValidator(field="score", min_value=0, max_value=100),
        ]
        pipeline = ValidationPipeline(validators=validators)

        def data_stream():
            yield {"score": 50}
            yield {"score": 150}
            yield {"score": 85}

        results = list(pipeline.validate_stream(data_stream()))

        assert len(results) == 3
        assert results[0].is_valid
        assert not results[1].is_valid
        assert results[2].is_valid

    def test_aggregate_results(self, context):
        """Test aggregating results."""
        validators = [
            ThresholdValidator(field="score", min_value=0, max_value=100),
        ]
        pipeline = ValidationPipeline(validators=validators)

        batch = [
            {"score": 50},
            {"score": 150},
        ]
        results = pipeline.validate_batch(batch)

        aggregated = pipeline.aggregate_results(results)

        assert not aggregated.is_valid
        assert aggregated.data["total_results"] == 2
        assert aggregated.data["valid_count"] == 1
        assert aggregated.data["invalid_count"] == 1

    def test_add_validator(self, context):
        """Test adding validators dynamically."""
        pipeline = ValidationPipeline()
        assert len(pipeline._validators) == 0

        validator = ThresholdValidator(field="score", min_value=0, max_value=100)
        pipeline.add_validator(validator)

        assert len(pipeline._validators) == 1

    def test_remove_validator(self, context):
        """Test removing validators."""
        validator = ThresholdValidator(field="score", min_value=0, max_value=100)
        pipeline = ValidationPipeline(validators=[validator])

        assert len(pipeline._validators) == 1
        pipeline.remove_validator(validator)
        assert len(pipeline._validators) == 0

    def test_set_handler(self, context):
        """Test setting handler."""
        pipeline = ValidationPipeline()
        new_handler = HaltHandler()
        pipeline.set_handler(new_handler)
        assert pipeline._handler is new_handler


# =============================================================================
# Handler Tests
# =============================================================================


class TestHaltHandler:
    """Tests for HaltHandler."""

    def test_can_handle_always(self):
        """Test that HaltHandler always handles."""
        handler = HaltHandler()
        result = ValidationResult()
        assert handler.can_handle(result)

    def test_handle_returns_halt(self):
        """Test that HaltHandler returns HALT action."""
        handler = HaltHandler()
        result = ValidationResult()
        context = ValidationContext(data={})
        action = handler.handle(result, context)
        assert action == ValidationAction.HALT


class TestSkipHandler:
    """Tests for SkipHandler."""

    def test_can_handle_always(self):
        """Test that SkipHandler always handles."""
        handler = SkipHandler()
        result = ValidationResult()
        assert handler.can_handle(result)

    def test_handle_returns_skip(self):
        """Test that SkipHandler returns SKIP action."""
        handler = SkipHandler()
        result = ValidationResult()
        context = ValidationContext(data={})
        action = handler.handle(result, context)
        assert action == ValidationAction.SKIP


class TestRetryHandler:
    """Tests for RetryHandler."""

    def test_can_handle_on_failure(self):
        """Test that RetryHandler handles failures."""
        handler = RetryHandler(max_retries=3)
        result = ValidationResult()
        result.add_error("field", "Error")
        assert handler.can_handle(result)

    def test_can_handle_with_warnings(self):
        """Test that RetryHandler with retry_on_warnings handles warnings."""
        handler = RetryHandler(retry_on_warnings=True)
        result = ValidationResult()
        result.add_warning("field", "Warning")
        assert handler.can_handle(result)

    def test_retry_when_available(self):
        """Test retrying when retries available."""
        handler = RetryHandler(max_retries=3, backoff_factor=0)
        result = ValidationResult()
        result.add_error("field", "Error")
        context = ValidationContext(data={}, max_retries=3)
        action = handler.handle(result, context)
        assert action == ValidationAction.RETRY
        assert context.retry_count == 1

    def test_halt_when_no_retries(self):
        """Test halting when no retries available."""
        handler = RetryHandler(max_retries=0)
        result = ValidationResult()
        result.add_error("field", "Error")
        context = ValidationContext(data={}, max_retries=0)
        action = handler.handle(result, context)
        assert action == ValidationAction.HALT


class TestConditionalHandler:
    """Tests for ConditionalHandler."""

    def test_condition_met(self):
        """Test when condition is met."""
        handler = ConditionalHandler(
            condition=lambda r: not r.is_valid,
            action=ValidationAction.RETRY,
        )
        result = ValidationResult()
        result.add_error("field", "Error")
        context = ValidationContext(data={})
        action = handler.handle(result, context)
        assert action == ValidationAction.RETRY

    def test_condition_not_met(self):
        """Test when condition is not met."""
        handler = ConditionalHandler(
            condition=lambda r: not r.is_valid,
            action=ValidationAction.RETRY,
        )
        result = ValidationResult()  # Valid result
        context = ValidationContext(data={})
        action = handler.handle(result, context)
        assert action == ValidationAction.HALT  # Fallback action


class TestChainHandler:
    """Tests for ChainHandler."""

    def test_chain_tries_each_handler(self):
        """Test that ChainHandler tries each handler in sequence."""
        handler1 = Mock(spec=ValidationHandler)
        handler1.can_handle.return_value = False
        handler2 = Mock(spec=ValidationHandler)
        handler2.can_handle.return_value = True
        handler2.handle.return_value = ValidationAction.SKIP

        chain = ChainHandler([handler1, handler2])
        result = ValidationResult()
        context = ValidationContext(data={})

        action = chain.handle(result, context)

        handler1.can_handle.assert_called_once()
        handler2.can_handle.assert_called_once()
        handler2.handle.assert_called_once()
        assert action == ValidationAction.SKIP

    def test_chain_stops_on_first_success(self):
        """Test that chain stops on first successful handler."""
        handler1 = Mock(spec=ValidationHandler)
        handler1.can_handle.return_value = True
        handler1.handle.return_value = ValidationAction.SKIP
        handler2 = Mock(spec=ValidationHandler)

        chain = ChainHandler([handler1, handler2])
        result = ValidationResult()
        context = ValidationContext(data={})

        action = chain.handle(result, context)

        handler1.can_handle.assert_called_once()
        handler2.can_handle.assert_not_called()
        assert action == ValidationAction.SKIP


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_validation_pipeline(self):
        """Test creating a pipeline with convenience function."""
        validators = [
            ThresholdValidator(field="score", min_value=0, max_value=100),
        ]
        pipeline = create_validation_pipeline(
            validators=validators,
            halt_on_error=True,
        )
        assert isinstance(pipeline, ValidationPipeline)
        assert len(pipeline._validators) == 1
        assert pipeline._halt_on_error is True

    def test_validate_and_get_errors(self):
        """Test getting error messages."""
        validators = [
            ThresholdValidator(field="score", min_value=0, max_value=100),
        ]
        errors = validate_and_get_errors({"score": 150}, validators)
        assert len(errors) == 1
        assert "exceeds maximum" in errors[0]

    def test_is_valid(self):
        """Test is_valid convenience function."""
        validators = [
            ThresholdValidator(field="score", min_value=0, max_value=100),
        ]
        assert is_valid({"score": 50}, validators)
        assert not is_valid({"score": 150}, validators)


# =============================================================================
# Integration Tests
# =============================================================================


class TestValidationPipelineIntegration:
    """Integration tests for the validation pipeline."""

    def test_full_validation_flow(self):
        """Test the complete validation flow with retries."""
        validators = [
            ThresholdValidator(field="score", min_value=0, max_value=100),
        ]
        handler = RetryHandler(max_retries=2, backoff_factor=0)
        pipeline = ValidationPipeline(
            validators=validators,
            handler=handler,
        )

        # First attempt fails, retries succeed after data modification
        data = {"score": 150}
        result = pipeline.validate(data)

        assert not result.is_valid  # Data never changes, so always fails
        assert result.retry_count <= 2

    def test_complex_validation_scenario(self):
        """Test a complex validation scenario."""
        validators = [
            PresenceValidator(field="email"),
            PatternValidator(field="email", pattern_type="email"),
            PresenceValidator(field="age"),
            TypeValidator(field="age", expected_type=int),
            ThresholdValidator(field="age", min_value=18, max_value=120),
        ]
        pipeline = ValidationPipeline(validators=validators)

        # Valid data
        data = {
            "email": "user@example.com",
            "age": 25,
        }
        result = pipeline.validate(data)
        assert result.is_valid

        # Invalid email
        data = {
            "email": "not-an-email",
            "age": 25,
        }
        result = pipeline.validate(data)
        assert not result.is_valid

        # Underage
        data = {
            "email": "user@example.com",
            "age": 15,
        }
        result = pipeline.validate(data)
        assert not result.is_valid


# Import validators used in tests
from victor.framework.validation.validators import (
    PresenceValidator,
    PatternValidator,
)
