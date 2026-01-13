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

"""Unit tests for validation framework validators."""

import pytest

from victor.framework.validation.pipeline import ValidationContext, ValidationResult
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
)


@pytest.fixture
def context() -> ValidationContext:
    """Create a validation context for tests."""
    return ValidationContext(data={})


# =============================================================================
# Threshold Validator Tests
# =============================================================================


class TestThresholdValidator:
    """Tests for ThresholdValidator."""

    def test_validate_within_range(self, context):
        """Test validation with value within range."""
        validator = ThresholdValidator(field="score", min_value=0, max_value=100)
        data = {"score": 85}
        result = validator.validate(data, context)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_below_minimum(self, context):
        """Test validation with value below minimum."""
        validator = ThresholdValidator(field="score", min_value=0, max_value=100)
        data = {"score": -5}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "below minimum" in result.errors[0].message

    def test_validate_above_maximum(self, context):
        """Test validation with value above maximum."""
        validator = ThresholdValidator(field="score", min_value=0, max_value=100)
        data = {"score": 150}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "exceeds maximum" in result.errors[0].message

    def test_validate_exclusive_bounds(self, context):
        """Test validation with exclusive bounds."""
        validator = ThresholdValidator(
            field="value",
            min_value=0,
            max_value=100,
            min_inclusive=False,
            max_inclusive=False,
        )
        data = {"value": 0}
        result = validator.validate(data, context)
        assert not result.is_valid

        data = {"value": 100}
        result = validator.validate(data, context)
        assert not result.is_valid

    def test_validate_missing_field(self, context):
        """Test validation with missing field."""
        validator = ThresholdValidator(field="score", min_value=0)
        data = {}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "missing" in result.errors[0].message.lower()

    def test_validate_non_numeric(self, context):
        """Test validation with non-numeric value."""
        validator = ThresholdValidator(field="score", min_value=0, max_value=100)
        data = {"score": "not a number"}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "numeric" in result.errors[0].message.lower()


# =============================================================================
# Range Validator Tests
# =============================================================================


class TestRangeValidator:
    """Tests for RangeValidator."""

    def test_validate_in_range(self, context):
        """Test validation with value in range."""
        validator = RangeValidator(field="age", min_value=18, max_value=65)
        data = {"age": 25}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_validate_below_range(self, context):
        """Test validation with value below range."""
        validator = RangeValidator(field="age", min_value=18, max_value=65)
        data = {"age": 15}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "at least" in result.errors[0].message.lower()

    def test_validate_above_range(self, context):
        """Test validation with value above range."""
        validator = RangeValidator(field="age", min_value=18, max_value=65)
        data = {"age": 70}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "at most" in result.errors[0].message.lower()

    def test_validate_exclusive_min(self, context):
        """Test validation with exclusive minimum."""
        validator = RangeValidator(
            field="value",
            min_value=10,
            max_value=20,
            exclusive_min=True,
        )
        data = {"value": 10}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "greater than" in result.errors[0].message.lower()

    def test_validate_none_value(self, context):
        """Test validation with None value."""
        validator = RangeValidator(field="age", min_value=18, max_value=65)
        data = {"age": None}
        result = validator.validate(data, context)
        assert result.is_valid  # None should be skipped


# =============================================================================
# Presence Validator Tests
# =============================================================================


class TestPresenceValidator:
    """Tests for PresenceValidator."""

    def test_validate_present(self, context):
        """Test validation with present field."""
        validator = PresenceValidator(field="email", required=True)
        data = {"email": "test@example.com"}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_validate_missing_required(self, context):
        """Test validation with missing required field."""
        validator = PresenceValidator(field="email", required=True)
        data = {}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "missing" in result.errors[0].message.lower()

    def test_validate_missing_optional(self, context):
        """Test validation with missing optional field."""
        validator = PresenceValidator(field="email", required=False)
        data = {}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_validate_empty_string(self, context):
        """Test validation with empty string."""
        validator = PresenceValidator(field="email", allow_empty=False)
        data = {"email": ""}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "empty" in result.errors[0].message.lower()

    def test_validate_empty_collection(self, context):
        """Test validation with empty collection."""
        validator = PresenceValidator(field="items", allow_empty=False)
        data = {"items": []}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "empty" in result.errors[0].message.lower()

    def test_validate_truthiness(self, context):
        """Test validation with truthiness check."""
        validator = PresenceValidator(
            field="value",
            check_truthiness=True,
        )
        data = {"value": 0}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "truthy" in result.errors[0].message.lower()


# =============================================================================
# Pattern Validator Tests
# =============================================================================


class TestPatternValidator:
    """Tests for PatternValidator."""

    def test_validate_matching_pattern(self, context):
        """Test validation with matching pattern."""
        validator = PatternValidator(
            field="email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )
        data = {"email": "test@example.com"}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_validate_non_matching_pattern(self, context):
        """Test validation with non-matching pattern."""
        validator = PatternValidator(
            field="email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )
        data = {"email": "not-an-email"}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "does not match" in result.errors[0].message.lower()

    def test_validate_email_pattern_type(self, context):
        """Test validation with built-in email pattern."""
        validator = PatternValidator(
            field="email",
            pattern_type="email",
        )
        data = {"email": "test@example.com"}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_validate_invalid_email(self, context):
        """Test validation with invalid email."""
        validator = PatternValidator(
            field="email",
            pattern_type="email",
        )
        data = {"email": "invalid-email"}
        result = validator.validate(data, context)
        assert not result.is_valid

    def test_validate_none_value(self, context):
        """Test validation with None value."""
        validator = PatternValidator(
            field="email",
            pattern_type="email",
        )
        data = {"email": None}
        result = validator.validate(data, context)
        assert result.is_valid  # None should be skipped

    def test_validate_non_string(self, context):
        """Test validation with non-string value."""
        validator = PatternValidator(
            field="email",
            pattern_type="email",
        )
        data = {"email": 123}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "string" in result.errors[0].message.lower()


# =============================================================================
# Type Validator Tests
# =============================================================================


class TestTypeValidator:
    """Tests for TypeValidator."""

    def test_validate_correct_type(self, context):
        """Test validation with correct type."""
        validator = TypeValidator(field="count", expected_type=int)
        data = {"count": 42}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_validate_incorrect_type(self, context):
        """Test validation with incorrect type."""
        validator = TypeValidator(field="count", expected_type=int)
        data = {"count": "42"}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "must be of type" in result.errors[0].message.lower()

    def test_validate_union_type(self, context):
        """Test validation with union type."""
        validator = TypeValidator(field="value", expected_type=(int, float))
        data = {"value": 3.14}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_validate_with_coercion(self, context):
        """Test validation with type coercion."""
        validator = TypeValidator(
            field="count",
            expected_type=int,
            coerce=True,
        )
        data = {"count": "42"}
        result = validator.validate(data, context)
        assert result.is_valid
        # Check that data was coerced
        assert isinstance(data["count"], int)

    def test_validate_type_from_string(self, context):
        """Test validation with type name string."""
        validator = TypeValidator(field="count", expected_type="int")
        data = {"count": 42}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_validate_none_value(self, context):
        """Test validation with None value."""
        validator = TypeValidator(field="count", expected_type=int)
        data = {"count": None}
        result = validator.validate(data, context)
        assert result.is_valid  # None should be skipped


# =============================================================================
# Length Validator Tests
# =============================================================================


class TestLengthValidator:
    """Tests for LengthValidator."""

    def test_validate_valid_length(self, context):
        """Test validation with valid length."""
        validator = LengthValidator(field="name", min_length=3, max_length=20)
        data = {"name": "John Doe"}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_validate_too_short(self, context):
        """Test validation with value too short."""
        validator = LengthValidator(field="name", min_length=3, max_length=20)
        data = {"name": "Jo"}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "below minimum" in result.errors[0].message.lower()

    def test_validate_too_long(self, context):
        """Test validation with value too long."""
        validator = LengthValidator(field="name", min_length=3, max_length=20)
        data = {"name": "A" * 25}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "exceeds maximum" in result.errors[0].message.lower()

    def test_validate_exact_length(self, context):
        """Test validation with exact length requirement."""
        validator = LengthValidator(field="code", exact_length=6)
        data = {"code": "12345"}
        result = validator.validate(data, context)
        assert not result.is_valid

        data = {"code": "123456"}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_validate_collection_length(self, context):
        """Test validation with collection."""
        validator = LengthValidator(field="items", min_length=1, max_length=10)
        data = {"items": [1, 2, 3]}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_validate_no_length(self, context):
        """Test validation with value that has no length."""
        validator = LengthValidator(field="value", min_length=1)
        data = {"value": 42}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "measurable length" in result.errors[0].message.lower()


# =============================================================================
# Composite Validator Tests
# =============================================================================


class TestCompositeValidator:
    """Tests for CompositeValidator."""

    def test_all_logic_all_pass(self, context):
        """Test ALL logic with all validators passing."""
        validator = CompositeValidator(
            validators=[
                ThresholdValidator(field="score", min_value=0, max_value=100),
                TypeValidator(field="score", expected_type=int),
            ],
            logic=CompositeLogic.ALL,
        )
        data = {"score": 85}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_all_logic_one_fails(self, context):
        """Test ALL logic with one validator failing."""
        validator = CompositeValidator(
            validators=[
                ThresholdValidator(field="score", min_value=0, max_value=100),
                TypeValidator(field="score", expected_type=str),
            ],
            logic=CompositeLogic.ALL,
        )
        data = {"score": 85}
        result = validator.validate(data, context)
        assert not result.is_valid

    def test_any_logic_one_passes(self, context):
        """Test ANY logic with one validator passing."""
        validator = CompositeValidator(
            validators=[
                ThresholdValidator(field="score", min_value=0, max_value=10),
                TypeValidator(field="score", expected_type=int),
            ],
            logic=CompositeLogic.ANY,
        )
        data = {"score": 85}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_any_logic_all_fail(self, context):
        """Test ANY logic with all validators failing."""
        validator = CompositeValidator(
            validators=[
                ThresholdValidator(field="score", min_value=0, max_value=10),
                TypeValidator(field="score", expected_type=str),
            ],
            logic=CompositeLogic.ANY,
        )
        data = {"score": 85}
        result = validator.validate(data, context)
        assert not result.is_valid

    def test_one_logic_exactly_one(self, context):
        """Test ONE logic with exactly one passing."""
        validator = CompositeValidator(
            validators=[
                ThresholdValidator(field="score", min_value=0, max_value=10),
                TypeValidator(field="score", expected_type=int),
            ],
            logic=CompositeLogic.ONE,
        )
        data = {"score": 85}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_one_logic_multiple_pass(self, context):
        """Test ONE logic with multiple passing."""
        validator = CompositeValidator(
            validators=[
                ThresholdValidator(field="score", min_value=0, max_value=100),
                TypeValidator(field="score", expected_type=int),
            ],
            logic=CompositeLogic.ONE,
        )
        data = {"score": 85}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "Multiple validators passed" in result.errors[0].message

    def test_none_logic_all_fail(self, context):
        """Test NONE logic with all validators failing."""
        validator = CompositeValidator(
            validators=[
                ThresholdValidator(field="score", min_value=0, max_value=10),
                TypeValidator(field="score", expected_type=str),
            ],
            logic=CompositeLogic.NONE,
        )
        data = {"score": 85}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_none_logic_one_passes(self, context):
        """Test NONE logic with one validator passing."""
        validator = CompositeValidator(
            validators=[
                TypeValidator(field="score", expected_type=int),
            ],
            logic=CompositeLogic.NONE,
        )
        data = {"score": 85}
        result = validator.validate(data, context)
        assert not result.is_valid


# =============================================================================
# Conditional Validator Tests
# =============================================================================


class TestConditionalValidator:
    """Tests for ConditionalValidator."""

    def test_condition_true_validates(self, context):
        """Test that validator runs when condition is true."""
        inner = ThresholdValidator(field="score", min_value=0, max_value=100)
        validator = ConditionalValidator(
            validator=inner,
            condition=lambda data: data.get("validate", False),
        )
        data = {"score": 85, "validate": True}
        result = validator.validate(data, context)
        assert result.is_valid

    def test_condition_false_skips(self, context):
        """Test that validator skips when condition is false."""
        inner = ThresholdValidator(field="score", min_value=0, max_value=10)
        validator = ConditionalValidator(
            validator=inner,
            condition=lambda data: data.get("validate", False),
        )
        data = {"score": 85, "validate": False}
        result = validator.validate(data, context)
        assert result.is_valid  # Skipped, so passes

    def test_condition_true_fails_validation(self, context):
        """Test that validation failure propagates."""
        inner = ThresholdValidator(field="score", min_value=0, max_value=10)
        validator = ConditionalValidator(
            validator=inner,
            condition=lambda data: data.get("validate", False),
        )
        data = {"score": 85, "validate": True}
        result = validator.validate(data, context)
        assert not result.is_valid


# =============================================================================
# Transforming Validator Tests
# =============================================================================


class TestTransformingValidator:
    """Tests for TransformingValidator."""

    def test_transform_before_validation(self, context):
        """Test that transform is applied before validation."""
        inner = ThresholdValidator(field="value", min_value=0, max_value=100)
        validator = TransformingValidator(
            field="value",
            validator=inner,
            transform=str.strip,
        )
        data = {"value": "  85  "}
        result = validator.validate(data, context)
        assert not result.is_valid  # String 85 won't pass numeric validation

    def test_transform_coerces_value(self, context):
        """Test that transform can coerce value."""
        inner = TypeValidator(field="value", expected_type=int, coerce=True)
        validator = TransformingValidator(
            field="value",
            validator=inner,
            transform=str.strip,
        )
        data = {"value": "  42  "}
        result = validator.validate(data, context)
        assert result.is_valid
        assert data["value"] == 42  # Value was coerced

    def test_transform_fails(self, context):
        """Test that transform failure is handled."""
        inner = ThresholdValidator(field="value", min_value=0, max_value=100)
        validator = TransformingValidator(
            field="value",
            validator=inner,
            transform=lambda x: int(x),  # Will fail on non-numeric
        )
        data = {"value": "not a number"}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "Transform failed" in result.errors[0].message


# =============================================================================
# Custom Validator Tests
# =============================================================================


class TestCustomValidator:
    """Tests for creating custom validators."""

    def test_custom_validator_protocol(self, context):
        """Test that custom validator can implement protocol."""

        class EvenNumberValidator:
            def __init__(self, field: str):
                self.field = field

            @property
            def name(self) -> str:
                return "EvenNumberValidator"

            def validate(self, data: dict, ctx: ValidationContext) -> ValidationResult:
                result = ValidationResult()
                value = data.get(self.field)
                if value is not None and isinstance(value, int) and value % 2 != 0:
                    result.add_error(self.field, "Value must be even", value=value)
                return result

        validator = EvenNumberValidator(field="count")
        data = {"count": 3}
        result = validator.validate(data, context)
        assert not result.is_valid
        assert "must be even" in result.errors[0].message

        data = {"count": 4}
        result = validator.validate(data, context)
        assert result.is_valid
