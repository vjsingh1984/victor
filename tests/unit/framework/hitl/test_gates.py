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

"""Tests for HITL gate patterns."""

import asyncio
import pytest

from victor.framework.hitl.gates import (
    ApprovalGate,
    ApprovalResponse,
    ChoiceInputGate,
    ChoiceResponse,
    ChoiceValidator,
    ConfirmationDialogGate,
    LengthValidator,
    PatternValidator,
    ReviewGate,
    RequiredValidator,
    ReviewResponse,
    TextResponse,
    TextInput,
    TextInputGate,
)


# =============================================================================
# Fallback Strategy Tests
# =============================================================================


class TestFallbackStrategies:
    """Tests for fallback strategies."""

    def test_abort_strategy(self):
        """Abort strategy should have correct values."""
        from victor.framework.hitl.protocols import FallbackStrategy

        strategy = FallbackStrategy.abort()
        assert strategy.behavior.value == "abort"

    def test_continue_with_default_strategy(self):
        """Continue strategy should store default value."""
        from victor.framework.hitl.protocols import FallbackStrategy

        strategy = FallbackStrategy.continue_with_default("default_value")
        assert strategy.behavior.value == "continue"
        assert strategy.default_value == "default_value"

    def test_skip_strategy(self):
        """Skip strategy should have correct values."""
        from victor.framework.hitl.protocols import FallbackStrategy

        strategy = FallbackStrategy.skip()
        assert strategy.behavior.value == "skip"

    def test_retry_strategy(self):
        """Retry strategy should store retry config."""
        from victor.framework.hitl.protocols import FallbackStrategy

        strategy = FallbackStrategy.retry(max_retries=5, delay=10.0)
        assert strategy.behavior.value == "retry"
        assert strategy.max_retries == 5
        assert strategy.retry_delay == 10.0


# =============================================================================
# Validator Tests
# =============================================================================


class TestRequiredValidator:
    """Tests for RequiredValidator."""

    @pytest.mark.asyncio
    async def test_valid_input_passes(self):
        """RequiredValidator should pass non-empty input."""
        validator = RequiredValidator()
        is_valid, error = await validator.validate("test input")
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_empty_string_fails(self):
        """RequiredValidator should fail on empty string."""
        validator = RequiredValidator()
        is_valid, error = await validator.validate("")
        assert is_valid is False
        assert error == "This field is required"

    @pytest.mark.asyncio
    async def test_whitespace_only_fails(self):
        """RequiredValidator should fail on whitespace only."""
        validator = RequiredValidator()
        is_valid, error = await validator.validate("   ")
        assert is_valid is False
        assert error == "This field is required"

    @pytest.mark.asyncio
    async def test_none_fails(self):
        """RequiredValidator should fail on None."""
        validator = RequiredValidator()
        is_valid, error = await validator.validate(None)
        assert is_valid is False
        assert error == "This field is required"


class TestLengthValidator:
    """Tests for LengthValidator."""

    @pytest.mark.asyncio
    async def test_valid_length_passes(self):
        """LengthValidator should pass valid length."""
        validator = LengthValidator(min_length=3, max_length=10)
        is_valid, error = await validator.validate("hello")
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_too_short_fails(self):
        """LengthValidator should fail on too short."""
        validator = LengthValidator(min_length=5, max_length=10)
        is_valid, error = await validator.validate("hi")
        assert is_valid is False
        assert "Minimum length" in error

    @pytest.mark.asyncio
    async def test_too_long_fails(self):
        """LengthValidator should fail on too long."""
        validator = LengthValidator(min_length=1, max_length=5)
        is_valid, error = await validator.validate("too long string")
        assert is_valid is False
        assert "Maximum length" in error

    @pytest.mark.asyncio
    async def test_non_string_fails(self):
        """LengthValidator should fail on non-string."""
        validator = LengthValidator()
        is_valid, error = await validator.validate(123)
        assert is_valid is False
        assert error == "Value must be a string"


class TestPatternValidator:
    """Tests for PatternValidator."""

    @pytest.mark.asyncio
    async def test_matching_pattern_passes(self):
        """PatternValidator should pass matching pattern."""
        validator = PatternValidator(r"^\d{3}-\d{3}-\d{4}$", "Invalid phone format")
        is_valid, error = await validator.validate("123-456-7890")
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_non_matching_fails(self):
        """PatternValidator should fail non-matching."""
        validator = PatternValidator(r"^\d{3}-\d{3}-\d{4}$", "Invalid phone format")
        is_valid, error = await validator.validate("not-a-phone")
        assert is_valid is False
        assert error == "Invalid phone format"


class TestChoiceValidator:
    """Tests for ChoiceValidator."""

    @pytest.mark.asyncio
    async def test_valid_choice_passes(self):
        """ChoiceValidator should pass valid choice."""
        validator = ChoiceValidator(["a", "b", "c"])
        is_valid, error = await validator.validate("b")
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_invalid_choice_fails(self):
        """ChoiceValidator should fail invalid choice."""
        validator = ChoiceValidator(["a", "b", "c"])
        is_valid, error = await validator.validate("d")
        assert is_valid is False
        assert "Must be one of" in error


# =============================================================================
# ApprovalGate Tests
# =============================================================================


class TestApprovalGate:
    """Tests for ApprovalGate."""

    def test_gate_initialization(self):
        """ApprovalGate should initialize with correct properties."""
        gate = ApprovalGate(
            title="Test Approval",
            description="Test description",
        )

        assert gate.gate_type == "approval"
        assert gate.title == "Test Approval"
        assert gate.prompt == "Test description"
        assert gate.timeout_seconds == 300.0
        assert gate.is_required is True

    def test_gate_with_context(self):
        """ApprovalGate should accept context."""
        gate = ApprovalGate(
            title="Test",
            description="Test",
            context={"key": "value"},
        )

        assert gate.context == {"key": "value"}

    def test_gate_with_custom_timeout(self):
        """ApprovalGate should accept custom timeout."""
        gate = ApprovalGate(
            title="Test",
            description="Test",
            timeout_seconds=60,
        )

        assert gate.timeout_seconds == 60.0

    def test_with_context_creates_new_gate(self):
        """with_context should create a new gate with merged context."""
        gate = ApprovalGate(
            title="Test",
            description="Test",
            context={"a": 1},
        )

        new_gate = gate.with_context({"b": 2})

        assert new_gate is not gate
        assert new_gate.context == {"a": 1, "b": 2}

    def test_with_timeout_creates_new_gate(self):
        """with_timeout should create a new gate with custom timeout."""
        gate = ApprovalGate(title="Test", description="Test")

        new_gate = gate.with_timeout(120)

        assert new_gate.timeout_seconds == 120

    @pytest.mark.asyncio
    async def test_execute_without_handler_returns_default(self):
        """Execute without handler should auto-approve for testing."""
        gate = ApprovalGate(title="Test", description="Test approval")

        result = await gate.execute()

        assert isinstance(result, ApprovalResponse)
        assert result.approved is True
        assert result.responder == "system"

    @pytest.mark.asyncio
    async def test_execute_with_handler_uses_response(self):
        """Execute with handler should use handler response."""

        async def mock_handler(**kwargs):
            return type(
                "MockResponse",
                (),
                {
                    "approved": False,
                    "reason": "Denied by test",
                    "responder": "test_handler",
                },
            )()

        gate = ApprovalGate(title="Test", description="Test")

        result = await gate.execute(handler=mock_handler)

        assert result.approved is False
        assert result.reason == "Denied by test"
        assert result.responder == "test_handler"


# =============================================================================
# TextInputGate Tests
# =============================================================================


class TestTextInputGate:
    """Tests for TextInputGate."""

    def test_text_input_initialization(self):
        """TextInputGate should initialize with correct properties."""
        gate = TextInputGate(
            title="Input Test",
            prompt="Enter text",
        )

        assert gate.gate_type == "text_input"
        assert gate.title == "Input Test"
        assert gate.placeholder == ""
        assert gate.required is True

    def test_text_input_with_placeholder(self):
        """TextInputGate should accept placeholder."""
        gate = TextInputGate(
            title="Test",
            prompt="Test",
            placeholder="Enter value here",
        )

        assert gate.placeholder == "Enter value here"

    @pytest.mark.asyncio
    async def test_execute_returns_default_value(self):
        """Execute without handler should return default value."""
        gate = TextInputGate(
            title="Test",
            prompt="Test",
            default_value="default text",
        )

        result = await gate.execute()

        assert isinstance(result, TextResponse)
        assert result.text == "default text"
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_required_validation_fails_on_empty(self):
        """Required gate should fail on empty input."""
        gate = TextInputGate(
            title="Test",
            prompt="Test",
            required=True,
        )

        # Simulate empty response
        async def mock_handler(**kwargs):
            return type("MockResponse", (), {"value": ""})()

        result = await gate.execute(handler=mock_handler)

        assert result.approved is False
        assert "required" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_length_validation(self):
        """Gate should validate length constraints."""
        gate = TextInputGate(
            title="Test",
            prompt="Test",
            min_length=5,
            max_length=10,
        )

        # Too short
        async def mock_handler_short(**kwargs):
            return type("MockResponse", (), {"value": "abc"})()

        result = await gate.execute(handler=mock_handler_short)
        assert result.approved is False

        # Valid
        async def mock_handler_valid(**kwargs):
            return type("MockResponse", (), {"value": "valid"})()

        result = await gate.execute(handler=mock_handler_valid)
        assert result.approved is True


# =============================================================================
# ChoiceInputGate Tests
# =============================================================================


class TestChoiceInputGate:
    """Tests for ChoiceInputGate."""

    def test_choice_initialization(self):
        """ChoiceInputGate should initialize with choices."""
        gate = ChoiceInputGate(
            title="Select Option",
            prompt="Choose one",
            choices=["a", "b", "c"],
        )

        assert gate.gate_type == "choice_input"
        assert gate.choices == ["a", "b", "c"]
        assert gate.default_index == 0

    @pytest.mark.asyncio
    async def test_execute_returns_default_choice(self):
        """Execute without handler should return default choice."""
        gate = ChoiceInputGate(
            title="Test",
            prompt="Test",
            choices=["a", "b", "c"],
            default_index=1,
        )

        result = await gate.execute()

        assert isinstance(result, ChoiceResponse)
        assert result.selected == "b"
        assert result.index == 1

    @pytest.mark.asyncio
    async def test_execute_with_handler(self):
        """Execute with handler should use selected value."""

        async def mock_handler(**kwargs):
            return type("MockResponse", (), {"value": "c"})()

        gate = ChoiceInputGate(
            title="Test",
            prompt="Test",
            choices=["a", "b", "c"],
        )

        result = await gate.execute(handler=mock_handler)

        assert result.selected == "c"
        assert result.index == 2

    @pytest.mark.asyncio
    async def test_invalid_choice_rejected(self):
        """Invalid choice should be rejected."""

        async def mock_handler(**kwargs):
            return type("MockResponse", (), {"value": "invalid"})()

        gate = ChoiceInputGate(
            title="Test",
            prompt="Test",
            choices=["a", "b", "c"],
        )

        result = await gate.execute(handler=mock_handler)

        assert result.approved is False


# =============================================================================
# ConfirmationDialogGate Tests
# =============================================================================


class TestConfirmationDialogGate:
    """Tests for ConfirmationDialogGate."""

    def test_confirmation_initialization(self):
        """ConfirmationDialogGate should initialize correctly."""
        gate = ConfirmationDialogGate(
            title="Confirm",
            prompt="Proceed?",
        )

        assert gate.gate_type == "confirmation"
        assert gate.default_approved is False
        assert gate.timeout_seconds == 60.0

    def test_confirmation_with_default_approved(self):
        """Should accept default_approved parameter."""
        gate = ConfirmationDialogGate(
            title="Confirm",
            prompt="Proceed?",
            default_approved=True,
        )

        assert gate.default_approved is True

    @pytest.mark.asyncio
    async def test_execute_uses_default(self):
        """Execute without handler should use default."""
        gate = ConfirmationDialogGate(
            title="Confirm",
            prompt="Proceed?",
            default_approved=True,
        )

        result = await gate.execute()

        assert result.approved is True


# =============================================================================
# ReviewGate Tests
# =============================================================================


class TestReviewGate:
    """Tests for ReviewGate."""

    def test_review_initialization(self):
        """ReviewGate should initialize with content."""
        gate = ReviewGate(
            title="Review Content",
            content="Content to review",
        )

        assert gate.gate_type == "review"
        assert gate.content == "Content to review"
        assert gate.allow_modifications is True

    def test_review_without_modifications(self):
        """Should support disallowing modifications."""
        gate = ReviewGate(
            title="Review",
            content="Content",
            allow_modifications=False,
        )

        assert gate.allow_modifications is False

    @pytest.mark.asyncio
    async def test_execute_auto_approves(self):
        """Execute without handler should auto-approve."""
        gate = ReviewGate(
            title="Review",
            content="Content",
        )

        result = await gate.execute()

        assert isinstance(result, ReviewResponse)
        assert result.approved is True
        assert result.modifications is None

    @pytest.mark.asyncio
    async def test_execute_with_modifications(self):
        """Execute with handler should support modifications."""

        async def mock_handler(**kwargs):
            return type(
                "MockResponse",
                (),
                {
                    "approved": True,
                    "modifications": {"change": "modified value"},
                    "reason": "Made improvements",
                },
            )()

        gate = ReviewGate(
            title="Review",
            content="Content",
            allow_modifications=True,
        )

        result = await gate.execute(handler=mock_handler)

        assert result.approved is True
        assert result.modifications == {"change": "modified value"}
        assert result.comments == "Made improvements"


# =============================================================================
# Alias Tests
# =============================================================================


class TestAliases:
    """Tests for backward compatibility aliases."""

    def test_text_input_alias(self):
        """TextInput should be an alias for TextInputGate."""
        assert TextInput is TextInputGate

    def test_text_input_creates_same_instance(self):
        """TextInput should create TextInputGate instances."""
        gate = TextInput(
            title="Test",
            prompt="Test",
        )

        assert isinstance(gate, TextInputGate)
