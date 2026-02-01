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

"""Tests for ValidationCoordinator.

This test file validates the ValidationCoordinator which handles:
- Intelligent response validation (quality scoring, grounding verification)
- Tool call validation (name format, enabled status)
- Context overflow checking
- Cancellation checking
- Input parameter validation

Test Strategy:
1. Test all validation methods with various inputs
2. Test error handling and edge cases
3. Test delegation to dependent coordinators
4. Test configuration options
5. Test result objects and their methods
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from victor.agent.coordinators.validation_coordinator import (
    ValidationCoordinator,
    ValidationCoordinatorConfig,
    ValidationResult,
    IntelligentValidationResult,
    ToolCallValidationResult,
    ContextValidationResult,
)


class TestValidationResult:
    """Test suite for ValidationResult base class."""

    def test_validation_result_initialization(self):
        """Test ValidationResult initialization with default values."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.metadata == {}

    def test_validation_result_with_errors(self):
        """Test ValidationResult with custom errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            metadata={"key": "value"},
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.metadata == {"key": "value"}

    def test_add_error_updates_validity(self):
        """Test that add_error sets is_valid to False."""
        result = ValidationResult(is_valid=True)
        result.add_error("Test error")

        assert result.is_valid is False
        assert "Test error" in result.errors

    def test_add_warning_preserves_validity(self):
        """Test that add_warning does not change is_valid."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")

        assert result.is_valid is True
        assert "Test warning" in result.warnings

    def test_add_multiple_errors(self):
        """Test adding multiple errors."""
        result = ValidationResult(is_valid=True)
        result.add_error("Error 1")
        result.add_error("Error 2")
        result.add_error("Error 3")

        assert len(result.errors) == 3
        assert result.is_valid is False

    def test_add_multiple_warnings(self):
        """Test adding multiple warnings."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")

        assert len(result.warnings) == 2
        assert result.is_valid is True


class TestIntelligentValidationResult:
    """Test suite for IntelligentValidationResult."""

    def test_initialization_with_defaults(self):
        """Test IntelligentValidationResult with default values."""
        result = IntelligentValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.quality_score == 0.5
        assert result.grounding_score == 0.5
        assert result.is_grounded is True
        assert result.grounding_issues == []
        assert result.should_finalize is False
        assert result.should_retry is False
        assert result.finalize_reason == ""
        assert result.grounding_feedback == ""

    def test_meets_quality_threshold_default(self):
        """Test meets_quality_threshold with default threshold."""
        result = IntelligentValidationResult(is_valid=True, quality_score=0.7)

        assert result.meets_quality_threshold() is True

    def test_meets_quality_threshold_custom(self):
        """Test meets_quality_threshold with custom threshold."""
        result = IntelligentValidationResult(is_valid=True, quality_score=0.6)

        assert result.meets_quality_threshold(0.5) is True
        assert result.meets_quality_threshold(0.7) is False

    def test_meets_quality_threshold_exact(self):
        """Test meets_quality_threshold at exact threshold boundary."""
        result = IntelligentValidationResult(is_valid=True, quality_score=0.5)

        assert result.meets_quality_threshold(0.5) is True

    def test_meets_grounding_threshold_default(self):
        """Test meets_grounding_threshold with default threshold."""
        result = IntelligentValidationResult(is_valid=True, grounding_score=0.8)

        assert result.meets_grounding_threshold() is True

    def test_meets_grounding_threshold_custom(self):
        """Test meets_grounding_threshold with custom threshold."""
        result = IntelligentValidationResult(is_valid=True, grounding_score=0.6)

        assert result.meets_grounding_threshold(0.5) is True
        assert result.meets_grounding_threshold(0.7) is False

    def test_meets_grounding_threshold_exact(self):
        """Test meets_grounding_threshold at exact threshold boundary."""
        result = IntelligentValidationResult(is_valid=True, grounding_score=0.7)

        assert result.meets_grounding_threshold(0.7) is True

    def test_all_fields_populated(self):
        """Test IntelligentValidationResult with all fields set."""
        result = IntelligentValidationResult(
            is_valid=False,
            quality_score=0.3,
            grounding_score=0.4,
            is_grounded=False,
            grounding_issues=["Hallucination detected"],
            should_finalize=True,
            should_retry=True,
            finalize_reason="Quality too low",
            grounding_feedback="Add more context",
            errors=["Quality score below threshold"],
            warnings=["Consider retrying"],
            metadata={"model": "test"},
        )

        assert result.is_valid is False
        assert result.quality_score == 0.3
        assert result.grounding_score == 0.4
        assert result.is_grounded is False
        assert len(result.grounding_issues) == 1
        assert result.should_finalize is True
        assert result.should_retry is True
        assert result.finalize_reason == "Quality too low"
        assert result.grounding_feedback == "Add more context"
        assert len(result.errors) == 1
        assert len(result.warnings) == 1


class TestToolCallValidationResult:
    """Test suite for ToolCallValidationResult."""

    def test_initialization_with_defaults(self):
        """Test ToolCallValidationResult with default values."""
        result = ToolCallValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.tool_calls is None
        assert result.filtered_count == 0
        assert result.remaining_content == ""

    def test_initialization_with_values(self):
        """Test ToolCallValidationResult with custom values."""
        tool_calls = [{"name": "test_tool", "arguments": {}}]
        result = ToolCallValidationResult(
            is_valid=True,
            tool_calls=tool_calls,
            filtered_count=2,
            remaining_content="Some text",
        )

        assert result.is_valid is True
        assert result.tool_calls == tool_calls
        assert result.filtered_count == 2
        assert result.remaining_content == "Some text"


class TestContextValidationResult:
    """Test suite for ContextValidationResult."""

    def test_initialization_with_defaults(self):
        """Test ContextValidationResult with default values."""
        result = ContextValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.is_overflow is False
        assert result.current_size == 0
        assert result.max_size == 0
        assert result.utilization_percent == 0.0

    def test_initialization_with_values(self):
        """Test ContextValidationResult with custom values."""
        result = ContextValidationResult(
            is_valid=True,
            is_overflow=False,
            current_size=150000,
            max_size=200000,
            utilization_percent=75.0,
        )

        assert result.is_valid is True
        assert result.is_overflow is False
        assert result.current_size == 150000
        assert result.max_size == 200000
        assert result.utilization_percent == 75.0


class TestValidationCoordinatorConfig:
    """Test suite for ValidationCoordinatorConfig."""

    def test_default_configuration(self):
        """Test ValidationCoordinatorConfig with default values."""
        config = ValidationCoordinatorConfig()

        assert config.enable_intelligent_validation is True
        assert config.enable_tool_call_validation is True
        assert config.enable_context_validation is True
        assert config.min_response_length == 50
        assert config.quality_threshold == 0.5
        assert config.grounding_threshold == 0.7
        assert config.max_garbage_chunks == 3

    def test_custom_configuration(self):
        """Test ValidationCoordinatorConfig with custom values."""
        config = ValidationCoordinatorConfig(
            enable_intelligent_validation=False,
            enable_tool_call_validation=False,
            enable_context_validation=False,
            min_response_length=100,
            quality_threshold=0.7,
            grounding_threshold=0.8,
            max_garbage_chunks=5,
        )

        assert config.enable_intelligent_validation is False
        assert config.enable_tool_call_validation is False
        assert config.enable_context_validation is False
        assert config.min_response_length == 100
        assert config.quality_threshold == 0.7
        assert config.grounding_threshold == 0.8
        assert config.max_garbage_chunks == 5


class TestValidationCoordinator:
    """Test suite for ValidationCoordinator."""

    # ========================================================================
    # Fixtures
    # ========================================================================

    @pytest.fixture
    def mock_intelligent_integration(self) -> Mock:
        """Create mock intelligent integration."""
        integration = AsyncMock()
        integration.validate_intelligent_response = AsyncMock()
        return integration

    @pytest.fixture
    def mock_context_manager(self) -> Mock:
        """Create mock context manager."""
        context = Mock()
        context.check_context_overflow = Mock(return_value=False)
        context.get_context_metrics = Mock()
        context.get_max_context_chars = Mock(return_value=200000)
        return context

    @pytest.fixture
    def mock_response_coordinator(self) -> Mock:
        """Create mock response coordinator."""
        response = Mock()
        response.is_valid_tool_name = Mock(return_value=True)
        response.parse_and_validate_tool_calls = Mock()
        return response

    @pytest.fixture
    def mock_cancel_event(self) -> Mock:
        """Create mock cancel event."""
        event = Mock()
        event.is_set = Mock(return_value=False)
        return event

    @pytest.fixture
    def mock_metrics_coordinator(self) -> Mock:
        """Create mock metrics coordinator."""
        metrics = Mock()
        metrics.is_cancellation_requested = Mock(return_value=False)
        metrics.is_cancelled = Mock(return_value=False)
        return metrics

    @pytest.fixture
    def coordinator(
        self,
        mock_intelligent_integration: Mock,
        mock_context_manager: Mock,
        mock_response_coordinator: Mock,
    ) -> ValidationCoordinator:
        """Create ValidationCoordinator with default mocks."""
        return ValidationCoordinator(
            intelligent_integration=mock_intelligent_integration,
            context_manager=mock_context_manager,
            response_coordinator=mock_response_coordinator,
        )

    @pytest.fixture
    def coordinator_with_cancel(
        self,
        mock_intelligent_integration: Mock,
        mock_context_manager: Mock,
        mock_response_coordinator: Mock,
        mock_cancel_event: Mock,
    ) -> ValidationCoordinator:
        """Create coordinator with cancel event."""
        return ValidationCoordinator(
            intelligent_integration=mock_intelligent_integration,
            context_manager=mock_context_manager,
            response_coordinator=mock_response_coordinator,
            cancel_event=mock_cancel_event,
        )

    @pytest.fixture
    def coordinator_with_metrics(
        self,
        mock_intelligent_integration: Mock,
        mock_context_manager: Mock,
        mock_response_coordinator: Mock,
        mock_metrics_coordinator: Mock,
    ) -> ValidationCoordinator:
        """Create coordinator with metrics coordinator."""
        return ValidationCoordinator(
            intelligent_integration=mock_intelligent_integration,
            context_manager=mock_context_manager,
            response_coordinator=mock_response_coordinator,
            metrics_coordinator=mock_metrics_coordinator,
        )

    # ========================================================================
    # Properties
    # ========================================================================

    def test_config_property(self, coordinator: ValidationCoordinator):
        """Test config property returns configuration."""
        config = coordinator.config

        assert isinstance(config, ValidationCoordinatorConfig)
        assert config.enable_intelligent_validation is True

    def test_config_property_custom(self):
        """Test config property with custom configuration."""
        custom_config = ValidationCoordinatorConfig(
            enable_intelligent_validation=False,
            quality_threshold=0.8,
        )
        coordinator = ValidationCoordinator(config=custom_config)

        assert coordinator.config == custom_config
        assert coordinator.config.quality_threshold == 0.8

    def test_intelligent_integration_property(self, coordinator: ValidationCoordinator):
        """Test intelligent_integration property."""
        assert coordinator.intelligent_integration is not None

    def test_context_manager_property(self, coordinator: ValidationCoordinator):
        """Test context_manager property."""
        assert coordinator.context_manager is not None

    def test_response_coordinator_property(self, coordinator: ValidationCoordinator):
        """Test response_coordinator property."""
        assert coordinator.response_coordinator is not None

    # ========================================================================
    # Intelligent Response Validation
    # ========================================================================

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_success(
        self, coordinator: ValidationCoordinator, mock_intelligent_integration: Mock
    ):
        """Test successful intelligent response validation."""
        # Setup
        mock_intelligent_integration.validate_intelligent_response.return_value = {
            "quality_score": 0.8,
            "grounding_score": 0.9,
            "is_grounded": True,
            "grounding_issues": [],
            "should_finalize": False,
            "should_retry": False,
            "finalize_reason": "",
            "grounding_feedback": "",
        }

        # Execute
        result = await coordinator.validate_intelligent_response(
            response="This is a well-grounded response with good quality.",
            query="Test query",
            tool_calls=2,
            task_type="analysis",
        )

        # Assert
        assert result is not None
        assert isinstance(result, IntelligentValidationResult)
        assert result.is_valid is True
        assert result.quality_score == 0.8
        assert result.grounding_score == 0.9
        assert result.is_grounded is True

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_below_threshold(
        self, coordinator: ValidationCoordinator, mock_intelligent_integration: Mock
    ):
        """Test validation when scores are below threshold."""
        # Setup
        mock_intelligent_integration.validate_intelligent_response.return_value = {
            "quality_score": 0.3,
            "grounding_score": 0.4,
            "is_grounded": False,
            "grounding_issues": ["Hallucination detected"],
            "should_finalize": True,
            "should_retry": True,
            "finalize_reason": "Quality too low",
            "grounding_feedback": "Add more context",
        }

        # Execute
        result = await coordinator.validate_intelligent_response(
            response="Poor quality response that exceeds minimum length for validation.",
            query="Test query",
            tool_calls=1,
            task_type="general",
        )

        # Assert
        assert result is not None
        assert result.is_valid is False
        assert result.quality_score == 0.3
        assert result.grounding_score == 0.4
        assert result.is_grounded is False
        assert len(result.errors) > 0
        assert "below threshold" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_short_response(
        self, coordinator: ValidationCoordinator
    ):
        """Test that short responses skip validation."""
        # Execute
        result = await coordinator.validate_intelligent_response(
            response="Short",
            query="Test",
            tool_calls=0,
            task_type="general",
        )

        # Assert - should return None for short responses
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_empty_response(
        self, coordinator: ValidationCoordinator
    ):
        """Test that empty responses skip validation."""
        # Execute
        result = await coordinator.validate_intelligent_response(
            response="",
            query="Test",
            tool_calls=0,
            task_type="general",
        )

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_whitespace_only(
        self, coordinator: ValidationCoordinator
    ):
        """Test that whitespace-only responses skip validation."""
        # Execute
        result = await coordinator.validate_intelligent_response(
            response="   \n\t  ",
            query="Test",
            tool_calls=0,
            task_type="general",
        )

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_disabled_integration(self):
        """Test validation when intelligent integration is None."""
        # Setup - coordinator without integration
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=Mock(),
            response_coordinator=Mock(),
        )

        # Execute
        result = await coordinator.validate_intelligent_response(
            response="A" * 100,
            query="Test",
            tool_calls=0,
            task_type="general",
        )

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_disabled_in_config(self):
        """Test validation when disabled in config."""
        # Setup
        config = ValidationCoordinatorConfig(enable_intelligent_validation=False)
        coordinator = ValidationCoordinator(
            intelligent_integration=AsyncMock(),
            context_manager=Mock(),
            response_coordinator=Mock(),
            config=config,
        )

        # Execute
        result = await coordinator.validate_intelligent_response(
            response="A" * 100,
            query="Test",
            tool_calls=0,
            task_type="general",
        )

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_integration_returns_none(
        self, coordinator: ValidationCoordinator, mock_intelligent_integration: Mock
    ):
        """Test when integration returns None (error case)."""
        # Setup
        mock_intelligent_integration.validate_intelligent_response.return_value = None

        # Execute
        result = await coordinator.validate_intelligent_response(
            response="A" * 100,
            query="Test",
            tool_calls=0,
            task_type="general",
        )

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_exception_handled(
        self, coordinator: ValidationCoordinator, mock_intelligent_integration: Mock
    ):
        """Test that exceptions are handled gracefully."""
        # Setup
        mock_intelligent_integration.validate_intelligent_response.side_effect = Exception(
            "Validation failed"
        )

        # Execute
        result = await coordinator.validate_intelligent_response(
            response="A" * 100,
            query="Test",
            tool_calls=0,
            task_type="general",
        )

        # Assert - should return None on error
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_partial_data(
        self, coordinator: ValidationCoordinator, mock_intelligent_integration: Mock
    ):
        """Test validation with partial data returned from integration."""
        # Setup - only return quality_score
        mock_intelligent_integration.validate_intelligent_response.return_value = {
            "quality_score": 0.6,
        }

        # Execute
        result = await coordinator.validate_intelligent_response(
            response="A" * 100,
            query="Test",
            tool_calls=0,
            task_type="general",
        )

        # Assert - should use defaults for missing fields
        assert result is not None
        assert result.quality_score == 0.6
        assert result.grounding_score == 0.5  # Default
        assert result.is_grounded is True  # Default

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_custom_min_length(self):
        """Test with custom min_response_length configuration."""
        # Setup
        config = ValidationCoordinatorConfig(min_response_length=10)
        coordinator = ValidationCoordinator(
            intelligent_integration=AsyncMock(),
            context_manager=Mock(),
            response_coordinator=Mock(),
            config=config,
        )

        # Execute - response is exactly 10 chars
        result = await coordinator.validate_intelligent_response(
            response="A" * 10,
            query="Test",
            tool_calls=0,
            task_type="general",
        )

        # Assert - integration should be called (not skipped)
        coordinator.intelligent_integration.validate_intelligent_response.assert_called_once()

    # ========================================================================
    # Tool Call Validation
    # ========================================================================

    def test_validate_tool_call_structure_valid_dict(self, coordinator: ValidationCoordinator):
        """Test tool call structure validation with valid dict."""
        tool_call = {"name": "test_tool", "arguments": {}}

        result = coordinator.validate_tool_call_structure(tool_call)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_tool_call_structure_missing_name(self, coordinator: ValidationCoordinator):
        """Test tool call structure validation with missing name."""
        tool_call = {"arguments": {}}

        result = coordinator.validate_tool_call_structure(tool_call)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "missing 'name' field" in result.errors[0].lower()

    def test_validate_tool_call_structure_not_dict(self, coordinator: ValidationCoordinator):
        """Test tool call structure validation with non-dict."""
        tool_call = "not a dict"

        result = coordinator.validate_tool_call_structure(tool_call)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "not a dict" in result.errors[0].lower()

    def test_validate_tool_call_structure_list(self, coordinator: ValidationCoordinator):
        """Test tool call structure validation with list."""
        tool_call = ["name", "arguments"]

        result = coordinator.validate_tool_call_structure(tool_call)

        assert result.is_valid is False
        assert "not a dict" in result.errors[0].lower()

    def test_validate_tool_call_structure_none(self, coordinator: ValidationCoordinator):
        """Test tool call structure validation with None."""
        tool_call = None

        result = coordinator.validate_tool_call_structure(tool_call)

        assert result.is_valid is False
        assert "not a dict" in result.errors[0].lower()

    def test_validate_tool_name_valid(self, coordinator: ValidationCoordinator):
        """Test tool name validation with valid name."""
        result = coordinator.validate_tool_name("read_file")

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_tool_name_empty(self, coordinator: ValidationCoordinator):
        """Test tool name validation with empty name."""
        result = coordinator.validate_tool_name("")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "empty" in result.errors[0].lower()

    def test_validate_tool_name_with_coordinator(
        self, coordinator: ValidationCoordinator, mock_response_coordinator: Mock
    ):
        """Test tool name validation with response coordinator."""
        mock_response_coordinator.is_valid_tool_name.return_value = False

        result = coordinator.validate_tool_name("invalid_tool")

        assert result.is_valid is False
        assert len(result.errors) == 1
        mock_response_coordinator.is_valid_tool_name.assert_called_once_with("invalid_tool")

    def test_validate_tool_name_without_coordinator(self):
        """Test tool name validation without response coordinator."""
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=None,
            response_coordinator=None,
        )

        # Valid name without coordinator
        result = coordinator.validate_tool_name("valid_tool")
        assert result.is_valid is True

        # Invalid characters without coordinator
        result = coordinator.validate_tool_name('tool"name"')
        assert result.is_valid is False
        assert "invalid characters" in result.errors[0].lower()

    def test_validate_tool_name_invalid_characters(self):
        """Test tool name validation with invalid characters (no coordinator)."""
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=None,
            response_coordinator=None,
        )

        # Test quote
        result = coordinator.validate_tool_name('tool"name"')
        assert result.is_valid is False

        # Test single quote
        result = coordinator.validate_tool_name("tool'name'")
        assert result.is_valid is False

        # Test newline
        result = coordinator.validate_tool_name("tool\nname")
        assert result.is_valid is False

        # Test tab
        result = coordinator.validate_tool_name("tool\tname")
        assert result.is_valid is False

        # Test backslash
        result = coordinator.validate_tool_name("tool\\name")
        assert result.is_valid is False

    def test_validate_and_filter_tool_calls_none(self, coordinator: ValidationCoordinator):
        """Test tool call validation with None."""
        result = coordinator.validate_and_filter_tool_calls(None, "Some content")

        assert result.is_valid is True
        assert result.tool_calls is None
        assert result.remaining_content == "Some content"
        assert result.filtered_count == 0

    def test_validate_and_filter_tool_calls_empty_list(self, coordinator: ValidationCoordinator):
        """Test tool call validation with empty list."""
        result = coordinator.validate_and_filter_tool_calls([], "Some content")

        assert result.is_valid is True
        assert result.tool_calls is None
        assert result.filtered_count == 0

    def test_validate_and_filter_tool_calls_valid(self, coordinator: ValidationCoordinator):
        """Test tool call validation with valid calls."""
        tool_calls = [
            {"name": "read_file", "arguments": {"path": "/tmp/test"}},
            {"name": "write_file", "arguments": {"path": "/tmp/test2"}},
        ]

        result = coordinator.validate_and_filter_tool_calls(tool_calls, "Content")

        assert result.is_valid is True
        assert result.tool_calls is not None
        assert result.filtered_count == 0

    def test_validate_and_filter_tool_calls_filters_invalid_structure(
        self, coordinator: ValidationCoordinator
    ):
        """Test tool call validation filters invalid structures."""
        tool_calls = [
            {"name": "valid_tool", "arguments": {}},
            "invalid_string",
            {"name": "another_valid", "arguments": {}},
        ]

        result = coordinator.validate_and_filter_tool_calls(tool_calls, "Content")

        assert result.tool_calls is not None
        assert result.filtered_count == 1
        assert len(result.errors) == 1

    def test_validate_and_filter_tool_calls_filters_invalid_names(
        self, coordinator: ValidationCoordinator
    ):
        """Test tool call validation filters invalid tool names."""
        # Test without response coordinator (basic validation only)
        coordinator_no_response = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=None,
            response_coordinator=None,
        )

        tool_calls = [
            {"name": "valid_tool", "arguments": {}},
            {"name": 'invalid"tool"', "arguments": {}},
        ]

        result = coordinator_no_response.validate_and_filter_tool_calls(tool_calls, "Content")

        assert result.tool_calls is not None
        assert result.filtered_count == 1
        assert len(result.errors) == 1

    def test_validate_and_filter_tool_calls_with_response_coordinator(
        self, coordinator: ValidationCoordinator, mock_response_coordinator: Mock
    ):
        """Test tool call validation delegates to response coordinator."""
        from victor.agent.coordinators.validation_coordinator import ToolCallValidationResult

        tool_calls = [{"name": "test", "arguments": {}}]
        parse_result = ToolCallValidationResult(
            is_valid=True,
            tool_calls=tool_calls,
            filtered_count=1,
            remaining_content="Remaining",
        )
        mock_response_coordinator.parse_and_validate_tool_calls.return_value = parse_result

        result = coordinator.validate_and_filter_tool_calls(tool_calls, "Content")

        assert result.tool_calls == tool_calls
        assert result.remaining_content == "Remaining"
        assert result.filtered_count == 1
        mock_response_coordinator.parse_and_validate_tool_calls.assert_called_once()

    def test_validate_and_filter_tool_calls_response_coordinator_exception(
        self, coordinator: ValidationCoordinator, mock_response_coordinator: Mock
    ):
        """Test tool call validation handles response coordinator exceptions."""
        mock_response_coordinator.parse_and_validate_tool_calls.side_effect = Exception(
            "Parse failed"
        )

        tool_calls = [{"name": "test", "arguments": {}}]

        result = coordinator.validate_and_filter_tool_calls(tool_calls, "Content")

        # Should add warning but not fail
        assert len(result.warnings) == 1
        assert "parse failed" in result.warnings[0].lower()

    def test_validate_and_filter_tool_calls_disabled_in_config(self):
        """Test tool call validation when disabled in config."""
        config = ValidationCoordinatorConfig(enable_tool_call_validation=False)
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=None,
            response_coordinator=Mock(),
            config=config,
        )

        tool_calls = [{"name": "test", "arguments": {}}]
        result = coordinator.validate_and_filter_tool_calls(tool_calls, "Content")

        # Response coordinator should not be called
        coordinator.response_coordinator.parse_and_validate_tool_calls.assert_not_called()

    # ========================================================================
    # Context Validation
    # ========================================================================

    def test_check_context_overflow_no_manager(self):
        """Test context overflow check without manager."""
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=None,
            response_coordinator=None,
        )

        result = coordinator.check_context_overflow(200000)

        assert result.is_valid is True
        assert result.is_overflow is False
        assert result.current_size == 0
        assert result.max_size == 200000

    def test_check_context_overflow_disabled_in_config(self):
        """Test context overflow check when disabled."""
        config = ValidationCoordinatorConfig(enable_context_validation=False)
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=Mock(),
            response_coordinator=None,
            config=config,
        )

        result = coordinator.check_context_overflow(200000)

        assert result.is_valid is True
        assert result.is_overflow is False
        # Manager should not be called
        coordinator.context_manager.check_context_overflow.assert_not_called()

    def test_check_context_overflow_no_overflow(
        self, coordinator: ValidationCoordinator, mock_context_manager: Mock
    ):
        """Test context overflow check when no overflow."""
        mock_context_manager.check_context_overflow.return_value = False
        mock_metrics = Mock()
        mock_metrics.total_chars = 100000
        mock_context_manager.get_context_metrics.return_value = mock_metrics

        result = coordinator.check_context_overflow(200000)

        assert result.is_valid is True
        assert result.is_overflow is False
        assert result.current_size == 100000
        assert result.utilization_percent == 50.0
        assert len(result.warnings) == 0

    def test_check_context_overflow_with_overflow(
        self, coordinator: ValidationCoordinator, mock_context_manager: Mock
    ):
        """Test context overflow check when overflow detected."""
        mock_context_manager.check_context_overflow.return_value = True
        mock_metrics = Mock()
        mock_metrics.total_chars = 190000
        mock_context_manager.get_context_metrics.return_value = mock_metrics

        result = coordinator.check_context_overflow(200000)

        assert result.is_overflow is True
        assert result.current_size == 190000
        assert result.utilization_percent == 95.0
        assert len(result.warnings) == 1
        assert "overflow risk" in result.warnings[0].lower()

    def test_check_context_overflow_zero_max(
        self, coordinator: ValidationCoordinator, mock_context_manager: Mock
    ):
        """Test context overflow check with zero max size."""
        mock_context_manager.check_context_overflow.return_value = False
        mock_metrics = Mock()
        mock_metrics.total_chars = 100000
        mock_context_manager.get_context_metrics.return_value = mock_metrics

        result = coordinator.check_context_overflow(0)

        # Should handle division by zero
        assert result.utilization_percent == 0.0

    def test_check_context_overflow_exception(
        self, coordinator: ValidationCoordinator, mock_context_manager: Mock
    ):
        """Test context overflow check handles exceptions."""
        mock_context_manager.check_context_overflow.side_effect = Exception("Check failed")

        result = coordinator.check_context_overflow(200000)

        # Should add warning but not fail
        assert len(result.warnings) == 1
        assert "check failed" in result.warnings[0].lower()

    def test_get_max_context_chars_with_manager(
        self, coordinator: ValidationCoordinator, mock_context_manager: Mock
    ):
        """Test get_max_context_chars with manager."""
        mock_context_manager.get_max_context_chars.return_value = 300000

        result = coordinator.get_max_context_chars()

        assert result == 300000

    def test_get_max_context_chars_without_manager(self):
        """Test get_max_context_chars without manager."""
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=None,
            response_coordinator=None,
        )

        result = coordinator.get_max_context_chars()

        assert result == 200000  # Default

    def test_get_max_context_chars_exception(
        self, coordinator: ValidationCoordinator, mock_context_manager: Mock
    ):
        """Test get_max_context_chars handles exceptions."""
        mock_context_manager.get_max_context_chars.side_effect = Exception("Failed")

        result = coordinator.get_max_context_chars()

        # Should return default on exception
        assert result == 200000

    # ========================================================================
    # Cancellation Checking
    # ========================================================================

    def test_is_cancelled_with_metrics_coordinator(
        self, coordinator_with_metrics: ValidationCoordinator, mock_metrics_coordinator: Mock
    ):
        """Test is_cancelled with metrics coordinator (preferred method)."""
        mock_metrics_coordinator.is_cancellation_requested.return_value = True

        result = coordinator_with_metrics.is_cancelled()

        assert result is True
        mock_metrics_coordinator.is_cancellation_requested.assert_called_once()

    def test_is_cancelled_with_metrics_coordinator_is_cancelled_method(
        self, coordinator_with_metrics: ValidationCoordinator, mock_metrics_coordinator: Mock
    ):
        """Test is_cancelled with metrics coordinator using is_cancelled method."""
        # Delete the first method to test fallback
        del mock_metrics_coordinator.is_cancellation_requested
        mock_metrics_coordinator.is_cancelled.return_value = True

        result = coordinator_with_metrics.is_cancelled()

        assert result is True
        mock_metrics_coordinator.is_cancelled.assert_called_once()

    def test_is_cancelled_with_cancel_event(
        self, coordinator_with_cancel: ValidationCoordinator, mock_cancel_event: Mock
    ):
        """Test is_cancelled with cancel event (legacy method)."""
        mock_cancel_event.is_set.return_value = True

        result = coordinator_with_cancel.is_cancelled()

        assert result is True
        mock_cancel_event.is_set.assert_called_once()

    def test_is_cancelled_metrics_coordinator_exception(
        self, coordinator_with_metrics: ValidationCoordinator, mock_metrics_coordinator: Mock
    ):
        """Test is_cancelled when metrics coordinator raises exception."""
        mock_metrics_coordinator.is_cancellation_requested.side_effect = Exception("Failed")

        result = coordinator_with_metrics.is_cancelled()

        # Should fall back gracefully
        assert result is False

    def test_is_cancelled_cancel_event_exception(
        self, coordinator_with_cancel: ValidationCoordinator, mock_cancel_event: Mock
    ):
        """Test is_cancelled when cancel event raises exception."""
        mock_cancel_event.is_set.side_effect = Exception("Failed")

        result = coordinator_with_cancel.is_cancelled()

        # Should handle gracefully
        assert result is False

    def test_is_cancelled_no_cancel_mechanisms(self):
        """Test is_cancelled when no cancel mechanisms configured."""
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=None,
            response_coordinator=None,
            cancel_event=None,
            metrics_coordinator=None,
        )

        result = coordinator.is_cancelled()

        assert result is False

    # ========================================================================
    # Input Parameter Validation
    # ========================================================================

    def test_validate_query_valid(self, coordinator: ValidationCoordinator):
        """Test query validation with valid input."""
        result = coordinator.validate_query("What is the meaning of life?")

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_query_empty(self, coordinator: ValidationCoordinator):
        """Test query validation with empty string."""
        result = coordinator.validate_query("")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "empty" in result.errors[0].lower()

    def test_validate_query_none(self, coordinator: ValidationCoordinator):
        """Test query validation with None."""
        result = coordinator.validate_query(None)

        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_validate_query_not_string(self, coordinator: ValidationCoordinator):
        """Test query validation with non-string type."""
        result = coordinator.validate_query(123)

        assert result.is_valid is False
        assert "string" in result.errors[0].lower()

    def test_validate_query_long_warning(self, coordinator: ValidationCoordinator):
        """Test query validation with very long query."""
        long_query = "A" * 100001

        result = coordinator.validate_query(long_query)

        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert "very long" in result.warnings[0].lower()

    def test_validate_task_type_valid(self, coordinator: ValidationCoordinator):
        """Test task type validation with valid type."""
        result = coordinator.validate_task_type("analysis")

        assert result.is_valid is True
        assert len(result.warnings) == 0

    def test_validate_task_type_empty(self, coordinator: ValidationCoordinator):
        """Test task type validation with empty type."""
        result = coordinator.validate_task_type("")

        assert result.is_valid is True  # Empty is allowed (warning only)
        assert len(result.warnings) == 1
        assert "empty" in result.warnings[0].lower()

    def test_validate_task_type_unknown(self, coordinator: ValidationCoordinator):
        """Test task type validation with unknown type."""
        result = coordinator.validate_task_type("unknown_type")

        assert result.is_valid is True  # Unknown is allowed (warning only)
        assert len(result.warnings) == 1
        assert "unknown" in result.warnings[0].lower()

    def test_validate_task_type_case_insensitive(self, coordinator: ValidationCoordinator):
        """Test task type validation is case-insensitive."""
        result = coordinator.validate_task_type("ANALYSIS")

        assert result.is_valid is True
        assert len(result.warnings) == 0

    def test_validate_task_type_all_known_types(self, coordinator: ValidationCoordinator):
        """Test task type validation with all known types."""
        known_types = [
            "general",
            "analysis",
            "edit",
            "debug",
            "test",
            "refactor",
            "documentation",
            "planning",
        ]

        for task_type in known_types:
            result = coordinator.validate_task_type(task_type)
            assert result.is_valid is True
            assert len(result.warnings) == 0

    # ========================================================================
    # Composite Validation
    # ========================================================================

    @pytest.mark.asyncio
    async def test_validate_request_all_valid(
        self, coordinator: ValidationCoordinator, mock_context_manager: Mock
    ):
        """Test validate_request with all valid inputs."""
        mock_context_manager.check_context_overflow.return_value = False

        result = await coordinator.validate_request(
            query="Test query",
            task_type="analysis",
            max_context_chars=200000,
        )

        assert result.is_valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_request_invalid_query(
        self, coordinator: ValidationCoordinator, mock_context_manager: Mock
    ):
        """Test validate_request with invalid query."""
        mock_context_manager.check_context_overflow.return_value = False

        result = await coordinator.validate_request(
            query="",
            task_type="analysis",
            max_context_chars=200000,
        )

        assert result.is_valid is False
        assert any("empty" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_request_with_warnings(
        self, coordinator: ValidationCoordinator, mock_context_manager: Mock
    ):
        """Test validate_request generates warnings for unknown task type."""
        mock_context_manager.check_context_overflow.return_value = False

        result = await coordinator.validate_request(
            query="Test query",
            task_type="unknown_type",
            max_context_chars=200000,
        )

        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("unknown" in warning.lower() for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_validate_request_context_overflow(
        self, coordinator: ValidationCoordinator, mock_context_manager: Mock
    ):
        """Test validate_request with context overflow."""
        mock_context_manager.check_context_overflow.return_value = True
        mock_metrics = Mock()
        mock_metrics.total_chars = 190000
        mock_context_manager.get_context_metrics.return_value = mock_metrics

        result = await coordinator.validate_request(
            query="Test query",
            task_type="analysis",
            max_context_chars=200000,
        )

        assert result.is_valid is True  # Overflow is warning, not error
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_validate_request_cancelled(
        self,
        coordinator: ValidationCoordinator,
        mock_context_manager: Mock,
        mock_metrics_coordinator: Mock,
    ):
        """Test validate_request when cancelled."""
        coordinator_with_cancel = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=mock_context_manager,
            response_coordinator=None,
            metrics_coordinator=mock_metrics_coordinator,
        )
        mock_context_manager.check_context_overflow.return_value = False
        mock_metrics_coordinator.is_cancellation_requested.return_value = True

        result = await coordinator_with_cancel.validate_request(
            query="Test query",
            task_type="analysis",
            max_context_chars=200000,
        )

        assert result.is_valid is False
        assert any("cancelled" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_request_multiple_issues(
        self, coordinator: ValidationCoordinator, mock_context_manager: Mock
    ):
        """Test validate_request with multiple validation issues."""
        mock_context_manager.check_context_overflow.return_value = True
        mock_metrics = Mock()
        mock_metrics.total_chars = 190000
        mock_context_manager.get_context_metrics.return_value = mock_metrics

        result = await coordinator.validate_request(
            query="",  # Invalid query
            task_type="unknown_type",  # Unknown task type
            max_context_chars=200000,
        )

        assert result.is_valid is False
        assert len(result.errors) >= 1  # At least query error
        assert len(result.warnings) >= 1  # At least task type warning

    # ========================================================================
    # Integration Tests
    # ========================================================================

    def test_coordinator_initialization_logging(
        self, mock_intelligent_integration: Mock, mock_context_manager: Mock
    ):
        """Test that coordinator logs initialization."""
        with patch("victor.agent.coordinators.validation_coordinator.logger") as mock_logger:
            coordinator = ValidationCoordinator(
                intelligent_integration=mock_intelligent_integration,
                context_manager=mock_context_manager,
                response_coordinator=None,
            )

            # Check debug was called
            mock_logger.debug.assert_called_once()
            args = mock_logger.debug.call_args[0][0]
            assert "validationcoordinator initialized" in args.lower()

    def test_coordinator_with_all_dependencies(
        self,
        mock_intelligent_integration: Mock,
        mock_context_manager: Mock,
        mock_response_coordinator: Mock,
        mock_cancel_event: Mock,
        mock_metrics_coordinator: Mock,
    ):
        """Test coordinator initialization with all dependencies."""
        coordinator = ValidationCoordinator(
            intelligent_integration=mock_intelligent_integration,
            context_manager=mock_context_manager,
            response_coordinator=mock_response_coordinator,
            cancel_event=mock_cancel_event,
            metrics_coordinator=mock_metrics_coordinator,
        )

        assert coordinator.intelligent_integration == mock_intelligent_integration
        assert coordinator.context_manager == mock_context_manager
        assert coordinator.response_coordinator == mock_response_coordinator
        assert coordinator._cancel_event == mock_cancel_event
        assert coordinator._metrics_coordinator == mock_metrics_coordinator


class TestValidationCoordinatorEdgeCases:
    """Test edge cases and error conditions for ValidationCoordinator."""

    @pytest.fixture
    def mock_response_coordinator(self) -> Mock:
        """Create mock response coordinator for edge case tests."""
        response = Mock()
        response.is_valid_tool_name = Mock(return_value=True)
        response.parse_and_validate_tool_calls = Mock()
        return response

    def test_validate_tool_name_with_unicode(self):
        """Test tool name validation with unicode characters."""
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=None,
            response_coordinator=None,
        )

        # Unicode should be valid (only specific chars are invalid)
        result = coordinator.validate_tool_name("工具_测试")
        assert result.is_valid is True

    def test_validate_query_with_unicode(self):
        """Test query validation with unicode characters."""
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=None,
            response_coordinator=None,
        )

        result = coordinator.validate_query("Hello 世界 🌍")
        assert result.is_valid is True

    def test_validate_tool_call_with_nested_dict(self):
        """Test tool call validation with nested dictionary."""
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=None,
            response_coordinator=None,
        )

        tool_call = {
            "name": "complex_tool",
            "arguments": {
                "nested": {"deep": {"value": 123}},
                "list": [1, 2, 3],
            },
        }

        result = coordinator.validate_tool_call_structure(tool_call)
        assert result.is_valid is True

    def test_validate_and_filter_tool_calls_all_filtered(self, mock_response_coordinator: Mock):
        """Test tool call validation when all calls are filtered."""
        coordinator = ValidationCoordinator(
            intelligent_integration=None,
            context_manager=None,
            response_coordinator=mock_response_coordinator,
            config=ValidationCoordinatorConfig(enable_tool_call_validation=True),
        )

        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
        ]

        # Mock response coordinator to filter all calls
        from victor.agent.coordinators.validation_coordinator import ToolCallValidationResult

        parse_result = ToolCallValidationResult(
            is_valid=True,
            tool_calls=None,  # All filtered
            filtered_count=2,
            remaining_content="",
        )
        mock_response_coordinator.parse_and_validate_tool_calls.return_value = parse_result

        result = coordinator.validate_and_filter_tool_calls(tool_calls, "Content")

        # All 2 filtered by response coordinator (0 filtered by basic validation + 2 by response coordinator)
        assert result.tool_calls is None
        assert result.filtered_count == 2
        mock_response_coordinator.parse_and_validate_tool_calls.assert_called_once()

    def test_validation_result_metadata_operations(self):
        """Test ValidationResult metadata operations."""
        result = ValidationResult(is_valid=True, metadata={"key1": "value1"})

        result.metadata["key2"] = "value2"
        result.metadata["key1"] = "updated"

        assert result.metadata["key1"] == "updated"
        assert result.metadata["key2"] == "value2"

    def test_intelligent_validation_result_threshold_methods(self):
        """Test IntelligentValidationResult threshold checking methods."""
        result = IntelligentValidationResult(
            is_valid=True,
            quality_score=0.6,
            grounding_score=0.8,
        )

        # Test quality thresholds
        assert result.meets_quality_threshold(0.5) is True
        assert result.meets_quality_threshold(0.6) is True
        assert result.meets_quality_threshold(0.7) is False

        # Test grounding thresholds
        assert result.meets_grounding_threshold(0.7) is True
        assert result.meets_grounding_threshold(0.8) is True
        assert result.meets_grounding_threshold(0.9) is False

    def test_context_validation_result_utilization_calculation(self):
        """Test ContextValidationResult utilization calculation."""
        # Note: utilization_percent is not auto-calculated in __init__
        # It's calculated by check_context_overflow method
        result = ContextValidationResult(
            is_valid=True,
            current_size=150000,
            max_size=200000,
            utilization_percent=75.0,  # Manually set for this test
        )

        # Utilization should be as set
        assert result.utilization_percent == 75.0
