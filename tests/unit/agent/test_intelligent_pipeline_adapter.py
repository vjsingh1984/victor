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

"""Unit tests for IntelligentPipelineAdapter.

Tests adapter for intelligent pipeline integration.
"""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
from dataclasses import dataclass, field

from victor.agent.adapters.intelligent_pipeline_adapter import (
    IntelligentPipelineAdapter,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockIntelligentValidationResult:
    """Mock IntelligentValidationResult."""

    quality_score: float
    grounding_score: float
    is_grounded: bool
    is_valid: bool
    grounding_issues: list = field(default_factory=list)
    should_finalize: bool = False
    should_retry: bool = False
    finalize_reason: str = ""
    grounding_feedback: str = ""


# ============================================================================
# IntelligentPipelineAdapter Tests
# ============================================================================


class TestIntelligentPipelineAdapter:
    """Test suite for IntelligentPipelineAdapter."""

    def test_initialization_with_no_components(self):
        """Test adapter initialization with no components."""
        adapter = IntelligentPipelineAdapter()

        assert adapter.intelligent_integration is None
        assert adapter.validation_coordinator is None
        assert adapter.is_enabled is False

    def test_initialization_with_components(self):
        """Test adapter initialization with components."""
        mock_integration = Mock()
        mock_validator = Mock()

        adapter = IntelligentPipelineAdapter(
            intelligent_integration=mock_integration,
            validation_coordinator=mock_validator,
        )

        assert adapter.intelligent_integration is mock_integration
        assert adapter.validation_coordinator is mock_validator
        assert adapter.is_enabled is True

    @pytest.mark.asyncio
    async def test_prepare_intelligent_request_with_integration(self):
        """Test preparing intelligent request with integration available."""
        mock_integration = AsyncMock()
        mock_integration.prepare_intelligent_request = AsyncMock(
            return_value={"mode": "analysis", "max_tools": 5}
        )

        adapter = IntelligentPipelineAdapter(intelligent_integration=mock_integration)

        result = await adapter.prepare_intelligent_request(
            task="Analyze code",
            task_type="analysis",
            conversation_state=None,
            unified_tracker=None,
        )

        assert result is not None
        assert result["mode"] == "analysis"
        assert result["max_tools"] == 5
        mock_integration.prepare_intelligent_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_intelligent_request_without_integration(self):
        """Test preparing intelligent request without integration."""
        adapter = IntelligentPipelineAdapter()

        result = await adapter.prepare_intelligent_request(
            task="Analyze code", task_type="analysis"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_prepare_intelligent_request_with_exception(self):
        """Test preparing intelligent request when integration raises exception."""
        mock_integration = Mock()
        mock_integration.prepare_intelligent_request = AsyncMock(
            side_effect=Exception("Integration failed")
        )

        adapter = IntelligentPipelineAdapter(intelligent_integration=mock_integration)

        result = await adapter.prepare_intelligent_request(
            task="Analyze code", task_type="analysis"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_with_result(self):
        """Test validating intelligent response with successful result."""
        mock_validator = Mock()
        mock_result = MockIntelligentValidationResult(
            quality_score=0.85,
            grounding_score=0.92,
            is_grounded=True,
            is_valid=True,
            grounding_issues=[],
            should_finalize=False,
            should_retry=False,
        )
        mock_validator.validate_intelligent_response = AsyncMock(return_value=mock_result)

        adapter = IntelligentPipelineAdapter(validation_coordinator=mock_validator)

        result = await adapter.validate_intelligent_response(
            response="The code shows good quality",
            query="Analyze code quality",
            tool_calls=3,
            task_type="analysis",
        )

        assert result is not None
        assert result["quality_score"] == 0.85
        assert result["grounding_score"] == 0.92
        assert result["is_grounded"] is True
        assert result["is_valid"] is True
        mock_validator.validate_intelligent_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_returns_none(self):
        """Test validating intelligent response when validation is skipped."""
        mock_validator = Mock()
        mock_validator.validate_intelligent_response = AsyncMock(return_value=None)

        adapter = IntelligentPipelineAdapter(validation_coordinator=mock_validator)

        result = await adapter.validate_intelligent_response(
            response="Short response",
            query="Test query",
            tool_calls=0,
            task_type="general",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_without_coordinator(self):
        """Test validating intelligent response without coordinator."""
        adapter = IntelligentPipelineAdapter()

        result = await adapter.validate_intelligent_response(
            response="Response", query="Query", tool_calls=1, task_type="edit"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_with_exception(self):
        """Test validating intelligent response when validation raises exception."""
        mock_validator = Mock()
        mock_validator.validate_intelligent_response = AsyncMock(
            side_effect=Exception("Validation failed")
        )

        adapter = IntelligentPipelineAdapter(validation_coordinator=mock_validator)

        result = await adapter.validate_intelligent_response(
            response="Response", query="Query", tool_calls=1, task_type="edit"
        )

        assert result is None

    def test_should_continue_intelligent_with_integration(self):
        """Test checking if should continue with integration available."""
        mock_integration = Mock()
        mock_integration.should_continue_intelligent = Mock(return_value=(True, "Good progress"))

        adapter = IntelligentPipelineAdapter(intelligent_integration=mock_integration)

        should_continue, reason = adapter.should_continue_intelligent()

        assert should_continue is True
        assert reason == "Good progress"
        mock_integration.should_continue_intelligent.assert_called_once()

    def test_should_continue_intelligent_without_integration(self):
        """Test checking if should continue without integration."""
        adapter = IntelligentPipelineAdapter()

        should_continue, reason = adapter.should_continue_intelligent()

        assert should_continue is True
        assert reason == "No intelligent integration"

    def test_should_continue_intelligent_with_exception(self):
        """Test checking if should continue when integration raises exception."""
        mock_integration = Mock()
        mock_integration.should_continue_intelligent = Mock(side_effect=Exception("Check failed"))

        adapter = IntelligentPipelineAdapter(intelligent_integration=mock_integration)

        should_continue, reason = adapter.should_continue_intelligent()

        assert should_continue is True  # Should continue on error
        assert "No intelligent integration" in reason or "Check failed" in reason

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome_with_integration(self):
        """Test recording intelligent outcome with integration available."""
        mock_integration = AsyncMock()
        mock_integration.record_intelligent_outcome = AsyncMock()

        adapter = IntelligentPipelineAdapter(intelligent_integration=mock_integration)

        await adapter.record_intelligent_outcome(
            success=True,
            quality_score=0.9,
            user_satisfied=True,
            completed=True,
        )

        mock_integration.record_intelligent_outcome.assert_called_once_with(
            success=True,
            quality_score=0.9,
            user_satisfied=True,
            completed=True,
        )

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome_without_integration(self):
        """Test recording intelligent outcome without integration."""
        adapter = IntelligentPipelineAdapter()

        # Should not raise exception
        await adapter.record_intelligent_outcome(
            success=True, quality_score=0.8, user_satisfied=False, completed=True
        )

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome_with_exception(self):
        """Test recording intelligent outcome when integration raises exception."""
        mock_integration = Mock()
        mock_integration.record_intelligent_outcome = AsyncMock(
            side_effect=Exception("Recording failed")
        )

        adapter = IntelligentPipelineAdapter(intelligent_integration=mock_integration)

        # Should not raise exception
        await adapter.record_intelligent_outcome(
            success=False, quality_score=0.5, user_satisfied=True, completed=False
        )

    @pytest.mark.asyncio
    async def test_full_integration_flow(self):
        """Test complete intelligent pipeline flow."""
        # Setup mocks
        mock_integration = Mock()
        mock_integration.prepare_intelligent_request = AsyncMock(
            return_value={"mode": "analysis", "max_tools": 10}
        )
        mock_integration.should_continue_intelligent = Mock(return_value=(True, "Good progress"))
        mock_integration.record_intelligent_outcome = AsyncMock()

        mock_validator = Mock()
        mock_result = MockIntelligentValidationResult(
            quality_score=0.88,
            grounding_score=0.95,
            is_grounded=True,
            is_valid=True,
            grounding_issues=[],
        )
        mock_validator.validate_intelligent_response = AsyncMock(return_value=mock_result)

        adapter = IntelligentPipelineAdapter(
            intelligent_integration=mock_integration,
            validation_coordinator=mock_validator,
        )

        # Prepare request
        request_data = await adapter.prepare_intelligent_request(
            task="Analyze code",
            task_type="analysis",
        )
        assert request_data is not None
        assert request_data["mode"] == "analysis"

        # Validate response
        validation_result = await adapter.validate_intelligent_response(
            response="Code analysis complete",
            query="Analyze code",
            tool_calls=5,
            task_type="analysis",
        )
        assert validation_result is not None
        assert validation_result["quality_score"] == 0.88

        # Check continuation
        should_continue, reason = adapter.should_continue_intelligent()
        assert should_continue is True

        # Record outcome
        await adapter.record_intelligent_outcome(
            success=True, quality_score=0.88, user_satisfied=True, completed=True
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
