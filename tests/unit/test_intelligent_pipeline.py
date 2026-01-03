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

"""Integration tests for IntelligentAgentPipeline."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.intelligent_pipeline import (
    IntelligentAgentPipeline,
    PipelineStats,
    RequestContext,
    ResponseResult,
    get_pipeline,
    clear_pipeline_cache,
)


class TestRequestContext:
    """Tests for RequestContext dataclass."""

    def test_creation(self):
        """Test creating a RequestContext."""
        context = RequestContext(
            system_prompt="You are a helpful assistant.",
            recommended_tool_budget=10,
            recommended_mode="explore",
            should_continue=True,
            mode_confidence=0.8,
        )

        assert context.system_prompt == "You are a helpful assistant."
        assert context.recommended_tool_budget == 10
        assert context.recommended_mode == "explore"
        assert context.should_continue is True
        assert context.mode_confidence == 0.8
        assert context.continuation_context is None
        assert context.profile_stats == {}

    def test_with_continuation_context(self):
        """Test RequestContext with continuation context."""
        context = RequestContext(
            system_prompt="test",
            recommended_tool_budget=5,
            recommended_mode="build",
            should_continue=True,
            continuation_context="Previous: Found 3 files",
        )

        assert context.continuation_context == "Previous: Found 3 files"


class TestResponseResult:
    """Tests for ResponseResult dataclass."""

    def test_creation(self):
        """Test creating a ResponseResult."""
        result = ResponseResult(
            is_valid=True,
            quality_score=0.85,
            grounding_score=0.9,
            is_grounded=True,
        )

        assert result.is_valid is True
        assert result.quality_score == 0.85
        assert result.grounding_score == 0.9
        assert result.is_grounded is True
        assert result.quality_details == {}
        assert result.grounding_issues == []
        assert result.improvement_suggestions == []
        assert result.learning_reward == 0.0

    def test_with_issues(self):
        """Test ResponseResult with grounding issues."""
        result = ResponseResult(
            is_valid=False,
            quality_score=0.4,
            grounding_score=0.3,
            is_grounded=False,
            grounding_issues=["Mentioned non-existent file", "Incorrect line number"],
            improvement_suggestions=["Verify file paths before mentioning"],
        )

        assert result.is_valid is False
        assert len(result.grounding_issues) == 2
        assert len(result.improvement_suggestions) == 1


class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = PipelineStats()

        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.avg_quality_score == 0.0
        assert stats.avg_grounding_score == 0.0
        assert stats.total_learning_reward == 0.0
        assert stats.mode_transitions == 0
        assert stats.circuit_breaker_trips == 0
        assert stats.retry_attempts == 0
        assert stats.cache_state == "cold"
        assert stats.profile_name == ""


class TestIntelligentAgentPipelineCreation:
    """Tests for IntelligentAgentPipeline creation and initialization."""

    def test_sync_creation(self):
        """Test synchronous creation of pipeline."""
        pipeline = IntelligentAgentPipeline(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test-profile",
            project_root="/tmp/test",
        )

        assert pipeline.provider_name == "ollama"
        assert pipeline.model == "qwen2.5:32b"
        assert pipeline.profile_name == "test-profile"
        assert pipeline.project_root == "/tmp/test"

    @pytest.mark.asyncio
    async def test_async_create_factory(self):
        """Test async factory method creates and initializes pipeline."""
        # Create pipeline without mocking - it handles component failures gracefully
        pipeline = await IntelligentAgentPipeline.create(
            provider_name="anthropic",
            model="claude-3",
            profile_name="claude-test",
        )

        assert pipeline.provider_name == "anthropic"
        assert pipeline.model == "claude-3"
        assert pipeline.profile_name == "claude-test"

    @pytest.mark.asyncio
    async def test_create_generates_profile_name_if_not_provided(self):
        """Should generate profile name from provider:model if not provided."""
        pipeline = await IntelligentAgentPipeline.create(
            provider_name="ollama",
            model="llama3.1:8b",
        )

        assert pipeline.profile_name == "ollama:llama3.1:8b"


class TestIntelligentAgentPipelinePrepareRequest:
    """Tests for prepare_request method."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        return IntelligentAgentPipeline(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
        )

    @pytest.mark.asyncio
    async def test_prepare_request_returns_context(self, pipeline):
        """prepare_request should return RequestContext."""
        context = await pipeline.prepare_request(
            task="Analyze the auth module",
            task_type="analysis",
        )

        assert isinstance(context, RequestContext)
        # should_continue depends on RL mode controller state - can be True or False
        # based on learned profile data, so we just check it's a bool
        assert isinstance(context.should_continue, bool)
        # Mode is determined by RL-based AdaptiveModeController
        # Valid modes: explore, build, review, finalize, complete
        assert context.recommended_mode in ("explore", "build", "review", "finalize", "complete")

    @pytest.mark.asyncio
    async def test_prepare_request_updates_stats(self, pipeline):
        """prepare_request should update pipeline stats."""
        initial_requests = pipeline._stats.total_requests

        await pipeline.prepare_request(
            task="Test task",
            task_type="general",
        )

        assert pipeline._stats.total_requests == initial_requests + 1

    @pytest.mark.asyncio
    async def test_prepare_request_with_continuation(self, pipeline):
        """prepare_request should handle continuation context."""
        context = await pipeline.prepare_request(
            task="Continue the analysis",
            task_type="analysis",
            continuation_context="Previously identified 5 modules",
        )

        assert context.continuation_context == "Previously identified 5 modules"

    @pytest.mark.asyncio
    async def test_prepare_request_uses_prompt_builder(self, pipeline):
        """Should use prompt builder when available."""
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value="Test system prompt")
        mock_builder.get_profile_stats = MagicMock(return_value={"test": "stats"})
        pipeline._prompt_builder = mock_builder

        context = await pipeline.prepare_request(
            task="Build a feature",
            task_type="create",
        )

        assert context.system_prompt == "Test system prompt"
        assert context.profile_stats == {"test": "stats"}
        mock_builder.build.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_request_uses_mode_controller(self, pipeline):
        """Should use mode controller when available."""
        mock_controller = MagicMock()
        mock_action = MagicMock()
        mock_action.target_mode = MagicMock()
        mock_action.target_mode.value = "build"
        mock_action.confidence = 0.9
        mock_action.should_continue = True
        mock_action.adjust_tool_budget = 5
        mock_controller.get_recommended_action = MagicMock(return_value=mock_action)
        mock_controller.get_optimal_tool_budget = MagicMock(return_value=15)
        pipeline._mode_controller = mock_controller

        context = await pipeline.prepare_request(
            task="Implement feature",
            task_type="create",
            tool_budget=10,
        )

        assert context.recommended_mode == "build"
        assert context.mode_confidence == 0.9
        assert context.should_continue is True


class TestIntelligentAgentPipelineProcessResponse:
    """Tests for process_response method."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        return IntelligentAgentPipeline(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
        )

    @pytest.mark.asyncio
    async def test_process_response_returns_result(self, pipeline):
        """process_response should return ResponseResult."""
        result = await pipeline.process_response(
            response="The auth module uses JWT tokens.",
            success=True,
        )

        assert isinstance(result, ResponseResult)

    @pytest.mark.asyncio
    async def test_process_response_updates_stats(self, pipeline):
        """process_response should update success stats."""
        await pipeline.process_response(
            response="Success response",
            success=True,
        )

        assert pipeline._stats.successful_requests == 1

    @pytest.mark.asyncio
    async def test_process_response_uses_quality_scorer(self, pipeline):
        """Should use quality scorer when available."""
        mock_scorer = MagicMock()
        mock_quality_result = MagicMock()
        mock_quality_result.overall_score = 0.85
        mock_quality_result.dimension_scores = []
        mock_quality_result.improvement_suggestions = ["Add more detail"]
        mock_scorer.score = AsyncMock(return_value=mock_quality_result)
        pipeline._quality_scorer = mock_scorer

        result = await pipeline.process_response(
            response="Test response",
            query="Test query",
            success=True,
        )

        assert result.quality_score == 0.85
        mock_scorer.score.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_response_uses_grounding_verifier(self, pipeline):
        """Should use grounding verifier when available."""
        mock_verifier = MagicMock()
        mock_grounding_result = MagicMock()
        mock_grounding_result.confidence = 0.75
        mock_grounding_result.is_grounded = True
        mock_grounding_result.issues = []
        mock_verifier.verify = AsyncMock(return_value=mock_grounding_result)
        pipeline._grounding_verifier = mock_verifier

        result = await pipeline.process_response(
            response="Test response",
            success=True,
        )

        assert result.grounding_score == 0.75
        assert result.is_grounded is True

    @pytest.mark.asyncio
    async def test_process_response_records_feedback(self, pipeline):
        """Should record feedback to prompt builder."""
        mock_builder = MagicMock()
        pipeline._prompt_builder = mock_builder

        await pipeline.process_response(
            response="Test",
            task_type="analysis",
            success=True,
        )

        mock_builder.record_feedback.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_response_records_mode_outcome(self, pipeline):
        """Should record outcome to mode controller."""
        mock_controller = MagicMock()
        mock_controller.record_outcome = MagicMock(return_value=0.5)
        pipeline._mode_controller = mock_controller

        result = await pipeline.process_response(
            response="Test",
            success=True,
        )

        assert result.learning_reward == 0.5
        mock_controller.record_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_response_notifies_observers(self, pipeline):
        """Should notify observers after processing."""
        notifications = []

        def observer(result: ResponseResult):
            notifications.append(result)

        pipeline.add_observer(observer)

        await pipeline.process_response(
            response="Test",
            success=True,
        )

        assert len(notifications) == 1
        assert isinstance(notifications[0], ResponseResult)


class TestIntelligentAgentPipelineShouldContinue:
    """Tests for should_continue method."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        return IntelligentAgentPipeline(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
        )

    def test_should_continue_delegates_to_mode_controller(self, pipeline):
        """Should delegate to mode controller when available."""
        mock_controller = MagicMock()
        mock_controller.should_continue = MagicMock(return_value=(False, "Budget exhausted"))
        pipeline._mode_controller = mock_controller

        should_continue, reason = pipeline.should_continue(
            tool_calls_made=10,
            tool_budget=10,
            quality_score=0.5,
            iteration_count=5,
            iteration_budget=20,
        )

        assert should_continue is False
        assert reason == "Budget exhausted"

    def test_should_continue_fallback_tool_budget(self, pipeline):
        """Should use fallback logic when tool budget exhausted."""
        should_continue, reason = pipeline.should_continue(
            tool_calls_made=10,
            tool_budget=10,
            quality_score=0.5,
            iteration_count=5,
            iteration_budget=20,
        )

        assert should_continue is False
        assert "budget" in reason.lower()

    def test_should_continue_fallback_iteration_budget(self, pipeline):
        """Should use fallback logic when iteration budget exhausted."""
        should_continue, reason = pipeline.should_continue(
            tool_calls_made=5,
            tool_budget=10,
            quality_score=0.5,
            iteration_count=20,
            iteration_budget=20,
        )

        assert should_continue is False
        assert "iteration" in reason.lower()

    def test_should_continue_fallback_high_quality(self, pipeline):
        """Should stop when quality is high."""
        should_continue, reason = pipeline.should_continue(
            tool_calls_made=3,
            tool_budget=10,
            quality_score=0.9,
            iteration_count=5,
            iteration_budget=20,
        )

        assert should_continue is False
        assert "quality" in reason.lower()

    def test_should_continue_true_when_ok(self, pipeline):
        """Should continue when budgets and quality allow."""
        should_continue, reason = pipeline.should_continue(
            tool_calls_made=3,
            tool_budget=10,
            quality_score=0.5,
            iteration_count=5,
            iteration_budget=20,
        )

        assert should_continue is True


class TestIntelligentAgentPipelineStats:
    """Tests for pipeline statistics."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        return IntelligentAgentPipeline(
            provider_name="test",
            model="test-model",
            profile_name="test-profile",
        )

    def test_get_stats_returns_pipeline_stats(self, pipeline):
        """get_stats should return PipelineStats."""
        stats = pipeline.get_stats()
        assert isinstance(stats, PipelineStats)
        assert stats.profile_name == "test-profile"

    def test_reset_session_clears_stats(self, pipeline):
        """reset_session should clear stats."""
        pipeline._stats.total_requests = 10
        pipeline._stats.successful_requests = 8

        pipeline.reset_session()

        assert pipeline._stats.total_requests == 0
        assert pipeline._stats.successful_requests == 0

    def test_get_learning_summary(self, pipeline):
        """get_learning_summary should return comprehensive info."""
        pipeline._stats.total_requests = 100
        pipeline._stats.successful_requests = 80

        summary = pipeline.get_learning_summary()

        assert summary["profile_name"] == "test-profile"
        assert summary["total_requests"] == 100
        assert summary["success_rate"] == 0.8


class TestIntelligentAgentPipelineObservers:
    """Tests for observer pattern."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        return IntelligentAgentPipeline(
            provider_name="test",
            model="test",
            profile_name="test",
        )

    def test_add_observer(self, pipeline):
        """Should add observer."""
        observer = MagicMock()
        pipeline.add_observer(observer)

        assert observer in pipeline._observers

    def test_remove_observer(self, pipeline):
        """Should remove observer."""
        observer = MagicMock()
        pipeline.add_observer(observer)
        pipeline.remove_observer(observer)

        assert observer not in pipeline._observers

    def test_remove_nonexistent_observer(self, pipeline):
        """Should handle removing nonexistent observer."""
        observer = MagicMock()
        pipeline.remove_observer(observer)  # Should not raise


class TestPipelineCache:
    """Tests for module-level pipeline cache."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear cache before each test."""
        clear_pipeline_cache()
        yield
        clear_pipeline_cache()

    @pytest.mark.asyncio
    async def test_get_pipeline_creates_new(self):
        """get_pipeline should create new pipeline on first call."""
        pipeline = await get_pipeline(
            provider_name="ollama",
            model="test-model-cache1",
        )

        assert pipeline.provider_name == "ollama"

    @pytest.mark.asyncio
    async def test_get_pipeline_returns_cached(self):
        """get_pipeline should return cached pipeline on subsequent calls."""
        pipeline1 = await get_pipeline(
            provider_name="ollama",
            model="test-model-cache2",
        )
        pipeline2 = await get_pipeline(
            provider_name="ollama",
            model="test-model-cache2",
        )

        assert pipeline1 is pipeline2

    @pytest.mark.asyncio
    async def test_get_pipeline_different_models(self):
        """get_pipeline should create different pipelines for different models."""
        pipeline1 = await get_pipeline(provider_name="ollama", model="model-a")
        pipeline2 = await get_pipeline(provider_name="ollama", model="model-b")

        assert pipeline1 is not pipeline2

    def test_clear_pipeline_cache(self):
        """clear_pipeline_cache should empty the cache."""
        # Import and directly manipulate the cache
        from victor.agent.intelligent_pipeline import _pipeline_cache

        _pipeline_cache["test"] = MagicMock()

        clear_pipeline_cache()

        assert len(_pipeline_cache) == 0


class TestIntelligentAgentPipelineResilience:
    """Tests for resilient execution."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        return IntelligentAgentPipeline(
            provider_name="ollama",
            model="test",
            profile_name="test",
        )

    @pytest.mark.asyncio
    async def test_execute_without_resilience(self, pipeline):
        """Should fall back to direct call without resilient executor."""
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)

        result = await pipeline.execute_with_resilience(
            provider=mock_provider,
            messages=[{"role": "user", "content": "test"}],
        )

        assert result is mock_response
        mock_provider.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_resilience(self, pipeline):
        """Should use resilient executor when available."""
        mock_executor = MagicMock()
        mock_response = MagicMock()
        mock_executor.execute = AsyncMock(return_value=mock_response)
        pipeline._resilient_executor = mock_executor

        mock_provider = MagicMock()

        result = await pipeline.execute_with_resilience(
            provider=mock_provider,
            messages=[{"role": "user", "content": "test"}],
        )

        assert result is mock_response
        mock_executor.execute.assert_called_once()


class TestIntelligentAgentPipelineIntegration:
    """Integration tests combining multiple pipeline features."""

    @pytest.fixture
    def pipeline_with_mocks(self):
        """Create a pipeline with all components mocked."""
        pipeline = IntelligentAgentPipeline(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="integration-test",
        )

        # Mock prompt builder
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value="System prompt")
        mock_builder.get_profile_stats = MagicMock(return_value={"requests": 10})
        mock_builder.record_feedback = MagicMock()
        pipeline._prompt_builder = mock_builder

        # Mock mode controller
        mock_controller = MagicMock()
        mock_action = MagicMock()
        mock_action.target_mode = MagicMock()
        mock_action.target_mode.value = "build"
        mock_action.confidence = 0.85
        mock_action.should_continue = True
        mock_action.adjust_tool_budget = 0
        mock_controller.get_recommended_action = MagicMock(return_value=mock_action)
        mock_controller.get_optimal_tool_budget = MagicMock(return_value=10)
        mock_controller.record_outcome = MagicMock(return_value=0.7)
        mock_controller.should_continue = MagicMock(return_value=(True, "Continue"))
        pipeline._mode_controller = mock_controller

        # Mock quality scorer
        mock_scorer = MagicMock()
        mock_quality_result = MagicMock()
        mock_quality_result.overall_score = 0.9
        mock_quality_result.dimension_scores = []
        mock_quality_result.improvement_suggestions = []
        mock_scorer.score = AsyncMock(return_value=mock_quality_result)
        pipeline._quality_scorer = mock_scorer

        return pipeline

    @pytest.mark.asyncio
    async def test_full_request_response_cycle(self, pipeline_with_mocks):
        """Test a complete request/response cycle."""
        # Prepare request
        context = await pipeline_with_mocks.prepare_request(
            task="Implement a new feature",
            task_type="create",
        )

        assert context.recommended_mode == "build"
        assert context.mode_confidence == 0.85
        assert context.system_prompt == "System prompt"

        # Process response
        result = await pipeline_with_mocks.process_response(
            response="Feature implemented successfully",
            query="Implement a new feature",
            success=True,
            task_type="create",
        )

        assert result.quality_score == 0.9
        assert result.learning_reward == 0.7

        # Check stats
        stats = pipeline_with_mocks.get_stats()
        assert stats.total_requests == 1
        assert stats.successful_requests == 1

    @pytest.mark.asyncio
    async def test_multiple_iterations(self, pipeline_with_mocks):
        """Test multiple request/response iterations."""
        for i in range(5):
            await pipeline_with_mocks.prepare_request(
                task=f"Task {i}",
                task_type="general",
            )
            await pipeline_with_mocks.process_response(
                response=f"Response {i}",
                success=True,
            )

        stats = pipeline_with_mocks.get_stats()
        assert stats.total_requests == 5
        assert stats.successful_requests == 5

    @pytest.mark.asyncio
    async def test_observer_receives_all_results(self, pipeline_with_mocks):
        """Observer should receive all response results."""
        results = []
        pipeline_with_mocks.add_observer(lambda r: results.append(r))

        for i in range(3):
            await pipeline_with_mocks.process_response(
                response=f"Response {i}",
                success=True,
            )

        assert len(results) == 3
