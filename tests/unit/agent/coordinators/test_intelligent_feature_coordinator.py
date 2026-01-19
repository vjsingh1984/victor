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

"""Tests for IntelligentFeatureCoordinator."""

import pytest

from victor.agent.coordinators.intelligent_feature_coordinator import (
    IntelligentFeatureCoordinator,
    create_intelligent_feature_coordinator,
)


class MockSettings:
    """Mock settings for testing."""

    def __init__(self, intelligent=False, qlearning=False):
        self.enable_intelligent_features = intelligent
        self.enable_qlearning = qlearning
        self.unified_embedding_model = "BAAI/bge-small-en-v1.5"


class MockQLearningCoordinator:
    """Mock Q-learning coordinator."""

    async def prepare_request(self, task: str, task_type: str):
        """Prepare intelligent request."""
        return {"task": task, "task_type": task_type, "optimized": True}


class MockEvaluationCoordinator:
    """Mock evaluation coordinator."""

    async def validate_response(self, response, expected_outcomes=None):
        """Validate response."""
        return {"validated": True, "quality_score": 0.9}

    async def record_outcome(self, task, task_type, outcome, metadata=None):
        """Record outcome."""
        return True


@pytest.fixture
def settings():
    """Fixture for settings."""
    return MockSettings(intelligent=True, qlearning=True)


@pytest.fixture
def qlearning_coordinator():
    """Fixture for Q-learning coordinator."""
    return MockQLearningCoordinator()


@pytest.fixture
def evaluation_coordinator():
    """Fixture for evaluation coordinator."""
    return MockEvaluationCoordinator()


@pytest.fixture
def coordinator(settings, qlearning_coordinator, evaluation_coordinator):
    """Fixture for IntelligentFeatureCoordinator."""
    return IntelligentFeatureCoordinator(
        settings=settings,
        qlearning_coordinator=qlearning_coordinator,
        evaluation_coordinator=evaluation_coordinator,
    )


class TestIntelligentFeatureCoordinator:
    """Test suite for IntelligentFeatureCoordinator."""

    def test_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator.is_intelligent_enabled() is True
        assert coordinator.is_qlearning_enabled() is True
        assert coordinator.are_embeddings_preloaded() is False

    def test_initialization_disabled(self):
        """Test initialization with features disabled."""
        settings = MockSettings(intelligent=False, qlearning=False)
        coordinator = IntelligentFeatureCoordinator(settings=settings)

        assert coordinator.is_intelligent_enabled() is False
        assert coordinator.is_qlearning_enabled() is False

    # ========================================================================
    # Intelligent Request Preparation
    # ========================================================================

    @pytest.mark.asyncio
    async def test_prepare_intelligent_request(self, coordinator):
        """Test preparing intelligent request."""
        request = await coordinator.prepare_intelligent_request("task description", "coding")

        assert request is not None
        assert request["task"] == "task description"
        assert request["task_type"] == "coding"
        assert request["optimized"] is True

    @pytest.mark.asyncio
    async def test_prepare_intelligent_request_disabled(self):
        """Test preparing request when disabled."""
        settings = MockSettings(intelligent=False)
        coordinator = IntelligentFeatureCoordinator(settings=settings)

        request = await coordinator.prepare_intelligent_request("task", "coding")

        assert request is None

    @pytest.mark.asyncio
    async def test_prepare_intelligent_request_no_qlearning(self):
        """Test preparing request without Q-learning coordinator."""
        settings = MockSettings(intelligent=True)
        coordinator = IntelligentFeatureCoordinator(settings=settings, qlearning_coordinator=None)

        request = await coordinator.prepare_intelligent_request("task", "coding")

        assert request is None

    @pytest.mark.asyncio
    async def test_prepare_intelligent_request_exception(self, coordinator, mocker):
        """Test preparing request with exception."""
        # Mock coordinator to raise exception
        mocker.patch.object(
            coordinator._qlearning_coordinator,
            "prepare_request",
            side_effect=Exception("Test error"),
        )

        request = await coordinator.prepare_intelligent_request("task", "coding")

        assert request is None  # Should return None on exception

    # ========================================================================
    # Intelligent Response Validation
    # ========================================================================

    @pytest.mark.asyncio
    async def test_validate_intelligent_response(self, coordinator):
        """Test validating intelligent response."""
        response = {"content": "test response"}
        validation = await coordinator.validate_intelligent_response(response)

        assert validation["validated"] is True
        assert validation["quality_score"] == 0.9

    @pytest.mark.asyncio
    async def test_validate_with_expected_outcomes(self, coordinator):
        """Test validation with expected outcomes."""
        response = {"content": "test response"}
        expected = ["outcome1", "outcome2"]

        validation = await coordinator.validate_intelligent_response(
            response, expected_outcomes=expected
        )

        assert validation["validated"] is True

    @pytest.mark.asyncio
    async def test_validate_no_evaluation_coordinator(self):
        """Test validation without evaluation coordinator."""
        coordinator = IntelligentFeatureCoordinator(settings=MockSettings())

        validation = await coordinator.validate_intelligent_response({})

        assert validation["validated"] is False
        assert "not available" in validation["reason"]

    @pytest.mark.asyncio
    async def test_validate_exception(self, coordinator, mocker):
        """Test validation with exception."""
        mocker.patch.object(
            coordinator._evaluation_coordinator,
            "validate_response",
            side_effect=Exception("Test error"),
        )

        validation = await coordinator.validate_intelligent_response({})

        assert validation["validated"] is False

    # ========================================================================
    # Intelligent Outcome Recording
    # ========================================================================

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome(self, coordinator):
        """Test recording intelligent outcome."""
        result = await coordinator.record_intelligent_outcome(
            task="test task",
            task_type="coding",
            outcome="success",
            metadata={"test": "data"},
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_record_outcome_no_metadata(self, coordinator):
        """Test recording outcome without metadata."""
        result = await coordinator.record_intelligent_outcome(
            task="test task",
            task_type="coding",
            outcome="success",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_record_outcome_no_evaluation_coordinator(self):
        """Test recording outcome without evaluation coordinator."""
        coordinator = IntelligentFeatureCoordinator(settings=MockSettings())

        result = await coordinator.record_intelligent_outcome("task", "coding", "success")

        assert result is False

    @pytest.mark.asyncio
    async def test_record_outcome_exception(self, coordinator, mocker):
        """Test recording outcome with exception."""
        mocker.patch.object(
            coordinator._evaluation_coordinator,
            "record_outcome",
            side_effect=Exception("Test error"),
        )

        result = await coordinator.record_intelligent_outcome("task", "coding", "success")

        assert result is False

    # ========================================================================
    # Embedding Preloading
    # ========================================================================

    @pytest.mark.asyncio
    async def test_preload_embeddings_already_loaded(self, coordinator):
        """Test preloading embeddings when already loaded."""
        coordinator._embeddings_preloaded = True

        result = await coordinator.preload_embeddings()

        assert result is True

    @pytest.mark.asyncio
    async def test_preload_embeddings_no_project_path(self, coordinator):
        """Test preloading embeddings without project path."""
        result = await coordinator.preload_embeddings(project_path=None)

        # Should succeed but may not preload any files
        assert result is True

    @pytest.mark.skip(reason="EmbeddingService is imported locally, difficult to mock")
    @pytest.mark.asyncio
    async def test_preload_embeddings_exception(self, coordinator, mocker):
        """Test preloading embeddings with exception."""
        # Skipped - EmbeddingService is imported inside the method
        # making it difficult to mock reliably in tests

    # ========================================================================
    # State Management
    # ========================================================================

    def test_get_state(self, coordinator):
        """Test getting coordinator state."""
        state = coordinator.get_state()

        assert state["intelligent_enabled"] is True
        assert state["qlearning_enabled"] is True
        assert state["embeddings_preloaded"] is False
        assert state["has_qlearning_coordinator"] is True
        assert state["has_evaluation_coordinator"] is True

    def test_get_state_disabled(self):
        """Test getting state with features disabled."""
        settings = MockSettings(intelligent=False, qlearning=False)
        coordinator = IntelligentFeatureCoordinator(settings=settings)

        state = coordinator.get_state()

        assert state["intelligent_enabled"] is False
        assert state["qlearning_enabled"] is False

    def test_reset(self, coordinator):
        """Test resetting coordinator."""
        coordinator._embeddings_preloaded = True
        coordinator.reset()

        assert coordinator.are_embeddings_preloaded() is False

    # ========================================================================
    # Computed Properties
    # ========================================================================

    def test_is_intelligent_enabled(self, coordinator):
        """Test checking if intelligent features enabled."""
        assert coordinator.is_intelligent_enabled() is True

    def test_is_intelligent_enabled_false(self):
        """Test checking when intelligent features disabled."""
        settings = MockSettings(intelligent=False)
        coordinator = IntelligentFeatureCoordinator(settings=settings)

        assert coordinator.is_intelligent_enabled() is False

    def test_is_qlearning_enabled(self, coordinator):
        """Test checking if Q-learning enabled."""
        assert coordinator.is_qlearning_enabled() is True

    def test_is_qlearning_enabled_false(self):
        """Test checking when Q-learning disabled."""
        settings = MockSettings(qlearning=False)
        coordinator = IntelligentFeatureCoordinator(settings=settings)

        assert coordinator.is_qlearning_enabled() is False

    def test_are_embeddings_preloaded(self, coordinator):
        """Test checking if embeddings preloaded."""
        assert coordinator.are_embeddings_preloaded() is False

        coordinator._embeddings_preloaded = True
        assert coordinator.are_embeddings_preloaded() is True


class TestCreateIntelligentFeatureCoordinator:
    """Test suite for create_intelligent_feature_coordinator factory."""

    def test_factory_function(self):
        """Test factory function creates coordinator."""
        settings = MockSettings()
        coordinator = create_intelligent_feature_coordinator(settings=settings)

        assert isinstance(coordinator, IntelligentFeatureCoordinator)

    def test_factory_function_with_coordinators(self):
        """Test factory function with all coordinators."""
        settings = MockSettings()
        qlearning = MockQLearningCoordinator()
        evaluation = MockEvaluationCoordinator()

        coordinator = create_intelligent_feature_coordinator(
            settings=settings,
            qlearning_coordinator=qlearning,
            evaluation_coordinator=evaluation,
        )

        assert isinstance(coordinator, IntelligentFeatureCoordinator)
        assert coordinator._qlearning_coordinator is qlearning
        assert coordinator._evaluation_coordinator is evaluation
