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

"""Tests for intelligent prompt builder with learning capabilities."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from victor.agent.intelligent_prompt_builder import (
    CacheState,
    ContextFragment,
    EmbeddingScheduler,
    IntelligentPromptBuilder,
    ProfileLearningStore,
    ProfileMetrics,
    PromptContext,
    PromptStrategy,
)


class TestProfileMetrics:
    """Tests for ProfileMetrics dataclass."""

    def test_creation(self):
        """Test creating profile metrics."""
        metrics = ProfileMetrics(
            profile_name="test-profile",
            provider="ollama",
            model="qwen2.5:32b",
        )

        assert metrics.profile_name == "test-profile"
        assert metrics.total_requests == 0
        assert metrics.avg_quality_score == 0.5

    def test_update_from_interaction(self):
        """Test updating metrics from interaction."""
        metrics = ProfileMetrics(
            profile_name="test",
            provider="ollama",
            model="test",
        )

        metrics.update_from_interaction(
            success=True,
            quality_score=0.8,
            response_time_ms=500.0,
            tool_calls=5,
            tool_budget=10,
            grounded=True,
        )

        assert metrics.total_requests == 1
        assert metrics.successful_completions == 1
        assert metrics.avg_quality_score > 0.5  # Updated toward 0.8
        assert metrics.avg_response_time_ms > 0

    def test_multiple_updates_use_exponential_average(self):
        """Multiple updates should use exponential moving average."""
        metrics = ProfileMetrics(
            profile_name="test",
            provider="ollama",
            model="test",
        )

        # First interaction - low quality
        metrics.update_from_interaction(
            success=True,
            quality_score=0.3,
            response_time_ms=100.0,
            tool_calls=2,
            tool_budget=10,
            grounded=True,
        )

        first_quality = metrics.avg_quality_score

        # Second interaction - high quality
        metrics.update_from_interaction(
            success=True,
            quality_score=0.9,
            response_time_ms=200.0,
            tool_calls=3,
            tool_budget=10,
            grounded=True,
        )

        # Quality should increase but not jump to 0.9 (EMA smoothing)
        assert metrics.avg_quality_score > first_quality
        assert metrics.avg_quality_score < 0.9

    def test_get_recommended_strategy_minimal(self):
        """High accuracy should recommend minimal strategy."""
        metrics = ProfileMetrics(
            profile_name="test",
            provider="ollama",
            model="test",
            total_requests=20,
            grounding_accuracy=0.95,
            tool_call_success_rate=0.95,
        )

        assert metrics.get_recommended_strategy() == PromptStrategy.MINIMAL

    def test_get_recommended_strategy_strict(self):
        """Low accuracy should recommend strict strategy."""
        metrics = ProfileMetrics(
            profile_name="test",
            provider="ollama",
            model="test",
            total_requests=20,
            grounding_accuracy=0.5,
            tool_call_success_rate=0.5,
        )

        assert metrics.get_recommended_strategy() == PromptStrategy.STRICT

    def test_get_recommended_strategy_adaptive_when_insufficient_data(self):
        """Should return adaptive when not enough data."""
        metrics = ProfileMetrics(
            profile_name="test",
            provider="ollama",
            model="test",
            total_requests=5,  # < 10
        )

        assert metrics.get_recommended_strategy() == PromptStrategy.ADAPTIVE


class TestContextFragment:
    """Tests for ContextFragment dataclass."""

    def test_relevance_score_calculation(self):
        """Test relevance score combines factors correctly."""
        fragment = ContextFragment(
            content="Test content",
            similarity=0.8,
            task_type="analysis",
            was_successful=True,
            timestamp=datetime.now(),
            source="conversation",
        )

        # Recent + successful + high similarity should give high relevance
        assert fragment.relevance_score > 0.5

    def test_recency_affects_relevance(self):
        """Older fragments should have lower relevance."""
        recent = ContextFragment(
            content="Recent",
            similarity=0.8,
            task_type="analysis",
            was_successful=True,
            timestamp=datetime.now(),
            source="conversation",
        )

        old = ContextFragment(
            content="Old",
            similarity=0.8,
            task_type="analysis",
            was_successful=True,
            timestamp=datetime.now() - timedelta(days=30),
            source="conversation",
        )

        assert recent.relevance_score > old.relevance_score

    def test_success_affects_relevance(self):
        """Successful interactions should have higher relevance."""
        successful = ContextFragment(
            content="Success",
            similarity=0.8,
            task_type="analysis",
            was_successful=True,
            timestamp=datetime.now(),
            source="conversation",
        )

        failed = ContextFragment(
            content="Failed",
            similarity=0.8,
            task_type="analysis",
            was_successful=False,
            timestamp=datetime.now(),
            source="conversation",
        )

        assert successful.relevance_score > failed.relevance_score


class TestProfileLearningStore:
    """Tests for ProfileLearningStore."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary learning store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_learning.db"
            yield ProfileLearningStore(db_path=db_path)

    def test_save_and_load_metrics(self, temp_store):
        """Test saving and loading profile metrics."""
        metrics = ProfileMetrics(
            profile_name="test-profile",
            provider="ollama",
            model="qwen2.5:32b",
            total_requests=100,
            avg_quality_score=0.75,
        )

        temp_store.save_metrics(metrics)
        loaded = temp_store.load_metrics("test-profile", "ollama", "qwen2.5:32b")

        assert loaded.profile_name == "test-profile"
        assert loaded.total_requests == 100
        assert loaded.avg_quality_score == 0.75

    def test_load_returns_new_metrics_if_not_found(self, temp_store):
        """Loading non-existent profile returns new metrics."""
        loaded = temp_store.load_metrics("nonexistent", "ollama", "test")

        assert loaded.profile_name == "nonexistent"
        assert loaded.total_requests == 0

    def test_record_interaction(self, temp_store):
        """Test recording interaction history."""
        temp_store.record_interaction(
            profile_name="test",
            task_type="analysis",
            success=True,
            quality_score=0.8,
            response_time_ms=500.0,
            tool_calls=5,
            tool_budget=10,
            grounded=True,
        )

        history = temp_store.get_recent_interactions("test", limit=10)
        assert len(history) == 1
        assert history[0]["task_type"] == "analysis"
        assert history[0]["success"] == 1


class TestEmbeddingScheduler:
    """Tests for EmbeddingScheduler."""

    def test_initial_state_is_cold(self):
        """Scheduler should start in cold state."""
        scheduler = EmbeddingScheduler()
        assert scheduler.state == CacheState.COLD

    def test_invalidate_resets_state(self):
        """Invalidate should reset to cold state."""
        scheduler = EmbeddingScheduler()
        scheduler._state = CacheState.WARM
        scheduler.invalidate()

        assert scheduler.state == CacheState.COLD


class TestIntelligentPromptBuilder:
    """Tests for IntelligentPromptBuilder."""

    @pytest.fixture
    def temp_learning_store(self):
        """Create a temporary learning store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_learning.db"
            yield ProfileLearningStore(db_path=db_path)

    def test_creation(self, temp_learning_store):
        """Test creating a prompt builder."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test-profile",
            learning_store=temp_learning_store,
        )

        assert builder.provider_name == "ollama"
        assert builder.model == "qwen2.5:32b"
        assert builder.profile_name == "test-profile"

    @pytest.mark.asyncio
    async def test_build_prompt_cloud_provider(self, temp_learning_store):
        """Cloud providers should get minimal prompts."""
        builder = IntelligentPromptBuilder(
            provider_name="anthropic",
            model="claude-3-opus",
            profile_name="claude-test",
            learning_store=temp_learning_store,
        )

        prompt = await builder.build(
            task="Analyze the authentication module",
            task_type="analysis",
        )

        assert "expert code analyst" in prompt.lower()
        assert len(prompt) < 2000  # Minimal prompt should be concise

    @pytest.mark.asyncio
    async def test_build_prompt_local_provider(self, temp_learning_store):
        """Local providers should get more structured prompts."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="llama3.1:8b",
            profile_name="local-llama",
            learning_store=temp_learning_store,
        )

        prompt = await builder.build(
            task="Analyze the authentication module",
            task_type="analysis",
        )

        # Local models should get more guidance
        assert "TOOL" in prompt or "tool" in prompt.lower()

    @pytest.mark.asyncio
    async def test_build_prompt_includes_task_hint(self, temp_learning_store):
        """Prompt should include task-specific hints."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
            learning_store=temp_learning_store,
        )

        prompt = await builder.build(
            task="Create a new file",
            task_type="create_simple",
        )

        assert "[CREATE]" in prompt or "create" in prompt.lower()

    @pytest.mark.asyncio
    async def test_build_prompt_includes_mode_hint(self, temp_learning_store):
        """Prompt should include mode context."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
            learning_store=temp_learning_store,
        )

        prompt = await builder.build(
            task="Build feature",
            task_type="create",
            current_mode="build",
        )

        assert "MODE" in prompt or "Build" in prompt

    def test_record_feedback_updates_metrics(self, temp_learning_store):
        """Recording feedback should update profile metrics."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
            learning_store=temp_learning_store,
        )

        initial_requests = builder._metrics.total_requests

        builder.record_feedback(
            task_type="analysis",
            success=True,
            quality_score=0.85,
            response_time_ms=1000.0,
            tool_calls=5,
            tool_budget=10,
            grounded=True,
        )

        assert builder._metrics.total_requests == initial_requests + 1

    def test_get_profile_stats(self, temp_learning_store):
        """get_profile_stats should return comprehensive stats."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
            learning_store=temp_learning_store,
        )

        stats = builder.get_profile_stats()

        assert "profile_name" in stats
        assert "provider" in stats
        assert "model" in stats
        assert "success_rate" in stats
        assert "cache_state" in stats

    def test_reset_learning(self, temp_learning_store):
        """reset_learning should clear learned metrics."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
            learning_store=temp_learning_store,
        )

        # Record some data
        builder.record_feedback(
            task_type="analysis",
            success=True,
            quality_score=0.9,
            response_time_ms=500.0,
            tool_calls=3,
            tool_budget=10,
            grounded=True,
        )

        # Reset
        builder.reset_learning()

        assert builder._metrics.total_requests == 0

    def test_observer_notification(self, temp_learning_store):
        """Observers should be notified on feedback."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
            learning_store=temp_learning_store,
        )

        notifications = []

        def observer(task_type, quality, success):
            notifications.append((task_type, quality, success))

        builder.add_observer(observer)

        builder.record_feedback(
            task_type="analysis",
            success=True,
            quality_score=0.8,
            response_time_ms=500.0,
            tool_calls=3,
            tool_budget=10,
            grounded=True,
        )

        assert len(notifications) == 1
        assert notifications[0][0] == "analysis"
        assert notifications[0][2] is True


class TestPromptStrategySelection:
    """Tests for prompt strategy selection logic."""

    @pytest.fixture
    def temp_learning_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_learning.db"
            yield ProfileLearningStore(db_path=db_path)

    def test_cloud_provider_gets_minimal_strategy(self, temp_learning_store):
        """Cloud providers should always use minimal strategy."""
        builder = IntelligentPromptBuilder(
            provider_name="anthropic",
            model="claude-3",
            profile_name="test",
            learning_store=temp_learning_store,
        )

        context = PromptContext(
            task="test",
            task_type="analysis",
            profile_name="test",
            provider="anthropic",
            model="claude-3",
        )

        strategy = builder._determine_strategy(context)
        assert strategy == PromptStrategy.MINIMAL

    def test_native_tool_model_gets_structured_strategy(self, temp_learning_store):
        """Models with native tool support get structured prompts."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
            learning_store=temp_learning_store,
        )

        context = PromptContext(
            task="test",
            task_type="analysis",
            profile_name="test",
            provider="ollama",
            model="qwen2.5:32b",
        )

        strategy = builder._determine_strategy(context)
        assert strategy == PromptStrategy.STRUCTURED

    def test_non_native_tool_model_gets_strict_strategy(self, temp_learning_store):
        """Models without native tool support get strict prompts."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="codellama:7b",
            profile_name="test",
            learning_store=temp_learning_store,
        )

        context = PromptContext(
            task="test",
            task_type="analysis",
            profile_name="test",
            provider="ollama",
            model="codellama:7b",
        )

        strategy = builder._determine_strategy(context)
        assert strategy == PromptStrategy.STRICT


class TestPromptGeneration:
    """Tests for prompt generation."""

    @pytest.fixture
    def temp_learning_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_learning.db"
            yield ProfileLearningStore(db_path=db_path)

    @pytest.mark.asyncio
    async def test_prompt_includes_grounding_rules(self, temp_learning_store):
        """Generated prompt should include grounding rules."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
            learning_store=temp_learning_store,
        )

        prompt = await builder.build(
            task="Analyze code",
            task_type="analysis",
        )

        assert "GROUNDING" in prompt or "grounding" in prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_with_continuation_context(self, temp_learning_store):
        """Prompt should include continuation context when provided."""
        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:32b",
            profile_name="test",
            learning_store=temp_learning_store,
        )

        prompt = await builder.build(
            task="Continue analysis",
            task_type="analysis",
            continuation_context="Previous: Found 3 modules needing review",
        )

        assert "Previous:" in prompt or "CONTINUATION" in prompt
