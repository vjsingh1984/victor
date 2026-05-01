"""Integration tests for Priority 4 Phase 2: Extended RL Learners.

Tests for:
- ExtendedModelSelectorLearner with HybridDecisionService
- ExtendedModeTransitionLearner with PhaseDetector
- ExtendedToolSelectorLearner with ToolPredictor and UsageAnalytics
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import sqlite3
from pathlib import Path
import tempfile

import pytest

from victor.agent.usage_analytics import UsageAnalytics
from victor.agent.context_phase_detector import PhaseDetector
from victor.agent.planning.tool_predictor import ToolPredictor
from victor.agent.conversation.state_machine import ConversationStage
from victor.core.shared_types import TaskPhase
from victor.framework.rl.base import RLOutcome, RLRecommendation


class TestExtendedModelSelectorLearner:
    """Test ExtendedModelSelectorLearner integration."""

    def setup_method(self):
        """Create test database connection."""
        self.db_conn = sqlite3.connect(":memory:")

    def teardown_method(self):
        """Close database connection."""
        if hasattr(self, "db_conn"):
            self.db_conn.close()

    def test_learner_initialization(self):
        """Test learner can be initialized with hybrid decision service."""
        from victor.framework.rl.learners.model_selector_extended import (
            ExtendedModelSelectorLearner,
        )

        learner = ExtendedModelSelectorLearner(name="model_selector", db_connection=self.db_conn)

        assert learner is not None
        assert hasattr(learner, "decision_service")
        assert hasattr(learner, "learn")
        assert hasattr(learner, "select_model")

    def test_learn_from_outcomes(self):
        """Test learning from model selection outcomes."""
        from victor.framework.rl.learners.model_selector_extended import (
            ExtendedModelSelectorLearner,
        )

        learner = ExtendedModelSelectorLearner(name="model_selector", db_connection=self.db_conn)

        # Create test outcomes
        outcomes = [
            RLOutcome(
                provider="system",
                model="hybrid",
                task_type="model_selection",
                success=True,
                quality_score=0.9,
                metadata={
                    "used_llm": False,  # Fast path
                    "decision_latency_ms": 50,  # Fast
                    "confidence": 0.95,
                },
            ),
            RLOutcome(
                provider="system",
                model="hybrid",
                task_type="model_selection",
                success=True,
                quality_score=0.85,
                metadata={
                    "used_llm": True,  # LLM fallback
                    "decision_latency_ms": 500,
                    "confidence": 0.6,
                },
            ),
        ]

        # Learn from outcomes
        recommendations = learner.learn(outcomes)

        # Should generate recommendations
        assert len(recommendations) > 0

        # Check recommendation types
        rec_types = {r.recommendation_type for r in recommendations}
        assert "decision_threshold" in rec_types

    def test_select_model_with_context(self):
        """Test model selection with context."""
        from victor.framework.rl.learners.model_selector_extended import (
            ExtendedModelSelectorLearner,
        )

        learner = ExtendedModelSelectorLearner(name="model_selector", db_connection=self.db_conn)

        # Select model for task
        model_name = learner.select_model(task_type="tool_call", context={"complexity": "low"})

        # Should return a model name
        assert model_name is not None
        assert isinstance(model_name, str)


class TestExtendedModeTransitionLearner:
    """Test ExtendedModeTransitionLearner integration."""

    def setup_method(self):
        """Create test database connection."""
        self.db_conn = sqlite3.connect(":memory:")

    def teardown_method(self):
        """Close database connection."""
        if hasattr(self, "db_conn"):
            self.db_conn.close()

    def test_learner_initialization(self):
        """Test learner can be initialized with phase detector."""
        from victor.framework.rl.learners.mode_transition_extended import (
            ExtendedModeTransitionLearner,
        )

        learner = ExtendedModeTransitionLearner(name="mode_transition", db_connection=self.db_conn)

        assert learner is not None
        assert hasattr(learner, "phase_detector")
        assert hasattr(learner, "transition_detector")
        assert hasattr(learner, "detect_phase")
        assert hasattr(learner, "should_transition")

    def test_detect_phase(self):
        """Test phase detection."""
        from victor.framework.rl.learners.mode_transition_extended import (
            ExtendedModeTransitionLearner,
        )

        learner = ExtendedModeTransitionLearner(name="mode_transition", db_connection=self.db_conn)

        # Detect phase from conversation state
        phase = learner.detect_phase(
            current_stage=ConversationStage.INITIAL,
            recent_tools=[],
            message_content="Let me explore the codebase",
        )

        # Should return a phase
        assert phase in TaskPhase
        assert isinstance(phase, TaskPhase)

    def test_should_transition(self):
        """Test phase transition validation."""
        from victor.framework.rl.learners.mode_transition_extended import (
            ExtendedModeTransitionLearner,
        )

        learner = ExtendedModeTransitionLearner(name="mode_transition", db_connection=self.db_conn)

        # Test valid transition
        can_transition = learner.should_transition(
            current_phase=TaskPhase.EXPLORATION, new_phase=TaskPhase.PLANNING
        )

        # Should allow transition
        assert isinstance(can_transition, bool)

    def test_learn_from_phase_transitions(self):
        """Test learning from phase transitions."""
        from victor.framework.rl.learners.mode_transition_extended import (
            ExtendedModeTransitionLearner,
        )

        learner = ExtendedModeTransitionLearner(name="mode_transition", db_connection=self.db_conn)

        # Create test outcomes
        outcomes = [
            RLOutcome(
                provider="system",
                model="phase_detector",
                task_type="mode_transition",
                success=True,
                quality_score=0.9,
                metadata={
                    "detected_phase": "planning",
                    "from_phase": "exploration",
                    "to_phase": "planning",
                    "transition_successful": True,
                },
            ),
        ]

        # Learn from outcomes
        recommendations = learner.learn(outcomes)

        # Should generate recommendations
        assert len(recommendations) > 0

        # Check recommendation types
        rec_types = {r.recommendation_type for r in recommendations}
        assert "phase_transition" in rec_types


class TestExtendedToolSelectorLearner:
    """Test ExtendedToolSelectorLearner integration."""

    def setup_method(self):
        """Create test database connection and reset UsageAnalytics."""
        self.db_conn = sqlite3.connect(":memory:")
        UsageAnalytics.reset_instance()

    def teardown_method(self):
        """Close database connection and reset UsageAnalytics."""
        if hasattr(self, "db_conn"):
            self.db_conn.close()
        UsageAnalytics.reset_instance()

    def test_learner_initialization(self):
        """Test learner can be initialized with predictor and analytics."""
        from victor.framework.rl.learners.tool_selector_extended import (
            ExtendedToolSelectorLearner,
        )

        learner = ExtendedToolSelectorLearner(name="tool_selector", db_connection=self.db_conn)

        assert learner is not None
        assert hasattr(learner, "predictor")
        assert hasattr(learner, "analytics")
        assert hasattr(learner, "predict_next_tool")

    def test_predict_next_tool(self):
        """Test tool prediction."""
        from victor.framework.rl.learners.tool_selector_extended import (
            ExtendedToolSelectorLearner,
        )

        learner = ExtendedToolSelectorLearner(name="tool_selector", db_connection=self.db_conn)

        # Train predictor with some patterns
        for _ in range(3):
            learner.predictor.cooccurrence_tracker.record_tool_sequence(
                tools=["search", "read"], task_type="bugfix", success=True
            )

        # Predict next tool
        tool_name = learner.predict_next_tool(
            task_description="Find and fix the bug",
            current_step="exploration",
            recent_tools=["search"],
            task_type="bugfix",
        )

        # Should predict a tool
        assert tool_name is not None
        assert isinstance(tool_name, str)

    def test_learn_from_outcomes(self):
        """Test learning from tool execution outcomes."""
        from victor.framework.rl.learners.tool_selector_extended import (
            ExtendedToolSelectorLearner,
        )

        learner = ExtendedToolSelectorLearner(name="tool_selector", db_connection=self.db_conn)

        # Record some tool executions
        for i in range(10):
            learner.analytics.record_tool_execution(
                tool_name="read",
                success=(i < 8),  # 80% success rate
                execution_time_ms=50.0,
                error=None if i < 8 else "Not found",
            )

        # Create test outcomes
        outcomes = [
            RLOutcome(
                provider="system",
                model="tool_executor",
                task_type="tool_execution",
                success=True,
                quality_score=0.9,
                metadata={
                    "tool_name": "read",
                    "tools_used": ["read"],
                    "task_type": "bugfix",
                },
            ),
        ]

        # Learn from outcomes
        recommendations = learner.learn(outcomes)

        # Should generate recommendations
        assert len(recommendations) > 0

        # Check recommendation types
        rec_types = {r.recommendation_type for r in recommendations}
        assert "tool_usage" in rec_types

    def test_get_tool_insights(self):
        """Test getting tool insights."""
        from victor.framework.rl.learners.tool_selector_extended import (
            ExtendedToolSelectorLearner,
        )

        learner = ExtendedToolSelectorLearner(name="tool_selector", db_connection=self.db_conn)

        # Record some executions
        for i in range(5):
            learner.analytics.record_tool_execution(
                tool_name="edit", success=True, execution_time_ms=100.0, error=None
            )

        # Get insights
        insights = learner.get_tool_insights("edit")

        # Should return insights
        assert isinstance(insights, dict)
        assert "success_rate" in insights
        assert "avg_execution_ms" in insights
        assert "execution_count" in insights


class TestExtendedLearnersIntegration:
    """Test integration between all three extended learners."""

    def setup_method(self):
        """Create test database connection."""
        self.db_conn = sqlite3.connect(":memory:")

    def teardown_method(self):
        """Close database connection."""
        if hasattr(self, "db_conn"):
            self.db_conn.close()

    def test_all_extended_learners_instantiable(self):
        """Test all extended learners can be instantiated."""
        from victor.framework.rl.learners.model_selector_extended import (
            ExtendedModelSelectorLearner,
        )
        from victor.framework.rl.learners.mode_transition_extended import (
            ExtendedModeTransitionLearner,
        )
        from victor.framework.rl.learners.tool_selector_extended import (
            ExtendedToolSelectorLearner,
        )

        model_learner = ExtendedModelSelectorLearner(
            name="model_selector", db_connection=self.db_conn
        )
        mode_learner = ExtendedModeTransitionLearner(
            name="mode_transition", db_connection=self.db_conn
        )
        tool_learner = ExtendedToolSelectorLearner(name="tool_selector", db_connection=self.db_conn)

        assert model_learner is not None
        assert mode_learner is not None
        assert tool_learner is not None

    def test_extended_learners_have_learn_method(self):
        """Test all extended learners have learn method."""
        from victor.framework.rl.learners.model_selector_extended import (
            ExtendedModelSelectorLearner,
        )
        from victor.framework.rl.learners.mode_transition_extended import (
            ExtendedModeTransitionLearner,
        )
        from victor.framework.rl.learners.tool_selector_extended import (
            ExtendedToolSelectorLearner,
        )

        model_learner = ExtendedModelSelectorLearner(
            name="model_selector", db_connection=self.db_conn
        )
        mode_learner = ExtendedModeTransitionLearner(
            name="mode_transition", db_connection=self.db_conn
        )
        tool_learner = ExtendedToolSelectorLearner(name="tool_selector", db_connection=self.db_conn)

        # All should have learn method
        assert hasattr(model_learner, "learn")
        assert hasattr(mode_learner, "learn")
        assert hasattr(tool_learner, "learn")

        # All should be callable
        assert callable(model_learner.learn)
        assert callable(mode_learner.learn)
        assert callable(tool_learner.learn)

    def test_extended_learners_produce_recommendations(self):
        """Test all extended learners produce recommendations."""
        from victor.framework.rl.learners.model_selector_extended import (
            ExtendedModelSelectorLearner,
        )
        from victor.framework.rl.learners.mode_transition_extended import (
            ExtendedModeTransitionLearner,
        )
        from victor.framework.rl.learners.tool_selector_extended import (
            ExtendedToolSelectorLearner,
        )

        # Create test outcomes
        outcomes = [
            RLOutcome(
                provider="test",
                model="test",
                task_type="test",
                success=True,
                quality_score=0.9,
                metadata={"test": "data"},
            )
        ]

        # Test each learner
        model_learner = ExtendedModelSelectorLearner(
            name="model_selector", db_connection=self.db_conn
        )
        mode_learner = ExtendedModeTransitionLearner(
            name="mode_transition", db_connection=self.db_conn
        )
        tool_learner = ExtendedToolSelectorLearner(name="tool_selector", db_connection=self.db_conn)

        model_recs = model_learner.learn(outcomes)
        mode_recs = mode_learner.learn(outcomes)
        tool_recs = tool_learner.learn(outcomes)

        # All should return lists
        assert isinstance(model_recs, list)
        assert isinstance(mode_recs, list)
        assert isinstance(tool_recs, list)

        # May be empty (no patterns yet), but should not error
