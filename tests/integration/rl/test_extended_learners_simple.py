"""Simple integration tests for Priority 4 Phase 2: Extended RL Learners."""

import sqlite3

import pytest

from victor.agent.usage_analytics import UsageAnalytics
from victor.agent.context_phase_detector import PhaseDetector
from victor.agent.planning.tool_predictor import ToolPredictor
from victor.agent.services.hybrid_decision_service import HybridDecisionService
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


class TestExtendedToolSelectorLearner:
    """Test ExtendedToolSelectorLearner integration."""

    def setup_method(self):
        """Create test database connection."""
        self.db_conn = sqlite3.connect(":memory:")
        UsageAnalytics.reset_instance()

    def teardown_method(self):
        """Close database connection."""
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

    def test_predict_next_tool(self):
        """Test tool prediction."""
        from victor.framework.rl.learners.tool_selector_extended import (
            ExtendedToolSelectorLearner,
        )
        from victor.agent.planning.cooccurrence_tracker import CooccurrenceTracker

        learner = ExtendedToolSelectorLearner(name="tool_selector", db_connection=self.db_conn)

        # Create a new tracker and train with patterns
        tracker = CooccurrenceTracker()
        for _ in range(3):
            tracker.record_tool_sequence(tools=["search", "read"], task_type="bugfix", success=True)

        # Update learner's predictor with trained tracker
        learner.predictor._cooccurrence_tracker = tracker

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


class TestAllExtendedLearners:
    """Test all three extended learners work together."""

    def setup_method(self):
        """Create test database connection."""
        self.db_conn = sqlite3.connect(":memory:")
        UsageAnalytics.reset_instance()

    def teardown_method(self):
        """Close database connection."""
        if hasattr(self, "db_conn"):
            self.db_conn.close()
        UsageAnalytics.reset_instance()

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

    def test_all_have_learn_method(self):
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

    def test_integrated_components_exist(self):
        """Test all integrated components exist and work."""
        # Test HybridDecisionService
        service = HybridDecisionService()
        assert service is not None

        # Test PhaseDetector
        detector = PhaseDetector()
        phase = detector.detect_phase(
            current_stage=ConversationStage.INITIAL, recent_tools=[], message_content="Test"
        )
        assert phase in TaskPhase

        # Test ToolPredictor
        predictor = ToolPredictor()
        predictions = predictor.predict_tools(
            task_description="Test", current_step="exploration", recent_tools=[], task_type="test"
        )
        assert isinstance(predictions, list)

        # Test UsageAnalytics
        analytics = UsageAnalytics.get_instance()
        assert analytics is not None
        summary = analytics.get_session_summary()
        assert isinstance(summary, dict)
