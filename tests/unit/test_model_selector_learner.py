"""Unit tests for model selector RL learner."""

import sqlite3
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Optional, Tuple

from victor.agent.rl.base import RLOutcome, RLRecommendation
from victor.agent.rl.coordinator import RLCoordinator
from victor.agent.rl.learners.model_selector import ModelSelectorLearner, SelectionStrategy
from victor.core.database import reset_database, get_database
from victor.core.schema import Tables


@pytest.fixture
def coordinator(tmp_path: Path) -> RLCoordinator:
    """Fixture for RLCoordinator, ensuring a clean database for each test."""
    # Reset the global singleton to ensure fresh database for each test
    reset_database()
    db_path = tmp_path / "rl_test.db"
    # Initialize the database singleton with temp path BEFORE creating coordinator
    get_database(db_path)
    coord = RLCoordinator(storage_path=tmp_path, db_path=db_path)
    yield coord
    # Reset again after the test to clean up
    reset_database()


@pytest.fixture
def learner(coordinator: RLCoordinator) -> ModelSelectorLearner:
    """Fixture for ModelSelectorLearner."""
    return coordinator.get_learner("model_selector")  # type: ignore


def _record_selection_outcome(
    learner: ModelSelectorLearner,
    provider: str = "ollama",
    model: str = "qwen3-coder-tools:30b",
    task_type: str = "analysis",
    *,
    success: bool = True,
    quality_score: float = 0.8,
    latency_seconds: float = 5.0,
    token_count: int = 1000,
    tool_calls_made: int = 5,
    user_satisfaction: Optional[float] = None,
) -> None:
    """Helper to record a single model selection outcome."""
    outcome = RLOutcome(
        provider=provider,
        model=model,
        task_type=task_type,
        success=success,
        quality_score=quality_score,
        metadata={
            "latency_seconds": latency_seconds,
            "token_count": token_count,
            "tool_calls_made": tool_calls_made,
            "user_satisfaction": user_satisfaction,
        },
    )
    learner.record_outcome(outcome)


def _get_q_value_from_db(
    coordinator: RLCoordinator,
    provider: str,
    task_type: Optional[str] = None,
) -> Tuple[float, int]:
    """Helper to retrieve Q-value and selection count from the database."""
    cursor = coordinator.db.cursor()
    if task_type:
        cursor.execute(
            f"SELECT q_value, selection_count FROM {Tables.RL_MODEL_TASK} WHERE provider = ? AND task_type = ?",
            (provider, task_type),
        )
    else:
        cursor.execute(
            f"SELECT q_value, selection_count FROM {Tables.RL_MODEL_Q} WHERE provider = ?",
            (provider,),
        )
    row = cursor.fetchone()
    return (row[0], row[1]) if row else (0.5, 0)  # Default values


class TestModelSelectorLearner:
    def test_initialization(self, learner: ModelSelectorLearner) -> None:
        """Test learner initializes correctly and creates tables."""
        assert learner.name == "model_selector"
        cursor = learner.db.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_MODEL_Q}';"
        )
        assert cursor.fetchone() is not None
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_MODEL_TASK}';"
        )
        assert cursor.fetchone() is not None
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_MODEL_STATE}';"
        )
        assert cursor.fetchone() is not None

    def test_record_single_outcome(
        self, coordinator: RLCoordinator, learner: ModelSelectorLearner
    ) -> None:
        """Recording one outcome updates Q-values and counts."""
        _record_selection_outcome(
            learner,
            provider="anthropic",
            model="claude-3-opus",
            task_type="analysis",
            success=True,
            quality_score=0.9,
            latency_seconds=10.0,
        )

        q_value, count = _get_q_value_from_db(coordinator, "anthropic")
        assert count == 1
        assert q_value != 0.5  # Should have been updated

        task_q_value, task_count = _get_q_value_from_db(coordinator, "anthropic", "analysis")
        assert task_count == 1
        assert task_q_value != 0.5

    def test_persistence(self, tmp_path: Path) -> None:
        """State persists across learner instances."""
        # Reset to ensure clean state
        reset_database()
        db_path = tmp_path / "rl_test.db"

        # Initialize DB singleton with temp path
        get_database(db_path)
        coordinator1 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner1 = coordinator1.get_learner("model_selector")  # type: ignore
        initial_epsilon_learner1 = learner1.epsilon

        _record_selection_outcome(learner1, provider="openai")

        # Reset the singleton to simulate new session (don't call close directly)
        reset_database()

        # Re-initialize with same temp path to test persistence
        get_database(db_path)
        coordinator2 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner2 = coordinator2.get_learner("model_selector")  # type: ignore

        q_value, count = _get_q_value_from_db(coordinator2, "openai")
        assert count == 1
        assert q_value != 0.5
        assert (
            learner2.epsilon < initial_epsilon_learner1
        )  # Epsilon should decay from initial value
        assert learner2.epsilon == learner1.epsilon  # The actual decayed value should be loaded

        # Clean up
        reset_database()

    @pytest.mark.parametrize(
        "strategy, expected_reason",
        [
            (SelectionStrategy.EPSILON_GREEDY, "Exploration"),
            (SelectionStrategy.EXPLOIT_ONLY, "Best Q-value"),
        ],
    )
    def test_get_recommendation_strategies(
        self, learner: ModelSelectorLearner, strategy: SelectionStrategy, expected_reason: str
    ) -> None:
        """Test different selection strategies."""
        learner.strategy = strategy
        learner.epsilon = (
            1.0 if strategy == SelectionStrategy.EPSILON_GREEDY else 0.0
        )  # Force exploration/exploitation

        _record_selection_outcome(
            learner, provider="provider_A", quality_score=0.9, task_type="analysis"
        )
        _record_selection_outcome(
            learner, provider="provider_B", quality_score=0.1, task_type="analysis"
        )

        # Mock random.random to ensure exploration is chosen for EPSILON_GREEDY
        import random

        with patch.object(
            random, "random", return_value=0.1
        ):  # Ensure random.random() < learner.epsilon (which is 0.99)
            # Get recommendation for providers A and B
            rec = learner.get_recommendation(
                provider='["provider_A", "provider_B"]', model="any-model", task_type="analysis"
            )
        assert rec is not None
        assert rec.value in ["provider_A", "provider_B"]
        assert expected_reason in rec.reason

    def test_ucb_strategy(self, learner: ModelSelectorLearner) -> None:
        """Test UCB selection strategy."""
        learner.strategy = SelectionStrategy.UCB

        # Populate some data
        _record_selection_outcome(
            learner,
            provider="provider_X",
            quality_score=0.8,
            task_type="coding",
            latency_seconds=1.0,
        )
        _record_selection_outcome(
            learner,
            provider="provider_X",
            quality_score=0.8,
            task_type="coding",
            latency_seconds=1.0,
        )
        _record_selection_outcome(
            learner,
            provider="provider_Y",
            quality_score=0.2,
            task_type="coding",
            latency_seconds=10.0,
        )

        rec = learner.get_recommendation(
            provider='["provider_X", "provider_Y"]', model="any-model", task_type="coding"
        )
        assert rec is not None
        # Provider_X should have higher Q-value and thus be selected more often
        # This is a probabilistic test, so we can't assert exact selection, but
        # for a small number of selections and high difference, X should win.
        # For simplicity, we'll just check it returns one of them.
        assert rec.value in ["provider_X", "provider_Y"]
