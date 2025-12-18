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

"""Centralized RL coordinator with unified SQLite storage."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from victor.agent.rl.base import BaseLearner, RLOutcome, RLRecommendation

logger = logging.getLogger(__name__)


class RLCoordinator:
    """Centralized coordinator for all RL learners.

    Responsibilities:
    - Manage unified SQLite database for all learners
    - Register and coordinate learners
    - Collect telemetry across verticals
    - Enable cross-vertical learning
    - Export metrics for monitoring

    Architecture:
    - Single SQLite database: ~/.victor/rl_data/rl.db
    - Each learner gets own tables (prefixed with learner name)
    - Shared outcomes table for cross-learner analysis
    - Telemetry table for monitoring

    Usage:
        coordinator = get_rl_coordinator()
        coordinator.record_outcome("continuation_patience", outcome, "coding")
        rec = coordinator.get_recommendation("continuation_patience", ...)
    """

    def __init__(self, storage_path: Path, db_path: Optional[Path] = None):
        """Initialize RL coordinator.

        Args:
            storage_path: Directory for RL data (e.g., ~/.victor/rl_data/)
            db_path: Path to SQLite database (defaults to ~/.victor/graph/graph.db)
        """
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Use existing graph database for RL tables
        if db_path is None:
            db_path = Path.home() / ".victor" / "graph" / "graph.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row

        # Registry of learners
        self._learners: Dict[str, BaseLearner] = {}

        # Ensure core tables exist
        self._ensure_core_tables()

        # Auto-register default learners
        self._register_default_learners()

        logger.info(f"RL: Coordinator initialized with database at {self.db_path}")

    def _ensure_core_tables(self) -> None:
        """Create core tables for telemetry and cross-learner analysis."""
        cursor = self.db.cursor()

        # Shared outcomes table for all learners
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                vertical TEXT NOT NULL,
                success INTEGER NOT NULL,
                quality_score REAL NOT NULL,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                created_at REAL DEFAULT (julianday('now'))
            )
            """
        )

        # Index for fast lookups
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rl_outcomes_learner
            ON rl_outcomes(learner_name, provider, model, task_type)
            """
        )

        # Telemetry table for monitoring
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )

        self.db.commit()
        logger.debug("RL: Core tables ensured")

    def _register_default_learners(self) -> None:
        """Register default learners.

        Note: Actual learner instances will be created lazily on first use
        to avoid circular imports and unnecessary initialization.
        """
        # Learners are registered when first accessed via get_learner()
        pass

    def register_learner(self, name: str, learner: BaseLearner) -> None:
        """Register a learner with the coordinator.

        Args:
            name: Learner name (e.g., "continuation_patience")
            learner: Learner instance
        """
        if name in self._learners:
            logger.warning(f"RL: Learner '{name}' already registered, replacing")

        self._learners[name] = learner
        logger.info(f"RL: Registered learner '{name}'")

    def get_learner(self, name: str) -> Optional[BaseLearner]:
        """Get a learner by name, creating it lazily if needed.

        Args:
            name: Learner name

        Returns:
            Learner instance or None if unknown
        """
        if name in self._learners:
            return self._learners[name]

        # Lazy initialization of learners
        learner = self._create_learner(name)
        if learner:
            self.register_learner(name, learner)

        return learner

    def _create_learner(self, name: str) -> Optional[BaseLearner]:
        """Create a learner instance on demand.

        Args:
            name: Learner name

        Returns:
            Learner instance or None if unknown
        """
        # Import here to avoid circular dependencies
        try:
            if name == "continuation_patience":
                from victor.agent.rl.learners.continuation_patience import (
                    ContinuationPatienceLearner,
                )

                return ContinuationPatienceLearner(
                    name=name, db_connection=self.db, learning_rate=0.1
                )
            elif name == "continuation_prompts":
                from victor.agent.rl.learners.continuation_prompts import (
                    ContinuationPromptLearner,
                )

                return ContinuationPromptLearner(
                    name=name, db_connection=self.db, learning_rate=0.1
                )
            elif name == "semantic_threshold":
                from victor.agent.rl.learners.semantic_threshold import (
                    SemanticThresholdLearner,
                )

                return SemanticThresholdLearner(
                    name=name, db_connection=self.db, learning_rate=0.1
                )
            elif name == "model_selector":
                from victor.agent.rl.learners.model_selector import ModelSelectorLearner

                return ModelSelectorLearner(
                    name=name, db_connection=self.db, learning_rate=0.1
                )
            else:
                logger.warning(f"RL: Unknown learner '{name}'")
                return None
        except ImportError as e:
            logger.warning(f"RL: Failed to import learner '{name}': {e}")
            return None

    def record_outcome(
        self,
        learner_name: str,
        outcome: RLOutcome,
        vertical: str = "coding",
    ) -> None:
        """Record an outcome for a specific learner.

        This is the main entry point for providing feedback to learners.

        Args:
            learner_name: Name of learner to update
            outcome: Outcome data
            vertical: Which vertical this came from (coding, devops, data_science)
        """
        outcome.vertical = vertical

        learner = self.get_learner(learner_name)
        if not learner:
            logger.warning(f"RL: Unknown learner '{learner_name}', skipping outcome")
            return

        try:
            # Record in learner-specific tables
            learner.record_outcome(outcome)

            # Record in shared outcomes table
            cursor = self.db.cursor()
            cursor.execute(
                """
                INSERT INTO rl_outcomes (
                    learner_name, provider, model, task_type, vertical,
                    success, quality_score, metadata, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    learner_name,
                    outcome.provider,
                    outcome.model,
                    outcome.task_type,
                    outcome.vertical,
                    1 if outcome.success else 0,
                    outcome.quality_score,
                    outcome.to_dict()["metadata"],  # JSON string
                    outcome.timestamp,
                ),
            )
            self.db.commit()

            logger.debug(
                f"RL: Recorded outcome for {learner_name} "
                f"({outcome.provider}:{outcome.model}:{outcome.task_type})"
            )

        except Exception as e:
            logger.error(f"RL: Failed to record outcome for {learner_name}: {e}")
            self.db.rollback()

    def get_recommendation(
        self,
        learner_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[RLRecommendation]:
        """Get recommendation from a learner.

        Args:
            learner_name: Name of learner to query
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Recommendation with value and confidence, or None if no data
        """
        learner = self.get_learner(learner_name)
        if not learner:
            logger.debug(f"RL: Unknown learner '{learner_name}', no recommendation")
            return None

        try:
            return learner.get_recommendation(provider, model, task_type)
        except Exception as e:
            logger.error(
                f"RL: Failed to get recommendation from {learner_name}: {e}"
            )
            return None

    def export_metrics(self) -> Dict[str, Any]:
        """Export all learned values and metrics for monitoring.

        Returns:
            Dictionary with metrics from all learners
        """
        metrics = {"coordinator": {"db_path": str(self.db_path), "learners": {}}}

        for name, learner in self._learners.items():
            try:
                metrics["coordinator"]["learners"][name] = learner.export_metrics()
            except Exception as e:
                logger.error(f"RL: Failed to export metrics for {name}: {e}")
                metrics["coordinator"]["learners"][name] = {"error": str(e)}

        # Add global stats
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM rl_outcomes")
        metrics["coordinator"]["total_outcomes"] = cursor.fetchone()[0]

        return metrics

    def get_all_recommendations(
        self, provider: str, model: str, task_type: str
    ) -> Dict[str, RLRecommendation]:
        """Get recommendations from all learners for given context.

        Useful for displaying all learned values for a specific provider/model/task.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Dictionary mapping learner name to recommendation
        """
        recommendations = {}

        for name in ["continuation_patience", "continuation_prompts", "semantic_threshold"]:
            rec = self.get_recommendation(name, provider, model, task_type)
            if rec:
                recommendations[name] = rec

        return recommendations

    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()
            logger.debug("RL: Database connection closed")


# Global singleton
_rl_coordinator: Optional[RLCoordinator] = None


def get_rl_coordinator() -> RLCoordinator:
    """Get global RL coordinator (lazy init).

    Returns:
        Global coordinator singleton
    """
    global _rl_coordinator
    if _rl_coordinator is None:
        storage_path = Path.home() / ".victor" / "rl_data"
        _rl_coordinator = RLCoordinator(storage_path)
    return _rl_coordinator
