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

"""Centralized RL coordinator with unified SQLite storage.

Provides both sync and async interfaces:
- Sync methods: `record_outcome()`, `get_recommendation()` (for sync callers)
- Async methods: `record_outcome_async()`, `get_recommendation_async()` (for async orchestrator)

The async methods use `asyncio.to_thread()` to offload SQLite operations to a thread pool,
preventing event loop blocking while maintaining the same synchronous SQLite internals.

Database:
    Uses the unified database at ~/.victor/victor.db via victor.core.database.
    All RL tables are consolidated in this single database for easier management.
"""

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from victor.agent.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.database import get_database
from victor.core.schema import Tables, Schema

logger = logging.getLogger(__name__)


class BatchedOutcomeWriter:
    """Batched writer for RL outcomes to reduce database commits.

    Instead of committing each outcome immediately, this class queues
    outcomes and writes them in batches. This significantly reduces
    I/O overhead when recording multiple outcomes.

    Example:
        writer = BatchedOutcomeWriter(coordinator, batch_size=50)
        for outcome in outcomes:
            writer.queue_outcome("learner", outcome, "vertical")
        writer.flush()  # Commit all queued outcomes

        # Or use as context manager
        with BatchedOutcomeWriter(coordinator) as writer:
            writer.queue_outcome("learner", outcome, "vertical")
        # Auto-flush on exit

    Attributes:
        coordinator: RLCoordinator to write to
        batch_size: Max outcomes per batch before auto-flush
        _queue: List of queued (learner_name, outcome, vertical) tuples
    """

    def __init__(
        self,
        coordinator: "RLCoordinator",
        batch_size: int = 50,
        auto_flush_on_exit: bool = True,
    ) -> None:
        """Initialize batched writer.

        Args:
            coordinator: RLCoordinator to write to
            batch_size: Max outcomes per batch before auto-flush
            auto_flush_on_exit: Flush on context manager exit
        """
        self.coordinator = coordinator
        self.batch_size = batch_size
        self.auto_flush_on_exit = auto_flush_on_exit
        self._queue: list[tuple[str, RLOutcome, str]] = []
        self._flush_count = 0

    def __enter__(self) -> "BatchedOutcomeWriter":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, optionally flushing."""
        if self.auto_flush_on_exit and self._queue:
            self.flush()

    def queue_outcome(
        self,
        learner_name: str,
        outcome: RLOutcome,
        vertical: str = "coding",
    ) -> None:
        """Queue an outcome for batched writing.

        Args:
            learner_name: Name of learner to update
            outcome: Outcome data
            vertical: Which vertical this came from
        """
        self._queue.append((learner_name, outcome, vertical))

        # Auto-flush if batch is full
        if len(self._queue) >= self.batch_size:
            self.flush()

    def flush(self) -> int:
        """Flush all queued outcomes to database.

        Writes all outcomes in a single transaction for efficiency.

        Returns:
            Number of outcomes written
        """
        if not self._queue:
            return 0

        count = len(self._queue)
        queue_copy = self._queue.copy()
        self._queue.clear()

        # Use coordinator's db connection for transaction
        db = self.coordinator.db
        cursor = db.cursor()

        try:
            from datetime import datetime as dt

            timestamp_now = dt.now().isoformat()

            for learner_name, outcome, vertical in queue_copy:
                outcome.vertical = vertical

                # Record in learner-specific tables
                learner = self.coordinator.get_learner(learner_name)
                if learner:
                    # Use learner's record_outcome but skip learner's commit
                    learner.record_outcome(outcome)

                # Record in shared outcomes table
                cursor.execute(
                    f"""
                    INSERT INTO {Tables.RL_OUTCOME} (
                        learner_name, learner_id, provider, model, task_type, vertical,
                        success, quality_score, metadata, timestamp, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        learner_name,
                        learner_name,
                        outcome.provider,
                        outcome.model,
                        outcome.task_type,
                        outcome.vertical or "general",
                        1 if outcome.success else 0,
                        outcome.quality_score,
                        outcome.to_dict()["metadata"],
                        timestamp_now,
                    ),
                )

            # Single commit for all outcomes
            db.commit()
            self._flush_count += count
            logger.debug(f"RL: Flushed batch of {count} outcomes")
            return count

        except Exception as e:
            logger.error(f"RL: Failed to flush {count} outcomes: {e}")
            db.rollback()
            # Re-queue failed outcomes
            self._queue.extend(queue_copy)
            raise

    def get_queue_size(self) -> int:
        """Get number of queued outcomes."""
        return len(self._queue)

    def get_flush_count(self) -> int:
        """Get total number of outcomes flushed."""
        return self._flush_count

    async def flush_async(self) -> int:
        """Async version of flush."""
        return await asyncio.to_thread(self.flush)


class RLCoordinator:
    """Centralized coordinator for all RL learners.

    Responsibilities:
    - Manage unified SQLite database for all learners
    - Register and coordinate learners
    - Collect telemetry across verticals
    - Enable cross-vertical learning
    - Export metrics for monitoring

    Architecture:
    - Uses unified database: ~/.victor/victor.db (via victor.core.database)
    - Each learner gets own tables (prefixed with learner name)
    - Shared outcomes table for cross-learner analysis
    - Telemetry table for monitoring

    Usage:
        coordinator = get_rl_coordinator()
        coordinator.record_outcome("continuation_patience", outcome, "coding")
        rec = coordinator.get_recommendation("continuation_patience", ...)
    """

    def __init__(self, storage_path: Optional[Path] = None, db_path: Optional[Path] = None):
        """Initialize RL coordinator.

        Args:
            storage_path: Directory for RL data (e.g., ~/.victor/rl_data/) - legacy, now ignored
            db_path: Path to SQLite database - legacy, now uses unified database
        """
        # Legacy storage_path kept for backward compatibility
        if storage_path is None:
            storage_path = Path.home() / ".victor" / "rl_data"
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Use unified database from victor.core.database
        self._db_manager = get_database()
        self.db = self._db_manager.get_connection()
        self.db_path = self._db_manager.db_path

        # Registry of learners
        self._learners: Dict[str, BaseLearner] = {}

        # Ensure core tables exist
        self._ensure_core_tables()

        # Auto-register default learners
        self._register_default_learners()

        # Connect to RL hooks and metrics for event-driven updates
        self._connect_hooks_and_metrics()

        logger.info(f"RL: Coordinator initialized with unified database at {self.db_path}")

    def _ensure_core_tables(self) -> None:
        """Create core tables for telemetry and cross-learner analysis."""
        cursor = self.db.cursor()

        # Shared outcomes table for all learners (uses schema constant)
        cursor.execute(Schema.RL_OUTCOME)

        # Index for fast lookups
        cursor.executescript(Schema.RL_OUTCOME_INDEXES)

        # Telemetry/metrics table for monitoring (uses schema constant)
        cursor.execute(Schema.RL_METRIC)

        self.db.commit()
        logger.debug("RL: Core tables ensured")

    def _register_default_learners(self) -> None:
        """Register default learners.

        Note: Actual learner instances will be created lazily on first use
        to avoid circular imports and unnecessary initialization.
        """
        # Learners are registered when first accessed via get_learner()
        pass

    def _connect_hooks_and_metrics(self) -> None:
        """Connect to RL hooks registry and metrics exporter.

        This enables event-driven learner activation and metrics collection.
        """
        try:
            # Connect to hooks registry
            from victor.agent.rl.hooks import get_rl_hooks

            hooks = get_rl_hooks(coordinator=self)
            logger.debug("RL: Connected to hooks registry")

            # Connect to metrics exporter
            from victor.agent.rl.metrics import get_rl_metrics

            get_rl_metrics(coordinator=self, hooks=hooks)
            logger.debug("RL: Connected to metrics exporter")

        except ImportError as e:
            logger.debug("RL: Hooks/metrics not available: %s", e)

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

                return SemanticThresholdLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "model_selector":
                from victor.agent.rl.learners.model_selector import ModelSelectorLearner

                return ModelSelectorLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "cache_eviction":
                from victor.agent.rl.learners.cache_eviction import CacheEvictionLearner

                return CacheEvictionLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "grounding_threshold":
                from victor.agent.rl.learners.grounding_threshold import GroundingThresholdLearner

                return GroundingThresholdLearner(
                    name=name, db_connection=self.db, learning_rate=0.1
                )
            elif name == "quality_weights":
                from victor.agent.rl.learners.quality_weights import QualityWeightLearner

                return QualityWeightLearner(name=name, db_connection=self.db, learning_rate=0.05)
            elif name == "tool_selector":
                from victor.agent.rl.learners.tool_selector import ToolSelectorLearner

                return ToolSelectorLearner(name=name, db_connection=self.db, learning_rate=0.05)
            elif name == "mode_transition":
                from victor.agent.rl.learners.mode_transition import ModeTransitionLearner

                return ModeTransitionLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "prompt_template":
                from victor.agent.rl.learners.prompt_template import PromptTemplateLearner

                return PromptTemplateLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "team_composition":
                from victor.agent.teams.learner import TeamCompositionLearner

                # TeamCompositionLearner has different signature - uses db_path instead of db_connection
                return TeamCompositionLearner(learning_rate=0.1)
            elif name == "cross_vertical":
                from victor.agent.rl.learners.cross_vertical import CrossVerticalLearner

                return CrossVerticalLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "workflow_execution":
                from victor.agent.rl.learners.workflow_execution import WorkflowExecutionLearner

                return WorkflowExecutionLearner(name=name, db_connection=self.db, learning_rate=0.1)
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
            # Note: Uses both learner_name (legacy NOT NULL) and learner_id (new schema)
            # Also includes timestamp (legacy NOT NULL) alongside created_at (new schema)
            from datetime import datetime as dt
            timestamp_now = dt.now().isoformat()
            cursor = self.db.cursor()
            cursor.execute(
                f"""
                INSERT INTO {Tables.RL_OUTCOME} (
                    learner_name, learner_id, provider, model, task_type, vertical,
                    success, quality_score, metadata, timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    learner_name,  # legacy column (NOT NULL)
                    learner_name,  # new column (for index)
                    outcome.provider,
                    outcome.model,
                    outcome.task_type,
                    outcome.vertical or "general",  # Default to "general" if None
                    1 if outcome.success else 0,
                    outcome.quality_score,
                    outcome.to_dict()["metadata"],  # JSON string
                    timestamp_now,  # legacy timestamp column (NOT NULL)
                ),
            )
            self.db.commit()

            # Also record telemetry metric
            self._record_metric(
                learner_name,
                "outcome_recorded",
                outcome.quality_score,
                {"success": outcome.success, "task_type": outcome.task_type},
            )

            logger.debug(
                f"RL: Recorded outcome for {learner_name} "
                f"({outcome.provider}:{outcome.model}:{outcome.task_type})"
            )

        except Exception as e:
            logger.error(f"RL: Failed to record outcome for {learner_name}: {e}")
            self.db.rollback()

    def _record_metric(
        self,
        learner_id: str,
        metric_type: str,
        metric_value: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a telemetry metric to rl_metric table.

        Args:
            learner_id: Learner name/ID
            metric_type: Type of metric (e.g., 'outcome_recorded', 'q_value_update')
            metric_value: Numeric metric value
            metadata: Optional additional metadata dict
        """
        try:
            import json
            cursor = self.db.cursor()
            cursor.execute(
                f"""
                INSERT INTO {Tables.RL_METRIC}
                (learner_id, metric_type, metric_value, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (
                    learner_id,
                    metric_type,
                    metric_value,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            self.db.commit()
        except Exception as e:
            logger.debug(f"RL: Failed to record metric: {e}")

    def record_metric(
        self,
        learner_id: str,
        metric_type: str,
        metric_value: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """Public interface to record telemetry metrics.

        Args:
            learner_id: Learner name/ID
            metric_type: Type of metric
            metric_value: Numeric metric value
            metadata: Optional additional metadata
        """
        self._record_metric(learner_id, metric_type, metric_value, metadata)

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
            logger.error(f"RL: Failed to get recommendation from {learner_name}: {e}")
            return None

    # =========================================================================
    # Async Wrappers - Non-blocking versions for async orchestrator
    # =========================================================================

    async def record_outcome_async(
        self,
        learner_name: str,
        outcome: RLOutcome,
        vertical: str = "coding",
    ) -> None:
        """Async version of record_outcome - offloads SQLite to thread pool.

        Use this from async code (like orchestrator.stream_chat) to avoid
        blocking the event loop during SQLite operations.

        Args:
            learner_name: Name of learner to update
            outcome: Outcome data
            vertical: Which vertical this came from (coding, devops, data_science)
        """
        await asyncio.to_thread(self.record_outcome, learner_name, outcome, vertical)

    async def get_recommendation_async(
        self,
        learner_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[RLRecommendation]:
        """Async version of get_recommendation - offloads SQLite to thread pool.

        Use this from async code to avoid blocking the event loop.

        Args:
            learner_name: Name of learner to query
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Recommendation with value and confidence, or None if no data
        """
        return await asyncio.to_thread(
            self.get_recommendation, learner_name, provider, model, task_type
        )

    async def get_all_recommendations_async(
        self, provider: str, model: str, task_type: str
    ) -> Dict[str, RLRecommendation]:
        """Async version of get_all_recommendations.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Dictionary mapping learner name to recommendation
        """
        return await asyncio.to_thread(self.get_all_recommendations, provider, model, task_type)

    async def export_metrics_async(self) -> Dict[str, Any]:
        """Async version of export_metrics.

        Returns:
            Dictionary with metrics from all learners
        """
        return await asyncio.to_thread(self.export_metrics)

    async def get_stats_async(self) -> Dict[str, Any]:
        """Async version of get_stats.

        Returns:
            Dictionary with coordinator statistics
        """
        return await asyncio.to_thread(self.get_stats)

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
        cursor.execute(f"SELECT COUNT(*) FROM {Tables.RL_OUTCOME}")
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

    def list_learners(self) -> list[str]:
        """List all registered learner names.

        Returns:
            List of learner names
        """
        return list(self._learners.keys())

    def export_all_learner_data(self) -> Dict[str, Any]:
        """Export all learner data for training/analysis.

        Returns:
            Dictionary mapping learner name to its exported data
        """
        data = {}
        for name, learner in self._learners.items():
            try:
                if hasattr(learner, "export_data"):
                    data[name] = learner.export_data()
                elif hasattr(learner, "get_q_table"):
                    data[name] = {"q_table": learner.get_q_table()}
                elif hasattr(learner, "export_metrics"):
                    data[name] = learner.export_metrics()
                else:
                    data[name] = {"type": type(learner).__name__}
            except Exception as e:
                logger.error(f"RL: Failed to export data for {name}: {e}")
                data[name] = {"error": str(e)}
        return data

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator-level statistics.

        Returns:
            Dictionary with coordinator statistics
        """
        # Count outcomes from database
        cursor = self.db.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {Tables.RL_OUTCOME}")
            total_outcomes = cursor.fetchone()[0]
        except Exception:
            total_outcomes = 0

        # Gather learner sample counts
        learner_samples = {}
        for name, learner in self._learners.items():
            try:
                if hasattr(learner, "get_sample_count"):
                    learner_samples[name] = learner.get_sample_count()
                elif hasattr(learner, "_sample_count"):
                    learner_samples[name] = getattr(learner, "_sample_count", 0)
                else:
                    learner_samples[name] = 0
            except Exception:
                learner_samples[name] = 0

        return {
            "total_outcomes": total_outcomes,
            "learner_count": len(self._learners),
            "learners": list(self._learners.keys()),
            "learner_samples": learner_samples,
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()
            logger.debug("RL: Database connection closed")


# Global singleton
_rl_coordinator: Optional[RLCoordinator] = None


def get_rl_coordinator() -> RLCoordinator:
    """Get global RL coordinator (lazy init).

    For sync callers. Use get_rl_coordinator_async() from async code.

    Returns:
        Global coordinator singleton
    """
    global _rl_coordinator
    if _rl_coordinator is None:
        _rl_coordinator = RLCoordinator()
    return _rl_coordinator


async def get_rl_coordinator_async() -> RLCoordinator:
    """Get global RL coordinator with async-safe initialization.

    Use this from async code to avoid blocking the event loop during
    coordinator initialization (which involves SQLite table creation).

    Returns:
        Global coordinator singleton
    """
    global _rl_coordinator
    if _rl_coordinator is None:
        # Initialize in thread to avoid blocking event loop
        _rl_coordinator = await asyncio.to_thread(RLCoordinator)
    return _rl_coordinator
