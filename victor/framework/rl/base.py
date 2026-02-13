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

"""Base classes and data models for RL framework."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.protocols.provider_adapter import IProviderAdapter

logger = logging.getLogger(__name__)


@dataclass
class RLOutcome:
    """Base class for RL outcome records.

    This is recorded after every session/operation to provide feedback
    for learning optimal parameter values.

    Attributes:
        provider: Provider name (e.g., "anthropic", "deepseek")
        model: Model name (e.g., "claude-3-opus", "deepseek-chat")
        task_type: Task type ("analysis", "action", "default")
        success: Whether the operation completed successfully
        quality_score: Quality score 0.0-1.0 from grounding/user feedback
        timestamp: ISO timestamp of outcome
        metadata: Domain-specific metadata (varies by learner)
        vertical: Which vertical this came from (coding, devops, data_science)
    """

    provider: str
    model: str
    task_type: str
    success: bool
    quality_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    vertical: str = "coding"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "provider": self.provider,
            "model": self.model,
            "task_type": self.task_type,
            "success": self.success,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp,
            "metadata": json.dumps(self.metadata),
            "vertical": self.vertical,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLOutcome":
        """Create from dictionary."""
        metadata = data.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return cls(
            provider=data["provider"],
            model=data["model"],
            task_type=data["task_type"],
            success=data["success"],
            quality_score=data["quality_score"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=metadata,
            vertical=data.get("vertical", "coding"),
        )


@dataclass
class RLRecommendation:
    """Base class for RL recommendations.

    Returned when querying a learner for optimal parameter value.

    Attributes:
        value: The recommended value (type varies by learner)
        confidence: Confidence in recommendation 0.0-1.0
        reason: Human-readable explanation of recommendation
        sample_size: Number of outcomes used to compute recommendation
        is_baseline: Whether this is a provider baseline (no learning yet)
    """

    value: Any
    confidence: float
    reason: str
    sample_size: int
    is_baseline: bool = False

    def __str__(self) -> str:
        """Human-readable representation."""
        baseline_str = " (baseline)" if self.is_baseline else ""
        return (
            f"RLRecommendation(value={self.value}, "
            f"confidence={self.confidence:.2f}, "
            f"samples={self.sample_size}{baseline_str})"
        )


class BaseLearner(ABC):
    """Abstract base class for all RL learners.

    Provides common infrastructure:
    - SQLite database access (via coordinator)
    - Provider adapter integration for baselines
    - Telemetry collection
    - Statistics tracking

    Subclasses implement domain-specific learning logic.
    """

    def __init__(
        self,
        name: str,
        db_connection: Any,  # sqlite3.Connection
        learning_rate: float = 0.1,
        provider_adapter: Optional["IProviderAdapter"] = None,
    ):
        """Initialize base learner.

        Args:
            name: Learner name (e.g., "continuation_patience")
            db_connection: SQLite database connection from coordinator
            learning_rate: How aggressively to adjust values (0.0-1.0)
            provider_adapter: Optional provider adapter for baselines
        """
        self.name = name
        self.db = db_connection
        self.learning_rate = learning_rate
        self.provider_adapter = provider_adapter

        # Ensure tables exist
        self._ensure_tables()

    @abstractmethod
    def _ensure_tables(self) -> None:
        """Ensure required database tables exist.

        Each learner creates its own tables for storing stats and outcomes.
        Table names should be prefixed with learner name to avoid conflicts.
        """
        pass

    @abstractmethod
    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record an outcome and update learned values.

        Args:
            outcome: Outcome data with provider, model, success, quality, etc.
        """
        pass

    @abstractmethod
    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommendation for given context.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Recommendation with value and confidence, or None if no data
        """
        pass

    @abstractmethod
    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (typically -1.0 to 1.0)
        """
        pass

    def _get_provider_baseline(self, param_name: str) -> Any:
        """Get baseline value from provider adapter if available.

        Args:
            param_name: Parameter name (e.g., "continuation_patience")

        Returns:
            Baseline value from provider capabilities, or None
        """
        if not self.provider_adapter:
            return None

        return getattr(self.provider_adapter.capabilities, param_name, None)

    def _get_context_key(self, provider: str, model: str, task_type: str) -> str:
        """Generate context key for storage.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Key string like "anthropic:claude-3-opus:analysis"
        """
        return f"{provider}:{model}:{task_type}"

    def export_metrics(self) -> Dict[str, Any]:
        """Export learner metrics for monitoring.

        Returns:
            Dictionary with learner stats and recommendations
        """
        cursor = self.db.cursor()

        # Get table name (each learner has its own stats table)
        stats_table = f"{self.name}_stats"

        try:
            cursor.execute(f"SELECT COUNT(*) FROM {stats_table}")
            total_contexts = cursor.fetchone()[0]

            cursor.execute(f"""
                SELECT SUM(total_sessions) FROM {stats_table}
                """)
            total_sessions = cursor.fetchone()[0] or 0

            return {
                "learner": self.name,
                "total_contexts": total_contexts,
                "total_sessions": total_sessions,
            }
        except Exception as e:
            logger.warning(f"RL: Failed to export metrics for {self.name}: {e}")
            return {"learner": self.name, "error": str(e)}
