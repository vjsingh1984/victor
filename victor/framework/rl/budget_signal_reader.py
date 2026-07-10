"""Read-only persistence adapter for RL-driven budget calibration.

Phase 2 of the calibration pipeline: load :class:`ToolBudgetSignals` and
:class:`DecisionOutcomeAggregate` from the *global* Victor database
(``~/.victor/victor.db``) using the existing data-access idiom shared by the
rest of the RL layer (``db.cursor()`` + ``Tables.*`` constants + ``dict(row)``).

Reuse-first design (no new SQLite path):
* The reader never writes and never owns a connection.
* ``db`` is injected for testability and lazily defaults to the canonical
  :func:`victor.core.database.get_database` accessor — the same idiom used in
  ``victor/framework/rl/learners/tool_selector.py``.
* Missing tables, absent columns, or empty results degrade to empty/cold-start
  signals (matching the try/except guards used elsewhere in the RL layer).

Schemas consumed (from ``victor/core/schema.py``):

    rl_tool_q        : tool_name, q_value, selection_count, success_count
    decision_outcome : success, quality_score, attributed_reward
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from victor.core.schema import Tables
from victor.framework.rl.budget_calibration import (
    DecisionOutcomeAggregate,
    ToolBudgetSignals,
)

logger = logging.getLogger(__name__)


class BudgetSignalReader:
    """Read aggregate RL signals for budget calibration from the global DB."""

    def __init__(self, db: Optional[Any] = None):
        """Initialize the reader.

        Args:
            db: Optional DB handle exposing ``cursor()``. When ``None`` the
                canonical ``get_database()`` accessor is used lazily on first
                read, keeping construction side-effect-free and testable.
        """
        self._db = db

    # ------------------------------------------------------------------
    # DB resolution
    # ------------------------------------------------------------------

    def _resolve_db(self, db: Optional[Any]) -> Any:
        """Return the provided db or lazily fetch the canonical global DB."""
        if db is not None:
            return db
        if self._db is not None:
            return self._db
        # Lazy import avoids a module-load dependency on the DB subsystem.
        from victor.core.database import get_database

        return get_database()

    # ------------------------------------------------------------------
    # Tool signals (rl_tool_q)
    # ------------------------------------------------------------------

    def load_tool_signals(self, db: Optional[Any] = None) -> Tuple[ToolBudgetSignals, ...]:
        """Load per-tool Q-value signals from ``rl_tool_q``.

        Returns an empty tuple when the table is absent/empty (cold-start safe).
        Q-values are clamped into [0, 1] to defend against legacy/default rows.
        """
        try:
            handle = self._resolve_db(db)
            cursor = handle.cursor()
            cursor.execute(
                f"SELECT tool_name, q_value, selection_count, success_count "
                f"FROM {Tables.RL_TOOL_Q}"
            )
            rows = [dict(r) for r in cursor.fetchall()]
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.debug(f"budget_calibration: rl_tool_q load failed: {exc}")
            return ()

        signals: list[ToolBudgetSignals] = []
        for row in rows:
            tool_name = row.get("tool_name")
            if not tool_name:
                continue
            selection = int(row.get("selection_count") or 0)
            success = int(row.get("success_count") or 0)
            q = float(row.get("q_value") or 0.0)
            q = max(0.0, min(1.0, q))  # clamp defensively
            rate = (success / selection) if selection > 0 else 0.0
            signals.append(
                ToolBudgetSignals(
                    tool_name=tool_name,
                    q_value=q,
                    selection_count=selection,
                    success_rate=rate,
                )
            )
        return tuple(signals)

    # ------------------------------------------------------------------
    # Decision outcomes (decision_outcome)
    # ------------------------------------------------------------------

    def load_decision_outcomes(self, db: Optional[Any] = None) -> DecisionOutcomeAggregate:
        """Aggregate trajectory outcomes from ``decision_outcome``.

        Returns a zeroed (cold-start) aggregate when the table is absent/empty.
        ``success`` NULLs are treated as failures; NULL rewards as 0.0.
        """
        try:
            handle = self._resolve_db(db)
            cursor = handle.cursor()
            cursor.execute(
                f"SELECT success, quality_score, attributed_reward FROM {Tables.DECISION_OUTCOME}"
            )
            rows = [dict(r) for r in cursor.fetchall()]
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.debug(f"budget_calibration: decision_outcome load failed: {exc}")
            return DecisionOutcomeAggregate(total=0, successes=0, failures=0, mean_reward=0.0)

        total = len(rows)
        if total == 0:
            return DecisionOutcomeAggregate(total=0, successes=0, failures=0, mean_reward=0.0)

        successes = sum(1 for r in rows if _truthy(r.get("success")))
        failures = total - successes
        rewards = [float(r.get("attributed_reward") or 0.0) for r in rows]
        mean_reward = sum(rewards) / total
        return DecisionOutcomeAggregate(
            total=total,
            successes=successes,
            failures=failures,
            mean_reward=mean_reward,
        )


def _truthy(value: Any) -> bool:
    """Coerce a possibly-NULL ``success`` flag into a boolean."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value >= 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False
