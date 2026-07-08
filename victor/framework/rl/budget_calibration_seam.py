"""RL budget-calibration application seam.

Phase 3: a single pure entry point that composes the reader -> calibrator ->
overlay pipeline and is gated by settings (default OFF = unchanged behavior).

This is the only function a session-bootstrap site needs to call. It never
mutates the baseline; when gating rejects calibration it returns the baseline
``ToolSettings`` by identity (so callers can cheaply detect "no change").

Gating contract
---------------
1. ``tool_budget_calibration_enabled`` is False (default) -> return baseline
   unchanged (identity).
2. Enabled but the calibrator's confidence is below
   ``tool_budget_calibration_min_confidence`` -> return baseline unchanged.
3. Enabled with sufficient confidence -> return a *new* ``ToolSettings`` with
   the calibrated overlay applied.

Safety
------
Any error in the reader/calibrator is caught and the baseline is returned,
so calibration can never break session bootstrap. Per CLAUDE.md, calibration
is opt-in (default False) and low-confidence recommendations always retain
the baseline.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from victor.config.settings import Settings
from victor.config.tool_settings import ToolSettings
from victor.framework.rl.budget_calibration import BudgetCalibrator
from victor.framework.rl.budget_signal_reader import BudgetSignalReader

logger = logging.getLogger(__name__)


def apply_budget_calibration(
    settings: Settings,
    db: Optional[Any] = None,
    explicit_override: bool = False,
) -> ToolSettings:
    """Apply RL-driven budget calibration to the session's tool settings.

    Args:
        settings: The active ``Settings``; only ``settings.tools`` is read.
        db: Optional DB handle exposing ``cursor()``. When ``None`` the
            reader lazily resolves the canonical global DB. Injected for
            testability.
        explicit_override: When ``True``, the caller has already applied an
            explicit (e.g. CLI ``--tool-budget``) override via
            ``SessionConfig.apply_to_settings``. Per FEP-0002 review decision
            #2, explicit overrides win absolutely, so the seam short-circuits
            to identity-return without reading the DB. Default ``False``.

    Returns:
        The (possibly calibrated) ``ToolSettings``. When calibration is
        disabled, low-confidence, or an explicit override is present, the
        baseline ``settings.tools`` is returned by identity. Never mutates the
        input.
    """
    baseline = settings.tools
    if baseline is None:
        return baseline  # nothing to calibrate

    # Gate 0 (precedence): an explicit override wins absolutely. Short-circuit
    # before any DB read so calibration can never re-tighten a user-set budget.
    if explicit_override:
        return baseline

    # Gate 1: opt-in toggle (default OFF -> zero behavior change).
    if not getattr(baseline, "tool_budget_calibration_enabled", False):
        return baseline

    min_conf = float(
        getattr(baseline, "tool_budget_calibration_min_confidence", 0.7)
    )

    try:
        reader = BudgetSignalReader()
        tools = reader.load_tool_signals(db=db)
        decisions = reader.load_decision_outcomes(db=db)
        rec = BudgetCalibrator().recommend(
            tools=tools, decisions=decisions, baseline=baseline
        )
    except Exception as exc:  # noqa: BLE001 - never break bootstrap
        logger.debug(f"budget_calibration: pipeline failed, retaining baseline: {exc}")
        return baseline

    # Gate 2: confidence threshold. A confidence of 0.0 means cold-start / no
    # usable signal (incl. DB errors the reader degraded) -> always retain the
    # baseline by identity; otherwise require confidence >= min_conf.
    if rec.confidence <= 0.0 or rec.confidence < min_conf:
        logger.debug(
            f"budget_calibration: confidence {rec.confidence:.2f} below threshold "
            f"{min_conf:.2f}; retaining baseline"
        )
        return baseline

    logger.info(
        f"budget_calibration: applying calibrated overlay "
        f"(confidence={rec.confidence:.2f}, budget={rec.recommended_tool_call_budget}, "
        f"rationale='{rec.rationale}')"
    )
    return rec.apply_to_settings(baseline)
