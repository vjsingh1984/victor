#!/usr/bin/env python
"""Dry-run inspector for RL-driven tool-budget calibration.

Reads the *existing* global Victor database (``~/.victor/victor.db``) and prints
what the calibration pipeline would recommend — **without applying anything**.
This is the evidence artifact referenced by FEP-0002 / ADR-017: run it against
real RL signal before deciding to opt in or wire the feature.

It reuses the exact same pure pipeline the runtime seam uses:

    BudgetSignalReader -> BudgetCalibrator.recommend()

so "inspect" and "apply" share one code path (no parallel harness).

Usage::

    python scripts/calibration_inspect.py            # current baseline budget
    python scripts/calibration_inspect.py --budget 800
    python scripts/calibration_inspect.py --min-confidence 0.7
    python scripts/calibration_inspect.py --json       # machine-readable

Exit code is 0 unless the calibration modules cannot be imported or the global
DB cannot be opened. It never writes to the database.

NOTE: This script intentionally does NOT read the ``tool_budget_calibration_*``
settings — it always computes a recommendation so you can evaluate the signal
independently of whether calibration is enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _baseline_tool_settings(budget: int) -> Any:
    """Return a default ``ToolSettings`` with the requested budget."""
    from victor.config.tool_settings import ToolSettings

    ts = ToolSettings()
    # ``tool_call_budget`` is the one field the calibrator reads off the
    # baseline; mirror it so the recommendation is relative to a known base.
    ts.tool_call_budget = budget
    return ts


def _run(
    budget: int, min_confidence: float, min_tool_samples: int, min_decision_total: int
) -> dict[str, Any]:
    """Execute the read -> calibrate path and return a result dict."""
    from victor.config.settings import Settings
    from victor.framework.rl.budget_calibration import (
        BudgetCalibrationConfig,
        BudgetCalibrator,
    )
    from victor.framework.rl.budget_signal_reader import BudgetSignalReader

    # Resolve a baseline ToolSettings off the canonical Settings() so the
    # recommendation reflects the real default budget when --budget is omitted.
    settings = Settings()
    baseline = _baseline_tool_settings(budget if budget > 0 else settings.tools.tool_call_budget)

    reader = BudgetSignalReader()
    tools = reader.load_tool_signals()
    decisions = reader.load_decision_outcomes()

    config = BudgetCalibrationConfig(
        min_tool_samples=min_tool_samples,
        min_decision_total=min_decision_total,
    )
    recommendation = BudgetCalibrator(config=config).recommend(
        tools=tools,
        decisions=decisions,
        baseline=baseline,
    )

    reliable_tool_count = sum(1 for t in tools if t.selection_count >= max(min_tool_samples, 1))
    return {
        "baseline_tool_call_budget": baseline.tool_call_budget,
        "recommended_tool_call_budget": recommendation.recommended_tool_call_budget,
        "confidence": recommendation.confidence,
        "rationale": recommendation.rationale,
        "min_confidence_gate": min_confidence,
        "would_apply": recommendation.confidence >= min_confidence
        and recommendation.confidence > 0.0,
        "signals": {
            "tool_signal_rows": len(tools),
            "reliable_tool_rows": reliable_tool_count,
            "decision_total": decisions.total,
            "decision_successes": decisions.successes,
            "decision_failures": decisions.failures,
            "decision_mean_reward": round(decisions.mean_reward, 4),
            "decision_success_rate": round(decisions.success_rate, 4),
            "decision_failure_rate": round(decisions.failure_rate, 4),
        },
        "recommendation_detail": {
            "recommended_relief_amount": recommendation.recommended_relief_amount,
            "recommended_relief_max_uses": recommendation.recommended_relief_max_uses,
            "early_stop_q_threshold": recommendation.early_stop_q_threshold,
            "relief_eligible_tools": list(recommendation.relief_eligible_tools),
        },
    }


def _print_human(result: dict[str, Any]) -> None:
    s = result["signals"]
    d = result["recommendation_detail"]
    print("=" * 60)
    print("RL Tool-Budget Calibration — INSPECT (no apply)")
    print("=" * 60)
    print(f"  baseline tool_call_budget : {result['baseline_tool_call_budget']}")
    print(f"  recommended budget        : {result['recommended_tool_call_budget']}")
    print(f"  confidence                : {result['confidence']:.3f}")
    print(f"  rationale                 : {result['rationale']}")
    print("-" * 60)
    print(
        f"  would_apply @ min_conf={result['min_confidence_gate']:.2f}?"
        f"  {result['would_apply']}"
    )
    print("-" * 60)
    print("  signals (from global DB):")
    print(
        f"    rl_tool_q rows          : {s['tool_signal_rows']} "
        f"(reliable: {s['reliable_tool_rows']})"
    )
    print(
        f"    decision_outcome total  : {s['decision_total']} "
        f"(succ {s['decision_successes']} / fail {s['decision_failures']})"
    )
    print(f"    success_rate            : {s['decision_success_rate']:.2%}")
    print(f"    failure_rate            : {s['decision_failure_rate']:.2%}")
    print(f"    mean_reward             : {s['decision_mean_reward']}")
    print("-" * 60)
    print("  recommendation detail:")
    print(f"    relief_amount           : {d['recommended_relief_amount']}")
    print(f"    relief_max_uses         : {d['recommended_relief_max_uses']}")
    print(f"    early_stop_q_threshold  : {d['early_stop_q_threshold']}")
    print(f"    relief_eligible_tools   : {d['relief_eligible_tools'] or '(none)'}")
    print("=" * 60)
    if not result["would_apply"]:
        print(
            "NOTE: confidence below gate — calibration would retain the "
            "baseline (identity-return). Nothing changes at runtime."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run inspector for RL tool-budget calibration.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Baseline tool_call_budget to calibrate against (default: real "
        "Settings().tools.tool_call_budget).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Confidence gate used to report would_apply (default 0.5, the " "seam default).",
    )
    parser.add_argument(
        "--min-tool-samples",
        type=int,
        default=20,
        help="Reliability gate for rl_tool_q rows (default 20).",
    )
    parser.add_argument(
        "--min-decision-total",
        type=int,
        default=50,
        help="Reliability gate for decision_outcome rows (default 50).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human report.",
    )
    args = parser.parse_args(argv)

    try:
        result = _run(
            budget=args.budget,
            min_confidence=args.min_confidence,
            min_tool_samples=args.min_tool_samples,
            min_decision_total=args.min_decision_total,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean failure
        print(f"calibration_inspect: failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
