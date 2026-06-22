# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Completion-strategy A/B harness for ADR-009 (enhanced vs rubric vs hybrid).

Runs a task battery via the real agent under each ``completion_strategy`` and reports the metrics
that matter for the keep/rubric/hybrid decision:
- **completion**: success rate;
- **efficiency / restatement proxy**: mean turns (FEP-0007's defect was under-scoring → extra
  retry turns restating the answer, so fewer turns for the same answer = less restatement);
- **variance**: turn-count spread.

The strategy is selected per run via the ``VICTOR_COMPLETION_STRATEGY`` env override (read at
AgenticLoop construction). Decision rule (match-or-beat): a strategy replaces ``enhanced`` only if it
keeps completion ≥ while reducing (or matching) turns + variance. Pair with the judge α-gate
(`rubric_judge_eval.py`) — only flip to a rubric path whose LLM judge clears α≥0.7.

Usage (needs a real model; authenticate first):
    eval "$(victor auth env -p zai 2>/dev/null | grep '^export')"
    python scripts/experiments/completion_ab.py --profile zai-coding --strategies enhanced,rubric,hybrid --runs 1
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

# Multi-turn, tool-driven coding tasks (shared with temperature_ab.py) — these exercise the
# under-scoring/restatement failure mode that single-turn QA cannot: enhanced may over-run, rubric/
# hybrid should stop once the relevant code is read.
try:
    from scripts.experiments.coding_tasks import MULTI_TURN_CODING_TASKS as TASKS
except ImportError:  # allow running as `python scripts/experiments/completion_ab.py`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from coding_tasks import MULTI_TURN_CODING_TASKS as TASKS


@dataclass
class RunMetrics:
    turns: int
    tool_calls: int
    success: bool
    answer_len: int


@dataclass
class StrategyResult:
    strategy: str
    runs: List[RunMetrics] = field(default_factory=list)

    def summary(self) -> dict:
        turns = [r.turns for r in self.runs]
        return {
            "n": len(self.runs),
            "completion_rate": (
                (sum(r.success for r in self.runs) / len(self.runs)) if self.runs else 0.0
            ),
            "mean_turns": statistics.mean(turns) if turns else 0.0,
            "turns_stdev": statistics.pstdev(turns) if len(turns) > 1 else 0.0,
            "mean_tool_calls": (
                statistics.mean([r.tool_calls for r in self.runs]) if self.runs else 0.0
            ),
            "mean_answer_len": (
                statistics.mean([r.answer_len for r in self.runs]) if self.runs else 0.0
            ),
        }


async def _run_one(task: str, profile: Optional[str]) -> RunMetrics:
    from victor.framework.agent import Agent

    kwargs = {"temperature": 0.0}  # pin temperature so the strategy is the only variable
    if profile:
        kwargs["profile"] = profile
    agent = await Agent.create(**kwargs)
    setter = getattr(agent, "set_max_iterations", None)
    if setter:
        setter(12)
    try:
        result = await agent.run(task)
    finally:
        close = getattr(agent, "close", None) or getattr(agent, "shutdown", None)
        if close:
            try:
                await close()
            except Exception:
                pass
    md = result.metadata or {}
    turns = int(
        md.get("iterations_count")
        or md.get("iterations")
        or md.get("turn_count")
        or md.get("iteration_count")
        or 0
    )
    if turns == 0:
        turns = len(result.tool_calls) + 1
    return RunMetrics(
        turns=turns,
        tool_calls=len(result.tool_calls),
        success=bool(result.success),
        answer_len=len(result.content or ""),
    )


async def main(strategies: List[str], runs: int, profile: Optional[str]) -> None:
    results = {s: StrategyResult(strategy=s) for s in strategies}
    for s in strategies:
        os.environ["VICTOR_COMPLETION_STRATEGY"] = s  # read at AgenticLoop construction
        for task in TASKS:
            for _ in range(runs):
                try:
                    results[s].runs.append(await _run_one(task, profile))
                except Exception as exc:
                    print(f"[warn] strategy={s} task={task[:40]!r} failed: {exc!r}")
    os.environ.pop("VICTOR_COMPLETION_STRATEGY", None)

    print("\n============== Completion-strategy A/B (ADR-009) ==============")
    header = f"{'strategy':>10} | {'n':>3} | {'complete':>8} | {'mean_turns':>10} | {'turns_sd':>8} | {'ans_len':>7}"
    print(header)
    print("-" * len(header))
    summ = {}
    for s in strategies:
        d = results[s].summary()
        summ[s] = d
        print(
            f"{s:>10} | {d['n']:>3} | {d['completion_rate']:>8.2f} | {d['mean_turns']:>10.2f} | "
            f"{d['turns_stdev']:>8.2f} | {d['mean_answer_len']:>7.0f}"
        )

    if "enhanced" in summ:
        base = summ["enhanced"]
        print("\nDecision (vs enhanced baseline):")
        for s in strategies:
            if s == "enhanced":
                continue
            d = summ[s]
            beats = (
                d["completion_rate"] >= base["completion_rate"]
                and d["mean_turns"] <= base["mean_turns"] + 0.25
                and d["turns_stdev"] <= base["turns_stdev"] + 0.25
            )
            verb = "✅ match-or-beats" if beats else "⚠️  does NOT beat"
            print(
                f"  {s}: {verb} enhanced (complete {d['completion_rate']:.2f} vs "
                f"{base['completion_rate']:.2f}, turns {d['mean_turns']:.2f} vs {base['mean_turns']:.2f})"
            )
        print(
            "  NOTE: only flip to a rubric/hybrid path whose LLM judge clears α≥0.7 "
            "(rubric_judge_eval.py); quality should be eyeballed before flipping."
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategies", default="enhanced,rubric,hybrid")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--profile", default="zai-coding")
    args = ap.parse_args()
    asyncio.run(
        main([s.strip() for s in args.strategies.split(",")], args.runs, args.profile or None)
    )
