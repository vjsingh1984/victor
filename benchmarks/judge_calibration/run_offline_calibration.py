#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Offline judge-calibration demo (EVR-2, ADR-011) — runs with zero LLM calls.

Runs the repo-local verifiable corpus through a scripted executor (which solves most tasks but
fakes some with "completion-without-effect" claims) and calibrates two heuristic judges against
the programmatic gold labels:

- **credulous**: trusts the agent's claim of completion — the failure mode ADR-011 exists to
  catch; expected to FAIL the α ≥ 0.7 gate (claims are always positive, gold is not).
- **evidence**: requires observable tool activity in the transcript before believing a
  completion claim; expected to PASS on this corpus.

To calibrate a *real* judge (rubric/LLM), implement the same
``(prompt, transcript, workspace) -> float`` callable around it and pass it to
``JudgeCalibrationHarness.run`` — the blinding contract and the report format are identical.
To generate trajectories with a *real* agent instead of the scripted executor, use
``victor.evaluation.calibration_agent_executor.make_agent_executor`` with a
``VictorAgentAdapter.from_profile(...)`` (an Ollama profile keeps the loop fully offline).

Usage:
    python benchmarks/judge_calibration/run_offline_calibration.py [--variants 8] [--out DIR]

    # Calibrate the LLM rubric judge via a Victor profile (provider/model/key from
    # profiles.yaml). If the key lives in the system keyring, bridge it into env first —
    # keyring is skipped without a TTY, but env always resolves:
    eval "$(victor auth env -p zai)"
    python benchmarks/judge_calibration/run_offline_calibration.py \
        --variants 8 --judge-profile zai-coding   # e.g. zai / glm-5.2 (1M context)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from victor.evaluation.calibration_corpus import default_corpus
from victor.evaluation.calibration_rubric_judge import make_rubric_judge
from victor.evaluation.judge_calibration_harness import (
    JudgeCalibrationHarness,
    Transcript,
    alternating_scripted_executor,
)


def credulous_judge(_prompt: str, transcript: Transcript, _workspace: Path) -> float:
    """Believes any completion claim in the final message."""
    final = transcript.final_message.lower()
    return 1.0 if ("done" in final or "complete" in final) else 0.0


def evidence_judge(_prompt: str, transcript: Transcript, _workspace: Path) -> float:
    """Scores from evidence of work (tool activity), not from what the agent claims."""
    return 1.0 if transcript.tool_steps() else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", type=int, default=8, help="Variants per task family (6 families)"
    )
    parser.add_argument("--out", type=Path, default=Path("benchmarks/judge_calibration/reports"))
    parser.add_argument(
        "--llm-judge-provider",
        default=None,
        help="Also calibrate the real ADR-009 LLM rubric judge via this provider "
        "(e.g. 'ollama' — keeps the run fully offline). One grading call per task.",
    )
    parser.add_argument(
        "--llm-judge-model",
        default="qwen2.5-coder:7b",
        help="Model for --llm-judge-provider grading calls",
    )
    parser.add_argument(
        "--llm-judge-base-url",
        default=None,
        help="Provider base URL override (e.g. http://<windows-host>:11434 from WSL)",
    )
    parser.add_argument(
        "--judge-profile",
        default=None,
        help="Resolve the LLM judge's provider/model/api-key from a Victor profile "
        "(profiles.yaml), e.g. --judge-profile cerebras-glm. Overrides the "
        "--llm-judge-provider/--llm-judge-model pair.",
    )
    parser.add_argument(
        "--judge-delay",
        type=float,
        default=2.0,
        help="Seconds to pace between LLM grading calls (strict provider RPM caps, "
        "e.g. Z.AI, need 15+; local Ollama can use 0)",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=1024,
        help="Token budget per grading call. Thinking-family judge models spend tokens "
        "on reasoning before the grade lines — if the integrity line reports "
        "ungradable>0, raise this (2048+).",
    )
    args = parser.parse_args()

    judges = {
        "credulous": credulous_judge,
        "evidence": evidence_judge,
        # The real ADR-009 no-LLM fallback path (HeuristicRubricJudge behind the
        # DimensionAwareFilter) — calibrated as shipped, binary completion verdicts.
        "rubric-heuristic": make_rubric_judge(),
    }
    llm_stats = None
    if args.judge_profile or args.llm_judge_provider:
        from victor.evaluation.calibration_rubric_judge import (
            JudgeCallStats,
            make_llm_rubric_judge,
            make_provider_complete_fn,
        )
        from victor.providers.registry import ProviderRegistry

        provider_name = args.llm_judge_provider
        model = args.llm_judge_model
        provider_kwargs = {}
        if args.llm_judge_base_url:
            provider_kwargs["base_url"] = args.llm_judge_base_url
        if args.judge_profile:
            import os

            from victor.config.settings import load_settings

            profiles = load_settings().load_profiles()
            if args.judge_profile not in profiles:
                parser.error(
                    f"profile '{args.judge_profile}' not found. "
                    f"Available: {', '.join(sorted(profiles))}"
                )
            profile = profiles[args.judge_profile]
            provider_name, model = profile.provider, profile.model
            # Pass through profile extras; ${ENV} references resolve like from_profile.
            api_key = getattr(profile, "api_key", None)
            if api_key and api_key.startswith("${") and api_key.endswith("}"):
                api_key = os.environ.get(api_key[2:-1])
            if api_key:
                provider_kwargs["api_key"] = api_key
            profile_base_url = getattr(profile, "base_url", None)
            if profile_base_url and not args.llm_judge_base_url:
                provider_kwargs["base_url"] = profile_base_url
        provider = ProviderRegistry.create(provider_name, **provider_kwargs)
        llm_stats = JudgeCallStats()
        judges["rubric-llm"] = make_llm_rubric_judge(
            make_provider_complete_fn(
                provider,
                model,
                max_tokens=args.judge_max_tokens,
                pre_call_delay_seconds=args.judge_delay,
                stats=llm_stats,
            ),
            stats=llm_stats,
        )
    exit_code = 0
    for name, judge in judges.items():
        harness = JudgeCalibrationHarness(default_corpus(variants=args.variants))
        # period=5 is coprime with the 6 task families, so scripted failures rotate
        # across every family instead of always hitting the same ones.
        report = harness.run(alternating_scripted_executor(period=5), judge)
        report.save(args.out / f"{name}.json")
        decision = report.gate_decision
        verdict = "TRUSTED" if decision.trusted else "NOT TRUSTED"
        if name == "rubric-llm" and llm_stats is not None:
            print(f"[rubric-llm] grading-call integrity: {llm_stats.summary()}")
            if not llm_stats.clean:
                verdict = "VOID"
                exit_code = 1
                if llm_stats.failures:
                    print(
                        f"[rubric-llm] ⚠ {llm_stats.failures} grading call(s) failed after "
                        "retries — failed calls degrade to neutral fallback scores, so this α "
                        "measures the error path, NOT the judge. Re-run with a higher "
                        "--judge-delay or a less rate-limited provider."
                    )
                if llm_stats.ungradable:
                    print(
                        f"[rubric-llm] ⚠ {llm_stats.ungradable} grading call(s) returned "
                        "output with NO parseable dimension scores — wrong output format, or a "
                        "thinking-family model exhausting its token budget on reasoning before "
                        "the grade lines. This α measures the parse-fallback path, NOT the "
                        "judge. Try --judge-max-tokens 2048+ or a non-thinking judge model."
                    )
        print(f"[{name}] n={len(report.samples)}  {verdict}  ({decision.reason})")
        for family, rel in report.per_family.items():
            alpha = rel.krippendorff_alpha
            shown = "n/a" if alpha is None or alpha != alpha else f"{alpha:.3f}"
            print(f"    {family:<12} n={rel.n:<3} α={shown}")
    print(f"\nReports written to {args.out}/")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
