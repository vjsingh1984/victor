#!/usr/bin/env python3
"""Benchmark startup KPIs for Victor import and Agent.create().

Measures:
- `import victor` cold + warm timings
- `Agent.create()` cold + warm timings

Usage:
    python scripts/benchmark_startup_kpi.py
    python scripts/benchmark_startup_kpi.py --iterations 10 --json
    python scripts/benchmark_startup_kpi.py --provider openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return float(ordered[idx])


@dataclass
class _TimingSummary:
    cold_ms: float
    warm_samples_ms: List[float]

    @property
    def warm_mean_ms(self) -> float:
        return statistics.fmean(self.warm_samples_ms) if self.warm_samples_ms else 0.0

    @property
    def warm_p95_ms(self) -> float:
        return _percentile(self.warm_samples_ms, 0.95)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cold_ms": self.cold_ms,
            "warm_samples_ms": self.warm_samples_ms,
            "warm_mean_ms": self.warm_mean_ms,
            "warm_p95_ms": self.warm_p95_ms,
        }


def _run_snippet(
    *,
    python_executable: str,
    code: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """Run an isolated Python snippet and parse JSON payload from stdout."""
    result = subprocess.run(
        [python_executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"benchmark subprocess failed: {details}")

    line = (result.stdout or "").strip().splitlines()
    if not line:
        raise RuntimeError("benchmark subprocess returned no output")
    try:
        return json.loads(line[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid benchmark payload: {line[-1]}") from exc


def _measure_import_victor(
    *,
    python_executable: str,
    iterations: int,
    timeout_seconds: float,
) -> _TimingSummary:
    code = textwrap.dedent(
        f"""
        import importlib
        import json
        import time

        warm_iters = {iterations}
        t0 = time.perf_counter()
        import victor  # noqa: F401
        cold_ms = (time.perf_counter() - t0) * 1000.0

        warm = []
        for _ in range(warm_iters):
            t = time.perf_counter()
            importlib.import_module("victor")
            warm.append((time.perf_counter() - t) * 1000.0)

        print(json.dumps({{"cold_ms": cold_ms, "warm_ms": warm}}))
        """
    )
    payload = _run_snippet(
        python_executable=python_executable,
        code=code,
        timeout_seconds=timeout_seconds,
    )
    return _TimingSummary(
        cold_ms=float(payload["cold_ms"]),
        warm_samples_ms=[float(v) for v in payload["warm_ms"]],
    )


def _measure_agent_create(
    *,
    python_executable: str,
    iterations: int,
    provider: str,
    model: str,
    timeout_seconds: float,
) -> _TimingSummary:
    provider_literal = json.dumps(provider)
    model_literal = json.dumps(model)
    code = textwrap.dedent(
        f"""
        import asyncio
        import json
        import time

        from victor.framework import Agent

        warm_iters = {iterations}
        provider = {provider_literal}
        model = {model_literal}

        async def run():
            t0 = time.perf_counter()
            agent = await Agent.create(
                provider=provider,
                model=model,
                enable_observability=False,
            )
            cold_ms = (time.perf_counter() - t0) * 1000.0
            await agent.close()

            warm = []
            for _ in range(warm_iters):
                t = time.perf_counter()
                agent = await Agent.create(
                    provider=provider,
                    model=model,
                    enable_observability=False,
                )
                warm.append((time.perf_counter() - t) * 1000.0)
                await agent.close()

            print(json.dumps({{"cold_ms": cold_ms, "warm_ms": warm}}))

        asyncio.run(run())
        """
    )
    payload = _run_snippet(
        python_executable=python_executable,
        code=code,
        timeout_seconds=timeout_seconds,
    )
    return _TimingSummary(
        cold_ms=float(payload["cold_ms"]),
        warm_samples_ms=[float(v) for v in payload["warm_ms"]],
    )


def _build_report(
    *,
    python_executable: str,
    iterations: int,
    provider: str,
    model: str,
    import_timing: _TimingSummary,
    create_timing: _TimingSummary,
) -> Dict[str, Any]:
    return {
        "python": python_executable,
        "iterations": iterations,
        "agent_create": {
            "provider": provider,
            "model": model,
            **create_timing.to_dict(),
        },
        "import_victor": import_timing.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark startup KPIs for import victor + Agent.create()."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of warm iterations for each measurement (default: 5).",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        help="Provider used for Agent.create() baseline (default: ollama).",
    )
    parser.add_argument(
        "--model",
        default="qwen3-coder:30b",
        help="Model used for Agent.create() baseline (default: qwen3-coder:30b).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for subprocess probes (default: current interpreter).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Per-probe timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON report only.",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be >= 1")

    import_timing = _measure_import_victor(
        python_executable=args.python,
        iterations=args.iterations,
        timeout_seconds=args.timeout_seconds,
    )
    create_timing = _measure_agent_create(
        python_executable=args.python,
        iterations=args.iterations,
        provider=args.provider,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )

    report = _build_report(
        python_executable=args.python,
        iterations=args.iterations,
        provider=args.provider,
        model=args.model,
        import_timing=import_timing,
        create_timing=create_timing,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("Startup KPI Benchmark")
    print(f"python: {report['python']}")
    print(f"iterations: {report['iterations']}")
    print("")
    print("import victor")
    print(f"  cold_ms: {report['import_victor']['cold_ms']:.2f}")
    print(f"  warm_mean_ms: {report['import_victor']['warm_mean_ms']:.2f}")
    print(f"  warm_p95_ms: {report['import_victor']['warm_p95_ms']:.2f}")
    print("")
    print("Agent.create()")
    print(f"  provider/model: {args.provider}/{args.model}")
    print(f"  cold_ms: {report['agent_create']['cold_ms']:.2f}")
    print(f"  warm_mean_ms: {report['agent_create']['warm_mean_ms']:.2f}")
    print(f"  warm_p95_ms: {report['agent_create']['warm_p95_ms']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
