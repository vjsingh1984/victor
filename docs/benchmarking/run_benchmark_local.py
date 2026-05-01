#!/usr/bin/env python3
"""Local Benchmark Runner — executes Victor benchmark tasks against Ollama.

Runs a representative subset of tasks from the benchmark suite using
the local Ollama provider, measures quality/latency/resource metrics,
and saves results for analysis.

Usage:
    python docs/benchmarking/run_benchmark_local.py
    python docs/benchmarking/run_benchmark_local.py --model qwen3-coder-tools:30b
    python docs/benchmarking/run_benchmark_local.py --tasks C1 R4 T1 A1 W1
"""

import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Selected tasks: one per category for a quick representative run
DEFAULT_TASKS = ["C1", "C4", "R4", "T1", "A1", "A2", "W1"]

# Import task definitions from main benchmark script
from docs.benchmarking.run_benchmark import TASK_REGISTRY


async def run_victor_task(
    task_id: str,
    task_def: Dict[str, Any],
    provider: str,
    model: str,
    run_number: int = 1,
) -> Dict[str, Any]:
    """Execute a single benchmark task using Victor agent.

    Returns a result dict with timing, output, and resource metrics.
    """
    import psutil
    from victor.framework.agent import Agent

    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()
    output = ""
    error = None
    success = False

    try:
        agent = await Agent.create(
            provider=provider,
            model=model,
            temperature=0.7,
            max_tokens=task_def.get("max_tokens", 2048),
        )

        result = await asyncio.wait_for(
            agent.run(task_def["prompt"]),
            timeout=task_def.get("timeout_seconds", 300),
        )

        output = result.content or ""
        success = len(output.strip()) > 0
        await agent.close()

    except asyncio.TimeoutError:
        error = "Timeout"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    duration_ms = (time.time() - start_time) * 1000
    end_memory = process.memory_info().rss / 1024 / 1024
    memory_delta = end_memory - start_memory

    return {
        "task_id": task_id,
        "task_name": task_def["name"],
        "category": task_def["category"],
        "complexity": task_def["complexity"],
        "framework": "victor",
        "provider": provider,
        "model": model,
        "run": run_number,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "success": success,
        "duration_ms": round(duration_ms, 2),
        "output_length": len(output),
        "output_preview": output[:500] if output else "",
        "memory_mb": round(memory_delta, 2),
        "error": error,
    }


async def run_benchmark(
    task_ids: List[str],
    provider: str = "ollama",
    model: str = "qwen3-coder-tools:30b",
    runs_per_task: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run benchmark suite and return results."""

    all_results = []
    print(f"\n{'='*70}")
    print(f"VICTOR BENCHMARK — Local Execution")
    print(f"Provider: {provider} | Model: {model}")
    print(f"Tasks: {len(task_ids)} | Runs per task: {runs_per_task}")
    print(f"{'='*70}\n")

    for task_id in task_ids:
        if task_id not in TASK_REGISTRY:
            print(f"  [SKIP] {task_id}: not found in TASK_REGISTRY")
            continue

        task_def = TASK_REGISTRY[task_id]
        print(
            f"[{task_id}] {task_def['name']} ({task_def['category']} / {task_def['complexity']})"
        )

        for run_num in range(1, runs_per_task + 1):
            run_label = f"  Run {run_num}/{runs_per_task}" if runs_per_task > 1 else " "
            print(f"{run_label} ...", end="", flush=True)

            result = await run_victor_task(task_id, task_def, provider, model, run_num)
            all_results.append(result)

            status = "OK" if result["success"] else "FAIL"
            dur = result["duration_ms"] / 1000
            print(
                f" {status} in {dur:.1f}s"
                f" | {result['output_length']} chars"
                f" | mem {result['memory_mb']:+.1f}MB"
                + (f" | err: {result['error'][:60]}" if result["error"] else "")
            )

    # Summary
    successes = sum(1 for r in all_results if r["success"])
    total = len(all_results)
    durations = [r["duration_ms"] for r in all_results if r["success"]]
    avg_dur = sum(durations) / len(durations) if durations else 0

    summary = {
        "framework": "victor",
        "provider": provider,
        "model": model,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tasks_total": total,
        "tasks_successful": successes,
        "tasks_failed": total - successes,
        "success_rate": round(successes / total * 100, 1) if total else 0,
        "avg_duration_ms": round(avg_dur, 2),
        "results": all_results,
    }

    print(f"\n{'='*70}")
    print(f"RESULTS: {successes}/{total} passed ({summary['success_rate']}%)")
    print(f"Avg duration (successful): {avg_dur/1000:.1f}s")
    print(f"{'='*70}\n")

    return summary


def save_results(summary: Dict[str, Any]) -> str:
    """Save results to JSON file."""
    results_dir = PROJECT_ROOT / "docs" / "benchmarking" / "results" / "victor"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"benchmark_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {filepath}")
    return str(filepath)


def _setup_debug_logging() -> None:
    """Configure logging so benchmark debug traces land in ~/.victor/logs/victor.log."""
    import logging
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    log_dir = Path.home() / ".victor" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "victor.log"

    root = logging.getLogger("victor")
    root.setLevel(logging.DEBUG)

    # File handler — DEBUG level for full visibility
    fh = RotatingFileHandler(str(log_file), maxBytes=10_485_760, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s - benchmark - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Suppress noisy libraries
    for lib in (
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
        "hpack",
        "h2",
        "anthropic",
        "openai",
        "sentence_transformers",
        "transformers",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run local Victor benchmarks")
    parser.add_argument("--model", default="qwen3-coder-tools:30b", help="Ollama model")
    parser.add_argument("--provider", default="ollama", help="Provider name")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Task IDs")
    parser.add_argument("--runs", type=int, default=1, help="Runs per task")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        _setup_debug_logging()
        print("[DEBUG] Logging enabled → ~/.victor/logs/victor.log")

    summary = await run_benchmark(
        task_ids=args.tasks,
        provider=args.provider,
        model=args.model,
        runs_per_task=args.runs,
    )

    save_results(summary)

    # Print per-task detail
    print("\nPer-task results:")
    print(f"{'Task':<6} {'Name':<25} {'Status':<6} {'Duration':>10} {'Chars':>8}")
    print("-" * 60)
    for r in summary["results"]:
        status = "OK" if r["success"] else "FAIL"
        dur = f"{r['duration_ms']/1000:.1f}s"
        print(
            f"{r['task_id']:<6} {r['task_name']:<25} {status:<6} {dur:>10} {r['output_length']:>8}"
        )


if __name__ == "__main__":
    asyncio.run(main())
