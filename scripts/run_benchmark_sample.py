#!/usr/bin/env python3
"""Run a sample of HumanEval tasks to verify benchmark framework.

This script runs a small sample of HumanEval tasks to demonstrate
the evaluation harness with real data and real test execution.

Usage:
    python scripts/run_benchmark_sample.py --profile claude-haiku --tasks 3
    python scripts/run_benchmark_sample.py --profile default --tasks 3
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.evaluation import (
    CodeQualityAnalyzer,
    HumanEvalRunner,
    TaskStatus,
    pass_at_k,
)


async def evaluate_solution(task, solution: str, analyzer: CodeQualityAnalyzer) -> dict:
    """Evaluate a solution by running actual tests."""
    import tempfile
    import subprocess

    result = {
        "task_id": task.task_id,
        "solution": solution,
        "passed": False,
        "error": None,
        "stdout": "",
        "stderr": "",
        "code_quality": None,
        "execution_time_ms": 0,
    }

    # Analyze code quality
    try:
        metrics = await analyzer.analyze(solution, language="python")
        result["code_quality"] = {
            "syntax_valid": metrics.syntax_valid,
            "type_coverage": metrics.type_coverage,
            "complexity": metrics.cyclomatic_complexity,
            "overall_score": metrics.get_overall_score(),
        }
    except Exception as e:
        result["code_quality"] = {"error": str(e)}

    # Create test file
    full_code = solution + "\n\n" + task.test_code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        start = time.time()
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            timeout=30,
        )
        result["execution_time_ms"] = int((time.time() - start) * 1000)
        result["stdout"] = proc.stdout.decode()[:500]
        result["stderr"] = proc.stderr.decode()[:500]
        result["passed"] = proc.returncode == 0
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout (30s)"
    except Exception as e:
        result["error"] = str(e)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return result


async def run_with_canonical_solutions(num_tasks: int = 3):
    """Run benchmark using canonical solutions to verify the framework works."""
    print("=" * 70)
    print("VICTOR BENCHMARK VERIFICATION")
    print("Using canonical solutions to verify framework functionality")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Tasks to run: {num_tasks}")
    print()

    # Load tasks
    runner = HumanEvalRunner()
    from victor.evaluation import EvaluationConfig, BenchmarkType

    config = EvaluationConfig(
        benchmark=BenchmarkType.HUMAN_EVAL,
        model="verification",
        max_tasks=num_tasks,
    )

    print("Loading HumanEval dataset from HuggingFace...")
    tasks = await runner.load_tasks(config)
    print(f"Loaded {len(tasks)} tasks")
    print()

    # Initialize analyzer
    analyzer = CodeQualityAnalyzer(use_ruff=False, use_radon=False)

    # Run evaluation
    results = []
    passed_count = 0

    print("Running evaluations with canonical (ground-truth) solutions:")
    print("-" * 70)

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task.task_id}")

        # Use the canonical solution
        solution = task.prompt + task.solution

        result = await evaluate_solution(task, solution, analyzer)
        results.append(result)

        if result["passed"]:
            passed_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        quality = result.get("code_quality", {})
        quality_score = quality.get("overall_score", 0) if quality else 0

        print(f"   Status: {status}")
        print(f"   Quality Score: {quality_score:.1f}/100")
        print(f"   Execution Time: {result['execution_time_ms']}ms")

        if result["error"]:
            print(f"   Error: {result['error']}")
        if result["stderr"] and not result["passed"]:
            print(f"   Stderr: {result['stderr'][:100]}...")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tasks:    {len(tasks)}")
    print(f"Passed:         {passed_count}")
    print(f"Failed:         {len(tasks) - passed_count}")
    print(f"Pass Rate:      {passed_count/len(tasks):.1%}")
    print()

    # Pass@k calculations
    print("Pass@k (based on canonical solutions):")
    for k in [1, 5, 10]:
        if k <= len(tasks):
            score = pass_at_k(len(tasks), passed_count, k)
            print(f"  Pass@{k}: {score:.2%}")

    # Average quality
    quality_scores = [
        r["code_quality"]["overall_score"]
        for r in results
        if r.get("code_quality") and "overall_score" in r["code_quality"]
    ]
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        print(f"\nAverage Code Quality: {avg_quality:.1f}/100")

    print()
    print("=" * 70)
    print("Note: These results use CANONICAL solutions from HumanEval dataset.")
    print("This verifies the evaluation framework works correctly.")
    print("To test Victor's actual code generation, use --mode=agent")
    print("=" * 70)

    return {
        "timestamp": datetime.now().isoformat(),
        "num_tasks": len(tasks),
        "passed": passed_count,
        "pass_rate": passed_count / len(tasks),
        "results": results,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark sample")
    parser.add_argument("--tasks", type=int, default=5, help="Number of tasks to run")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    results = await run_with_canonical_solutions(args.tasks)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
