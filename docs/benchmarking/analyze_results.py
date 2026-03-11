#!/usr/bin/env python3
"""Statistical Significance Analysis for Competitive Benchmarks.

Analyzes benchmark results across frameworks, computing per-task metrics,
confidence intervals, variance flags, and weighted composite scores per
the scoring methodology defined in scoring-methodology.md.

Usage:
    # Analyze all frameworks with results
    python docs/benchmarking/analyze_results.py

    # Analyze specific frameworks
    python docs/benchmarking/analyze_results.py --frameworks victor langgraph crewai

    # Output JSON report
    python docs/benchmarking/analyze_results.py --output report.json

    # Verbose mode
    python docs/benchmarking/analyze_results.py -v
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "docs" / "benchmarking" / "results"

# Scoring dimension weights from scoring-methodology.md
DIMENSION_WEIGHTS = {
    "task_success_rate": 0.40,
    "output_quality": 0.20,
    "execution_speed": 0.10,
    "resource_efficiency": 0.15,
    "reliability": 0.10,
    "developer_experience": 0.05,
}

# High-variance threshold: std dev > 20% of mean
HIGH_VARIANCE_THRESHOLD = 0.20

# Confidence level for CI (1.96 for 95%)
Z_SCORE_95 = 1.96


def load_results(framework: str) -> List[Dict[str, Any]]:
    """Load all benchmark result files for a framework.

    Args:
        framework: Framework name (e.g., "victor")

    Returns:
        List of result dictionaries from all run files.
    """
    framework_dir = RESULTS_DIR / framework
    if not framework_dir.exists():
        return []

    results = []
    for result_file in sorted(framework_dir.glob("*.json")):
        with open(result_file) as f:
            data = json.load(f)
        # Handle both single-run and multi-run result formats
        if "results" in data and isinstance(data["results"], list):
            for r in data["results"]:
                r.setdefault("framework", framework)
                results.append(r)
        else:
            data.setdefault("framework", framework)
            results.append(data)

    return results


def group_by_task(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by task_id.

    Args:
        results: Flat list of result records.

    Returns:
        Dictionary mapping task_id to list of run results.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        task_id = r.get("task_id", "unknown")
        grouped.setdefault(task_id, []).append(r)
    return grouped


def compute_mean(values: List[float]) -> float:
    """Compute arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_std_dev(values: List[float], mean: Optional[float] = None) -> float:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    if mean is None:
        mean = compute_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_confidence_interval(
    mean: float, std_dev: float, n: int
) -> Tuple[float, float]:
    """Compute 95% confidence interval.

    Args:
        mean: Sample mean.
        std_dev: Sample standard deviation.
        n: Sample size.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if n < 2:
        return (mean, mean)
    margin = Z_SCORE_95 * (std_dev / math.sqrt(n))
    return (mean - margin, mean + margin)


def is_high_variance(std_dev: float, mean: float) -> bool:
    """Check if std dev exceeds 20% of mean (high-variance flag)."""
    if mean == 0:
        return std_dev > 0
    return (std_dev / abs(mean)) > HIGH_VARIANCE_THRESHOLD


def compute_task_metrics(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute per-task metrics across multiple runs.

    Args:
        runs: List of run results for a single task.

    Returns:
        Dictionary with mean, std_dev, ci_lower, ci_upper, high_variance flag,
        and per-metric breakdowns.
    """
    n = len(runs)
    durations = [r.get("duration_ms", 0) for r in runs]
    qualities = [r.get("output_quality", 0) for r in runs if r.get("output_quality") is not None]
    memories = [r.get("memory_mb", 0) for r in runs]
    successes = sum(1 for r in runs if r.get("success", False))

    dur_mean = compute_mean(durations)
    dur_std = compute_std_dev(durations, dur_mean)
    dur_ci = compute_confidence_interval(dur_mean, dur_std, n)

    qual_mean = compute_mean(qualities) if qualities else 0.0
    qual_std = compute_std_dev(qualities, qual_mean) if qualities else 0.0
    qual_ci = compute_confidence_interval(qual_mean, qual_std, len(qualities)) if qualities else (0.0, 0.0)

    mem_mean = compute_mean(memories)
    mem_std = compute_std_dev(memories, mem_mean)

    success_rate = successes / n if n > 0 else 0.0

    return {
        "runs": n,
        "success_rate": round(success_rate, 4),
        "duration_ms": {
            "mean": round(dur_mean, 2),
            "std_dev": round(dur_std, 2),
            "ci_95_lower": round(dur_ci[0], 2),
            "ci_95_upper": round(dur_ci[1], 2),
            "high_variance": is_high_variance(dur_std, dur_mean),
        },
        "output_quality": {
            "mean": round(qual_mean, 2),
            "std_dev": round(qual_std, 2),
            "ci_95_lower": round(qual_ci[0], 2),
            "ci_95_upper": round(qual_ci[1], 2),
            "high_variance": is_high_variance(qual_std, qual_mean),
        },
        "memory_mb": {
            "mean": round(mem_mean, 2),
            "std_dev": round(mem_std, 2),
            "high_variance": is_high_variance(mem_std, mem_mean),
        },
    }


def compute_composite_score(
    task_metrics: Dict[str, Dict[str, Any]],
    all_framework_metrics: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    framework_name: str = "",
    dx_score: float = 3.0,
) -> Dict[str, Any]:
    """Compute weighted composite score across all tasks for a framework.

    Uses the 6-dimension scoring from scoring-methodology.md:
    1. Task Success Rate (40%)
    2. Output Quality (20%)
    3. Execution Speed (10%)
    4. Resource Efficiency (15%)
    5. Reliability (10%)
    6. Developer Experience (5%)

    Args:
        task_metrics: Per-task metrics for this framework.
        all_framework_metrics: Metrics for all frameworks (for relative scoring).
        framework_name: Name of this framework.
        dx_score: Developer experience score (1-5, default 3.0).

    Returns:
        Dictionary with dimension scores and overall composite.
    """
    if not task_metrics:
        return {"overall": 0.0, "dimensions": {}}

    # 1. Task Success Rate (40%)
    success_rates = [m["success_rate"] for m in task_metrics.values()]
    avg_success = compute_mean(success_rates)
    success_score = avg_success * 40

    # 2. Output Quality (20%)
    quality_means = [
        m["output_quality"]["mean"]
        for m in task_metrics.values()
        if m["output_quality"]["mean"] > 0
    ]
    avg_quality = compute_mean(quality_means) if quality_means else 0.0
    quality_score = (avg_quality / 5.0) * 20

    # 3. Execution Speed (10%) — relative to slowest framework
    dur_means = [m["duration_ms"]["mean"] for m in task_metrics.values()]
    median_duration = sorted(dur_means)[len(dur_means) // 2] if dur_means else 0

    slowest_median = median_duration  # default: self is slowest
    if all_framework_metrics:
        for fw, fw_metrics in all_framework_metrics.items():
            fw_durs = [m["duration_ms"]["mean"] for m in fw_metrics.values()]
            if fw_durs:
                fw_median = sorted(fw_durs)[len(fw_durs) // 2]
                slowest_median = max(slowest_median, fw_median)

    if slowest_median > 0:
        speed_score = max(0, 1 - (median_duration / slowest_median)) * 10
    else:
        speed_score = 0.0

    # 4. Resource Efficiency (15%) — relative to worst
    mem_means = [m["memory_mb"]["mean"] for m in task_metrics.values()]
    avg_memory = compute_mean(mem_means) if mem_means else 0.0

    highest_memory = avg_memory
    if all_framework_metrics:
        for fw, fw_metrics in all_framework_metrics.items():
            fw_mems = [m["memory_mb"]["mean"] for m in fw_metrics.values()]
            if fw_mems:
                fw_avg_mem = compute_mean(fw_mems)
                highest_memory = max(highest_memory, fw_avg_mem)

    if highest_memory > 0:
        resource_score = max(0, 1 - (avg_memory / highest_memory)) * 15
    else:
        resource_score = 0.0

    # 5. Reliability (10%) — based on error rate across all runs
    total_runs = sum(m["runs"] for m in task_metrics.values())
    total_successes = sum(
        m["success_rate"] * m["runs"] for m in task_metrics.values()
    )
    error_rate = 1 - (total_successes / total_runs) if total_runs > 0 else 1.0
    reliability_score = (1 - error_rate) * 10

    # 6. Developer Experience (5%) — externally provided score
    dx_normalized = (dx_score / 5.0) * 5

    overall = (
        success_score
        + quality_score
        + speed_score
        + resource_score
        + reliability_score
        + dx_normalized
    )

    return {
        "overall": round(overall, 2),
        "dimensions": {
            "task_success_rate": round(success_score, 2),
            "output_quality": round(quality_score, 2),
            "execution_speed": round(speed_score, 2),
            "resource_efficiency": round(resource_score, 2),
            "reliability": round(reliability_score, 2),
            "developer_experience": round(dx_normalized, 2),
        },
    }


def find_statistical_ties(
    framework_scores: Dict[str, Dict[str, Any]],
    framework_task_metrics: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Identify frameworks with overlapping confidence intervals.

    Per scoring-methodology.md: frameworks with overlapping CIs are
    considered statistically tied.

    Args:
        framework_scores: Composite scores per framework.
        framework_task_metrics: Per-task metrics per framework.

    Returns:
        List of tie descriptions.
    """
    ties = []
    frameworks = list(framework_scores.keys())

    for i in range(len(frameworks)):
        for j in range(i + 1, len(frameworks)):
            fw_a, fw_b = frameworks[i], frameworks[j]
            metrics_a = framework_task_metrics[fw_a]
            metrics_b = framework_task_metrics[fw_b]

            # Compute overall score CI from per-task quality CIs
            scores_a = [m["output_quality"]["mean"] for m in metrics_a.values() if m["output_quality"]["mean"] > 0]
            scores_b = [m["output_quality"]["mean"] for m in metrics_b.values() if m["output_quality"]["mean"] > 0]

            if not scores_a or not scores_b:
                continue

            mean_a = compute_mean(scores_a)
            std_a = compute_std_dev(scores_a, mean_a)
            ci_a = compute_confidence_interval(mean_a, std_a, len(scores_a))

            mean_b = compute_mean(scores_b)
            std_b = compute_std_dev(scores_b, mean_b)
            ci_b = compute_confidence_interval(mean_b, std_b, len(scores_b))

            # Check overlap
            if ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]:
                ties.append({
                    "frameworks": [fw_a, fw_b],
                    "dimension": "output_quality",
                    "ci_a": [round(ci_a[0], 2), round(ci_a[1], 2)],
                    "ci_b": [round(ci_b[0], 2), round(ci_b[1], 2)],
                    "verdict": "statistically_tied",
                })

            # Also check duration CIs
            durs_a = [m["duration_ms"]["mean"] for m in metrics_a.values()]
            durs_b = [m["duration_ms"]["mean"] for m in metrics_b.values()]

            if durs_a and durs_b:
                dur_mean_a = compute_mean(durs_a)
                dur_std_a = compute_std_dev(durs_a, dur_mean_a)
                dur_ci_a = compute_confidence_interval(dur_mean_a, dur_std_a, len(durs_a))

                dur_mean_b = compute_mean(durs_b)
                dur_std_b = compute_std_dev(durs_b, dur_mean_b)
                dur_ci_b = compute_confidence_interval(dur_mean_b, dur_std_b, len(durs_b))

                if dur_ci_a[0] <= dur_ci_b[1] and dur_ci_b[0] <= dur_ci_a[1]:
                    ties.append({
                        "frameworks": [fw_a, fw_b],
                        "dimension": "execution_speed",
                        "ci_a": [round(dur_ci_a[0], 2), round(dur_ci_a[1], 2)],
                        "ci_b": [round(dur_ci_b[0], 2), round(dur_ci_b[1], 2)],
                        "verdict": "statistically_tied",
                    })

    return ties


def collect_high_variance_flags(
    framework_task_metrics: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[Dict[str, str]]:
    """Collect all high-variance flags across frameworks and tasks.

    Args:
        framework_task_metrics: Per-task metrics per framework.

    Returns:
        List of high-variance flag records.
    """
    flags = []
    for fw, tasks in framework_task_metrics.items():
        for task_id, metrics in tasks.items():
            for metric_name in ("duration_ms", "output_quality", "memory_mb"):
                metric = metrics.get(metric_name, {})
                if metric.get("high_variance", False):
                    flags.append({
                        "framework": fw,
                        "task_id": task_id,
                        "metric": metric_name,
                        "mean": metric.get("mean", 0),
                        "std_dev": metric.get("std_dev", 0),
                    })
    return flags


def print_summary_report(
    framework_scores: Dict[str, Dict[str, Any]],
    framework_task_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    ties: List[Dict[str, Any]],
    high_variance: List[Dict[str, str]],
    verbose: bool = False,
) -> None:
    """Print formatted summary report to console.

    Args:
        framework_scores: Composite scores per framework.
        framework_task_metrics: Per-task metrics per framework.
        ties: Statistical tie records.
        high_variance: High-variance flag records.
        verbose: Whether to show detailed per-task breakdown.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK ANALYSIS REPORT")
    print("=" * 70)

    # Overall scores table
    print("\n## Overall Composite Scores\n")
    print(f"{'Framework':<15} {'Overall':>8} {'Success':>8} {'Quality':>8} "
          f"{'Speed':>8} {'Resource':>8} {'Reliab.':>8} {'DX':>8}")
    print("-" * 83)

    for fw in sorted(framework_scores, key=lambda x: framework_scores[x]["overall"], reverse=True):
        s = framework_scores[fw]
        d = s["dimensions"]
        print(
            f"{fw:<15} {s['overall']:>8.1f} "
            f"{d['task_success_rate']:>8.1f} "
            f"{d['output_quality']:>8.1f} "
            f"{d['execution_speed']:>8.1f} "
            f"{d['resource_efficiency']:>8.1f} "
            f"{d['reliability']:>8.1f} "
            f"{d['developer_experience']:>8.1f}"
        )

    # High-variance warnings
    if high_variance:
        print(f"\n## High-Variance Warnings ({len(high_variance)} flags)\n")
        for flag in high_variance:
            ratio = (flag["std_dev"] / flag["mean"] * 100) if flag["mean"] else 0
            print(
                f"  ! {flag['framework']}/{flag['task_id']} "
                f"{flag['metric']}: std_dev={flag['std_dev']:.1f} "
                f"({ratio:.0f}% of mean={flag['mean']:.1f})"
            )

    # Statistical ties
    if ties:
        print(f"\n## Statistical Ties ({len(ties)} found)\n")
        for tie in ties:
            print(
                f"  ~ {tie['frameworks'][0]} vs {tie['frameworks'][1]} "
                f"on {tie['dimension']}: "
                f"CI_A={tie['ci_a']}, CI_B={tie['ci_b']}"
            )

    # Per-task breakdown (verbose)
    if verbose:
        print("\n## Per-Task Breakdown\n")
        for fw in sorted(framework_task_metrics):
            print(f"\n### {fw}\n")
            print(f"  {'Task':<8} {'Runs':>5} {'Success':>8} "
                  f"{'Dur(ms)':>10} {'Quality':>8} {'Mem(MB)':>8}")
            print("  " + "-" * 55)
            for task_id in sorted(framework_task_metrics[fw]):
                m = framework_task_metrics[fw][task_id]
                dur = m["duration_ms"]["mean"]
                qual = m["output_quality"]["mean"]
                mem = m["memory_mb"]["mean"]
                hv = " !" if any(
                    m[k].get("high_variance", False)
                    for k in ("duration_ms", "output_quality", "memory_mb")
                ) else ""
                print(
                    f"  {task_id:<8} {m['runs']:>5} "
                    f"{m['success_rate']:>8.1%} "
                    f"{dur:>10.1f} {qual:>8.1f} {mem:>8.1f}{hv}"
                )

    # Minimum runs warning
    for fw, tasks in framework_task_metrics.items():
        low_run_tasks = [tid for tid, m in tasks.items() if m["runs"] < 3]
        if low_run_tasks:
            print(f"\n  Warning: {fw} has < 3 runs for tasks: {', '.join(low_run_tasks)}")
            print("  Minimum 3 runs per task required for reliable statistical analysis.")

    print("\n" + "=" * 70)


def analyze(
    frameworks: List[str],
    verbose: bool = False,
    dx_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run full statistical analysis across frameworks.

    Args:
        frameworks: List of framework names to analyze.
        verbose: Whether to print detailed output.
        dx_scores: Optional developer experience scores per framework (1-5).

    Returns:
        Complete analysis report as dictionary.
    """
    if dx_scores is None:
        dx_scores = {}

    framework_task_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    framework_scores: Dict[str, Dict[str, Any]] = {}

    # Phase 1: Load and compute per-task metrics for each framework
    for fw in frameworks:
        results = load_results(fw)
        if not results:
            print(f"  No results found for {fw}, skipping.")
            continue

        grouped = group_by_task(results)
        task_metrics = {}
        for task_id, runs in grouped.items():
            task_metrics[task_id] = compute_task_metrics(runs)

        framework_task_metrics[fw] = task_metrics

    if not framework_task_metrics:
        print("No results found for any framework.")
        return {"error": "No results found"}

    # Phase 2: Compute composite scores (needs all frameworks for relative metrics)
    for fw, task_metrics in framework_task_metrics.items():
        dx = dx_scores.get(fw, 3.0)
        framework_scores[fw] = compute_composite_score(
            task_metrics,
            all_framework_metrics=framework_task_metrics,
            framework_name=fw,
            dx_score=dx,
        )

    # Phase 3: Statistical comparisons
    ties = find_statistical_ties(framework_scores, framework_task_metrics)
    high_variance = collect_high_variance_flags(framework_task_metrics)

    # Phase 4: Report
    print_summary_report(
        framework_scores, framework_task_metrics, ties, high_variance, verbose
    )

    return {
        "frameworks": list(framework_task_metrics.keys()),
        "scores": framework_scores,
        "task_metrics": {
            fw: {tid: m for tid, m in tasks.items()}
            for fw, tasks in framework_task_metrics.items()
        },
        "statistical_ties": ties,
        "high_variance_flags": high_variance,
    }


def discover_frameworks() -> List[str]:
    """Discover frameworks that have result directories."""
    if not RESULTS_DIR.exists():
        return []
    return [
        d.name
        for d in sorted(RESULTS_DIR.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze competitive benchmark results with statistical significance"
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=None,
        help="Framework names to analyze (default: auto-discover from results/)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save JSON report to file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-task breakdown",
    )

    args = parser.parse_args()

    frameworks = args.frameworks or discover_frameworks()
    if not frameworks:
        print("No frameworks specified and no results found in", RESULTS_DIR)
        print("\nUsage: python docs/benchmarking/analyze_results.py --frameworks victor langgraph crewai")
        sys.exit(1)

    print(f"Analyzing frameworks: {', '.join(frameworks)}")

    report = analyze(frameworks, verbose=args.verbose)

    if args.output and "error" not in report:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to: {output_path}")


if __name__ == "__main__":
    main()
