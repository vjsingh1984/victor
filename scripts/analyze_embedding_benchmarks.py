#!/usr/bin/env python
"""
Generate performance analysis reports for embedding operations benchmarks.

This script analyzes benchmark results and generates detailed comparison reports
with visualizations and recommendations.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


class BenchmarkAnalyzer:
    """Analyze and report on embedding operation benchmarks."""

    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.results = self._load_results()

    def _load_results(self) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        if not self.results_path.exists():
            raise FileNotFoundError(f"Benchmark results not found: {self.results_path}")

        with open(self.results_path, "r") as f:
            return json.load(f)

    def extract_metrics(self) -> Dict[str, Dict[str, float]]:
        """Extract key metrics from benchmark results."""
        metrics = {}

        for benchmark_name, benchmark_data in self.results.get("benchmarks", {}).items():
            # Extract key statistics
            metrics[benchmark_name] = {
                "mean": benchmark_data.get("stats", {}).get("mean", 0),
                "stddev": benchmark_data.get("stats", {}).get("stddev", 0),
                "min": benchmark_data.get("stats", {}).get("min", 0),
                "max": benchmark_data.get("stats", {}).get("max", 0),
                "rounds": benchmark_data.get("stats", {}).get("rounds", 0),
            }

        return metrics

    def generate_comparison_table(self) -> str:
        """Generate Markdown comparison table."""
        metrics = self.extract_metrics()

        lines = [
            "# Embedding Operations Performance Comparison",
            "",
            "## Benchmark Results Summary",
            "",
            "| Benchmark | Mean (μs) | StdDev (μs) | Min (μs) | Max (μs) | Throughput |",
            "|-----------|-----------|-------------|----------|----------|-------------|",
        ]

        for name, stats in sorted(metrics.items()):
            # Calculate throughput if applicable
            operations_per_sec = 1_000_000 / stats["mean"] if stats["mean"] > 0 else 0

            lines.append(
                f"| {name} | {stats['mean']:.2f} | {stats['stddev']:.2f} | "
                f"{stats['min']:.2f} | {stats['max']:.2f} | {operations_per_sec:,.0f} ops/s |"
            )

        return "\n".join(lines)

    def analyze_algorithm_comparison(self) -> str:
        """Analyze and compare different algorithms."""
        lines = [
            "",
            "## Algorithm Comparison Analysis",
            "",
        ]

        # Compare sorting vs argpartition
        sorting_key = None
        partition_key = None

        for key in self.results.get("benchmarks", {}).keys():
            if "sorting" in key and "argpartition" not in key:
                sorting_key = key
            elif "argpartition" in key:
                partition_key = key

        if sorting_key and partition_key:
            sorting_time = self.results["benchmarks"][sorting_key]["stats"]["mean"]
            partition_time = self.results["benchmarks"][partition_key]["stats"]["mean"]
            speedup = sorting_time / partition_time if partition_time > 0 else 0

            lines.extend(
                [
                    f"### Sorting vs Argpartition",
                    f"- **Sorting (O(n log n))**: {sorting_time:.2f} μs",
                    f"- **Argpartition (O(n))**: {partition_time:.2f} μs",
                    f"- **Speedup**: {speedup:.2f}x",
                    f"- **Recommendation**: {'Use argpartition' if speedup > 1 else 'Use sorting'}",
                    "",
                ]
            )

        # Compare loop vs vectorized
        loop_key = None
        vectorized_key = None

        for key in self.results.get("benchmarks", {}).keys():
            if "loop" in key and "vectorized" not in key:
                loop_key = key
            elif "vectorized" in key:
                vectorized_key = key

        if loop_key and vectorized_key:
            loop_time = self.results["benchmarks"][loop_key]["stats"]["mean"]
            vectorized_time = self.results["benchmarks"][vectorized_key]["stats"]["mean"]
            speedup = loop_time / vectorized_time if vectorized_time > 0 else 0

            lines.extend(
                [
                    f"### Loop vs Vectorized Operations",
                    f"- **Loop-based**: {loop_time:.2f} μs",
                    f"- **Vectorized**: {vectorized_time:.2f} μs",
                    f"- **Speedup**: {speedup:.2f}x",
                    f"- **Recommendation**: Always use vectorized operations",
                    "",
                ]
            )

        return "\n".join(lines)

    def generate_recommendations(self) -> str:
        """Generate performance optimization recommendations."""
        lines = [
            "",
            "## Performance Optimization Recommendations",
            "",
        ]

        # Analyze cache performance
        cache_hit = None
        cache_miss = None

        for key, data in self.results.get("benchmarks", {}).items():
            if "cache_hit" in key:
                cache_hit = data["stats"]["mean"]
            elif "cache_miss" in key:
                cache_miss = data["stats"]["mean"]

        if cache_hit and cache_miss:
            speedup = cache_miss / cache_hit if cache_hit > 0 else 0
            lines.extend(
                [
                    f"### Cache Optimization",
                    f"- **Cache hit**: {cache_hit:.2f} μs",
                    f"- **Cache miss**: {cache_miss:.2f} μs",
                    f"- **Speedup**: {speedup:.2f}x",
                    f"- **Recommendation**: Implement aggressive caching for repeated queries",
                    "",
                ]
            )

        # Analyze batch size scaling
        batch_sizes = {}
        for key, data in self.results.get("benchmarks", {}).items():
            if "small_batch" in key:
                batch_sizes["small"] = data["stats"]["mean"]
            elif "medium_batch" in key:
                batch_sizes["medium"] = data["stats"]["mean"]
            elif "large_batch" in key:
                batch_sizes["large"] = data["stats"]["mean"]

        if batch_sizes:
            lines.extend(
                [
                    "### Batch Size Recommendations",
                ]
            )
            for size, time in sorted(batch_sizes.items()):
                lines.append(f"- **{size.capitalize()}**: {time:.2f} μs")
            lines.append("")

        # General recommendations
        lines.extend(
            [
                "### General Best Practices",
                "",
                "1. **Use Vectorized Operations**: Always prefer NumPy vectorized operations over loops",
                "2. **Pre-normalize Embeddings**: Normalize once during indexing, not at query time",
                "3. **Use Argpartition for Top-K**: O(n) vs O(n log n) for large datasets",
                "4. **Implement Caching**: Cache frequently accessed embeddings and results",
                "5. **Batch Processing**: Process multiple queries together when possible",
                "6. **Memory Efficiency**: Use float32 instead of float64 for embeddings",
                "",
                "### Implementation Priorities",
                "",
                "1. **High Priority**: Vectorization, pre-normalization",
                "2. **Medium Priority**: Top-k optimization, caching",
                "3. **Low Priority**: Micro-optimizations, memory alignment",
                "",
            ]
        )

        return "\n".join(lines)

    def generate_performance_profile(self) -> str:
        """Generate performance profile for different use cases."""
        lines = [
            "",
            "## Performance Profiles by Use Case",
            "",
            "### Real-time Search (< 10ms latency)",
            "- **Batch Size**: 1-10 queries",
            "- **Corpus Size**: < 10K embeddings",
            "- **Algorithm**: Pre-normalized + vectorized + argpartition",
            "- **Caching**: Aggressive (80%+ hit rate)",
            "",
            "### Batch Processing (100ms-1s latency)",
            "- **Batch Size**: 100-1000 queries",
            "- **Corpus Size**: 10K-100K embeddings",
            "- **Algorithm**: Matrix multiplication + batch top-k",
            "- **Caching**: Moderate (40-60% hit rate)",
            "",
            "### Offline Indexing (> 1s acceptable)",
            "- **Batch Size**: 1000+ queries",
            "- **Corpus Size**: 100K+ embeddings",
            "- **Algorithm**: Full matrix operations",
            "- **Caching**: Minimal (write-through only)",
            "",
        ]

        return "\n".join(lines)

    def generate_full_report(self, output_path: str = None) -> str:
        """Generate complete analysis report."""
        report_parts = [
            self.generate_comparison_table(),
            self.analyze_algorithm_comparison(),
            self.generate_recommendations(),
            self.generate_performance_profile(),
            "",
            "## Running the Benchmarks",
            "",
            "```bash",
            "# Run all benchmarks",
            "pytest tests/benchmark/benchmarks/test_embedding_operations_baseline.py -v",
            "",
            "# Save results",
            "pytest tests/benchmark/benchmarks/test_embedding_operations_baseline.py --benchmark-save=baseline",
            "",
            "# Compare with previous run",
            "pytest tests/benchmark/benchmarks/test_embedding_operations_baseline.py --benchmark-compare=baseline",
            "",
            "# Generate histogram",
            "pytest tests/benchmark/benchmarks/test_embedding_operations_baseline.py --benchmark-histogram",
            "```",
            "",
            "---",
            "",
            "*Report generated by analyze_embedding_benchmarks.py*",
        ]

        report = "\n".join(report_parts)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Report saved to: {output_path}")

        return report


def run_benchmarks(output_file: str = None) -> bool:
    """Run the embedding benchmarks and save results."""
    cmd = [
        "pytest",
        "tests/benchmark/benchmarks/test_embedding_operations_baseline.py",
        "-v",
        "--benchmark-only",
        "--benchmark-sort=name",
    ]

    if output_file:
        cmd.extend([f"--benchmark-save={output_file}"])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmarks: {e}", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Analyze embedding operation benchmarks")
    parser.add_argument("action", choices=["run", "analyze", "report"], help="Action to perform")
    parser.add_argument(
        "--results",
        default=".benchmarks/embedding_operations_baseline.json",
        help="Path to benchmark results JSON file",
    )
    parser.add_argument("--output", help="Output path for generated report")
    parser.add_argument("--save", help="Name to save benchmark results")

    args = parser.parse_args()

    if args.action == "run":
        success = run_benchmarks(args.save)
        if not success:
            sys.exit(1)

    elif args.action == "analyze":
        analyzer = BenchmarkAnalyzer(args.results)
        report = analyzer.generate_full_report(args.output)
        print(report)

    elif args.action == "report":
        # Generate and print report from benchmark file
        analyzer = BenchmarkAnalyzer(args.results)
        report = analyzer.generate_full_report(args.output)
        print(report)


if __name__ == "__main__":
    main()
