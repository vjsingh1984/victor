#!/usr/bin/env python3
"""P4 Multi-Provider Excellence - Performance Benchmarking

This script benchmarks P4 features by running test queries and comparing results:
- Semantic-only vs Hybrid search
- With vs without query expansion
- With vs without deduplication
- Different threshold values

Usage:
    python scripts/p4_benchmark.py --queries queries.txt
    python scripts/p4_benchmark.py --queries queries.txt --profile p4_default
    python scripts/p4_benchmark.py --compare-configs config1.yaml config2.yaml
    python scripts/p4_benchmark.py --quick  # Run built-in benchmark queries
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark query."""

    query: str
    config_name: str
    results_count: int
    execution_time_ms: float
    top_5_files: List[str] = field(default_factory=list)
    avg_score: float = 0.0
    has_exact_match: bool = False
    search_mode: str = "unknown"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    enable_hybrid_search: bool = False
    hybrid_semantic_weight: float = 0.6
    hybrid_keyword_weight: float = 0.4
    semantic_threshold: float = 0.5
    query_expansion: bool = True
    max_expansions: int = 5


# Built-in benchmark queries for quick testing
BENCHMARK_QUERIES = [
    "tool registration",
    "error handling",
    "semantic search",
    "provider initialization",
    "conversation history",
    "file operations",
    "git integration",
    "database connection",
    "authentication logic",
    "configuration loading",
]


def load_queries(queries_file: Optional[str]) -> List[str]:
    """Load benchmark queries from file or use built-in."""
    if queries_file:
        with open(queries_file, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return BENCHMARK_QUERIES


async def run_query_benchmark(
    query: str, config: BenchmarkConfig, root_path: str = "."
) -> BenchmarkResult:
    """Run a single query with given configuration."""
    from victor.config.settings import Settings
    from victor_coding.codebase.indexer import CodebaseIndex

    # Create settings with benchmark config
    settings = Settings()
    settings.enable_hybrid_search = config.enable_hybrid_search
    settings.hybrid_search_semantic_weight = config.hybrid_semantic_weight
    settings.hybrid_search_keyword_weight = config.hybrid_keyword_weight
    settings.semantic_similarity_threshold = config.semantic_threshold
    settings.semantic_query_expansion_enabled = config.query_expansion
    settings.semantic_max_query_expansions = config.max_expansions

    # Create index
    index = CodebaseIndex(
        root_path=root_path,
        use_embeddings=True,
        embedding_config={
            "vector_store": "lancedb",
            "embedding_model_type": "sentence-transformers",
            "embedding_model_name": "all-MiniLM-L12-v2",
            "persist_directory": str(Path(root_path) / ".victor" / "embeddings"),
            "extra_config": {},
        },
    )

    # Ensure indexed
    await index.ensure_indexed(auto_reindex=False)

    # Run query with timing
    start_time = time.monotonic()

    results = await index.semantic_search(
        query=query,
        max_results=10,
        similarity_threshold=config.semantic_threshold,
        expand_query=config.query_expansion,
    )

    execution_time_ms = (time.monotonic() - start_time) * 1000

    # Extract result details
    top_5_files = [r.get("file_path", "") for r in results[:5]]
    avg_score = sum(r.get("score", 0.0) for r in results) / len(results) if results else 0.0
    search_mode = results[0].get("search_mode", "semantic") if results else "semantic"

    # Check for exact matches (high score)
    has_exact_match = any(r.get("score", 0.0) > 0.9 for r in results)

    return BenchmarkResult(
        query=query,
        config_name=config.name,
        results_count=len(results),
        execution_time_ms=execution_time_ms,
        top_5_files=top_5_files,
        avg_score=avg_score,
        has_exact_match=has_exact_match,
        search_mode=search_mode,
    )


async def run_benchmark_suite(
    queries: List[str], configs: List[BenchmarkConfig], root_path: str = "."
) -> Dict[str, List[BenchmarkResult]]:
    """Run full benchmark suite across all configs."""
    results_by_config: Dict[str, List[BenchmarkResult]] = {config.name: [] for config in configs}

    total_runs = len(queries) * len(configs)
    current_run = 0

    print(f"Running {total_runs} benchmark queries...")
    print()

    for config in configs:
        print(f"Testing configuration: {config.name}")

        for query in queries:
            current_run += 1
            print(f"  [{current_run}/{total_runs}] {query}...", end="", flush=True)

            try:
                result = await run_query_benchmark(query, config, root_path)
                results_by_config[config.name].append(result)
                print(f" {result.results_count} results in {result.execution_time_ms:.0f}ms")
            except Exception as e:
                print(f" ERROR: {e}")

        print()

    return results_by_config


def print_benchmark_report(
    results_by_config: Dict[str, List[BenchmarkResult]], configs: List[BenchmarkConfig]
):
    """Print comprehensive benchmark report."""
    print("=" * 80)
    print("P4 PERFORMANCE BENCHMARK REPORT")
    print("=" * 80)
    print()

    # Summary table
    print("CONFIGURATION SUMMARY")
    print("-" * 80)
    print(f"{'Config':<20} {'Queries':<8} {'Avg Time':<12} {'Avg Results':<12} {'Success %':<10}")
    print("-" * 80)

    for config in configs:
        results = results_by_config[config.name]
        if not results:
            continue

        avg_time = sum(r.execution_time_ms for r in results) / len(results)
        avg_results = sum(r.results_count for r in results) / len(results)
        success_rate = sum(1 for r in results if r.results_count > 0) / len(results)

        print(
            f"{config.name:<20} {len(results):<8} {avg_time:<12.0f} {avg_results:<12.1f} {success_rate:<10.1%}"
        )

    print()

    # Detailed comparison
    if len(configs) >= 2:
        print()
        print("DETAILED COMPARISON")
        print("-" * 80)

        # Compare first two configs
        config1 = configs[0]
        config2 = configs[1]
        results1 = results_by_config[config1.name]
        results2 = results_by_config[config2.name]

        print(f"Comparing: {config1.name} vs {config2.name}")
        print()

        # Per-query comparison
        print(f"{'Query':<30} {config1.name:<15} {config2.name:<15} {'Winner'}")
        print("-" * 80)

        for r1, r2 in zip(results1, results2):
            query_short = r1.query[:28] + "..." if len(r1.query) > 28 else r1.query

            # Determine winner based on results count (higher is better)
            if r1.results_count > r2.results_count:
                winner = f"✓ {config1.name}"
            elif r2.results_count > r1.results_count:
                winner = f"✓ {config2.name}"
            else:
                # Tie on count, use avg score
                if r1.avg_score > r2.avg_score:
                    winner = f"✓ {config1.name} (score)"
                elif r2.avg_score > r1.avg_score:
                    winner = f"✓ {config2.name} (score)"
                else:
                    winner = "Tie"

            print(f"{query_short:<30} {r1.results_count:<15} {r2.results_count:<15} {winner}")

        # Overall winner
        print()
        print("OVERALL WINNER")
        print("-" * 80)

        wins1 = sum(1 for r1, r2 in zip(results1, results2) if r1.results_count > r2.results_count)
        wins2 = sum(1 for r1, r2 in zip(results1, results2) if r2.results_count > r1.results_count)

        if wins1 > wins2:
            print(f"✓ {config1.name} wins {wins1} out of {len(results1)} queries")
        elif wins2 > wins1:
            print(f"✓ {config2.name} wins {wins2} out of {len(results2)} queries")
        else:
            print("Tie!")

    print()

    # Recommendations
    print()
    print("RECOMMENDATIONS")
    print("-" * 80)

    # Find best overall config
    best_config = None
    best_score = 0.0

    for config in configs:
        results = results_by_config[config.name]
        if not results:
            continue

        # Score: weighted average of success rate (70%) and avg results (30%)
        success_rate = sum(1 for r in results if r.results_count > 0) / len(results)
        avg_results_norm = min(sum(r.results_count for r in results) / len(results) / 10.0, 1.0)
        score = 0.7 * success_rate + 0.3 * avg_results_norm

        if score > best_score:
            best_score = score
            best_config = config

    if best_config:
        print(f"✅ Best Configuration: {best_config.name}")
        print(f"   Score: {best_score:.2f}")
        print()
        print("   Settings:")
        print(f"   - Hybrid Search: {best_config.enable_hybrid_search}")
        if best_config.enable_hybrid_search:
            print(f"   - Semantic Weight: {best_config.hybrid_semantic_weight}")
            print(f"   - Keyword Weight: {best_config.hybrid_keyword_weight}")
        print(f"   - Threshold: {best_config.semantic_threshold}")
        print(f"   - Query Expansion: {best_config.query_expansion}")
        if best_config.query_expansion:
            print(f"   - Max Expansions: {best_config.max_expansions}")

    print()
    print("=" * 80)


def export_results(results_by_config: Dict[str, List[BenchmarkResult]], output_file: str):
    """Export benchmark results to JSON."""
    export_data = {}

    for config_name, results in results_by_config.items():
        export_data[config_name] = [
            {
                "query": r.query,
                "results_count": r.results_count,
                "execution_time_ms": r.execution_time_ms,
                "top_5_files": r.top_5_files,
                "avg_score": r.avg_score,
                "has_exact_match": r.has_exact_match,
                "search_mode": r.search_mode,
            }
            for r in results
        ]

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"✅ Results exported to: {output_file}")


async def main_async(args):
    """Async main function."""
    # Load queries
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} benchmark queries")
    print()

    # Define configurations to test
    if args.compare_configs:
        # TODO: Load configs from files
        print("ERROR: --compare-configs not yet implemented")
        return
    else:
        # Default: test P4 disabled vs enabled
        configs = [
            BenchmarkConfig(
                name="P4-Disabled",
                enable_hybrid_search=False,
                semantic_threshold=0.7,  # Legacy threshold
                query_expansion=False,
            ),
            BenchmarkConfig(
                name="P4-Default",
                enable_hybrid_search=True,
                hybrid_semantic_weight=0.6,
                hybrid_keyword_weight=0.4,
                semantic_threshold=0.5,
                query_expansion=True,
                max_expansions=5,
            ),
            BenchmarkConfig(
                name="P4-Precision",
                enable_hybrid_search=True,
                hybrid_semantic_weight=0.9,
                hybrid_keyword_weight=0.1,
                semantic_threshold=0.65,
                query_expansion=True,
                max_expansions=3,
            ),
            BenchmarkConfig(
                name="P4-Recall",
                enable_hybrid_search=True,
                hybrid_semantic_weight=0.5,
                hybrid_keyword_weight=0.5,
                semantic_threshold=0.35,
                query_expansion=True,
                max_expansions=10,
            ),
        ]

    # Run benchmark
    results = await run_benchmark_suite(queries, configs, args.root)

    # Print report
    print_benchmark_report(results, configs)

    # Export if requested
    if args.export:
        export_results(results, args.export)


def main():
    parser = argparse.ArgumentParser(
        description="P4 Multi-Provider Excellence Performance Benchmarking"
    )
    parser.add_argument(
        "--queries",
        metavar="FILE",
        help="File with benchmark queries (one per line)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with built-in queries",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory to search (default: current directory)",
    )
    parser.add_argument(
        "--export",
        metavar="FILE",
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--compare-configs",
        nargs=2,
        metavar=("CONFIG1", "CONFIG2"),
        help="Compare two configuration files (YAML)",
    )
    parser.add_argument(
        "--profile",
        help="Use specific profile from profiles.yaml",
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
