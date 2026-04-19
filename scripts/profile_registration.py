#!/usr/bin/env python
"""Profile registration performance to identify bottlenecks.

This script uses cProfile and memory profiling to analyze registration
operations and identify O(n²) complexity hotspots.
"""

import cProfile
import pstats
import time
import tracemalloc
from pathlib import Path
from typing import List

# Set up path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.tools.registry import ToolRegistry
from victor.tools.base import BaseTool
from victor.tools.enums import AccessMode, CostTier, DangerLevel, ExecutionCategory, Priority
from victor.tools.metadata import ToolMetadata
from typing import Dict, Any, List


class MockTool(BaseTool):
    """Mock tool for performance testing."""

    def __init__(self, name: str, description: str, tags: List[str] = None):
        self._name = name
        self._description = description
        self._tags = tags or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self._name,
            description=self._description,
            category=ExecutionCategory.READ_ONLY,
            access_mode=AccessMode.READONLY,
            cost_tier=CostTier.FREE,
            danger_level=DangerLevel.SAFE,
            priority=Priority.MEDIUM,
            tags=self._tags,
        )

    async def execute(self, _exec_ctx, **kwargs):
        from victor.tools.base import ToolResult
        return ToolResult(success=True, output="test output")


def profile_registration(n_items: int) -> pstats.Stats:
    """Profile registration of N items.

    Args:
        n_items: Number of items to register

    Returns:
        Profile statistics
    """
    print(f"\n=== Profiling registration of {n_items} items ===")

    registry = ToolRegistry()

    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Start memory tracking
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    # Register items
    start_time = time.perf_counter()
    for i in range(n_items):
        tool = MockTool(
            name=f"tool_{i}",
            description=f"Test tool {i}",
            tags=[f"tag_{i % 10}", f"category_{i % 5}"]
        )
        registry.register(tool)

    end_time = time.perf_counter()
    snapshot2 = tracemalloc.take_snapshot()

    # Stop profiling
    profiler.disable()
    tracemalloc.stop()

    # Report timing
    duration_ms = (end_time - start_time) * 1000
    print(f"Total time: {duration_ms:.2f}ms")
    print(f"Time per item: {duration_ms/n_items:.3f}ms")

    # Report memory
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("\nTop 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)

    # Get statistics
    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    # Print top 20 functions by time
    print("\n=== Top 20 functions by cumulative time ===")
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    # Print top 20 functions by time (called)
    print("\n=== Top 20 functions by own time ===")
    stats.sort_stats('time')
    stats.print_stats(20)

    return stats


def profile_lookup(registry: ToolRegistry, n_lookups: int = 100):
    """Profile lookup operations.

    Args:
        registry: ToolRegistry with registered tools
        n_lookups: Number of lookup operations to perform
    """
    print(f"\n=== Profiling {n_lookups} lookup operations ===")

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.perf_counter()
    for i in range(n_lookups):
        # Simulate lookup operations
        registry.get(f"tool_{i % 100}")

    end_time = time.perf_counter()
    profiler.disable()

    duration_ms = (end_time - start_time) * 1000
    print(f"Total time: {duration_ms:.2f}ms")
    print(f"Time per lookup: {duration_ms/n_lookups:.3f}ms")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.print_stats(10)


def main():
    """Run profiling benchmarks."""
    print("=" * 70)
    print("REGISTRATION PERFORMANCE PROFILING")
    print("=" * 70)

    # Test different scales
    scales = [10, 50, 100, 500, 1000]

    results = []
    for scale in scales:
        stats = profile_registration(scale)
        results.append((scale, stats))

        print("\n" + "=" * 70 + "\n")

    # Print summary
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"{'Scale':<10} {'Time (ms)':<15} {'Per Item (ms)':<15}")
    print("-" * 40)
    for scale, _ in results:
        pass  # Timing already printed in profile_registration

    # Save profiling data
    print("\n=== Saving profiling data ===")
    stats = profile_registration(100)
    stats.dump_stats('registration_profile.prof')
    print("Profile saved to registration_profile.prof")
    print("View with: snakeviz registration_profile.prof")
    print("Or: python -m pstats registration_profile.prof")


if __name__ == "__main__":
    main()
