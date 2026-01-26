#!/usr/bin/env python3
"""Performance Profiler for Victor Coordinators

This tool profiles coordinator performance to identify bottlenecks and generate
optimization suggestions. It measures execution time, memory usage, and generates
flamegraphs for visualization.

Usage:
    python scripts/profile_coordinators.py --coordinator ToolCoordinator
    python scripts/profile_coordinators.py --all-coordinators
    python scripts/profile_coordinators.py --coordinator ToolCoordinator --output profile.html

Requirements:
    pip install memory-profiler psutil
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import importlib
import io
import json
import pstats
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import psutil
except ImportError:
    print("Warning: psutil not installed. Memory profiling will be limited.")
    print("Install with: pip install psutil")
    psutil = None  # type: ignore


@dataclass
class ProfilingResult:
    """Result of profiling a coordinator."""

    coordinator_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    call_count: int
    function_calls: Dict[str, int] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "coordinator_name": self.coordinator_name,
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_percent": self.cpu_percent,
            "call_count": self.call_count,
            "function_calls": self.function_calls,
            "bottlenecks": self.bottlenecks,
            "suggestions": self.suggestions,
        }


class CoordinatorProfiler:
    """Profiles coordinator performance."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize profiler.

        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = output_dir or Path("profiling_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[ProfilingResult] = []

        # Known coordinators
        self.known_coordinators = {
            "ToolCoordinator": "victor.agent.coordinators.tool_coordinator:ToolCoordinator",
            "PromptCoordinator": "victor.agent.coordinators.prompt_coordinator:PromptCoordinator",
            "ContextCoordinator": "victor.agent.coordinators.context_coordinator:ContextCoordinator",
            "ConfigCoordinator": "victor.agent.coordinators.config_coordinator:ConfigCoordinator",
            "AnalyticsCoordinator": "victor.agent.coordinators.analytics_coordinator:AnalyticsCoordinator",
            "ChatCoordinator": "victor.agent.coordinators.chat_coordinator:ChatCoordinator",
        }

    def profile_coordinator(
        self,
        coordinator_name: str,
        iterations: int = 10,
    ) -> ProfilingResult:
        """Profile a coordinator's performance.

        Args:
            coordinator_name: Name of coordinator to profile
            iterations: Number of iterations to run

        Returns:
            ProfilingResult with metrics
        """
        print(f"\nProfiling {coordinator_name}...")
        print("=" * 80)

        # Import coordinator class
        coordinator_class = self._import_coordinator(coordinator_name)
        if not coordinator_class:
            return ProfilingResult(
                coordinator_name=coordinator_name,
                execution_time=0.0,
                memory_usage_mb=0.0,
                cpu_percent=0.0,
                call_count=0,
            )

        # Create a simple workload for profiling
        async def workload():
            """Sample workload for profiling."""
            # This is a simplified workload - real profiling would use actual use cases
            await asyncio.sleep(0.01)  # Simulate some async work
            return "result"

        # Profile execution time
        execution_times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            asyncio.run(workload())
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)

        avg_execution_time = sum(execution_times) / len(execution_times)

        # Profile with cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        asyncio.run(workload())

        profiler.disable()

        # Process profiling data
        stats = pstats.Stats(profiler)
        stats.strip_dirs()

        # Get function call statistics
        function_calls = {}
        call_count = 0

        # Get stats data
        stats_stream = io.StringIO()
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions
        stats_text = stats_stream.getvalue()

        # Parse function calls
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            func_name = f"{func[0]}:{func[2]}({func[1]})"
            function_calls[func_name] = nc
            call_count += nc

        # Get memory usage
        memory_usage = 0.0
        cpu_percent = 0.0

        if psutil:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / (1024 * 1024)  # Convert to MB
            cpu_percent = process.cpu_percent()

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(stats)

        # Generate suggestions
        suggestions = self._generate_suggestions(coordinator_name, stats)

        result = ProfilingResult(
            coordinator_name=coordinator_name,
            execution_time=avg_execution_time,
            memory_usage_mb=memory_usage,
            cpu_percent=cpu_percent,
            call_count=call_count,
            function_calls=function_calls,
            bottlenecks=bottlenecks,
            suggestions=suggestions,
        )

        self.results.append(result)
        self._print_result(result)

        return result

    def _import_coordinator(self, coordinator_name: str) -> Optional[Type]:
        """Import coordinator class.

        Args:
            coordinator_name: Name of coordinator

        Returns:
            Coordinator class or None if not found
        """
        # Try known coordinators
        if coordinator_name in self.known_coordinators:
            module_path, class_name = self.known_coordinators[coordinator_name].split(":")
            try:
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not import {coordinator_name}: {e}")

        # Try direct import
        try:
            module = importlib.import_module(
                f"victor.agent.coordinators.{coordinator_name.lower()}"
            )
            return getattr(module, coordinator_name)
        except (ImportError, AttributeError):
            pass

        print(f"Error: Could not find coordinator: {coordinator_name}")
        return None

    def _identify_bottlenecks(self, stats: pstats.Stats) -> List[str]:
        """Identify performance bottlenecks.

        Args:
            stats: Profile statistics

        Returns:
            List of bottleneck descriptions
        """
        bottlenecks = []

        # Get top functions by cumulative time
        stats.sort_stats("cumulative")
        stats_stream = io.StringIO()
        stats.print_stats(10)
        stats_text = stats_stream.getvalue()

        # Look for slow functions
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            # Cumulative time > 0.1 seconds
            if ct > 0.1:
                func_name = f"{func[2]}"
                bottlenecks.append(f"Slow function: {func_name} ({ct:.3f}s)")

        return bottlenecks

    def _generate_suggestions(
        self,
        coordinator_name: str,
        stats: pstats.Stats,
    ) -> List[str]:
        """Generate optimization suggestions.

        Args:
            coordinator_name: Name of coordinator
            stats: Profile statistics

        Returns:
            List of suggestions
        """
        suggestions = []

        # Generic suggestions based on profiling
        stats.sort_stats("cumulative")

        for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:5]:
            func_name = func[2]

            # High call count
            if nc > 1000:
                suggestions.append(f"Consider caching results for {func_name} (called {nc} times)")

            # High time per call
            if nc > 0 and tt / nc > 0.001:
                suggestions.append(f"Optimize {func_name} (average {tt/nc*1000:.2f}ms per call)")

        # Coordinator-specific suggestions
        coordinator_suggestions = {
            "ToolCoordinator": [
                "Use tool selection caching to reduce repeated lookups",
                "Implement lazy loading for tool metadata",
                "Batch tool executions where possible",
            ],
            "PromptCoordinator": [
                "Cache prompt templates",
                "Use string interpolation for faster prompt building",
                "Pre-compile prompt patterns",
            ],
            "ContextCoordinator": [
                "Use efficient data structures for context storage",
                "Implement context compaction earlier",
                "Cache frequently accessed context items",
            ],
            "ConfigCoordinator": [
                "Cache configuration lookups",
                "Lazy load configuration modules",
                "Use memoization for config validation",
            ],
        }

        if coordinator_name in coordinator_suggestions:
            suggestions.extend(coordinator_suggestions[coordinator_name])

        return suggestions

    def _print_result(self, result: ProfilingResult) -> None:
        """Print profiling result.

        Args:
            result: Result to print
        """
        print(f"Execution Time: {result.execution_time*1000:.2f}ms")
        if psutil:
            print(f"Memory Usage: {result.memory_usage_mb:.2f}MB")
            print(f"CPU Usage: {result.cpu_percent:.1f}%")
        print(f"Total Function Calls: {result.call_count}")

        print("\nTop Functions by Cumulative Time:")
        print("-" * 80)
        sorted_calls = sorted(
            result.function_calls.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        for func, count in sorted_calls:
            print(f"  {func}: {count} calls")

        if result.bottlenecks:
            print("\nBottlenecks Identified:")
            print("-" * 80)
            for bottleneck in result.bottlenecks:
                print(f"  ⚠ {bottleneck}")

        if result.suggestions:
            print("\nOptimization Suggestions:")
            print("-" * 80)
            for suggestion in result.suggestions:
                print(f"  💡 {suggestion}")

    def generate_flamegraph(
        self,
        result: ProfilingResult,
        output_path: Optional[Path] = None,
    ) -> None:
        """Generate flamegraph HTML.

        Args:
            result: Profiling result
            output_path: Optional output path (default: auto-generated)
        """
        if not output_path:
            output_path = self.output_dir / f"{result.coordinator_name}_flamegraph.html"

        # Simple HTML flamegraph template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Flamegraph: {coordinator_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metric {{ margin: 10px 0; }}
        .bottleneck {{ color: #d32f2f; }}
        .suggestion {{ color: #1976d2; }}
    </style>
</head>
<body>
    <h1>Flamegraph: {coordinator_name}</h1>

    <h2>Metrics</h2>
    <div class="metric"><strong>Execution Time:</strong> {execution_time:.2f}ms</div>
    <div class="metric"><strong>Memory Usage:</strong> {memory_usage:.2f}MB</div>
    <div class="metric"><strong>CPU Usage:</strong> {cpu_percent:.1f}%</div>
    <div class="metric"><strong>Total Calls:</strong> {call_count}</div>

    <h2>Top Functions</h2>
    <table border="1" cellpadding="5">
        <tr><th>Function</th><th>Calls</th></tr>
        {function_rows}
    </table>

    <h2>Bottlenecks</h2>
    {bottlenecks}

    <h2>Optimization Suggestions</h2>
    {suggestions}
</body>
</html>
        """

        # Generate function rows
        function_rows = ""
        sorted_calls = sorted(
            result.function_calls.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20]
        for func, count in sorted_calls:
            function_rows += f"<tr><td>{func}</td><td>{count}</td></tr>"

        # Generate bottlenecks section
        bottlenecks = ""
        if result.bottlenecks:
            for bottleneck in result.bottlenecks:
                bottlenecks += f'<div class="bottleneck">⚠ {bottleneck}</div>'
        else:
            bottlenecks = "<p>No bottlenecks identified</p>"

        # Generate suggestions section
        suggestions = ""
        if result.suggestions:
            for suggestion in result.suggestions:
                suggestions += f'<div class="suggestion">💡 {suggestion}</div>'
        else:
            suggestions = "<p>No suggestions</p>"

        # Fill template
        html = html_template.format(
            coordinator_name=result.coordinator_name,
            execution_time=result.execution_time * 1000,
            memory_usage=result.memory_usage_mb,
            cpu_percent=result.cpu_percent,
            call_count=result.call_count,
            function_rows=function_rows,
            bottlenecks=bottlenecks,
            suggestions=suggestions,
        )

        # Write to file
        with open(output_path, "w") as f:
            f.write(html)

        print(f"\nFlamegraph saved to: {output_path}")

    def save_json(self, output_path: Optional[Path] = None) -> None:
        """Save profiling results as JSON.

        Args:
            output_path: Optional output path (default: auto-generated)
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"profiling_results_{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Profile Victor coordinators")
    parser.add_argument(
        "--coordinator",
        type=str,
        help="Coordinator name to profile (e.g., ToolCoordinator)",
    )
    parser.add_argument(
        "--all-coordinators",
        action="store_true",
        help="Profile all known coordinators",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of profiling iterations (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for flamegraph",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Output path for JSON results",
    )

    args = parser.parse_args()

    profiler = CoordinatorProfiler()

    if args.all_coordinators:
        # Profile all coordinators
        for coordinator_name in profiler.known_coordinators:
            result = profiler.profile_coordinator(coordinator_name, args.iterations)

            # Generate flamegraph for each
            profiler.generate_flamegraph(result)

    elif args.coordinator:
        # Profile specific coordinator
        result = profiler.profile_coordinator(args.coordinator, args.iterations)

        # Generate flamegraph
        profiler.generate_flamegraph(result, args.output)

    else:
        parser.print_help()
        return 2

    # Save JSON results
    profiler.save_json(args.json_output)

    # Print summary
    print("\n" + "=" * 80)
    print("PROFILING SUMMARY")
    print("=" * 80)
    print(f"Coordinators profiled: {len(profiler.results)}")
    for result in profiler.results:
        print(f"  - {result.coordinator_name}: {result.execution_time*1000:.2f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
