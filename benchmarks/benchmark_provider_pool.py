#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Performance benchmarks for provider pool.

Measures performance improvements from provider pooling and load balancing:
- Latency reduction under load
- Throughput improvement
- Failover performance
- Comparison of load balancing strategies
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.providers.provider_pool import (
    ProviderPool,
    ProviderPoolConfig,
    create_provider_pool,
)
from victor.providers.load_balancer import LoadBalancerType
from victor.providers.base import BaseProvider, Message, CompletionResponse
from victor.providers.health_monitor import HealthStatus


class BenchmarkProvider(BaseProvider):
    """Mock provider for benchmarking with configurable latency."""

    def __init__(
        self,
        name: str,
        latency_ms: float = 100,
        failure_rate: float = 0.0,
    ):
        super().__init__()
        self._name = name
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.request_count = 0
        self.failure_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: List = None,
        **kwargs,
    ) -> CompletionResponse:
        """Simulate chat with configurable latency."""
        self.request_count += 1

        # Simulate failure
        if self.failure_rate > 0 and (self.request_count % int(1 / self.failure_rate)) == 0:
            self.failure_count += 1
            raise RuntimeError(f"Provider {self._name} simulated failure")

        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        return CompletionResponse(
            content=f"Response from {self._name}",
            model=model,
            role="assistant",
        )

    async def stream(self, messages, *, model, **kwargs):
        """Simulate streaming."""
        await asyncio.sleep(self.latency_ms / 1000.0)
        yield StreamChunk(content=f"Stream from {self._name}")

    async def close(self) -> None:
        """Cleanup."""
        pass


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""

    name: str
    num_requests: int = 100
    num_providers: int = 3
    concurrency: int = 10
    latency_ms: float = 100
    failure_rate: float = 0.0
    load_balancer: LoadBalancerType = LoadBalancerType.ROUND_ROBIN


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    config: BenchmarkConfig
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_sec: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    pool_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "config": {
                "num_requests": self.config.num_requests,
                "num_providers": self.config.num_providers,
                "concurrency": self.config.concurrency,
                "latency_ms": self.config.latency_ms,
                "failure_rate": self.config.failure_rate,
                "load_balancer": self.config.load_balancer.value,
            },
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_duration_sec": round(self.total_duration_sec, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "throughput_rps": round(self.throughput_rps, 2),
            "pool_stats": self.pool_stats,
        }


class ProviderPoolBenchmark:
    """Benchmark suite for provider pool."""

    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []

    async def run_benchmark(
        self,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            config: Benchmark configuration

        Returns:
            BenchmarkResult with metrics
        """
        print(f"\n{'='*60}")
        print(f"Running: {config.name}")
        print(f"  Requests: {config.num_requests}")
        print(f"  Providers: {config.num_providers}")
        print(f"  Concurrency: {config.concurrency}")
        print(f"  Latency: {config.latency_ms}ms")
        print(f"  Failure Rate: {config.failure_rate:.1%}")
        print(f"  Load Balancer: {config.load_balancer.value}")
        print(f"{'='*60}")

        # Create providers
        providers = {}
        for i in range(config.num_providers):
            provider_id = f"provider-{i}"
            providers[provider_id] = BenchmarkProvider(
                name=provider_id,
                latency_ms=config.latency_ms,
                failure_rate=config.failure_rate,
            )

        # Create pool
        pool_config = ProviderPoolConfig(
            load_balancer=config.load_balancer,
            enable_warmup=False,
            max_retries=3,
        )

        pool = await create_provider_pool(
            name=f"benchmark-{config.name}",
            providers=providers,
            config=pool_config,
        )

        # Run benchmark
        latencies = []
        start_time = time.time()
        successful = 0
        failed = 0

        semaphore = asyncio.Semaphore(config.concurrency)

        async def make_request(idx: int) -> None:
            """Make a single request."""
            nonlocal successful, failed

            async with semaphore:
                req_start = time.time()
                try:
                    await pool.chat(
                        [Message(role="user", content=f"Request {idx}")],
                        model="test-model",
                    )
                    successful += 1
                    latency_ms = (time.time() - req_start) * 1000
                    latencies.append(latency_ms)
                except Exception as e:
                    failed += 1

        # Launch all requests
        tasks = [make_request(i) for i in range(config.num_requests)]
        await asyncio.gather(*tasks)

        total_duration = time.time() - start_time

        # Calculate metrics
        latencies.sort()
        result = BenchmarkResult(
            name=config.name,
            config=config,
            total_requests=config.num_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_duration_sec=total_duration,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            p50_latency_ms=latencies[len(latencies) // 2] if latencies else 0,
            p95_latency_ms=latencies[int(len(latencies) * 0.95)] if latencies else 0,
            p99_latency_ms=latencies[int(len(latencies) * 0.99)] if latencies else 0,
            throughput_rps=successful / total_duration if total_duration > 0 else 0,
            pool_stats=pool.get_pool_stats(),
        )

        # Cleanup
        await pool.close()

        print(f"\nResults:")
        print(f"  Successful: {successful}/{config.num_requests}")
        print(f"  Failed: {failed}/{config.num_requests}")
        print(f"  Duration: {total_duration:.2f}s")
        print(f"  Throughput: {result.throughput_rps:.2f} req/s")
        print(f"  Avg Latency: {result.avg_latency_ms:.2f}ms")
        print(f"  P50 Latency: {result.p50_latency_ms:.2f}ms")
        print(f"  P95 Latency: {result.p95_latency_ms:.2f}ms")
        print(f"  P99 Latency: {result.p99_latency_ms:.2f}ms")

        self.results.append(result)
        return result

    async def run_all_benchmarks(self) -> None:
        """Run comprehensive benchmark suite."""
        print("\n" + "=" * 60)
        print("PROVIDER池 BENCHMARK SUITE")
        print("=" * 60)

        # Benchmark 1: Baseline - Single provider
        await self.run_benchmark(
            BenchmarkConfig(
                name="baseline_single_provider",
                num_requests=100,
                num_providers=1,
                concurrency=1,
                latency_ms=100,
                failure_rate=0.0,
                load_balancer=LoadBalancerType.ROUND_ROBIN,
            )
        )

        # Benchmark 2: Pool with 3 providers
        await self.run_benchmark(
            BenchmarkConfig(
                name="pool_3_providers",
                num_requests=100,
                num_providers=3,
                concurrency=3,
                latency_ms=100,
                failure_rate=0.0,
                load_balancer=LoadBalancerType.ROUND_ROBIN,
            )
        )

        # Benchmark 3: High concurrency
        await self.run_benchmark(
            BenchmarkConfig(
                name="high_concurrency",
                num_requests=300,
                num_providers=5,
                concurrency=20,
                latency_ms=100,
                failure_rate=0.0,
                load_balancer=LoadBalancerType.LEAST_CONNECTIONS,
            )
        )

        # Benchmark 4: Compare load balancers
        for lb_strategy in [
            LoadBalancerType.ROUND_ROBIN,
            LoadBalancerType.LEAST_CONNECTIONS,
            LoadBalancerType.ADAPTIVE,
        ]:
            await self.run_benchmark(
                BenchmarkConfig(
                    name=f"lb_comparison_{lb_strategy.value}",
                    num_requests=200,
                    num_providers=3,
                    concurrency=10,
                    latency_ms=100,
                    failure_rate=0.0,
                    load_balancer=lb_strategy,
                )
            )

        # Benchmark 5: With failures
        await self.run_benchmark(
            BenchmarkConfig(
                name="with_failures",
                num_requests=200,
                num_providers=3,
                concurrency=5,
                latency_ms=100,
                failure_rate=0.1,  # 10% failure rate
                load_balancer=LoadBalancerType.ADAPTIVE,
            )
        )

        # Benchmark 6: Variable latency
        providers_config = [
            (
                "variable_latency",
                [
                    ("fast", 50),
                    ("medium", 150),
                    ("slow", 300),
                ],
            ),
        ]

        # Generate comparison report
        self.generate_report()

    def generate_report(self) -> None:
        """Generate markdown report with results."""
        report_lines = [
            "# Provider Pool Benchmark Report",
            f"\nGenerated: {datetime.now().isoformat()}",
            "\n## Summary",
            "\nThis report compares the performance of different provider pool configurations.",
            "\n## Benchmark Results",
            "\n",
        ]

        # Table header
        report_lines.extend(
            [
                "| Benchmark | Throughput (req/s) | Avg Latency (ms) | P95 Latency (ms) | Success Rate |",
                "|-----------|-------------------|------------------|------------------|--------------|",
            ]
        )

        for result in self.results:
            success_rate = (
                result.successful_requests / result.total_requests * 100
                if result.total_requests > 0
                else 0
            )
            report_lines.append(
                f"| {result.name} | {result.throughput_rps:.2f} | "
                f"{result.avg_latency_ms:.2f} | {result.p95_latency_ms:.2f} | "
                f"{success_rate:.1f}% |"
            )

        # Detailed results
        report_lines.append("\n## Detailed Results\n")

        for result in self.results:
            report_lines.extend(
                [
                    f"### {result.name}",
                    f"\n**Configuration:**",
                    f"- Requests: {result.config.num_requests}",
                    f"- Providers: {result.config.num_providers}",
                    f"- Concurrency: {result.config.concurrency}",
                    f"- Latency: {result.config.latency_ms}ms",
                    f"- Load Balancer: {result.config.load_balancer.value}",
                    f"\n**Results:**",
                    f"- Total Requests: {result.total_requests}",
                    f"- Successful: {result.successful_requests}",
                    f"- Failed: {result.failed_requests}",
                    f"- Duration: {result.total_duration_sec:.2f}s",
                    f"- Throughput: {result.throughput_rps:.2f} req/s",
                    f"- Avg Latency: {result.avg_latency_ms:.2f}ms",
                    f"- P50 Latency: {result.p50_latency_ms:.2f}ms",
                    f"- P95 Latency: {result.p95_latency_ms:.2f}ms",
                    f"- P99 Latency: {result.p99_latency_ms:.2f}ms",
                    "\n",
                ]
            )

        # Performance improvements
        report_lines.extend(
            [
                "## Performance Analysis",
                "\n### Key Findings",
            ]
        )

        if len(self.results) >= 2:
            baseline = self.results[0]
            pool_3 = self.results[1]

            throughput_improvement = (
                (pool_3.throughput_rps - baseline.throughput_rps) / baseline.throughput_rps * 100
                if baseline.throughput_rps > 0
                else 0
            )

            latency_reduction = (
                (baseline.avg_latency_ms - pool_3.avg_latency_ms) / baseline.avg_latency_ms * 100
                if baseline.avg_latency_ms > 0
                else 0
            )

            report_lines.extend(
                [
                    f"- **Throughput Improvement:** {throughput_improvement:.1f}% "
                    f"({baseline.throughput_rps:.2f} → {pool_3.throughput_rps:.2f} req/s)",
                    f"- **Latency Reduction:** {latency_reduction:.1f}% "
                    f"({baseline.avg_latency_ms:.2f}ms → {pool_3.avg_latency_ms:.2f}ms)",
                ]
            )

        report_lines.append("\n### Load Balancer Comparison")

        # Extract load balancer comparison results
        lb_results = [r for r in self.results if r.name.startswith("lb_comparison_")]

        if lb_results:
            best_throughput = max(lb_results, key=lambda r: r.throughput_rps)
            lowest_latency = min(lb_results, key=lambda r: r.avg_latency_ms)

            report_lines.extend(
                [
                    f"- **Best Throughput:** {best_throughput.config.load_balancer.value} "
                    f"({best_throughput.throughput_rps:.2f} req/s)",
                    f"- **Lowest Latency:** {lowest_latency.config.load_balancer.value} "
                    f"({lowest_latency.avg_latency_ms:.2f}ms)",
                ]
            )

        # Save report
        report_path = self.output_dir / "provider_pool_benchmark_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        # Save JSON results
        json_path = self.output_dir / "provider_pool_benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(
                [r.to_dict() for r in self.results],
                f,
                indent=2,
            )

        print(f"\n{'='*60}")
        print(f"Reports saved:")
        print(f"  {report_path}")
        print(f"  {json_path}")
        print(f"{'='*60}")


async def main():
    """Run benchmarks."""
    benchmark = ProviderPoolBenchmark()
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
