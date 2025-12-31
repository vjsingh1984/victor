#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# Apache License 2.0 - See LICENSE
"""RL Benchmark Harness for apples-to-apples model comparison.

Design Patterns Used:
- Strategy Pattern: Different complexity levels with consistent interfaces
- Template Method: Batch execution with customizable prompts
- Observer Pattern: Metrics collection and Q-value tracking
- Factory Pattern: Profile-based model instantiation

Key Features:
- Same prompt for ALL models within a batch (apples-to-apples)
- Varying complexity across batches
- Parallel cloud provider execution
- Sequential Ollama execution (VRAM constraint)
- Automatic Q-value tracking and analysis
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Task complexity levels for benchmarking."""

    SIMPLE = "simple"  # Single file read, basic questions
    MEDIUM = "medium"  # Multi-file exploration, analysis
    COMPLEX = "complex"  # Cross-codebase tracing, synthesis


@dataclass
class BenchmarkPrompt:
    """A benchmark prompt with metadata."""

    text: str
    complexity: ComplexityLevel
    requires_tools: bool = True
    expected_tool_count: int = 1
    tags: List[str] = field(default_factory=list)


@dataclass
class BatchResult:
    """Result from a single model in a batch."""

    profile: str
    provider: str
    success: bool
    duration_seconds: float
    exit_code: int
    output_file: Path
    error: Optional[str] = None


@dataclass
class BatchMetrics:
    """Aggregated metrics for a batch."""

    batch_id: int
    complexity: ComplexityLevel
    prompt: str
    results: List[BatchResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def avg_duration(self) -> float:
        successes = [r.duration_seconds for r in self.results if r.success]
        return sum(successes) / len(successes) if successes else 0.0


# Benchmark prompts organized by complexity
BENCHMARK_PROMPTS: Dict[ComplexityLevel, List[BenchmarkPrompt]] = {
    ComplexityLevel.SIMPLE: [
        BenchmarkPrompt(
            "Read README.md and summarize the project's main purpose in 2 sentences.",
            ComplexityLevel.SIMPLE,
            requires_tools=True,
            expected_tool_count=1,
            tags=["file-read", "summarization"],
        ),
        BenchmarkPrompt(
            "What is the version of Python required by this project? Check pyproject.toml.",
            ComplexityLevel.SIMPLE,
            requires_tools=True,
            expected_tool_count=1,
            tags=["file-read", "extraction"],
        ),
    ],
    ComplexityLevel.MEDIUM: [
        BenchmarkPrompt(
            "List all files in victor/agent/ and identify which one contains the main orchestration logic. Explain its role briefly.",
            ComplexityLevel.MEDIUM,
            requires_tools=True,
            expected_tool_count=2,
            tags=["directory-listing", "file-read", "analysis"],
        ),
        BenchmarkPrompt(
            "Search for 'CostTier' in the codebase and explain what tiers exist and their purpose.",
            ComplexityLevel.MEDIUM,
            requires_tools=True,
            expected_tool_count=2,
            tags=["code-search", "analysis"],
        ),
        BenchmarkPrompt(
            "Read victor/tools/base.py and identify the abstract methods that tool implementations must define.",
            ComplexityLevel.MEDIUM,
            requires_tools=True,
            expected_tool_count=1,
            tags=["file-read", "code-analysis"],
        ),
    ],
    ComplexityLevel.COMPLEX: [
        BenchmarkPrompt(
            "Analyze the tool calling adapter system in victor/agent/tool_calling/. What providers are supported and how do their implementations differ?",
            ComplexityLevel.COMPLEX,
            requires_tools=True,
            expected_tool_count=3,
            tags=["multi-file", "architecture", "comparison"],
        ),
        BenchmarkPrompt(
            "Trace the flow from user input to tool execution. Start from cli.py, follow through orchestrator.py, and identify the key function calls.",
            ComplexityLevel.COMPLEX,
            requires_tools=True,
            expected_tool_count=4,
            tags=["tracing", "multi-file", "flow-analysis"],
        ),
    ],
}


class RLBenchmarkHarness:
    """Harness for running RL benchmark experiments.

    Ensures apples-to-apples comparison by:
    1. Running identical prompts for all models in a batch
    2. Tracking complexity level per batch
    3. Collecting consistent metrics
    """

    # Cloud providers that can run in parallel
    CLOUD_PROFILES = [
        "claude-haiku",
        "gemini-2.5-flash",
        "grok-fast",
        "deepseek",
        "groqcloud",
    ]

    # Ollama profiles that must run sequentially (VRAM constraint)
    OLLAMA_PROFILES = [
        "default",  # qwen2.5-coder-tools:32b on 192.168.1.20
    ]

    # Mock/test providers to exclude from Q-table analysis
    MOCK_PROVIDERS = {"mock", "mock_provider", "dummy", "dummy-stream", "test"}

    def __init__(
        self,
        project_dir: Path = Path("/Users/vijaysingh/code/codingagent"),
        output_dir: Optional[Path] = None,
        timeout_seconds: int = 120,
    ):
        self.project_dir = project_dir
        self.output_dir = output_dir or Path(
            f"/tmp/rl-benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds
        self.batch_metrics: List[BatchMetrics] = []

    async def run_single_model(
        self,
        profile: str,
        prompt: str,
        batch_id: int,
    ) -> BatchResult:
        """Run a single model with the given prompt."""
        output_file = self.output_dir / f"batch{batch_id}_{profile}.log"
        start_time = time.time()

        cmd = ["victor", "chat", "-p", profile, "--plain", prompt]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.project_dir),
            )
            duration = time.time() - start_time

            # Write output to file
            with open(output_file, "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write(f"\n--- STDERR ---\n{result.stderr}")

            # Determine provider from profile (simplified mapping)
            provider = self._profile_to_provider(profile)

            return BatchResult(
                profile=profile,
                provider=provider,
                success=result.returncode == 0,
                duration_seconds=duration,
                exit_code=result.returncode,
                output_file=output_file,
                error=result.stderr if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return BatchResult(
                profile=profile,
                provider=self._profile_to_provider(profile),
                success=False,
                duration_seconds=duration,
                exit_code=124,  # Timeout exit code
                output_file=output_file,
                error="Timeout exceeded",
            )
        except Exception as e:
            return BatchResult(
                profile=profile,
                provider=self._profile_to_provider(profile),
                success=False,
                duration_seconds=0,
                exit_code=1,
                output_file=output_file,
                error=str(e),
            )

    def _profile_to_provider(self, profile: str) -> str:
        """Map profile name to provider for Q-table tracking."""
        mapping = {
            "claude-haiku": "anthropic",
            "claude-sonnet": "anthropic",
            "gemini-2.5-flash": "google",
            "gemini-2.5-pro": "google",
            "grok-fast": "xai",
            "grok": "xai",
            "deepseek": "deepseek",
            "groqcloud": "groq",
            "groq-mixtral": "groq",
            "default": "ollama",
            "ollama-32b": "ollama",
        }
        return mapping.get(profile, profile.split("-")[0])

    async def run_batch(
        self,
        batch_id: int,
        prompt: BenchmarkPrompt,
        include_ollama: bool = False,
    ) -> BatchMetrics:
        """Run a batch with the same prompt for all models.

        Cloud models run in parallel, Ollama runs sequentially.
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"BATCH {batch_id} - {prompt.complexity.value.upper()}")
        logger.info(f"{'='*50}")
        logger.info(f"Prompt: {prompt.text[:80]}...")

        metrics = BatchMetrics(
            batch_id=batch_id,
            complexity=prompt.complexity,
            prompt=prompt.text,
        )

        # Run cloud providers in parallel
        logger.info(f"\nRunning {len(self.CLOUD_PROFILES)} cloud providers in parallel...")
        cloud_tasks = [
            self.run_single_model(profile, prompt.text, batch_id) for profile in self.CLOUD_PROFILES
        ]
        cloud_results = await asyncio.gather(*cloud_tasks)

        for result in cloud_results:
            status = "ok" if result.success else "FAIL"
            logger.info(
                f"  [{batch_id}] {result.profile}: {status} ({result.duration_seconds:.1f}s)"
            )
            metrics.results.append(result)

        # Run Ollama sequentially if requested
        if include_ollama:
            logger.info(f"\nRunning {len(self.OLLAMA_PROFILES)} Ollama models sequentially...")
            for profile in self.OLLAMA_PROFILES:
                result = await self.run_single_model(profile, prompt.text, batch_id)
                status = "ok" if result.success else "FAIL"
                logger.info(
                    f"  [{batch_id}] {result.profile}: {status} ({result.duration_seconds:.1f}s)"
                )
                metrics.results.append(result)

        self.batch_metrics.append(metrics)
        return metrics

    async def run_experiment(
        self,
        num_batches: int = 10,
        include_ollama: bool = False,
    ) -> Dict[str, Any]:
        """Run a full experiment with multiple batches.

        Each batch uses the same prompt for all models (apples-to-apples).
        Batches cycle through different complexity levels.
        """
        logger.info(f"Starting RL Benchmark Experiment")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Batches: {num_batches}, Include Ollama: {include_ollama}")

        # Collect all prompts and cycle through them
        all_prompts = []
        for complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX]:
            all_prompts.extend(BENCHMARK_PROMPTS[complexity])

        for batch_id in range(1, num_batches + 1):
            prompt_idx = (batch_id - 1) % len(all_prompts)
            prompt = all_prompts[prompt_idx]
            await self.run_batch(batch_id, prompt, include_ollama)

        # Generate summary
        return self.generate_summary()

    def generate_summary(self) -> Dict[str, Any]:
        """Generate experiment summary with Q-value analysis."""
        # Aggregate by provider
        provider_stats: Dict[str, Dict[str, Any]] = {}

        for batch in self.batch_metrics:
            for result in batch.results:
                provider = result.provider
                if provider in self.MOCK_PROVIDERS:
                    continue

                if provider not in provider_stats:
                    provider_stats[provider] = {
                        "total": 0,
                        "success": 0,
                        "total_duration": 0.0,
                        "by_complexity": {
                            c.value: {"total": 0, "success": 0} for c in ComplexityLevel
                        },
                    }

                stats = provider_stats[provider]
                stats["total"] += 1
                if result.success:
                    stats["success"] += 1
                    stats["total_duration"] += result.duration_seconds
                stats["by_complexity"][batch.complexity.value]["total"] += 1
                if result.success:
                    stats["by_complexity"][batch.complexity.value]["success"] += 1

        # Calculate success rates
        summary = {
            "experiment_id": self.output_dir.name,
            "total_batches": len(self.batch_metrics),
            "timestamp": datetime.now().isoformat(),
            "provider_rankings": [],
        }

        for provider, stats in sorted(
            provider_stats.items(),
            key=lambda x: x[1]["success"] / max(x[1]["total"], 1),
            reverse=True,
        ):
            success_rate = stats["success"] / max(stats["total"], 1)
            avg_duration = stats["total_duration"] / max(stats["success"], 1)

            summary["provider_rankings"].append(
                {
                    "provider": provider,
                    "success_rate": round(success_rate, 3),
                    "total_runs": stats["total"],
                    "successful_runs": stats["success"],
                    "avg_duration_seconds": round(avg_duration, 2),
                    "by_complexity": {
                        c: {
                            "success_rate": round(v["success"] / max(v["total"], 1), 3),
                            "runs": v["total"],
                        }
                        for c, v in stats["by_complexity"].items()
                        if v["total"] > 0
                    },
                }
            )

        # Save summary
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*50}")
        logger.info("EXPERIMENT SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total batches: {summary['total_batches']}")
        logger.info("\nProvider Rankings (by success rate):")
        for rank in summary["provider_rankings"]:
            logger.info(
                f"  {rank['provider']:12} "
                f"success={rank['success_rate']:.1%} "
                f"({rank['successful_runs']}/{rank['total_runs']}) "
                f"avg={rank['avg_duration_seconds']:.1f}s"
            )
        logger.info(f"\nFull summary: {summary_file}")

        return summary

    def get_filtered_q_table(self) -> Dict[str, float]:
        """Get Q-table excluding mock/test providers."""
        q_table_path = Path.home() / ".victor" / "rl_q_tables.json"

        if not q_table_path.exists():
            return {}

        with open(q_table_path) as f:
            data = json.load(f)

        q_table = data.get("q_table", {})
        return {k: v for k, v in q_table.items() if k not in self.MOCK_PROVIDERS}


async def main():
    """Run benchmark experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="RL Benchmark Harness")
    parser.add_argument("--batches", type=int, default=10, help="Number of batches")
    parser.add_argument("--ollama", action="store_true", help="Include Ollama models")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per model (seconds)")
    args = parser.parse_args()

    harness = RLBenchmarkHarness(timeout_seconds=args.timeout)
    summary = await harness.run_experiment(
        num_batches=args.batches,
        include_ollama=args.ollama,
    )

    # Show filtered Q-table
    print("\n" + "=" * 50)
    print("FILTERED Q-TABLE (excluding mock providers)")
    print("=" * 50)
    q_table = harness.get_filtered_q_table()
    for provider, q_value in sorted(q_table.items(), key=lambda x: -x[1]):
        print(f"  {provider:12} Q={q_value:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
