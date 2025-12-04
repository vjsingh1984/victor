#!/usr/bin/env python3
"""Run full HumanEval benchmark with Victor agent.

This script uses the existing evaluation harness to run the complete
HumanEval benchmark using Victor's LLM providers.

Usage:
    # Run with default (Ollama) profile, parallelism=1
    python scripts/run_full_benchmark.py --profile default --parallel 1 --tasks 10

    # Run with claude-haiku profile, parallelism=4
    python scripts/run_full_benchmark.py --profile claude-haiku --parallel 4 --tasks 10

    # Run full benchmark (164 tasks)
    python scripts/run_full_benchmark.py --profile claude-haiku --parallel 4

    # Target specific Ollama host directly (bypasses tiered URL selection)
    python scripts/run_full_benchmark.py --profile default --base-url http://192.168.1.20:11434 --model gpt-oss:latest

    # Run two benchmarks in parallel on different hosts (in separate terminals)
    # Terminal 1: gpt-oss on 192.168.1.20
    python scripts/run_full_benchmark.py --profile default --base-url http://192.168.1.20:11434 --model gpt-oss:latest --output /tmp/bench_gpt_oss.json

    # Terminal 2: qwen3-coder on 192.168.1.73
    python scripts/run_full_benchmark.py --profile default --base-url http://192.168.1.73:11434 --model qwen3-coder:30b --output /tmp/bench_qwen3.json
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.config.settings import load_settings
from victor.evaluation import (
    BenchmarkType,
    EvaluationConfig,
    HumanEvalRunner,
    TaskStatus,
    get_harness,
    pass_at_k,
)
from victor.evaluation.protocol import BenchmarkTask
from victor.providers.base import Message
from victor.providers.registry import ProviderRegistry

# Set log level to WARN for cleaner output
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def print_banner(profile: str, num_tasks: int, parallel: int):
    """Print benchmark banner."""
    print()
    print("=" * 70)
    print("  VICTOR AI CODING ASSISTANT - HumanEval Benchmark")
    print("=" * 70)
    print()
    print(f"Profile:     {profile}")
    print(f"Tasks:       {num_tasks if num_tasks else 'ALL (164)'}")
    print(f"Parallelism: {parallel}")
    print(f"Timestamp:   {datetime.now().isoformat()}")
    print()


def create_agent_callback(profile: str, timeout: int = 120, base_url: str = None, model_override: str = None):
    """Create an agent callback that uses Victor's providers.

    Returns a callback function compatible with EvaluationHarness.run_evaluation().

    Args:
        profile: Profile name from profiles.yaml
        timeout: Request timeout in seconds
        base_url: Override base URL (useful for targeting specific Ollama hosts)
        model_override: Override model name from profile
    """
    # Load settings and profile
    settings = load_settings()
    profiles = settings.load_profiles()

    if profile not in profiles:
        raise ValueError(f"Profile '{profile}' not found. Available: {list(profiles.keys())}")

    profile_config = profiles[profile]
    model_name = model_override or profile_config.model
    print(f"Provider:    {profile_config.provider}/{model_name}")

    # Get provider settings
    provider_settings = settings.get_provider_settings(profile_config.provider)

    # Override base_url if specified (for direct host targeting)
    if base_url:
        provider_settings["base_url"] = base_url
        print(f"Base URL:    {base_url} (override)")

    provider = ProviderRegistry.create(
        profile_config.provider,
        model=model_name,
        **provider_settings,
    )

    async def agent_callback(task: BenchmarkTask) -> str:
        """Generate code for a single task using the LLM."""
        prompt = f"""You are an expert Python programmer. Complete the following function.
Your response must contain ONLY the complete Python code, starting with any necessary imports.
Do not include any explanations, markdown formatting, or code blocks - just the raw Python code.

{task.prompt}

Complete the function above. Write the implementation that passes all the test cases.
"""

        messages = [Message(role="user", content=prompt)]

        try:
            response = await asyncio.wait_for(
                provider.chat(
                    messages=messages,
                    model=model_name,
                    temperature=0.2,
                    max_tokens=1024,
                ),
                timeout=timeout,
            )

            generated_code = response.content.strip()

            # Clean up markdown code blocks if present
            if "```python" in generated_code:
                match = re.search(r"```python\n(.*?)```", generated_code, re.DOTALL)
                if match:
                    generated_code = match.group(1).strip()
            elif "```" in generated_code:
                match = re.search(r"```\n(.*?)```", generated_code, re.DOTALL)
                if match:
                    generated_code = match.group(1).strip()

            return generated_code

        except asyncio.TimeoutError:
            return task.prompt  # Return prompt only (will fail tests)
        except Exception as e:
            logger.warning(f"Error generating code for {task.task_id}: {e}")
            return task.prompt

    return agent_callback


async def run_benchmark(
    profile: str,
    num_tasks: int,
    parallel: int,
    output_file: str = None,
    base_url: str = None,
    model_override: str = None,
):
    """Run the benchmark using the existing evaluation harness.

    Args:
        profile: Profile name from profiles.yaml
        num_tasks: Number of tasks to run (None for all)
        parallel: Parallelism level
        output_file: Output JSON file path
        base_url: Override base URL (for targeting specific Ollama hosts)
        model_override: Override model name from profile
    """
    print_banner(profile, num_tasks, parallel)

    # Register runner with harness
    harness = get_harness()
    harness.register_runner(HumanEvalRunner())

    # Configure evaluation
    config = EvaluationConfig(
        benchmark=BenchmarkType.HUMAN_EVAL,
        model=model_override or profile,
        max_tasks=num_tasks,
        parallel_tasks=parallel,
        timeout_per_task=120,
    )

    # Create agent callback
    print("Initializing provider...")
    agent_callback = create_agent_callback(profile, base_url=base_url, model_override=model_override)
    print()

    # Run evaluation using the harness
    print("-" * 70)
    print("Running benchmark (progress shown in real-time)...")
    print("-" * 70)
    print(flush=True)

    # Progress callback to print results as they complete
    def print_progress(task_idx: int, total_tasks: int, task_result):
        status = "PASS" if task_result.is_success else "FAIL"
        quality = task_result.code_quality.get_overall_score() if task_result.code_quality else 0
        duration = task_result.duration_seconds or 0

        print(f"[{task_idx+1}/{total_tasks}] {task_result.task_id}")
        print(f"    Status: {status}")
        print(f"    Quality: {quality:.1f}/100")
        print(f"    Time: {duration:.1f}s")
        if task_result.error_message:
            print(f"    Error: {task_result.error_message[:80]}")
        print(flush=True)

    result = await harness.run_evaluation(config, agent_callback, progress_callback=print_progress)

    # Get metrics from harness
    metrics = result.get_metrics()

    # Calculate Pass@k
    n = metrics["total_tasks"]
    c = metrics["passed"]
    pass_at_1 = pass_at_k(n, c, 1)
    pass_at_5 = pass_at_k(n, c, 5) if n >= 5 else None
    pass_at_10 = pass_at_k(n, c, 10) if n >= 10 else None

    # Print summary
    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"Profile:          {profile}")
    print(f"Total Tasks:      {metrics['total_tasks']}")
    print(f"Passed:           {metrics['passed']}")
    print(f"Failed:           {metrics['failed']}")
    print(f"Errors:           {metrics['errors']}")
    print(f"Timeouts:         {metrics['timeouts']}")
    print(f"Pass Rate:        {metrics['pass_rate']:.1%}")
    print()
    print("Pass@k Metrics:")
    print(f"  Pass@1:         {pass_at_1:.2%}")
    if pass_at_5:
        print(f"  Pass@5:         {pass_at_5:.2%}")
    if pass_at_10:
        print(f"  Pass@10:        {pass_at_10:.2%}")
    print()
    print(f"Total Duration:   {metrics['duration_seconds']:.1f}s")
    print(f"Avg per Task:     {metrics['duration_seconds']/metrics['total_tasks']:.1f}s")
    print()
    print("=" * 70)

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "profile": profile,
        "metrics": metrics,
        "pass_at_k": {
            "pass_at_1": pass_at_1,
            "pass_at_5": pass_at_5,
            "pass_at_10": pass_at_10,
        },
    }

    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path(f"/tmp/victor_benchmark_{profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Also generate markdown report
    report = harness.generate_report(result, format="markdown")
    report_path = output_path.with_suffix('.md')
    report_path.write_text(report)
    print(f"Report saved to: {report_path}")

    return result


async def main():
    parser = argparse.ArgumentParser(
        description="Run HumanEval benchmark with Victor agent"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Profile to use (default, claude-haiku, etc.)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="Number of tasks to run (default: all 164)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Parallelism level (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override base URL (e.g., http://192.168.1.20:11434 for specific Ollama host)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name from profile (e.g., gpt-oss:latest)",
    )

    args = parser.parse_args()

    await run_benchmark(
        profile=args.profile,
        num_tasks=args.tasks,
        parallel=args.parallel,
        output_file=args.output,
        base_url=args.base_url,
        model_override=args.model,
    )


if __name__ == "__main__":
    asyncio.run(main())
