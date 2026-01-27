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

"""Benchmark CLI commands for Victor.

Provides CLI commands for running industry-standard AI coding benchmarks:
- SWE-bench: Real-world GitHub issue resolution
- HumanEval: Code generation from docstrings
- MBPP: Basic Python programming problems
- Framework comparison: Compare against other AI coding assistants

Usage:
    victor benchmark list                       # List available benchmarks
    victor benchmark run swe-bench --max-tasks 10
    victor benchmark run humaneval --model claude-3-sonnet
    victor benchmark run mbpp --output results.json
    victor benchmark compare --benchmark swe-bench
    victor benchmark leaderboard
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from victor.ui.commands.utils import setup_logging

benchmark_app = typer.Typer(
    name="benchmark",
    help="Run AI coding benchmarks and compare against other frameworks.",
)
console = Console()


def _configure_log_level(log_level: Optional[str], command: str = "benchmark") -> None:
    """Configure logging for benchmark commands using centralized config.

    Uses the centralized logging config system with proper priority chain:
    1. CLI argument (log_level)
    2. Environment variable (VICTOR_LOG_LEVEL)
    3. User config (~/.victor/config.yaml)
    4. Command-specific override from package config
    5. Package defaults (WARNING console, INFO file)

    Args:
        log_level: CLI-provided log level (highest priority)
        command: Command name for command-specific config lookup
    """
    # Validate log level if provided
    if log_level is not None:
        log_level = log_level.upper()
        valid_levels = {"DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"}

        if log_level not in valid_levels:
            console.print(
                f"[bold red]Error:[/] Invalid log level '{log_level}'. "
                f"Valid options: {', '.join(sorted(valid_levels))}"
            )
            raise typer.Exit(1)

        if log_level == "WARN":
            log_level = "WARNING"

    # Use centralized logging config
    setup_logging(command=command, cli_log_level=log_level, stream=sys.stderr)


@benchmark_app.command("list")
def list_benchmarks() -> None:
    """List available benchmarks and their descriptions."""
    from victor.evaluation.protocol import BenchmarkType

    table = Table(title="Available Benchmarks")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Tasks", style="green")
    table.add_column("Source", style="dim")

    benchmarks = [
        ("swe-bench", "Real-world GitHub issue resolution", "~2300", "Princeton NLP"),
        ("swe-bench-lite", "Curated subset of SWE-bench", "~300", "Princeton NLP"),
        ("humaneval", "Code generation from docstrings", "164", "OpenAI"),
        ("mbpp", "Mostly Basic Python Problems", "974", "Google Research"),
        ("mbpp-test", "MBPP test split", "500", "Google Research"),
    ]

    for name, desc, tasks, source in benchmarks:
        table.add_row(name, desc, tasks, source)

    console.print(table)
    console.print("\n[dim]Run with: victor benchmark run <benchmark-name>[/]")


@benchmark_app.command("setup")
def setup_benchmark(
    benchmark: str = typer.Argument(..., help="Benchmark to setup: swe-bench, swe-bench-lite"),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-n", help="Maximum number of repos to setup"
    ),
    force_reindex: bool = typer.Option(
        False, "--force", "-f", help="Force re-index even if already done"
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
) -> None:
    """Setup benchmark repos (clone + index). Run before 'run' for faster execution.

    This downloads and indexes the repositories needed for the benchmark.
    Indexes are cached in ~/.victor/swe_bench_cache/ and reused across runs.

    Example:
        victor benchmark setup swe-bench --max-tasks 5
        victor benchmark run swe-bench --max-tasks 5
    """
    _configure_log_level(log_level)

    from victor.evaluation.protocol import BenchmarkType, EvaluationConfig
    from victor.evaluation.benchmarks import SWEBenchRunner

    # Only SWE-bench needs setup (repo cloning)
    benchmark_lower = benchmark.lower().replace("_", "-")
    if benchmark_lower not in ("swe-bench", "swe-bench-lite"):
        console.print(f"[yellow]Setup not needed for {benchmark}[/]")
        console.print("Only SWE-bench benchmarks require repo setup.")
        return

    async def run_setup():
        from victor.evaluation.swe_bench_loader import SWEBenchWorkspaceManager

        # Create runner to load tasks
        if benchmark_lower == "swe-bench-lite":
            runner = SWEBenchRunner(split="lite")
        else:
            runner = SWEBenchRunner()

        # Load tasks (model is required but not used for setup)
        config = EvaluationConfig(
            benchmark=BenchmarkType.SWE_BENCH,
            model="setup-only",  # Not used for cloning/indexing
            max_tasks=max_tasks,
        )

        console.print(f"[bold]Setting up {benchmark} benchmark...[/]")
        tasks = await runner.load_tasks(config)
        console.print(f"Found {len(tasks)} tasks to setup")

        # Setup workspace manager
        workspace_manager = SWEBenchWorkspaceManager()

        # Group tasks by repo (to avoid duplicate clones)
        repos_seen = set()
        unique_tasks = []
        for task in tasks:
            if task.repo and task.repo not in repos_seen:
                repos_seen.add(task.repo)
                unique_tasks.append(task)

        console.print(f"Unique repositories: {len(unique_tasks)}")

        # Setup each unique repo
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            setup_task = progress.add_task("Setting up repos...", total=len(unique_tasks))

            for i, task in enumerate(unique_tasks):
                repo_name = (
                    task.repo.split("/")[-1].replace(".git", "") if task.repo else task.task_id
                )
                progress.update(setup_task, description=f"[{i+1}/{len(unique_tasks)}] {repo_name}")

                try:
                    # Check if already indexed
                    if not force_reindex and workspace_manager.is_repo_indexed(task):
                        console.print(f"  [green]✓[/] {repo_name} (cached)")
                    else:
                        await workspace_manager.setup_repo_with_indexes(
                            task,
                            force_reindex=force_reindex,
                        )
                        console.print(f"  [green]✓[/] {repo_name} (indexed)")
                except Exception as e:
                    console.print(f"  [red]✗[/] {repo_name}: {e}")

                progress.advance(setup_task)

        console.print("\n[bold green]Setup complete![/]")
        console.print(f"Cached repos: {workspace_manager.cache_dir}")
        console.print(f"\nNow run: victor benchmark run {benchmark}")

    asyncio.run(run_setup())


@benchmark_app.command("run")
def run_benchmark(
    benchmark: str = typer.Argument(..., help="Benchmark to run: swe-bench, humaneval, mbpp"),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-n", help="Maximum number of tasks to run"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use (default: from profile)"
    ),
    profile: str = typer.Option("default", "--profile", "-p", help="Victor profile to use"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for results (JSON)"
    ),
    timeout: int = typer.Option(300, "--timeout", "-t", help="Timeout per task in seconds"),
    max_turns: int = typer.Option(10, "--max-turns", help="Maximum conversation turns per task"),
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel tasks"),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="Resume from checkpoint if previous run was interrupted"
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
) -> None:
    """Run a benchmark evaluation."""
    _configure_log_level(log_level)

    from victor.config.settings import Settings
    from victor.evaluation.protocol import BenchmarkType, EvaluationConfig
    from victor.evaluation.benchmarks import (
        SWEBenchRunner,
        HumanEvalRunner,
        MBPPRunner,
    )

    # Map benchmark name to type and runner
    benchmark_map = {
        "swe-bench": (BenchmarkType.SWE_BENCH, SWEBenchRunner),
        "swe-bench-lite": (BenchmarkType.SWE_BENCH, lambda: SWEBenchRunner(split="lite")),
        "humaneval": (BenchmarkType.HUMAN_EVAL, HumanEvalRunner),
        "mbpp": (BenchmarkType.MBPP, MBPPRunner),
        "mbpp-test": (BenchmarkType.MBPP, lambda: MBPPRunner(split="test")),
    }

    benchmark_lower = benchmark.lower().replace("_", "-")
    if benchmark_lower not in benchmark_map:
        console.print(f"[bold red]Error:[/] Unknown benchmark: {benchmark}")
        console.print(f"Available: {', '.join(benchmark_map.keys())}")
        raise typer.Exit(1)

    bench_type, runner_factory = benchmark_map[benchmark_lower]
    runner = runner_factory() if callable(runner_factory) else runner_factory

    # Load profile to get model if not specified
    effective_model = model
    effective_provider = None
    if not effective_model:
        settings = Settings()
        try:
            profiles = settings.load_profiles()
            profile_config = profiles.get(profile)
            if profile_config:
                # ProfileConfig is a Pydantic model with .model_name attribute
                effective_model = profile_config.model_name
                effective_provider = profile_config.provider
            else:
                console.print(
                    f"[yellow]Warning:[/] Profile '{profile}' not found, using default model"
                )
                effective_model = "claude-3-sonnet"
        except Exception as e:
            console.print(f"[yellow]Warning:[/] Could not load profile: {e}")
            effective_model = "claude-3-sonnet"

    # Build config
    config = EvaluationConfig(
        benchmark=bench_type,
        model=effective_model,
        max_tasks=max_tasks,
        timeout_per_task=timeout,
        max_turns=max_turns,
        parallel_tasks=parallel,
    )

    console.print(f"\n[bold cyan]Running {benchmark} benchmark[/]")
    console.print(f"Model: {config.model}")
    if max_tasks:
        console.print(f"Max tasks: {max_tasks}")
    console.print(f"Timeout: {timeout}s per task")
    console.print()

    async def run_async():
        from victor.evaluation.harness import EvaluationHarness
        from victor.evaluation.agent_adapter import VictorAgentAdapter
        from victor.evaluation.protocol import BenchmarkTask

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading tasks...", total=None)

            # Create harness and register runner
            harness = EvaluationHarness()
            harness.register_runner(runner)

            # Load tasks first to show count
            tasks = await runner.load_tasks(config)
            progress.update(task, description=f"Loaded {len(tasks)} tasks")

            if not tasks:
                console.print("[yellow]No tasks to run[/]")
                return None

            # Create agent adapter from profile
            progress.update(task, description="Initializing agent...")
            try:
                adapter = VictorAgentAdapter.from_profile(
                    profile=profile,
                    model_override=model,  # Use explicit model if provided
                    timeout=timeout,
                )

                # Create workspace manager for SWE-bench (uses caching)
                from victor.evaluation.swe_bench_loader import SWEBenchWorkspaceManager

                workspace_manager = SWEBenchWorkspaceManager()

                # Create a callback that returns code AND metrics for token tracking
                async def agent_callback(benchmark_task: BenchmarkTask) -> dict:
                    """Run agent on task and return generated code with metrics.

                    Two-phase approach:
                    1. Setup phase (optional, run 'victor benchmark setup' first):
                       - Clones repo to ~/.victor/swe_bench_cache/<hash>/
                       - Builds indexes in repo's .victor/ directory
                    2. Execution phase (this callback):
                       - Uses cached repo with pre-built indexes
                       - Agent works in target repo, not victor's codebase

                    Returns a dict with:
                    - code: The generated patch or code
                    - tokens_input: Input tokens used
                    - tokens_output: Output tokens used
                    - tokens_used: Total tokens used
                    - tool_calls: Number of tool calls
                    - turns: Number of conversation turns
                    """
                    import os

                    # Check if repo is already setup (Phase 1 completed)
                    cached_repo = workspace_manager.get_cached_repo_path(benchmark_task)
                    if cached_repo and workspace_manager.is_repo_indexed(benchmark_task):
                        # Use cached+indexed repo directly
                        work_dir = cached_repo
                        console.print(f"  [dim]Using indexed repo: {cached_repo.name}[/]")
                    else:
                        # Setup on-the-fly (slower, but works without explicit setup)
                        console.print(
                            "  [dim]Setting up repo (run 'victor benchmark setup' for faster execution)...[/]"
                        )
                        await workspace_manager.setup_repo_with_indexes(benchmark_task)
                        work_dir = workspace_manager.get_cached_repo_path(benchmark_task)

                    # Checkout the specific base commit for this task
                    if benchmark_task.base_commit and work_dir:
                        checkout_proc = await asyncio.create_subprocess_exec(
                            "git",
                            "checkout",
                            benchmark_task.base_commit,
                            cwd=work_dir,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await checkout_proc.communicate()

                    # Change to target repo directory so agent uses its indexes
                    original_cwd = os.getcwd()
                    os.chdir(work_dir)
                    try:
                        trace = await adapter.execute_task(benchmark_task, work_dir)
                    except asyncio.CancelledError:
                        # On timeout/cancellation, store partial data for harness
                        partial = adapter.get_partial_trace()
                        logger.info(
                            "Task cancelled - partial trace: "
                            f"tool_calls={partial['tool_calls']}, turns={partial['turns']}, "
                            f"tokens={partial['tokens_used']}"
                        )
                        # Store partial data in a container the harness can access
                        # We use a mutable dict attached to the function object
                        setattr(
                            agent_callback,
                            "_partial_data",
                            {
                                "code": partial.get("code", ""),
                                "tokens_input": partial.get("tokens_input", 0),
                                "tokens_output": partial.get("tokens_output", 0),
                                "tokens_used": partial.get("tokens_used", 0),
                                "tool_calls": partial.get("tool_calls", 0),
                                "turns": partial.get("turns", 0),
                            },
                        )
                        # Re-raise so wait_for properly times out
                        raise
                    finally:
                        os.chdir(original_cwd)

                    # Return code with metrics for harness to populate TaskResult
                    return {
                        "code": trace.generated_patch or trace.generated_code or "",
                        "tokens_input": trace.token_usage.input_tokens,
                        "tokens_output": trace.token_usage.output_tokens,
                        "tokens_used": trace.token_usage.total_tokens,
                        "tool_calls": len(trace.tool_calls),
                        "turns": trace.turns,
                    }

            except Exception as e:
                console.print(f"[red]Error initializing agent:[/] {e}")
                import traceback

                traceback.print_exc()
                return None

            # Progress callback
            def on_progress(task_idx: int, total: int, result):
                progress.update(
                    task, description=f"Task {task_idx + 1}/{total}: {result.status.value}"
                )

            # Run evaluation
            if resume:
                progress.update(task, description="Resuming evaluation...")
            else:
                progress.update(task, description="Running evaluation...")
            result = await harness.run_evaluation(
                config=config,
                agent_callback=agent_callback,
                progress_callback=on_progress,
                resume=resume,
            )

            return result

    result = asyncio.run(run_async())

    if result is None:
        raise typer.Exit(1)

    # Display results
    console.print("\n[bold green]Results[/]")

    metrics = result.get_metrics()
    results_table = Table()
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="white")

    results_table.add_row("Total Tasks", str(metrics["total_tasks"]))
    results_table.add_row("Passed", f"[green]{metrics['passed']}[/]")
    results_table.add_row("Failed", f"[red]{metrics['failed']}[/]")
    results_table.add_row("Errors", f"[yellow]{metrics['errors']}[/]")
    results_table.add_row("Pass Rate", f"[bold]{metrics['pass_rate']:.1%}[/]")
    results_table.add_row("Duration", f"{metrics['duration_seconds']:.1f}s")
    results_table.add_row("Total Tokens", f"{metrics['total_tokens']:,}")

    console.print(results_table)

    # Save results if output specified
    if output:
        output_data = {
            "benchmark": benchmark,
            "model": config.model,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "task_results": [
                {
                    "task_id": r.task_id,
                    "status": r.status.value,
                    "tests_passed": r.tests_passed,
                    "tests_total": r.tests_total,
                    "duration": r.duration_seconds,
                }
                for r in result.task_results
            ],
        }
        output.write_text(json.dumps(output_data, indent=2))
        console.print(f"\n[dim]Results saved to {output}[/]")


@benchmark_app.command("compare")
def compare_frameworks(
    benchmark: str = typer.Option(
        "swe-bench", "--benchmark", "-b", help="Benchmark for comparison"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for comparison report"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, markdown, json"
    ),
) -> None:
    """Compare Victor against other AI coding frameworks."""
    from victor.evaluation.benchmarks import (
        Framework,
        FRAMEWORK_CAPABILITIES,
        PUBLISHED_RESULTS,
    )
    from victor.evaluation.protocol import BenchmarkType

    # Map benchmark name to type
    benchmark_type_map = {
        "swe-bench": BenchmarkType.SWE_BENCH,
        "humaneval": BenchmarkType.HUMAN_EVAL,
        "mbpp": BenchmarkType.MBPP,
    }

    benchmark_lower = benchmark.lower().replace("_", "-")
    if benchmark_lower not in benchmark_type_map:
        console.print(f"[bold red]Error:[/] Unknown benchmark: {benchmark}")
        raise typer.Exit(1)

    bench_type = benchmark_type_map[benchmark_lower]

    console.print(f"\n[bold cyan]Framework Comparison: {benchmark}[/]\n")

    # Get published results
    if bench_type not in PUBLISHED_RESULTS:
        console.print("[yellow]No published results available for this benchmark[/]")
        console.print("Run 'victor benchmark run' to generate Victor results")
        raise typer.Exit(0)

    table = Table(title=f"Published Results: {benchmark}")
    table.add_column("Framework", style="cyan")
    table.add_column("Model", style="white")
    table.add_column("Pass Rate", style="green")
    table.add_column("Source", style="dim")

    results = PUBLISHED_RESULTS[bench_type]
    for framework, data in sorted(
        results.items(), key=lambda x: x[1].get("pass_rate", 0), reverse=True
    ):
        table.add_row(
            framework.value,
            data.get("model", "unknown"),
            f"{data.get('pass_rate', 0):.1%}",
            data.get("source", ""),
        )

    console.print(table)

    if format == "markdown" and output:
        md_lines = [
            f"# Framework Comparison: {benchmark}",
            "",
            "| Framework | Model | Pass Rate | Source |",
            "|-----------|-------|-----------|--------|",
        ]
        for framework, data in results.items():
            md_lines.append(
                f"| {framework.value} | {data.get('model', 'unknown')} | "
                f"{data.get('pass_rate', 0):.1%} | {data.get('source', '')} |"
            )
        output.write_text("\n".join(md_lines))
        console.print(f"\n[dim]Report saved to {output}[/]")

    elif format == "json" and output:
        output.write_text(
            json.dumps(
                {
                    "benchmark": benchmark,
                    "results": {f.value: d for f, d in results.items()},
                },
                indent=2,
            )
        )
        console.print(f"\n[dim]Report saved to {output}[/]")


@benchmark_app.command("leaderboard")
def show_leaderboard(
    benchmark: str = typer.Option("swe-bench", "--benchmark", "-b", help="Benchmark to show"),
) -> None:
    """Show the leaderboard for a benchmark."""
    from victor.evaluation.benchmarks import PUBLISHED_RESULTS
    from victor.evaluation.protocol import BenchmarkType

    benchmark_type_map = {
        "swe-bench": BenchmarkType.SWE_BENCH,
        "humaneval": BenchmarkType.HUMAN_EVAL,
        "mbpp": BenchmarkType.MBPP,
    }

    benchmark_lower = benchmark.lower().replace("_", "-")
    if benchmark_lower not in benchmark_type_map:
        console.print(f"[bold red]Error:[/] Unknown benchmark: {benchmark}")
        raise typer.Exit(1)

    bench_type = benchmark_type_map[benchmark_lower]

    console.print(f"\n[bold cyan]Leaderboard: {benchmark}[/]\n")

    if bench_type not in PUBLISHED_RESULTS:
        console.print("[yellow]No results available[/]")
        raise typer.Exit(0)

    results = PUBLISHED_RESULTS[bench_type]
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("pass_rate", 0), reverse=True)

    table = Table()
    table.add_column("Rank", style="bold")
    table.add_column("Framework", style="cyan")
    table.add_column("Pass Rate", style="green")

    for i, (framework, data) in enumerate(sorted_results, 1):
        medal = ""
        if i == 1:
            medal = "🥇 "
        elif i == 2:
            medal = "🥈 "
        elif i == 3:
            medal = "🥉 "

        table.add_row(
            f"{medal}{i}",
            framework.value,
            f"{data.get('pass_rate', 0):.1%}",
        )

    console.print(table)
    console.print("\n[dim]Based on published benchmark results[/]")


@benchmark_app.command("capabilities")
def show_capabilities() -> None:
    """Show capabilities comparison across frameworks."""
    from victor.evaluation.benchmarks import Framework, FRAMEWORK_CAPABILITIES

    console.print("\n[bold cyan]Framework Capabilities Comparison[/]\n")

    table = Table()
    table.add_column("Capability", style="cyan")

    # Add framework columns
    frameworks = [Framework.VICTOR, Framework.AIDER, Framework.CLAUDE_CODE, Framework.CURSOR]
    for f in frameworks:
        table.add_column(f.value, justify="center")

    capabilities = [
        ("Code Generation", "code_generation"),
        ("Code Editing", "code_editing"),
        ("Multi-file Edit", "multi_file_editing"),
        ("Tool Use", "tool_use"),
        ("Autonomous Mode", "autonomous_mode"),
        ("Planning", "planning"),
        ("Local Models", "local_models"),
        ("Air-gapped", "air_gapped"),
        ("MCP Support", "mcp_support"),
        ("Open Source", "open_source"),
    ]

    for cap_name, cap_attr in capabilities:
        row = [cap_name]
        for f in frameworks:
            if f in FRAMEWORK_CAPABILITIES:
                cap = FRAMEWORK_CAPABILITIES[f]
                value = getattr(cap, cap_attr, False)
                row.append("[green]✓[/]" if value else "[red]✗[/]")
            else:
                row.append("[dim]?[/]")
        table.add_row(*row)

    console.print(table)
