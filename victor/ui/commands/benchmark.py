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

from victor.core.async_utils import run_sync
from victor.ui.commands.utils import setup_logging

benchmark_app = typer.Typer(
    name="benchmark",
    help="Run AI coding benchmarks and compare against other frameworks.",
)
console = Console()


def _configure_log_level(
    log_level: Optional[str],
    command: str = "benchmark",
    debug_modules: Optional[str] = None,
) -> None:
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
    setup_logging(
        command=command,
        cli_log_level=log_level,
        stream=sys.stderr,
        cli_debug_modules=debug_modules,
    )


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
    benchmark: str = typer.Argument(
        ..., help="Benchmark to setup: swe-bench, swe-bench-lite"
    ),
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

    # Only SWE-bench needs setup (repo cloning)
    benchmark_lower = benchmark.lower().replace("_", "-")
    if benchmark_lower not in ("swe-bench", "swe-bench-lite"):
        console.print(f"[yellow]Setup not needed for {benchmark}[/]")
        console.print("Only SWE-bench benchmarks require repo setup.")
        return
    run_sync(
        _setup_benchmark_async(
            benchmark=benchmark,
            benchmark_lower=benchmark_lower,
            max_tasks=max_tasks,
            force_reindex=force_reindex,
        )
    )


async def _run_git_with_timeout(cmd, cwd, timeout=60):
    """Run a git command with timeout protection.

    Kills the process if it doesn't complete within timeout seconds.
    Prevents silent hangs from blocking the benchmark pipeline.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, stdout, stderr
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning("Git command timed out after %ds: %s", timeout, " ".join(cmd))
        raise


@benchmark_app.command("run")
def run_benchmark(
    benchmark: str = typer.Argument(
        ..., help="Benchmark to run: swe-bench, humaneval, mbpp"
    ),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-n", help="Maximum number of tasks to run"
    ),
    start_task: int = typer.Option(
        0, "--start-task", help="Skip first N tasks (0-indexed, for targeting specific tasks)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use (default: from profile)"
    ),
    profile: str = typer.Option(
        "default", "--profile", "-p", help="Victor profile to use"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for results (JSON)"
    ),
    timeout: int = typer.Option(
        300, "--timeout", "-t", help="Timeout per task in seconds"
    ),
    max_turns: int = typer.Option(
        10, "--max-turns", help="Maximum conversation turns per task"
    ),
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel tasks"),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from checkpoint if previous run was interrupted",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Override provider (e.g., deepseek, openai, xai)"
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
    debug_modules: Optional[str] = typer.Option(
        None,
        "--debug-modules",
        help="Comma-separated modules for DEBUG logging (e.g. code_search,agent_adapter)",
    ),
    no_edge_model: bool = typer.Option(
        False,
        "--no-edge-model",
        help="Disable edge model micro-decisions during benchmark",
    ),
) -> None:
    """Run a benchmark evaluation."""
    _configure_log_level(log_level, debug_modules=debug_modules)

    # Disable edge model if requested
    if no_edge_model:
        from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

        get_feature_flag_manager().disable(FeatureFlag.USE_EDGE_MODEL)

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
        "swe-bench-lite": (
            BenchmarkType.SWE_BENCH,
            lambda: SWEBenchRunner(split="lite"),
        ),
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
                # ProfileConfig is a Pydantic model with .model attribute
                effective_model = profile_config.model
                effective_provider = profile_config.provider
            else:
                console.print(
                    f"[yellow]Warning:[/] Profile '{profile}' not found, using default model"
                )
                effective_model = "claude-3-sonnet"
        except Exception as e:
            console.print(f"[yellow]Warning:[/] Could not load profile: {e}")
            effective_model = "claude-3-sonnet"

    # Build config — account for start_task offset in max_tasks
    effective_max_tasks = max_tasks
    if start_task > 0 and max_tasks is not None:
        effective_max_tasks = max_tasks + start_task

    config = EvaluationConfig(
        benchmark=bench_type,
        model=effective_model,
        max_tasks=effective_max_tasks,
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
    result = run_sync(
        _run_benchmark_async(
            runner=runner,
            config=config,
            profile=profile,
            model=model,
            timeout=timeout,
            max_turns=max_turns,
            resume=resume,
            provider_override=provider,
            start_task=start_task,
        )
    )

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


async def _setup_benchmark_async(
    *,
    benchmark: str,
    benchmark_lower: str,
    max_tasks: Optional[int],
    force_reindex: bool,
) -> None:
    from victor.evaluation.protocol import BenchmarkType, EvaluationConfig
    from victor.evaluation.benchmarks import SWEBenchRunner
    from victor.evaluation.swe_bench_loader import SWEBenchWorkspaceManager

    runner = (
        SWEBenchRunner(split="lite")
        if benchmark_lower == "swe-bench-lite"
        else SWEBenchRunner()
    )
    config = EvaluationConfig(
        benchmark=BenchmarkType.SWE_BENCH,
        model="setup-only",
        max_tasks=max_tasks,
    )

    console.print(f"[bold]Setting up {benchmark} benchmark...[/]")
    tasks = await runner.load_tasks(config)
    console.print(f"Found {len(tasks)} tasks to setup")

    workspace_manager = SWEBenchWorkspaceManager()
    repos_seen = set()
    unique_tasks = []
    for task in tasks:
        if task.repo and task.repo not in repos_seen:
            repos_seen.add(task.repo)
            unique_tasks.append(task)

    console.print(f"Unique repositories: {len(unique_tasks)}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        setup_task = progress.add_task("Setting up repos...", total=len(unique_tasks))

        for i, task in enumerate(unique_tasks):
            repo_name = (
                task.repo.split("/")[-1].replace(".git", "")
                if task.repo
                else task.task_id
            )
            progress.update(
                setup_task, description=f"[{i+1}/{len(unique_tasks)}] {repo_name}"
            )

            try:
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


async def _run_benchmark_async(
    *,
    runner,
    config,
    profile: str,
    model: Optional[str],
    timeout: int,
    max_turns: int = 10,
    resume: bool,
    provider_override: Optional[str] = None,
    start_task: int = 0,
):
    from victor.evaluation.harness import EvaluationHarness
    from victor.evaluation.agent_adapter import VictorAgentAdapter
    from victor.evaluation.protocol import BenchmarkTask
    from victor.evaluation.swe_bench_loader import SWEBenchWorkspaceManager

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading tasks...", total=None)

        harness = EvaluationHarness()
        harness.register_runner(runner)

        tasks = await runner.load_tasks(config)

        # Skip tasks for targeted execution (--start-task)
        if start_task > 0 and start_task < len(tasks):
            tasks = tasks[start_task:]
            console.print(
                f"[dim]Skipped {start_task} tasks (starting from index {start_task})[/]"
            )
            # Override runner.load_tasks to return filtered set
            # (harness calls load_tasks internally)
            _original_load = runner.load_tasks

            async def _filtered_load(cfg):
                return tasks

            runner.load_tasks = _filtered_load

        progress.update(task, description=f"Loaded {len(tasks)} tasks")

        if not tasks:
            console.print("[yellow]No tasks to run[/]")
            return None

        progress.update(task, description="Initializing agent...")
        try:
            from victor.evaluation.agent_adapter import AdapterConfig

            adapter_config = AdapterConfig(
                total_timeout=timeout,
                max_turns=max_turns,
                min_turn_timeout=max(240, timeout // max(max_turns, 1)),
            )
            # Bootstrap edge model decision service into the container
            # so tool selection, prompt focus, and stage detection can use it.
            from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

            _edge_enabled = get_feature_flag_manager().is_enabled(
                FeatureFlag.USE_EDGE_MODEL
            )
            if _edge_enabled:
                try:
                    from victor.agent.edge_model import (
                        EdgeModelConfig,
                        create_edge_decision_service,
                    )
                    from victor.core import get_container
                    from victor.agent.services.protocols.decision_service import (
                        LLMDecisionServiceProtocol,
                    )
                    from victor.core.container import ServiceLifetime

                    edge_service = create_edge_decision_service(EdgeModelConfig())
                    if edge_service:
                        container = get_container()
                        container.register(
                            LLMDecisionServiceProtocol,
                            lambda c: edge_service,
                            ServiceLifetime.SINGLETON,
                        )
                        console.print("[dim]Edge model enabled for micro-decisions[/]")
                except Exception as e:
                    console.print(f"[dim]Edge model unavailable: {e}[/]")
            else:
                console.print("[dim]Edge model disabled (--no-edge-model)[/]")

            if provider_override:
                # Direct provider creation bypassing profile's provider
                from victor.config.settings import load_settings
                from victor.config.api_keys import get_api_key
                from victor.providers.registry import ProviderRegistry
                from victor.agent.orchestrator import AgentOrchestrator

                settings = load_settings()
                profiles = settings.load_profiles()
                profile_config = profiles.get(profile, profiles.get("default"))
                # Use explicit --model, else provider's default model
                _PROVIDER_DEFAULT_MODELS = {
                    "deepseek": "deepseek-chat",
                    "anthropic": "claude-sonnet-4-20250514",
                    "openai": "gpt-4o",
                    "google": "gemini-2.0-flash",
                    "xai": "grok-3",
                    "ollama": "gemma4:31b",
                }
                if model:
                    effective_model = model
                else:
                    effective_model = _PROVIDER_DEFAULT_MODELS.get(provider_override)
                    if not effective_model and profile_config:
                        effective_model = profile_config.model
                    if not effective_model:
                        effective_model = "deepseek-chat"

                api_key = get_api_key(provider_override)
                provider = ProviderRegistry.create(
                    provider_override,
                    api_key=api_key,
                    settings=settings,
                    timeout=timeout,
                )

                # Only enable thinking for providers that support it
                use_thinking = (
                    hasattr(provider, "supports_thinking")
                    and provider.supports_thinking()
                )

                orchestrator = AgentOrchestrator(
                    settings=settings,
                    provider=provider,
                    model=effective_model,
                    provider_name=provider_override,
                    thinking=use_thinking,
                )
                adapter = VictorAgentAdapter(orchestrator, adapter_config)
            else:
                adapter = VictorAgentAdapter.from_profile(
                    profile=profile,
                    model_override=model,
                    timeout=timeout,
                    config=adapter_config,
                )
            workspace_manager = SWEBenchWorkspaceManager()

            async def agent_callback(benchmark_task: BenchmarkTask) -> dict:
                """Run agent on task and return generated code with metrics."""
                cached_repo = workspace_manager.get_cached_repo_path(benchmark_task)
                if cached_repo and workspace_manager.is_repo_indexed(benchmark_task):
                    work_dir = cached_repo
                    console.print(f"  [dim]Using indexed repo: {cached_repo.name}[/]")
                else:
                    console.print(
                        "  [dim]Setting up repo (run 'victor benchmark setup' for faster execution)...[/]"
                    )
                    await workspace_manager.setup_repo_with_indexes(benchmark_task)
                    work_dir = workspace_manager.get_cached_repo_path(benchmark_task)

                if benchmark_task.base_commit and work_dir:
                    # Reset workspace to base_commit with clean working tree.
                    # Fetch the specific commit first (may be outside shallow depth).
                    for git_cmd in [
                        [
                            "git",
                            "fetch",
                            "--depth",
                            "1",
                            "origin",
                            benchmark_task.base_commit,
                        ],
                        ["git", "checkout", "--force", benchmark_task.base_commit],
                        ["git", "clean", "-fd", "-e", ".victor"],
                    ]:
                        try:
                            await _run_git_with_timeout(git_cmd, work_dir, timeout=60)
                        except asyncio.TimeoutError:
                            console.print(
                                f"  [yellow]Git command timed out: {git_cmd[1]}[/]"
                            )

                # Pre-warm code_search index BEFORE task timer starts.
                # Timeout protected — if pre-warm hangs, skip and continue.
                try:
                    from victor.tools.code_search_tool import _get_or_build_index
                    from victor.config.settings import load_settings

                    _prewarm_settings = load_settings()
                    await asyncio.wait_for(
                        _get_or_build_index(
                            work_dir, _prewarm_settings, force_reindex=False
                        ),
                        timeout=120,
                    )
                    console.print("  [dim]Code search index pre-warmed[/]")
                except asyncio.TimeoutError:
                    logger.warning("Index pre-warm timed out after 120s, skipping")
                except Exception as e:
                    logger.debug(f"Index pre-warm skipped: {e}")

                original_cwd = os.getcwd()
                os.chdir(work_dir)
                try:
                    trace = await adapter.execute_task(benchmark_task, work_dir)
                except asyncio.CancelledError:
                    partial = adapter.get_partial_trace()
                    logger.info(
                        "Task cancelled - partial trace: tool_calls=%s, turns=%s, tokens=%s",
                        partial["tool_calls"],
                        partial["turns"],
                        partial["tokens_used"],
                    )
                    agent_callback._partial_data = {
                        "code": partial.get("code", ""),
                        "tokens_input": partial.get("tokens_input", 0),
                        "tokens_output": partial.get("tokens_output", 0),
                        "tokens_used": partial.get("tokens_used", 0),
                        "tool_calls": partial.get("tool_calls", 0),
                        "turns": partial.get("turns", 0),
                    }
                    raise
                finally:
                    os.chdir(original_cwd)

                # Capture actual file changes as a git diff patch.
                # The agent edits files via the edit tool (modifies on disk),
                # but the evaluation harness needs a patch string to apply
                # on a fresh repo clone and run tests.
                patch = trace.generated_patch or trace.generated_code or ""
                if not patch and work_dir:
                    try:
                        diff_proc = await asyncio.create_subprocess_exec(
                            "git",
                            "diff",
                            cwd=work_dir,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        diff_out, _ = await diff_proc.communicate()
                        if diff_out:
                            patch = diff_out.decode("utf-8", errors="replace")
                            logger.info(
                                "Captured git diff patch: %d bytes from workspace",
                                len(patch),
                            )
                    except Exception as e:
                        logger.debug("Failed to capture git diff: %s", e)

                return {
                    "code": patch,
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

        def on_progress(task_idx: int, total: int, result):
            progress.update(
                task,
                description=f"Task {task_idx + 1}/{total}: {result.status.value}",
            )

        progress.update(
            task,
            description="Resuming evaluation..." if resume else "Running evaluation...",
        )
        return await harness.run_evaluation(
            config=config,
            agent_callback=agent_callback,
            progress_callback=on_progress,
            resume=resume,
        )


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
    benchmark: str = typer.Option(
        "swe-bench", "--benchmark", "-b", help="Benchmark to show"
    ),
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
    sorted_results = sorted(
        results.items(), key=lambda x: x[1].get("pass_rate", 0), reverse=True
    )

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
    frameworks = [
        Framework.VICTOR,
        Framework.AIDER,
        Framework.CLAUDE_CODE,
        Framework.CURSOR,
    ]
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
