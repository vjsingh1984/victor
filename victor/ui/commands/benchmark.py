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
    benchmark: str = typer.Argument(..., help="Benchmark to run: swe-bench, humaneval, mbpp"),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-n", help="Maximum number of tasks to run"
    ),
    start_task: int = typer.Option(
        0, "--start-task", help="Skip first N tasks (0-indexed, for targeting specific tasks)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use (default: from profile)"
    ),
    profile: str = typer.Option("default", "--profile", "-p", help="Victor profile to use"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for results (JSON)"
    ),
    timeout: int = typer.Option(420, "--timeout", "-t", help="Timeout per task in seconds"),
    max_turns: int = typer.Option(10, "--max-turns", help="Maximum conversation turns per task"),
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
    # Resolve start_task — when called programmatically (not via Typer CLI),
    # the default may be a typer.OptionInfo rather than int.
    if not isinstance(start_task, int):
        start_task = 0
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

    # Extended token metrics (if available)
    cached = metrics.get("cached_tokens", 0)
    reasoning = metrics.get("reasoning_tokens", 0)
    cost_micros = metrics.get("cost_usd_micros", 0)
    if cached > 0:
        results_table.add_row("Cached Tokens", f"[dim]{cached:,}[/]")
    if reasoning > 0:
        results_table.add_row("Reasoning Tokens", f"[dim]{reasoning:,}[/]")
    if cost_micros > 0:
        cost_usd = cost_micros / 1_000_000
        results_table.add_row("API Cost", f"[dim]${cost_usd:.4f}[/]")

    console.print(results_table)

    # Auto-evolve: run GEPA evolution after benchmark completes
    # This is the "post-session hook" from Memory Scaling — the agent
    # improves its prompts automatically after each benchmark run.
    if metrics["total_tasks"] >= 5:  # Only evolve with enough data
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("prompt_optimizer")
            if learner:
                from victor.agent.prompt_builder import ASI_TOOL_EFFECTIVENESS_GUIDANCE

                # Determine provider from model name
                model_lower = (config.model or "").lower()
                provider = "default"
                for prefix, prov in [
                    ("gpt", "openai"), ("grok", "xai"),
                    ("deepseek", "deepseek"), ("haiku", "anthropic"),
                    ("claude", "anthropic"),
                ]:
                    if prefix in model_lower:
                        provider = prov
                        break

                candidate = learner.evolve(
                    "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
                    ASI_TOOL_EFFECTIVENESS_GUIDANCE,
                    provider=provider,
                )
                if candidate:
                    # Seed from this run's results
                    for _ in range(metrics["passed"]):
                        candidate.update(True)
                    for _ in range(metrics["failed"] + metrics.get("errors", 0)):
                        candidate.update(False)
                    learner._save_candidate(candidate)
                    console.print(
                        f"\n[dim]Auto-evolved prompt gen-{candidate.generation} "
                        f"for {provider} (mean={candidate.mean:.2f})[/]"
                    )
        except Exception as e:
            logger.debug("Auto-evolve skipped: %s", e)

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
        SWEBenchRunner(split="lite") if benchmark_lower == "swe-bench-lite" else SWEBenchRunner()
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
            repo_name = task.repo.split("/")[-1].replace(".git", "") if task.repo else task.task_id
            progress.update(setup_task, description=f"[{i+1}/{len(unique_tasks)}] {repo_name}")

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
            console.print(f"[dim]Skipped {start_task} tasks (starting from index {start_task})[/]")
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

            _edge_enabled = get_feature_flag_manager().is_enabled(FeatureFlag.USE_EDGE_MODEL)
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
                    hasattr(provider, "supports_thinking") and provider.supports_thinking()
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

            _warmed_repos: set = set()  # Track repos that have been pre-warmed this session

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
                            console.print(f"  [yellow]Git command timed out: {git_cmd[1]}[/]")

                # Pre-warm code_search index BEFORE task timer starts.
                # This is a fixed cost — NOT deducted from the per-task timeout.
                # Only do full pre-warm once per repo; subsequent tasks skip.
                repo_key = str(work_dir) if work_dir else ""
                if repo_key in _warmed_repos:
                    logger.info(
                        "Index already warmed for %s, skipping pre-warm",
                        work_dir.name if work_dir else "?",
                    )
                    console.print("  Code search index pre-warmed (cached)")
                else:
                    try:
                        from victor.tools.code_search_tool import _get_or_build_index
                        from victor.config.settings import load_settings

                        _prewarm_settings = load_settings()
                        await asyncio.wait_for(
                            _get_or_build_index(work_dir, _prewarm_settings, force_reindex=False),
                            timeout=300,  # Allow up to 5 min — outside task timer
                        )
                        _warmed_repos.add(repo_key)
                        console.print("  Code search index pre-warmed")
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Index pre-warm timed out after 300s, code_search may be slow"
                        )
                        _warmed_repos.add(repo_key)  # Don't retry on next task
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


@benchmark_app.command("evolve")
def evolve_prompts(
    provider: str = typer.Option("all", "--provider", "-p", help="Provider to evolve (or 'all')"),
    section: str = typer.Option("all", "--section", "-s", help="Section to evolve (or 'all')"),
    compliance: bool = typer.Option(False, "--compliance", help="Show GEPA compliance scorecard"),
) -> None:
    """Evolve prompts using GEPA + benchmark trace data.

    Reads execution traces from usage.jsonl and evaluation results,
    then evolves prompt sections using the configured strategy
    (GEPA, MIPROv2, CoT distillation).

    Examples:
        victor benchmark evolve                 # Evolve all
        victor benchmark evolve -p openai       # Evolve for OpenAI
        victor benchmark evolve --compliance    # Show compliance
    """
    from rich.table import Table

    try:
        from victor.framework.rl.coordinator import get_rl_coordinator

        coordinator = get_rl_coordinator()
        learner = coordinator.get_learner("prompt_optimizer")
        if learner is None:
            console.print("[yellow]Prompt optimizer not available[/]")
            return

        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner
        from victor.agent.prompt_builder import ASI_TOOL_EFFECTIVENESS_GUIDANCE

        sections = PromptOptimizerLearner.EVOLVABLE_SECTIONS
        if section != "all":
            matched = [s for s in sections if section.upper() in s]
            sections = matched if matched else sections

        providers = ["openai", "xai", "deepseek", "anthropic"]
        if provider != "all":
            providers = [provider]

        section_text = {
            "ASI_TOOL_EFFECTIVENESS_GUIDANCE": ASI_TOOL_EFFECTIVENESS_GUIDANCE,
            "GROUNDING_RULES": "",
            "COMPLETION_GUIDANCE": "",
            "FEW_SHOT_EXAMPLES": "",
        }

        results = Table(title="GEPA Evolution Results")
        results.add_column("Provider", style="cyan")
        results.add_column("Section", style="dim")
        results.add_column("Gen", style="green")
        results.add_column("Mean", style="bold")
        results.add_column("Chars", style="dim")

        # Load ALL evaluation results (not just latest) to compute
        # aggregate pass rates per provider across all benchmark runs.
        import glob as _glob
        import json as _json
        from pathlib import Path as _Path

        eval_dir = _Path.home() / ".victor" / "evaluations"
        provider_agg = {}  # provider → {"pass": N, "fail": N}
        model_to_provider = {
            "gpt": "openai",
            "grok": "xai",
            "deepseek": "deepseek",
            "haiku": "anthropic",
            "claude": "anthropic",
        }
        for ef in sorted(_glob.glob(str(eval_dir / "eval_swe_bench_*.json"))):
            try:
                with open(ef) as _f:
                    edata = _json.load(_f)
                emodel = edata.get("config", {}).get("model", "")
                etasks = edata.get("tasks", [])
                # Map model to provider
                p_name = None
                for prefix, prov in model_to_provider.items():
                    if prefix in emodel.lower():
                        p_name = prov
                        break
                if not p_name or len(etasks) < 1:
                    continue
                if p_name not in provider_agg:
                    provider_agg[p_name] = {"pass": 0, "fail": 0}
                for t in etasks:
                    if t.get("status") == "passed":
                        provider_agg[p_name]["pass"] += 1
                    else:
                        provider_agg[p_name]["fail"] += 1
            except Exception:
                pass

        # Also collect trace-based success from usage.jsonl
        traces = learner._collect_traces(limit=200)
        trace_pass = sum(1 for t in traces if t.success)
        trace_fail = len(traces) - trace_pass
        console.print(
            f"[dim]Seeding from {len(list(_glob.glob(str(eval_dir / 'eval_*.json'))))} "
            f"eval files + {len(traces)} traces[/]"
        )

        # Default seed for providers without benchmark data
        default_seed = (max(trace_pass, 5), max(trace_fail, 3))

        for p in providers:
            for s in sections:
                current = section_text.get(s, "")
                candidate = learner.evolve(s, current, provider=p)
                if candidate:
                    # Seed from aggregate benchmark data or defaults
                    agg = provider_agg.get(p)
                    if agg and (agg["pass"] + agg["fail"]) > 0:
                        # Scale to max 15 to avoid overwhelming priors
                        total = agg["pass"] + agg["fail"]
                        scale = min(15 / max(total, 1), 1.0)
                        passes = max(1, int(agg["pass"] * scale))
                        fails = max(1, int(agg["fail"] * scale))
                    else:
                        # No benchmark data — use trace-based defaults
                        passes, fails = default_seed
                    for _ in range(passes):
                        candidate.update(True)
                    for _ in range(fails):
                        candidate.update(False)
                    learner._save_candidate(candidate)
                    results.add_row(
                        p,
                        s[:20],
                        str(candidate.generation),
                        f"{candidate.mean:.2f}",
                        str(len(candidate.text)),
                    )
                else:
                    results.add_row(p, s[:20], "-", "-", "[dim]no change[/]")

        console.print(results)
        metrics = learner.export_metrics()
        console.print(f"\n[dim]Total candidates: {metrics['total_candidates']}[/]")

    except Exception as e:
        console.print(f"[red]Evolution failed:[/] {e}")
        import traceback

        console.print(traceback.format_exc())

    if compliance:
        _show_compliance_scorecard()


def _show_compliance_scorecard() -> None:
    """Show GEPA prompt compliance analysis from usage.jsonl traces."""
    import gzip
    import json
    from collections import defaultdict
    from pathlib import Path
    from rich.table import Table

    events = []
    logs_dir = Path.home() / ".victor" / "logs"
    for path in sorted(logs_dir.glob("usage.*.jsonl.gz")) + [logs_dir / "usage.jsonl"]:
        if not path.exists():
            continue
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            mode = "rt" if path.suffix == ".gz" else "r"
            with opener(str(path), mode) as f:
                for line in f:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass

    sessions = defaultdict(list)
    for e in events:
        if e.get("event_type") in ("tool_call", "tool_result"):
            sessions[e.get("session_id", "?")].append(e)

    # Rule checks
    total_sessions = len(sessions)
    search_first = sum(
        1
        for evts in sessions.values()
        if any(
            e.get("data", {}).get("tool_name") == "code_search"
            for e in evts
            if e.get("event_type") == "tool_call"
        )
        and next(
            (
                e.get("data", {}).get("tool_name")
                for e in evts
                if e.get("event_type") == "tool_call"
            ),
            None,
        )
        == "code_search"
    )

    shell_total = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call" and e.get("data", {}).get("tool_name") == "shell"
    )
    shell_search = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call"
        and e.get("data", {}).get("tool_name") == "shell"
        and any(
            s in str(e.get("data", {}).get("tool_args", {}).get("cmd", "")).lower()
            for s in ["grep ", "rg ", " ag "]
        )
    )

    total_reads = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call" and e.get("data", {}).get("tool_name") == "read"
    )
    victor_reads = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call"
        and e.get("data", {}).get("tool_name") == "read"
        and "victor" in str(e.get("data", {}).get("tool_args", {}).get("path", "")).lower()
        and "swe_bench" not in str(e.get("data", {}).get("tool_args", {}))
    )

    cs_total = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call"
        and e.get("data", {}).get("tool_name") == "code_search"
    )
    cs_semantic = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call"
        and e.get("data", {}).get("tool_name") == "code_search"
        and e.get("data", {}).get("tool_args", {}).get("mode", "semantic") == "semantic"
    )

    table = Table(title="GEPA Compliance Scorecard")
    table.add_column("Rule", style="cyan")
    table.add_column("Compliance", style="bold")
    table.add_column("Target", style="dim")

    rules = [
        ("Search first", search_first * 100 // max(total_sessions, 1), 50),
        (
            "No shell search",
            (shell_total - shell_search) * 100 // max(shell_total, 1) if shell_total else 100,
            80,
        ),
        (
            "No workspace contamination",
            (total_reads - victor_reads) * 100 // max(total_reads, 1),
            95,
        ),
        (
            "Semantic search preferred",
            cs_semantic * 100 // max(cs_total, 1) if cs_total else 100,
            80,
        ),
    ]

    for name, pct, target in rules:
        status = (
            "[green]✓[/]"
            if pct >= target
            else "[yellow]⚠[/]" if pct >= target * 0.6 else "[red]✗[/]"
        )
        table.add_row(f"{status} {name}", f"{pct}%", f"{target}%")

    console.print(f"\n")
    console.print(table)

    # Efficiency scaling: tool calls per session over time
    session_tools = []
    for sid, evts in sessions.items():
        tc = sum(1 for e in evts if e.get("event_type") == "tool_call")
        if tc > 0:
            session_tools.append(tc)

    if len(session_tools) >= 4:
        first_half = session_tools[: len(session_tools) // 2]
        second_half = session_tools[len(session_tools) // 2 :]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        delta = avg_second - avg_first
        direction = "↓" if delta < 0 else "↑" if delta > 0 else "="
        console.print(
            f"[dim]Efficiency: {avg_first:.0f} → {avg_second:.0f} tools/session "
            f"({direction}{abs(delta):.0f}) across {len(session_tools)} sessions[/]"
        )

    console.print(f"[dim]Based on {len(events):,} events across {total_sessions} sessions[/]")
