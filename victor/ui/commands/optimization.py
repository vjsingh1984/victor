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

"""CLI commands for workflow and prompt optimization.

This module provides command-line interface commands for interacting
with the workflow and prompt optimization systems.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import click

from victor.core.async_utils import run_sync
from victor.framework.rl import create_prompt_rollout_experiment_async
from victor.optimization import (
    WorkflowOptimizer,
    OptimizationConfig,
)
from victor.experiments.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


@click.group()
def opt():
    """Workflow optimization commands."""
    pass


@opt.command()
@click.argument("workflow_id")
@click.option(
    "--min-executions",
    default=3,
    help="Minimum number of executions required for analysis",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file for profile (JSON format)",
)
def profile(
    workflow_id: str,
    min_executions: int,
    output: Optional[str],
):
    """Profile a workflow to detect bottlenecks.

    Example:
        victor opt profile my_workflow --min-executions 5
    """
    run_sync(_profile_async(workflow_id, min_executions, output))


@opt.command()
@click.argument("workflow_id")
@click.option(
    "--min-executions",
    default=3,
    help="Minimum number of executions required",
)
@click.option(
    "--max-suggestions",
    default=10,
    help="Maximum number of suggestions to return",
)
@click.option(
    "--min-confidence",
    default=0.6,
    type=float,
    help="Minimum confidence threshold",
)
@click.option(
    "--min-improvement",
    default=0.1,
    type=float,
    help="Minimum expected improvement threshold",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file for suggestions (JSON format)",
)
def suggest(
    workflow_id: str,
    min_executions: int,
    max_suggestions: int,
    min_confidence: float,
    min_improvement: float,
    output: Optional[str],
):
    """Get optimization suggestions for a workflow.

    Example:
        victor opt suggest my_workflow --max-suggestions 5
    """
    run_sync(
        _suggest_async(
            workflow_id=workflow_id,
            min_executions=min_executions,
            max_suggestions=max_suggestions,
            min_confidence=min_confidence,
            min_improvement=min_improvement,
            output=output,
        )
    )


@opt.command()
@click.argument("workflow_id")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--algorithm",
    default="hill_climbing",
    type=click.Choice(["hill_climbing", "simulated_annealing"]),
    help="Search algorithm to use",
)
@click.option(
    "--max-iterations",
    default=50,
    help="Maximum iterations for search",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Validate variants before applying",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file for optimized variant (JSON format)",
)
def optimize(
    workflow_id: str,
    config_file: str,
    algorithm: str,
    max_iterations: int,
    validate: bool,
    output: Optional[str],
):
    """Run optimization on a workflow.

    Example:
        victor opt optimize my_workflow config.json --algorithm hill_climbing
    """
    run_sync(
        _optimize_async(
            workflow_id=workflow_id,
            config_file=config_file,
            algorithm=algorithm,
            max_iterations=max_iterations,
            validate=validate,
            output=output,
        )
    )


@opt.command()
@click.argument("variant_file", type=click.Path(exists=True))
@click.argument("workflow_id")
@click.option(
    "--test-inputs",
    type=click.Path(exists=True),
    help="File containing test inputs for validation (JSON format)",
)
def validate(
    variant_file: str,
    workflow_id: str,
    test_inputs: Optional[str],
):
    """Validate a workflow variant.

    Example:
        victor opt validate variant.json my_workflow --test-inputs tests.json
    """
    run_sync(_validate_async(variant_file, workflow_id, test_inputs))


@opt.command("prompt-rollout")
@click.argument("section_name")
@click.argument("provider")
@click.argument("treatment_hash")
@click.option(
    "--control-hash",
    default=None,
    help="Optional control candidate hash. Uses current active candidate when omitted.",
)
@click.option(
    "--traffic-split",
    default=0.1,
    type=click.FloatRange(min=0.0, max=1.0, min_open=True, max_open=True),
    help="Traffic share to allocate to the treatment candidate.",
)
@click.option(
    "--min-samples-per-variant",
    default=50,
    type=click.IntRange(min=1),
    help="Minimum samples per variant before promotion analysis.",
)
def prompt_rollout(
    section_name: str,
    provider: str,
    treatment_hash: str,
    control_hash: Optional[str],
    traffic_split: float,
    min_samples_per_variant: int,
) -> None:
    """Start a benchmark-gated prompt rollout experiment.

    Example:
        victor opt prompt-rollout GROUNDING_RULES anthropic candidate_hash
    """
    if not 0.0 < traffic_split < 1.0:
        raise click.BadParameter(
            "traffic_split must be between 0 and 1 (exclusive)",
            param_hint="traffic_split",
        )
    if min_samples_per_variant < 1:
        raise click.BadParameter(
            "min_samples_per_variant must be at least 1",
            param_hint="min_samples_per_variant",
        )

    run_sync(
        _prompt_rollout_async(
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
            control_hash=control_hash,
            traffic_split=traffic_split,
            min_samples_per_variant=min_samples_per_variant,
        )
    )


async def _profile_async(
    workflow_id: str,
    min_executions: int,
    output: Optional[str],
) -> None:
    tracker = ExperimentTracker()
    optimizer = WorkflowOptimizer(experiment_tracker=tracker)

    click.echo(f"Profiling workflow: {workflow_id}")
    workflow_profile = await optimizer.analyze_workflow(
        workflow_id=workflow_id,
        min_executions=min_executions,
    )

    if not workflow_profile:
        click.echo(f"Unable to profile workflow: {workflow_id}", err=True)
        return

    click.echo(f"\n{'='*60}")
    click.echo(f"Workflow Profile: {workflow_id}")
    click.echo(f"{'='*60}")
    click.echo(f"Total duration: {workflow_profile.total_duration:.2f}s")
    click.echo(f"Total cost: ${workflow_profile.total_cost:.4f}")
    click.echo(f"Total tokens: {workflow_profile.total_tokens}")
    click.echo(f"Success rate: {workflow_profile.success_rate:.1%}")
    click.echo(f"Executions analyzed: {workflow_profile.num_executions}")

    if workflow_profile.bottlenecks:
        click.echo(f"\nBottlenecks detected ({len(workflow_profile.bottlenecks)}):")
        for i, bottleneck in enumerate(workflow_profile.bottlenecks, 1):
            click.echo(f"  {i}. {bottleneck}")
            click.echo(f"     Severity: {bottleneck.severity.value}")
            click.echo(f"     Suggestion: {bottleneck.suggestion}")
    else:
        click.echo("\nNo bottlenecks detected!")

    if workflow_profile.opportunities:
        click.echo(f"\nOptimization opportunities ({len(workflow_profile.opportunities)}):")
        for i, opportunity in enumerate(workflow_profile.opportunities[:5], 1):
            click.echo(f"  {i}. {opportunity.strategy_type.value}: {opportunity.target}")
            click.echo(f"     Expected improvement: {opportunity.expected_improvement:.1%}")
            click.echo(f"     Risk level: {opportunity.risk_level.value}")
            click.echo(f"     Confidence: {opportunity.confidence:.1%}")
    else:
        click.echo("\nNo optimization opportunities identified!")

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(workflow_profile.to_dict(), f, indent=2)
        click.echo(f"\nProfile saved to: {output}")


async def _suggest_async(
    *,
    workflow_id: str,
    min_executions: int,
    max_suggestions: int,
    min_confidence: float,
    min_improvement: float,
    output: Optional[str],
) -> None:
    tracker = ExperimentTracker()
    optimizer = WorkflowOptimizer(
        experiment_tracker=tracker,
        config=OptimizationConfig(
            min_confidence=min_confidence,
            min_improvement=min_improvement,
        ),
    )

    click.echo(f"Generating optimization suggestions for: {workflow_id}")
    suggestions = await optimizer.suggest_optimizations(
        workflow_id=workflow_id,
        min_executions=min_executions,
        max_suggestions=max_suggestions,
    )

    if not suggestions:
        click.echo(f"No optimization suggestions found for: {workflow_id}")
        return

    click.echo(f"\nFound {len(suggestions)} optimization suggestions:\n")
    for i, suggestion in enumerate(suggestions, 1):
        click.echo(f"{i}. {suggestion.strategy_type.value.upper()}: {suggestion.target}")
        click.echo(f"   Description: {suggestion.description}")
        click.echo(f"   Expected improvement: {suggestion.expected_improvement:.1%}")
        click.echo(f"   Risk level: {suggestion.risk_level.value}")
        click.echo(f"   Confidence: {suggestion.confidence:.1%}")

        if suggestion.estimated_cost_reduction > 0:
            click.echo(f"   Estimated cost reduction: ${suggestion.estimated_cost_reduction:.4f}")
        if suggestion.estimated_duration_reduction > 0:
            click.echo(f"   Estimated time saved: {suggestion.estimated_duration_reduction:.2f}s")
        click.echo()

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([suggestion.to_dict() for suggestion in suggestions], f, indent=2)
        click.echo(f"Suggestions saved to: {output}")


async def _optimize_async(
    *,
    workflow_id: str,
    config_file: str,
    algorithm: str,
    max_iterations: int,
    validate: bool,
    output: Optional[str],
) -> None:
    with open(config_file, "r") as f:
        workflow_config = json.load(f)

    tracker = ExperimentTracker()
    optimizer = WorkflowOptimizer(
        experiment_tracker=tracker,
        config=OptimizationConfig(
            search_algorithm=algorithm,
            max_iterations=max_iterations,
            enable_validation=validate,
        ),
    )

    click.echo(f"Optimizing workflow: {workflow_id}")
    click.echo(f"Algorithm: {algorithm}")
    click.echo(f"Max iterations: {max_iterations}")
    click.echo(f"Validation: {'enabled' if validate else 'disabled'}")

    result = await optimizer.optimize_workflow(
        workflow_id=workflow_id,
        workflow_config=workflow_config,
    )

    if not result or not result.best_variant:
        click.echo(f"Optimization failed for workflow: {workflow_id}", err=True)
        return

    click.echo(f"\n{'='*60}")
    click.echo("Optimization Results")
    click.echo(f"{'='*60}")
    click.echo(f"Iterations: {result.iterations}")
    click.echo(f"Converged: {result.converged}")
    click.echo(f"Best score: {result.best_score:.3f}")

    variant = result.best_variant
    click.echo(f"\nBest variant: {variant.variant_id}")
    click.echo(f"Changes applied: {len(variant.changes)}")
    click.echo(f"Expected improvement: {variant.expected_improvement:.1%}")
    click.echo(f"Risk level: {variant.risk_level}")

    if variant.estimated_cost_reduction > 0:
        click.echo(f"Estimated cost reduction: ${variant.estimated_cost_reduction:.4f}")
    if variant.estimated_duration_reduction > 0:
        click.echo(f"Estimated time saved: {variant.estimated_duration_reduction:.2f}s")

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.best_variant.to_dict(), f, indent=2)
        click.echo(f"\nOptimized variant saved to: {output}")


async def _validate_async(
    variant_file: str,
    workflow_id: str,
    test_inputs: Optional[str],
) -> None:
    with open(variant_file, "r") as f:
        variant_data = json.load(f)

    from victor.optimization.generator import WorkflowVariant

    variant = WorkflowVariant(
        variant_id=variant_data["variant_id"],
        base_workflow_id=variant_data["base_workflow_id"],
        changes=[],
        expected_improvement=variant_data.get("expected_improvement", 0.0),
        risk_level=variant_data.get("risk_level", "medium"),
        config=variant_data.get("config", {}),
    )

    test_data = None
    if test_inputs:
        with open(test_inputs, "r") as f:
            test_data = json.load(f)

    tracker = ExperimentTracker()
    optimizer = WorkflowOptimizer(experiment_tracker=tracker)

    click.echo(f"Validating variant: {variant.variant_id}")
    workflow_profile = await optimizer.analyze_workflow(workflow_id=workflow_id)
    if not workflow_profile:
        click.echo(f"Cannot validate without profile for: {workflow_id}", err=True)
        return

    result = await optimizer.validate_variant(
        variant=variant,
        profile=workflow_profile,
        test_inputs=test_data,
    )

    click.echo(f"\n{'='*60}")
    click.echo("Validation Results")
    click.echo(f"{'='*60}")
    click.echo(f"Mode: {result.mode.value}")
    click.echo(f"Duration score: {result.duration_score:.3f}")
    click.echo(f"Cost score: {result.cost_score:.3f}")
    click.echo(f"Quality score: {result.quality_score:.3f}")
    click.echo(f"Overall score: {result.overall_score:.3f}")
    click.echo(f"Confidence: {result.confidence:.1%}")
    click.echo(f"Recommend: {'Yes' if result.recommendation else 'No'}")

    if result.metrics:
        click.echo("\nAdditional metrics:")
        for key, value in result.metrics.items():
            click.echo(f"  {key}: {value}")


async def _prompt_rollout_async(
    *,
    section_name: str,
    provider: str,
    treatment_hash: str,
    control_hash: Optional[str],
    traffic_split: float,
    min_samples_per_variant: int,
) -> None:
    click.echo("Starting prompt rollout experiment:")
    click.echo(f"  Section: {section_name}")
    click.echo(f"  Provider: {provider}")
    click.echo(f"  Treatment hash: {treatment_hash}")
    if control_hash:
        click.echo(f"  Control hash: {control_hash}")
    click.echo(f"  Traffic split: {traffic_split:.1%}")
    click.echo(f"  Min samples per variant: {min_samples_per_variant}")

    try:
        experiment_id = await create_prompt_rollout_experiment_async(
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
            control_hash=control_hash,
            traffic_split=traffic_split,
            min_samples_per_variant=min_samples_per_variant,
        )
    except ValueError as exc:
        click.echo(f"Cannot start prompt rollout: {exc}", err=True)
        return

    if not experiment_id:
        click.echo(
            f"Unable to start prompt rollout experiment for section: {section_name}",
            err=True,
        )
        return

    click.echo(f"Prompt rollout experiment started: {experiment_id}")


# Register commands with CLI
# Note: This would typically be done in the main CLI setup
def register_optimization_commands(cli_group):
    """Register optimization commands with CLI group.

    Args:
        cli_group: Click group to add commands to
    """
    cli_group.add_command(opt)
