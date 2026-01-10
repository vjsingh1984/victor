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

"""CLI commands for workflow optimization.

This module provides command-line interface commands for interacting
with the workflow optimization system.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path

import click

from victor.optimization import (
    WorkflowOptimizer,
    OptimizationConfig,
    WorkflowProfile,
    OptimizationOpportunity,
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
    async def run_profile():
        tracker = ExperimentTracker()
        optimizer = WorkflowOptimizer(experiment_tracker=tracker)

        click.echo(f"Profiling workflow: {workflow_id}")

        profile = await optimizer.analyze_workflow(
            workflow_id=workflow_id,
            min_executions=min_executions,
        )

        if not profile:
            click.echo(f"Unable to profile workflow: {workflow_id}", err=True)
            return

        # Display results
        click.echo(f"\n{'='*60}")
        click.echo(f"Workflow Profile: {workflow_id}")
        click.echo(f"{'='*60}")
        click.echo(f"Total duration: {profile.total_duration:.2f}s")
        click.echo(f"Total cost: ${profile.total_cost:.4f}")
        click.echo(f"Total tokens: {profile.total_tokens}")
        click.echo(f"Success rate: {profile.success_rate:.1%}")
        click.echo(f"Executions analyzed: {profile.num_executions}")

        # Display bottlenecks
        if profile.bottlenecks:
            click.echo(f"\nBottlenecks detected ({len(profile.bottlenecks)}):")
            for i, bottleneck in enumerate(profile.bottlenecks, 1):
                click.echo(f"  {i}. {bottleneck}")
                click.echo(f"     Severity: {bottleneck.severity.value}")
                click.echo(f"     Suggestion: {bottleneck.suggestion}")
        else:
            click.echo("\nNo bottlenecks detected!")

        # Display opportunities
        if profile.opportunities:
            click.echo(f"\nOptimization opportunities ({len(profile.opportunities)}):")
            for i, opp in enumerate(profile.opportunities[:5], 1):
                click.echo(f"  {i}. {opp.strategy_type.value}: {opp.target}")
                click.echo(f"     Expected improvement: {opp.expected_improvement:.1%}")
                click.echo(f"     Risk level: {opp.risk_level.value}")
                click.echo(f"     Confidence: {opp.confidence:.1%}")
        else:
            click.echo("\nNo optimization opportunities identified!")

        # Save to file if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)

            click.echo(f"\nProfile saved to: {output}")

    asyncio.run(run_profile())


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
    async def run_suggest():
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

        # Save to file if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump([s.to_dict() for s in suggestions], f, indent=2)

            click.echo(f"Suggestions saved to: {output}")

    asyncio.run(run_suggest())


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
    async def run_optimize():
        # Load workflow config
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

        # Display results
        click.echo(f"\n{'='*60}")
        click.echo(f"Optimization Results")
        click.echo(f"{'='*60}")
        click.echo(f"Iterations: {result.iterations}")
        click.echo(f"Converged: {result.converged}")
        click.echo(f"Best score: {result.best_score:.3f}")

        if result.best_variant:
            variant = result.best_variant
            click.echo(f"\nBest variant: {variant.variant_id}")
            click.echo(f"Changes applied: {len(variant.changes)}")
            click.echo(f"Expected improvement: {variant.expected_improvement:.1%}")
            click.echo(f"Risk level: {variant.risk_level}")

            if variant.estimated_cost_reduction > 0:
                click.echo(f"Estimated cost reduction: ${variant.estimated_cost_reduction:.4f}")

            if variant.estimated_duration_reduction > 0:
                click.echo(f"Estimated time saved: {variant.estimated_duration_reduction:.2f}s")

        # Save to file if requested
        if output and result.best_variant:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(result.best_variant.to_dict(), f, indent=2)

            click.echo(f"\nOptimized variant saved to: {output}")

    asyncio.run(run_optimize())


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
    async def run_validate():
        # Load variant
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

        # Load test inputs if provided
        test_data = None
        if test_inputs:
            with open(test_inputs, "r") as f:
                test_data = json.load(f)

        tracker = ExperimentTracker()
        optimizer = WorkflowOptimizer(experiment_tracker=tracker)

        click.echo(f"Validating variant: {variant.variant_id}")

        # Get profile for baseline
        profile = await optimizer.analyze_workflow(
            workflow_id=workflow_id,
        )

        if not profile:
            click.echo(f"Cannot validate without profile for: {workflow_id}", err=True)
            return

        # Validate variant
        result = await optimizer.validate_variant(
            variant=variant,
            profile=profile,
            test_inputs=test_data,
        )

        # Display results
        click.echo(f"\n{'='*60}")
        click.echo(f"Validation Results")
        click.echo(f"{'='*60}")
        click.echo(f"Mode: {result.mode.value}")
        click.echo(f"Duration score: {result.duration_score:.3f}")
        click.echo(f"Cost score: {result.cost_score:.3f}")
        click.echo(f"Quality score: {result.quality_score:.3f}")
        click.echo(f"Overall score: {result.overall_score:.3f}")
        click.echo(f"Confidence: {result.confidence:.1%}")
        click.echo(f"Recommend: {'Yes' if result.recommendation else 'No'}")

        if result.metrics:
            click.echo(f"\nAdditional metrics:")
            for key, value in result.metrics.items():
                click.echo(f"  {key}: {value}")

    asyncio.run(run_validate())


# Register commands with CLI
# Note: This would typically be done in the main CLI setup
def register_optimization_commands(cli_group):
    """Register optimization commands with CLI group.

    Args:
        cli_group: Click group to add commands to
    """
    cli_group.add_command(opt)
