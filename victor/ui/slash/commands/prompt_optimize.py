# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prompt optimization slash command (GEPA v2).

Commands:
- /prompt-optimize: Run evolution cycle on all evolvable sections
- /prompt-optimize ASI: Evolve a specific section
- /prompt-optimize --status: Show candidates and scores
- /prompt-optimize --pareto: Show Pareto frontier (v2)
- /prompt-optimize --tier [economic|balanced|performance]: Show/set tier

Example:
    /prompt-optimize                    # Evolve all sections
    /prompt-optimize ASI                # Evolve ASI_TOOL_EFFECTIVENESS_GUIDANCE
    /prompt-optimize --status           # Show current candidates
    /prompt-optimize --pareto           # Show Pareto frontier
    /prompt-optimize --tier performance # Force performance tier
"""

from __future__ import annotations

import logging

from rich.panel import Panel
from rich.table import Table

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class PromptOptimizeCommand(BaseSlashCommand):
    """Run GEPA-inspired prompt evolution cycle."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="prompt-optimize",
            description="Evolve system prompt sections using execution trace analysis (GEPA)",
            usage="/prompt-optimize [section|--status]",
            aliases=["optimize-prompt", "evolve-prompt"],
            category="advanced",
            requires_agent=False,
        )

    def execute(self, ctx: CommandContext) -> None:
        args = ctx.args if ctx.args else []
        show_status = "--status" in args or "status" in args

        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("prompt_optimizer")

            if learner is None:
                ctx.console.print(
                    "[yellow]Prompt optimizer learner not available. "
                    "Ensure RL coordinator is initialized.[/]"
                )
                return

            if show_status:
                self._show_status(ctx, learner, coordinator.db_path)
                return

            # Determine which sections to evolve
            from victor.framework.rl.learners.prompt_optimizer import (
                PromptOptimizerLearner,
            )

            sections = PromptOptimizerLearner.EVOLVABLE_SECTIONS
            if args and args[0] not in ("--status", "status"):
                # User specified a section name or abbreviation
                target = args[0].upper()
                matched = [s for s in sections if target in s]
                if matched:
                    sections = matched
                else:
                    ctx.console.print(
                        f"[red]Unknown section '{args[0]}'.[/] " f"Available: {', '.join(sections)}"
                    )
                    return

            # Get current prompt text for each section
            from victor.agent.prompt_builder import (
                ASI_TOOL_EFFECTIVENESS_GUIDANCE,
                COMPLETION_GUIDANCE,
                GROUNDING_RULES,
            )
            from victor.framework.init_synthesizer import SYNTHESIS_RULES

            section_text = {
                "ASI_TOOL_EFFECTIVENESS_GUIDANCE": ASI_TOOL_EFFECTIVENESS_GUIDANCE,
                "GROUNDING_RULES": GROUNDING_RULES,
                "COMPLETION_GUIDANCE": COMPLETION_GUIDANCE,
                "FEW_SHOT_EXAMPLES": "",  # No static text; MIPROv2 mines from traces
                "INIT_SYNTHESIS_RULES": SYNTHESIS_RULES,
            }

            # Run evolution
            results = Table(title="GEPA Prompt Evolution Results")
            results.add_column("Section", style="cyan")
            results.add_column("Provider", style="magenta")
            results.add_column("Ordinal", style="green")
            results.add_column("Status", style="bold")
            results.add_column("Change", style="dim")
            results.add_column("Lineage", style="dim")

            for section in sections:
                current = section_text.get(section)
                if current is None:
                    # Section not registered in section_text at all
                    results.add_row(section, "-", "-", "[yellow]Not available[/]", "-", "-")
                    continue

                candidate = learner.evolve(section, current)
                if candidate:
                    # Compute text diff size
                    added = len(candidate.text) - len(current)
                    change = f"+{added} chars" if added > 0 else f"{added} chars"
                    results.add_row(
                        section,
                        candidate.provider,
                        str(candidate.generation),
                        "[green]Evolved[/]",
                        change,
                        f"{candidate.parent_hash[:8]} -> {candidate.text_hash[:8]}",
                    )
                else:
                    results.add_row(section, "-", "-", "[dim]No change[/]", "-", "-")

            ctx.console.print(results)
            ctx.console.print(
                Panel(
                    f"Persisted in: [bold]{coordinator.db_path}[/]\n"
                    "Ordinal is candidate creation order per (section, provider). Shared parent "
                    "hashes indicate sibling candidates from the same baseline, not a strict "
                    "linear prompt chain.",
                    title="Prompt Evolution Notes",
                    border_style="blue",
                )
            )

        except Exception as e:
            ctx.console.print(f"[red]Prompt optimization failed:[/] {e}")
            logger.exception("Prompt optimization error")

    def _show_status(self, ctx: CommandContext, learner, db_path) -> None:
        """Display current prompt candidates and their scores."""
        metrics = learner.export_metrics()

        if metrics["total_candidates"] == 0:
            ctx.console.print(
                Panel(
                    "No evolved prompt candidates yet.\n\n"
                    "Run [bold]/prompt-optimize[/] to start evolution cycle.\n"
                    "Requires execution trace data in ~/.victor/logs/usage.jsonl\n"
                    f"Stored in: {db_path}",
                    title="Prompt Optimizer Status",
                    border_style="blue",
                )
            )
            return

        table = Table(title="Prompt Candidates")
        table.add_column("Section", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Ordinal", style="green")
        table.add_column("Parent", style="dim")
        table.add_column("Hash", style="dim")
        table.add_column("Live", style="bold")
        table.add_column("Bench", style="bold")
        table.add_column("α/β", style="dim")
        table.add_column("Mean", style="bold")
        table.add_column("Samples", style="dim")
        table.add_column("Strategy", style="dim")
        table.add_column("Preview", style="dim", max_width=50)

        for row in learner.export_candidate_rows():
            table.add_row(
                row["section"],
                row["provider"],
                str(row["ordinal"]),
                row["parent_hash"][:8],
                row["text_hash"][:8],
                "yes" if row["active"] else "no",
                "yes" if row["benchmark_passed"] else "no",
                f"{row['alpha']:.1f}/{row['beta']:.1f}",
                f"{row['mean']:.2f}",
                str(row["sample_count"]),
                row["strategy"],
                row["preview"],
            )

        ctx.console.print(table)

        # GEPA v2: Pareto info
        pareto_info = metrics.get("pareto", {})
        if pareto_info:
            pareto_table = Table(title="Pareto Frontier (GEPA v2)")
            pareto_table.add_column("Section", style="cyan")
            pareto_table.add_column("Hash", style="dim")
            pareto_table.add_column("Gen", style="green")
            pareto_table.add_column("Coverage", style="bold")
            pareto_table.add_column("Chars", style="dim")

            for section, info in pareto_info.items():
                for c in info.get("candidates", []):
                    pareto_table.add_row(
                        section,
                        c["hash"][:8],
                        str(c["gen"]),
                        str(c["coverage"]),
                        str(c["chars"]),
                    )
            ctx.console.print(pareto_table)

        ctx.console.print(
            f"\n[dim]Total: {metrics['total_candidates']} candidates "
            f"across {len(metrics['sections'])} section/provider keys"
            f"{' (Pareto v2)' if metrics.get('use_pareto') else ''}[/]"
        )
        ctx.console.print(f"[dim]Database: {db_path}[/]")
        ctx.console.print(
            Panel(
                "Ordinals are creation order, not guaranteed linear ancestry. "
                "Compare parent/hash pairs to see sibling candidates evolved from the same baseline.",
                title="Lineage Semantics",
                border_style="blue",
            )
        )
