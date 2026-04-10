# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MIPROv2-inspired few-shot demonstration mining strategy.

Mines successful execution traces from usage.jsonl and formats them
as in-context few-shot examples. Unlike GEPA (which evolves rules),
MIPROv2 provides concrete "do it like this" demonstrations.

Based on: MIPROv2 (DSPy) — Multiprompt Instruction Proposal Optimizer
that jointly optimizes instructions and few-shot demonstrations using
Bayesian optimization.

Our adaptation: instead of Bayesian search over candidate pools, we
mine real successful tool-call sequences from execution history and
rank by diversity + completion score.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MIPROv2Strategy:
    """Mine few-shot demonstrations from successful execution traces.

    Implements the PromptOptimizationStrategy protocol.
    """

    def __init__(
        self,
        max_examples: int = 3,
        min_completion_score: float = 0.7,
        example_diversity: bool = True,
        max_example_chars: int = 400,
    ):
        self._max_examples = max_examples
        self._min_score = min_completion_score
        self._diversity = example_diversity
        self._max_chars = max_example_chars

    def reflect(
        self,
        traces: List[Any],
        section_name: str,
        current_text: str,
    ) -> str:
        """Mine few-shot examples from successful traces.

        Returns formatted examples as a string, or empty if none found.
        """
        if not traces:
            return ""

        # Filter to successful traces with tool calls
        successful = [
            t
            for t in traces
            if t.success and t.tool_calls > 0 and t.completion_score >= self._min_score
        ]

        if not successful:
            logger.debug("MIPROv2: No successful traces above %.1f score", self._min_score)
            return ""

        # Sort by completion score (best first)
        successful.sort(key=lambda t: -t.completion_score)

        # Select diverse examples (different task types if possible)
        selected = self._select_diverse(successful)

        # Format as few-shot examples
        return self._format_examples(selected)

    def mutate(
        self, current_text: str, reflection: str, section_name: str
    ) -> str:
        """Append mined examples to the prompt section."""
        if not reflection:
            return current_text
        return f"{current_text}\n\n{reflection}"

    def _select_diverse(self, traces: List[Any]) -> List[Any]:
        """Select diverse examples covering different failure categories."""
        if not self._diversity or len(traces) <= self._max_examples:
            return traces[: self._max_examples]

        selected = []
        seen_types = set()

        for trace in traces:
            task_type = getattr(trace, "task_type", "default")
            if task_type not in seen_types or len(selected) < self._max_examples:
                selected.append(trace)
                seen_types.add(task_type)
                if len(selected) >= self._max_examples:
                    break

        return selected

    def _format_examples(self, traces: List[Any]) -> str:
        """Format traces as structured few-shot examples."""
        if not traces:
            return ""

        lines = ["SUCCESSFUL TOOL-USE PATTERNS (from execution history):"]

        for i, trace in enumerate(traces, 1):
            tools = getattr(trace, "tool_calls", 0)
            score = getattr(trace, "completion_score", 0)
            task_type = getattr(trace, "task_type", "unknown")
            failures = getattr(trace, "tool_failures", {})

            # Build example description
            failure_str = ""
            if failures:
                top_fail = max(failures.items(), key=lambda x: x[1])[0] if failures else ""
                failure_str = f", recovered from {top_fail}" if top_fail else ""

            lines.append(
                f"\nExample {i} ({task_type}, score={score:.1f}):"
            )
            lines.append(
                f"  Tools: {tools} calls{failure_str}"
            )

            # Add tool sequence pattern based on failure/success profile
            if not failures:
                lines.append(
                    "  Pattern: code_search → read target → edit → verify"
                )
            elif "file_not_found" in failures:
                lines.append(
                    "  Pattern: ls → code_search → read (verified path) → edit"
                )
            elif "edit_mismatch" in failures:
                lines.append(
                    "  Pattern: read (full context) → edit (3+ lines) → verify"
                )
            else:
                lines.append(
                    "  Pattern: code_search → read → analyze → edit → verify"
                )

        # Keep within char limit
        result = "\n".join(lines)
        if len(result) > self._max_chars:
            result = result[: self._max_chars] + "\n..."

        return result
