# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Chain-of-Thought Distillation strategy.

Extracts step-by-step reasoning patterns from successful traces of
strong models (e.g., GPT-5.4 at 70%) and distills them into structured
guidance for weaker models (e.g., DeepSeek at 25%, Haiku at 25%).

Based on: Symbolic Chain-of-Thought Distillation (ACL 2023) — train
smaller models on rationales sampled from larger teacher models.

Our adaptation: instead of fine-tuning, we inject successful reasoning
patterns as structured prompts, teaching weaker models the step-by-step
approach that works for stronger models.
"""

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


class CoTDistillationStrategy:
    """Distill reasoning from strong model traces into prompt guidance.

    Implements the PromptOptimizationStrategy protocol.
    """

    def __init__(
        self,
        min_source_score: float = 0.7,
        max_steps: int = 5,
        min_score_gap: float = 0.15,
        llm_service: Any = None,
    ):
        self._min_score = min_source_score
        self._max_steps = max_steps
        self._min_gap = min_score_gap
        self._llm = llm_service

    def reflect(
        self,
        traces: List[Any],
        section_name: str,
        current_text: str,
    ) -> str:
        """Extract reasoning patterns from successful strong-model traces.

        Identifies traces where the source provider succeeded and builds
        a generalizable step-by-step reasoning template.
        """
        if not traces:
            return ""

        # Find high-scoring successful traces
        strong = [
            t
            for t in traces
            if t.success
            and t.completion_score >= self._min_score
            and t.tool_calls >= 3  # Need enough steps for a meaningful chain
        ]

        if not strong:
            logger.debug(
                "CoT: No strong traces found (need score >= %.1f, tools >= 3)",
                self._min_score,
            )
            return ""

        # Select the best trace for distillation
        best = max(strong, key=lambda t: t.completion_score)

        return self._distill_reasoning(best)

    def mutate(self, current_text: str, reflection: str, section_name: str) -> str:
        """Append distilled reasoning template to the prompt."""
        if not reflection:
            return current_text
        return f"{current_text}\n\n{reflection}"

    def _distill_reasoning(self, trace: Any) -> str:
        """Convert a successful trace into a step-by-step reasoning template.

        Extracts the tool-call sequence and generalizes it into a
        reusable reasoning pattern.
        """
        tools = getattr(trace, "tool_calls", 0)
        score = getattr(trace, "completion_score", 0)
        failures = getattr(trace, "tool_failures", {})

        # Build reasoning chain from trace profile
        steps = []

        # Step 1: Always start with discovery
        steps.append(
            "1. DISCOVER: Use code_search(query='relevant term') to find the "
            "file(s) related to the issue. Do NOT guess file paths."
        )

        # Step 2: Read and understand
        steps.append(
            "2. READ: Read the identified file(s) to understand the current "
            "implementation. Use search= parameter for large files."
        )

        # Step 3: Plan the fix
        steps.append(
            "3. PLAN: Before editing, identify exactly which lines need to "
            "change and what the fix should be. State your plan."
        )

        # Step 4: Execute with precision
        if failures and "edit_mismatch" in failures:
            steps.append(
                "4. EDIT: Copy old_str EXACTLY from the file (use 3+ lines of "
                "surrounding context). If the edit fails, re-read the file and "
                "adjust the old_str — do NOT guess."
            )
        else:
            steps.append(
                "4. EDIT: Apply the fix using edit() with exact old_str "
                "copied from the file. Include sufficient context for uniqueness."
            )

        # Step 5: Verify (if tools permit)
        if tools > 5:
            steps.append(
                "5. VERIFY: Read the modified file to confirm the edit was "
                "applied correctly. Signal TASK_DONE when verified."
            )

        # Limit to max_steps
        steps = steps[: self._max_steps]

        header = f"STEP-BY-STEP APPROACH (distilled from {score:.0%} success rate):"
        return f"{header}\n" + "\n".join(steps)
