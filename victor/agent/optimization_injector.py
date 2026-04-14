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

"""Optimization injector for prompt composition.

Consolidates all prompt optimization strategy outputs (GEPA, MIPROv2, CoT,
failure hints) into a single interface consumed by PromptComposer.

Key capability: Real-time failure hint injection. After a tool rollback,
the injector maps the error to one of 13 GEPA failure categories and
provides corrective guidance in the NEXT turn's user prefix.

Research basis:
- arXiv:2507.19457 — GEPA Pareto frontier + Thompson Sampling (ICLR 2026)
- arXiv:2406.11695 — MIPROv2 KNN few-shot demonstration mining
- arXiv:2601.08884 — Structured failure taxonomy with corrective hints
- arXiv:2604.07645 — PRIME semantic trace zones (SUCCESS/FAILURE/RECOVERY)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Failure category → corrective hint mapping (from GEPA arXiv:2601.08884).
# These are injected into user messages in real-time after tool failures,
# not just during offline evolution.
FAILURE_HINTS: Dict[str, str] = {
    "file_not_found": (
        "The file was not found. Use ls() to check the directory, "
        "or code_search(query='...') to find the correct path."
    ),
    "read_directory": (
        "You tried to read a directory. Use ls() for directories, read() for files."
    ),
    "permission_denied": (
        "Permission denied. Check file permissions or use a different path."
    ),
    "edit_mismatch": (
        "Your edit failed because old_str did not match the file content. "
        "RE-READ the file at the exact location and COPY the text "
        "character-by-character from the read output. Do NOT type from memory."
    ),
    "edit_ambiguous": (
        "Your edit matched multiple locations. Include 3+ surrounding lines "
        "of context in old_str to make the match unique."
    ),
    "edit_syntax": (
        "The edit produced invalid syntax. Check indentation and ensure "
        "new_str preserves correct Python/language syntax."
    ),
    "tool_not_found": (
        "Tool not found. Use only tools listed in the available set. "
        "Check spelling. Use code_search or read as fallbacks."
    ),
    "timeout": (
        "The operation timed out. Use more targeted searches and avoid "
        "reading entire large directories."
    ),
    "tool_error": (
        "Tool call error. Check the arguments match the expected schema "
        "and review the error message before retrying."
    ),
    "search_no_results": (
        "Search returned no results. Broaden the query, try alternative "
        "keywords or partial names, or use ls() to browse manually."
    ),
    "shell_error": (
        "Shell command failed. Check syntax and ensure required tools "
        "are available. Use absolute paths for reliability."
    ),
    "test_failure": (
        "Tests failed. Read the output carefully, identify which assertion "
        "failed and why, then fix the root cause."
    ),
    "other": (
        "An error occurred. Read the error message carefully and diagnose "
        "the root cause before retrying."
    ),
}

# Patterns for categorizing failures from error messages.
_FAILURE_PATTERNS: List[tuple] = [
    ("edit_mismatch", r"old_str not found|transaction rolled back|Failed to commit"),
    ("edit_ambiguous", r"ambiguous|multiple matches|appears \d+ times"),
    ("file_not_found", r"file not found|no such file|FileNotFoundError"),
    ("read_directory", r"is a directory|IsADirectoryError"),
    ("permission_denied", r"permission denied|PermissionError"),
    ("edit_syntax", r"syntax error|SyntaxError|IndentationError"),
    ("tool_not_found", r"tool.*not found|unknown tool"),
    ("timeout", r"timed? ?out|TimeoutError"),
    ("search_no_results", r"no results|0 results|no matches"),
    ("shell_error", r"command not found|exit code [1-9]"),
    ("test_failure", r"FAILED|AssertionError|test.*fail"),
    ("tool_error", r"missing required|invalid argument|parameter"),
]


def categorize_failure(error: str) -> str:
    """Categorize a failure error message into one of 13 GEPA categories.

    Args:
        error: The error message string from a failed tool call.

    Returns:
        Failure category string (e.g., "edit_mismatch", "file_not_found").
    """
    if not error:
        return "other"
    error_lower = error.lower()
    for category, pattern in _FAILURE_PATTERNS:
        if re.search(pattern, error_lower, re.IGNORECASE):
            return category
    return "other"


class OptimizationInjector:
    """Consolidates all prompt optimization strategy outputs.

    Provides a single interface for PromptComposer to collect:
    - GEPA-evolved sections (per-session, Thompson Sampling)
    - MIPROv2 few-shot examples (per-query, KNN)
    - CoT distillation hints (per-session)
    - Real-time failure hints (per-turn, after errors)

    Usage:
        injector = OptimizationInjector()
        evolved = injector.get_evolved_sections("deepseek", "deepseek-chat", "edit")
        hint = injector.get_failure_hint("edit_mismatch", "old_str not found")
    """

    def __init__(self) -> None:
        self._section_cache: Dict[str, Optional[str]] = {}
        self._last_failure_category: Optional[str] = None
        self._last_failure_error: Optional[str] = None

    def clear_session_cache(self) -> None:
        """Clear per-session cache (called on workspace switch)."""
        self._section_cache.clear()

    def get_evolved_sections(
        self,
        provider: str = "",
        model: str = "",
        task_type: str = "default",
    ) -> List[str]:
        """Get GEPA-evolved sections for user prefix injection.

        Returns evolved versions when confidence > threshold,
        static fallback for ASI_TOOL_EFFECTIVENESS otherwise.

        Args:
            provider: Provider name for per-provider evolution.
            model: Model name.
            task_type: Task type for context.

        Returns:
            List of evolved section texts to include in user prefix.
        """
        results: List[str] = []

        evolvable_sections = [
            "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
            "GROUNDING_RULES",
            "COMPLETION_GUIDANCE",
            "INIT_SYNTHESIS_RULES",
        ]

        for section_name in evolvable_sections:
            # Check session cache first
            if section_name in self._section_cache:
                cached = self._section_cache[section_name]
                if cached:
                    results.append(cached)
                continue

            evolved = self._sample_evolved(section_name, provider, model, task_type)
            self._section_cache[section_name] = evolved
            if evolved:
                results.append(evolved)

        # Always include static ASI_TOOL_EFFECTIVENESS if no evolved version
        if not any("TOOL EFFECTIVENESS" in r for r in results):
            from victor.agent.prompt_builder import ASI_TOOL_EFFECTIVENESS_GUIDANCE

            results.insert(0, ASI_TOOL_EFFECTIVENESS_GUIDANCE)

        if results:
            evolved_count = sum(1 for r in results if "TOOL EFFECTIVENESS" not in r)
            if evolved_count > 0:
                logger.info(
                    "[OptimizationInjector] Serving %d evolved + %d static sections "
                    "for %s/%s",
                    evolved_count,
                    len(results) - evolved_count,
                    provider or "default",
                    model or "default",
                )

        return results

    def get_few_shots(self, query: str) -> Optional[str]:
        """Get MIPROv2 KNN-selected few-shot examples.

        Unlike evolved sections (per-session), few-shots can be
        per-query to match the current task context.

        Args:
            query: Current user message for KNN similarity matching.

        Returns:
            Few-shot examples text or None.
        """
        # Check session cache (MIPROv2 is still session-level for now)
        if "FEW_SHOT_EXAMPLES" in self._section_cache:
            return self._section_cache["FEW_SHOT_EXAMPLES"]

        evolved = self._sample_evolved("FEW_SHOT_EXAMPLES", "", "", "default")
        self._section_cache["FEW_SHOT_EXAMPLES"] = evolved
        return evolved

    def get_failure_hint(
        self,
        category: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[str]:
        """Get a corrective hint for a specific failure category.

        Called by PromptComposer after a tool failure to inject
        actionable guidance into the NEXT turn's user prefix.

        Args:
            category: Pre-categorized failure type (e.g., "edit_mismatch").
            error: Raw error message (will be categorized if category is None).

        Returns:
            Corrective hint string or None.
        """
        if not category and error:
            category = categorize_failure(error)
        if not category:
            return None

        hint = FAILURE_HINTS.get(category)
        if not hint:
            return None

        return f"PREVIOUS ERROR: {hint}"

    def record_failure(self, tool_name: str, error: str) -> str:
        """Record a tool failure for hint injection on next turn.

        Called by the execution coordinator when a tool call fails.

        Args:
            tool_name: Name of the failed tool.
            error: Error message.

        Returns:
            The categorized failure type.
        """
        category = categorize_failure(error)
        self._last_failure_category = category
        self._last_failure_error = error
        logger.info(
            "[OptimizationInjector] Recorded failure: tool=%s category=%s",
            tool_name,
            category,
        )
        return category

    def _sample_evolved(
        self,
        section_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[str]:
        """Sample an evolved section from GEPA via Thompson Sampling.

        Gated by prompt_optimization.enabled setting. Returns None if
        no candidates exist or confidence is below threshold.
        """
        try:
            from victor.config.settings import get_settings

            po = getattr(get_settings(), "prompt_optimization", None)
            if po is None or not po.enabled:
                return None
            strategies = po.get_strategies_for_section(section_name)
            if not strategies:
                return None
        except Exception:
            return None

        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("prompt_optimizer")
            if learner is None:
                return None

            rec = learner.get_recommendation(
                provider or "",
                model or "",
                task_type,
                section_name=section_name,
            )
            if rec and rec.confidence > 0.6 and not rec.is_baseline:
                logger.info(
                    "[OptimizationInjector] Using evolved '%s' (gen=%s, conf=%.2f)",
                    section_name,
                    rec.reason,
                    rec.confidence,
                )
                return rec.value
        except Exception:
            pass

        return None
