# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""GEPA-Inspired Prompt Optimizer.

Evolves system prompt sections using execution trace analysis and
LLM-driven reflection, following the GEPA methodology (ICLR 2026):
  1. Collect execution traces (tool calls, failures, outcomes)
  2. Reflect on failure patterns using LLM
  3. Mutate prompt sections based on reflection
  4. Select best candidates via Thompson Sampling

Uses strategy pattern — GEPAStrategy is the default, but can be
swapped for alternatives (random mutation, manual, etc.).

Usage:
    from victor.framework.rl.learners.prompt_optimizer import (
        PromptOptimizerLearner,
    )

    learner = PromptOptimizerLearner("prompt_optimizer", db)
    candidate = learner.evolve("ASI_TOOL_EFFECTIVENESS_GUIDANCE", current_text)
    recommendation = learner.get_recommendation("ollama", "qwen3", "action",
                                                 section_name="ASI_TOOL_EFFECTIVENESS_GUIDANCE")
"""

import glob
import gzip
import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ExecutionTrace:
    """Summary of one agent session's execution for prompt evolution."""

    session_id: str
    task_type: str
    provider: str
    model: str
    tool_calls: int
    tool_failures: Dict[str, int]  # category → count
    success: bool
    completion_score: float
    tokens_used: int


@dataclass
class PromptCandidate:
    """An evolved prompt section candidate with Bayesian scoring.

    Candidates are scoped to (section_name, provider) so each provider
    can evolve independently. A cheap model may need more explicit guidance
    while a stronger model benefits from concise prompts.
    """

    section_name: str
    text: str
    text_hash: str
    generation: int
    parent_hash: str
    provider: str = "default"  # Provider scope (e.g., "xai", "anthropic", "ollama")
    scores: Dict[str, float] = field(default_factory=dict)
    alpha: float = 1.0
    beta_val: float = 1.0
    sample_count: int = 0

    def sample(self) -> float:
        """Thompson Sampling: draw from Beta distribution."""
        return random.betavariate(max(self.alpha, 0.01), max(self.beta_val, 0.01))

    def update(self, success: bool) -> None:
        """Update Beta posteriors."""
        if success:
            self.alpha += 1.0
        else:
            self.beta_val += 1.0
        self.sample_count += 1

    @property
    def mean(self) -> float:
        """Posterior mean."""
        return self.alpha / (self.alpha + self.beta_val)


@dataclass
class Objective:
    """An optimization objective with weight."""

    name: str
    weight: float
    direction: str = "maximize"


# ---------------------------------------------------------------------------
# Strategy Protocol + GEPA Default
# ---------------------------------------------------------------------------


class PromptOptimizationStrategy(Protocol):
    """Strategy interface for prompt evolution approaches."""

    def reflect(
        self,
        traces: List[ExecutionTrace],
        section_name: str,
        current_text: str,
    ) -> str:
        """Analyze traces and produce a reflection/diagnosis."""
        ...

    def mutate(
        self, current_text: str, reflection: str, section_name: str
    ) -> str:
        """Generate mutated prompt section text."""
        ...


class GEPAStrategy:
    """GEPA-inspired: reflect on execution traces, then mutate prompt text.

    Uses LLM for reflection + mutation when available. Falls back to
    heuristic reflection (failure frequency analysis) when LLM unavailable.
    """

    def __init__(self, llm_service: Any = None):
        self._llm = llm_service

    def reflect(
        self,
        traces: List[ExecutionTrace],
        section_name: str,
        current_text: str,
    ) -> str:
        """Analyze traces and produce natural language reflection."""
        # Aggregate failure patterns
        total = len(traces)
        successes = sum(1 for t in traces if t.success)
        all_failures: Dict[str, int] = {}
        total_tool_calls = 0
        total_tokens = 0

        for trace in traces:
            total_tool_calls += trace.tool_calls
            total_tokens += trace.tokens_used
            for category, count in trace.tool_failures.items():
                all_failures[category] = all_failures.get(category, 0) + count

        # Build heuristic reflection from failure frequencies
        lines = [
            f"Analysis of {total} execution traces:",
            f"- Success rate: {successes}/{total} ({100*successes/max(total,1):.0f}%)",
            f"- Avg tool calls: {total_tool_calls/max(total,1):.1f}",
            f"- Avg tokens: {total_tokens/max(total,1):.0f}",
        ]
        if all_failures:
            lines.append("- Top failure categories:")
            for cat, count in sorted(
                all_failures.items(), key=lambda x: -x[1]
            )[:5]:
                lines.append(f"  - {cat}: {count}")

        reflection = "\n".join(lines)

        # If LLM available, enhance with LLM-driven reflection
        if self._llm is not None:
            try:
                from victor.agent.services.protocols.decision_service import (
                    DecisionType,
                )

                llm_reflection = self._llm.decide_sync(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    {
                        "message_excerpt": (
                            f"Reflect on these execution failures for prompt section "
                            f"'{section_name}':\n{reflection}\n\nCurrent prompt:\n"
                            f"{current_text[:500]}\n\nWhat specific changes would "
                            f"reduce failures and improve success rate?"
                        ),
                    },
                )
                if llm_reflection.source != "timeout_fallback":
                    reflection += f"\n\nLLM Reflection:\n{llm_reflection.result}"
            except Exception:
                pass  # LLM reflection is best-effort

        return reflection

    def mutate(
        self, current_text: str, reflection: str, section_name: str
    ) -> str:
        """Generate mutated prompt text based on reflection.

        If LLM unavailable, applies heuristic mutations based on common
        failure patterns identified in the reflection.
        """
        # If LLM available, use it for intelligent mutation
        if self._llm is not None:
            try:
                from victor.agent.services.protocols.decision_service import (
                    DecisionType,
                )

                result = self._llm.decide_sync(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    {
                        "message_excerpt": (
                            f"Improve this prompt section based on the reflection.\n\n"
                            f"Current '{section_name}':\n{current_text}\n\n"
                            f"Reflection:\n{reflection}\n\n"
                            f"Generate improved version (same length ±20%, "
                            f"specific and actionable):"
                        ),
                    },
                )
                if (
                    result.source != "timeout_fallback"
                    and result.result
                    and len(str(result.result)) > 20
                ):
                    return str(result.result)
            except Exception:
                pass

        # Heuristic mutation: append failure-specific guidance
        mutations = []
        if "file_not_found" in reflection.lower():
            mutations.append(
                "- Verify file paths with ls() before reading them."
            )
        if "edit" in reflection.lower() and "mismatch" in reflection.lower():
            mutations.append(
                "- When editing, read the file first and copy old_str exactly."
            )
        if "timeout" in reflection.lower():
            mutations.append(
                "- Keep tool calls focused. Avoid redundant reads of the same file."
            )

        if mutations:
            return current_text + "\n" + "\n".join(mutations)
        return current_text


# ---------------------------------------------------------------------------
# Core Learner
# ---------------------------------------------------------------------------

# Minimum traces required before evolution
MIN_TRACES_FOR_EVOLUTION = 5
MIN_SAMPLES_FOR_CONFIDENCE = 3


class PromptOptimizerLearner(BaseLearner):
    """Evolves system prompt sections using GEPA-inspired trace analysis.

    Registered as 'prompt_optimizer' in the RL coordinator. Opt-in only:
    prompt evolution must be triggered explicitly via evolve(), and
    candidates are only used when confidence exceeds threshold.
    """

    EVOLVABLE_SECTIONS = [
        "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
        "GROUNDING_RULES",
    ]

    DEFAULT_OBJECTIVES = [
        Objective("completion_score", weight=0.5),
        Objective("tool_effectiveness", weight=0.3),
        Objective("token_efficiency", weight=0.2),
    ]

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Any = None,
        strategy: Optional[PromptOptimizationStrategy] = None,
    ):
        self._strategy: PromptOptimizationStrategy = strategy or GEPAStrategy()
        self._candidates: Dict[str, List[PromptCandidate]] = {}
        super().__init__(name, db_connection, learning_rate, provider_adapter)
        self._load_candidates()

    def _ensure_tables(self) -> None:
        """Create the prompt candidate table."""
        from victor.core.schema import Schema

        try:
            self.db.executescript(Schema.AGENT_PROMPT_CANDIDATE)
            logger.debug("Prompt optimizer tables ensured")
        except Exception as e:
            logger.warning("Failed to create prompt optimizer tables: %s", e)

    def _load_candidates(self) -> None:
        """Load candidates from DB into memory.

        Candidates are keyed by (section_name, provider) for provider-aware
        prompt evolution. The dict key is "section_name::provider".
        """
        from victor.core.schema import Tables

        try:
            cursor = self.db.execute(
                f"SELECT section_name, provider, text_hash, text, generation, parent_hash, "
                f"completion_score, token_efficiency, tool_effectiveness, "
                f"alpha, beta, sample_count "
                f"FROM {Tables.AGENT_PROMPT_CANDIDATE}"
            )
            for row in cursor.fetchall():
                candidate = PromptCandidate(
                    section_name=row[0],
                    provider=row[1] or "default",
                    text_hash=row[2],
                    text=row[3],
                    generation=row[4],
                    parent_hash=row[5] or "",
                    scores={
                        "completion_score": row[6],
                        "token_efficiency": row[7],
                        "tool_effectiveness": row[8],
                    },
                    alpha=row[9],
                    beta_val=row[10],
                    sample_count=row[11],
                )
                key = self._candidate_key(row[0], row[1] or "default")
                self._candidates.setdefault(key, []).append(candidate)
            total = sum(len(v) for v in self._candidates.values())
            if total:
                logger.info("Loaded %d prompt candidates from database", total)
        except Exception as e:
            logger.debug("Failed to load prompt candidates: %s", e)

    @staticmethod
    def _candidate_key(section_name: str, provider: str = "default") -> str:
        """Build the dict key for a (section, provider) pair."""
        return f"{section_name}::{provider}"

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Update posteriors for the active candidate."""
        section = outcome.metadata.get("prompt_section")
        if not section:
            return

        provider = outcome.provider or "default"
        # Try provider-specific first, then default
        for key in [
            self._candidate_key(section, provider),
            self._candidate_key(section, "default"),
        ]:
            candidates = self._candidates.get(key, [])
            active = [c for c in candidates if c.sample_count > 0]
            if active:
                candidate = active[-1]
                success = outcome.success and outcome.quality_score >= 0.5
                candidate.update(success)
                candidate.scores["completion_score"] = (
                    candidate.scores.get("completion_score", 0.0) * 0.9
                    + outcome.quality_score * 0.1
                )
                self._save_candidate(candidate)
                return

    def get_recommendation(
        self,
        provider: str,
        model: str,
        task_type: str,
        section_name: Optional[str] = None,
    ) -> Optional[RLRecommendation]:
        """Thompson Sampling over candidates for a section.

        Looks up provider-specific candidates first, then falls back to
        'default' provider candidates. This enables per-provider prompt
        evolution while sharing globally evolved prompts as baseline.
        """
        if not section_name:
            return None

        # Try provider-specific first, then default
        candidates = self._candidates.get(
            self._candidate_key(section_name, provider or "default"), []
        )
        if not candidates:
            candidates = self._candidates.get(
                self._candidate_key(section_name, "default"), []
            )
        if not candidates:
            return None

        # Thompson Sampling: sample from each candidate's Beta distribution
        best = max(candidates, key=lambda c: c.sample())
        confidence = min(
            best.sample_count / (MIN_SAMPLES_FOR_CONFIDENCE * 2), 1.0
        )

        return RLRecommendation(
            value=best.text,
            confidence=confidence,
            reason=f"GEPA gen-{best.generation} (α={best.alpha:.1f}, β={best.beta_val:.1f})",
            sample_size=best.sample_count,
            is_baseline=best.sample_count < MIN_SAMPLES_FOR_CONFIDENCE,
        )

    def evolve(
        self,
        section_name: str,
        current_text: str,
        provider: str = "default",
    ) -> Optional[PromptCandidate]:
        """Run one GEPA evolution cycle for a section.

        Args:
            section_name: Which prompt section to evolve
            current_text: Current text of the section
            provider: Provider scope (e.g., "xai", "ollama", "default")

        Steps:
        1. Collect execution traces from usage.jsonl + evaluation results
        2. Reflect on failure patterns (via strategy)
        3. Mutate prompt text (via strategy)
        4. Store new candidate

        Returns:
            New PromptCandidate, or None if insufficient data
        """
        traces = self._collect_traces(limit=50)
        if len(traces) < MIN_TRACES_FOR_EVOLUTION:
            logger.info(
                "Not enough traces for evolution (%d < %d)",
                len(traces),
                MIN_TRACES_FOR_EVOLUTION,
            )
            return None

        # Reflect
        reflection = self._strategy.reflect(traces, section_name, current_text)
        logger.info("GEPA reflection for '%s':\n%s", section_name, reflection)

        # Mutate
        new_text = self._strategy.mutate(current_text, reflection, section_name)
        if new_text == current_text:
            logger.info("Mutation produced no change for '%s'", section_name)
            return None

        # Create candidate
        text_hash = hashlib.md5(new_text.encode()).hexdigest()[:12]
        parent_hash = hashlib.md5(current_text.encode()).hexdigest()[:12]
        key = self._candidate_key(section_name, provider)
        generation = self._get_max_generation(key) + 1

        candidate = PromptCandidate(
            section_name=section_name,
            provider=provider,
            text=new_text,
            text_hash=text_hash,
            generation=generation,
            parent_hash=parent_hash,
        )

        self._candidates.setdefault(key, []).append(candidate)
        self._save_candidate(candidate)

        logger.info(
            "GEPA evolved '%s' to gen-%d (hash=%s)",
            section_name,
            generation,
            text_hash,
        )
        return candidate

    def _get_max_generation(self, section_name: str) -> int:
        """Get the highest generation number for a section."""
        candidates = self._candidates.get(section_name, [])
        if not candidates:
            return 0
        return max(c.generation for c in candidates)

    def _save_candidate(self, candidate: PromptCandidate) -> None:
        """Persist a candidate to the database."""
        from victor.core.schema import Tables

        try:
            self.db.execute(
                f"INSERT OR REPLACE INTO {Tables.AGENT_PROMPT_CANDIDATE} "
                f"(section_name, provider, text_hash, text, generation, parent_hash, "
                f"completion_score, token_efficiency, tool_effectiveness, "
                f"alpha, beta, sample_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    candidate.section_name,
                    candidate.provider,
                    candidate.text_hash,
                    candidate.text,
                    candidate.generation,
                    candidate.parent_hash,
                    candidate.scores.get("completion_score", 0.0),
                    candidate.scores.get("token_efficiency", 0.0),
                    candidate.scores.get("tool_effectiveness", 0.0),
                    candidate.alpha,
                    candidate.beta_val,
                    candidate.sample_count,
                ),
            )
            self.db.commit()
        except Exception as e:
            logger.warning("Failed to save prompt candidate: %s", e)

    def _collect_traces(self, limit: int = 50) -> List[ExecutionTrace]:
        """Collect execution traces from usage.jsonl files."""
        traces: List[ExecutionTrace] = []

        try:
            from victor.config.settings import get_project_paths

            logs_dir = get_project_paths().global_logs_dir
        except Exception:
            logs_dir = Path.home() / ".victor" / "logs"

        # Read from all usage.jsonl files (current + rotated .gz)
        jsonl_files = sorted(logs_dir.glob("usage.*.jsonl.gz")) + [
            logs_dir / "usage.jsonl"
        ]

        sessions: Dict[str, Dict[str, Any]] = {}
        for jsonl_path in jsonl_files:
            if not jsonl_path.exists():
                continue
            try:
                opener = (
                    gzip.open if jsonl_path.suffix == ".gz" else open
                )
                mode = "rt" if jsonl_path.suffix == ".gz" else "r"
                with opener(jsonl_path, mode) as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            sid = event.get("session_id", "")
                            etype = event.get("event_type", "")
                            data = event.get("data", {})

                            if sid not in sessions:
                                sessions[sid] = {
                                    "tool_calls": 0,
                                    "failures": {},
                                    "provider": "",
                                    "model": "",
                                    "task_type": "default",
                                    "tokens": 0,
                                }

                            if etype == "tool_call":
                                sessions[sid]["tool_calls"] += 1
                            elif etype == "tool_result":
                                if not data.get("success", True):
                                    error = str(
                                        data.get("error")
                                        or data.get("result", {}).get(
                                            "error", ""
                                        )
                                    )
                                    cat = self._categorize_failure(error)
                                    sessions[sid]["failures"][cat] = (
                                        sessions[sid]["failures"].get(cat, 0)
                                        + 1
                                    )
                            elif etype == "task_classification":
                                sessions[sid]["task_type"] = data.get(
                                    "task_type", "default"
                                )
                        except (json.JSONDecodeError, KeyError):
                            continue
            except Exception:
                continue

        # Convert to ExecutionTrace objects (most recent first)
        for sid, data in list(sessions.items())[-limit:]:
            if data["tool_calls"] > 0:
                has_failures = bool(data["failures"])
                traces.append(
                    ExecutionTrace(
                        session_id=sid,
                        task_type=data["task_type"],
                        provider=data.get("provider", "unknown"),
                        model=data.get("model", "unknown"),
                        tool_calls=data["tool_calls"],
                        tool_failures=data["failures"],
                        success=not has_failures,
                        completion_score=0.5 if has_failures else 0.8,
                        tokens_used=data.get("tokens", 0),
                    )
                )

        return traces

    @staticmethod
    def _categorize_failure(error: str) -> str:
        """Categorize a tool failure error message."""
        error_lower = error.lower()
        if "file not found" in error_lower or "directory not found" in error_lower:
            return "file_not_found"
        if "old_str not found" in error_lower:
            return "edit_mismatch"
        if "ambiguous" in error_lower or "found 2 times" in error_lower:
            return "edit_ambiguous"
        if "cannot read directory" in error_lower:
            return "read_directory"
        if "tool" in error_lower and "not found" in error_lower:
            return "tool_not_found"
        if "timeout" in error_lower:
            return "timeout"
        return "other"

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward from outcome."""
        return (
            0.4 * (1.0 if outcome.success else 0.0)
            + 0.4 * outcome.quality_score
            + 0.2 * outcome.metadata.get("tool_effectiveness", 0.5)
        )

    def export_metrics(self) -> Dict[str, Any]:
        """Export optimizer metrics."""
        return {
            "total_candidates": sum(
                len(v) for v in self._candidates.values()
            ),
            "sections": {
                name: len(candidates)
                for name, candidates in self._candidates.items()
            },
            "max_generation": max(
                (
                    max((c.generation for c in cands), default=0)
                    for cands in self._candidates.values()
                ),
                default=0,
            ),
        }
