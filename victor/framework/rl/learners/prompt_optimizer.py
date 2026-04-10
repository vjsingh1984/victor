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
class ToolCallTrace:
    """Individual tool call within a session (ASI detail)."""

    tool_name: str
    arguments_summary: str = ""
    reasoning_before: str = ""
    success: bool = True
    result_summary: str = ""
    error_detail: str = ""
    duration_ms: float = 0.0


@dataclass
class ExecutionTrace:
    """Summary of one agent session's execution for prompt evolution.

    When GEPA v2 trace enrichment is enabled, tool_call_details contains
    per-call ASI data. Otherwise, tool_calls is an int count (v1 compat).
    """

    session_id: str
    task_type: str
    provider: str
    model: str
    tool_calls: int
    tool_failures: Dict[str, int]  # category → count
    success: bool
    completion_score: float
    tokens_used: int
    # GEPA v2: detailed per-call traces (ASI)
    tool_call_details: List["ToolCallTrace"] = field(default_factory=list)


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

    def mutate(self, current_text: str, reflection: str, section_name: str) -> str:
        """Generate mutated prompt section text."""
        ...


class GEPAStrategy:
    """GEPA-inspired: reflect on execution traces, then mutate prompt text.

    Uses LLM for reflection + mutation when available. Falls back to
    heuristic reflection (failure frequency analysis) when LLM unavailable.

    LLM sources (in order of preference):
    1. Explicit llm_service (decision service)
    2. Ollama local model (free, fast — default: qwen3.5:2b)
    3. Heuristic fallback (no LLM needed)
    """

    def __init__(
        self,
        llm_service: Any = None,
        ollama_model: str = "qwen3.5:2b",
        ollama_url: str = "http://localhost:11434",
    ):
        self._llm = llm_service
        self._provider_name = "ollama"
        self._model = ollama_model
        self._provider = None  # Lazy-loaded

    def _get_provider(self) -> Any:
        """Get or create provider via Victor's provider abstraction.

        Uses the configured provider name to instantiate. Defaults to
        ollama (free, local). Set _provider_name to None to disable.
        """
        if self._provider is not None:
            return self._provider
        if not self._provider_name:
            return None
        try:
            if self._provider_name == "ollama":
                from victor.providers.ollama_provider import OllamaProvider

                self._provider = OllamaProvider()
            else:
                from importlib import import_module

                mod = import_module(f"victor.providers.{self._provider_name}_provider")
                cls_name = f"{self._provider_name.title()}Provider"
                self._provider = getattr(mod, cls_name)()
            return self._provider
        except Exception as e:
            logger.debug("Failed to create %s provider: %s", self._provider_name, e)
            return None

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Call LLM via Victor's provider abstraction (free local or cloud).

        Prepends /no_think for Qwen models to suppress verbose reasoning.
        """
        provider = self._get_provider()
        if provider is None:
            return None
        try:
            from victor.core.async_utils import run_sync_in_thread
            from victor.providers.base import Message

            # Suppress thinking for Qwen models
            effective_prompt = prompt
            if "qwen" in self._model.lower():
                effective_prompt = f"/no_think\n{prompt}"

            messages = [Message(role="user", content=effective_prompt)]
            response = run_sync_in_thread(
                provider.chat(
                    messages=messages,
                    model=self._model,
                    max_tokens=max_tokens,
                    temperature=0.7,
                ),
                timeout=30.0,
            )
            content = response.content if response else ""
            # Strip thinking artifacts (Qwen3, DeepSeek R1)
            import re

            if "<think>" in content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            # Strip "Thinking Process:" preamble
            if "Thinking Process:" in content:
                parts = re.split(r"\n(?=TOOL EFFECTIVENESS|[A-Z]{3,}[:\s])", content)
                for part in reversed(parts):
                    if "Thinking Process" not in part and len(part.strip()) > 50:
                        content = part.strip()
                        break
            if content and len(content) > 20:
                return content.strip()
        except Exception as e:
            logger.debug("LLM call via %s failed: %s", self._provider_name, e)
        return None

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
            for cat, count in sorted(all_failures.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {cat}: {count}")

        reflection = "\n".join(lines)

        # Enhance with LLM-driven reflection (provider → decision service → skip)
        llm_prompt = (
            f"You are analyzing execution traces for an AI coding agent.\n\n"
            f"{reflection}\n\n"
            f"Current prompt section '{section_name}':\n{current_text[:500]}\n\n"
            f"What 3 specific, actionable changes to this prompt guidance would "
            f"reduce the failure patterns above? Be concise — bullet points only."
        )

        # Try provider abstraction first (Ollama by default, free + local)
        llm_result = self._call_llm(llm_prompt)
        if llm_result:
            reflection += f"\n\nLLM Reflection ({self._provider_name}/{self._model}):\n{llm_result}"
            return reflection

        # Try decision service if available
        if self._llm is not None:
            try:
                from victor.agent.services.protocols.decision_service import (
                    DecisionType,
                )

                llm_reflection = self._llm.decide_sync(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    {
                        "message_excerpt": llm_prompt,
                    },
                )
                if llm_reflection.source != "timeout_fallback":
                    reflection += f"\n\nLLM Reflection:\n{llm_reflection.result}"
            except Exception:
                pass  # LLM reflection is best-effort

        return reflection

    def mutate(self, current_text: str, reflection: str, section_name: str) -> str:
        """Generate mutated prompt text based on reflection.

        Uses provider abstraction for LLM mutation, falls back to
        heuristic mutations based on failure patterns.
        """
        mutation_prompt = (
            f"Improve this prompt section for an AI coding agent based on "
            f"the execution analysis below.\n\n"
            f"Current '{section_name}':\n{current_text}\n\n"
            f"Reflection on failures:\n{reflection}\n\n"
            f"Generate an improved version. Requirements:\n"
            f"- Keep same length (±20%)\n"
            f"- Be specific and actionable\n"
            f"- Address the failure patterns from the reflection\n"
            f"- Output ONLY the improved prompt text, no explanation\n\n"
            f"Improved version:"
        )

        # Try provider abstraction (Ollama by default)
        llm_result = self._call_llm(mutation_prompt, max_tokens=800)
        if llm_result and len(llm_result) > 50:
            return llm_result

        # Heuristic mutation: append failure-specific guidance
        mutations = []
        if "file_not_found" in reflection.lower():
            mutations.append("- Verify file paths with ls() before reading them.")
        if "edit" in reflection.lower() and "mismatch" in reflection.lower():
            mutations.append("- When editing, read the file first and copy old_str exactly.")
        if "timeout" in reflection.lower():
            mutations.append("- Keep tool calls focused. Avoid redundant reads of the same file.")

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
        "COMPLETION_GUIDANCE",
        "FEW_SHOT_EXAMPLES",
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
        use_pareto: bool = False,
        max_prompt_chars: int = 1500,
    ):
        self._strategy: PromptOptimizationStrategy = strategy or GEPAStrategy()
        self._candidates: Dict[str, List[PromptCandidate]] = {}
        self._use_pareto = use_pareto
        self._max_prompt_chars = max_prompt_chars
        self._pareto_frontiers: Dict[str, Any] = {}  # section → ParetoFrontier
        super().__init__(name, db_connection, learning_rate, provider_adapter)
        self._load_candidates()
        if self._use_pareto:
            self._init_pareto_frontiers()

    def _ensure_tables(self) -> None:
        """Create the prompt candidate table and GEPA v2 extensions."""
        from victor.core.schema import Schema

        try:
            self.db.executescript(Schema.AGENT_PROMPT_CANDIDATE)
            self.db.executescript(Schema.AGENT_PROMPT_PARETO_INSTANCE)
            # Migrate: add v2 columns to existing table
            for col_def, default in [
                ("instance_scores TEXT", "'{}'"),
                ("coverage_count INTEGER", "0"),
                ("is_on_frontier INTEGER", "1"),
                ("char_length INTEGER", "0"),
            ]:
                try:
                    self.db.execute(
                        f"ALTER TABLE agent_prompt_candidate "
                        f"ADD COLUMN {col_def} DEFAULT {default}"
                    )
                except Exception:
                    pass  # Column already exists
            self.db.commit()
            logger.debug("Prompt optimizer tables ensured (v2)")
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
            candidates = self._candidates.get(self._candidate_key(section_name, "default"), [])
        if not candidates:
            return None

        # Thompson Sampling: sample from each candidate's Beta distribution
        best = max(candidates, key=lambda c: c.sample())
        confidence = min(best.sample_count / (MIN_SAMPLES_FOR_CONFIDENCE * 2), 1.0)

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
        # Use v2 trace collection when Pareto mode is enabled
        if self._use_pareto:
            traces = self._collect_traces_v2(limit=50)
        else:
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

        # Prompt bloat control: hard-truncate
        if self._max_prompt_chars and len(new_text) > self._max_prompt_chars:
            new_text = new_text[: self._max_prompt_chars]
            logger.info(
                "GEPA bloat control: truncated '%s' to %d chars",
                section_name,
                self._max_prompt_chars,
            )

        # Reject if over 2x the limit (likely garbage output)
        if self._max_prompt_chars and len(new_text) > 2 * self._max_prompt_chars:
            logger.warning(
                "GEPA rejected mutation for '%s': %d chars > 2x limit",
                section_name,
                len(new_text),
            )
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

        # GEPA v2: Add to Pareto frontier
        if self._use_pareto:
            if section_name not in self._pareto_frontiers:
                from victor.framework.rl.pareto import ParetoFrontier

                self._pareto_frontiers[section_name] = ParetoFrontier(max_candidates=20)
            self._pareto_frontiers[section_name].add_candidate(
                text_hash=text_hash,
                text=new_text,
                generation=generation,
            )

        logger.info(
            "GEPA evolved '%s' to gen-%d (hash=%s, %d chars%s)",
            section_name,
            generation,
            text_hash,
            len(new_text),
            ", pareto" if self._use_pareto else "",
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
        jsonl_files = sorted(logs_dir.glob("usage.*.jsonl.gz")) + [logs_dir / "usage.jsonl"]

        sessions: Dict[str, Dict[str, Any]] = {}
        for jsonl_path in jsonl_files:
            if not jsonl_path.exists():
                continue
            try:
                opener = gzip.open if jsonl_path.suffix == ".gz" else open
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
                                        data.get("error") or data.get("result", {}).get("error", "")
                                    )
                                    cat = self._categorize_failure(error)
                                    sessions[sid]["failures"][cat] = (
                                        sessions[sid]["failures"].get(cat, 0) + 1
                                    )
                            elif etype == "task_classification":
                                sessions[sid]["task_type"] = data.get("task_type", "default")
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

    # ------------------------------------------------------------------
    # GEPA v2: Pareto support
    # ------------------------------------------------------------------

    def _init_pareto_frontiers(self) -> None:
        """Initialize Pareto frontiers from existing candidates."""
        try:
            from victor.framework.rl.pareto import ParetoFrontier
        except ImportError:
            logger.warning("Pareto module not available, disabling Pareto mode")
            self._use_pareto = False
            return

        for key, candidates in self._candidates.items():
            section = key.split("::")[0] if "::" in key else key
            if section not in self._pareto_frontiers:
                self._pareto_frontiers[section] = ParetoFrontier(max_candidates=20)
            frontier = self._pareto_frontiers[section]
            for c in candidates:
                frontier.add_candidate(
                    text_hash=c.text_hash,
                    text=c.text,
                    generation=c.generation,
                    instance_scores=c.scores,
                )

    def get_pareto_frontier(self, section_name: str) -> Optional[Any]:
        """Get the Pareto frontier for a section (if Pareto mode enabled)."""
        return self._pareto_frontiers.get(section_name)

    def _collect_traces_v2(self, limit: int = 50) -> List[ExecutionTrace]:
        """Collect enriched execution traces (GEPA v2 with ASI detail).

        Reads the enriched JSONL events which include reasoning_before_call,
        result_summary, error_detail, and duration_ms per tool call.
        Falls back to v1 collection if enriched fields are absent.
        """
        traces: List[ExecutionTrace] = []

        try:
            from victor.config.settings import get_project_paths

            logs_dir = get_project_paths().global_logs_dir
        except Exception:
            logs_dir = Path.home() / ".victor" / "logs"

        jsonl_files = sorted(logs_dir.glob("usage.*.jsonl.gz")) + [logs_dir / "usage.jsonl"]

        sessions: Dict[str, Dict[str, Any]] = {}
        for jsonl_path in jsonl_files:
            if not jsonl_path.exists():
                continue
            try:
                opener = gzip.open if jsonl_path.suffix == ".gz" else open
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
                                    "details": [],  # v2: per-call details
                                }

                            if etype == "tool_call":
                                sessions[sid]["tool_calls"] += 1
                                # v2: capture detail
                                detail = ToolCallTrace(
                                    tool_name=data.get("tool_name", ""),
                                    arguments_summary=str(data.get("arguments_sanitized", ""))[
                                        :200
                                    ],
                                    reasoning_before=str(data.get("reasoning_before_call", ""))[
                                        :500
                                    ],
                                )
                                sessions[sid]["details"].append(detail)

                            elif etype == "tool_result":
                                success = data.get("success", True)
                                # Update the last detail entry
                                details = sessions[sid]["details"]
                                if details:
                                    last = details[-1]
                                    last.success = success
                                    last.duration_ms = data.get("duration_ms", 0)
                                    last.result_summary = str(data.get("result_summary", ""))[:500]
                                    last.error_detail = str(data.get("error_detail", ""))[:500]

                                if not success:
                                    error = str(
                                        data.get("error_detail")
                                        or data.get("error")
                                        or data.get("result", {}).get("error", "")
                                    )
                                    cat = self._categorize_failure(error)
                                    sessions[sid]["failures"][cat] = (
                                        sessions[sid]["failures"].get(cat, 0) + 1
                                    )

                            elif etype == "task_classification":
                                sessions[sid]["task_type"] = data.get("task_type", "default")
                        except (json.JSONDecodeError, KeyError):
                            continue
            except Exception:
                continue

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
                        tool_call_details=data.get("details", []),
                    )
                )

        return traces

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward from outcome."""
        return (
            0.4 * (1.0 if outcome.success else 0.0)
            + 0.4 * outcome.quality_score
            + 0.2 * outcome.metadata.get("tool_effectiveness", 0.5)
        )

    def export_metrics(self) -> Dict[str, Any]:
        """Export optimizer metrics."""
        pareto_info = {}
        for section, frontier in self._pareto_frontiers.items():
            pareto_info[section] = {
                "frontier_size": frontier.size,
                "candidates": [
                    {
                        "hash": e.text_hash,
                        "gen": e.generation,
                        "coverage": e.coverage_count,
                        "chars": e.char_length,
                    }
                    for e in frontier.get_frontier()
                ],
            }

        return {
            "total_candidates": sum(len(v) for v in self._candidates.values()),
            "sections": {name: len(candidates) for name, candidates in self._candidates.items()},
            "max_generation": max(
                (
                    max((c.generation for c in cands), default=0)
                    for cands in self._candidates.values()
                ),
                default=0,
            ),
            "use_pareto": self._use_pareto,
            "pareto": pareto_info,
        }
