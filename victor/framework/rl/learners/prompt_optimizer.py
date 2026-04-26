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

import gzip
import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic Trace Zones (inspired by arXiv:2604.07645 — PRIME)
# ---------------------------------------------------------------------------


class TraceZone(str, Enum):
    """Semantic zones for GEPA trace organization."""

    SUCCESS = "successful_strategies"
    FAILURE = "failure_patterns"
    RECOVERY = "recovery_patterns"


def classify_trace_zone(trace) -> TraceZone:
    """Classify an execution trace into a semantic zone.

    - RECOVERY: successful despite having tool failures (retry worked)
    - FAILURE: score < 0.5 or not successful
    - SUCCESS: everything else (score >= 0.5, no failures)
    """
    has_failures = bool(getattr(trace, "tool_failures", None))
    is_success = getattr(trace, "success", False)
    score = getattr(trace, "completion_score", 0.0)

    if is_success and has_failures:
        return TraceZone.RECOVERY
    if not is_success or score < 0.5:
        return TraceZone.FAILURE
    return TraceZone.SUCCESS


# ---------------------------------------------------------------------------
# Trace Quality Scoring (inspired by arXiv:2604.07877 — MemReader)
# ---------------------------------------------------------------------------

TRACE_QUALITY_THRESHOLD = 0.3


def score_trace_quality(trace) -> float:
    """Score trace quality for GEPA reflection value (MemReader-inspired).

    Returns 0.0-1.0. Traces below TRACE_QUALITY_THRESHOLD are noise.
    Criteria: substance, completeness, richness, coherence.
    """
    score = 0.0

    tool_calls = getattr(trace, "tool_calls", 0)
    if isinstance(tool_calls, int):
        if tool_calls >= 5:
            score += 0.3
        elif tool_calls >= 3:
            score += 0.2
        elif tool_calls >= 1:
            score += 0.1

    details = getattr(trace, "tool_call_details", [])
    if details:
        populated = sum(
            1 for d in details if getattr(d, "result_summary", "") or getattr(d, "error_detail", "")
        )
        completeness = populated / max(len(details), 1)
        score += 0.3 * completeness

    reasoning_count = sum(1 for d in details if getattr(d, "reasoning_before", ""))
    if details and reasoning_count / max(len(details), 1) > 0.5:
        score += 0.2

    failures = getattr(trace, "tool_failures", {})
    if failures:
        categorized = sum(v for k, v in failures.items() if k != "other")
        total = sum(failures.values())
        if total > 0 and categorized / total > 0.5:
            score += 0.2
        else:
            score += 0.1
    elif getattr(trace, "success", False):
        score += 0.15

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Capability Gap Analysis (inspired by arXiv:2604.05336 — TRACE)
# ---------------------------------------------------------------------------


@dataclass
class CapabilityGap:
    """A specific capability deficiency identified from trace contrast."""

    capability: str
    failure_rate: float
    failure_count: int
    example_errors: List[str]


FAILURE_TO_CAPABILITY: Dict[str, str] = {
    "edit_mismatch": "edit_precision",
    "edit_ambiguous": "edit_precision",
    "edit_syntax": "edit_precision",
    "file_not_found": "path_resolution",
    "read_directory": "path_resolution",
    "permission_denied": "path_resolution",
    "search_no_results": "search_strategy",
    "tool_not_found": "tool_knowledge",
    "tool_error": "tool_knowledge",
    "timeout": "execution_efficiency",
    "shell_error": "execution_efficiency",
    "test_failure": "verification",
    "other": "other",
}


def analyze_capability_gaps(traces) -> List[CapabilityGap]:
    """Contrast success vs failure zones to find dominant gaps (TRACE-inspired)."""
    capability_failures: Dict[str, int] = {}
    capability_errors: Dict[str, List[str]] = {}
    total_failures = 0

    for trace in traces:
        zone = classify_trace_zone(trace)
        if zone != TraceZone.FAILURE:
            continue
        for cat, count in getattr(trace, "tool_failures", {}).items():
            capability = FAILURE_TO_CAPABILITY.get(cat, "other")
            capability_failures[capability] = capability_failures.get(capability, 0) + count
            total_failures += count
            for detail in getattr(trace, "tool_call_details", []):
                if not getattr(detail, "success", True) and getattr(detail, "error_detail", ""):
                    errors = capability_errors.setdefault(capability, [])
                    if len(errors) < 3:
                        errors.append(getattr(detail, "error_detail", "")[:200])

    if not total_failures:
        return []

    gaps = []
    for cap, count in sorted(capability_failures.items(), key=lambda x: -x[1]):
        gaps.append(
            CapabilityGap(
                capability=cap,
                failure_rate=count / total_failures,
                failure_count=count,
                example_errors=capability_errors.get(cap, []),
            )
        )
    return gaps[:3]


# ---------------------------------------------------------------------------
# Structured Failure Taxonomy (inspired by arXiv:2601.08884)
# ---------------------------------------------------------------------------
# Each failure category maps to a corrective "Prompt Hint" that feeds into
# GEPA's reflection step, giving the mutation LLM actionable guidance
# instead of raw category names. Add new categories by adding entries.

FAILURE_HINTS: Dict[str, str] = {
    "file_not_found": (
        "Verify file paths with ls() before reading. "
        "Use code_search to find files by name or content."
    ),
    "read_directory": (
        "Use ls() for directories, read() only for files. "
        "Check the path is a file, not a directory."
    ),
    "permission_denied": (
        "Check file permissions. Avoid writing to read-only paths. "
        "Use a working directory the agent has write access to."
    ),
    "edit_mismatch": (
        "Read the complete file before editing. Copy old_str exactly from "
        "tool output — character for character, including whitespace and indentation."
    ),
    "edit_ambiguous": (
        "Include 3+ surrounding context lines in old_str to make the match unique. "
        "If the string appears multiple times, add distinguishing context."
    ),
    "edit_syntax": (
        "Validate that new_str preserves correct syntax. Check indentation matches "
        "the surrounding code. Run a linter after editing if available."
    ),
    "tool_not_found": (
        "Use only tools listed in the available tools. Check tool name spelling. "
        "Use ls or code_search as universal fallbacks."
    ),
    "timeout": (
        "Keep tool calls focused. Avoid reading entire large directories. "
        "Use targeted searches instead of broad scans. Limit shell command duration."
    ),
    "tool_error": (
        "Check tool arguments match the expected schema. Review the error message "
        "and adjust arguments before retrying."
    ),
    "search_no_results": (
        "Broaden the search query. Try alternative keywords, partial names, "
        "or regex patterns. Fall back to ls() + grep for manual search."
    ),
    "shell_error": (
        "Check command syntax. Ensure required tools (git, npm, etc.) are "
        "installed. Use absolute paths for reliability."
    ),
    "test_failure": (
        "Read the test output carefully. Identify which assertion failed and why. "
        "Fix the root cause, not the symptom."
    ),
    "other": (
        "Read the error message carefully. Diagnose the root cause before "
        "retrying. Avoid repeating the same failing operation."
    ),
}


def get_failure_hint(category: str) -> str:
    """Get the corrective prompt hint for a failure category."""
    return FAILURE_HINTS.get(category, "")


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
    # Per-section token counts for efficiency tracking
    section_tokens: Dict[str, int] = field(default_factory=dict)
    # GEPA v2: detailed per-call traces (ASI)
    tool_call_details: List["ToolCallTrace"] = field(default_factory=list)
    # Credit assignment signals (FEP-0001 Phase 3)
    # Per-tool credit values from CreditTrackingService
    credit_signals: List[Dict[str, Any]] = field(default_factory=list)
    # Optional agent-level summary for multi-agent team runs
    agent_guidance: Optional[str] = None


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
    benchmark_score: float = 0.0
    benchmark_runs: int = 0
    benchmark_passed: bool = False
    is_active: bool = False
    strategy_name: str = "gepa"
    requires_benchmark: bool = False

    def sample(self) -> float:
        """Thompson Sampling: draw from Beta distribution with staleness decay.

        Candidates with many samples have their posteriors slightly decayed
        toward uncertainty (0.5), giving newer candidates a fair chance.
        Decay factor: 0.95^(samples/20) — halves certainty after ~280 samples.
        """
        decay = 0.95 ** (self.sample_count / 20.0)
        # Decay posteriors toward prior (1,1) — increases uncertainty
        eff_alpha = 1.0 + (self.alpha - 1.0) * decay
        eff_beta = 1.0 + (self.beta_val - 1.0) * decay
        return random.betavariate(max(eff_alpha, 0.01), max(eff_beta, 0.01))

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
        **kwargs: Any,
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
        **kwargs: Any,
    ) -> str:
        """Analyze traces and produce natural language reflection."""
        del kwargs
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

        # Enrich with credit assignment data (FEP-0001 Phase 3)
        credit_traces = [t for t in traces if t.credit_signals]
        if credit_traces:
            tool_credits: Dict[str, List[float]] = {}
            for trace in credit_traces:
                for cs in trace.credit_signals:
                    tname = cs.get("tool_name", "unknown")
                    tool_credits.setdefault(tname, []).append(cs.get("credit", 0.0))
            if tool_credits:
                lines.append("- Tool credit attribution:")
                for tool, credits in sorted(
                    tool_credits.items(), key=lambda x: sum(x[1]), reverse=True
                )[:5]:
                    avg = sum(credits) / len(credits)
                    lines.append(f"  - {tool}: avg_credit={avg:+.2f} ({len(credits)} calls)")

        agent_guidance_blocks = []
        for trace in traces:
            guidance = getattr(trace, "agent_guidance", None)
            if guidance and guidance not in agent_guidance_blocks:
                agent_guidance_blocks.append(guidance)
        if agent_guidance_blocks:
            lines.append("- Agent execution credit:")
            lines.extend(agent_guidance_blocks[:2])

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
        "INIT_SYNTHESIS_RULES",  # Only the RULES section, frame stays fixed
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
        self._extra_strategies: Dict[str, List["PromptOptimizationStrategy"]] = {}
        # Section-specific strategy overrides (e.g., FEW_SHOT_EXAMPLES → MIPROv2)
        self._candidates: Dict[str, List[PromptCandidate]] = {}
        self._use_pareto = use_pareto
        self._max_prompt_chars = max_prompt_chars
        self._pareto_frontiers: Dict[str, Any] = {}  # section → ParetoFrontier
        super().__init__(name, db_connection, learning_rate, provider_adapter)
        self._load_candidates()
        if self._use_pareto:
            self._init_pareto_frontiers()
        self._init_section_strategies()

    @staticmethod
    def _load_prompt_optimization_settings() -> Any:
        """Load prompt-optimization settings, tolerating bootstrap-time failures."""
        try:
            from victor.config.settings import get_settings

            settings = get_settings()
            return getattr(settings, "prompt_optimization", None)
        except Exception:
            return None

    def _init_section_strategies(self) -> None:
        """Initialize section-specific strategies from config."""
        try:
            from victor.config.prompt_optimization_settings import (
                PromptOptimizationSettings,
            )
            from victor.framework.rl.learners.strategy_registry import (
                build_prompt_strategy,
            )

            po_settings = self._load_prompt_optimization_settings()
            if po_settings is None:
                po_settings = PromptOptimizationSettings(enabled=True)

            self._extra_strategies = {}
            for section_name in self.EVOLVABLE_SECTIONS:
                strategy_names = po_settings.get_strategies_for_section(section_name)
                strategies = []
                for strategy_name in strategy_names:
                    strategy = build_prompt_strategy(
                        strategy_name,
                        settings=po_settings,
                        gepa_strategy=self._strategy,
                    )
                    if strategy is not None:
                        strategies.append(strategy)
                self._extra_strategies[section_name] = strategies

            logger.debug(
                "Section strategies initialized: %s",
                {k: [type(s).__name__ for s in v] for k, v in self._extra_strategies.items()},
            )
        except ImportError:
            logger.debug("Strategy imports failed, using default GEPA only")

    def _strategies_for_section(self, section_name: str) -> List["PromptOptimizationStrategy"]:
        """Return the configured strategy chain for a section."""
        if section_name in self._extra_strategies:
            return list(self._extra_strategies[section_name])
        return [self._strategy]

    def _collect_learning_traces(self, limit: int = 50) -> List[ExecutionTrace]:
        """Collect and merge traces for prompt optimization."""
        if self._use_pareto:
            jsonl_traces = self._collect_traces_v2(limit=limit)
        else:
            jsonl_traces = self._collect_traces(limit=limit)

        conv_traces = self._collect_traces_from_conversations(limit=limit)
        traces = self._merge_traces(jsonl_traces, conv_traces)

        if conv_traces:
            logger.info(
                "Unified traces: %d from JSONL + %d from conversations = %d unique",
                len(jsonl_traces),
                len(conv_traces),
                len(traces),
            )
        return traces

    def _apply_section_strategies(
        self,
        section_name: str,
        current_text: str,
        traces: List[ExecutionTrace],
        *,
        query: Optional[str] = None,
    ) -> str:
        """Apply the configured strategy chain for a section."""
        new_text = current_text
        for strat in self._strategies_for_section(section_name):
            strat_name = type(strat).__name__
            reflection = strat.reflect(traces, section_name, new_text, query=query)
            if reflection:
                logger.info(
                    "%s reflection for '%s':\n%s", strat_name, section_name, reflection[:200]
                )
                new_text = strat.mutate(new_text, reflection, section_name)
        return new_text

    def get_query_aware_few_shots(self, query: str) -> Optional[str]:
        """Render MIPROv2 few-shot examples tailored to the current query."""
        if not query or not query.strip():
            return None
        if not self._strategies_for_section("FEW_SHOT_EXAMPLES"):
            return None

        traces = self._collect_learning_traces(limit=50)
        if not traces:
            return None

        few_shots = self._apply_section_strategies(
            "FEW_SHOT_EXAMPLES",
            "",
            traces,
            query=query,
        ).strip()
        return few_shots or None

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
                ("benchmark_score REAL", "0.0"),
                ("benchmark_runs INTEGER", "0"),
                ("benchmark_passed INTEGER", "0"),
                ("strategy_name TEXT", "'gepa'"),
                ("requires_benchmark INTEGER", "0"),
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
                f"alpha, beta, sample_count, benchmark_score, benchmark_runs, "
                f"benchmark_passed, is_active, strategy_name, requires_benchmark "
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
                    benchmark_score=row[12] or 0.0,
                    benchmark_runs=row[13] or 0,
                    benchmark_passed=bool(row[14]),
                    is_active=bool(row[15]),
                    strategy_name=row[16] or "gepa",
                    requires_benchmark=bool(row[17]),
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

    @staticmethod
    def _strategy_name_for_candidate(strategies: List["PromptOptimizationStrategy"]) -> str:
        """Derive a stable strategy name for persisted candidate metadata."""
        if not strategies:
            return "gepa"
        primary = type(strategies[0]).__name__
        if primary.endswith("Strategy"):
            primary = primary[: -len("Strategy")]
        return primary.lower()

    @staticmethod
    def _requires_benchmark_for_candidate(strategies: List["PromptOptimizationStrategy"]) -> bool:
        """Whether a strategy chain should remain shadow-only until benchmark approval."""
        return any(bool(getattr(strategy, "requires_benchmark_gate", False)) for strategy in strategies)

    @staticmethod
    def _servable_candidates(candidates: List[PromptCandidate]) -> List[PromptCandidate]:
        """Filter out pending candidates that are explicitly benchmark-gated."""
        return [
            candidate
            for candidate in candidates
            if not candidate.requires_benchmark or candidate.benchmark_passed
        ]

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
            active = [c for c in candidates if c.is_active] or [
                c for c in candidates if c.sample_count > 0
            ]
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

        provider_candidates = self._servable_candidates(
            self._candidates.get(self._candidate_key(section_name, provider or "default"), [])
        )
        default_candidates = self._servable_candidates(
            self._candidates.get(self._candidate_key(section_name, "default"), [])
        )

        candidates = provider_candidates or default_candidates
        if not candidates:
            return None

        active_candidates = [c for c in candidates if c.is_active]
        approved_candidates = [c for c in candidates if c.benchmark_passed]

        if active_candidates:
            candidates = [c for c in active_candidates if c.benchmark_passed] or active_candidates
        elif approved_candidates:
            candidates = approved_candidates

        # Hybrid: if Pareto enabled, restrict Thompson to frontier candidates
        if self._use_pareto:
            key = self._candidate_key(section_name, provider or "default")
            frontier = self._pareto_frontiers.get(key)
            if frontier:
                frontier_hashes = {e.text_hash for e in frontier.get_frontier()}
                if frontier_hashes:
                    frontier_candidates = [c for c in candidates if c.text_hash in frontier_hashes]
                    if frontier_candidates:
                        candidates = frontier_candidates
                        logger.debug(
                            "Pareto frontier filtered %d → %d candidates for %s",
                            len(candidates) + len(frontier_candidates),
                            len(frontier_candidates),
                            section_name,
                        )

        # Thompson Sampling: sample from (frontier) candidates' Beta distributions
        best = max(candidates, key=lambda c: c.sample())
        evidence_count = max(best.sample_count, best.benchmark_runs)
        confidence = min(evidence_count / (MIN_SAMPLES_FOR_CONFIDENCE * 2), 1.0)

        reason_parts = [f"GEPA gen-{best.generation} (α={best.alpha:.1f}, β={best.beta_val:.1f})"]
        if best.is_active:
            reason_parts.append("active")
        if best.benchmark_passed:
            reason_parts.append(
                f"bench={best.benchmark_score:.2f}/{best.benchmark_runs}"
                if best.benchmark_runs
                else "bench-approved"
            )

        return RLRecommendation(
            value=best.text,
            confidence=confidence,
            reason=", ".join(reason_parts),
            sample_size=evidence_count,
            is_baseline=best.sample_count < MIN_SAMPLES_FOR_CONFIDENCE,
        )

    def evolve(
        self,
        section_name: str,
        current_text: str,
        provider: str = "default",
        query: Optional[str] = None,
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
        traces = self._collect_learning_traces(limit=50)

        # Enrich traces with credit signals (FEP-0001 Phase 3)
        self._enrich_traces_with_credit(traces)

        if len(traces) < MIN_TRACES_FOR_EVOLUTION:
            logger.info(
                "Not enough traces for evolution (%d < %d)",
                len(traces),
                MIN_TRACES_FOR_EVOLUTION,
            )
            return None

        # Apply strategies sequentially (layered composition)
        new_text = self._apply_section_strategies(
            section_name,
            current_text,
            traces,
            query=query,
        )
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
        strategies = self._strategies_for_section(section_name)

        candidate = PromptCandidate(
            section_name=section_name,
            provider=provider,
            text=new_text,
            text_hash=text_hash,
            generation=generation,
            parent_hash=parent_hash,
            strategy_name=self._strategy_name_for_candidate(strategies),
            requires_benchmark=self._requires_benchmark_for_candidate(strategies),
        )

        self._candidates.setdefault(key, []).append(candidate)
        self._save_candidate(candidate)

        # Prune: keep only top N candidates per section (by mean score)
        MAX_CANDIDATES_PER_SECTION = 10
        section_candidates = self._candidates.get(key, [])
        if len(section_candidates) > MAX_CANDIDATES_PER_SECTION:
            # Keep the highest-mean candidates
            section_candidates.sort(key=lambda c: -c.mean)
            pruned = section_candidates[MAX_CANDIDATES_PER_SECTION:]
            self._candidates[key] = section_candidates[:MAX_CANDIDATES_PER_SECTION]
            # Remove pruned from DB
            from victor.core.schema import Tables

            for p_candidate in pruned:
                try:
                    self.db.execute(
                        f"DELETE FROM {Tables.AGENT_PROMPT_CANDIDATE} WHERE text_hash = ?",
                        (p_candidate.text_hash,),
                    )
                except Exception:
                    pass
            self.db.commit()
            logger.info(
                "Pruned %d candidates from %s (kept top %d)",
                len(pruned),
                key,
                MAX_CANDIDATES_PER_SECTION,
            )

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
                f"alpha, beta, sample_count, benchmark_score, benchmark_runs, "
                f"benchmark_passed, is_active, strategy_name, requires_benchmark) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    candidate.benchmark_score,
                    candidate.benchmark_runs,
                    int(candidate.benchmark_passed),
                    int(candidate.is_active),
                    candidate.strategy_name,
                    int(candidate.requires_benchmark),
                ),
            )
            self.db.commit()
        except Exception as e:
            logger.warning("Failed to save prompt candidate: %s", e)

    def _find_candidate(
        self,
        section_name: str,
        provider: str,
        text_hash: str,
    ) -> Optional[PromptCandidate]:
        """Find a specific candidate by section/provider/hash."""
        candidates = self._candidates.get(self._candidate_key(section_name, provider), [])
        for candidate in candidates:
            if candidate.text_hash == text_hash:
                return candidate
        return None

    def record_benchmark_result(
        self,
        section_name: str,
        provider: str,
        text_hash: str,
        score: float,
        passed: bool,
    ) -> Optional[PromptCandidate]:
        """Record a benchmark result for a candidate.

        Uses a running average for benchmark score and remembers whether
        the candidate has ever passed its gating benchmark.
        """
        candidate = self._find_candidate(section_name, provider, text_hash)
        if candidate is None:
            return None

        previous_runs = candidate.benchmark_runs
        cumulative_score = candidate.benchmark_score * previous_runs
        candidate.benchmark_runs += 1
        candidate.benchmark_score = (cumulative_score + score) / candidate.benchmark_runs
        candidate.benchmark_passed = candidate.benchmark_passed or bool(passed)
        self._save_candidate(candidate)
        return candidate

    def promote_candidate(
        self,
        section_name: str,
        provider: str,
        text_hash: str,
    ) -> Optional[PromptCandidate]:
        """Promote a candidate to active status for its section/provider."""
        key = self._candidate_key(section_name, provider)
        candidates = self._candidates.get(key, [])
        if not candidates:
            return None

        target: Optional[PromptCandidate] = None
        for candidate in candidates:
            if candidate.text_hash == text_hash:
                target = candidate
                break

        if target is None:
            return None
        if target.requires_benchmark and not target.benchmark_passed:
            raise ValueError("cannot promote candidate before benchmark gating passes")
        if target.benchmark_runs > 0 and not target.benchmark_passed:
            raise ValueError("cannot promote candidate that has failed benchmark gating")

        for candidate in candidates:
            candidate.is_active = candidate.text_hash == text_hash
            self._save_candidate(candidate)
        return target

    def build_rollout_experiment_config(
        self,
        section_name: str,
        provider: str,
        treatment_hash: str,
        *,
        control_hash: Optional[str] = None,
        traffic_split: float = 0.1,
        min_samples_per_variant: int = 100,
    ) -> Any:
        """Build an A/B experiment config for safely rolling out a prompt candidate."""
        from victor.framework.rl.experiment_coordinator import (
            ExperimentConfig,
            Variant,
            VariantType,
        )

        key = self._candidate_key(section_name, provider)
        candidates = self._candidates.get(key, [])
        if not candidates:
            raise ValueError(f"no candidates found for {section_name}/{provider}")

        treatment = next((c for c in candidates if c.text_hash == treatment_hash), None)
        if treatment is None:
            raise ValueError(f"unknown treatment candidate: {treatment_hash}")
        if treatment.requires_benchmark and not treatment.benchmark_passed:
            raise ValueError("cannot create rollout experiment before benchmark gating passes")

        if control_hash:
            control = next(
                (c for c in candidates if c.text_hash == control_hash and c.text_hash != treatment_hash),
                None,
            )
        else:
            approved_controls = [
                candidate
                for candidate in candidates
                if candidate.text_hash != treatment_hash and candidate.benchmark_passed
            ]
            active_controls = [candidate for candidate in approved_controls if candidate.is_active]
            control = active_controls[0] if active_controls else None
            if control is None and approved_controls:
                control = max(
                    approved_controls,
                    key=lambda c: (c.benchmark_score, c.benchmark_runs, c.sample_count, c.generation),
                )
        if control is None:
            raise ValueError("no approved control candidate available for rollout")

        experiment_id = (
            f"prompt_optimizer_{section_name.lower()}_{provider or 'default'}_{treatment_hash}"
        )
        return ExperimentConfig(
            experiment_id=experiment_id,
            name=f"Prompt rollout for {section_name}",
            description=(
                f"Roll out prompt candidate {treatment_hash} against control {control.text_hash} "
                f"for section {section_name} on provider {provider or 'default'}."
            ),
            control=Variant(
                name=control.text_hash,
                type=VariantType.CONTROL,
                config={
                    "learner": "prompt_optimizer",
                    "section_name": section_name,
                    "provider": provider,
                    "text_hash": control.text_hash,
                    "strategy_name": control.strategy_name,
                },
                description=f"Approved control prompt ({control.strategy_name})",
            ),
            treatment=Variant(
                name=treatment.text_hash,
                type=VariantType.TREATMENT,
                config={
                    "learner": "prompt_optimizer",
                    "section_name": section_name,
                    "provider": provider,
                    "text_hash": treatment.text_hash,
                    "strategy_name": treatment.strategy_name,
                },
                description=f"Candidate rollout prompt ({treatment.strategy_name})",
            ),
            traffic_split=traffic_split,
            min_samples_per_variant=min_samples_per_variant,
        )

    def create_rollout_experiment(
        self,
        coordinator: Any,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        control_hash: Optional[str] = None,
        traffic_split: float = 0.1,
        min_samples_per_variant: int = 100,
    ) -> str:
        """Create and start a rollout experiment for an approved candidate."""
        config = self.build_rollout_experiment_config(
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
            control_hash=control_hash,
            traffic_split=traffic_split,
            min_samples_per_variant=min_samples_per_variant,
        )
        if not coordinator.create_experiment(config):
            raise ValueError(f"experiment already exists: {config.experiment_id}")
        if not coordinator.start_experiment(config.experiment_id):
            raise ValueError(f"failed to start experiment: {config.experiment_id}")
        return config.experiment_id

    def rollback_active_candidate(
        self,
        section_name: str,
        provider: str,
        failed_text_hash: Optional[str] = None,
    ) -> Optional[PromptCandidate]:
        """Rollback the active candidate to the best prior approved candidate."""
        key = self._candidate_key(section_name, provider)
        candidates = self._candidates.get(key, [])
        if not candidates:
            return None

        failed_hash = failed_text_hash
        if failed_hash is None:
            active = next((c for c in candidates if c.is_active), None)
            failed_hash = active.text_hash if active else None

        for candidate in candidates:
            if failed_hash and candidate.text_hash == failed_hash:
                candidate.is_active = False
                self._save_candidate(candidate)

        fallback_candidates = [
            c for c in candidates if c.text_hash != failed_hash and c.benchmark_passed
        ]
        if not fallback_candidates:
            return None

        fallback = max(
            fallback_candidates,
            key=lambda c: (c.benchmark_score, c.benchmark_runs, c.sample_count, c.generation),
        )
        for candidate in candidates:
            candidate.is_active = candidate.text_hash == fallback.text_hash
            self._save_candidate(candidate)
        return fallback

    def seed_from_evaluations(self, eval_dir: Optional[Path] = None) -> int:
        """Load evaluation results and update Pareto instance scores.

        Reads eval_swe_bench_*.json files and updates each frontier
        candidate's per-instance scores for multi-objective selection.

        Returns:
            Number of instance scores updated
        """
        if not self._use_pareto:
            return 0

        if eval_dir is None:
            eval_dir = Path.home() / ".victor" / "evaluations"

        import glob as _glob

        updated = 0
        for eval_file in sorted(_glob.glob(str(eval_dir / "eval_swe_bench_*.json"))):
            try:
                with open(eval_file) as f:
                    data = json.load(f)
                model = data.get("config", {}).get("model", "unknown")
                for task in data.get("tasks", []):
                    instance_id = f"{task['task_id']}::{model}"
                    score = 1.0 if task.get("status") == "passed" else 0.0
                    for frontier in self._pareto_frontiers.values():
                        for entry in frontier.get_frontier():
                            frontier.update_instance_score(entry.text_hash, instance_id, score)
                            updated += 1
            except Exception:
                continue

        if updated:
            logger.info("Seeded %d Pareto instance scores from evaluations", updated)
        return updated

    def _select_challenging_traces(
        self, traces: List[ExecutionTrace], max_traces: int = 20
    ) -> List[ExecutionTrace]:
        """SIMBA-inspired: bias selection toward challenging examples.

        Scores each trace by challenge value. Recovery patterns are most
        valuable, followed by high-failure, borderline scores, detailed errors.
        Returns 70/30 mix of challenging/easy traces for contrast.
        """
        if len(traces) < max_traces:
            return traces

        scored = []
        for trace in traces:
            challenge = 0.0
            zone = classify_trace_zone(trace)
            if zone == TraceZone.RECOVERY:
                challenge += 0.4
            total_failures = sum(getattr(trace, "tool_failures", {}).values())
            challenge += 0.3 * min(total_failures / 5.0, 1.0)
            score = getattr(trace, "completion_score", 0.0)
            if 0.1 < score < 0.7:
                challenge += 0.2 * (1.0 - score)
            details = getattr(trace, "tool_call_details", [])
            has_errors = any(getattr(d, "error_detail", "") for d in details)
            if has_errors:
                challenge += 0.1
            scored.append((trace, challenge))

        scored.sort(key=lambda x: -x[1])
        n_challenging = int(max_traces * 0.7)
        n_easy = max_traces - n_challenging
        challenging = [t for t, _ in scored[:n_challenging]]
        easy = [t for t in traces if classify_trace_zone(t) == TraceZone.SUCCESS][:n_easy]
        return challenging + easy

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

        # Convert to ExecutionTrace objects with quality scoring
        # Quality filter: skip sessions with < 2 tool calls (likely API errors)
        for sid, data in list(sessions.items())[-limit:]:
            if data["tool_calls"] < 2:
                continue  # Skip trivially broken sessions

            total_failures = sum(data["failures"].values())
            total_calls = data["tool_calls"]
            failure_rate = total_failures / max(total_calls, 1)

            # Quality-based completion score (not just binary)
            # Low failure rate = high quality trace, worth learning from
            completion_score = max(0.0, 1.0 - failure_rate * 1.5)

            traces.append(
                ExecutionTrace(
                    session_id=sid,
                    task_type=data["task_type"],
                    provider=data.get("provider", "unknown"),
                    model=data.get("model", "unknown"),
                    tool_calls=total_calls,
                    tool_failures=data["failures"],
                    success=failure_rate < 0.3,
                    completion_score=completion_score,
                    tokens_used=data.get("tokens", 0),
                )
            )

        # Sort by quality — high-quality traces first for GEPA reflection
        traces.sort(key=lambda t: -t.completion_score)
        return traces

    @staticmethod
    def _categorize_failure(error: str) -> str:
        """Categorize a tool failure error message into a structured category.

        The order of checks matters: more specific patterns come before
        more generic ones to ensure correct classification.

        Returns one of the 13 FAILURE_HINTS keys.
        """
        lower = error.lower()
        # Filesystem errors
        if "not found" in lower and ("file" in lower or "path" in lower):
            return "file_not_found"
        if "directory" in lower and ("read" in lower or "cannot" in lower):
            return "read_directory"
        if "permission denied" in lower or "access denied" in lower:
            return "permission_denied"
        # Edit errors
        if "old_str" in lower and "not found" in lower:
            return "edit_mismatch"
        if "ambiguous" in lower or ("match" in lower and "found" in lower and "times" in lower):
            return "edit_ambiguous"
        if "syntax error" in lower and ("edit" in lower or "after" in lower):
            return "edit_syntax"
        # Tool errors
        if "tool" in lower and "not found" in lower:
            return "tool_not_found"
        if "timeout" in lower or "timed out" in lower:
            return "timeout"
        # Search errors
        if "no results" in lower or "no matches" in lower:
            return "search_no_results"
        # Test failures
        if "test failed" in lower or ("assertion" in lower and "fail" in lower):
            return "test_failure"
        # Shell command errors
        if "command" in lower and ("fail" in lower or "error" in lower):
            return "shell_error"
        # Generic tool errors (after all specific checks)
        if "error" in lower and "tool" in lower:
            return "tool_error"
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

    def _collect_traces_from_conversations(self, limit: int = 50) -> List[ExecutionTrace]:
        """Collect execution traces from ConversationStore SQLite DB.

        Converts normalized session+message data into ExecutionTrace
        objects that all prompt optimization strategies can consume.
        This supplements JSONL-based traces with richer historical data
        (provider metadata, model family, message counts, duration).
        """
        traces: List[ExecutionTrace] = []
        try:
            from victor.agent.conversation.store import ConversationStore
            from victor.agent.conversation.types import MessageRole
        except ImportError:
            return traces

        try:
            store = ConversationStore()
        except Exception:
            logger.debug("ConversationStore unavailable for trace collection")
            return traces

        try:
            # Get sessions with enough messages to be meaningful
            sessions = store.get_rl_training_data(limit=limit, min_messages=3)
        except Exception as e:
            logger.debug("Failed to query RL training data: %s", e)
            return traces

        for sess in sessions:
            session_id = sess.get("session_id", "")
            provider = sess.get("provider") or "unknown"
            model = sess.get("model") or "unknown"
            tool_msg_count = sess.get("tool_messages") or 0

            # Skip sessions with no tool usage
            if tool_msg_count < 2:
                continue

            # Build tool call details from individual messages
            details: List[ToolCallTrace] = []
            failures: Dict[str, int] = {}
            try:
                session_obj = store.get_session(session_id)
                if session_obj:
                    for msg in session_obj.messages:
                        if msg.role == MessageRole.TOOL_CALL:
                            details.append(
                                ToolCallTrace(
                                    tool_name=msg.tool_name or "",
                                    arguments_summary=msg.content[:200],
                                    reasoning_before="",
                                )
                            )
                        elif msg.role == MessageRole.TOOL:
                            is_error = "error" in msg.content.lower()[:200]
                            if details:
                                last = details[-1]
                                last.success = not is_error
                                last.result_summary = msg.content[:500]
                                if is_error:
                                    last.error_detail = msg.content[:500]
                            if is_error:
                                cat = self._categorize_failure(msg.content[:300])
                                failures[cat] = failures.get(cat, 0) + 1
            except Exception:
                pass  # Fall back to aggregate-only

            total_tool_calls = max(tool_msg_count // 2, 1)
            total_failures = sum(failures.values())
            failure_rate = total_failures / max(total_tool_calls, 1)

            traces.append(
                ExecutionTrace(
                    session_id=session_id,
                    task_type="default",
                    provider=provider,
                    model=model,
                    tool_calls=total_tool_calls,
                    tool_failures=failures,
                    success=failure_rate < 0.3,
                    completion_score=max(0.0, 1.0 - failure_rate * 1.5),
                    tokens_used=0,
                    tool_call_details=details,
                )
            )

        traces.sort(key=lambda t: -t.completion_score)
        return traces

    @staticmethod
    def _merge_traces(
        *trace_lists: List[ExecutionTrace],
    ) -> List[ExecutionTrace]:
        """Merge multiple trace lists, deduplicating by session_id.

        When the same session_id appears in multiple sources, the
        version with more tool_call_details wins (richer data).
        """
        by_id: Dict[str, ExecutionTrace] = {}
        for traces in trace_lists:
            for t in traces:
                existing = by_id.get(t.session_id)
                if existing is None:
                    by_id[t.session_id] = t
                elif len(t.tool_call_details) > len(existing.tool_call_details):
                    # Prefer the richer trace
                    by_id[t.session_id] = t
        merged = list(by_id.values())
        merged.sort(key=lambda t: -t.completion_score)
        return merged

    def _enrich_traces_with_credit(self, traces: List[ExecutionTrace]) -> None:
        """Enrich execution traces with credit assignment signals.

        Pulls recent credit signals from CreditTrackingService (if available
        via DI container) and attaches per-tool credit data to traces.
        This gives GEPA concrete per-tool value attribution for targeted
        prompt mutations.
        """
        try:
            from victor.core import get_container
            from victor.framework.rl.credit_tracking_service import CreditTrackingService

            container = get_container()
            service = container.get_optional(CreditTrackingService)
            if service is None:
                return

            tool_summary = service.get_tool_credit_summary()
            agent_guidance = service.generate_agent_guidance()
            if not tool_summary and not agent_guidance:
                return

            # For each trace, attach credit data for its tools
            for trace in traces:
                credit_data = []
                for detail in trace.tool_call_details:
                    tool_name = getattr(detail, "tool_name", "")
                    if tool_name in tool_summary:
                        credit_data.append(
                            {
                                "tool_name": tool_name,
                                "credit": tool_summary[tool_name]["avg_credit"],
                                "total_credit": tool_summary[tool_name]["total_credit"],
                                "call_count": tool_summary[tool_name]["call_count"],
                            }
                        )
                if credit_data:
                    trace.credit_signals = credit_data
                if agent_guidance:
                    trace.agent_guidance = agent_guidance
        except Exception:
            pass  # Credit enrichment is best-effort

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
