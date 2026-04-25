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

"""Unified prompt assembly pipeline.

Consolidates PromptComposer, SystemPromptCoordinator, and orchestrator
frozen-prompt glue into a single module that owns:
- Provider tier detection and content routing
- System prompt building + freezing (Tier A/B) or per-turn rebuild (Tier C)
- Per-turn user prefix composition (GEPA, credit, failure hints, skills)
- Parallel read budget hints
- RL prompt_used event emission
- Shell variant resolution and task classification

Replaces:
- victor.agent.prompt_composer.PromptComposer
- victor.agent.coordinators.system_prompt_coordinator.SystemPromptCoordinator
- Orchestrator's _system_prompt_frozen flag and manual TurnContext bridging

Usage:
    pipeline = UnifiedPromptPipeline(
        provider=provider,
        builder=system_prompt_builder,
        registry=content_registry,
        optimizer=optimization_injector,
        task_analyzer=task_analyzer,
        get_context_window=lambda: 128000,
    )

    # Session start
    system_prompt = pipeline.build_system_prompt(project_context=init_md)

    # Per-turn
    prefix = pipeline.compose_turn_prefix(user_message, turn_context)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.tools.core_tool_aliases import CORE_TOOL_ALIASES

if TYPE_CHECKING:
    from victor.agent.content_registry import ContentCategory, ContentItem, ContentRegistry
    from victor.agent.optimization_injector import OptimizationInjector
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


_MODEL_FACING_TOOL_ALIAS_PATTERN = re.compile(
    r"\b("
    + "|".join(sorted((re.escape(name) for name in CORE_TOOL_ALIASES), key=len, reverse=True))
    + r")\b"
)


# ============================================================================
# Tier Detection (from prompt_composer.py)
# ============================================================================


class ProviderTier(Enum):
    """Provider capability tier based on caching support.

    Tier A: API billing discount + KV prefix cache (Anthropic, OpenAI, DeepSeek, etc.)
    Tier B: KV prefix cache only, no billing discount (Ollama, LMStudio, vLLM)
    Tier C: No caching support (unknown/custom providers)
    """

    API_AND_KV = "api_and_kv"
    KV_ONLY = "kv_only"
    NO_CACHE = "no_cache"


class Placement(Enum):
    """Where a content item should be placed."""

    SYSTEM_PROMPT = "system"
    USER_PREFIX = "user_prefix"
    TOOL_DEFINITIONS = "tools"
    OMITTED = "omitted"


def detect_provider_tier(provider: Any) -> ProviderTier:
    """Detect the caching tier from a provider instance."""
    if provider is None:
        return ProviderTier.NO_CACHE

    api_cache = hasattr(provider, "supports_prompt_caching") and provider.supports_prompt_caching()
    kv_cache = (
        hasattr(provider, "supports_kv_prefix_caching") and provider.supports_kv_prefix_caching()
    )

    if api_cache:
        return ProviderTier.API_AND_KV
    if kv_cache:
        return ProviderTier.KV_ONLY
    return ProviderTier.NO_CACHE


# ============================================================================
# Content Router (from prompt_composer.py)
# ============================================================================


class ContentRouter:
    """Routes content items to placements based on provider tier and category."""

    def __init__(self, tier: ProviderTier, edge_sections: Optional[Set[str]] = None):
        self._tier = tier
        self._edge_sections = edge_sections

    @property
    def tier(self) -> ProviderTier:
        return self._tier

    def route(self, item: "ContentItem", is_evolved: bool = False) -> Placement:
        """Decide where a content item should be placed."""
        from victor.agent.content_registry import ContentCategory

        if is_evolved:
            return Placement.USER_PREFIX

        if item.category in (ContentCategory.DYNAMIC, ContentCategory.EPHEMERAL):
            return Placement.USER_PREFIX

        if self._tier == ProviderTier.API_AND_KV:
            return Placement.SYSTEM_PROMPT

        if self._tier == ProviderTier.KV_ONLY:
            if self._edge_sections is not None:
                if item.section_group not in self._edge_sections and not item.required:
                    return Placement.OMITTED
            return Placement.SYSTEM_PROMPT

        # Tier C
        if item.category == ContentCategory.SEMI_STATIC:
            return Placement.USER_PREFIX
        if item.required:
            return Placement.SYSTEM_PROMPT
        if self._edge_sections is not None:
            if item.section_group not in self._edge_sections:
                return Placement.OMITTED
        return Placement.OMITTED


# ============================================================================
# Turn Context (from prompt_composer.py)
# ============================================================================


@dataclass
class TurnContext:
    """Per-turn context for composing user prefix."""

    provider_name: str = ""
    model: str = ""
    task_type: str = "default"
    active_skill_prompt: Optional[str] = None
    last_turn_failed: bool = False
    last_failure_category: Optional[str] = None
    last_failure_error: Optional[str] = None
    reminder_text: Optional[str] = None


@dataclass
class PromptCompletenessAssessment:
    """Compact view of whether a prompt is ready for autonomous execution."""

    score: float
    required_files: List[str] = field(default_factory=list)
    required_outputs: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    missing_elements: List[str] = field(default_factory=list)
    ambiguous_reference: bool = False
    needs_clarification: bool = False
    search_first: bool = False

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize assessment details for testing and observability."""
        return {
            "score": round(self.score, 4),
            "required_files": list(self.required_files),
            "required_outputs": list(self.required_outputs),
            "constraints": list(self.constraints),
            "missing_elements": list(self.missing_elements),
            "ambiguous_reference": self.ambiguous_reference,
            "needs_clarification": self.needs_clarification,
            "search_first": self.search_first,
        }

    def render_guidance(self) -> str:
        """Render a compact execution contract for prompt injection."""
        lines = ["Prompt execution contract:"]
        if self.required_files:
            lines.append(f"- Scope: {', '.join(self.required_files[:3])}")
        if self.required_outputs:
            lines.append(f"- Deliverables: {', '.join(self.required_outputs[:3])}")
        if self.constraints:
            lines.append(f"- Constraints: {', '.join(self.constraints[:3])}")
        if self.missing_elements:
            lines.append(f"- Missing: {', '.join(self.missing_elements[:3])}")
        if self.search_first:
            lines.append(
                "- Search first: locate relevant files with code_search, overview, or read before editing. Do not guess paths."
            )
        if self.needs_clarification:
            lines.append(
                "- Ask one targeted clarification before editing files or taking irreversible actions."
            )
        return "\n".join(lines)


# ============================================================================
# Unified Prompt Pipeline
# ============================================================================


class UnifiedPromptPipeline:
    """Single owner of all prompt assembly decisions.

    Replaces PromptComposer + SystemPromptCoordinator + orchestrator
    frozen-prompt glue code.

    Lifecycle:
      1. __init__: detect tier, create router
      2. build_system_prompt(): assemble + freeze (Tier A/B) or pass-through (Tier C)
      3. compose_turn_prefix(): per-turn dynamic content (GEPA, credit, hints, skills)
      4. unfreeze(): on workspace switch
    """

    def __init__(
        self,
        provider: Any,
        builder: Any,
        registry: Optional["ContentRegistry"] = None,
        optimizer: Optional["OptimizationInjector"] = None,
        task_analyzer: Optional["TaskAnalyzer"] = None,
        get_context_window: Optional[Callable[[], int]] = None,
        session_id: str = "",
        edge_sections: Optional[Set[str]] = None,
        enable_prompt_completeness_guard: Optional[bool] = None,
    ):
        self._tier = detect_provider_tier(provider)
        self._builder = builder
        self._registry = registry
        self._optimizer = optimizer
        self._task_analyzer = task_analyzer
        self._get_context_window = get_context_window or (lambda: 128000)
        self._session_id = session_id
        if enable_prompt_completeness_guard is None:
            try:
                from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

                enable_prompt_completeness_guard = get_feature_flag_manager().is_enabled(
                    FeatureFlag.USE_PROMPT_COMPLETENESS_GUARD
                )
            except Exception:
                enable_prompt_completeness_guard = False
        self.enable_prompt_completeness_guard = enable_prompt_completeness_guard

        self._router = ContentRouter(self._tier, edge_sections)
        self._frozen_prompt: Optional[str] = None
        self._last_prompt_completeness_assessment: Optional[PromptCompletenessAssessment] = None

        # Extract provider/model names for RL events
        self._provider_name = getattr(builder, "provider_name", "") or ""
        self._model_name = getattr(builder, "model", "") or ""

        logger.info(
            "UnifiedPromptPipeline initialized: tier=%s, provider=%s",
            self._tier.value,
            self._provider_name,
        )

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------

    @property
    def tier(self) -> ProviderTier:
        """Provider caching tier."""
        return self._tier

    @property
    def is_frozen(self) -> bool:
        """Whether system prompt is frozen (Tier A/B after first build)."""
        return self._frozen_prompt is not None

    @property
    def builder(self) -> Any:
        """Underlying SystemPromptBuilder (backward compat)."""
        return self._builder

    @property
    def last_prompt_completeness_assessment(self) -> Optional[PromptCompletenessAssessment]:
        """Latest prompt completeness assessment for observability/testing."""
        return self._last_prompt_completeness_assessment

    # ----------------------------------------------------------------
    # System Prompt (session-level)
    # ----------------------------------------------------------------

    def build_system_prompt(self, project_context: Optional[str] = None) -> str:
        """Build system prompt, freezing for Tier A/B providers.

        For Tier A/B: Built once, frozen for the session. Subsequent calls
        return the frozen prompt without re-building.
        For Tier C: Rebuilt every call (no cache benefit from freezing).

        Args:
            project_context: Optional project context (init.md) to append.

        Returns:
            The final system prompt string.
        """
        # Return frozen prompt for Tier A/B
        if self._frozen_prompt is not None:
            return self._frozen_prompt

        # Build base prompt from SystemPromptBuilder
        base_prompt = self._builder.build()

        # Append project context
        if project_context:
            base_prompt = f"{base_prompt}\n\n{project_context}"

        # Add parallel read budget hint for large context windows
        budget_hint = self._get_parallel_read_budget()
        if budget_hint:
            base_prompt = f"{base_prompt}\n\n{budget_hint}"

        # Credit guidance: Tier C only (system prompt rebuilt per-turn)
        # Tier A/B credit goes in compose_turn_prefix() instead
        if self._tier == ProviderTier.NO_CACHE:
            credit = self._get_credit_guidance()
            if credit:
                base_prompt = f"{base_prompt}\n\n{credit}"

        # Emit RL event
        self._emit_prompt_used_event(base_prompt)

        # Freeze for Tier A/B
        if self._tier in (ProviderTier.API_AND_KV, ProviderTier.KV_ONLY):
            self._frozen_prompt = base_prompt

        return base_prompt

    def unfreeze(self) -> None:
        """Unfreeze system prompt (called on workspace switch)."""
        self._frozen_prompt = None
        if self._optimizer:
            self._optimizer.clear_session_cache()

    # ----------------------------------------------------------------
    # Turn Prefix (per-turn)
    # ----------------------------------------------------------------

    def compose_turn_prefix(
        self,
        user_message: str,
        turn_context: TurnContext,
    ) -> str:
        """Compose per-turn user prefix with all dynamic optimization content.

        Assembles GEPA-evolved sections, MIPROv2 few-shots, failure hints,
        skills, context reminders, and credit guidance into a single prefix.

        Args:
            user_message: Current user message (for KNN few-shot selection).
            turn_context: Per-turn context with failure state, skill, etc.

        Returns:
            Prefix string to prepend to user message, or empty string.
        """
        from victor.framework.prompt_dictionary_compressor import compress_prompt_blocks
        from victor.framework.prompt_document import PromptBlock, PromptDocument

        document = PromptDocument()
        self._last_prompt_completeness_assessment = None
        next_priority = 10

        def add_block(name: str, content: Optional[str]) -> None:
            nonlocal next_priority
            if not content:
                return
            normalized = content.strip()
            if not normalized:
                return
            document.upsert(
                PromptBlock(
                    name=f"{name}_{next_priority}",
                    content=normalized,
                    priority=next_priority,
                    header="",
                    kind="turn_prefix",
                )
            )
            next_priority += 10

        if self.enable_prompt_completeness_guard:
            guidance = self._build_prompt_completeness_guidance(user_message, turn_context)
            add_block("prompt_completeness", guidance)

        # 1. GEPA/MIPROv2/CoT evolved sections
        if self._optimizer:
            evolved = self._optimizer.get_evolved_sections(
                provider=turn_context.provider_name,
                model=turn_context.model,
                task_type=turn_context.task_type,
            )
            for index, section in enumerate(evolved):
                add_block(
                    f"evolved_section_{index}",
                    self._canonicalize_system_guidance_text(section),
                )

            # 2. MIPROv2 few-shots (KNN per-query)
            few_shots = self._optimizer.get_few_shots(user_message)
            if few_shots:
                add_block("few_shots", self._canonicalize_system_guidance_text(few_shots))

            # 3. Failure hints (after rollback/error)
            if turn_context.last_turn_failed:
                hint = self._optimizer.get_failure_hint(
                    turn_context.last_failure_category,
                    turn_context.last_failure_error,
                )
                if hint:
                    add_block("failure_hint", self._canonicalize_system_guidance_text(hint))

        # 4. Active skill prompt
        add_block("active_skill_prompt", turn_context.active_skill_prompt)

        # 5. Context reminders
        add_block("context_reminder", turn_context.reminder_text)

        # 6. Credit guidance: Tier A/B only (system prompt is frozen)
        # Tier C credit is already in build_system_prompt()
        if self._tier != ProviderTier.NO_CACHE:
            credit = self._get_credit_guidance()
            if credit:
                add_block("credit_guidance", self._canonicalize_system_guidance_text(credit))

        # 7. Online tool reputation (mid-turn feedback from current session)
        reputation = self._get_tool_reputation_guidance()
        if reputation:
            add_block("tool_reputation", self._canonicalize_system_guidance_text(reputation))

        if not document.iter_renderable_blocks():
            return ""

        compression = compress_prompt_blocks(
            block.content for block in document.iter_renderable_blocks()
        )
        reminder_body = compression.compressed_prompt
        return "<system-reminder>\n" + reminder_body + "\n</system-reminder>\n\n"

    # ----------------------------------------------------------------
    # Migrated from SystemPromptCoordinator
    # ----------------------------------------------------------------

    def resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell aliases to the appropriate enabled shell variant."""
        try:
            from victor.agent.shell_resolver import resolve_shell_variant

            return resolve_shell_variant(tool_name, None, None)
        except (ImportError, Exception) as e:
            logger.debug("Shell resolver unavailable: %s", e)
            return tool_name

    def classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task type based on keywords."""
        if self._task_analyzer:
            try:
                return self._task_analyzer.classify_keywords(user_message)
            except Exception as e:
                logger.debug("Task classification failed: %s", e)
        return {"task_type": "default", "confidence": 0.0}

    def classify_task_with_context(
        self, user_message: str, history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Classify task type with conversation history context."""
        if self._task_analyzer:
            try:
                return self._task_analyzer.classify_with_context(user_message, history or [])
            except Exception as e:
                logger.debug("Task classification with context failed: %s", e)
        return {"task_type": "default", "confidence": 0.0}

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _get_credit_guidance(self) -> Optional[str]:
        """Get credit-driven tool effectiveness guidance if available."""
        try:
            from victor.core import get_container
            from victor.framework.rl.credit_tracking_service import CreditTrackingService

            container = get_container()
            service = container.get_optional(CreditTrackingService)
            if service is None:
                return None
            return service.generate_tool_guidance()
        except Exception:
            return None

    def _get_tool_reputation_guidance(self) -> Optional[str]:
        """Get online tool reputation guidance from ToolPipeline.

        The ToolReputationTracker on the ToolPipeline updates after every
        tool execution. This pulls its current guidance for mid-turn injection.
        """
        try:
            from victor.core import get_container
            from victor.agent.tool_pipeline import ToolPipeline

            container = get_container()
            pipeline = container.get_optional(ToolPipeline)
            if pipeline is None:
                return None
            tracker = getattr(pipeline, "_tool_reputation", None)
            if tracker is None:
                return None
            return tracker.get_selection_guidance()
        except Exception:
            return None

    def _get_parallel_read_budget(self) -> Optional[str]:
        """Get parallel read budget hint for large context windows."""
        context_window = self._get_context_window()
        if context_window < 32768:
            return None

        try:
            from victor.agent.context_compactor import calculate_parallel_read_budget

            budget = calculate_parallel_read_budget(context_window)
            return budget.to_prompt_hint()
        except (ImportError, Exception) as e:
            logger.debug("Parallel read budget unavailable: %s", e)
            return None

    def _build_prompt_completeness_guidance(
        self,
        user_message: str,
        turn_context: TurnContext,
    ) -> Optional[str]:
        """Build a compact execution contract when the prompt needs reinforcement."""
        assessment = self._assess_prompt_completeness(user_message, turn_context)
        self._last_prompt_completeness_assessment = assessment
        if assessment is None:
            return None
        if not (
            assessment.required_files
            or assessment.required_outputs
            or assessment.constraints
            or assessment.missing_elements
            or assessment.search_first
        ):
            return None
        return assessment.render_guidance()

    def _assess_prompt_completeness(
        self,
        user_message: str,
        turn_context: TurnContext,
    ) -> Optional[PromptCompletenessAssessment]:
        """Heuristically assess whether the prompt is specific enough to execute."""
        message = user_message.strip()
        if not message:
            return None

        message_lower = message.lower()
        required_files = self._extract_required_files(message)
        required_outputs = self._extract_required_outputs(message)
        constraints = self._extract_constraints(message)
        scope_hints = self._extract_scope_hints(message)

        action_markers = (
            "fix",
            "add",
            "update",
            "implement",
            "refactor",
            "review",
            "audit",
            "analyze",
            "search",
            "create",
            "write",
            "benchmark",
            "optimize",
            "generate",
            "compare",
            "tabulate",
            "distill",
            "summarize",
        )
        report_markers = (
            "review",
            "audit",
            "analyze",
            "compare",
            "summarize",
            "benchmark",
            "report",
            "findings",
            "table",
            "tabulate",
        )
        action_task = turn_context.task_type not in {"", "default", "chat", "help"} or any(
            marker in message_lower for marker in action_markers
        )
        report_task = turn_context.task_type in {"review", "analysis", "benchmark"} or any(
            marker in message_lower for marker in report_markers
        )
        target_present = bool(required_files or scope_hints)
        deliverable_present = bool(required_outputs) or (action_task and not report_task)
        ambiguous_reference = bool(
            re.search(r"\b(it|this|that|same thing|same one|above|below)\b", message_lower)
        ) and not target_present
        search_first = bool(action_task and scope_hints and not required_files and not ambiguous_reference)

        missing: List[str] = []
        if action_task and not target_present:
            missing.append("target artifact or scope")
        if report_task and not required_outputs:
            missing.append("expected deliverable")
        if ambiguous_reference and "target artifact or scope" not in missing:
            missing.append("target artifact or scope")

        goal_present = action_task or len(message) >= 12
        score = 0.0
        if goal_present:
            score += 0.4
        if target_present:
            score += 0.35
        if deliverable_present:
            score += 0.15
        if constraints:
            score += 0.1
        if ambiguous_reference:
            score = max(0.0, score - 0.25)

        needs_clarification = bool(missing) and (
            ambiguous_reference
            or "target artifact or scope" in missing
            or "expected deliverable" in missing
        )

        if not (required_files or required_outputs or constraints or needs_clarification):
            return None

        return PromptCompletenessAssessment(
            score=min(score, 1.0),
            required_files=required_files,
            required_outputs=required_outputs,
            constraints=constraints,
            missing_elements=missing,
            ambiguous_reference=ambiguous_reference,
            needs_clarification=needs_clarification,
            search_first=search_first,
        )

    def _extract_required_files(self, user_message: str) -> List[str]:
        """Extract file paths from the prompt with task analyzer fallback."""
        if self._task_analyzer and hasattr(self._task_analyzer, "extract_required_files_from_prompt"):
            try:
                paths = self._task_analyzer.extract_required_files_from_prompt(user_message)
                return self._unique_nonempty(paths)
            except Exception as e:
                logger.debug("Task analyzer file extraction failed: %s", e)

        matches = re.findall(
            r"(?:^|\s|[\"'\-])((?:\.{0,2}/)?[\w./-]+/[\w.-]+\.[a-z]{1,10})(?:\s|[\"']|$|[,;:.\)]|\Z)",
            user_message,
            flags=re.IGNORECASE,
        )
        return self._unique_nonempty(matches)

    def _extract_required_outputs(self, user_message: str) -> List[str]:
        """Extract output requirements from the prompt with task analyzer fallback."""
        if self._task_analyzer and hasattr(self._task_analyzer, "extract_required_outputs_from_prompt"):
            try:
                outputs = self._task_analyzer.extract_required_outputs_from_prompt(user_message)
                return self._unique_nonempty(outputs)
            except Exception as e:
                logger.debug("Task analyzer output extraction failed: %s", e)

        message_lower = user_message.lower()
        outputs: List[str] = []
        if re.search(r"findings?\s*table|table\s+of\s+findings?", message_lower):
            outputs.append("findings table")
        if re.search(r"summary|summarize", message_lower):
            outputs.append("summary")
        if re.search(r"\btests?\b", message_lower):
            outputs.append("tests")
        if re.search(r"\bbenchmark\b", message_lower):
            outputs.append("benchmark results")
        return self._unique_nonempty(outputs)

    def _extract_scope_hints(self, user_message: str) -> List[str]:
        """Extract scope hints when no explicit file path is present."""
        hints = re.findall(r"`([^`]{3,80})`", user_message)
        return self._unique_nonempty(hints)

    def _extract_constraints(self, user_message: str) -> List[str]:
        """Extract brief execution constraints from the prompt."""
        patterns = (
            r"\bwithout\s+[^,.;\n]+",
            r"\bpreserve\s+[^,.;\n]+",
            r"\bavoid\s+[^,.;\n]+",
            r"\bmust\s+[^,.;\n]+",
            r"\bonly\s+[^,.;\n]+",
            r"\bexactly\s+[^,.;\n]+",
            r"\bat\s+least\s+[^,.;\n]+",
            r"\bdo not\s+[^,.;\n]+",
            r"\bdon't\s+[^,.;\n]+",
        )
        constraints: List[str] = []
        for pattern in patterns:
            constraints.extend(
                match.group(0).strip() for match in re.finditer(pattern, user_message, re.IGNORECASE)
            )
        return self._unique_nonempty(constraints)

    def _unique_nonempty(self, values: List[str]) -> List[str]:
        """Preserve insertion order while removing empty/duplicate values."""
        deduped: List[str] = []
        seen: Set[str] = set()
        for value in values:
            cleaned = value.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        return deduped

    def _emit_prompt_used_event(self, prompt: str) -> None:
        """Emit PROMPT_USED event for RL prompt template learner."""
        try:
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            is_local = self._provider_name.lower() in {"ollama", "lmstudio", "vllm"}
            prompt_lower = prompt.lower()

            event = RLEvent(
                type=RLEventType.PROMPT_USED,
                success=True,
                quality_score=0.5,
                provider=self._provider_name,
                model=self._model_name,
                task_type="general",
                metadata={
                    "prompt_style": "detailed" if is_local else "structured",
                    "prompt_length": len(prompt),
                    "has_examples": "example" in prompt_lower or "e.g." in prompt_lower,
                    "has_thinking_prompt": "step by step" in prompt_lower,
                    "has_constraints": "must" in prompt_lower or "always" in prompt_lower,
                    "session_id": self._session_id,
                },
            )
            hooks.emit(event)
        except Exception as e:
            logger.debug("Failed to emit prompt_used event: %s", e)

    def _canonicalize_system_guidance_text(self, text: Optional[str]) -> str:
        """Normalize legacy core-tool aliases in system-owned dynamic guidance.

        Applies only to text generated by Victor itself (optimizer, credit,
        reputation hints) before that text is injected into the model-facing
        prompt surface.
        """
        if not text:
            return ""
        return _MODEL_FACING_TOOL_ALIAS_PATTERN.sub(
            lambda match: CORE_TOOL_ALIASES[match.group(0)],
            text,
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "UnifiedPromptPipeline",
    "ProviderTier",
    "Placement",
    "ContentRouter",
    "TurnContext",
    "PromptCompletenessAssessment",
    "detect_provider_tier",
]
