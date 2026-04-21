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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.content_registry import ContentCategory, ContentItem, ContentRegistry
    from victor.agent.optimization_injector import OptimizationInjector
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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
    ):
        self._tier = detect_provider_tier(provider)
        self._builder = builder
        self._registry = registry
        self._optimizer = optimizer
        self._task_analyzer = task_analyzer
        self._get_context_window = get_context_window or (lambda: 128000)
        self._session_id = session_id

        self._router = ContentRouter(self._tier, edge_sections)
        self._frozen_prompt: Optional[str] = None

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
        parts: List[str] = []

        # 1. GEPA/MIPROv2/CoT evolved sections
        if self._optimizer:
            evolved = self._optimizer.get_evolved_sections(
                provider=turn_context.provider_name,
                model=turn_context.model,
                task_type=turn_context.task_type,
            )
            parts.extend(evolved)

            # 2. MIPROv2 few-shots (KNN per-query)
            few_shots = self._optimizer.get_few_shots(user_message)
            if few_shots:
                parts.append(few_shots)

            # 3. Failure hints (after rollback/error)
            if turn_context.last_turn_failed:
                hint = self._optimizer.get_failure_hint(
                    turn_context.last_failure_category,
                    turn_context.last_failure_error,
                )
                if hint:
                    parts.append(hint)

        # 4. Active skill prompt
        if turn_context.active_skill_prompt:
            parts.append(turn_context.active_skill_prompt)

        # 5. Context reminders
        if turn_context.reminder_text:
            parts.append(turn_context.reminder_text)

        # 6. Credit guidance: Tier A/B only (system prompt is frozen)
        # Tier C credit is already in build_system_prompt()
        if self._tier != ProviderTier.NO_CACHE:
            credit = self._get_credit_guidance()
            if credit:
                parts.append(credit)

        # 7. Online tool reputation (mid-turn feedback from current session)
        reputation = self._get_tool_reputation_guidance()
        if reputation:
            parts.append(reputation)

        if not parts:
            return ""

        return "<system-reminder>\n" + "\n\n".join(parts) + "\n</system-reminder>\n\n"

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


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "UnifiedPromptPipeline",
    "ProviderTier",
    "Placement",
    "ContentRouter",
    "TurnContext",
    "detect_provider_tier",
]
