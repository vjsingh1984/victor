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

"""Unified prompt composition pipeline (DEPRECATED).

.. deprecated::
    Superseded by ``victor.agent.prompt_pipeline.UnifiedPromptPipeline``.
    Import from ``victor.agent.prompt_pipeline`` for new code.

Routes prompt content to the correct placement (system prompt, user message
prefix, tool definitions, or omitted) based on provider caching capabilities
and content lifecycle category.

Architecture:
- System prompt (frozen): Tools, project context, static guidance — cacheable
- User message prefix (dynamic): GEPA/MIPROv2/CoT evolved sections, failure
  hints, reminders, skills — adapts per-turn without breaking cache
- Tool definitions: FULL (Tier A), COMPACT (Tier B), STUB (Tier C)

Research basis:
- arXiv:2601.06007 — System-prompt-only caching optimal (41-80% cost reduction)
- arXiv:2507.19457 — GEPA Pareto frontier + reflection (ICLR 2026)
- arXiv:2404.13208 — Instruction hierarchy (safety in system prompt)
- arXiv:2311.04934 — Prompt Cache modular attention reuse
- arXiv:2503.01163 — Thompson Sampling for prompt strategy selection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from victor.agent.content_registry import ContentCategory, ContentItem, ContentRegistry

if TYPE_CHECKING:
    from victor.agent.optimization_injector import OptimizationInjector

logger = logging.getLogger(__name__)


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
    """Detect the caching tier from a provider instance.

    Args:
        provider: A provider instance with supports_prompt_caching()
                  and supports_kv_prefix_caching() methods.

    Returns:
        ProviderTier based on provider capabilities.
    """
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


class ContentRouter:
    """Routes content items to placements based on provider tier and category.

    The routing logic implements the decision matrix from the architecture plan:
    - DYNAMIC/EPHEMERAL content always goes to user prefix (all tiers)
    - STATIC content goes to system prompt (Tier A/B) or rebuilt per-turn (Tier C)
    - SEMI-STATIC is tier-dependent: system prompt (Tier A/B) or user prefix (Tier C)
    - Edge model sections can filter what's included for Tier B/C
    """

    def __init__(
        self,
        tier: ProviderTier,
        edge_sections: Optional[Set[str]] = None,
    ):
        self._tier = tier
        self._edge_sections = edge_sections

    @property
    def tier(self) -> ProviderTier:
        return self._tier

    def route(self, item: ContentItem, is_evolved: bool = False) -> Placement:
        """Decide where a content item should be placed.

        Args:
            item: The content item to route.
            is_evolved: Whether this is a GEPA/MIPROv2-evolved version
                        (always goes to user prefix regardless of category).

        Returns:
            Placement decision.
        """
        # Evolved content always goes to user prefix — dynamic by definition
        if is_evolved:
            return Placement.USER_PREFIX

        # Dynamic and ephemeral content always goes to user prefix
        if item.category in (ContentCategory.DYNAMIC, ContentCategory.EPHEMERAL):
            return Placement.USER_PREFIX

        # Tier A: Everything cacheable goes to system prompt (90% discount)
        if self._tier == ProviderTier.API_AND_KV:
            return Placement.SYSTEM_PROMPT

        # Tier B: Static/semi-static in system prompt (frozen for KV stability)
        if self._tier == ProviderTier.KV_ONLY:
            # Edge model can filter non-required sections
            if self._edge_sections is not None:
                if item.section_group not in self._edge_sections and not item.required:
                    return Placement.OMITTED
            return Placement.SYSTEM_PROMPT

        # Tier C: Minimize system prompt, move semi-static to user prefix
        if item.category == ContentCategory.SEMI_STATIC:
            return Placement.USER_PREFIX
        if item.required:
            return Placement.SYSTEM_PROMPT
        # Edge model filtering for non-required static items
        if self._edge_sections is not None:
            if item.section_group not in self._edge_sections:
                return Placement.OMITTED
        return Placement.OMITTED


@dataclass
class TurnContext:
    """Per-turn context for composing user prefix.

    Carries the state needed by compose_user_prefix() to assemble
    dynamic content for the current turn.
    """

    provider_name: str = ""
    model: str = ""
    task_type: str = "default"
    active_skill_prompt: Optional[str] = None
    last_turn_failed: bool = False
    last_failure_category: Optional[str] = None
    last_failure_error: Optional[str] = None
    reminder_text: Optional[str] = None


class PromptComposer:
    """Unified prompt composition pipeline.

    Replaces the scattered prompt assembly logic across SystemPromptBuilder,
    orchestrator inline prefix injection, and prompt_builder optimization methods.

    Lifecycle:
    1. build_system_prompt() — Called once at session start (or workspace switch)
    2. compose_user_prefix() — Called every turn with dynamic content
    3. Both are tier-aware: frozen system prompt for Tier A/B, dynamic for Tier C

    Usage:
        composer = PromptComposer(provider, registry, injector)
        sys_prompt = composer.build_system_prompt()
        prefix = composer.compose_user_prefix(user_msg, turn_ctx)
    """

    def __init__(
        self,
        provider: Any,
        registry: ContentRegistry,
        optimization_injector: Optional[OptimizationInjector] = None,
        edge_service: Any = None,
    ):
        self._tier = detect_provider_tier(provider)
        self._registry = registry
        self._optimizer = optimization_injector
        self._edge_service = edge_service
        self._router = ContentRouter(self._tier)
        self._frozen_system_prompt: Optional[str] = None

        logger.info(
            "PromptComposer initialized: tier=%s, optimizer=%s",
            self._tier.value,
            self._optimizer is not None,
        )

    @property
    def tier(self) -> ProviderTier:
        return self._tier

    @property
    def is_frozen(self) -> bool:
        return self._frozen_system_prompt is not None

    def build_system_prompt(
        self,
        base_prompt: str = "",
        project_context: Optional[str] = None,
    ) -> str:
        """Build and optionally freeze the system prompt.

        For Tier A/B: Built once, frozen for the session.
        For Tier C: Built fresh each call (no caching benefit).

        Args:
            base_prompt: Provider-specific base prompt from SystemPromptBuilder.
            project_context: Project context (init.md content) to append.

        Returns:
            The assembled system prompt string.
        """
        # Return frozen prompt if available (Tier A/B)
        if self._frozen_system_prompt is not None:
            return self._frozen_system_prompt

        sections = [base_prompt] if base_prompt else []

        # Route static content items to system prompt
        for item in self._registry.get_by_category(ContentCategory.STATIC):
            placement = self._router.route(item)
            if placement == Placement.SYSTEM_PROMPT and item.default_text:
                sections.append(item.default_text)

        # Route semi-static content to system prompt (Tier A/B)
        if project_context:
            sections.append(project_context)

        prompt = "\n\n".join(s for s in sections if s.strip())

        # Freeze for Tier A/B (cache optimization)
        if self._tier in (ProviderTier.API_AND_KV, ProviderTier.KV_ONLY):
            self._frozen_system_prompt = prompt
            logger.info(
                "[PromptComposer] System prompt frozen: tier=%s, tokens=~%d",
                self._tier.value,
                len(prompt) // 4,
            )

        return prompt

    def unfreeze(self) -> None:
        """Unfreeze system prompt (called on workspace switch)."""
        self._frozen_system_prompt = None
        # Clear optimizer session cache so GEPA re-samples
        if self._optimizer:
            self._optimizer.clear_session_cache()

    def _get_credit_guidance(self) -> Optional[str]:
        """Get credit-driven tool guidance from CreditTrackingService."""
        try:
            from victor.core import get_container

            container = get_container()
            service = container.get_optional("credit_tracking_service")
            if service is None:
                return None
            return service.generate_tool_guidance()
        except Exception:
            return None

    def compose_user_prefix(
        self,
        user_message: str,
        turn_context: TurnContext,
    ) -> str:
        """Compose per-turn user prefix with all dynamic optimization content.

        Assembles GEPA-evolved sections, MIPROv2 few-shots, failure hints,
        skills, and context reminders into a single prefix string.

        Args:
            user_message: The current user message (for KNN few-shot selection).
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

        # 6. Credit-driven tool guidance (FEP-0001 Phase 3 feedback loop)
        credit_guidance = self._get_credit_guidance()
        if credit_guidance:
            parts.append(credit_guidance)

        if not parts:
            return ""

        prefix = "<system-reminder>\n" + "\n\n".join(parts) + "\n</system-reminder>\n\n"

        # Audit logging: emit event and log summary of injected content
        if parts:
            part_names = []
            if turn_context.last_turn_failed:
                part_names.append(f"failure_hint({turn_context.last_failure_category})")
            if turn_context.active_skill_prompt:
                part_names.append("skill")
            if turn_context.reminder_text:
                part_names.append("reminders")
            # Count GEPA/MIPROv2 sections (everything not skill/reminder/hint)
            opt_count = len(parts)
            if turn_context.active_skill_prompt:
                opt_count -= 1
            if turn_context.reminder_text:
                opt_count -= 1
            if turn_context.last_turn_failed:
                opt_count -= 1
            if opt_count > 0:
                part_names.insert(0, f"gepa_sections({opt_count})")

            logger.info(
                "[PromptComposer] User prefix injected: %s, ~%d tokens",
                "+".join(part_names) if part_names else "optimization",
                len(prefix) // 4,
            )

            # Emit observability event (gated by bus availability)
            try:
                from victor.observability.bus import get_event_bus

                bus = get_event_bus()
                if bus:
                    bus.emit(
                        "prompt.user_prefix_injected",
                        {
                            "sections": part_names,
                            "token_estimate": len(prefix) // 4,
                            "tier": self._tier.value,
                            "has_failure_hint": turn_context.last_turn_failed,
                            "provider": turn_context.provider_name,
                        },
                    )
            except Exception:
                pass  # Observability is best-effort

        return prefix
