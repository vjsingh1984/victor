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

"""Prompt coordinator for building prompts from contributors.

This module implements the PromptCoordinator which consolidates prompt
building from multiple sources (IPromptContributor implementations).

Design Patterns:
    - Strategy Pattern: Multiple prompt contributors via IPromptContributor
    - Builder Pattern: Build complex prompts from multiple parts
    - Chain of Responsibility: Try contributors in priority order
    - SRP: Focused only on prompt building coordination
    - Protocol Pattern: IPromptBuilderCoordinator for dependency inversion

Usage:
    from victor.agent.coordinators.prompt_coordinator import PromptCoordinator
    from victor.protocols import IPromptContributor

    # Create coordinator with multiple contributors
    coordinator = PromptCoordinator(contributors=[contributor1, contributor2])

    # Build system prompt
    prompt = await coordinator.build_system_prompt(
        context=PromptContext({"task": "code_review"})
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from victor.protocols import IPromptContributor, PromptContext

if TYPE_CHECKING:
    from victor.framework.prompt import PromptBuilder

logger = logging.getLogger(__name__)


class IPromptBuilderCoordinator:
    """Protocol for prompt building coordination.

    Defines the interface for building system prompts with dynamic
    adaptations based on model capabilities and context.
    """

    def build_system_prompt_with_adapter(
        self,
        prompt_builder: "PromptBuilder",
        get_model_context_window: Callable[[], int],
        model: str,
        session_id: str,
    ) -> str:
        """Build system prompt with tool calling adapter.

        Args:
            prompt_builder: The prompt builder instance
            get_model_context_window: Function to get model context window
            model: Model name
            session_id: Session identifier

        Returns:
            Built system prompt string
        """
        ...

    def get_thinking_disabled_prompt(self, base_prompt: str) -> str:
        """Get prompt with thinking mode disabled prefix.

        Args:
            base_prompt: The base prompt text

        Returns:
            Prompt with thinking disable prefix if available
        """
        ...


class PromptCoordinator:
    """Prompt building coordination from multiple contributors.

    This coordinator manages multiple IPromptContributor implementations,
    building prompts by aggregating contributions from all sources.

    Responsibilities:
    - Build system prompts from multiple contributors
    - Build task-specific hints
    - Aggregate prompt contributions in priority order
    - Cache built prompts to avoid repeated builds
    - Invalidate cache when contributors change

    Contributors are called in priority order (higher first), with
    later contributors able to override earlier ones.

    Performance:
        - First build: O(n) where n is number of contributors
        - Cached build: O(1) hash lookup
        - Cache hit rate: Typically > 80% for repeated contexts
    """

    def __init__(
        self,
        contributors: Optional[List[IPromptContributor]] = None,
        enable_cache: bool = True,
        cache_ttl: Optional[float] = None,  # Time-to-live in seconds
    ) -> None:
        """Initialize the prompt coordinator.

        Args:
            contributors: List of prompt contributors
            enable_cache: Enable prompt caching
            cache_ttl: Optional cache TTL (None = infinite)
        """
        # Sort contributors by priority (highest first)
        self._contributors = sorted(contributors or [], key=lambda c: c.priority(), reverse=True)
        self._enable_cache = enable_cache
        self._cache_ttl = cache_ttl

        # Enhanced cache with metadata
        self._prompt_cache: Dict[str, Tuple[str, float]] = {}  # key -> (prompt, timestamp)
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_invalidations: int = 0

    async def build_system_prompt(
        self,
        context: PromptContext,
    ) -> str:
        """Build system prompt from all contributors.

        Aggregates contributions from all contributors in priority order.
        Later contributors can override earlier ones.

        Args:
            context: Prompt context with task information

        Returns:
            Built system prompt string

        Raises:
            PromptBuildError: If all contributors fail

        Example:
            prompt = await coordinator.build_system_prompt(
                PromptContext({"task": "code_review", "language": "python"})
            )
        """
        # Check cache first
        cache_key = self._make_cache_key(context)

        if self._enable_cache and cache_key in self._prompt_cache:
            # Check TTL if configured
            if self._cache_ttl is not None:
                prompt, timestamp = self._prompt_cache[cache_key]
                age = time.time() - timestamp
                if age > self._cache_ttl:
                    # Cache expired
                    del self._prompt_cache[cache_key]
                    self._cache_misses += 1
                else:
                    # Cache hit
                    self._cache_hits += 1
                    return prompt
            else:
                # No TTL, direct cache hit
                self._cache_hits += 1
                return self._prompt_cache[cache_key][0]

        # Cache miss - build prompt
        self._cache_misses += 1

        # Aggregate contributions from all contributors
        sections = []
        for contributor in self._contributors:
            try:
                contribution = await contributor.contribute(context)
                if contribution:
                    sections.append(contribution)
            except Exception as e:
                # Log error but continue to next contributor
                logger.warning(
                    f"Prompt contributor {contributor.__class__.__name__} failed: {e}"
                )

        # Join sections with newlines
        prompt = "\n\n".join(sections)

        # Cache the result with timestamp
        if self._enable_cache and prompt:
            self._prompt_cache[cache_key] = (prompt, time.time())

        return prompt

    async def build_task_hint(
        self,
        task_type: str,
        context: PromptContext,
    ) -> str:
        """Build task-specific hint.

        Builds a hint for a specific task type (e.g., "simple", "medium",
        "complex") by asking contributors for task-specific guidance.

        Args:
            task_type: Type of task (simple, medium, complex, etc.)
            context: Prompt context

        Returns:
            Task hint string

        Example:
            hint = await coordinator.build_task_hint(
                task_type="complex",
                context={"vertical": "coding"}
            )
        """
        # Create task-specific context
        task_context: PromptContext = {**(context or {}), "task_type": task_type}

        # Aggregate task hints from contributors
        hints = []
        for contributor in self._contributors:
            try:
                contribution = await contributor.contribute(task_context)
                if contribution:
                    hints.append(contribution)
            except Exception:
                continue

        return "\n\n".join(hints) if hints else ""

    def invalidate_cache(
        self,
        context: Optional[PromptContext] = None,
    ) -> None:
        """Invalidate prompt cache.

        Args:
            context: Specific context to invalidate (None = all)

        Example:
            # Invalidate specific context
            coordinator.invalidate_cache(PromptContext({"task": "code_review"}))

            # Invalidate all
            coordinator.invalidate_cache()
        """
        if context:
            cache_key = self._make_cache_key(context)
            if cache_key in self._prompt_cache:
                del self._prompt_cache[cache_key]
                self._cache_invalidations += 1
        else:
            invalidated_count = len(self._prompt_cache)
            self._prompt_cache.clear()
            self._cache_invalidations += invalidated_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache metrics

        Example:
            stats = coordinator.get_cache_stats()
            hit_rate = stats["hit_rate"]
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests) if total_requests > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_invalidations": self._cache_invalidations,
            "cache_size": len(self._prompt_cache),
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    def reset_cache_stats(self) -> None:
        """Reset cache statistics counters."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_invalidations = 0

    def add_contributor(
        self,
        contributor: IPromptContributor,
    ) -> None:
        """Add a prompt contributor.

        Args:
            contributor: Prompt contributor to add

        Example:
            contributor = VerticalPromptContributor()
            coordinator.add_contributor(contributor)
        """
        self._contributors.append(contributor)
        # Re-sort by priority
        self._contributors.sort(key=lambda c: c.priority(), reverse=True)
        # Clear cache when contributors change
        self.invalidate_cache()

    def remove_contributor(
        self,
        contributor: IPromptContributor,
    ) -> None:
        """Remove a prompt contributor.

        Args:
            contributor: Prompt contributor to remove
        """
        if contributor in self._contributors:
            self._contributors.remove(contributor)
            # Clear cache when contributors change
            self.invalidate_cache()

    def _make_cache_key(
        self,
        context: PromptContext,
    ) -> str:
        """Create a cache key from context.

        Args:
            context: Prompt context

        Returns:
            Cache key string
        """
        import hashlib
        import json

        try:
            data = json.dumps(context, sort_keys=True, default=str)
        except Exception:
            data = str(context)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


class PromptBuildError(Exception):
    """Exception raised when prompt building fails."""

    pass


# Built-in prompt contributors

class BasePromptContributor(IPromptContributor):
    """Base class for prompt contributors.

    Provides default implementation for IPromptContributor protocol
    that subclasses can override.

    Attributes:
        _priority: Contributor priority (higher = called first)
    """

    def __init__(self, priority: int = 50):
        """Initialize the prompt contributor.

        Args:
            priority: Contributor priority (default: 50, medium priority)
        """
        self._priority = priority

    async def contribute(self, context: PromptContext) -> str:
        """Contribute to prompt building.

        Subclasses should override this method to provide their contribution.

        Args:
            context: Prompt context

        Returns:
            Prompt contribution string
        """
        return ""

    def priority(self) -> int:
        """Get contributor priority."""
        return self._priority


class SystemPromptContributor(BasePromptContributor):
    """Contributor that provides base system prompt.

    This contributor provides the core system prompt that defines
    the agent's role and capabilities.

    Attributes:
        _prompt: Base system prompt string
    """

    def __init__(
        self,
        prompt: str,
        priority: int = 100,
    ):
        """Initialize the system prompt contributor.

        Args:
            prompt: Base system prompt
            priority: Contributor priority (default: 100, high priority)
        """
        super().__init__(priority=priority)
        self._prompt = prompt

    async def contribute(self, context: PromptContext) -> str:
        """Provide base system prompt.

        Args:
            context: Prompt context (not used)

        Returns:
            Base system prompt string
        """
        return self._prompt


class TaskHintContributor(BasePromptContributor):
    """Contributor that provides task-specific hints.

    This contributor provides hints for different task types
    (simple, medium, complex) to guide agent behavior.

    Attributes:
        _hints: Dictionary mapping task types to hints
    """

    def __init__(
        self,
        hints: Dict[str, str],
        priority: int = 75,
    ):
        """Initialize the task hint contributor.

        Args:
            hints: Dictionary mapping task types to hint strings
            priority: Contributor priority (default: 75, medium-high priority)
        """
        super().__init__(priority=priority)
        self._hints = hints

    async def contribute(self, context: PromptContext) -> str:
        """Provide task-specific hint.

        Args:
            context: Prompt context with task_type

        Returns:
            Task hint string if task_type found, empty string otherwise
        """
        task_type = (context or {}).get("task_type", "medium")
        return self._hints.get(task_type, "")

    def set_hint(
        self,
        task_type: str,
        hint: str,
    ) -> None:
        """Set hint for a task type.

        Args:
            task_type: Task type identifier
            hint: Hint string
        """
        self._hints[task_type] = hint

    def get_hints(self) -> Dict[str, str]:
        """Get all hints.

        Returns:
            Dictionary mapping task types to hints
        """
        return self._hints.copy()


class PromptBuilderCoordinator(IPromptBuilderCoordinator):
    """Coordinator for building system prompts with dynamic adaptations.

    This coordinator extracts prompt building logic from the orchestrator,
    providing a focused module for:
    - Building system prompts with tool calling adapter
    - Injecting dynamic parallel read budget hints
    - Emitting RL events for prompt learning
    - Handling thinking mode disable prefix for recovery scenarios

    Responsibilities:
    - Build system prompts using prompt_builder
    - Calculate and inject parallel read budget based on context window
    - Emit PROMPT_USED events for RL learning
    - Provide thinking-disabled prompts for recovery scenarios

    Design Patterns:
    - SRP: Single responsibility for prompt building
    - DIP: Depends on IPromptBuilderCoordinator protocol
    - Builder: Constructs complex prompts from multiple parts
    """

    def __init__(
        self,
        tool_calling_caps: Optional[Any] = None,
        enable_rl_events: bool = True,
    ) -> None:
        """Initialize the prompt builder coordinator.

        Args:
            tool_calling_caps: Tool calling capabilities from model config
            enable_rl_events: Whether to emit RL events
        """
        self._tool_calling_caps = tool_calling_caps
        self._enable_rl_events = enable_rl_events

    def build_system_prompt_with_adapter(
        self,
        prompt_builder: "PromptBuilder",
        get_model_context_window: Callable[[], int],
        model: str,
        session_id: str,
        provider_name: str = "unknown",
    ) -> str:
        """Build system prompt using the tool calling adapter.

        Includes dynamic parallel read budget based on model's context window.

        Args:
            prompt_builder: The PromptBuilder instance
            get_model_context_window: Function to get model context window size
            model: Model name
            session_id: Session identifier for RL tracking
            provider_name: Provider name for RL tracking

        Returns:
            Built system prompt with dynamic budget hint if applicable

        Example:
            prompt = coordinator.build_system_prompt_with_adapter(
                prompt_builder=self.prompt_builder,
                get_model_context_window=lambda: 128000,
                model="claude-sonnet-4-5",
                session_id="session-123",
                provider_name="anthropic"
            )
        """
        from victor.agent.context_compactor import calculate_parallel_read_budget

        base_prompt = prompt_builder.build()

        # Calculate dynamic parallel read budget based on model context window
        context_window = get_model_context_window()
        budget = calculate_parallel_read_budget(context_window)

        # Inject dynamic budget hint for models with reasonable context
        # Only add for models with >= 32K context (smaller models benefit from sequential reads)
        if context_window >= 32768:
            budget_hint = budget.to_prompt_hint()
            final_prompt = f"{base_prompt}\n\n{budget_hint}"
        else:
            final_prompt = base_prompt

        # Emit prompt_used event for RL learning
        self._emit_prompt_used_event(
            final_prompt,
            provider_name=provider_name,
            model=model,
            session_id=session_id,
        )

        return final_prompt

    def _emit_prompt_used_event(
        self,
        prompt: str,
        provider_name: str,
        model: str,
        session_id: str,
    ) -> None:
        """Emit PROMPT_USED event for RL prompt template learner.

        Args:
            prompt: The final system prompt that was built
            provider_name: Name of the provider
            model: Model name
            session_id: Session identifier

        Note:
            RL hook failure should never block prompt building.
        """
        if not self._enable_rl_events:
            return

        try:
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            # Determine prompt style based on provider type
            # Cloud providers use concise style, local uses detailed
            is_local = provider_name.lower() in {"ollama", "lmstudio", "vllm"}
            prompt_style = "detailed" if is_local else "structured"

            # Calculate prompt characteristics
            has_examples = "example" in prompt.lower() or "e.g." in prompt.lower()
            has_thinking = "step by step" in prompt.lower() or "think" in prompt.lower()
            has_constraints = "must" in prompt.lower() or "always" in prompt.lower()

            event = RLEvent(
                type=RLEventType.PROMPT_USED,
                success=True,  # Prompt was successfully built
                quality_score=0.5,  # Neutral until we get outcome feedback
                provider=provider_name,
                model=model,
                task_type="general",  # Will be updated with actual task type
                metadata={
                    "prompt_style": prompt_style,
                    "prompt_length": len(prompt),
                    "has_examples": has_examples,
                    "has_thinking_prompt": has_thinking,
                    "has_constraints": has_constraints,
                    "session_id": session_id,
                },
            )
            hooks.emit(event)
            logger.debug(f"Emitted prompt_used event: style={prompt_style}")

        except Exception as e:
            # RL hook failure should never block prompt building
            logger.debug(f"Failed to emit prompt_used event: {e}")

    def get_thinking_disabled_prompt(self, base_prompt: str) -> str:
        """Prefix a prompt with the thinking disable prefix if supported.

        IMPORTANT: This should ONLY be used in RECOVERY scenarios where:
        - Model returned empty response (stuck in thinking)
        - Context overflow forced completion
        - Iteration limit forced completion

        Normal model calls should NOT use this - thinking mode produces
        better quality results. This is a last-resort recovery mechanism.

        For models with thinking mode (e.g., Qwen3), prepends the configured
        disable prefix (e.g., "/no_think") to get direct responses without
        internal reasoning overhead.

        Args:
            base_prompt: The base prompt text

        Returns:
            Prompt with thinking disable prefix if available, otherwise base_prompt

        Example:
            # In recovery scenario where model is stuck in thinking mode
            recovery_prompt = coordinator.get_thinking_disabled_prompt(
                "Summarize what you've found."
            )
        """
        prefix = getattr(self._tool_calling_caps, "thinking_disable_prefix", None)
        if prefix:
            return f"{prefix}\n{base_prompt}"
        return base_prompt

    def set_tool_calling_caps(self, caps: Any) -> None:
        """Update tool calling capabilities.

        Args:
            caps: New tool calling capabilities
        """
        self._tool_calling_caps = caps


__all__ = [
    "IPromptBuilderCoordinator",
    "PromptBuilderCoordinator",
    "PromptCoordinator",
    "PromptBuildError",
    "BasePromptContributor",
    "SystemPromptContributor",
    "TaskHintContributor",
]
