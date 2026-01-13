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

from typing import Any, Dict, List, Optional

from victor.protocols import IPromptContributor, PromptContext


class PromptCoordinator:
    """Prompt building coordination from multiple contributors.

    This coordinator manages multiple IPromptContributor implementations,
    building prompts by aggregating contributions from all sources.

    Responsibilities:
    - Build system prompts from multiple contributors
    - Build task-specific hints
    - Aggregate prompt contributions in priority order
    - Cache built prompts to avoid repeated builds

    Contributors are called in priority order (higher first), with
    later contributors able to override earlier ones.
    """

    def __init__(
        self,
        contributors: Optional[List[IPromptContributor]] = None,
        enable_cache: bool = True,
    ) -> None:
        """Initialize the prompt coordinator.

        Args:
            contributors: List of prompt contributors
            enable_cache: Enable prompt caching
        """
        # Sort contributors by priority (highest first)
        self._contributors = sorted(contributors or [], key=lambda c: c.priority(), reverse=True)
        self._enable_cache = enable_cache
        self._prompt_cache: Dict[str, str] = {}

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
            return self._prompt_cache[cache_key]

        # Aggregate contributions from all contributors
        sections = []
        for contributor in self._contributors:
            try:
                contribution = await contributor.contribute(context)
                if contribution:
                    sections.append(contribution)
            except Exception as e:
                # Log error but continue to next contributor
                import logging

                logging.getLogger(__name__).warning(
                    f"Prompt contributor {contributor.__class__.__name__} failed: {e}"
                )

        # Join sections with newlines
        prompt = "\n\n".join(sections)

        # Cache the result
        if self._enable_cache and prompt:
            self._prompt_cache[cache_key] = prompt

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
            self._prompt_cache.pop(cache_key, None)
        else:
            self._prompt_cache.clear()

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


__all__ = [
    "PromptCoordinator",
    "PromptBuildError",
    "BasePromptContributor",
    "SystemPromptContributor",
    "TaskHintContributor",
]
