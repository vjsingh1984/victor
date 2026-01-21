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

"""Prompt building coordination for system prompt construction.

This coordinator provides a centralized interface for building and managing
system prompts, including mode-specific prompts, hints, and thinking mode handling.
"""

import logging
from typing import Any, Dict, List, Optional

from victor.agent.coordinators.prompt_builder_protocol import (
    IPromptBuilderCoordinator,
    PromptContext,
    PromptBuilderCoordinatorConfig,
)

logger = logging.getLogger(__name__)


class PromptBuilderCoordinator(IPromptBuilderCoordinator):
    """Coordinator for prompt building operations.

    Handles system prompt construction from contributors, mode-specific
    modifications, tool hints, and thinking mode adjustments.

    Responsibilities:
    - Build system prompts from contributors
    - Apply mode-specific prompt modifications
    - Manage thinking mode prompts
    - Apply tool hints and guidance
    - Cache built prompts

    Attributes:
        config: Coordinator configuration
        prompt_contributors: List of prompt contributors
        base_prompt: Base system prompt
        prompt_cache: Cache for built prompts
    """

    def __init__(
        self,
        config: PromptBuilderCoordinatorConfig,
        base_prompt: str = "",
    ):
        """Initialize PromptBuilderCoordinator.

        Args:
            config: Coordinator configuration
            base_prompt: Base system prompt
        """
        self._config = config
        self._base_prompt = base_prompt
        self._prompt_contributors: List[Any] = []
        self._prompt_cache: Dict[str, str] = {}

    def get_system_prompt(self) -> str:
        """Get current system prompt.

        Returns:
            Current system prompt string
        """
        return self._base_prompt

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt.

        Args:
            prompt: New system prompt
        """
        self._base_prompt = prompt
        self.invalidate_prompt_cache()

    def append_to_system_prompt(self, content: str) -> None:
        """Append content to system prompt.

        Args:
            content: Content to append
        """
        self._base_prompt += "\n\n" + content
        self.invalidate_prompt_cache()

    def build_prompt(
        self,
        context: Optional[PromptContext] = None,
        include_hints: bool = True,
    ) -> str:
        """Build complete system prompt.

        Combines base system prompt with:
        - Mode-specific content
        - Tool hints
        - Vertical context
        - Thinking mode adjustments

        Args:
            context: Prompt building context
            include_hints: Whether to include tool hints

        Returns:
            Complete system prompt
        """
        context = context or PromptContext()

        # Check cache
        if self._config.cache_enabled:
            cache_key = self._get_cache_key(context, include_hints)
            if cache_key in self._prompt_cache:
                return self._prompt_cache[cache_key]

        # Build prompt
        prompt = self._base_prompt

        # Apply mode-specific modifications
        prompt = self.apply_mode_prompt(prompt, context.mode)

        # Apply tool hints
        if include_hints and self._config.include_tool_hints:
            prompt = self.apply_tool_hints(prompt, context.tool_set)

        # Apply thinking mode adjustments
        if not context.thinking_enabled:
            prompt = self.build_thinking_disabled_prompt(prompt)

        # Apply contributors
        for contributor in self._prompt_contributors:
            try:
                contribution = contributor.get_contribution(context)
                if contribution:
                    prompt += "\n\n" + contribution
            except Exception as e:
                logger.warning(f"Prompt contributor failed: {e}")

        # Enforce max length
        if len(prompt) > self._config.max_prompt_length:
            logger.warning(
                f"Prompt exceeds max length ({len(prompt)} > {self._config.max_prompt_length}), truncating"
            )
            prompt = prompt[: self._config.max_prompt_length]

        # Cache result
        if self._config.cache_enabled:
            cache_key = self._get_cache_key(context, include_hints)
            self._prompt_cache[cache_key] = prompt

        return prompt

    def build_thinking_disabled_prompt(self, base_prompt: str) -> str:
        """Build prompt for thinking-disabled mode.

        Removes thinking-related instructions when thinking mode is disabled
        (e.g., for models that don't support extended thinking).

        Args:
            base_prompt: Base prompt with thinking instructions

        Returns:
            Prompt without thinking instructions
        """
        # Remove thinking-related sections
        thinking_keywords = [
            "thinking process",
            "step-by-step reasoning",
            "thought process",
            "reasoning steps",
        ]

        lines = base_prompt.split("\n")
        filtered_lines = []
        skip = False

        for line in lines:
            line_lower = line.lower()
            # Check if line starts thinking section
            if any(keyword in line_lower for keyword in thinking_keywords):
                skip = True
                continue

            # Check if line ends thinking section
            if skip and line.strip() and not any(keyword in line_lower for keyword in thinking_keywords):
                skip = False

            if not skip:
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def apply_mode_prompt(self, base_prompt: str, mode: str) -> str:
        """Apply mode-specific prompt modifications.

        Args:
            base_prompt: Base system prompt
            mode: Agent mode (build, plan, explore)

        Returns:
            Mode-modified prompt
        """
        mode_modifications = {
            "build": "\n\nYou are in BUILD mode. You can make full edits to files.",
            "plan": "\n\nYou are in PLAN mode. Focus on planning and analysis. Limited edits allowed.",
            "explore": "\n\nYou are in EXPLORE mode. Focus on exploration and understanding. No edits allowed.",
        }

        modification = mode_modifications.get(mode, "")
        return base_prompt + modification

    def apply_tool_hints(
        self,
        base_prompt: str,
        tool_set: Optional[Any] = None,
    ) -> str:
        """Apply tool hints to prompt.

        Adds guidance about available tools and when to use them.

        Args:
            base_prompt: Base system prompt
            tool_set: Available tools

        Returns:
            Prompt with tool hints
        """
        if not tool_set:
            return base_prompt

        hints = "\n\nAvailable Tools:\n"
        # Add tool hints from tool_set
        # This is a simplified version - full implementation would query tool_set
        hints += "- You have access to various tools for file operations, code analysis, and more.\n"
        hints += "- Use tools when appropriate to accomplish tasks.\n"
        hints += "- Always check if a tool exists before using it.\n"

        return base_prompt + hints

    def get_prompt_contributors(self) -> List[Any]:
        """Get registered prompt contributors.

        Returns:
            List of prompt contributors
        """
        return self._prompt_contributors.copy()

    def register_prompt_contributor(self, contributor: Any) -> None:
        """Register a prompt contributor.

        Args:
            contributor: Prompt contributor instance
        """
        self._prompt_contributors.append(contributor)
        self.invalidate_prompt_cache()

    def invalidate_prompt_cache(self) -> None:
        """Invalidate cached prompts.

        Forces prompt rebuild on next call to build_prompt().
        """
        self._prompt_cache.clear()

    def _get_cache_key(self, context: PromptContext, include_hints: bool) -> str:
        """Generate cache key for prompt.

        Args:
            context: Prompt building context
            include_hints: Whether hints are included

        Returns:
            Cache key string
        """
        parts = [
            context.mode,
            str(context.thinking_enabled),
            str(include_hints),
        ]
        return "|".join(parts)


def create_prompt_builder_coordinator(
    config: Optional[PromptBuilderCoordinatorConfig] = None,
    base_prompt: str = "",
) -> PromptBuilderCoordinator:
    """Factory function to create PromptBuilderCoordinator.

    Args:
        config: Coordinator configuration (optional)
        base_prompt: Base system prompt

    Returns:
        Configured PromptBuilderCoordinator instance
    """
    if config is None:
        config = PromptBuilderCoordinatorConfig()

    return PromptBuilderCoordinator(
        config=config,
        base_prompt=base_prompt,
    )


__all__ = [
    "PromptBuilderCoordinator",
    "create_prompt_builder_coordinator",
]
