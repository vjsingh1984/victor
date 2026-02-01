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

"""Protocol for prompt building coordination.

This protocol defines the interface for building and managing system prompts,
including mode-specific prompts, hints, and thinking mode handling.
"""

from typing import Any, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.tools import ToolSet


class PromptContext:
    """Context for prompt building.

    Attributes:
        mode: Current agent mode (build, plan, explore)
        thinking_enabled: Whether thinking mode is enabled
        vertical_context: Vertical-specific context
        tool_set: Available tools
        conversation_stage: Current conversation stage
    """

    def __init__(
        self,
        mode: str = "build",
        thinking_enabled: bool = False,
        vertical_context: Optional[Any] = None,
        tool_set: Optional["ToolSet"] = None,
        conversation_stage: str = "initial",
    ) -> None:
        self.mode = mode
        self.thinking_enabled = thinking_enabled
        self.vertical_context = vertical_context
        self.tool_set = tool_set
        self.conversation_stage = conversation_stage


class IPromptBuilderCoordinator(Protocol):
    """Protocol for prompt building coordination.

    This protocol defines the interface for building and managing system prompts.
    The coordinator is responsible for:
    - Building system prompts from contributors
    - Applying mode-specific prompt modifications
    - Managing thinking mode prompts
    - Applying tool hints and guidance
    - Caching built prompts

    Example:
        coordinator = container.get(IPromptBuilderCoordinator)
        context = PromptContext(mode="plan", thinking_enabled=True)
        prompt = coordinator.build_prompt(context)
    """

    def get_system_prompt(self) -> str:
        """Get current system prompt.

        Returns:
            Current system prompt string
        """
        ...

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt.

        Args:
            prompt: New system prompt
        """
        ...

    def append_to_system_prompt(self, content: str) -> None:
        """Append content to system prompt.

        Args:
            content: Content to append
        """
        ...

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
        ...

    def build_thinking_disabled_prompt(self, base_prompt: str) -> str:
        """Build prompt for thinking-disabled mode.

        Removes thinking-related instructions when thinking mode is disabled
        (e.g., for models that don't support extended thinking).

        Args:
            base_prompt: Base prompt with thinking instructions

        Returns:
            Prompt without thinking instructions
        """
        ...

    def apply_mode_prompt(self, base_prompt: str, mode: str) -> str:
        """Apply mode-specific prompt modifications.

        Args:
            base_prompt: Base system prompt
            mode: Agent mode (build, plan, explore)

        Returns:
            Mode-modified prompt
        """
        ...

    def apply_tool_hints(
        self,
        base_prompt: str,
        tool_set: Optional["ToolSet"] = None,
    ) -> str:
        """Apply tool hints to prompt.

        Adds guidance about available tools and when to use them.

        Args:
            base_prompt: Base system prompt
            tool_set: Available tools

        Returns:
            Prompt with tool hints
        """
        ...

    def get_prompt_contributors(self) -> list[Any]:
        """Get registered prompt contributors.

        Returns:
            List of prompt contributors
        """
        ...

    def register_prompt_contributor(self, contributor: Any) -> None:
        """Register a prompt contributor.

        Args:
            contributor: Prompt contributor instance
        """
        ...

    def invalidate_prompt_cache(self) -> None:
        """Invalidate cached prompts.

        Forces prompt rebuild on next call to build_prompt().
        """
        ...


class PromptBuilderCoordinatorConfig:
    """Configuration for PromptBuilderCoordinator.

    Attributes:
        cache_enabled: Whether to cache built prompts
        include_tool_hints: Whether to include tool hints by default
        include_thinking_instructions: Whether to include thinking instructions
        max_prompt_length: Maximum prompt length in characters
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        include_tool_hints: bool = True,
        include_thinking_instructions: bool = True,
        max_prompt_length: int = 50000,
    ) -> None:
        self.cache_enabled = cache_enabled
        self.include_tool_hints = include_tool_hints
        self.include_thinking_instructions = include_thinking_instructions
        self.max_prompt_length = max_prompt_length


__all__ = [
    "IPromptBuilderCoordinator",
    "PromptContext",
    "PromptBuilderCoordinatorConfig",
]
