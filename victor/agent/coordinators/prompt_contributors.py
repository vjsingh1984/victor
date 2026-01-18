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

"""Built-in prompt contributors for PromptCoordinator.

This module provides concrete implementations of IPromptContributor
for common prompt building scenarios.

Design Patterns:
    - Strategy Pattern: Each contributor implements a different prompt strategy
    - Template Method: Base classes provide template for subclasses
    - OCP: New contributors can be added without modifying existing code
    - SRP: Each contributor has a single, focused responsibility

Usage:
    from victor.agent.coordinators.prompt_contributors import (
        VerticalPromptContributor,
        ContextPromptContributor,
        ProjectInstructionsContributor,
    )

    vertical_contributor = VerticalPromptContributor(
        vertical_name="coding",
        system_prompt="You are an expert software developer..."
    )

    project_contributor = ProjectInstructionsContributor(
        root_path="/path/to/project"
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.protocols import IPromptContributor, PromptContext

logger = logging.getLogger(__name__)


class VerticalPromptContributor(IPromptContributor):
    """Contributor that provides vertical-specific system prompts.

    This contributor adds domain-specific guidance based on the active
    vertical (coding, research, devops, etc.).

    Attributes:
        _vertical_name: Name of the vertical (e.g., "coding", "research")
        _system_prompt: Vertical-specific system prompt
        _priority: Contributor priority (default: 100, high priority)
    """

    def __init__(
        self,
        vertical_name: str,
        system_prompt: str,
        priority: int = 100,
    ):
        """Initialize the vertical prompt contributor.

        Args:
            vertical_name: Name of the vertical
            system_prompt: Vertical-specific system prompt
            priority: Contributor priority (default: 100)
        """
        self._vertical_name = vertical_name
        self._system_prompt = system_prompt
        self._priority = priority

    async def contribute(self, context: PromptContext) -> str:
        """Provide vertical-specific system prompt.

        Args:
            context: Prompt context

        Returns:
            Vertical-specific system prompt if vertical matches, empty string otherwise
        """
        context_vertical = (context or {}).get("vertical_name", "")
        if context_vertical == self._vertical_name:
            return self._system_prompt
        return ""

    def priority(self) -> int:
        """Get contributor priority."""
        return self._priority


class ContextPromptContributor(IPromptContributor):
    """Contributor that provides context-aware prompt additions.

    This contributor adds context-specific guidance based on the current
    task, mode, stage, or other context factors.

    Attributes:
        _context_handlers: Dictionary mapping context keys to handlers
        _priority: Contributor priority (default: 75, medium-high priority)
    """

    def __init__(
        self,
        context_handlers: Optional[Dict[str, Any]] = None,
        priority: int = 75,
    ):
        """Initialize the context prompt contributor.

        Args:
            context_handlers: Dictionary mapping context keys to handler functions
            priority: Contributor priority (default: 75)
        """
        self._context_handlers = context_handlers or {}
        self._priority = priority

    async def contribute(self, context: PromptContext) -> str:
        """Provide context-aware prompt additions.

        Args:
            context: Prompt context

        Returns:
            Context-specific prompt additions
        """
        if not context:
            return ""

        contributions = []

        # Call each context handler
        for key, handler in self._context_handlers.items():
            if key in context:
                try:
                    if callable(handler):
                        contribution = handler(context[key], context)
                        if contribution:
                            contributions.append(contribution)
                    elif isinstance(handler, str):
                        # Static string contribution
                        contributions.append(handler)
                except Exception as e:
                    logger.warning(f"Context handler for '{key}' failed: {e}")

        return "\n\n".join(contributions)

    def priority(self) -> int:
        """Get contributor priority."""
        return self._priority

    def add_handler(self, key: str, handler: Any) -> None:
        """Add a context handler.

        Args:
            key: Context key to handle
            handler: Handler function or static string
        """
        self._context_handlers[key] = handler

    def remove_handler(self, key: str) -> None:
        """Remove a context handler.

        Args:
            key: Context key to remove
        """
        self._context_handlers.pop(key, None)


class ProjectInstructionsContributor(IPromptContributor):
    """Contributor that loads project-specific instructions from .victor.md.

    This contributor loads and injects project-specific context and instructions
    from .victor/init.md or similar project configuration files.

    Attributes:
        _root_path: Root directory to search for project files
        _enabled: Whether project context loading is enabled
        _priority: Contributor priority (default: 50, medium priority)
    """

    def __init__(
        self,
        root_path: Optional[str] = None,
        enabled: bool = True,
        priority: int = 50,
    ):
        """Initialize the project instructions contributor.

        Args:
            root_path: Root directory for project context (default: current directory)
            enabled: Whether project context loading is enabled
            priority: Contributor priority (default: 50)
        """
        self._root_path = Path(root_path) if root_path else Path.cwd()
        self._enabled = enabled
        self._priority = priority
        self._project_context: Optional[Any] = None
        self._content_cache: Optional[str] = None

    def _load_project_context(self) -> Optional[Any]:
        """Load project context from filesystem.

        Returns:
            ProjectContext instance if found and loaded, None otherwise
        """
        if not self._enabled:
            return None

        # Lazy load project context
        if self._project_context is None:
            try:
                from victor.context.project_context import ProjectContext

                project_context = ProjectContext(str(self._root_path))
                if project_context.load():
                    self._project_context = project_context
                    logger.info(f"Loaded project context from {project_context.context_file}")
                else:
                    logger.debug("No project context file found")
            except Exception as e:
                logger.warning(f"Failed to load project context: {e}")

        return self._project_context

    async def contribute(self, context: PromptContext) -> str:
        """Provide project-specific instructions.

        Args:
            context: Prompt context (not used directly)

        Returns:
            Project context formatted for system prompt injection
        """
        if not self._enabled:
            return ""

        project_context = self._load_project_context()
        if project_context is None:
            return ""

        # Use cached content if available
        if self._content_cache is None:
            self._content_cache = project_context.get_system_prompt_addition()

        return self._content_cache

    def priority(self) -> int:
        """Get contributor priority."""
        return self._priority

    def invalidate_cache(self) -> None:
        """Invalidate cached project content.

        Call this when the project context file may have changed.
        """
        self._content_cache = None
        if self._project_context is not None:
            # Reload on next access
            try:
                from victor.context.project_context import ProjectContext

                ProjectContext.clear_cache()
            except Exception:
                pass


class ModeAwareContributor(IPromptContributor):
    """Contributor that provides mode-specific prompt additions.

    This contributor adds guidance based on the agent's current mode
    (build, plan, explore).

    Attributes:
        _mode_prompts: Dictionary mapping modes to prompt additions
        _priority: Contributor priority (default: 80, medium-high priority)
    """

    def __init__(
        self,
        mode_prompts: Dict[str, str],
        priority: int = 80,
    ):
        """Initialize the mode-aware contributor.

        Args:
            mode_prompts: Dictionary mapping mode names to prompt additions
            priority: Contributor priority (default: 80)

        Example:
            mode_prompts = {
                "plan": "Focus on thorough analysis and planning.",
                "build": "Focus on implementation and testing.",
                "explore": "Focus on investigation and discovery.",
            }
        """
        self._mode_prompts = mode_prompts
        self._priority = priority

    async def contribute(self, context: PromptContext) -> str:
        """Provide mode-specific prompt additions.

        Args:
            context: Prompt context with mode key

        Returns:
            Mode-specific prompt addition if mode found, empty string otherwise
        """
        if not context:
            return ""

        mode = context.get("mode", "")
        return self._mode_prompts.get(mode, "")

    def priority(self) -> int:
        """Get contributor priority."""
        return self._priority

    def set_mode_prompt(self, mode: str, prompt: str) -> None:
        """Set prompt for a specific mode.

        Args:
            mode: Mode name
            prompt: Prompt addition for this mode
        """
        self._mode_prompts[mode] = prompt

    def get_supported_modes(self) -> List[str]:
        """Get list of supported modes.

        Returns:
            List of mode names
        """
        return list(self._mode_prompts.keys())


class StageAwareContributor(IPromptContributor):
    """Contributor that provides stage-specific prompt additions.

    This contributor adds guidance based on the conversation stage
    (INITIAL, PLANNING, READING, ANALYSIS, EXECUTION, VERIFICATION, COMPLETION).

    Attributes:
        _stage_prompts: Dictionary mapping stages to prompt additions
        _priority: Contributor priority (default: 70, medium priority)
    """

    def __init__(
        self,
        stage_prompts: Dict[str, str],
        priority: int = 70,
    ):
        """Initialize the stage-aware contributor.

        Args:
            stage_prompts: Dictionary mapping stage names to prompt additions
            priority: Contributor priority (default: 70)

        Example:
            stage_prompts = {
                "INITIAL": "Focus on understanding the user's request.",
                "EXECUTION": "Focus on implementing changes carefully.",
                "VERIFICATION": "Focus on thorough testing.",
            }
        """
        self._stage_prompts = stage_prompts
        self._priority = priority

    async def contribute(self, context: PromptContext) -> str:
        """Provide stage-specific prompt additions.

        Args:
            context: Prompt context with stage key

        Returns:
            Stage-specific prompt addition if stage found, empty string otherwise
        """
        if not context:
            return ""

        stage = context.get("stage", "")
        return self._stage_prompts.get(stage, "")

    def priority(self) -> int:
        """Get contributor priority."""
        return self._priority

    def set_stage_prompt(self, stage: str, prompt: str) -> None:
        """Set prompt for a specific stage.

        Args:
            stage: Stage name
            prompt: Prompt addition for this stage
        """
        self._stage_prompts[stage] = prompt

    def get_supported_stages(self) -> List[str]:
        """Get list of supported stages.

        Returns:
            List of stage names
        """
        return list(self._stage_prompts.keys())


class DynamicPromptContributor(IPromptContributor):
    """Contributor that uses a dynamic async function for contribution.

    This contributor allows custom prompt building logic via an async function,
    providing maximum flexibility for complex prompt scenarios.

    Attributes:
        _contributor_func: Async function that takes context and returns prompt string
        _priority: Contributor priority
    """

    def __init__(
        self,
        contributor_func: Any,
        priority: int = 60,
    ):
        """Initialize the dynamic prompt contributor.

        Args:
            contributor_func: Async function that takes PromptContext and returns str
            priority: Contributor priority (default: 60)

        Example:
            async def my_contributor(context: PromptContext) -> str:
                if context.get("task_type") == "debugging":
                    return "Focus on identifying root causes."
                return ""

            contributor = DynamicPromptContributor(my_contributor, priority=90)
        """
        self._contributor_func = contributor_func
        self._priority = priority

    async def contribute(self, context: PromptContext) -> str:
        """Provide dynamic prompt contribution.

        Args:
            context: Prompt context

        Returns:
            Prompt string from contributor function

        Raises:
            Exception: If contributor function raises an exception
        """
        if not callable(self._contributor_func):
            return ""

        result = self._contributor_func(context)

        # Handle both sync and async functions
        if isinstance(result, str):
            return result
        elif hasattr(result, "__await__"):
            # It's a coroutine
            return await result
        else:
            return ""

    def priority(self) -> int:
        """Get contributor priority."""
        return self._priority


__all__ = [
    "VerticalPromptContributor",
    "ContextPromptContributor",
    "ProjectInstructionsContributor",
    "ModeAwareContributor",
    "StageAwareContributor",
    "DynamicPromptContributor",
]
