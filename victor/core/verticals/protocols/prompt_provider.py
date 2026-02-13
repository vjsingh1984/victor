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

"""Prompt Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for prompt contribution.
Following ISP, these protocols are focused on a single responsibility:
contributing to system prompts and task hints.

Usage:
    from victor.core.verticals.protocols.prompt_provider import (
        PromptContributorProtocol,
    )

    class CodingPromptContributor(PromptContributorProtocol):
        def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
            return {
                "edit": TaskTypeHint(
                    task_type="edit",
                    hint="[EDIT] Read target file first.",
                    tool_budget=5,
                ),
            }
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Protocol, runtime_checkable

from victor.core.vertical_types import TaskTypeHint

# =============================================================================
# Prompt Contributor Protocol
# =============================================================================


@runtime_checkable
class PromptContributorProtocol(Protocol):
    """Protocol for contributing to system prompts.

    Verticals can contribute domain-specific task hints and system
    prompt sections without modifying framework code.

    Example:
        class CodingPromptContributor(PromptContributorProtocol):
            def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
                return {
                    "edit": TaskTypeHint(
                        task_type="edit",
                        hint="[EDIT] Read target file first, then modify.",
                        tool_budget=5,
                        priority_tools=["read_file", "edit_files"],
                    ),
                }

            def get_system_prompt_section(self) -> str:
                return "When modifying code, always run tests afterward."
    """

    @abstractmethod
    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        """Get task-type-specific prompt hints.

        Returns:
            Dict mapping task types to their hints
        """
        ...

    def get_system_prompt_section(self) -> str:
        """Get a section to append to the system prompt.

        Returns:
            Additional system prompt text (or empty string)
        """
        return ""

    def get_grounding_rules(self) -> str:
        """Get vertical-specific grounding rules.

        Returns:
            Grounding rules text (or empty string for default)
        """
        return ""

    def get_priority(self) -> int:
        """Get priority for prompt section ordering.

        Lower values appear first.

        Returns:
            Priority value (default 50)
        """
        return 50


__all__ = [
    "PromptContributorProtocol",
    "TaskTypeHint",
]
