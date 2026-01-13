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

"""Prompt contributor protocol for dependency inversion.

This module defines the IPromptContributor protocol that enables
dependency injection for prompt building, following the
Dependency Inversion Principle (DIP).

Design Principles:
    - DIP: PromptCoordinator depends on this protocol, not concrete contributors
    - OCP: New prompt contributors can be added without modifying existing code
    - ISP: Protocol contains only prompt-related methods

Usage:
    class TaskTypePromptContributor(IPromptContributor):
        async def contribute(self, context: PromptContext) -> str:
            if context["task_type"] == "coding":
                return "\\nFocus on writing clean, well-documented code."
            return ""

        def priority(self) -> int:
            return 100
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class IPromptContributor(Protocol):
    """Protocol for prompt content contributors.

    Implementations contribute prompt fragments that are merged
    into the final system prompt. Contributors are called in
    priority order, with higher priority contributors being
    called later (able to override earlier contributions).

    Use cases:
    - Task-specific hints (coding, research, data analysis)
    - Context-aware instructions (Git status, file type)
    - Mode-specific behavior (plan, build, explore)
    - Vertical-specific prompts (coding assistant, research assistant)
    - Dynamic instructions based on conversation state
    """

    async def contribute(self, context: PromptContext) -> str:
        """Generate prompt contribution based on context.

        Args:
            context: Prompt building context with vertical, mode, task type, etc.

        Returns:
            Prompt contribution string (can be empty)

        Example:
            async def contribute(self, context: PromptContext) -> str:
                if context.get("task_type") == "code_review":
                    return "\\nFocus on code quality, security, and performance."
                return ""
        """
        ...

    def priority(self) -> int:
        """Priority for merging contributions.

        Higher priority contributors are called later and can
        override or extend earlier contributions.

        Returns:
            Priority value (0-1000+, higher = later in chain)

        Example Priority Levels:
            - 0-99: Base prompts (role definition, general behavior)
            - 100-499: Context-specific (file type, Git status)
            - 500-999: Task-specific (coding, research, testing)
            - 1000+: Critical overrides (safety, compliance)
        """
        ...


# Type alias for prompt context
PromptContext = Dict[str, Any]
"""Context for prompt building.

Common keys:
    - vertical: str - Vertical name (coding, research, etc.)
    - mode: str - Agent mode (build, plan, explore)
    - task_type: Optional[str] - Task classification
    - stage: Optional[str] - Conversation stage
    - metadata: Dict[str, Any] - Additional context
"""


__all__ = ["IPromptContributor", "PromptContext"]
