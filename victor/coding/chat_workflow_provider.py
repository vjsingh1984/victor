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

"""Coding Chat Workflow Provider for Phase 2.

This module provides the CodingChatWorkflowProvider which exposes chat
workflows for the coding vertical using BaseYAMLWorkflowProvider pattern.

Phase 2: Vertical Chat Workflow Definition
==========================================
- Defines chat workflows in YAML (chat.yaml)
- Registers escape hatches for complex conditions
- Provides automatic workflow triggers
- Integrates with vertical through step handler (Phase 4)

Usage:
    # Provider is automatically registered with workflow registry
    provider = CodingChatWorkflowProvider()
    workflows = provider.get_workflows()

    # Execute chat workflow
    result = await provider.run_compiled_workflow(
        "coding_chat",
        {"user_message": "Fix the bug in user.py"}
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from victor.framework.workflows.base_yaml_provider import BaseYAMLWorkflowProvider

logger = logging.getLogger(__name__)


class CodingChatWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides chat workflows for coding vertical.

    This provider exposes chat-specific workflows that implement the
    agentic chat loop using StateGraph for execution. Workflows are
    defined in YAML with escape hatches for complex conditions.

    Workflows Provided:
        - coding_chat: Full agentic chat loop with planning
        - quick_chat: Simplified chat for quick questions

    Example:
        provider = CodingChatWorkflowProvider()

        # Get available workflows
        workflows = provider.get_workflows()
        print(f"Available workflows: {list(workflows.keys())}")

        # Execute coding chat
        result = await provider.run_compiled_workflow(
            "coding_chat",
            {"user_message": "Implement a binary search tree"}
        )
    """

    def _get_escape_hatches_module(self) -> str:
        """Return the escape hatches module for coding vertical.

        The escape hatches module contains:
        - CONDITIONS: Complex condition functions for YAML workflows
        - TRANSFORMS: Transform functions for state transitions

        For chat workflows, this includes:
        - chat_task_complexity: Determine task complexity
        - has_pending_tool_calls: Check for pending tools
        - can_continue_iteration: Check iteration limit
        - update_conversation_with_tool_results: Update conversation state
        - format_coding_response: Format final response

        Returns:
            Module path string "victor.coding.escape_hatches"
        """
        return "victor.coding.escape_hatches"

    def _get_workflows_directory(self) -> Path:
        """Return the directory containing chat workflow YAML files.

        By default, this returns victor/coding/workflows/ which contains
        the chat.yaml file.

        Returns:
            Path to workflows directory
        """
        # Use parent directory of this file
        return Path(__file__).parent / "workflows"

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatic workflow triggers based on user input patterns.

        These patterns are used to automatically select the appropriate
        chat workflow based on the user's message content.

        Returns:
            List of (regex_pattern, workflow_name) tuples

        Example:
            # Patterns for automatic workflow selection
            [
                (r"(fix|debug|bug)", "coding_chat"),  # Bug fixing uses full chat
                (r"quick|simple|what is", "quick_chat"),  # Quick questions
            ]
        """
        return [
            # Coding tasks with implementation keywords -> full chat
            (
                r"(implement|create|build|design|refactor|migrate|add feature)",
                "coding_chat",
            ),
            # Bug fixing -> full chat
            (r"(fix|debug|bug|issue|error)", "coding_chat"),
            # Test-related -> full chat
            (r"(test|spec|coverage)", "coding_chat"),
            # Quick questions -> quick chat
            (r"(what is|how do|explain|quick|simple)", "quick_chat"),
            # Default fallback -> full chat
            (r".*", "coding_chat"),
        ]

    def get_workflow_for_task_type(self, task_type: str) -> str | None:
        """Get recommended workflow for a specific task type.

        Args:
            task_type: Type of task (e.g., "edit", "create", "debug")

        Returns:
            Workflow name or None if no mapping exists
        """
        # Map task types to workflows
        task_type_mapping = {
            # Implementation tasks
            "create": "coding_chat",
            "edit": "coding_chat",
            "refactor": "coding_chat",
            # Debugging tasks
            "debug": "coding_chat",
            "fix": "coding_chat",
            # Simple tasks
            "explain": "quick_chat",
            "search": "quick_chat",
            "read": "quick_chat",
        }

        return task_type_mapping.get(task_type.lower())

    def __repr__(self) -> str:
        """Return string representation of provider."""
        return f"CodingChatWorkflowProvider(workflows={len(self.get_workflows())})"


# =============================================================================
# Provider Factory
# =============================================================================

_provider_instance: CodingChatWorkflowProvider | None = None


def get_chat_workflow_provider() -> CodingChatWorkflowProvider:
    """Get or create the singleton chat workflow provider.

    This function ensures only one instance of the provider is created,
    which is important for performance and consistency.

    Returns:
        CodingChatWorkflowProvider singleton instance

    Example:
        provider = get_chat_workflow_provider()
        workflows = provider.get_workflows()
    """
    global _provider_instance

    if _provider_instance is None:
        _provider_instance = CodingChatWorkflowProvider()
        logger.info(f"Created chat workflow provider: {_provider_instance}")

    return _provider_instance


__all__ = [
    "CodingChatWorkflowProvider",
    "get_chat_workflow_provider",
]
