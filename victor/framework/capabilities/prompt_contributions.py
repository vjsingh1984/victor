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

"""Generic prompt contribution capability provider (Phase 3)."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptContribution:
    """A prompt contribution configuration."""

    name: str
    task_type: str
    hint: str
    tool_budget: int = 15
    priority: int = 50
    system_section: str = ""


class PromptContributionCapability:
    """Generic prompt contribution capability provider.

    Provides common prompt contributions used across verticals:
    - Task type hints
    - System prompt sections
    - Grounding rules

    This reduces duplication in prompt engineering and ensures
    consistent prompting behavior across verticals.

    Phase 3: Generic Capabilities - Framework Layer
    """

    COMMON_HINTS = [
        PromptContribution(
            name="read_first",
            task_type="edit",
            hint="Always read the file before making edits",
            tool_budget=5,
        ),
        PromptContribution(
            name="verify_changes",
            task_type="edit",
            hint="Verify changes compile and pass tests",
            tool_budget=10,
        ),
        PromptContribution(
            name="search_code",
            task_type="search",
            hint="Use grep to search code before reading",
            tool_budget=10,
        ),
    ]

    def __init__(
        self,
        contributions: Optional[List[PromptContribution]] = None,
    ):
        """Initialize prompt contribution capability.

        Args:
            contributions: List of contributions (uses COMMON if None)
        """
        self.contributions = contributions or self.COMMON_HINTS.copy()

    def get_task_hints(self) -> Dict[str, Dict[str, Any]]:
        """Get all task hints as dictionary.

        Returns:
            Dictionary mapping task type to hint configuration
        """
        hints = {}

        for contrib in self.contributions:
            hints[contrib.task_type] = {
                "hint": contrib.hint,
                "tool_budget": contrib.tool_budget,
            }

        return hints

    def get_contributors(self) -> List[Any]:
        """Get prompt contributors for vertical.

        Returns:
            List of prompt contributor adapters (if available)
        """
        # Try to import PromptContributorAdapter
        try:
            from victor.core.verticals.prompt_adapter import PromptContributorAdapter

            contributors = []
            for contrib in self.contributions:
                adapter = PromptContributorAdapter.from_dict(
                    task_hints={
                        contrib.task_type: {
                            "hint": contrib.hint,
                            "tool_budget": contrib.tool_budget,
                        }
                    },
                    system_prompt_section=contrib.system_section,
                    priority=contrib.priority,
                )
                contributors.append(adapter)

            return contributors
        except ImportError:
            # Return empty list if adapter not available
            return []
