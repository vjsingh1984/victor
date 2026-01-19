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

"""Vertical-specific prompt sections.

VerticalPromptSection provides domain-specific content for different verticals.
"""

from dataclasses import dataclass


@dataclass
class VerticalPromptSection:
    """Vertical-specific prompt section.

    Provides domain-specific content for a particular vertical (e.g., coding,
    research, devops, data_analysis). This section associates content with
    a specific vertical identifier.

    Attributes:
        vertical: The vertical identifier (e.g., "coding", "research")
        content: The vertical-specific content
        priority: Ordering priority (lower = earlier in prompt). Default is 40.

    Example:
        section = VerticalPromptSection(
            vertical="coding",
            content="You are an expert software developer...",
            priority=40
        )
        rendered = section.render()
        # "You are an expert software developer..."
    """

    vertical: str
    content: str
    priority: int = 40

    def render(self) -> str:
        """Render section content.

        Returns:
            The vertical-specific content as-is

        Example:
            section = VerticalPromptSection(
                vertical="coding",
                content="Expert developer"
            )
            assert section.render() == "Expert developer"
        """
        return self.content

    def with_priority(self, priority: int) -> "VerticalPromptSection":
        """Create new section with different priority.

        Args:
            priority: New priority value

        Returns:
            New VerticalPromptSection with updated priority

        Example:
            section = VerticalPromptSection("coding", "Content", priority=50)
            higher_priority = section.with_priority(10)
            # higher_priority.priority == 10
            # section.priority == 50 (unchanged)
        """
        return VerticalPromptSection(
            vertical=self.vertical,
            content=self.content,
            priority=priority,
        )

    def __repr__(self) -> str:
        """Return string representation of the section."""
        content_preview = self.content[:30] if len(self.content) > 30 else self.content
        return (
            f"VerticalPromptSection("
            f"vertical={self.vertical}, "
            f"priority={self.priority}, "
            f"content={content_preview}...)"
        )
