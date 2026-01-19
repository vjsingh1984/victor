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

"""Checklist sections for prompts.

Checklist sections provide verification checklists for quality assurance.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChecklistSection:
    """Checklist section for prompts.

    Provides a verification checklist for quality assurance.
    Items are rendered as Markdown checkbox format.

    Attributes:
        checklist: List of checklist item strings
        priority: Ordering priority (lower = earlier in prompt). Default is 30.

    Example:
        section = ChecklistSection(
            checklist=["Tests pass", "Code compiles"],
            priority=30
        )
        rendered = section.render()
        # "- [ ] Tests pass\\n- [ ] Code compiles"
    """

    checklist: List[str]
    priority: int = 30

    def render(self) -> str:
        """Render section as checklist.

        Returns:
            Checklist formatted as Markdown checkboxes (one per line)

        Example:
            section = ChecklistSection(checklist=["Item 1", "Item 2"])
            rendered = section.render()
            assert rendered == "- [ ] Item 1\\n- [ ] Item 2"
        """
        return "\n".join(f"- [ ] {item}" for item in self.checklist)

    def add_item(self, item: str) -> "ChecklistSection":
        """Add a checklist item.

        Creates a new ChecklistSection with the additional item.
        Original section is unchanged (immutable pattern).

        Args:
            item: Checklist item string to add

        Returns:
            New ChecklistSection with the item added

        Example:
            base = ChecklistSection(checklist=["Item 1"])
            extended = base.add_item("Item 2")
            # extended.checklist == ["Item 1", "Item 2"]
            # base.checklist == ["Item 1"] (unchanged)
        """
        new_items = list(self.checklist) + [item]
        return ChecklistSection(checklist=new_items, priority=self.priority)

    def __repr__(self) -> str:
        """Return string representation of the section."""
        return f"ChecklistSection(priority={self.priority}, items={len(self.checklist)})"
