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

"""Grounding context sections for prompts.

Grounding sections provide contextual information with variable substitution,
ensuring prompts are properly grounded in relevant context.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GroundingSection:
    """Grounding context section for prompts.

    Provides contextual information with support for variable substitution.
    Variables are substituted using {variable_name} syntax.

    Attributes:
        content: The grounding content template with {variable} placeholders
        priority: Ordering priority (lower = earlier in prompt). Default is 10.
        variables: Dictionary mapping variable names to values

    Example:
        section = GroundingSection(
            content="Working on {project} in {language}",
            variables={"project": "Victor", "language": "Python"}
        )
        rendered = section.render()  # "Working on Victor in Python"

        # Add more variables later
        new_section = section.with_variables(stage="development")
        rendered = new_section.render()  # "Working on Victor in Python"
    """

    content: str
    priority: int = 10
    variables: Dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Render section with variables substituted.

        Returns:
            Content with all {variable} placeholders replaced with their values

        Example:
            section = GroundingSection(
                content="Project: {name}",
                variables={"name": "Victor"}
            )
            assert section.render() == "Project: Victor"
        """
        rendered_content = self.content
        for key, value in self.variables.items():
            placeholder = f"{{{key}}}"
            rendered_content = rendered_content.replace(placeholder, str(value))
        return rendered_content

    def with_variables(self, **kwargs: Any) -> "GroundingSection":
        """Create new section with additional variables.

        Existing variables are preserved, and new variables are added.
        If a variable with the same name exists, it will be overridden.

        Args:
            **kwargs: Variable names and values to add

        Returns:
            New GroundingSection with merged variables

        Example:
            base = GroundingSection("Lang: {lang}", variables={"lang": "Python"})
            extended = base.with_variables(version="3.11", env="production")
            # extended.variables == {"lang": "Python", "version": "3.11", "env": "production"}
        """
        new_vars = {**self.variables, **kwargs}
        return GroundingSection(
            content=self.content,
            priority=self.priority,
            variables=new_vars,
        )

    def __repr__(self) -> str:
        """Return string representation of the section."""
        return (
            f"GroundingSection("
            f"content={self.content[:30]}..., "
            f"priority={self.priority}, "
            f"variables={list(self.variables.keys())})"
        )
