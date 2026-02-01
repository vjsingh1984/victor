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

"""Rule sections for prompts.

Rule sections provide ordered lists of behavioral rules or guidelines.
"""

from dataclasses import dataclass


@dataclass
class RuleSection:
    """Rule section for prompts.

    Provides an ordered list of behavioral rules or guidelines.
    Rules are rendered as a bulleted list.

    Attributes:
        rules: List of rule strings
        priority: Ordering priority (lower = earlier in prompt). Default is 20.

    Example:
        section = RuleSection(
            rules=["Always read files first", "Verify changes"],
            priority=20
        )
        rendered = section.render()
        # "- Always read files first\n- Verify changes"
    """

    rules: list[str]
    priority: int = 20

    def render(self) -> str:
        """Render section as bullet list.

        Returns:
            Rules formatted as bulleted list (one per line)

        Example:
            section = RuleSection(rules=["Rule 1", "Rule 2"])
            rendered = section.render()
            assert rendered == "- Rule 1\\n- Rule 2"
        """
        return "\n".join(f"- {rule}" for rule in self.rules)

    def add_rule(self, rule: str) -> "RuleSection":
        """Add a rule to the section.

        Creates a new RuleSection with the additional rule.
        Original section is unchanged (immutable pattern).

        Args:
            rule: Rule string to add

        Returns:
            New RuleSection with the rule added

        Example:
            base = RuleSection(rules=["Rule 1"])
            extended = base.add_rule("Rule 2")
            # extended.rules == ["Rule 1", "Rule 2"]
            # base.rules == ["Rule 1"] (unchanged)
        """
        new_rules = list(self.rules) + [rule]
        return RuleSection(rules=new_rules, priority=self.priority)

    def __repr__(self) -> str:
        """Return string representation of the section."""
        return f"RuleSection(priority={self.priority}, rules={len(self.rules)})"


@dataclass
class SafetyRuleSection(RuleSection):
    """Specialized rule section for safety rules.

    Inherits from RuleSection but provides semantic clarity for safety-related rules.
    This is a marker class that can be used for type checking and filtering.

    Example:
        section = SafetyRuleSection(
            rules=["Never expose credentials", "Validate all input"],
            priority=60
        )
    """

    pass
