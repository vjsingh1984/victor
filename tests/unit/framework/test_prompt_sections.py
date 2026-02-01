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

"""Tests for prompt section classes.

Tests the structured section classes that provide type-safe prompt construction.
"""


from victor.framework.prompt_sections.grounding import GroundingSection
from victor.framework.prompt_sections.rules import RuleSection, SafetyRuleSection
from victor.framework.prompt_sections.checklist import ChecklistSection
from victor.framework.prompt_sections.vertical import VerticalPromptSection


class TestGroundingSection:
    """Tests for GroundingSection."""

    def test_render_basic(self):
        """Test rendering basic content without variables."""
        section = GroundingSection(content="Hello World")
        assert section.render() == "Hello World"

    def test_render_with_variables(self):
        """Test rendering with variable substitution."""
        section = GroundingSection(content="Hello {name}", variables={"name": "World"})
        assert section.render() == "Hello World"

    def test_render_multiple_variables(self):
        """Test rendering with multiple variables."""
        section = GroundingSection(
            content="Project: {project}, Stage: {stage}",
            variables={"project": "Victor", "stage": "dev"},
        )
        rendered = section.render()
        assert "Project: Victor" in rendered
        assert "Stage: dev" in rendered

    def test_with_variables_adds_new(self):
        """Test with_variables adds new variables."""
        section = GroundingSection("Test {a}", variables={"a": 1})
        new_section = section.with_variables(b=2)
        assert new_section.variables == {"a": 1, "b": 2}

    def test_with_variables_overwrites_existing(self):
        """Test with_variables overwrites existing variables."""
        section = GroundingSection("Test {a}", variables={"a": 1})
        new_section = section.with_variables(a=2)
        assert new_section.variables == {"a": 2}

    def test_with_variables_preserves_original(self):
        """Test with_variables preserves original section."""
        section = GroundingSection("Test {a}", variables={"a": 1})
        new_section = section.with_variables(b=2)
        # Original should be unchanged
        assert section.variables == {"a": 1}

    def test_priority_default(self):
        """Test default priority value."""
        section = GroundingSection("Test")
        assert section.priority == 10

    def test_priority_custom(self):
        """Test custom priority value."""
        section = GroundingSection("Test", priority=50)
        assert section.priority == 50

    def test_repr(self):
        """Test string representation."""
        section = GroundingSection(
            "This is a very long content that should be truncated", variables={"key": "value"}
        )
        repr_str = repr(section)
        assert "GroundingSection" in repr_str
        assert "priority=" in repr_str
        assert "key" in repr_str


class TestRuleSection:
    """Tests for RuleSection."""

    def test_render_empty(self):
        """Test rendering empty rules list."""
        section = RuleSection(rules=[])
        assert section.render() == ""

    def test_render_single_rule(self):
        """Test rendering single rule."""
        section = RuleSection(rules=["Rule 1"])
        assert section.render() == "- Rule 1"

    def test_render_multiple_rules(self):
        """Test rendering multiple rules."""
        section = RuleSection(rules=["Rule 1", "Rule 2", "Rule 3"])
        rendered = section.render()
        assert "- Rule 1" in rendered
        assert "- Rule 2" in rendered
        assert "- Rule 3" in rendered
        assert rendered.count("\n") == 2  # 2 newlines for 3 items

    def test_add_rule(self):
        """Test adding a rule."""
        section = RuleSection(rules=["Rule 1"])
        new_section = section.add_rule("Rule 2")
        assert new_section.rules == ["Rule 1", "Rule 2"]
        assert section.rules == ["Rule 1"]  # Original unchanged

    def test_add_rule_multiple(self):
        """Test adding multiple rules."""
        section = RuleSection(rules=["Rule 1"])
        new_section = section.add_rule("Rule 2").add_rule("Rule 3")
        assert new_section.rules == ["Rule 1", "Rule 2", "Rule 3"]

    def test_priority_default(self):
        """Test default priority value."""
        section = RuleSection(rules=["Test"])
        assert section.priority == 20

    def test_priority_custom(self):
        """Test custom priority value."""
        section = RuleSection(rules=["Test"], priority=50)
        assert section.priority == 50

    def test_repr(self):
        """Test string representation."""
        section = RuleSection(rules=["Rule 1", "Rule 2"], priority=30)
        repr_str = repr(section)
        assert "RuleSection" in repr_str
        assert "priority=30" in repr_str
        assert "rules=2" in repr_str


class TestSafetyRuleSection:
    """Tests for SafetyRuleSection."""

    def test_inherits_from_rule_section(self):
        """Test that SafetyRuleSection inherits from RuleSection."""
        section = SafetyRuleSection(rules=["Safety rule 1"])
        assert isinstance(section, RuleSection)

    def test_render_same_as_rule_section(self):
        """Test rendering is same as RuleSection."""
        rules = ["Safety rule 1", "Safety rule 2"]
        safety_section = SafetyRuleSection(rules=rules)
        rule_section = RuleSection(rules=rules)
        assert safety_section.render() == rule_section.render()


class TestChecklistSection:
    """Tests for ChecklistSection."""

    def test_render_empty(self):
        """Test rendering empty checklist."""
        section = ChecklistSection(checklist=[])
        assert section.render() == ""

    def test_render_single_item(self):
        """Test rendering single checklist item."""
        section = ChecklistSection(checklist=["Item 1"])
        assert section.render() == "- [ ] Item 1"

    def test_render_multiple_items(self):
        """Test rendering multiple checklist items."""
        section = ChecklistSection(checklist=["Item 1", "Item 2", "Item 3"])
        rendered = section.render()
        assert "- [ ] Item 1" in rendered
        assert "- [ ] Item 2" in rendered
        assert "- [ ] Item 3" in rendered
        assert rendered.count("\n") == 2  # 2 newlines for 3 items

    def test_add_item(self):
        """Test adding a checklist item."""
        section = ChecklistSection(checklist=["Item 1"])
        new_section = section.add_item("Item 2")
        assert new_section.checklist == ["Item 1", "Item 2"]
        assert section.checklist == ["Item 1"]  # Original unchanged

    def test_add_item_multiple(self):
        """Test adding multiple items."""
        section = ChecklistSection(checklist=["Item 1"])
        new_section = section.add_item("Item 2").add_item("Item 3")
        assert new_section.checklist == ["Item 1", "Item 2", "Item 3"]

    def test_priority_default(self):
        """Test default priority value."""
        section = ChecklistSection(checklist=["Test"])
        assert section.priority == 30

    def test_priority_custom(self):
        """Test custom priority value."""
        section = ChecklistSection(checklist=["Test"], priority=50)
        assert section.priority == 50

    def test_repr(self):
        """Test string representation."""
        section = ChecklistSection(checklist=["Item 1", "Item 2"], priority=40)
        repr_str = repr(section)
        assert "ChecklistSection" in repr_str
        assert "priority=40" in repr_str
        assert "items=2" in repr_str


class TestVerticalPromptSection:
    """Tests for VerticalPromptSection."""

    def test_render_basic(self):
        """Test rendering basic content."""
        section = VerticalPromptSection(vertical="coding", content="You are a coder")
        assert section.render() == "You are a coder"

    def test_render_multiline(self):
        """Test rendering multiline content."""
        content = """You are an expert.

Your skills:
- Coding
- Testing"""
        section = VerticalPromptSection(vertical="coding", content=content)
        rendered = section.render()
        assert "You are an expert" in rendered
        assert "Your skills:" in rendered
        assert "- Coding" in rendered

    def test_with_priority(self):
        """Test changing priority."""
        section = VerticalPromptSection(vertical="coding", content="Content", priority=50)
        new_section = section.with_priority(10)
        assert new_section.priority == 10
        assert section.priority == 50  # Original unchanged

    def test_with_priority_preserves_content(self):
        """Test with_priority preserves other attributes."""
        section = VerticalPromptSection(vertical="coding", content="Content")
        new_section = section.with_priority(20)
        assert new_section.vertical == "coding"
        assert new_section.content == "Content"

    def test_priority_default(self):
        """Test default priority value."""
        section = VerticalPromptSection(vertical="coding", content="Test")
        assert section.priority == 40

    def test_vertical_attribute(self):
        """Test vertical attribute is preserved."""
        section = VerticalPromptSection(vertical="research", content="Content")
        assert section.vertical == "research"

    def test_repr(self):
        """Test string representation."""
        section = VerticalPromptSection(
            vertical="coding",
            content="This is a very long content that should be truncated",
            priority=50,
        )
        repr_str = repr(section)
        assert "VerticalPromptSection" in repr_str
        assert "vertical=coding" in repr_str
        assert "priority=50" in repr_str
        assert "..." in repr_str  # Content truncated


class TestSectionImmutability:
    """Tests for section immutability patterns."""

    def test_grounding_section_immutability(self):
        """Test GroundingSection with_variables returns new instance."""
        original = GroundingSection("Test {a}", variables={"a": 1})
        modified = original.with_variables(b=2)

        # Different instances
        assert original is not modified
        # Original unchanged
        assert original.variables == {"a": 1}
        # Modified has both
        assert modified.variables == {"a": 1, "b": 2}

    def test_rule_section_immutability(self):
        """Test RuleSection add_rule returns new instance."""
        original = RuleSection(rules=["Rule 1"])
        modified = original.add_rule("Rule 2")

        # Different instances
        assert original is not modified
        # Original unchanged
        assert original.rules == ["Rule 1"]
        # Modified has both
        assert modified.rules == ["Rule 1", "Rule 2"]

    def test_checklist_section_immutability(self):
        """Test ChecklistSection add_item returns new instance."""
        original = ChecklistSection(checklist=["Item 1"])
        modified = original.add_item("Item 2")

        # Different instances
        assert original is not modified
        # Original unchanged
        assert original.checklist == ["Item 1"]
        # Modified has both
        assert modified.checklist == ["Item 1", "Item 2"]

    def test_vertical_section_immutability(self):
        """Test VerticalPromptSection with_priority returns new instance."""
        original = VerticalPromptSection(vertical="coding", content="Content", priority=50)
        modified = original.with_priority(10)

        # Different instances
        assert original is not modified
        # Original unchanged
        assert original.priority == 50
        # Modified has new priority
        assert modified.priority == 10
