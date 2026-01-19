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

"""Unit tests for victor.framework.prompts.common_prompts module."""

import pytest

from victor.framework.prompts.common_prompts import (
    AnalysisWorkflow,
    BugFixWorkflow,
    CodeCreationWorkflow,
    ChecklistBuilder,
    GroundingMode,
    GroundingRulesBuilder,
    SafetyRulesBuilder,
    SystemPromptBuilder,
    TaskCategory,
    TaskHint,
    TaskHintBuilder,
    WorkflowTemplate,
)


class TestGroundingRulesBuilder:
    """Tests for GroundingRulesBuilder."""

    def test_minimal_grounding(self):
        """Test building minimal grounding rules."""
        grounding = GroundingRulesBuilder().minimal().build()
        assert "GROUNDING:" in grounding
        assert "tool output only" in grounding
        assert "Never invent file paths" in grounding

    def test_extended_grounding(self):
        """Test building extended grounding rules."""
        grounding = GroundingRulesBuilder().extended().build()
        assert "CRITICAL" in grounding
        assert "NEVER fabricate" in grounding
        assert "NEVER ignore" in grounding
        assert "quote EXACTLY" in grounding

    def test_custom_grounding(self):
        """Test building custom grounding rules."""
        custom_rules = "Always verify with actual data"
        grounding = GroundingRulesBuilder().custom(custom_rules).build()
        assert grounding == custom_rules

    def test_default_is_minimal(self):
        """Test that default mode is minimal."""
        grounding = GroundingRulesBuilder().build()
        assert "tool output only" in grounding


class TestTaskHintBuilder:
    """Tests for TaskHintBuilder."""

    def test_basic_task_hint(self):
        """Test building a basic task hint."""
        hint = (
            TaskHintBuilder()
            .for_task_type("edit")
            .with_category(TaskCategory.MODIFICATION)
            .with_guidance("[EDIT] Read target file first")
            .build()
        )
        assert hint.task_type == "edit"
        assert hint.category == TaskCategory.MODIFICATION
        assert hint.hint == "[EDIT] Read target file first"
        assert hint.tool_budget == 10  # default

    def test_task_hint_with_all_fields(self):
        """Test building task hint with all fields."""
        hint = (
            TaskHintBuilder()
            .for_task_type("debug")
            .with_category(TaskCategory.ANALYSIS)
            .with_guidance("[DEBUG] Find and fix bugs")
            .with_tool_budget(15)
            .with_priority_tools(["read", "grep", "shell"])
            .with_workflow(["Read error", "Find cause", "Fix bug"])
            .with_rules(["Make minimal changes", "Test after fix"])
            .with_anti_patterns(["Don't guess", "Don't refactor"])
            .build()
        )
        assert hint.task_type == "debug"
        assert hint.tool_budget == 15
        assert hint.priority_tools == ["read", "grep", "shell"]
        assert hint.workflow_steps == ["Read error", "Find cause", "Fix bug"]
        assert hint.rules == ["Make minimal changes", "Test after fix"]
        assert hint.anti_patterns == ["Don't guess", "Don't refactor"]

    def test_task_hint_to_dict(self):
        """Test converting TaskHint to dictionary."""
        hint = (
            TaskHintBuilder()
            .for_task_type("create")
            .with_category(TaskCategory.CREATION)
            .with_guidance("[CREATE] Write new code")
            .build()
        )
        hint_dict = hint.to_dict()
        assert hint_dict["task_type"] == "create"
        assert hint_dict["category"] == "creation"
        assert hint_dict["hint"] == "[CREATE] Write new code"
        assert hint_dict["tool_budget"] == 10

    def test_task_hint_builder_missing_task_type(self):
        """Test that builder fails when task_type is not set."""
        with pytest.raises(ValueError, match="task_type must be set"):
            TaskHintBuilder().with_guidance("Some hint").build()

    def test_task_hint_builder_missing_hint(self):
        """Test that builder fails when hint is not set."""
        with pytest.raises(ValueError, match="hint must be set"):
            TaskHintBuilder().for_task_type("edit").build()


class TestSystemPromptBuilder:
    """Tests for SystemPromptBuilder."""

    def test_identity_only(self):
        """Test building system prompt with only identity."""
        prompt = SystemPromptBuilder().with_identity("You are an expert assistant").build()
        assert prompt == "You are an expert assistant"

    def test_identity_with_capabilities(self):
        """Test building system prompt with identity and capabilities."""
        prompt = (
            SystemPromptBuilder()
            .with_identity("You are Victor")
            .with_capabilities(["Code analysis", "Test generation"])
            .build()
        )
        assert "You are Victor" in prompt
        assert "Code analysis" in prompt
        assert "Test generation" in prompt
        assert "Your capabilities:" in prompt

    def test_full_system_prompt(self):
        """Test building full system prompt with all sections."""
        prompt = (
            SystemPromptBuilder()
            .with_identity("You are Victor")
            .with_capabilities(["Code analysis"])
            .with_guidelines(["Understand before modifying", "Make incremental changes"])
            .with_best_practices(["Test frequently", "Document changes"])
            .with_tool_usage("Use read before editing files")
            .build()
        )
        assert "You are Victor" in prompt
        assert "Your capabilities:" in prompt
        assert "Guidelines:" in prompt
        assert "Best practices:" in prompt
        assert "Use read before editing" in prompt

    def test_empty_builder(self):
        """Test building with empty builder."""
        prompt = SystemPromptBuilder().build()
        assert prompt == ""


class TestChecklistBuilder:
    """Tests for ChecklistBuilder."""

    def test_single_item(self):
        """Test building checklist with single item."""
        checklist = ChecklistBuilder().add_item("Check code compiles").build()
        assert "- [ ] Check code compiles" in checklist
        assert "## Checklist" in checklist

    def test_multiple_items(self):
        """Test building checklist with multiple items."""
        checklist = (
            ChecklistBuilder()
            .add_item("Check code compiles")
            .add_item("Run tests")
            .add_item("Update docs")
            .build()
        )
        assert "- [ ] Check code compiles" in checklist
        assert "- [ ] Run tests" in checklist
        assert "- [ ] Update docs" in checklist

    def test_add_items_batch(self):
        """Test adding multiple items at once."""
        items = ["Item 1", "Item 2", "Item 3"]
        checklist = ChecklistBuilder().add_items(items).build()
        for item in items:
            assert f"- [ ] {item}" in checklist

    def test_custom_title(self):
        """Test building checklist with custom title."""
        checklist = ChecklistBuilder().with_title("Security Checklist").add_item("No secrets").build()
        assert "## Security Checklist" in checklist

    def test_empty_checklist(self):
        """Test building empty checklist."""
        checklist = ChecklistBuilder().build()
        assert checklist == ""


class TestSafetyRulesBuilder:
    """Tests for SafetyRulesBuilder."""

    def test_single_rule(self):
        """Test building safety rules with single rule."""
        rules = SafetyRulesBuilder().add_rule("Never expose credentials").build()
        assert "## Safety Rules" in rules
        assert "- Never expose credentials" in rules

    def test_multiple_rules(self):
        """Test building safety rules with multiple rules."""
        rules = (
            SafetyRulesBuilder()
            .add_rule("Never expose credentials")
            .add_rule("Validate all input")
            .add_rule("Use least privilege")
            .build()
        )
        assert "- Never expose credentials" in rules
        assert "- Validate all input" in rules
        assert "- Use least privilege" in rules

    def test_add_rules_batch(self):
        """Test adding multiple rules at once."""
        rule_list = ["Rule 1", "Rule 2", "Rule 3"]
        rules = SafetyRulesBuilder().add_rules(rule_list).build()
        for rule in rule_list:
            assert f"- {rule}" in rules

    def test_empty_safety_rules(self):
        """Test building empty safety rules."""
        rules = SafetyRulesBuilder().build()
        assert rules == ""


class TestWorkflowTemplate:
    """Tests for workflow templates."""

    def test_bug_fix_workflow(self):
        """Test BugFixWorkflow template."""
        workflow = BugFixWorkflow()
        steps = workflow.get_steps()
        assert len(steps) == 5
        assert "UNDERSTAND" in steps[0]
        assert "LOCATE" in steps[1]
        assert "ANALYZE" in steps[2]
        assert "FIX" in steps[3]
        assert "VERIFY" in steps[4]

    def test_bug_fix_workflow_render(self):
        """Test rendering BugFixWorkflow."""
        workflow = BugFixWorkflow()
        rendered = workflow.render()
        assert "Workflow:" in rendered
        assert "1. UNDERSTAND" in rendered
        assert "5. VERIFY" in rendered

    def test_code_creation_workflow(self):
        """Test CodeCreationWorkflow template."""
        workflow = CodeCreationWorkflow()
        steps = workflow.get_steps()
        assert len(steps) == 3
        assert "UNDERSTAND" in steps[0]
        assert "IMPLEMENT" in steps[1]
        assert "VERIFY" in steps[2]

    def test_analysis_workflow(self):
        """Test AnalysisWorkflow template."""
        workflow = AnalysisWorkflow()
        steps = workflow.get_steps()
        assert len(steps) == 4
        assert "EXPLORE" in steps[0]
        assert "ANALYZE" in steps[1]
        assert "SYNTHESIZE" in steps[2]
        assert "REPORT" in steps[3]


class TestTaskHint:
    """Tests for TaskHint dataclass."""

    def test_task_hint_creation(self):
        """Test creating TaskHint directly."""
        hint = TaskHint(
            task_type="edit",
            category=TaskCategory.MODIFICATION,
            hint="[EDIT] Read first",
            tool_budget=5,
            priority_tools=["read", "edit"],
        )
        assert hint.task_type == "edit"
        assert hint.category == TaskCategory.MODIFICATION
        assert hint.hint == "[EDIT] Read first"
        assert hint.tool_budget == 5
        assert hint.priority_tools == ["read", "edit"]
        assert hint.workflow_steps is None  # default

    def test_task_hint_defaults(self):
        """Test TaskHint with default values."""
        hint = TaskHint(
            task_type="general",
            category=TaskCategory.GENERAL,
            hint="[GENERAL] Do the task",
        )
        assert hint.tool_budget == 10  # default
        assert hint.priority_tools == []  # default
        assert hint.workflow_steps is None
        assert hint.rules is None
        assert hint.anti_patterns is None


class TestEnums:
    """Tests for enum values."""

    def test_grounding_mode_values(self):
        """Test GroundingMode enum values."""
        assert GroundingMode.MINIMAL.value == "minimal"
        assert GroundingMode.EXTENDED.value == "extended"
        assert GroundingMode.CUSTOM.value == "custom"

    def test_task_category_values(self):
        """Test TaskCategory enum values."""
        assert TaskCategory.CREATION.value == "creation"
        assert TaskCategory.MODIFICATION.value == "modification"
        assert TaskCategory.ANALYSIS.value == "analysis"
        assert TaskCategory.EXECUTION.value == "execution"
        assert TaskCategory.VERIFICATION.value == "verification"
        assert TaskCategory.GENERAL.value == "general"


class TestBuilderIntegration:
    """Integration tests for builder combinations."""

    def test_build_complete_prompt(self):
        """Test building a complete prompt using multiple builders."""
        # Build grounding
        grounding = GroundingRulesBuilder().extended().build()

        # Build task hint
        hint = (
            TaskHintBuilder()
            .for_task_type("debug")
            .with_category(TaskCategory.ANALYSIS)
            .with_guidance("[DEBUG] Find bugs efficiently")
            .with_tool_budget(12)
            .with_priority_tools(["read", "grep", "test"])
            .build()
        )

        # Build system prompt
        system_prompt = (
            SystemPromptBuilder()
            .with_identity("You are Victor, an expert debugger")
            .with_guidelines(["Read error traces", "Find root cause", "Make minimal fixes"])
            .build()
        )

        # Verify all components are built
        assert "CRITICAL" in grounding
        assert hint.task_type == "debug"
        assert "Victor" in system_prompt
