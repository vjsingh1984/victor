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

"""Tests for TaskMilestoneMonitor - Goal-aware task progress tracking."""

from victor.agent.milestone_monitor import (
    TaskType,
    Milestone,
    TaskMilestoneMonitor,
)


class TestTaskTypeDetection:
    """Test task type detection from prompts."""

    def test_detect_edit_task_with_add_keyword(self):
        """Detect EDIT task when 'add' keyword is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "Add a version property to the BaseTool class"
        assert tracker.detect_task_type(prompt) == TaskType.EDIT

    def test_detect_edit_task_with_modify_keyword(self):
        """Detect EDIT task when 'modify' keyword is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "Modify the execute method to handle errors"
        assert tracker.detect_task_type(prompt) == TaskType.EDIT

    def test_detect_edit_task_with_change_keyword(self):
        """Detect EDIT task when 'change' keyword is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "Change the default timeout to 60 seconds"
        assert tracker.detect_task_type(prompt) == TaskType.EDIT

    def test_detect_edit_task_with_fix_keyword(self):
        """Detect BUG_FIX task when 'fix the bug' pattern is present.

        Note: "Fix the bug" is now correctly classified as BUG_FIX (not EDIT)
        since the unified classification module prioritizes bug-fix patterns.
        """
        tracker = TaskMilestoneMonitor()
        prompt = "Fix the bug in the authentication module"
        # BUG_FIX is more accurate than EDIT for bug fix requests
        assert tracker.detect_task_type(prompt) == TaskType.BUG_FIX

    def test_detect_search_task_with_find_keyword(self):
        """Detect SEARCH task when 'find' keyword is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "Find all classes that inherit from BaseTool"
        assert tracker.detect_task_type(prompt) == TaskType.SEARCH

    def test_detect_search_task_with_list_keyword(self):
        """Detect SEARCH task when 'list' keyword is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "List all the API endpoints in the codebase"
        assert tracker.detect_task_type(prompt) == TaskType.SEARCH

    def test_detect_search_task_with_where_keyword(self):
        """Detect SEARCH task when 'where' keyword is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "Where is the database connection configured?"
        assert tracker.detect_task_type(prompt) == TaskType.SEARCH

    def test_detect_create_task_with_create_keyword(self):
        """Detect CREATE task when 'create' keyword is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "Create a new Python script called fibonacci.py"
        assert tracker.detect_task_type(prompt) == TaskType.CREATE

    def test_detect_create_simple_task_with_write_keyword(self):
        """Detect CREATE_SIMPLE task for standalone function requests."""
        tracker = TaskMilestoneMonitor()
        prompt = "Write a function to calculate prime numbers"
        # Standalone function requests without file context are CREATE_SIMPLE
        assert tracker.detect_task_type(prompt) == TaskType.CREATE_SIMPLE

    def test_detect_create_task_with_file_context(self):
        """Detect CREATE task when file path context is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "Write a new module in victor/tools/ for handling HTTP requests"
        assert tracker.detect_task_type(prompt) == TaskType.CREATE

    def test_detect_analyze_task_with_analyze_keyword(self):
        """Detect ANALYZE task when 'analyze' keyword is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "Analyze the code in the agent module"
        assert tracker.detect_task_type(prompt) == TaskType.ANALYZE

    def test_detect_analyze_task_with_count_keyword(self):
        """Detect ANALYZE task when 'count' keyword is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "Count the number of Python files in the project"
        assert tracker.detect_task_type(prompt) == TaskType.ANALYZE

    def test_detect_design_task_with_design_keyword(self):
        """Detect DESIGN task when 'design' keyword is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "Design a plugin system for the application"
        assert tracker.detect_task_type(prompt) == TaskType.DESIGN

    def test_detect_design_task_with_outline_keyword(self):
        """Detect DESIGN task when 'outline' keyword is present."""
        tracker = TaskMilestoneMonitor()
        prompt = "Outline the components needed for a rate limiter"
        assert tracker.detect_task_type(prompt) == TaskType.DESIGN

    def test_detect_design_task_with_explain_keyword(self):
        """Detect DESIGN task with 'explain' keyword (conceptual)."""
        tracker = TaskMilestoneMonitor()
        prompt = "Explain the difference between lists and tuples"
        assert tracker.detect_task_type(prompt) == TaskType.DESIGN

    def test_default_to_general_for_ambiguous_prompt(self):
        """Default to GENERAL for prompts without clear task indicators."""
        tracker = TaskMilestoneMonitor()
        prompt = "Help me with this code"
        assert tracker.detect_task_type(prompt) == TaskType.GENERAL


class TestTargetFileExtraction:
    """Test extraction of target files from prompts."""

    def test_extract_single_file_path(self):
        """Extract a single file path from prompt."""
        tracker = TaskMilestoneMonitor()
        prompt = "Add a property to victor/tools/base.py"
        tracker.detect_task_type(prompt)
        assert "victor/tools/base.py" in tracker.progress.target_files

    def test_extract_multiple_file_paths(self):
        """Extract multiple file paths from prompt."""
        tracker = TaskMilestoneMonitor()
        prompt = "Compare victor/agent/orchestrator.py with victor/agent/tool_executor.py"
        tracker.detect_task_type(prompt)
        assert "victor/agent/orchestrator.py" in tracker.progress.target_files
        assert "victor/agent/tool_executor.py" in tracker.progress.target_files

    def test_extract_class_name_as_target(self):
        """Extract class names mentioned in prompt."""
        tracker = TaskMilestoneMonitor()
        prompt = "Add a version property to the BaseTool class"
        tracker.detect_task_type(prompt)
        assert "BaseTool" in tracker.progress.target_entities


class TestMilestoneTracking:
    """Test milestone updates from tool calls."""

    def test_target_read_milestone_on_read_file(self):
        """Set TARGET_READ milestone when target file is read."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Edit victor/tools/base.py to add version")

        # Simulate read_file tool call
        tracker.update_from_tool_call(
            tool_name="read_file",
            args={"path": "victor/tools/base.py"},
            result={"success": True, "content": "..."},
        )

        assert Milestone.TARGET_READ in tracker.progress.milestones

    def test_change_made_milestone_on_edit_files(self):
        """Set CHANGE_MADE milestone when edit_files is used."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Add version property to base.py")

        tracker.update_from_tool_call(
            tool_name="edit_files",
            args={"file_path": "victor/tools/base.py", "changes": "..."},
            result={"success": True},
        )

        assert Milestone.CHANGE_MADE in tracker.progress.milestones

    def test_change_made_milestone_on_write_file(self):
        """Set CHANGE_MADE milestone when write_file is used."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Create a new file")

        tracker.update_from_tool_call(
            tool_name="write_file",
            args={"path": "new_file.py", "content": "..."},
            result={"success": True},
        )

        assert Milestone.CHANGE_MADE in tracker.progress.milestones

    def test_search_complete_milestone_on_code_search(self):
        """Set SEARCH_COMPLETE milestone after code_search finds results."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Find all classes inheriting from BaseTool")

        tracker.update_from_tool_call(
            tool_name="code_search",
            args={"query": "class.*BaseTool"},
            result={"success": True, "results": [{"file": "foo.py"}]},
        )

        assert Milestone.SEARCH_COMPLETE in tracker.progress.milestones


class TestForceActionDecision:
    """Test should_force_action logic."""

    def test_force_action_after_target_read_for_edit_task(self):
        """Force action after target file is read for EDIT task."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Edit victor/tools/base.py")

        # Read target file
        tracker.update_from_tool_call(
            tool_name="read_file", args={"path": "victor/tools/base.py"}, result={"success": True}
        )

        # Simulate 3 iterations
        tracker.increment_iteration()
        tracker.increment_iteration()
        tracker.increment_iteration()

        should_force, hint = tracker.should_force_action()
        assert should_force is True
        assert "edit_files" in hint.lower() or "make the change" in hint.lower()

    def test_no_force_action_before_reading_target(self):
        """Don't force action before target file is read."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Edit victor/tools/base.py")

        tracker.increment_iteration()
        tracker.increment_iteration()
        tracker.increment_iteration()

        should_force, hint = tracker.should_force_action()
        assert should_force is False

    def test_no_force_action_for_design_task(self):
        """Don't force action for DESIGN tasks (no tools needed)."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Outline the components for a rate limiter")

        tracker.increment_iteration()
        tracker.increment_iteration()

        should_force, hint = tracker.should_force_action()
        assert should_force is False


class TestGoalSatisfaction:
    """Test is_goal_satisfied logic."""

    def test_edit_task_satisfied_after_change_made(self):
        """EDIT task is satisfied after CHANGE_MADE milestone."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Add version to base.py")

        tracker.update_from_tool_call(
            tool_name="edit_files", args={"file_path": "base.py"}, result={"success": True}
        )

        assert tracker.is_goal_satisfied() is True

    def test_search_task_satisfied_after_search_complete(self):
        """SEARCH task is satisfied after SEARCH_COMPLETE milestone."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Find all tool classes")

        tracker.update_from_tool_call(
            tool_name="code_search",
            args={"query": "class.*Tool"},
            result={"success": True, "results": [{"file": "a.py"}]},
        )

        assert tracker.is_goal_satisfied() is True

    def test_create_task_satisfied_after_file_written(self):
        """CREATE task is satisfied after file is written."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Create a new script")

        tracker.update_from_tool_call(
            tool_name="write_file", args={"path": "new.py"}, result={"success": True}
        )

        assert tracker.is_goal_satisfied() is True

    def test_design_task_always_satisfied(self):
        """DESIGN tasks are satisfied immediately (no tools needed)."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Design a plugin system")

        # Design tasks don't need tools
        assert tracker.is_goal_satisfied() is True


class TestTaskProgressReset:
    """Test reset functionality for new conversations."""

    def test_reset_clears_milestones(self):
        """Reset clears all milestones."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Edit base.py")
        tracker.update_from_tool_call("read_file", {"path": "base.py"}, {"success": True})

        tracker.reset()

        assert len(tracker.progress.milestones) == 0

    def test_reset_clears_iteration_count(self):
        """Reset clears iteration count."""
        tracker = TaskMilestoneMonitor()
        tracker.increment_iteration()
        tracker.increment_iteration()

        tracker.reset()

        assert tracker.iteration_count == 0


class TestTaskConfig:
    """Test task-specific configuration."""

    def test_edit_task_has_low_exploration_limit(self):
        """EDIT tasks should have low exploration iteration limit."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Add a property")

        config = tracker.get_task_config()
        assert config.max_exploration_iterations <= 4

    def test_search_task_has_higher_exploration_limit(self):
        """SEARCH tasks should have higher exploration limit."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Find all classes")

        config = tracker.get_task_config()
        assert config.max_exploration_iterations >= 6

    def test_task_config_includes_required_tools(self):
        """Task config should include required tools for the task type."""
        tracker = TaskMilestoneMonitor()
        tracker.detect_task_type("Edit base.py")

        config = tracker.get_task_config()
        assert "edit_files" in config.required_tools


class TestYAMLConfigLoader:
    """Test loading configuration from YAML file."""

    def test_load_config_from_yaml(self):
        """Load task config from YAML file."""
        from victor.agent.milestone_monitor import TaskToolConfigLoader

        loader = TaskToolConfigLoader()
        config = loader.load_config()

        # Should have task_types key
        assert "task_types" in config
        assert "edit" in config["task_types"]
        assert "search" in config["task_types"]

    def test_get_stage_tools_for_task(self):
        """Get stage-specific tools for a task type."""
        from victor.agent.milestone_monitor import TaskToolConfigLoader

        loader = TaskToolConfigLoader()
        tools = loader.get_stage_tools("edit", "executing")

        assert "edit_files" in tools
        assert "read_file" in tools

    def test_get_force_action_hint(self):
        """Get force action hint for task type."""
        from victor.agent.milestone_monitor import TaskToolConfigLoader

        loader = TaskToolConfigLoader()
        hint = loader.get_force_action_hint("edit", "after_target_read")

        assert "edit_files" in hint.lower()

    def test_fallback_when_yaml_missing(self):
        """Fallback to hardcoded defaults if YAML missing."""
        from victor.agent.milestone_monitor import TaskToolConfigLoader

        loader = TaskToolConfigLoader(config_path="/nonexistent/path.yaml")
        config = loader.load_config()

        # Should return defaults
        assert config is not None
