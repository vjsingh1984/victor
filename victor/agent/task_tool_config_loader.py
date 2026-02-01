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

"""Task-aware tool configuration loader.

This module provides TaskToolConfigLoader which loads task-specific tool
configuration from YAML, enabling intelligent tool selection based on task type
and conversation stage.
"""

import logging
from pathlib import Path
from typing import Any, Optional, cast

import yaml

logger = logging.getLogger(__name__)


# Default path to task tool config
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "task_tool_config.yaml"


class TaskToolConfigLoader:
    """Loads task-aware tool configuration from YAML.

    This class loads the task_tool_config.yaml file which defines:
    - Task-type specific tool recommendations
    - Stage-based tool availability
    - Force-action hints and thresholds
    """

    # Fallback config when YAML is not available
    DEFAULT_CONFIG: dict[str, Any] = {
        "task_types": {
            "edit": {
                "max_exploration_iterations": 8,  # Increased from 3 - edits often need context
                "force_action_after_target_read": True,
                "required_tools": ["edit_files", "read_file"],
                "stage_tools": {
                    "initial": ["list_directory", "code_search"],
                    "reading": ["read_file", "code_search"],
                    "executing": ["edit_files", "read_file"],
                    "verifying": ["read_file", "run_tests"],
                },
                "force_action_hints": {
                    "after_target_read": "Use edit_files to make the change.",
                    "max_iterations": "Please make the change or explain blockers.",
                },
            },
            "search": {
                "max_exploration_iterations": 8,
                "force_action_after_target_read": False,
                "required_tools": ["code_search", "read_file"],
                "stage_tools": {
                    "initial": ["list_directory", "code_search"],
                    "reading": ["read_file", "code_search"],
                    "executing": ["read_file"],
                    "verifying": ["read_file"],
                },
                "force_action_hints": {
                    "max_iterations": "Please summarize your findings.",
                },
            },
            "create": {
                "max_exploration_iterations": 8,  # Increased from 3 - creates often need context
                "force_action_after_target_read": False,
                "required_tools": ["write_file"],
                "stage_tools": {
                    "initial": ["list_directory", "read_file"],
                    "reading": ["read_file"],
                    "executing": ["write_file", "edit_files"],
                    "verifying": ["read_file", "run_tests"],
                },
                "force_action_hints": {
                    "max_iterations": "Please create the file.",
                },
            },
            "create_simple": {
                "max_exploration_iterations": 1,
                "force_action_after_target_read": False,
                "skip_exploration": True,
                "required_tools": ["write_file"],
                "stage_tools": {
                    "initial": ["write_file"],
                    "reading": [],
                    "executing": ["write_file"],
                    "verifying": ["read_file"],
                },
                "force_action_hints": {
                    "immediate": "Create the code directly using write_file.",
                    "max_iterations": "Please create the file now.",
                },
            },
            "analyze": {
                "max_exploration_iterations": 20,
                "force_action_after_target_read": False,
                "required_tools": ["read_file", "execute_bash"],
                "stage_tools": {
                    "initial": ["list_directory", "code_search"],
                    "reading": ["read_file", "code_search"],
                    "executing": ["execute_bash"],
                    "verifying": ["read_file"],
                },
                "force_action_hints": {
                    "max_iterations": "Please summarize your analysis.",
                },
            },
            "design": {
                # Architecture/design questions require thorough codebase exploration
                # Increased from 2 to 20 to allow proper analysis
                "max_exploration_iterations": 20,
                "force_action_after_target_read": False,
                "needs_tools": True,  # Design tasks NEED tools to explore the codebase
                "required_tools": ["list_directory", "read_file", "code_search"],
                "stage_tools": {
                    "initial": [
                        "list_directory",
                        "code_search",
                        "read_file",
                        "get_project_overview",
                    ],
                    "reading": ["read_file", "code_search", "list_directory"],
                    "executing": [],
                    "verifying": ["read_file"],
                },
                "force_action_hints": {
                    "max_iterations": "Please summarize the architecture and provide your recommendations.",
                },
            },
            "general": {
                "max_exploration_iterations": 15,
                "force_action_after_target_read": False,
                "required_tools": ["read_file", "list_directory"],
                "stage_tools": {
                    "initial": ["list_directory", "code_search", "read_file"],
                    "reading": ["read_file", "code_search"],
                    "executing": ["edit_files", "write_file"],
                    "verifying": ["read_file", "run_tests"],
                },
                "force_action_hints": {
                    "max_iterations": "Please complete the task.",
                },
            },
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the config loader.

        Args:
            config_path: Path to the YAML config file. Uses default if not specified.
        """
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._config: Optional[dict[str, Any]] = None

    def load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        if self._config is not None:
            return self._config

        try:
            if self._config_path.exists():
                with open(self._config_path) as f:
                    self._config = yaml.safe_load(f)
                    logger.debug(f"Loaded task tool config from {self._config_path}")
            else:
                logger.warning(f"Task tool config not found at {self._config_path}, using defaults")
                self._config = self.DEFAULT_CONFIG
        except Exception as e:
            logger.error(f"Failed to load task tool config: {e}")
            self._config = self.DEFAULT_CONFIG

        return self._config

    def get_stage_tools(self, task_type: str, stage: str) -> list[str]:
        """Get tools available for a specific task type and stage.

        Args:
            task_type: The task type (edit, search, create, etc.)
            stage: The conversation stage (initial, reading, executing, verifying)

        Returns:
            List of tool names
        """
        config = self.load_config()
        task_config = config.get("task_types", {}).get(task_type, {})
        stage_tools = task_config.get("stage_tools", {})
        result = stage_tools.get(stage, [])
        return cast(list[str], result) if isinstance(result, list) else []

    def get_force_action_hint(self, task_type: str, hint_type: str) -> str:
        """Get force action hint for a task type.

        Args:
            task_type: The task type
            hint_type: Type of hint (after_target_read, max_iterations)

        Returns:
            Hint message string
        """
        config = self.load_config()
        task_config = config.get("task_types", {}).get(task_type, {})
        hints = task_config.get("force_action_hints", {})
        result = hints.get(hint_type, "Please proceed with the task.")
        return str(result) if result else "Please proceed with the task."

    def get_task_config(self, task_type: str) -> dict[str, Any]:
        """Get full configuration for a task type.

        Args:
            task_type: The task type

        Returns:
            Task configuration dictionary
        """
        config = self.load_config()
        result = config.get("task_types", {}).get(task_type, {})
        return cast(dict[str, Any], result) if isinstance(result, dict) else {}


__all__ = [
    "TaskToolConfigLoader",
    "DEFAULT_CONFIG_PATH",
]
