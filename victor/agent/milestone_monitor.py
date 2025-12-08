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

"""Goal-aware milestone tracking for intelligent orchestration.

.. deprecated::
    This module is deprecated. Use `victor.agent.unified_task_tracker.UnifiedTaskTracker`
    instead, which consolidates TaskMilestoneMonitor and LoopDetector into a single
    unified system.

This module provides task type detection and milestone tracking to enable
the orchestrator to make smarter decisions about when to force action,
when exploration is complete, and what tools are needed.

The primary class is `TaskMilestoneMonitor`.

For loop detection and budget enforcement, see `victor.agent.progress_tracker.LoopDetector`.

The TaskMilestoneMonitor addresses the issue of over-exploration by:
1. Detecting task type from the prompt (EDIT, SEARCH, CREATE, etc.)
2. Tracking milestones (TARGET_READ, CHANGE_MADE, etc.)
3. Deciding when to nudge the LLM to take action
4. Determining when the goal is satisfied
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import yaml

# Import canonical TaskType from task_classifier (single source of truth)
from victor.embeddings.task_classifier import TaskType

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
    DEFAULT_CONFIG: Dict[str, Any] = {
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
        self._config: Optional[Dict[str, Any]] = None

    def load_config(self) -> Dict[str, Any]:
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

    def get_stage_tools(self, task_type: str, stage: str) -> List[str]:
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
        return cast(List[str], result) if isinstance(result, list) else []

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

    def get_task_config(self, task_type: str) -> Dict[str, Any]:
        """Get full configuration for a task type.

        Args:
            task_type: The task type

        Returns:
            Task configuration dictionary
        """
        config = self.load_config()
        result = config.get("task_types", {}).get(task_type, {})
        return cast(Dict[str, Any], result) if isinstance(result, dict) else {}


class Milestone(Enum):
    """Milestones that can be achieved during task execution."""

    TARGET_IDENTIFIED = "target_identified"  # Found the file/entity to work on
    TARGET_READ = "target_read"  # Read the target file
    CHANGE_MADE = "change_made"  # Used edit_files/write_file
    CHANGE_VERIFIED = "change_verified"  # Verified the change (tests/read)
    SEARCH_COMPLETE = "search_complete"  # Found requested items


@dataclass
class TaskConfig:
    """Configuration for a specific task type."""

    max_exploration_iterations: int = 8
    required_tools: Set[str] = field(default_factory=set)
    completion_milestones: Set[Milestone] = field(default_factory=set)
    force_action_after_target_read: bool = False
    needs_tools: bool = True


# Task-specific configurations
TASK_CONFIGS: Dict[TaskType, TaskConfig] = {
    TaskType.EDIT: TaskConfig(
        max_exploration_iterations=3,
        required_tools={"edit_files", "read_file"},
        completion_milestones={Milestone.CHANGE_MADE},
        force_action_after_target_read=True,
        needs_tools=True,
    ),
    TaskType.SEARCH: TaskConfig(
        max_exploration_iterations=8,
        required_tools={"code_search", "semantic_code_search", "read_file", "list_directory"},
        completion_milestones={Milestone.SEARCH_COMPLETE},
        force_action_after_target_read=False,
        needs_tools=True,
    ),
    TaskType.CREATE: TaskConfig(
        max_exploration_iterations=3,
        required_tools={"write_file", "execute_bash"},
        completion_milestones={Milestone.CHANGE_MADE},
        force_action_after_target_read=False,
        needs_tools=True,
    ),
    TaskType.CREATE_SIMPLE: TaskConfig(
        max_exploration_iterations=1,  # Skip exploration - go straight to write_file
        required_tools={"write_file"},
        completion_milestones={Milestone.CHANGE_MADE},
        force_action_after_target_read=False,
        needs_tools=True,
    ),
    TaskType.ANALYZE: TaskConfig(
        max_exploration_iterations=20,
        required_tools={"read", "shell", "ls", "grep"},  # Canonical short names
        completion_milestones={Milestone.SEARCH_COMPLETE},
        force_action_after_target_read=False,
        needs_tools=True,
    ),
    TaskType.DESIGN: TaskConfig(
        max_exploration_iterations=2,
        required_tools=set(),  # No tools typically needed
        completion_milestones=set(),  # Always satisfied
        force_action_after_target_read=False,
        needs_tools=False,  # Conceptual tasks don't need tools
    ),
    TaskType.GENERAL: TaskConfig(
        max_exploration_iterations=15,
        required_tools={"read", "ls"},  # Canonical short names
        completion_milestones=set(),
        force_action_after_target_read=False,
        needs_tools=True,
    ),
    TaskType.ACTION: TaskConfig(
        max_exploration_iterations=25,  # Allow extensive iteration for multi-step actions (web search, git ops)
        required_tools={"shell", "web", "fetch", "write"},  # Canonical short names
        completion_milestones={Milestone.CHANGE_MADE},
        force_action_after_target_read=False,
        needs_tools=True,
    ),
    TaskType.ANALYSIS_DEEP: TaskConfig(
        max_exploration_iterations=30,  # Allow extensive exploration for deep analysis
        required_tools={"read", "grep", "search", "ls"},  # Canonical short names
        completion_milestones={Milestone.SEARCH_COMPLETE},
        force_action_after_target_read=False,
        needs_tools=True,
    ),
}


@dataclass
class TaskProgress:
    """Tracks progress toward task completion."""

    task_type: TaskType = TaskType.GENERAL
    milestones: Set[Milestone] = field(default_factory=set)
    target_files: Set[str] = field(default_factory=set)
    target_entities: Set[str] = field(default_factory=set)  # Classes, functions, etc.
    files_read: Set[str] = field(default_factory=set)
    files_modified: Set[str] = field(default_factory=set)
    iteration_count: int = 0  # Productive iterations (tool calls made)
    total_turns: int = 0  # Total turns including empty continuation prompts


class TaskMilestoneMonitor:
    """Tracks task milestones and provides intelligent orchestration hints.

    This class is responsible for:
    1. Detecting task type from the user prompt
    2. Extracting target files/entities from the prompt
    3. Tracking milestones as tools are executed
    4. Deciding when to force the LLM to take action (proactive nudging)
    5. Determining when the goal is satisfied

    For reactive loop detection and budget enforcement, see LoopDetector.

    Example:
        >>> monitor = TaskMilestoneMonitor()
        >>> monitor.detect_task_type("Add a version property to BaseTool in base.py")
        TaskType.EDIT
        >>> monitor.update_from_tool_call("read_file", {"path": "base.py"}, {"success": True})
        >>> monitor.should_force_action()
        (True, "You have read the target file. Now use edit_files to make the change.")

    """

    # Patterns for detecting task types
    EDIT_PATTERNS = [
        r"\b(add|modify|change|update|edit|fix|remove|delete|refactor|rename)\b",
        r"\b(insert|append|prepend|replace)\b",
    ]
    SEARCH_PATTERNS = [
        r"\b(find|locate|search|where|show me|list all|look for)\b",
        r"\b(which files?|what files?)\b",
    ]
    CREATE_PATTERNS = [
        r"\b(create|write|generate|make|build|implement)\s+(a |an |new |the |\w+ )*(file|script|module|function|class)\b",
        r"\b(create|write)\s+.+?\.(py|js|ts|go|rs|java)\b",
    ]
    # Patterns for simple standalone code generation (no codebase exploration needed)
    # These patterns match standalone code generation requests without file path context
    CREATE_SIMPLE_PATTERNS = [
        # "create a simple Python function", "write a simple script"
        r"\b(create|write|generate|make|implement)\s+(a |an )?simple\s+\w*\s*(function|class|script)\b",
        # "create a function that calculates", "write a class that handles"
        r"\b(create|write|generate)\s+(a |an )?(\w+\s+)?(function|class)\s+(that|which|to)\b",
        # "write me a function", "generate me some code"
        r"\b(write|generate)\s+me\s+(a |an )?(\w+\s+)?(function|class|code)\b",
        # "create factorial function", "make fibonacci class"
        r"^(create|write|generate|make)\s+(a |an )?(\w+\s+)?(function|class)\s+\w+",
        # "just write/create" patterns - explicit simplicity request
        r"\bjust\s+(create|write|generate|make)\s+(a |an )?",
    ]
    ANALYZE_PATTERNS = [
        r"\b(analyze|count|measure|calculate|how many|statistics)\b",
        r"\b(total lines?|number of)\b",
    ]
    DESIGN_PATTERNS = [
        r"\b(design|plan|outline|architecture|explain|describe|what is|how does)\b",
        r"\b(without (reading|looking|checking))\b",
        r"\b(just list|just outline|conceptually)\b",
    ]

    # Patterns for extracting file paths
    FILE_PATH_PATTERN = r"(?:^|[\s,])([a-zA-Z0-9_/.-]+(?:\.py|\.js|\.ts|\.go|\.rs|\.java|\.yaml|\.yml|\.json|\.md))\b"

    # Patterns for extracting class/function names
    ENTITY_PATTERN = r"\b(class|function|method)\s+([A-Z][a-zA-Z0-9_]*|\w+)"
    CLASS_NAME_PATTERN = r"\b([A-Z][a-zA-Z0-9_]+)\s+class\b|\bthe\s+([A-Z][a-zA-Z0-9_]+)\b"

    def __init__(self, use_semantic_classification: bool = True):
        """Initialize the progress tracker.

        Args:
            use_semantic_classification: If True, use embedding-based task type
                classification instead of regex patterns. More robust but requires
                initialization time for embeddings. Defaults to True.
        """
        self.progress = TaskProgress()
        self.iteration_count = 0
        self._raw_prompt = ""
        self._use_semantic = use_semantic_classification
        self._task_classifier = None

        # Model-specific exploration settings (can be overridden by orchestrator)
        self._exploration_multiplier: float = 1.0
        self._continuation_patience: int = 3

        # Lazy initialization of classifier (use singleton to avoid duplicate loading)
        if self._use_semantic:
            try:
                from victor.embeddings.task_classifier import TaskTypeClassifier

                self._task_classifier = TaskTypeClassifier.get_instance()
            except ImportError:
                logger.warning("TaskTypeClassifier not available, falling back to regex patterns")
                self._use_semantic = False

    def set_model_exploration_settings(
        self, exploration_multiplier: float = 1.0, continuation_patience: int = 3
    ) -> None:
        """Set model-specific exploration settings.

        Called by the orchestrator after loading model capabilities.

        Args:
            exploration_multiplier: Multiplier for max_exploration_iterations
            continuation_patience: Number of empty turns allowed before forcing
        """
        self._exploration_multiplier = exploration_multiplier
        self._continuation_patience = continuation_patience
        logger.debug(
            f"Model exploration settings: multiplier={exploration_multiplier}, "
            f"patience={continuation_patience}"
        )

    def reset(self) -> None:
        """Reset the tracker for a new conversation."""
        self.progress = TaskProgress()
        self.iteration_count = 0
        self._raw_prompt = ""

    def detect_task_type(self, prompt: str) -> TaskType:
        """Detect the task type from a user prompt.

        Uses semantic embedding-based classification if available,
        falls back to regex patterns otherwise.

        Args:
            prompt: The user's prompt text

        Returns:
            The detected TaskType
        """
        self._raw_prompt = prompt

        # Extract targets first (needed for file context check)
        self._extract_targets(prompt)

        # Try semantic classification first (more robust)
        if self._use_semantic and self._task_classifier:
            try:
                result = self._task_classifier.classify_sync(prompt)
                if result.confidence >= 0.35:  # Minimum confidence threshold
                    # TaskType is now imported from task_classifier - no mapping needed
                    self.progress.task_type = result.task_type
                    task_config = TASK_CONFIGS.get(result.task_type)
                    max_exp = task_config.max_exploration_iterations if task_config else "N/A"
                    logger.info(
                        f"Task type detected: {result.task_type.value} "
                        f"(semantic, confidence={result.confidence:.2f}, max_exploration={max_exp})"
                    )
                    return result.task_type
                else:
                    logger.info(
                        f"Semantic classification: {result.task_type.value} "
                        f"(confidence={result.confidence:.2f} < 0.35 threshold), falling back to regex"
                    )
            except Exception as e:
                logger.warning(f"Semantic classification failed, using regex fallback: {e}")

        # Fall back to regex-based detection
        return self._detect_task_type_regex(prompt)

    def _detect_task_type_regex(self, prompt: str) -> TaskType:
        """Detect task type using regex patterns (fallback method).

        Args:
            prompt: The user's prompt text

        Returns:
            The detected TaskType
        """
        prompt_lower = prompt.lower()

        # Check patterns in priority order
        # DESIGN first (if they say "without reading" or "just outline")
        for pattern in self.DESIGN_PATTERNS:
            if re.search(pattern, prompt_lower):
                self.progress.task_type = TaskType.DESIGN
                logger.debug(f"Detected task type: DESIGN from pattern: {pattern}")
                return TaskType.DESIGN

        # EDIT patterns
        for pattern in self.EDIT_PATTERNS:
            if re.search(pattern, prompt_lower):
                self.progress.task_type = TaskType.EDIT
                logger.debug(f"Detected task type: EDIT from pattern: {pattern}")
                return TaskType.EDIT

        # CREATE_SIMPLE patterns (check first - simple code gen without context)
        # Only if no file path is specified in the prompt
        has_file_context = len(self.progress.target_files) > 0

        if not has_file_context:
            for pattern in self.CREATE_SIMPLE_PATTERNS:
                if re.search(pattern, prompt_lower):
                    self.progress.task_type = TaskType.CREATE_SIMPLE
                    logger.debug(f"Detected task type: CREATE_SIMPLE from pattern: {pattern}")
                    return TaskType.CREATE_SIMPLE

        # CREATE patterns (check before SEARCH since "create" is more specific)
        for pattern in self.CREATE_PATTERNS:
            if re.search(pattern, prompt_lower):
                self.progress.task_type = TaskType.CREATE
                logger.debug(f"Detected task type: CREATE from pattern: {pattern}")
                return TaskType.CREATE

        # SEARCH patterns
        for pattern in self.SEARCH_PATTERNS:
            if re.search(pattern, prompt_lower):
                self.progress.task_type = TaskType.SEARCH
                logger.debug(f"Detected task type: SEARCH from pattern: {pattern}")
                return TaskType.SEARCH

        # ANALYZE patterns
        for pattern in self.ANALYZE_PATTERNS:
            if re.search(pattern, prompt_lower):
                self.progress.task_type = TaskType.ANALYZE
                logger.debug(f"Detected task type: ANALYZE from pattern: {pattern}")
                return TaskType.ANALYZE

        # Default to GENERAL
        self.progress.task_type = TaskType.GENERAL
        logger.debug("Detected task type: GENERAL (no specific pattern matched)")
        return TaskType.GENERAL

    def _extract_targets(self, prompt: str) -> None:
        """Extract target files and entities from the prompt.

        Args:
            prompt: The user's prompt text
        """
        # Extract file paths
        file_matches = re.findall(self.FILE_PATH_PATTERN, prompt)
        self.progress.target_files = set(file_matches)

        # Extract class/entity names
        entity_matches = re.findall(self.ENTITY_PATTERN, prompt)
        for match in entity_matches:
            if isinstance(match, tuple):
                self.progress.target_entities.add(match[1])
            else:
                self.progress.target_entities.add(match)

        # Also look for PascalCase names that might be classes
        class_matches = re.findall(self.CLASS_NAME_PATTERN, prompt)
        for match in class_matches:
            if isinstance(match, tuple):
                for name in match:
                    if name:
                        self.progress.target_entities.add(name)
            elif match:
                self.progress.target_entities.add(match)

        logger.debug(
            f"Extracted targets - files: {self.progress.target_files}, "
            f"entities: {self.progress.target_entities}"
        )

    def update_from_tool_call(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Update progress based on a tool call.

        Args:
            tool_name: Name of the tool that was called
            args: Arguments passed to the tool
            result: Result returned by the tool
        """
        # Check if the result indicates success
        is_success = self._is_successful_result(result)

        if not is_success:
            logger.debug(f"Tool {tool_name} did not succeed, skipping milestone update")
            return

        # Track files read
        if tool_name == "read_file":
            file_path = args.get("path", "")
            self.progress.files_read.add(file_path)

            # Check if this was a target file
            if self._is_target_file(file_path):
                self.progress.milestones.add(Milestone.TARGET_READ)
                logger.info(f"Milestone achieved: TARGET_READ ({file_path})")

        # Track files modified
        elif tool_name in ("edit_files", "write_file"):
            file_path = args.get("file_path") or args.get("path", "")
            self.progress.files_modified.add(file_path)
            self.progress.milestones.add(Milestone.CHANGE_MADE)
            logger.info(f"Milestone achieved: CHANGE_MADE ({file_path})")

        # Track search completion
        elif tool_name in ("code_search", "semantic_code_search"):
            results = result.get("results", []) if isinstance(result, dict) else []
            if results:
                self.progress.milestones.add(Milestone.SEARCH_COMPLETE)
                logger.info("Milestone achieved: SEARCH_COMPLETE")

        # Track test execution as verification
        elif tool_name in ("run_tests", "execute_bash"):
            if "test" in str(args).lower():
                self.progress.milestones.add(Milestone.CHANGE_VERIFIED)
                logger.info("Milestone achieved: CHANGE_VERIFIED")

    def _is_successful_result(self, result: Any) -> bool:
        """Check if a tool result indicates success.

        Args:
            result: The result from a tool call

        Returns:
            True if the result indicates success
        """
        if isinstance(result, dict):
            return bool(result.get("success", True))
        return True

    def _is_target_file(self, file_path: str) -> bool:
        """Check if a file path matches any target file.

        Args:
            file_path: The file path to check

        Returns:
            True if the file is a target file
        """
        # Direct match
        if file_path in self.progress.target_files:
            return True

        # Partial match (filename without directory)
        for target in self.progress.target_files:
            if file_path.endswith(target) or target.endswith(file_path):
                return True

        # Check if file likely contains target entity
        for entity in self.progress.target_entities:
            if entity.lower() in file_path.lower():
                return True

        # For EDIT tasks, consider the first file read as the target if no targets specified
        if (
            self.progress.task_type == TaskType.EDIT
            and not self.progress.target_files
            and not self.progress.files_read
        ):
            return True

        return False

    def increment_iteration(self) -> None:
        """Increment the productive iteration counter (called on successful tool call)."""
        self.iteration_count += 1
        self.progress.iteration_count = self.iteration_count
        # Also increment total turns when a tool call happens
        self.progress.total_turns += 1

    def increment_turn(self) -> None:
        """Increment total turns without incrementing productive iterations.

        Called when a turn happens without a tool call (e.g., continuation prompts).
        This helps track models that need more "thinking" turns.
        """
        self.progress.total_turns += 1
        logger.debug(
            f"Turn without tool call: total_turns={self.progress.total_turns}, "
            f"productive_iterations={self.iteration_count}"
        )

    def should_force_action(self) -> Tuple[bool, Optional[str]]:
        """Determine if the LLM should be forced to take action.

        Returns:
            A tuple of (should_force, hint_message)
        """
        config = self.get_task_config()

        # Don't force action for design tasks
        if not config.needs_tools:
            return False, None

        # Don't force action before we've done some exploration
        if self.iteration_count < 2:
            return False, None

        # Calculate effective iteration count that accounts for model "thinking" overhead
        # Models that need more turns per tool call get a higher effective limit
        total_turns = self.progress.total_turns
        productive = self.iteration_count

        # Start with base max, apply model-specific multiplier
        base_max = config.max_exploration_iterations
        model_adjusted_max = int(base_max * self._exploration_multiplier)

        # Productivity ratio: what fraction of turns resulted in tool calls?
        # Lower ratio = model needs more thinking, we should be more patient
        if total_turns > 0 and productive > 0:
            productivity_ratio = productive / total_turns
            # If model only makes tool calls 50% of turns, give it additional buffer
            # Clamp multiplier between 1.0 and 2.0 to avoid extreme values
            productivity_multiplier = (
                min(2.0, max(1.0, 1.0 / productivity_ratio)) if productivity_ratio > 0 else 1.5
            )
            effective_max = int(model_adjusted_max * productivity_multiplier)
            logger.debug(
                f"Exploration limits: base={base_max}, model_adjusted={model_adjusted_max}, "
                f"effective={effective_max} (productivity={productivity_ratio:.2f}, "
                f"multiplier={self._exploration_multiplier})"
            )
        else:
            effective_max = model_adjusted_max

        # Force action for EDIT tasks after target is read
        if (
            config.force_action_after_target_read
            and Milestone.TARGET_READ in self.progress.milestones
            and Milestone.CHANGE_MADE not in self.progress.milestones
            and self.iteration_count >= effective_max
        ):
            hint = (
                "You have gathered enough information about the target file. "
                "Now use edit_files to make the requested change."
            )
            logger.info(f"Forcing action for EDIT task after {self.iteration_count} iterations")
            return True, hint

        # Force completion if we've exceeded exploration limit (using effective max)
        if self.iteration_count > effective_max:
            if self.progress.task_type == TaskType.SEARCH:
                hint = (
                    "You have done extensive exploration. "
                    "Please summarize your findings and provide the answer."
                )
            else:
                hint = (
                    "You have gathered sufficient context. "
                    "Please complete the task or explain what's blocking you."
                )
            logger.info(
                f"Forcing completion after {self.iteration_count} iterations "
                f"(task_type={self.progress.task_type.value}, "
                f"base_max={config.max_exploration_iterations}, effective_max={effective_max}, "
                f"total_turns={total_turns}, "
                f"milestones={[m.value for m in self.progress.milestones]}, "
                f"files_read={len(self.progress.files_read)})"
            )
            return True, hint

        return False, None

    def is_goal_satisfied(self) -> bool:
        """Check if the task goal has been satisfied.

        Returns:
            True if the goal is satisfied based on milestones
        """
        config = self.get_task_config()

        # Design tasks are always satisfied (no tools needed)
        if not config.needs_tools:
            return True

        # Check if completion milestones are met
        if config.completion_milestones:
            return bool(self.progress.milestones & config.completion_milestones)

        return False

    def get_task_config(self) -> TaskConfig:
        """Get the configuration for the current task type.

        Returns:
            TaskConfig for the current task type
        """
        return TASK_CONFIGS.get(self.progress.task_type, TASK_CONFIGS[TaskType.GENERAL])

    def get_required_tools(self) -> Set[str]:
        """Get the required tools for the current task type.

        Returns:
            Set of tool names required for this task
        """
        config = self.get_task_config()
        return config.required_tools

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of current progress.

        Returns:
            Dictionary with progress information
        """
        config = self.get_task_config()
        total_turns = self.progress.total_turns
        productive = self.iteration_count
        productivity_ratio = productive / total_turns if total_turns > 0 else 1.0

        return {
            "task_type": self.progress.task_type.value,
            "milestones": [m.value for m in self.progress.milestones],
            "target_files": list(self.progress.target_files),
            "target_entities": list(self.progress.target_entities),
            "files_read": list(self.progress.files_read),
            "files_modified": list(self.progress.files_modified),
            "iteration_count": self.iteration_count,
            "total_turns": total_turns,
            "productivity_ratio": round(productivity_ratio, 2),
            "max_exploration": config.max_exploration_iterations,
            "goal_satisfied": self.is_goal_satisfied(),
        }
