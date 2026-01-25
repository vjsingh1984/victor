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

"""Agent modes for different operational contexts.

Inspired by OpenCode's agent modes, this module provides Build, Plan, and
Explore modes that modify agent behavior for different tasks.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from victor.agent.protocols import ModeControllerProtocol

logger = logging.getLogger(__name__)


class AgentMode(str, Enum):
    """Available agent modes.

    This is the CANONICAL AgentMode enum with 3 base modes.

    Inherits from str to allow direct string comparison (e.g., AgentMode.BUILD == "build").

    Semantic Variants (different modes for different purposes):
    - AgentMode (here): Base 3 modes (BUILD, PLAN, EXPLORE) for core agent control
    - RLAgentMode: Extended 5 modes for RL state machine (includes REVIEW, COMPLETE)
    - AdaptiveAgentMode: Extended 5 modes for adaptive control
    """

    BUILD = "build"
    PLAN = "plan"
    EXPLORE = "explore"


@dataclass
class OperationalModeConfig:
    """Rich configuration for operational agent modes (BUILD/PLAN/EXPLORE).

    This provides detailed operational control including:
    - Tool access control (allowed/disallowed tools)
    - System prompt modifications
    - Behavioral settings (confirmations, planning output)
    - Sandbox restrictions

    Renamed from ModeConfig to be semantically distinct:
    - ModeConfig (victor.core.mode_config): Simple tool budget/iteration config
    - OperationalModeConfig: Rich operational mode configuration
    """

    name: str
    description: str
    # Tools that are allowed in this mode
    allowed_tools: Set[str] = field(default_factory=set)
    # Tools that are disallowed in this mode (takes precedence)
    disallowed_tools: Set[str] = field(default_factory=set)
    # Whether all tools are allowed by default
    allow_all_tools: bool = False
    # System prompt additions for this mode
    system_prompt_addition: str = ""
    # Whether to require confirmation for file writes
    require_write_confirmation: bool = False
    # Whether to show detailed planning output
    verbose_planning: bool = False
    # Maximum files to modify in a single operation
    max_files_per_operation: int = 0  # 0 = no limit
    # Tool priority adjustments (tool_name -> priority_boost)
    tool_priorities: Dict[str, float] = field(default_factory=dict)
    # Exploration multiplier - increases exploration limits in explore/plan modes
    exploration_multiplier: float = 1.0
    # Sandbox directory for limited edits (relative to project root)
    # If set, edits are only allowed within this directory
    sandbox_dir: Optional[str] = None
    # Whether edits are allowed in sandbox even if disallowed_tools has edit
    allow_sandbox_edits: bool = False


# Default mode configurations
MODE_CONFIGS: Dict[AgentMode, OperationalModeConfig] = {
    AgentMode.BUILD: OperationalModeConfig(
        name="Build",
        description="Implementation mode for creating and modifying code",
        allow_all_tools=True,
        disallowed_tools=set(),  # All tools available
        system_prompt_addition="""
You are in BUILD mode - focused on TAKING ACTION and implementing changes.

ACTION-FIRST PRINCIPLE:
- STOP excessive reading/exploration. You have enough context.
- When asked to edit/create/modify: USE the edit or write tool NOW
- Do NOT read the same file multiple times before editing
- After reading a file ONCE, immediately proceed to editing

IMPLEMENTATION WORKFLOW:
1. Read the target file ONCE to understand current state
2. IMMEDIATELY use edit_files() or write_file() to make changes
3. Run tests if applicable
4. Commit your changes

ANTI-PATTERNS TO AVOID:
- Reading a file 3+ times without editing (you already have the content)
- Saying "Let me read..." when you've already read the file
- Planning to edit without actually calling the edit tool
- Exploration loops without taking action

When the user asks you to edit a file, your NEXT tool call should be edit_files() or write_file().
""",
        require_write_confirmation=False,
        verbose_planning=False,
        max_files_per_operation=0,  # No limit
        tool_priorities={
            "edit": 1.5,  # Boost edit tool priority in BUILD mode (canonical name)
            "edit_files": 1.5,  # Alias for edit
            "write_file": 1.5,  # Boost write tool priority
            "bash": 1.2,
            "git_status": 1.0,
            "read_file": 0.9,  # Slightly lower read priority to encourage action
        },
        exploration_multiplier=5.0,  # 5x exploration for reading before writing (was 2x)
    ),
    AgentMode.PLAN: OperationalModeConfig(
        name="Plan",
        description="Planning mode for analysis and strategy before implementation",
        allow_all_tools=False,
        allowed_tools={
            # Read-only and analysis tools
            "read_file",
            "list_directory",
            "code_search",
            "semantic_code_search",
            "glob",
            "grep",
            "git_status",
            "git_diff",
            "git_log",
            "dependency_graph",
            "code_review",
            # Planning-specific
            "plan_files",
            # Sandbox editing (limited to .victor/sandbox/)
            "write_file",
            "edit_files",
        },
        disallowed_tools={
            # No direct bash/shell or git modifications
            "bash",  # Could modify things outside sandbox
            "shell",  # Also shell tool (same as bash)
            "git_commit",
            "git_push",
        },
        system_prompt_addition="""
You are in PLAN mode - focused on analysis and planning before implementation.

SANDBOX EDITING: You can create/edit files ONLY in the `.victor/sandbox/` directory.
This is useful for:
- Drafting code snippets before implementation
- Creating plan documents
- Testing small code samples

RESTRICTIONS:
- DO NOT modify files outside `.victor/sandbox/`
- All edits to main codebase will be blocked
- Use /mode build to switch to full implementation when ready

WORKFLOW:
1. Explore the codebase thoroughly to understand the structure
2. Identify potential issues and edge cases
3. Create a clear, step-by-step implementation plan
4. Save your plan with /plan save
5. Draft code in .victor/sandbox/ if helpful
6. Use /mode build when ready to implement
""",
        require_write_confirmation=True,
        verbose_planning=True,
        max_files_per_operation=5,  # Limited edits in sandbox
        tool_priorities={
            "code_search": 1.3,
            "semantic_code_search": 1.3,
            "read_file": 1.2,
            "dependency_graph": 1.2,
            "plan_files": 1.5,
        },
        exploration_multiplier=10.0,  # 10x exploration in plan mode (effectively unlimited like Claude Code)
        sandbox_dir=".victor/sandbox",
        allow_sandbox_edits=True,
    ),
    AgentMode.EXPLORE: OperationalModeConfig(
        name="Explore",
        description="Exploration mode for understanding code without modifications",
        allow_all_tools=False,
        allowed_tools={
            # Read-only tools only
            "read_file",
            "list_directory",
            "code_search",
            "semantic_code_search",
            "glob",
            "grep",
            "git_status",
            "git_diff",
            "git_log",
            "git_show",
            "dependency_graph",
            "code_review",
            "web_search",
            "web_fetch",
            # Sandbox notes (limited to .victor/sandbox/)
            "write_file",
        },
        disallowed_tools={
            "edit_files",  # No editing main codebase
            "bash",
            "shell",  # Also shell tool (same as bash)
            "git_commit",
            "git_push",
            "git_checkout",
        },
        system_prompt_addition="""
You are in EXPLORE mode - focused on understanding and navigating code.

NOTES: You can create notes ONLY in the `.victor/sandbox/` directory.
This is useful for:
- Saving findings and observations
- Creating documentation drafts

RESTRICTIONS:
- DO NOT modify files outside `.victor/sandbox/`
- Use /mode plan for structured planning
- Use /mode build for implementation

WORKFLOW:
1. Answer questions about the codebase clearly
2. Navigate through code to trace functionality
3. Explain architecture, patterns, and design decisions
4. Save notes in .victor/sandbox/ if helpful
""",
        require_write_confirmation=True,
        verbose_planning=False,
        max_files_per_operation=3,  # Limited notes in sandbox
        tool_priorities={
            "read_file": 1.3,
            "code_search": 1.2,
            "semantic_code_search": 1.2,
            "list_directory": 1.1,
        },
        exploration_multiplier=20.0,  # 20x exploration in explore mode (effectively unlimited like Claude Code)
        sandbox_dir=".victor/sandbox",
        allow_sandbox_edits=True,
    ),
}


class AgentModeController:
    """Controls agent mode state and transitions.

    Note: Previously named `ModeManager`. Alias kept for backward compatibility.
    """

    def __init__(self, initial_mode: AgentMode = AgentMode.BUILD):
        """Initialize the mode controller.

        Args:
            initial_mode: Starting mode (defaults to BUILD)
        """
        self._current_mode = initial_mode
        self._mode_history: List[AgentMode] = [initial_mode]
        self._callbacks: List[Callable[[AgentMode, AgentMode], None]] = []
        logger.info(f"AgentModeController initialized in {initial_mode.value} mode")

    @property
    def current_mode(self) -> AgentMode:
        """Get the current mode."""
        return self._current_mode

    @property
    def config(self) -> OperationalModeConfig:
        """Get the current mode configuration."""
        return MODE_CONFIGS[self._current_mode]

    def switch_mode(self, new_mode: AgentMode) -> bool:
        """Switch to a new mode.

        Args:
            new_mode: The mode to switch to

        Returns:
            True if switch was successful
        """
        if new_mode == self._current_mode:
            logger.debug(f"Already in {new_mode.value} mode")
            return True

        old_mode = self._current_mode
        self._current_mode = new_mode
        self._mode_history.append(new_mode)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(old_mode, new_mode)
            except Exception as e:
                logger.error(f"Mode change callback failed: {e}")

        # Emit RL event for mode transition
        self._emit_mode_transition_event(old_mode, new_mode)

        logger.info(f"Switched from {old_mode.value} to {new_mode.value} mode")
        return True

    def previous_mode(self) -> Optional[AgentMode]:
        """Switch to the previous mode.

        Returns:
            The mode switched to, or None if no history
        """
        if len(self._mode_history) < 2:
            return None

        # Remove current from history
        self._mode_history.pop()
        prev = self._mode_history[-1]
        self._current_mode = prev
        return prev

    def register_callback(self, callback: Callable[[AgentMode, AgentMode], None]) -> None:
        """Register a callback for mode changes.

        Args:
            callback: Function called with (old_mode, new_mode)
        """
        self._callbacks.append(callback)

    def _emit_mode_transition_event(self, old_mode: AgentMode, new_mode: AgentMode) -> None:
        """Emit RL event for mode transition.

        This activates the mode_transition learner to learn optimal
        mode transitions based on task context.

        Args:
            old_mode: Mode transitioning from
            new_mode: Mode transitioning to
        """
        try:
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is not None:
                event = RLEvent(
                    type=RLEventType.MODE_TRANSITION,
                    mode_from=old_mode.value,
                    mode_to=new_mode.value,
                    success=True,  # Transition always succeeds
                    quality_score=0.5,  # Will be updated by outcome recording
                    metadata={
                        "history_length": len(self._mode_history),
                    },
                )

                hooks.emit(event)

        except Exception as e:
            logger.debug(f"Mode transition event emission failed: {e}")

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in the current mode.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is allowed
        """
        config = self.config

        # Check disallowed list first (takes precedence)
        if tool_name in config.disallowed_tools:
            return False

        # If allow_all_tools, it's allowed unless disallowed
        if config.allow_all_tools:
            return True

        # Otherwise, check allowed list
        return tool_name in config.allowed_tools

    def get_tool_priority(self, tool_name: str) -> float:
        """Get priority adjustment for a tool in current mode.

        Args:
            tool_name: Name of the tool

        Returns:
            Priority multiplier (1.0 = no adjustment)
        """
        return self.config.tool_priorities.get(tool_name, 1.0)

    def get_exploration_multiplier(self) -> float:
        """Get exploration limit multiplier for current mode.

        Higher multipliers allow more exploration iterations.
        - BUILD: 5.0x (reading before writing)
        - PLAN: 10.0x (thorough analysis)
        - EXPLORE: 20.0x (exploration is primary goal)

        Returns:
            Multiplier value
        """
        return self.config.exploration_multiplier

    def get_system_prompt_addition(self) -> str:
        """Get additional system prompt text for current mode.

        Returns:
            Additional prompt text
        """
        return self.config.system_prompt_addition

    def get_status(self) -> Dict[str, Any]:
        """Get current mode status.

        Returns:
            Dictionary with mode information
        """
        config = self.config
        return {
            "mode": self._current_mode.value,
            "name": config.name,
            "description": config.description,
            "write_confirmation_required": config.require_write_confirmation,
            "verbose_planning": config.verbose_planning,
        }

    def get_mode_list(self) -> List[Dict[str, Any]]:
        """Get list of available modes.

        Returns:
            List of mode info dictionaries
        """
        return [
            {
                "mode": mode.value,
                "name": MODE_CONFIGS[mode].name,
                "description": MODE_CONFIGS[mode].description,
                "current": str(mode == self._current_mode),
            }
            for mode in AgentMode
        ]


# Global instance (legacy - prefer DI container)
_mode_controller: Optional[AgentModeController] = None


def get_mode_controller() -> AgentModeController:
    """Get or create the mode controller.

    Resolution order:
    1. Check DI container (preferred)
    2. Fall back to module-level singleton (legacy)

    Returns:
        AgentModeController instance
    """
    global _mode_controller

    # Try DI container first
    try:
        from victor.core.container import get_container

        container = get_container()
        # Try to get AgentModeController from container
        try:
            controller = container.get(AgentModeController)
            return controller
        except Exception:
            pass  # Not registered
    except Exception:
        pass  # Fall back to legacy singleton

    # Legacy fallback
    if _mode_controller is None:
        _mode_controller = AgentModeController()
    return _mode_controller


def set_mode_controller(controller: AgentModeController) -> None:
    """Set the global mode controller instance.

    Note: This sets the legacy module-level singleton. If using DI container,
    use container.register_or_replace() instead.
    """
    global _mode_controller
    _mode_controller = controller


def reset_mode_controller() -> None:
    """Reset the global mode controller (for testing).

    Note: This only resets the legacy module-level singleton. If using DI
    container, use reset_container() as well.
    """
    global _mode_controller
    _mode_controller = None
