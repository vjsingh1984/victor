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

"""File Context Coordinator for tracking files and outputs.

This coordinator manages:
- Required files extraction from prompts
- Required outputs extraction from prompts
- Observed files tracking
- File reading session tracking
- Nudge management for file completion

Design Pattern: Coordinator Pattern
- Centralizes file context tracking logic
- Delegates file/outputs extraction to TaskAnalyzer
- Provides clean API for orchestrator

Phase 6 Refactoring: Extracted from AgentOrchestrator
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Set

if TYPE_CHECKING:
    from victor.agent.task_analyzer import TaskAnalyzer

logger = logging.getLogger(__name__)


class FileContextCoordinator:
    """Coordinator for file context tracking and management.

    Manages required files, required outputs, observed files, and
    file reading session state for task completion tracking.

    Attributes:
        _task_analyzer: TaskAnalyzer for extracting files/outputs from prompts
        _required_files: List of required file paths
        _required_outputs: List of required output descriptions
        _observed_files: Set of files that have been observed
        _read_files_session: Set of files read in current session
        _all_files_read_nudge_sent: Whether nudge has been sent

    Example:
        coordinator = FileContextCoordinator(task_analyzer)
        coordinator.extract_requirements("Read main.py and utils.py")
        files = coordinator.get_required_files()
        coordinator.mark_observed("main.py")
    """

    def __init__(
        self,
        task_analyzer: "TaskAnalyzer",
    ):
        """Initialize FileContextCoordinator.

        Args:
            task_analyzer: TaskAnalyzer for extracting requirements from prompts
        """
        self._task_analyzer = task_analyzer
        self._required_files: List[str] = []
        self._required_outputs: List[str] = []
        self._observed_files: Set[str] = set()
        self._read_files_session: Set[str] = set()
        self._all_files_read_nudge_sent: bool = False

    # ========================================================================
    # Requirement Extraction
    # ========================================================================

    def extract_requirements(self, user_message: str) -> None:
        """Extract required files and outputs from user prompt.

        Updates internal state with extracted requirements.

        Args:
            user_message: User's prompt text
        """
        self._required_files = self._extract_required_files_from_prompt(user_message)
        self._required_outputs = self._extract_required_outputs_from_prompt(user_message)

        logger.debug(
            f"Extracted {len(self._required_files)} required files, "
            f"{len(self._required_outputs)} required outputs"
        )

    def _extract_required_files_from_prompt(self, user_message: str) -> List[str]:
        """Extract file paths mentioned in user prompt for task completion tracking.

        Delegates to TaskAnalyzer.extract_required_files_from_prompt().

        Args:
            user_message: The user's prompt text

        Returns:
            List of file paths mentioned in the prompt
        """
        return self._task_analyzer.extract_required_files_from_prompt(user_message)

    def _extract_required_outputs_from_prompt(self, user_message: str) -> List[str]:
        """Extract output requirements from user prompt.

        Delegates to TaskAnalyzer.extract_required_outputs_from_prompt().

        Args:
            user_message: The user's prompt text

        Returns:
            List of required output types (e.g., ["findings table", "top-3 fixes"])
        """
        return self._task_analyzer.extract_required_outputs_from_prompt(user_message)

    # ========================================================================
    # File Tracking
    # ========================================================================

    def mark_observed(self, file_path: str) -> None:
        """Mark a file as observed.

        Args:
            file_path: Path to the observed file
        """
        self._observed_files.add(file_path)

    def mark_many_observed(self, file_paths: List[str]) -> None:
        """Mark multiple files as observed.

        Args:
            file_paths: List of file paths
        """
        self._observed_files.update(file_paths)

    def get_observed_files(self) -> Set[str]:
        """Get set of observed files.

        Returns:
            Set of observed file paths
        """
        return self._observed_files.copy()

    def set_observed_files(self, files: Set[str]) -> None:
        """Set observed files (for state restoration).

        Args:
            files: Set of file paths
        """
        self._observed_files = files.copy()

    # ========================================================================
    # Required Files
    # ========================================================================

    def get_required_files(self) -> List[str]:
        """Get list of required files.

        Returns:
            List of required file paths
        """
        return self._required_files.copy()

    def set_required_files(self, files: List[str]) -> None:
        """Set required files (for state restoration).

        Args:
            files: List of required file paths
        """
        self._required_files = files.copy()

    # ========================================================================
    # Required Outputs
    # ========================================================================

    def get_required_outputs(self) -> List[str]:
        """Get list of required outputs.

        Returns:
            List of required output descriptions
        """
        return self._required_outputs.copy()

    def set_required_outputs(self, outputs: List[str]) -> None:
        """Set required outputs (for state restoration).

        Args:
            outputs: List of required output descriptions
        """
        self._required_outputs = outputs.copy()

    # ========================================================================
    # File Reading Session
    # ========================================================================

    def add_to_read_session(self, file_path: str) -> None:
        """Add file to current read session.

        Args:
            file_path: Path to file being read
        """
        self._read_files_session.add(file_path)

    def get_read_session(self) -> Set[str]:
        """Get files read in current session.

        Returns:
            Set of file paths read this session
        """
        return self._read_files_session.copy()

    def set_read_session(self, files: Set[str]) -> None:
        """Set read session files (for state restoration).

        Args:
            files: Set of file paths
        """
        self._read_files_session = files.copy()

    def clear_read_session(self) -> None:
        """Clear current read session."""
        self._read_files_session.clear()

    # ========================================================================
    # Nudge Management
    # ========================================================================

    def get_nudge_sent(self) -> bool:
        """Check if file completion nudge has been sent.

        Returns:
            True if nudge sent, False otherwise
        """
        return self._all_files_read_nudge_sent

    def set_nudge_sent(self, sent: bool) -> None:
        """Set nudge sent status.

        Args:
            sent: Whether nudge has been sent
        """
        self._all_files_read_nudge_sent = sent

    # ========================================================================
    # State Management
    # ========================================================================

    def get_state(self) -> dict[str, Any]:
        """Get coordinator state for checkpointing.

        Returns:
            Dictionary with coordinator state
        """
        return {
            "required_files": self._required_files.copy(),
            "required_outputs": self._required_outputs.copy(),
            "observed_files": list(self._observed_files),
            "read_files_session": list(self._read_files_session),
            "all_files_read_nudge_sent": self._all_files_read_nudge_sent,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore coordinator state from checkpoint.

        Args:
            state: State dictionary from checkpoint
        """
        self._required_files = state.get("required_files", [])
        self._required_outputs = state.get("required_outputs", [])
        self._observed_files = set(state.get("observed_files", []))
        self._read_files_session = set(state.get("read_files_session", []))
        self._all_files_read_nudge_sent = state.get("all_files_read_nudge_sent", False)

    def reset(self) -> None:
        """Reset all coordinator state."""
        self._required_files = []
        self._required_outputs = []
        self._observed_files = set()
        self._read_files_session = set()
        self._all_files_read_nudge_sent = False

    # ========================================================================
    # Computed Properties
    # ========================================================================

    def are_all_required_files_observed(self) -> bool:
        """Check if all required files have been observed.

        Returns:
            True if all required files observed, False otherwise
        """
        if not self._required_files:
            return False
        return all(f in self._observed_files for f in self._required_files)

    def get_missing_files(self) -> List[str]:
        """Get list of required files not yet observed.

        Returns:
            List of unobserved required file paths
        """
        return [f for f in self._required_files if f not in self._observed_files]

    def get_file_observation_progress(self) -> dict[str, Any]:
        """Get progress on file observation.

        Returns:
            Dictionary with progress metrics:
            - total_required: Total number of required files
            - total_observed: Number of required files observed
            - missing_files: List of unobserved files
            - progress_percent: Percentage of required files observed
        """
        missing = self.get_missing_files()
        total = len(self._required_files)
        observed = total - len(missing)

        return {
            "total_required": total,
            "total_observed": observed,
            "missing_files": missing,
            "progress_percent": (observed / total * 100) if total > 0 else 0,
        }


def create_file_context_coordinator(
    task_analyzer: "TaskAnalyzer",
) -> FileContextCoordinator:
    """Factory function to create FileContextCoordinator.

    Args:
        task_analyzer: TaskAnalyzer for extracting requirements

    Returns:
        Configured FileContextCoordinator instance
    """
    return FileContextCoordinator(task_analyzer=task_analyzer)


__all__ = [
    "FileContextCoordinator",
    "create_file_context_coordinator",
]
