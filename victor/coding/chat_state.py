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

"""Coding-specific chat state for Phase 3 domain logic extraction.

This module provides the CodingChatState class which extends the domain-agnostic
MutableChatState with coding-specific state management.

Phase 3: Extract Domain Logic from ChatCoordinator/AgentOrchestrator
===================================================================
Moves coding-specific state from orchestrator into dedicated state class:
- Required files tracking for task completion
- Required outputs tracking for task completion
- Read files session for progress tracking
- File read nudge notification state

Architecture:
    - MutableChatState (domain-agnostic, base)
    - CodingChatState (domain-specific, extends MutableChatState)
    - State protocol abstraction (framework layer)

Usage:
    state = CodingChatState()
    state.set_required_files(["main.py", "utils.py"])
    state.track_file_read("main.py")
    state.check_all_files_read()  # Returns True when all files read
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor.framework.protocols import MutableChatState

logger = logging.getLogger(__name__)


class CodingChatState(MutableChatState):
    """Coding-specific chat state with file tracking.

    This class extends MutableChatState with coding-specific state management:
    - Required files tracking: Files that must be read for task completion
    - Required outputs tracking: Expected outputs from the task
    - Read files session: Track which required files have been read
    - Nudge notification: Track if "all files read" nudge has been sent

    The state is designed to be used by chat workflows that need to track
    task completion progress without tight coupling to the orchestrator.

    Example:
        state = CodingChatState()
        state.set_required_files(["main.py", "utils.py"])
        state.track_file_read("main.py")
        state.check_all_files_read()  # Returns False
        state.track_file_read("utils.py")
        state.check_all_files_read()  # Returns True
    """

    def __init__(self) -> None:
        """Initialize the coding chat state."""
        super().__init__()
        # Coding-specific state
        self._required_files: List[str] = []
        self._required_outputs: List[str] = []
        self._read_files_session: set[str] = set()
        self._all_files_read_nudge_sent: bool = False

    # ========================================================================
    # Required Files
    # ========================================================================

    @property
    def required_files(self) -> List[str]:
        """Get the list of required files for task completion.

        Returns:
            List of file paths that must be read
        """
        return self._required_files.copy()

    def set_required_files(self, files: List[str]) -> None:
        """Set the list of required files for task completion.

        Args:
            files: List of file paths that must be read
        """
        self._required_files = list(files)
        # Update metadata for workflow access
        self.set_metadata("required_files", files)

    def add_required_file(self, file_path: str) -> None:
        """Add a single file to the required files list.

        Args:
            file_path: Path to the file
        """
        if file_path not in self._required_files:
            self._required_files.append(file_path)
            self.set_metadata("required_files", self._required_files)

    # ========================================================================
    # Required Outputs
    # ========================================================================

    @property
    def required_outputs(self) -> List[str]:
        """Get the list of required outputs for task completion.

        Returns:
            List of expected outputs
        """
        return self._required_outputs.copy()

    def set_required_outputs(self, outputs: List[str]) -> None:
        """Set the list of required outputs for task completion.

        Args:
            outputs: List of expected outputs
        """
        self._required_outputs = list(outputs)
        # Update metadata for workflow access
        self.set_metadata("required_outputs", outputs)

    def add_required_output(self, output: str) -> None:
        """Add a single output to the required outputs list.

        Args:
            output: Expected output description
        """
        if output not in self._required_outputs:
            self._required_outputs.append(output)
            self.set_metadata("required_outputs", self._required_outputs)

    # ========================================================================
    # Read Files Session
    # ========================================================================

    def track_file_read(self, file_path: str) -> None:
        """Track that a file has been read.

        Args:
            file_path: Path to the file that was read
        """
        self._read_files_session.add(file_path)
        logger.debug(f"Tracked file read: {file_path}")

    def has_read_file(self, file_path: str) -> bool:
        """Check if a file has been read.

        Args:
            file_path: Path to check

        Returns:
            True if file has been read
        """
        return file_path in self._read_files_session

    def get_read_files(self) -> List[str]:
        """Get list of files that have been read.

        Returns:
            List of file paths
        """
        return sorted(self._read_files_session)

    def clear_read_files_session(self) -> None:
        """Clear the read files session (e.g., for new task)."""
        self._read_files_session.clear()
        self._all_files_read_nudge_sent = False
        logger.debug("Cleared read files session")

    def check_all_files_read(self) -> bool:
        """Check if all required files have been read.

        Returns:
            True if all required files have been read
        """
        if not self._required_files:
            return False
        return self._read_files_session.issuperset(set(self._required_files))

    def get_unread_files(self) -> List[str]:
        """Get list of required files that haven't been read yet.

        Returns:
            List of unread file paths
        """
        return [f for f in self._required_files if f not in self._read_files_session]

    # ========================================================================
    # Nudge Notification State
    # ========================================================================

    @property
    def all_files_read_nudge_sent(self) -> bool:
        """Check if the "all files read" nudge has been sent.

        Returns:
            True if nudge has been sent
        """
        return self._all_files_read_nudge_sent

    def mark_nudge_sent(self) -> None:
        """Mark that the nudge notification has been sent."""
        self._all_files_read_nudge_sent = True
        logger.info("All files read nudge marked as sent")

    def reset_nudge_state(self) -> None:
        """Reset the nudge notification state."""
        self._all_files_read_nudge_sent = False

    # ========================================================================
    # Workflow Integration
    # ========================================================================

    def extract_requirements_from_message(self, user_message: str) -> None:
        """Extract and set requirements from user message.

        This method analyzes the user message to extract:
        - Required files (file paths mentioned)
        - Required outputs (expected results)

        Args:
            user_message: User's input message
        """
        # Use the prompt requirement extractor
        from victor.agent.prompt_requirement_extractor import extract_prompt_requirements

        # Extract requirements (this populates counts and patterns)
        requirements = extract_prompt_requirements(user_message)

        # Set in state (note: file paths are extracted elsewhere via patterns)
        # The requirement counts are available in the requirements object
        if hasattr(requirements, "file_count"):
            logger.debug(f"Extracted {requirements.file_count} required files")

        if hasattr(requirements, "output_count"):
            logger.debug(f"Extracted {requirements.output_count} required outputs")

        # Update metadata with requirement info
        self.set_metadata("requirements_extracted", {
            "file_count": getattr(requirements, "file_count", 0),
            "output_count": getattr(requirements, "output_count", 0),
        })

    # ========================================================================
    # State Management
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization.

        Returns:
            Dictionary representation of state
        """
        base_dict = super().to_dict()
        base_dict.update({
            "required_files": self._required_files.copy(),
            "required_outputs": self._required_outputs.copy(),
            "read_files_session": sorted(self._read_files_session),
            "all_files_read_nudge_sent": self._all_files_read_nudge_sent,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodingChatState":
        """Create state from dictionary.

        Args:
            data: Dictionary representation of state

        Returns:
            New CodingChatState instance
        """
        state = cls()
        state._required_files = data.get("required_files", [])
        state._required_outputs = data.get("required_outputs", [])
        state._read_files_session = set(data.get("read_files_session", []))
        state._all_files_read_nudge_sent = data.get("all_files_read_nudge_sent", False)

        # Restore base state
        state._messages = data.get("messages", [])
        state._iteration_count = data.get("iteration_count", 0)
        state._metadata = data.get("metadata", {})

        return state

    def __repr__(self) -> str:
        """Return string representation of state."""
        return (
            f"CodingChatState("
            f"files={len(self._messages)}, "
            f"iteration={self._iteration_count}, "
            f"required_files={len(self._required_files)}, "
            f"read_files={len(self._read_files_session)}, "
            f"metadata_keys={list(self._metadata.keys())}"
            f")"
        )


__all__ = [
    "CodingChatState",
]
