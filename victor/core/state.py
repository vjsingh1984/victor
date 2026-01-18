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

"""Core conversation state types.

This module provides the canonical conversation stage enum used across
the entire Victor codebase. It is moved from victor.agent.conversation_state
to victor.core.state to enforce proper layer boundaries (Framework should
not depend on Agent).

This is the SINGLE SOURCE OF TRUTH for conversation stages.
The Framework's Stage enum is an alias to this ConversationStage.
"""

from __future__ import annotations

from enum import Enum


class ConversationStage(str, Enum):
    """Stages in a typical coding assistant conversation.

    This is the canonical source for conversation stages.
    Uses string values for serialization compatibility.

    Note: Framework's `Stage` enum is now an alias to this class.

    Stages:
    - INITIAL: First interaction
    - PLANNING: Understanding scope, planning approach
    - READING: Reading files, gathering context
    - ANALYSIS: Analyzing code, understanding structure
    - EXECUTION: Making changes, running commands
    - VERIFICATION: Testing, validating
    - COMPLETION: Summarizing, done
    """

    INITIAL = "initial"  # First interaction
    PLANNING = "planning"  # Understanding scope, planning approach
    READING = "reading"  # Reading files, gathering context
    ANALYSIS = "analysis"  # Analyzing code, understanding structure
    EXECUTION = "execution"  # Making changes, running commands
    VERIFICATION = "verification"  # Testing, validating
    COMPLETION = "completion"  # Summarizing, done


__all__ = ["ConversationStage"]
