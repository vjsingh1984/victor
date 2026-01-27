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

"""Core team role types.

This module provides the canonical SubAgentRole enum used across
the Victor codebase. Previously in agent/subagents/base,
now moved to core/teams to enforce proper layer boundaries (Framework
should not depend on Agent).

This is the SINGLE SOURCE OF TRUTH for sub-agent roles.
"""

from __future__ import annotations

from enum import Enum


class SubAgentRole(str, Enum):
    """Role specialization for sub-agents.

    Each role has specific capabilities and constraints:

    - RESEARCHER: Read-only exploration (read, search, code_search, web_search)
    - PLANNER: Task breakdown and planning (read, ls, search, plan_files)
    - EXECUTOR: Code changes and execution (read, write, edit, shell, test, git)
    - REVIEWER: Quality checks and testing (read, search, test, git_diff, shell)
    - TESTER: Test writing and running (read, write to tests/, test, shell)
    """

    RESEARCHER = "researcher"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    TESTER = "tester"

    def __str__(self) -> str:
        """Return the string value instead of the enum name."""
        return self.value

    def __repr__(self) -> str:
        """Return a custom repr with lowercase name."""
        return f"<{self.__class__.__name__}.{self.name.lower()}: '{self.value}'>"


__all__ = ["SubAgentRole"]
