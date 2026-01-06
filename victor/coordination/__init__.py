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

"""Multi-agent coordination module.

This module provides unified coordination for multi-agent systems,
consolidating multiple coordinator implementations into a single,
configurable system.

Architecture:
    - Single coordinator with mode-based configuration
    - Formation strategies for different execution patterns
    - Event system integration
    - Clean dependency hierarchy (no circular dependencies)

Example:
    from victor.coordination import create_coordinator

    coordinator = create_coordinator(
        formation="hierarchical",
        mode="production",
        config=coordinator_config
    )
"""

from victor.teams import create_coordinator
from victor.teams.types import (
    AgentMessage,
    MessageType,
    TeamFormation,
)

__all__ = [
    "create_coordinator",
    "AgentMessage",
    "MessageType",
    "TeamFormation",
]
