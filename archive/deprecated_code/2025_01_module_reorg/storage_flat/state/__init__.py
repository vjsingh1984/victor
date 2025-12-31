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

"""Victor State - Standalone state machine package.

This module has moved to victor.storage.state.
Import from victor.storage.state instead for new code.

This module provides backward-compatible re-exports.
"""

from __future__ import annotations

# Re-export from agent.conversation_state for backward compatibility
from victor.agent.conversation_state import (
    ConversationStage,
    ConversationState,
    ConversationStateMachine,
    STAGE_KEYWORDS,
)

# New standalone components - re-export from new location
from victor.storage.state.machine import (
    StateMachine,
    StateConfig,
    StateTransition,
)
from victor.storage.state.protocols import (
    StateProtocol,
    StageProtocol,
    TransitionValidatorProtocol,
)

__all__ = [
    # Conversation-specific (re-exports)
    "ConversationStage",
    "ConversationState",
    "ConversationStateMachine",
    "STAGE_KEYWORDS",
    # Generic state machine
    "StateMachine",
    "StateConfig",
    "StateTransition",
    # Protocols
    "StateProtocol",
    "StageProtocol",
    "TransitionValidatorProtocol",
]

__version__ = "0.2.0"
