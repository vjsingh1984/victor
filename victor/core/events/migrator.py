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

"""Migration utilities for consolidating event systems.

This module provided conversion functions from the legacy VictorEvent system
to the canonical Event system. NOTE: Migration is complete, these functions
are no longer needed but kept for reference.

Topic Mapping Reference (OLD -> NEW):
- Tool events: (EventCategory.TOOL, "tool.start") -> "tool.start"
- Model events: (EventCategory.MODEL, "llm.request") -> "model.request"
- State events: (EventCategory.STATE, "state.change") -> "state.transition"
- Lifecycle events: (EventCategory.LIFECYCLE, "session.start") -> "lifecycle.session.start"
"""

from __future__ import annotations

import logging


# Migration complete - VictorEvent and EventCategory have been removed
# TOPIC_MAPPING kept for reference only
TOPIC_MAPPING: dict[str, str] = {
    # Tool events
    "tool.start": "tool.start",
    "tool.complete": "tool.result",
    "tool.error": "tool.error",
    "tool.end": "tool.end",
    "tool.output": "tool.output",
    # Model events
    "llm.request": "model.request",
    "llm.response": "model.response",
    "llm.error": "model.error",
    "llm.start": "model.start",
    "llm.end": "model.end",
    # State events
    "state.change": "state.transition",
    "state.snapshot": "state.snapshot",
    "state.restore": "state.restore",
    # Lifecycle events
    "session.start": "lifecycle.session.start",
    "session.end": "lifecycle.session.end",
    "agent.init": "lifecycle.agent.init",
    "agent.shutdown": "lifecycle.agent.shutdown",
    # Error events
    "error.occurred": "error.raised",
    "error.recovered": "error.recovered",
}

logger = logging.getLogger(__name__)


# =============================================================================
# Conversion Functions
# =============================================================================

# NOTE: Legacy event migration functions have been removed.
# Current event conversion is handled by victor/core/events/adapter.py
