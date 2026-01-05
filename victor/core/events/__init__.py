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

"""Unified Event System for Victor.

This package provides a unified event taxonomy and utilities for event handling
across all Victor subsystems. The taxonomy consolidates fragmented event types
from various modules into a single, hierarchical system.

Key Components:
- UnifiedEventType: Hierarchical enum of all event types
- Mapping functions: Convert legacy event types to unified taxonomy
- Utility functions: Query and filter events by category

Usage:
    from victor.core.events import (
        UnifiedEventType,
        map_workflow_event,
        get_events_by_category,
    )

    # Use unified event types directly
    event_type = UnifiedEventType.WORKFLOW_NODE_START
    print(event_type.category)  # "workflow"

    # Map from legacy event types
    from victor.workflows.streaming import WorkflowEventType
    unified = map_workflow_event(WorkflowEventType.NODE_START)

    # Query events by category
    tool_events = get_events_by_category("tool")

Migration Guide:
    Legacy event types from the following modules can be mapped to the
    unified taxonomy:

    - victor.workflows.streaming.WorkflowEventType -> map_workflow_event()
    - victor.observability.event_bus.EventCategory -> map_event_category()

    While legacy event types continue to work, new code should use
    UnifiedEventType directly for consistency.

See Also:
    - victor.core.events.taxonomy: Full documentation and implementation
    - victor.observability.event_bus: EventBus for event pub/sub
    - victor.workflows.observability: Workflow observability integration
"""

from victor.core.events.taxonomy import (
    # Core enum
    UnifiedEventType,
    # Mapping functions
    map_workflow_event,
    map_event_category,
    map_framework_event,
    map_tool_event,
    map_agent_event,
    map_system_event,
    # Utility functions
    get_all_categories,
    get_events_by_category,
    is_valid_event_type,
    # Deprecation helpers
    emit_deprecation_warning,
)

__all__ = [
    # Core enum
    "UnifiedEventType",
    # Mapping functions
    "map_workflow_event",
    "map_event_category",
    "map_framework_event",
    "map_tool_event",
    "map_agent_event",
    "map_system_event",
    # Utility functions
    "get_all_categories",
    "get_events_by_category",
    "is_valid_event_type",
    # Deprecation helpers
    "emit_deprecation_warning",
]
