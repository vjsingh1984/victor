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

"""Vertical context container for orchestrator state.

MIGRATION NOTICE: This module has moved to victor.core.verticals.context
to enforce proper layer boundaries. This file is kept for backward compatibility
and imports from the new location.

For new code, import from victor.core.verticals:
    from victor.core.verticals import VerticalContext, create_vertical_context

This module provides a structured container for all vertical-related state,
replacing scattered private attributes with a proper, protocol-compliant
data structure.

Design Philosophy:
- Single source of truth for vertical state
- Protocol-based access for SOLID compliance
- Backward compatibility with existing code
- Type-safe access to all vertical configuration

Capability Config Storage Pattern
----------------------------------
The `capability_configs` dict provides centralized storage for vertical
capability configurations, replacing direct orchestrator attribute assignment.
This pattern follows SOLID principles by avoiding tight coupling between
verticals and the orchestrator implementation.

OLD Pattern (avoid - creates tight coupling):
    # In vertical integration code
    orchestrator.rag_config = {"chunk_size": 512}
    orchestrator.source_verification_config = {"strict": True}
    orchestrator.code_style = {"max_line_length": 100}

NEW Pattern (preferred - decoupled via context):
    # In vertical integration code
    context.set_capability_config("rag_config", {"chunk_size": 512})
    context.set_capability_config("source_verification_config", {"strict": True})
    context.set_capability_config("code_style", {"max_line_length": 100})

    # Or bulk apply
    context.apply_capability_configs({
        "rag_config": {"chunk_size": 512},
        "source_verification_config": {"strict": True},
        "code_style": {"max_line_length": 100},
    })

    # Retrieve elsewhere
    rag_config = context.get_capability_config("rag_config", {})
    if rag_config.get("strict"):
        # Apply strict mode
        ...

Benefits:
- Verticals don't need to know orchestrator's internal structure
- Easy to add new configs without modifying orchestrator class
- Type-safe via protocol methods
- Clear separation of concerns

Usage:
    from victor.agent.vertical_context import VerticalContext

    # Create context
    context = VerticalContext(name="coding")

    # Apply vertical configuration
    context.apply_stages(stages)
    context.apply_middleware(middleware_list)
    context.apply_safety_patterns(patterns)

    # Store capability configs
    context.set_capability_config("rag_config", {"chunk_size": 512})
    context.apply_capability_configs({
        "code_style": {"max_line_length": 100},
        "test_framework": "pytest",
    })

    # Query context
    if context.has_middleware:
        for mw in context.middleware:
            await mw.before_tool_call(...)

    # Retrieve capability configs
    code_style = context.get_capability_config("code_style", {})
"""

from __future__ import annotations

# Import from new canonical location to enforce layer boundaries
from victor.core.verticals.context import (
    VerticalContext,
    VerticalContextProtocol,
    MutableVerticalContextProtocol,
    create_vertical_context,
)

# Re-export for backward compatibility
__all__ = [
    "VerticalContext",
    "VerticalContextProtocol",
    "MutableVerticalContextProtocol",
    "create_vertical_context",
]
