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

"""Centralized capability system for verticals (Phase 5).

This module provides a consolidated, YAML-based configuration system for
vertical capabilities, replacing scattered capability definitions.

New in Phase 5:
- CapabilityDefinition: Declarative capability definitions
- CapabilityRegistry: Thread-safe singleton for capability management
- CapabilityHandler: Auto-generated handlers for configure/get operations
- Entry point discovery for external capabilities
- YAML-first capability loading

Example:
    from victor.core.capabilities import (
        CapabilityRegistry,
        CapabilityDefinition,
        CapabilityType,
    )

    # Get registry singleton
    registry = CapabilityRegistry.get_instance()

    # Register capability
    definition = CapabilityDefinition(
        name="git_safety",
        capability_type=CapabilityType.SAFETY,
        description="Git safety rules",
    )
    registry.register(definition)

    # Get handler
    handler = registry.get_handler("git_safety")
    handler.configure(context, {"block_force_push": True})

Legacy API (backward compatible):
    from victor.core.capabilities import CapabilityLoader

    # Load capabilities for a vertical
    loader = CapabilityLoader.from_vertical("coding")
    capabilities = loader.list_capabilities()
"""

# Legacy API (Phase 4 and earlier)
from victor.core.capabilities.base_loader import (
    Capability,
    CapabilityLoader,
    CapabilitySet,
    CapabilityType as LegacyCapabilityType,
)

# New API (Phase 5)
from victor.core.capabilities.types import (
    CapabilityDefinition,
    CapabilityType,
    ConfigSchema,
)
from victor.core.capabilities.registry import (
    CapabilityRegistry,
    validate_capability_yaml,
)
from victor.core.capabilities.handler import (
    CapabilityHandler,
    VerticalContextProtocol,
)

__all__ = [
    # Legacy API (backward compatible)
    "Capability",
    "CapabilityLoader",
    "CapabilitySet",
    "LegacyCapabilityType",
    # New API (Phase 5)
    "CapabilityDefinition",
    "CapabilityType",
    "ConfigSchema",
    "CapabilityRegistry",
    "validate_capability_yaml",
    "CapabilityHandler",
    "VerticalContextProtocol",
]
